"""Window aggregation and sentiment-pop detection.

A *pop* is unusual chatter about a symbol: a spike in mention volume
and/or a sharp tone shift versus a rolling baseline.  Each pop reports
two numbers side by side:

- ``bullishness_10`` — pure tone on the 0–10 scale (linear map of polarity)
- ``conviction_10`` — tone weighted by chatter volume

so a 9/10 tone with 5 mentions (~5.0 conviction) reads differently from
a 6/10 tone with 200 mentions (~9.0 conviction).
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone

from .models import Mention, ScoredMention, SentimentPop, SentimentWindow
from .scoring import score_mentions


def aggregate(
    scored: list[ScoredMention],
    symbol: str,
    start: datetime,
    end: datetime,
    baseline: list[SentimentWindow] | None = None,
) -> SentimentWindow:
    """Fold scored mentions into one window, with optional baseline stats."""
    n = len(scored)
    mean_p = statistics.fmean([s.polarity for s in scored]) if n else 0.0

    volume_zscore: float | None = None
    tone_shift: float | None = None
    if baseline:
        counts = [w.n_mentions for w in baseline if w.symbol == symbol]
        tones = [w.mean_polarity for w in baseline if w.symbol == symbol]
        if len(counts) >= 3:
            mu, sd = statistics.fmean(counts), statistics.pstdev(counts) or 1.0
            volume_zscore = (n - mu) / sd
        if tones:
            tone_shift = mean_p - statistics.fmean(tones)

    return SentimentWindow(
        symbol=symbol.upper(), start=start, end=end, n_mentions=n,
        mean_polarity=round(mean_p, 4),
        volume_zscore=round(volume_zscore, 2) if volume_zscore is not None else None,
        tone_shift=round(tone_shift, 4) if tone_shift is not None else None,
    )


def burst_window(
    scored: list[ScoredMention],
    symbol: str,
    start: datetime,
    end: datetime,
    n_slices: int = 7,
) -> SentimentWindow:
    """Burst detection without history: slice the window by time.

    The window is cut into ``n_slices`` equal time slices.  The most recent
    slice is the "now" under test; the earlier slices form the baseline.
    The returned window covers the *whole* window (so tone reflects all
    chatter) but its ``volume_zscore``/``tone_shift`` measure how the
    latest slice deviates from the earlier ones — a burst detector that
    needs no stored history.
    """
    span = (end - start).total_seconds() or 1.0
    counts = [0] * n_slices
    tones: list[list[float]] = [[] for _ in range(n_slices)]
    for s in scored:
        ts = s.mention.timestamp
        idx = int((ts - start).total_seconds() / span * n_slices)
        idx = max(0, min(n_slices - 1, idx))
        counts[idx] += 1
        tones[idx].append(s.polarity)

    # z-score of the *latest slice's* count against the earlier slices:
    # a burst is chatter concentrating right now.
    base_counts = counts[:-1]
    volume_zscore: float | None = None
    if len(base_counts) >= 3:
        mu = statistics.fmean(base_counts)
        sd = statistics.pstdev(base_counts) or 1.0
        volume_zscore = round((counts[-1] - mu) / sd, 2)
    base_tones = [p for t in tones[:-1] for p in t]
    tone_shift: float | None = None
    if base_tones:
        tone_shift = round(statistics.fmean(tones[-1] or [0.0])
                           - statistics.fmean(base_tones), 4)

    n = len(scored)
    mean_p = statistics.fmean([s.polarity for s in scored]) if n else 0.0
    return SentimentWindow(
        symbol=symbol.upper(), start=start, end=end, n_mentions=n,
        mean_polarity=round(mean_p, 4),
        volume_zscore=volume_zscore, tone_shift=tone_shift,
    )


def detect_pops(
    windows: list[SentimentWindow],
    scored_by_symbol: dict[str, list[ScoredMention]] | None = None,
    min_mentions: int = 5,
    min_volume_zscore: float = 2.0,
    min_abs_tone: float = 0.25,
) -> list[SentimentPop]:
    """Keep windows that are loud, tonal, or both; rank loudest-first.

    A window pops when it clears the mention floor AND either its volume
    z-score or its absolute tone clears its bar.  Ranking blends the two
    so a quiet-but-violent tone shift can outrank a loud-but-flat window.
    """
    pops = []
    for w in windows:
        if w.n_mentions < min_mentions:
            continue
        loud = w.volume_zscore is not None and w.volume_zscore >= min_volume_zscore
        tonal = abs(w.mean_polarity) >= min_abs_tone
        if not (loud or tonal):
            continue
        drivers: tuple[str, ...] = ()
        if scored_by_symbol and w.symbol in scored_by_symbol:
            hits = Counter(h for s in scored_by_symbol[w.symbol] for h in s.hits)
            drivers = tuple(h for h, _ in hits.most_common(5))
        pops.append(SentimentPop(window=w, drivers=drivers))
    pops.sort(
        key=lambda p: (
            p.window.volume_zscore or 0.0,
            abs(p.window.mean_polarity),
            p.window.conviction,
        ),
        reverse=True,
    )
    return pops


def window_bounds(hours: int, end: datetime | None = None) -> tuple[datetime, datetime]:
    """(start, end) for a trailing window of ``hours``."""
    end = end or datetime.now(timezone.utc)
    return end - timedelta(hours=hours), end
