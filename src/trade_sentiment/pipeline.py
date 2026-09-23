"""End-to-end scan: fetch → score → aggregate → pops.

``scan()`` is the one call the agents, dashboards, and CLI share.
Source fetching runs in a thread pool (network is the bottleneck);
scoring and aggregation are CPU-cheap and stay on the calling thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from . import aggregation as agg
from . import sources as src
from .models import Mention, SentimentPop
from .scoring import score_mentions


def fetch_all(
    symbols: list[str],
    source_names: list[str] | None = None,
    limit_per_source: int = 50,
    max_workers: int = 6,
) -> dict[str, list[Mention]]:
    """Fetch mentions for every (symbol, source) pair in parallel."""
    names = source_names or list(src.SOURCES)
    jobs = [(s, src.get_source(n)) for s in symbols for n in names]
    mentions: dict[str, list[Mention]] = {s.upper(): [] for s in symbols}

    def _one(job):
        symbol, source = job
        try:
            return symbol.upper(), source.fetch(symbol, limit_per_source)
        except Exception:
            return symbol.upper(), []  # one bad source must not kill the scan

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for symbol, found in pool.map(_one, jobs):
            mentions[symbol].extend(found)
    return mentions


def scan(
    symbols: list[str],
    source_names: list[str] | None = None,
    window_hours: int = 24,
    baseline_windows: int = 7,
    min_mentions: int = 5,
    limit_per_source: int = 50,
) -> list[SentimentPop]:
    """Scan ``symbols`` for sentiment pops.

    Returns pops ranked loudest-first, each carrying ``bullishness_10``
    (pure tone) and ``conviction_10`` (tone × volume) plus a plain-English
    ``verdict`` such as "XYZ has a pop in sentiment of 9/10 bullishness".
    """
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("at least one symbol is required")
    end = agg.window_bounds(window_hours)[1]
    start = end - timedelta(hours=window_hours)

    mentions = fetch_all(symbols, source_names, limit_per_source)

    # baseline: slice the window by time; the most recent slice is tested
    # against the earlier slices, so a burst needs no stored history.
    pops = []
    scored_by_symbol = {}
    windows = []
    for symbol in symbols:
        scored = score_mentions(mentions[symbol])
        scored_by_symbol[symbol] = scored
        windows.append(agg.burst_window(scored, symbol, start, end,
                                       n_slices=max(baseline_windows, 3)))
    return agg.detect_pops(windows, scored_by_symbol, min_mentions=min_mentions)
