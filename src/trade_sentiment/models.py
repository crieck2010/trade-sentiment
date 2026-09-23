"""Plain-data models.  No I/O, no network, no UI imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SentimentLabel(str, Enum):
    VERY_BEARISH = "very bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very bullish"


def label_for(polarity: float) -> SentimentLabel:
    """Bucket a -1..1 polarity onto the five-point spectrum."""
    if polarity <= -0.6:
        return SentimentLabel.VERY_BEARISH
    if polarity <= -0.2:
        return SentimentLabel.BEARISH
    if polarity < 0.2:
        return SentimentLabel.NEUTRAL
    if polarity < 0.6:
        return SentimentLabel.BULLISH
    return SentimentLabel.VERY_BULLISH


def bullishness_10(polarity: float) -> float:
    """Linear tone score on the 0–10 scale: -1 → 0, 0 → 5, +1 → 10."""
    return round(max(0.0, min(10.0, (polarity + 1.0) / 2.0 * 10.0)), 1)


def conviction_10(bullishness: float, n_mentions: int) -> float:
    """Conviction score on the 0–10 scale: tone weighted by chatter volume.

    ``volume_factor`` grows from 0.5 (a lone voice) toward 1.5 (a crowd of
    100+ mentions), so a 9/10 tone with 5 mentions scores ~5.0 conviction
    while a 6/10 tone with 200 mentions scores ~9.0.  The two numbers are
    reported side by side precisely to separate strong-but-quiet sentiment
    from moderate-but-loud sentiment.
    """
    volume_factor = 0.5 + min(n_mentions, 100) / 100.0
    return round(max(0.0, min(10.0, bullishness * volume_factor)), 1)


@dataclass(frozen=True)
class Mention:
    """One piece of public chatter about a symbol."""

    id: str
    source: str  # "reddit" | "stocktwits" | "news"
    symbol: str
    text: str
    timestamp: datetime
    author: str = ""
    engagement: int = 0  # likes + comments + reshares, source-dependent
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source": self.source, "symbol": self.symbol,
            "text": self.text, "timestamp": self.timestamp.isoformat(),
            "author": self.author, "engagement": self.engagement,
            "url": self.url,
        }


@dataclass(frozen=True)
class ScoredMention:
    """A mention with its tone score attached."""

    mention: Mention
    polarity: float  # -1.0 (very bearish) .. +1.0 (very bullish)
    magnitude: float  # 0..1, how much sentiment-bearing language was found
    label: SentimentLabel
    hits: tuple[str, ...] = ()  # lexicon terms that fired, for explainability

    def to_dict(self) -> dict:
        return {
            **self.mention.to_dict(),
            "polarity": round(self.polarity, 3),
            "magnitude": round(self.magnitude, 3),
            "label": self.label.value,
            "hits": list(self.hits),
        }


@dataclass(frozen=True)
class SentimentWindow:
    """Aggregated tone + chatter for one symbol over one time window."""

    symbol: str
    start: datetime
    end: datetime
    n_mentions: int
    mean_polarity: float
    volume_zscore: float | None  # vs rolling baseline; None when no baseline
    tone_shift: float | None  # mean_polarity minus baseline mean

    @property
    def bullishness(self) -> float:
        return bullishness_10(self.mean_polarity)

    @property
    def conviction(self) -> float:
        return conviction_10(self.bullishness, self.n_mentions)

    @property
    def label(self) -> SentimentLabel:
        return label_for(self.mean_polarity)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "n_mentions": self.n_mentions,
            "mean_polarity": round(self.mean_polarity, 3),
            "bullishness_10": self.bullishness,
            "conviction_10": self.conviction,
            "label": self.label.value,
            "volume_zscore": (round(self.volume_zscore, 2)
                              if self.volume_zscore is not None else None),
            "tone_shift": (round(self.tone_shift, 3)
                           if self.tone_shift is not None else None),
        }


@dataclass(frozen=True)
class SentimentPop:
    """A detected surge: unusual chatter and/or a tone shift for a symbol."""

    window: SentimentWindow
    drivers: tuple[str, ...] = ()  # top lexicon terms behind the tone
    verdict: str = ""

    def __post_init__(self):
        if not self.verdict:
            w = self.window
            object.__setattr__(
                self, "verdict",
                f"{w.symbol} has a pop in sentiment of "
                f"{w.bullishness}/10 bullishness "
                f"(conviction {w.conviction}/10, {w.label.value}) — "
                f"{w.n_mentions} mentions in the window.",
            )

    def to_dict(self) -> dict:
        return {
            **self.window.to_dict(),
            "drivers": list(self.drivers),
            "verdict": self.verdict,
        }
