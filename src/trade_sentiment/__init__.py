"""Social/news sentiment engine for the trade-suite.

Monitors public chatter (Reddit, StockTwits, news RSS) about a ticker,
scores the language on a very-bearish → very-bullish spectrum, and emits
sentiment pops such as "XYZ has a pop in sentiment of 9/10 bullishness".

The core is stdlib-only and deterministic: a finance-tuned lexicon scorer,
per-symbol rolling baselines, and z-score pop detection.  Every source
adapter is lazy and keyless for the v0.1.0 sources; the X adapter ships in a
later version once an API key is available.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .aggregation import aggregate, detect_pops
from .models import (
    Mention,
    ScoredMention,
    SentimentLabel,
    SentimentPop,
    SentimentWindow,
)
from .pipeline import scan
from .scoring import score_text

__all__ = [
    "Mention",
    "ScoredMention",
    "SentimentLabel",
    "SentimentPop",
    "SentimentWindow",
    "aggregate",
    "detect_pops",
    "scan",
    "score_text",
    "__version__",
]
