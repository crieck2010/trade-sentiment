"""Interoperability bridges.  All sibling imports are lazy.

- :func:`to_agent_ideas` — sentiment pops as research-desk ideas for
  ``trade-agents`` (consumed by the ``sentiment_scout`` in v0.1.2).
- :func:`to_signal_overlay` — per-symbol direction map usable as a
  signal filter alongside ``trade-strategies`` output.
- :func:`to_dashboard_rows` — flat rows for the dashboard data viewers.
"""

from __future__ import annotations

from .models import SentimentPop


def to_agent_ideas(pops: list[SentimentPop]) -> list[dict]:
    """Convert pops to trade-agents idea dicts.

    Direction follows tone: bullish pops → LONG ideas, bearish pops →
    SHORT ideas.  ``score`` blends conviction and tone so the PM agent
    ranks loud, confident sentiment first.
    """
    ideas = []
    for pop in pops:
        w = pop.window
        direction = "LONG" if w.mean_polarity >= 0 else "SHORT"
        score = round(min(1.0, (w.conviction / 10.0) * 0.6
                          + abs(w.mean_polarity) * 0.4), 3)
        ideas.append({
            "agent": "sentiment_scout",
            "symbol": w.symbol,
            "strategy": "sentiment_momentum",
            "direction": direction,
            "score": score,
            "conviction": w.conviction,
            "bullishness_10": w.bullishness,
            "conviction_10": w.conviction,
            "n_mentions": w.n_mentions,
            "volume_zscore": w.volume_zscore,
            "tone_shift": w.tone_shift,
            "drivers": list(pop.drivers),
            "thesis": pop.verdict,
            "params": {"window_hours": 24},
        })
    return ideas


def to_signal_overlay(
    pops: list[SentimentPop],
    min_conviction: float = 5.0,
) -> dict[str, str]:
    """{symbol: "LONG"|"SHORT"} for pops clearing the conviction bar.

    Use as a gate: only take strategy signals that agree with the overlay,
    or size up when they do.
    """
    overlay = {}
    for pop in pops:
        w = pop.window
        if w.conviction >= min_conviction:
            overlay[w.symbol] = "LONG" if w.mean_polarity >= 0 else "SHORT"
    return overlay


def to_dashboard_rows(pops: list[SentimentPop]) -> list[dict]:
    """Flat, JSON-safe rows for dashboard tables."""
    return [p.to_dict() for p in pops]


def describe() -> dict:
    """Machine-readable capability summary for the suite registry."""
    return {
        "name": "trade-sentiment",
        "capabilities": ["scan", "score_text", "to_agent_ideas",
                         "to_signal_overlay", "to_dashboard_rows"],
        "outputs": ["SentimentPop", "idea dicts", "signal overlay"],
    }
