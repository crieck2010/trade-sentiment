"""Archive example: record-on-fetch + point-in-time query.

Uses an in-memory archive and synthetic scored mentions — no network.
Shows the two timestamps and why the query predicate matters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trade_sentiment import Archive, Mention, ScoredMention, score_text

UTC = timezone.utc


def fake_scored(symbol: str, text: str, published: datetime) -> ScoredMention:
    m = Mention(id="ex1", source="news", symbol=symbol, text=text,
                timestamp=published, url="https://example.com/x")
    sm = score_text(text)
    return ScoredMention(mention=m, polarity=sm.polarity,
                         magnitude=sm.magnitude, label=sm.label,
                         hits=sm.hits)


def main() -> None:
    arc = Archive(":memory:")

    t = datetime(2026, 9, 20, 14, 0, tzinfo=UTC)  # headline published 14:00
    fetched = t + timedelta(hours=6)  # ...but scraped at 20:00
    arc.record_scored(
        fake_scored("XYZ", "Earnings beat, guidance raised sharply", t),
        recorded_at=fetched)

    # A decision made at 15:00 — one hour after publication — could NOT
    # have known this observation (it was only scraped at 20:00).
    early = arc.query("XYZ", as_of=t + timedelta(hours=1))
    print(f"knowable at 15:00: {len(early)} observations  (no lookahead)")

    # The same decision made at 21:00 sees it.
    late = arc.query("XYZ", as_of=t + timedelta(hours=7))
    print(f"knowable at 21:00: {len(late)} observations")
    for o in late:
        print(f"  observed_at={o.observed_at:%H:%M} "
              f"recorded_at={o.recorded_at:%H:%M} "
              f"polarity={o.polarity:+.2f} label={o.label}")

    print("coverage:", arc.coverage("XYZ"))


if __name__ == "__main__":
    main()
