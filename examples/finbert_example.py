"""FinBERT adapter example.

Runs fully WITHOUT torch/transformers: a tiny fake scorer stands in for
the model so the blending math and the scan hook are demonstrable
anywhere.  Set TRADE_SENTIMENT_FINBERT_LIVE=1 with the ML stack installed
to run the real model instead.
"""

from __future__ import annotations

import os

from trade_sentiment import finbert, scoring
from trade_sentiment.models import Mention

TEXTS = [
    "earnings beat, guidance raised — best quarter in years",
    "lawsuit filed, guidance cut, not exactly a blowout",
    "the meeting is scheduled for Tuesday",
]


def _mentions():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return [Mention(id=str(i), source="demo", symbol="XYZ", text=t,
                    timestamp=now) for i, t in enumerate(TEXTS)]


def main() -> None:
    lexicon_scored = scoring.score_mentions(_mentions())
    print("lexicon polarities:",
          [s.polarity for s in lexicon_scored])

    live = os.environ.get("TRADE_SENTIMENT_FINBERT_LIVE") == "1"
    if live:
        scorer = finbert.FinBERTScorer().load()
        print("using live FinBERT:",
              finbert.FINBERT_MODEL)
    else:
        # Fake stand-in: pretends the model read the second text as
        # strongly negative (negation scope the lexicon mishandles).
        class FakeScorer:
            def polarity(self, texts):
                return [0.9 if "beat" in t else
                        -0.8 if "lawsuit" in t else 0.0 for t in texts]

        scorer = FakeScorer()
        print("using fake scorer (set TRADE_SENTIMENT_FINBERT_LIVE=1 "
              "with torch+transformers for the real model)")

    for blend in (0.0, 0.5, 1.0):
        blended = finbert.rescore_with_finbert(
            lexicon_scored, blend=blend, scorer=scorer)
        print(f"blend={blend}:",
              [(round(s.polarity, 3), s.label.value) for s in blended])

    # whole-scan hook: one shared scorer across symbols
    rescorer = finbert.finbert_rescorer(blend=0.5) \
        if live else lambda scored: finbert.rescore_with_finbert(
            scored, blend=0.5, scorer=scorer)
    print("scan-ready rescorer built:", callable(rescorer))


if __name__ == "__main__":
    main()
