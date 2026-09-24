"""Tests for the FinBERT rescoring adapter.

Everything here runs WITHOUT torch/transformers installed: the math is
pure-Python and the model is replaced by a duck-typed fake scorer.  The
single live-model test is gated behind TRADE_SENTIMENT_FINBERT_LIVE=1
(it downloads ~440 MB on first run) and skips otherwise.
"""

from __future__ import annotations

import builtins
import os

import pytest

from trade_sentiment import finbert as fb
from trade_sentiment.models import Mention, ScoredMention, label_for
from trade_sentiment.scoring import score_mentions


def _mention(text: str, symbol: str = "XYZ") -> Mention:
    from datetime import datetime, timezone
    return Mention(id="t", source="test", symbol=symbol, text=text,
                   timestamp=datetime.now(timezone.utc))


class FakeScorer:
    """Duck-typed stand-in for FinBERTScorer: fixed polarity per text."""

    def __init__(self, mapping: dict[str, float] | None = None):
        self.mapping = mapping or {}
        self.calls: list[list[str]] = []

    def polarity(self, texts):
        self.calls.append(list(texts))
        return [self.mapping.get(t, 0.0) for t in texts]


class TestLogitsToPolarity:
    def test_strong_positive(self):
        assert fb.logits_to_polarity([3.0, 0.0, 0.0]) == pytest.approx(
            0.8642, abs=1e-3)

    def test_strong_negative(self):
        assert fb.logits_to_polarity([0.0, 3.0, 0.0]) == pytest.approx(
            -0.8642, abs=1e-3)

    def test_neutral_mass_pushes_to_zero(self):
        assert fb.logits_to_polarity([0.0, 0.0, 5.0]) == pytest.approx(
            0.0, abs=1e-2)

    def test_uniform_is_zero(self):
        assert fb.logits_to_polarity([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_custom_label_order(self):
        # head ordered (negative, neutral, positive): index 2 is positive
        p = fb.logits_to_polarity([0.0, 0.0, 3.0],
                                  label_order=("negative", "neutral",
                                               "positive"))
        assert p == pytest.approx(0.8642, abs=1e-3)

    def test_missing_labels_rejected(self):
        with pytest.raises(fb.FinBERTError):
            fb.logits_to_polarity([1.0, 2.0], label_order=("a", "b"))

    def test_wrong_logit_count_rejected(self):
        with pytest.raises(fb.FinBERTError):
            fb.logits_to_polarity([1.0, 2.0])


class TestRequireFinbert:
    def test_missing_stack_raises_helpful_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch", "transformers") or \
                    name.startswith(("torch.", "transformers.")):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(fb.FinBERTError, match="pip install torch"):
            fb.require_finbert()

    def test_scorer_load_fails_soft_without_stack(self):
        # torch truly absent in this env; load() must raise FinBERTError,
        # never ImportError, so callers can catch one type.
        with pytest.raises(fb.FinBERTError):
            fb.FinBERTScorer().load()


class TestRescoreWithFinbert:
    def _scored(self):
        return score_mentions([
            _mention("earnings beat, guidance raised"),
            _mention("lawsuit filed, guidance cut"),
        ])

    def test_blend_zero_returns_lexicon(self):
        scored = self._scored()
        fake = FakeScorer({"earnings beat, guidance raised": 1.0})
        out = fb.rescore_with_finbert(scored, blend=0.0, scorer=fake)
        assert [s.polarity for s in out] == [s.polarity for s in scored]
        assert fake.calls == []  # model never consulted

    def test_blend_one_takes_finbert(self):
        scored = self._scored()
        fake = FakeScorer({"earnings beat, guidance raised": 0.9,
                           "lawsuit filed, guidance cut": -0.7})
        out = fb.rescore_with_finbert(scored, blend=1.0, scorer=fake)
        assert out[0].polarity == pytest.approx(0.9)
        assert out[1].polarity == pytest.approx(-0.7)

    def test_blend_half_is_convex_combination(self):
        scored = self._scored()
        fake = FakeScorer({"earnings beat, guidance raised": 1.0,
                           "lawsuit filed, guidance cut": -1.0})
        out = fb.rescore_with_finbert(scored, blend=0.5, scorer=fake)
        for sm, new in zip(scored, out):
            f = fake.mapping[sm.mention.text]
            assert new.polarity == pytest.approx(
                round(0.5 * sm.polarity + 0.5 * f, 4))
            assert new.label == label_for(new.polarity)
            assert new.hits == sm.hits  # lexicon explanation preserved
            assert new.magnitude == sm.magnitude

    def test_failing_scorer_keeps_lexicon(self):
        class Boom:
            def polarity(self, texts):
                raise RuntimeError("gpu melted")
        scored = self._scored()
        out = fb.rescore_with_finbert(scored, blend=0.5, scorer=Boom())
        assert [s.polarity for s in out] == [s.polarity for s in scored]

    def test_none_scorer_without_stack_keeps_lexicon(self, monkeypatch):
        # scorer=None triggers a real load attempt; with no torch the
        # FinBERTError path must return lexicon scores, not raise.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        scored = self._scored()
        out = fb.rescore_with_finbert(scored, blend=0.5, scorer=None)
        assert [s.polarity for s in out] == [s.polarity for s in scored]

    def test_empty_input(self):
        assert fb.rescore_with_finbert([], blend=0.5,
                                       scorer=FakeScorer()) == []


class TestFinbertRescorer:
    def test_builds_reusable_callable(self, monkeypatch):
        created = []

        class FakeFinBERTScorer(fb.FinBERTScorer):
            def __init__(self, *a, **k):
                created.append((a, k))
                self._fake = FakeScorer({"good news": 0.8})

            def load(self):
                return self

            def polarity(self, texts):
                return self._fake.polarity(texts)

        monkeypatch.setattr(fb, "FinBERTScorer", FakeFinBERTScorer)
        rescorer = fb.finbert_rescorer(blend=0.5)
        assert created  # built eagerly, so load cost is paid once
        scored = score_mentions([_mention("good news")])
        out1 = rescorer(scored)
        out2 = rescorer(scored)
        assert len(created) == 1  # shared across calls/symbols
        assert out1[0].polarity == out2[0].polarity


class TestScanRescorerHook:
    def test_rescorer_applied_in_scan(self):
        from unittest.mock import patch
        from trade_sentiment import pipeline, sources as src

        class FakeSource(src.Source):
            name = "fake"
            def fetch(self, symbol, limit=50):
                return [_mention("moon rocket squeeze", symbol)] * 12

        fake = FakeSource()

        def flip(scored):
            return [ScoredMention(mention=s.mention, polarity=-s.polarity,
                                  magnitude=s.magnitude,
                                  label=label_for(-s.polarity), hits=s.hits)
                    for s in scored]

        with patch.dict(src.SOURCES, {"fake": lambda: fake}):
            plain = pipeline.scan(["XYZ"], source_names=["fake"],
                                  min_mentions=5)
            flipped = pipeline.scan(["XYZ"], source_names=["fake"],
                                    min_mentions=5, rescorer=flip)
        assert plain[0].window.mean_polarity > 0
        assert flipped[0].window.mean_polarity < 0

    def test_failing_rescorer_does_not_kill_scan(self):
        from unittest.mock import patch
        from trade_sentiment import pipeline, sources as src

        class FakeSource(src.Source):
            name = "fake"
            def fetch(self, symbol, limit=50):
                return [_mention("moon rocket squeeze", symbol)] * 12

        def boom(scored):
            raise RuntimeError("nope")

        with patch.dict(src.SOURCES, {"fake": lambda: FakeSource()}):
            pops = pipeline.scan(["XYZ"], source_names=["fake"],
                                 min_mentions=5, rescorer=boom)
        assert len(pops) == 1  # lexicon scores stood in


class TestDescribe:
    def test_capability_summary(self):
        d = fb.describe()
        assert d["name"] == "trade-sentiment.finbert"
        assert "rescore_with_finbert" in d["capabilities"]
        assert d["optional_dependencies"] == ["torch", "transformers"]


class TestCli:
    def test_scan_rescore_flags_parse(self):
        from trade_sentiment.cli import build_parser
        args = build_parser().parse_args(
            ["scan", "AAPL", "--rescore", "finbert", "--blend", "0.7"])
        assert args.rescore == "finbert" and args.blend == 0.7

    def test_score_model_flag_parses(self):
        from trade_sentiment.cli import build_parser
        args = build_parser().parse_args(["score", "--model", "finbert",
                                          "good news"])
        assert args.model == "finbert"

    def test_score_finbert_without_stack_exits_3(self, capsys):
        # torch absent here: the CLI must fail cleanly, not traceback.
        from trade_sentiment.cli import main
        rc = main(["score", "--model", "finbert", "good news"])
        assert rc == 3
        assert "finbert unavailable" in capsys.readouterr().err


@pytest.mark.skipif(
    os.environ.get("TRADE_SENTIMENT_FINBERT_LIVE") != "1",
    reason="live FinBERT test: set TRADE_SENTIMENT_FINBERT_LIVE=1 to run "
           "(downloads ~440 MB on first run)",
)
class TestFinbertLive:
    def test_live_model_scores_in_range(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        scorer = fb.FinBERTScorer(batch_size=8).load()
        polarities = scorer.polarity([
            "The company reported record earnings and raised guidance.",
            "The company filed for bankruptcy protection.",
            "The meeting is scheduled for Tuesday.",
        ])
        assert polarities[0] > 0.3
        assert polarities[1] < -0.3
        assert abs(polarities[2]) < 0.5
        assert all(-1.0 <= p <= 1.0 for p in polarities)
