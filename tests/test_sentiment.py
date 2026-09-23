"""Tests for trade-sentiment.  Network is mocked; nothing hits the wire."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from trade_sentiment import (
    Mention,
    SentimentLabel,
    aggregate,
    detect_pops,
    scan,
    score_text,
)
from trade_sentiment import aggregation as agg
from trade_sentiment import sources as src
from trade_sentiment.adapters import (
    describe,
    to_agent_ideas,
    to_dashboard_rows,
    to_signal_overlay,
)
from trade_sentiment.cli import build_parser, main
from trade_sentiment.models import (
    SentimentPop,
    SentimentWindow,
    bullishness_10,
    conviction_10,
    label_for,
)
from trade_sentiment.scoring import rescore_with_llm, score_mentions


def _mention(text: str, symbol: str = "XYZ",
             ts: datetime | None = None) -> Mention:
    return Mention(id="t1", source="reddit", symbol=symbol, text=text,
                   timestamp=ts or datetime.now(timezone.utc))


NOW = datetime.now(timezone.utc)


# -- models ----------------------------------------------------------------
class TestModels:
    def test_label_buckets(self):
        assert label_for(-1.0) is SentimentLabel.VERY_BEARISH
        assert label_for(-0.61) is SentimentLabel.VERY_BEARISH
        assert label_for(-0.4) is SentimentLabel.BEARISH
        assert label_for(0.0) is SentimentLabel.NEUTRAL
        assert label_for(0.19) is SentimentLabel.NEUTRAL
        assert label_for(0.4) is SentimentLabel.BULLISH
        assert label_for(0.9) is SentimentLabel.VERY_BULLISH

    def test_bullishness_scale(self):
        assert bullishness_10(-1.0) == 0.0
        assert bullishness_10(0.0) == 5.0
        assert bullishness_10(1.0) == 10.0
        assert bullishness_10(0.8) == 9.0

    def test_conviction_separates_quiet_from_loud(self):
        quiet = conviction_10(9.0, 5)    # strong tone, few mentions
        loud = conviction_10(6.0, 200)   # moderate tone, crowd
        assert quiet < loud
        assert 0.0 <= quiet <= 10.0 and 0.0 <= loud <= 10.0

    def test_conviction_bounds(self):
        assert conviction_10(10.0, 10_000) == 10.0
        assert conviction_10(0.0, 0) == 0.0

    def test_verdict_text(self):
        w = SentimentWindow(symbol="XYZ", start=NOW - timedelta(hours=24),
                            end=NOW, n_mentions=42, mean_polarity=0.8,
                            volume_zscore=3.1, tone_shift=0.5)
        pop = SentimentPop(window=w)
        assert "XYZ" in pop.verdict and "9.0/10" in pop.verdict
        assert "bullishness" in pop.verdict

    def test_window_to_dict(self):
        w = SentimentWindow(symbol="XYZ", start=NOW - timedelta(hours=1),
                            end=NOW, n_mentions=3, mean_polarity=-0.5,
                            volume_zscore=None, tone_shift=None)
        d = w.to_dict()
        assert d["bullishness_10"] == 2.5
        assert d["label"] == "bearish"
        assert d["volume_zscore"] is None


# -- scoring ----------------------------------------------------------------
class TestScoring:
    def test_bullish_text(self):
        sm = score_text("Huge breakout, earnings beat, upgrading to strong buy 🚀")
        assert sm.polarity > 0.3
        assert sm.label in (SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH)

    def test_bearish_text(self):
        sm = score_text("Total dump, guidance cut, bankruptcy risk. Sell everything.")
        assert sm.polarity < -0.3
        assert sm.label in (SentimentLabel.BEARISH, SentimentLabel.VERY_BEARISH)

    def test_neutral_text(self):
        sm = score_text("The company reported results on Tuesday morning.")
        assert sm.polarity == 0.0
        assert sm.label is SentimentLabel.NEUTRAL

    def test_negation_flips(self):
        sm = score_text("This is not bullish at all")
        assert sm.polarity < 0

    def test_intensifier_amplifies(self):
        mild = score_text("bullish")
        strong = score_text("extremely bullish")
        assert strong.polarity > mild.polarity

    def test_emoji_scores(self):
        sm = score_text("📉📉")
        assert sm.polarity < 0

    def test_empty_text_neutral(self):
        sm = score_text("")
        assert sm.polarity == 0.0 and sm.magnitude == 0.0

    def test_hits_recorded(self):
        sm = score_text("breakout squeeze")
        assert "breakout" in sm.hits and "squeeze" in sm.hits

    def test_batch(self):
        out = score_mentions([_mention("moon"), _mention("dump")])
        assert len(out) == 2 and out[0].polarity > 0 > out[1].polarity

    def test_llm_blend(self):
        sm = score_text("moon")
        (resc,) = rescore_with_llm([sm], advisor=lambda t: -1.0, blend=0.5)
        assert resc.polarity == pytest.approx((sm.polarity - 1.0) / 2, abs=0.01)

    def test_llm_failure_falls_back(self):
        sm = score_text("moon")
        def boom(t):
            raise RuntimeError("nope")
        (resc,) = rescore_with_llm([sm], advisor=boom)
        assert resc.polarity == sm.polarity


# -- sources (mocked network) ----------------------------------------------
REDDIT_JSON = json.dumps({"data": {"children": [
    {"data": {"id": "a1", "title": "XYZ to the moon",
               "selftext": "breakout", "author": "u1", "score": 10,
               "num_comments": 2, "created_utc": 1_700_000_000,
               "permalink": "/r/x"}}]}}).encode()

ST_JSON = json.dumps({"messages": [
    {"id": 1, "body": "XYZ dump incoming", "created_at": "2024-01-01T00:00:00Z",
     "user": {"username": "bob"}, "likes": {"total": 3},
     "entities": {"sentiment": {"basic": "Bearish"}}}]}).encode()

NEWS_RSS = b"""<rss><channel><item>
<title>XYZ beats estimates - Reuters</title>
<link>http://x</link><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
<source>Reuters</source></item></channel></rss>"""


def _urlopen(data: bytes):
    resp = io.BytesIO(data)
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: False
    return resp


class TestSources:
    def test_reddit_parses(self):
        with patch("trade_sentiment.sources._get", return_value=REDDIT_JSON):
            ms = src.RedditSource().fetch("XYZ")
        assert len(ms) == 1 and ms[0].symbol == "XYZ"
        assert ms[0].engagement == 12

    def test_stocktwits_parses(self):
        with patch("trade_sentiment.sources._get", return_value=ST_JSON):
            ms = src.StockTwitsSource().fetch("XYZ")
        assert len(ms) == 1 and ms[0].author == "bob"

    def test_news_parses(self):
        with patch("trade_sentiment.sources._get", return_value=NEWS_RSS):
            ms = src.NewsSource().fetch("XYZ")
        assert len(ms) == 1 and "Reuters" in ms[0].author

    def test_network_failure_returns_empty(self):
        with patch("trade_sentiment.sources._get",
                   side_effect=RuntimeError("down")):
            assert src.RedditSource().fetch("XYZ") == []

    def test_get_source(self):
        assert isinstance(src.get_source("reddit"), src.RedditSource)
        with pytest.raises(KeyError):
            src.get_source("x_twitter")

    def test_registry_has_three(self):
        assert set(src.SOURCES) == {"reddit", "stocktwits", "news"}


# -- aggregation ------------------------------------------------------------
def _scored(texts: list[str], start: datetime, step_s: float):
    return score_mentions([
        _mention(t, ts=start + timedelta(seconds=i * step_s))
        for i, t in enumerate(texts)
    ])


class TestAggregation:
    def test_aggregate_mean(self):
        start, end = NOW - timedelta(hours=24), NOW
        scored = _scored(["moon", "moon", "dump"], start, 3600)
        w = aggregate(scored, "XYZ", start, end)
        assert w.n_mentions == 3
        assert w.mean_polarity > 0

    def test_burst_detects_recent_surge(self):
        start, end = NOW - timedelta(hours=7), NOW
        # quiet history, then a burst of chatter in the last slice
        texts = ["the company filed a form"] * 6 + ["moon rocket squeeze"] * 30
        scored = _scored(texts, start, 3600)
        w = agg.burst_window(scored, "XYZ", start, end, n_slices=7)
        assert w.volume_zscore is not None and w.volume_zscore > 2.0
        assert w.mean_polarity > 0.2

    def test_burst_flat_chatter_no_zscore_spike(self):
        start, end = NOW - timedelta(hours=7), NOW
        texts = ["the company filed a form"] * 14
        scored = _scored(texts, start, 1800)  # even 2-per-slice spread
        w = agg.burst_window(scored, "XYZ", start, end, n_slices=7)
        assert w.volume_zscore is not None and abs(w.volume_zscore) < 2.0

    def test_detect_pops_filters_and_ranks(self):
        start, end = NOW - timedelta(hours=24), NOW
        loud = SentimentWindow("AAA", start, end, 50, 0.5, 4.0, 0.3)
        tonal = SentimentWindow("BBB", start, end, 10, 0.9, 0.5, 0.6)
        flat = SentimentWindow("CCC", start, end, 50, 0.05, 0.5, 0.0)
        tiny = SentimentWindow("DDD", start, end, 2, 0.9, 9.0, 0.8)
        pops = detect_pops([flat, tonal, loud, tiny])
        assert [p.window.symbol for p in pops] == ["AAA", "BBB"]
        assert pops[0].drivers == ()

    def test_detect_pops_drivers(self):
        start, end = NOW - timedelta(hours=24), NOW
        scored = _scored(["moon rocket"] * 8, start, 3600)
        w = SentimentWindow("XYZ", start, end, 8, 0.8, 3.0, 0.5)
        (pop,) = detect_pops([w], {"XYZ": scored})
        assert "moon" in pop.drivers


# -- pipeline ----------------------------------------------------------------
class FakeSource(src.Source):
    name = "fake"
    def __init__(self, texts: list[str]):
        self.texts = texts
    def fetch(self, symbol: str, limit: int = 50) -> list[Mention]:
        return [_mention(t, symbol) for t in self.texts[:limit]]


class TestPipeline:
    def test_scan_finds_pop(self):
        fake = FakeSource(["moon rocket squeeze 🚀"] * 12)
        with patch.dict(src.SOURCES, {"fake": lambda: fake}):
            pops = scan(["XYZ"], source_names=["fake"], min_mentions=5)
        assert len(pops) == 1
        assert pops[0].window.symbol == "XYZ"
        assert "bullishness" in pops[0].verdict

    def test_scan_empty_symbols_rejected(self):
        with pytest.raises(ValueError):
            scan([])

    def test_scan_bad_source_does_not_kill(self):
        class Bad(src.Source):
            name = "bad"
            def fetch(self, symbol, limit=50):
                raise RuntimeError("boom")
        with patch.dict(src.SOURCES, {"bad": Bad, "fake": lambda: FakeSource(["moon"] * 8)}):
            pops = scan(["XYZ"], source_names=["bad", "fake"], min_mentions=5)
        assert len(pops) == 1


# -- adapters -----------------------------------------------------------------
class TestAdapters:
    def _pop(self, polarity: float, n: int) -> SentimentPop:
        w = SentimentWindow("XYZ", NOW - timedelta(hours=1), NOW, n,
                            polarity, 3.0, 0.4)
        return SentimentPop(window=w, drivers=("moon",))

    def test_to_agent_ideas_direction(self):
        ideas = to_agent_ideas([self._pop(0.7, 40), self._pop(-0.7, 40)])
        assert [i["direction"] for i in ideas] == ["LONG", "SHORT"]
        assert all(i["agent"] == "sentiment_scout" for i in ideas)
        assert all(0 <= i["score"] <= 1 for i in ideas)

    def test_to_signal_overlay_threshold(self):
        pops = [self._pop(0.8, 100), self._pop(-0.8, 3)]
        overlay = to_signal_overlay(pops, min_conviction=5.0)
        assert overlay == {"XYZ": "LONG"} or overlay.get("XYZ") == "LONG"

    def test_to_dashboard_rows(self):
        rows = to_dashboard_rows([self._pop(0.5, 10)])
        assert rows[0]["verdict"].startswith("XYZ")

    def test_describe(self):
        assert describe()["name"] == "trade-sentiment"


# -- cli ----------------------------------------------------------------------
class TestCLI:
    def test_score(self, capsys):
        assert main(["score", "moon rocket"]) == 0
        out = capsys.readouterr().out
        assert "polarity" in out

    def test_sources(self, capsys):
        assert main(["sources"]) == 0
        assert "reddit" in capsys.readouterr().out

    def test_scan_uses_pipeline(self, capsys):
        pop = SentimentPop(SentimentWindow(
            "XYZ", NOW - timedelta(hours=1), NOW, 10, 0.8, 3.0, 0.4))
        with patch("trade_sentiment.cli.scan", return_value=[pop]):
            assert main(["scan", "XYZ"]) == 0
        assert "XYZ" in capsys.readouterr().out

    def test_scan_no_symbols_fails(self):
        assert main(["scan"]) == 2

    def test_config_after_subcommand(self, tmp_path, capsys):
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"symbols": ["XYZ"]}))
        pop = SentimentPop(SentimentWindow(
            "XYZ", NOW - timedelta(hours=1), NOW, 10, 0.8, 3.0, 0.4))
        with patch("trade_sentiment.cli.scan", return_value=[pop]) as m:
            assert main(["scan", "--config", str(cfg)]) == 0
        assert m.call_args[0][0] == ["XYZ"]
