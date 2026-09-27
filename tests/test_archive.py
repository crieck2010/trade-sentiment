"""Tests for the point-in-time sentiment archive.

The load-bearing test is ``TestNoLookahead``: observations recorded after
decision time T must be invisible to ``query(symbol, as_of=T)``, and an
observation timestamped T must never incorporate information from after T.
Network is never touched; the provider fixtures below are synthetic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from trade_sentiment import (
    Archive,
    ArchivedObservation,
    Mention,
    ScoredMention,
    score_text,
)
from trade_sentiment.archive import DEFAULT_ARCHIVE_PATH

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)  # decision time in the tests


def _scored(text: str, symbol: str = "XYZ",
            ts: datetime | None = None) -> ScoredMention:
    m = Mention(id="t1", source="reddit", symbol=symbol, text=text,
                timestamp=ts or T0, url="https://example.com/x")
    sm = score_text(text)
    return ScoredMention(mention=m, polarity=sm.polarity,
                         magnitude=sm.magnitude, label=sm.label,
                         hits=sm.hits)


@pytest.fixture
def arc() -> Archive:
    return Archive(":memory:")


# -- basic roundtrip -------------------------------------------------------
class TestRoundtrip:
    def test_record_and_query(self, arc: Archive):
        arc.record(symbol="xyz", source="reddit", observed_at=T0,
                   recorded_at=T0 + timedelta(hours=1), polarity=0.7,
                   magnitude=0.8, label="bullish", text="to the moon",
                   url="https://example.com/a")
        rows = arc.query("XYZ", as_of=T0 + timedelta(hours=2))
        assert len(rows) == 1
        o = rows[0]
        assert isinstance(o, ArchivedObservation)
        assert o.symbol == "XYZ"  # upper-cased on write
        assert o.polarity == pytest.approx(0.7)
        assert o.observed_at == T0
        assert o.recorded_at == T0 + timedelta(hours=1)

    def test_record_scored_uses_publication_timestamp(self, arc: Archive):
        # observed_at must be the SOURCE publication time, not fetch time.
        pub = T0 - timedelta(days=10)
        sm = _scored("Huge breakout, earnings beat", ts=pub)
        fetch = T0  # fetched ten days after publication
        arc.record_scored(sm, recorded_at=fetch, scorer="lexicon")
        rows = arc.query("XYZ", as_of=fetch)
        assert len(rows) == 1
        assert rows[0].observed_at == pub
        assert rows[0].recorded_at == fetch
        assert rows[0].meta["scorer"] == "lexicon"
        assert rows[0].meta["mention_id"] == "t1"

    def test_observed_after_recorded_rejected(self, arc: Archive):
        with pytest.raises(ValueError, match="never incorporate"):
            arc.record(symbol="XYZ", source="reddit",
                       observed_at=T0 + timedelta(seconds=1),
                       recorded_at=T0, polarity=0.1)

    def test_future_publication_timestamp_rejected(self, arc: Archive):
        # A provider that stamps observed_at with fetch time (or later)
        # fails loudly instead of poisoning the archive.
        sm = _scored("late headline", ts=T0 + timedelta(hours=5))
        with pytest.raises(ValueError):
            arc.record_scored(sm, recorded_at=T0)

    def test_empty_coverage_is_honest(self, arc: Archive):
        assert arc.coverage("XYZ") is None
        assert arc.coverage() is None

    def test_coverage_bounds(self, arc: Archive):
        arc.record(symbol="XYZ", source="news", observed_at=T0,
                   recorded_at=T0 + timedelta(hours=1))
        arc.record(symbol="XYZ", source="news",
                   observed_at=T0 + timedelta(days=2),
                   recorded_at=T0 + timedelta(days=2, hours=1))
        cov = arc.coverage("XYZ")
        assert cov is not None
        assert cov["n_observations"] == 2
        assert cov["from_observed"].startswith("2026-09-01")
        assert cov["to_observed"].startswith("2026-09-03")


# -- the load-bearing no-lookahead contract --------------------------------
class TestNoLookahead:
    def test_late_recordings_excluded(self, arc: Archive):
        """Observations recorded after T are invisible at as_of=T."""
        early = arc.record(symbol="XYZ", source="reddit",
                           observed_at=T0 - timedelta(days=2),
                           recorded_at=T0 - timedelta(days=1),
                           polarity=0.5, text="steady chatter")
        arc.record(symbol="XYZ", source="reddit",
                   observed_at=T0 - timedelta(hours=1),
                   recorded_at=T0 + timedelta(hours=1),  # known only later
                   polarity=0.95, text="earnings leak")
        rows = arc.query("XYZ", as_of=T0)
        assert [r.text for r in rows] == ["steady chatter"]
        assert all(r.recorded_at <= T0 for r in rows)

        # ...but the same observation is visible once we move as_of past it.
        rows = arc.query("XYZ", as_of=T0 + timedelta(hours=2))
        assert len(rows) == 2

    def test_query_range_is_point_in_time(self, arc: Archive):
        """query_range(start, end) == knowledge available at ``end``."""
        arc.record(symbol="XYZ", source="news", observed_at=T0,
                   recorded_at=T0 + timedelta(days=5),  # scraped late
                   polarity=-0.8, text="scandal breaks")
        rows = arc.query_range("XYZ", T0 - timedelta(days=1), T0)
        assert rows == []  # the late scrape must not leak into T0's replay

    def test_observation_timestamped_T_uses_no_post_T_info(self, arc: Archive):
        """Provider fixture that would leak if the archive were miswired.

        The fixture below simulates a provider that publishes at T but is
        only fetched at T+10d — and that TEMPTINGLY holds ``future_news``
        text describing events at T+9d.  A miswired archive that recorded
        the provider's current (future) text, or stamped observed_at with
        the fetch time, would fail these assertions.
        """

        class LeakyFixture:
            # text the provider published at T (past)
            published_at_T = "Solid quarter, revenue up, guidance steady"
            # text describing the FUTURE — must never enter the archive
            future_news = "CEO resigns amid accounting fraud (T+9d)"

            def fetch(self, symbol, limit=50):
                # mentions carry the publication timestamp, not fetch time
                return [Mention(id="fx1", source="news", symbol=symbol,
                                text=self.published_at_T, timestamp=T0,
                                url="https://example.com/past")]

        with patch("trade_sentiment.pipeline.fetch_all") as fake_fetch:
            prov = LeakyFixture()
            fake_fetch.side_effect = (
                lambda symbols, *a, **k:
                {s: prov.fetch(s) for s in symbols})
            # fetch happens at T+10d
            now = T0 + timedelta(days=10)
            with patch("trade_sentiment.pipeline.datetime") as dt_mock:
                from datetime import datetime as real_dt
                dt_mock.now.side_effect = (
                    lambda tz=None: now if tz else now.replace(tzinfo=None))
                from trade_sentiment import scan
                scan(["XYZ"], source_names=["news"], archive=arc,
                     window_hours=24 * 30, min_mentions=1)

        rows = arc.query("XYZ", as_of=T0 + timedelta(days=11))
        assert len(rows) == 1
        o = rows[0]
        assert o.observed_at == T0, \
            "observed_at must be the publication time, not the fetch time"
        assert o.recorded_at >= T0 + timedelta(days=10)
        assert o.text == LeakyFixture.published_at_T
        assert "fraud" not in o.text, "future information leaked into archive"

        # At decision time T the observation is NOT yet knowable
        # (recorded at T+10d) — no lookahead, even though observed_at <= T.
        assert arc.query("XYZ", as_of=T0) == []

    def test_contract_cannot_be_opted_out(self, arc: Archive):
        """No query flag bypasses recorded_at <= as_of."""
        arc.record(symbol="XYZ", source="reddit", observed_at=T0,
                   recorded_at=T0 + timedelta(hours=1), polarity=0.9)
        for kwargs in ({}, {"sources": ["reddit"]}, {"limit": 100},
                       {"since": T0 - timedelta(days=1)}):
            assert arc.query("XYZ", as_of=T0, **kwargs) == []


# -- retention --------------------------------------------------------------
class TestRetention:
    def test_prune_by_observed_at(self, arc: Archive):
        arc.record(symbol="XYZ", source="news", observed_at=T0,
                   recorded_at=T0)
        arc.record(symbol="XYZ", source="news",
                   observed_at=T0 - timedelta(days=40),
                   recorded_at=T0 - timedelta(days=40))
        removed = arc.prune(30, now=T0)
        assert removed == 1
        assert arc.coverage("XYZ")["n_observations"] == 1

    def test_prune_does_not_create_lookahead(self, arc: Archive):
        # pruning only narrows coverage; it never makes late-recorded
        # observations visible at earlier decision times.
        arc.record(symbol="XYZ", source="reddit",
                   observed_at=T0 - timedelta(days=10),
                   recorded_at=T0, polarity=0.4)
        arc.prune(30, now=T0)
        assert arc.query("XYZ", as_of=T0 - timedelta(days=1)) == []


# -- scan wiring ------------------------------------------------------------
class TestScanWiring:
    def test_scan_without_archive_unchanged(self):
        from trade_sentiment import scan
        with patch("trade_sentiment.pipeline.fetch_all") as fake_fetch:
            fake_fetch.return_value = {
                "XYZ": [_scored("great quarter, beat estimates").mention]}
            pops = scan(["XYZ"], window_hours=24, min_mentions=1)
        assert isinstance(pops, list)

    def test_scan_records_on_fetch(self, arc: Archive):
        from trade_sentiment import scan
        mentions = [_scored("great quarter, beat estimates",
                            ts=T0 - timedelta(hours=2)).mention,
                    _scored("analysts upgrade to buy",
                            ts=T0 - timedelta(hours=1)).mention]
        with patch("trade_sentiment.pipeline.fetch_all") as fake_fetch:
            fake_fetch.return_value = {"XYZ": mentions}
            scan(["XYZ"], window_hours=24, min_mentions=1, archive=arc)
        # recorded_at is the real fetch time (now); as_of must be after it
        rows = arc.query("XYZ",
                         as_of=datetime.now(timezone.utc) + timedelta(minutes=5))
        assert len(rows) == 2
        assert {o.text for o in rows} == {
            "great quarter, beat estimates", "analysts upgrade to buy"}
        assert all(o.observed_at < o.recorded_at for o in rows)

    def test_archive_failure_never_kills_scan(self):
        from trade_sentiment import scan
        broken = Archive(":memory:")
        broken.close()  # writes now raise
        with patch("trade_sentiment.pipeline.fetch_all") as fake_fetch:
            fake_fetch.return_value = {
                "XYZ": [_scored("great quarter, beat estimates").mention]}
            pops = scan(["XYZ"], window_hours=24, min_mentions=1,
                        archive=broken)
        assert isinstance(pops, list)


# -- CLI --------------------------------------------------------------------
class TestArchiveCLI:
    def test_archive_query_cli(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        arc = Archive(db)
        arc.record(symbol="XYZ", source="reddit", observed_at=T0,
                   recorded_at=T0 + timedelta(minutes=5), polarity=0.6,
                   text="steady")
        arc.close()
        from trade_sentiment.cli import main
        rc = main(["archive", "query", "XYZ",
                   "--as-of", "2026-09-02T00:00:00+00:00",
                   "--db", str(db)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "steady" in out

    def test_archive_query_hides_future_recordings(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        arc = Archive(db)
        arc.record(symbol="XYZ", source="reddit", observed_at=T0,
                   recorded_at=T0 + timedelta(days=3), polarity=0.9,
                   text="future leak")
        arc.close()
        from trade_sentiment.cli import main
        rc = main(["archive", "query", "XYZ",
                   "--as-of", "2026-09-02T00:00:00+00:00",
                   "--db", str(db)])
        assert rc == 0
        assert "future leak" not in capsys.readouterr().out

    def test_archive_coverage_cli(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        from trade_sentiment.cli import main
        assert main(["archive", "coverage", "--db", str(db)]) == 0
        assert "archive is empty" in capsys.readouterr().out

    def test_archive_prune_cli(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        arc = Archive(db)
        arc.record(symbol="XYZ", source="news",
                   observed_at=T0 - timedelta(days=90),
                   recorded_at=T0 - timedelta(days=90))
        arc.close()
        from trade_sentiment.cli import main
        assert main(["archive", "prune", "--db", str(db),
                     "--days", "30"]) == 0
        assert "pruned 1" in capsys.readouterr().out

    def test_scan_archive_cli_flag(self, tmp_path):
        db = tmp_path / "a.db"
        from trade_sentiment.cli import main
        with patch("trade_sentiment.pipeline.fetch_all") as fake_fetch:
            fake_fetch.return_value = {
                "XYZ": [_scored("great quarter, beat estimates",
                                ts=T0 - timedelta(hours=1)).mention]}
            rc = main(["scan", "XYZ", "--sources", "news",
                       "--archive", str(db), "--min-mentions", "1"])
        assert rc == 0
        assert Archive(db).coverage("XYZ")["n_observations"] == 1


def test_default_path_documented():
    assert str(DEFAULT_ARCHIVE_PATH).endswith("sentiment-archive.db")
