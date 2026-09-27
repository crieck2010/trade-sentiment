"""Point-in-time sentiment archive.  Stdlib sqlite3, suite pattern.

Every sentiment observation is recorded with **both** timestamps:

- ``observed_at`` — the as-of time the sentiment *refers to*
  (for a scored mention: the source publication timestamp).  An
  observation timestamped T must never incorporate information that only
  became available after T.
- ``recorded_at`` — when the observation *entered the archive*
  (the fetch time).  Always >= ``observed_at``.

No-lookahead contract (load-bearing — see ``tests/test_archive.py`` and
``docs/ARCHIVE.md``): any screening decision made at decision time D may
use **only** observations with ``recorded_at <= D``.  ``Archive.query()``
enforces this predicate in SQL and it cannot be opted out of.
Filtering on ``observed_at`` alone is not point-in-time safe: a headline
published at 09:30 but scraped at 16:00 can carry information (edits,
engagement counts, corrections) from 16:00.

Coverage: the archive accumulates observations from deployment forward.
There is no history before the first recorded observation — never
fabricate history (see ``docs/BACKFILL.md`` for the GDELT investigation
and the honest gap statement).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import ArchivedObservation, ScoredMention

DEFAULT_ARCHIVE_PATH = (
    Path.home() / ".trade-sentiment" / "sentiment-archive.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'mention',
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    polarity REAL,
    magnitude REAL,
    label TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_obs_symbol_recorded
    ON observations(symbol, recorded_at);
CREATE INDEX IF NOT EXISTS idx_obs_symbol_observed
    ON observations(symbol, observed_at);
"""

_MAX_TEXT = 2000  # raw reference text is truncated at record time


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Archive:
    """SQLite point-in-time store for sentiment observations.

    ``path`` may be a file path, ``":memory:"``, or None (the default
    on-disk location ``~/.trade-sentiment/sentiment-archive.db``).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            self.path = DEFAULT_ARCHIVE_PATH
        else:
            self.path = Path(path) if path != ":memory:" else ":memory:"
        if self.path != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(self.path) if self.path != ":memory:" else ":memory:")
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- writes ------------------------------------------------------------
    def record(
        self,
        *,
        symbol: str,
        source: str,
        observed_at: datetime,
        polarity: float | None = None,
        magnitude: float | None = None,
        label: str = "",
        text: str = "",
        url: str = "",
        topic: str = "",
        kind: str = "mention",
        recorded_at: datetime | None = None,
        meta: dict | None = None,
    ) -> int:
        """Record one observation; returns the row id.

        Raises ``ValueError`` if ``observed_at`` is after ``recorded_at``
        — an observation can never refer to a time after it was archived.
        This invariant is what makes the archive point-in-time safe.
        """
        rec = recorded_at or datetime.now(timezone.utc)
        obs = observed_at
        if obs.tzinfo is None:
            obs = obs.replace(tzinfo=timezone.utc)
        if rec.tzinfo is None:
            rec = rec.replace(tzinfo=timezone.utc)
        if obs > rec:
            raise ValueError(
                f"observed_at ({obs.isoformat()}) is after recorded_at "
                f"({rec.isoformat()}): an observation must never incorporate "
                "information from after it was archived")
        cur = self._db.execute(
            """INSERT INTO observations
               (symbol, topic, source, kind, observed_at, recorded_at,
                polarity, magnitude, label, text, url, meta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol.upper(), topic, source, kind, _utc_iso(obs),
             _utc_iso(rec), polarity, magnitude, label, text[:_MAX_TEXT],
             url, json.dumps(meta or {})),
        )
        self._db.commit()
        return cur.lastrowid

    def record_scored(
        self,
        scored: ScoredMention,
        *,
        recorded_at: datetime | None = None,
        scorer: str = "lexicon",
        topic: str = "",
    ) -> int:
        """Archive one scored mention (record-on-fetch).

        ``observed_at`` comes from the mention's source publication
        timestamp — never the fetch time.  ``recorded_at`` defaults to
        now (the fetch).  If the publication timestamp is in the future
        relative to the recording time the write is rejected: a miswired
        provider that stamps ``observed_at`` with fetch time fails loudly
        here instead of silently poisoning the archive.
        """
        m = scored.mention
        return self.record(
            symbol=m.symbol, source=m.source, observed_at=m.timestamp,
            polarity=scored.polarity, magnitude=scored.magnitude,
            label=scored.label.value, text=m.text, url=m.url,
            topic=topic, kind="mention", recorded_at=recorded_at,
            meta={"scorer": scorer, "hits": list(scored.hits),
                  "mention_id": m.id, "author": m.author,
                  "engagement": m.engagement},
        )

    # -- reads -------------------------------------------------------------
    def query(
        self,
        symbol: str,
        as_of: datetime | str,
        *,
        sources: list[str] | None = None,
        topic: str | None = None,
        since: datetime | str | None = None,
        limit: int | None = None,
    ) -> list[ArchivedObservation]:
        """Point-in-time query: observations knowable at ``as_of``.

        Returns only rows with ``recorded_at <= as_of`` — this predicate
        is the no-lookahead contract and is applied unconditionally.
        Rows are ordered by ``observed_at`` (oldest first).  ``since``
        bounds ``observed_at`` from below; ``limit`` caps the row count.
        """
        if isinstance(as_of, str):
            as_of = _parse_iso(as_of)
        if isinstance(since, str):
            since = _parse_iso(since)
        sql = ("SELECT * FROM observations WHERE symbol = ? "
               "AND recorded_at <= ?")
        params: list = [symbol.upper(), _utc_iso(as_of)]
        if sources:
            sql += " AND source IN (%s)" % ",".join("?" * len(sources))
            params.extend(sources)
        if topic is not None:
            sql += " AND topic = ?"
            params.append(topic)
        if since is not None:
            sql += " AND observed_at >= ?"
            params.append(_utc_iso(since))
        sql += " ORDER BY observed_at ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_obs(r)
                for r in self._db.execute(sql, params).fetchall()]

    def query_range(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        **kwargs,
    ) -> list[ArchivedObservation]:
        """Replay helper: what was knowable about ``[start, end]`` at ``end``.

        Equivalent to ``query(symbol, as_of=end, since=start)`` — the
        decision made at ``end`` sees only observations recorded by
        ``end``, even those observed earlier in the window.
        """
        return self.query(symbol, end, since=start, **kwargs)

    def coverage(self, symbol: str | None = None) -> dict | None:
        """Min/max ``observed_at`` and row counts.

        ``coverage(symbol)`` for one symbol, ``coverage()`` for the whole
        archive.  Returns None when the archive is empty — no history
        before deployment exists, and this says so plainly.
        """
        if symbol is None:
            row = self._db.execute(
                "SELECT MIN(observed_at) lo, MAX(observed_at) hi, "
                "COUNT(*) n, COUNT(DISTINCT symbol) nsym FROM observations"
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT MIN(observed_at) lo, MAX(observed_at) hi, "
                "COUNT(*) n FROM observations WHERE symbol = ?",
                (symbol.upper(),)).fetchone()
        if row["n"] == 0:
            return None
        out = {"from_observed": row["lo"], "to_observed": row["hi"],
               "n_observations": row["n"]}
        if symbol is None:
            out["n_symbols"] = row["nsym"]
        return out

    # -- retention ----------------------------------------------------------
    def prune(self, days: int, *,
              now: datetime | None = None) -> int:
        """Delete observations with ``observed_at`` older than ``days``.

        Returns the number of rows removed.  Pruning is by
        ``observed_at`` (what the sentiment refers to), not
        ``recorded_at`` — after pruning, point-in-time replays with
        ``as_of`` earlier than the new coverage start are incomplete, and
        ``coverage()`` will say so.  Retention is a coverage statement,
        not a lookahead fix: pruning never repairs a lookahead leak.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = _utc_iso(now - timedelta(days=days))
        cur = self._db.execute(
            "DELETE FROM observations WHERE observed_at < ?", (cutoff,))
        self._db.commit()
        return cur.rowcount

    def close(self) -> None:
        self._db.close()

    def _row_to_obs(self, row: sqlite3.Row) -> ArchivedObservation:
        return ArchivedObservation(
            symbol=row["symbol"], source=row["source"],
            observed_at=_parse_iso(row["observed_at"]),
            recorded_at=_parse_iso(row["recorded_at"]),
            polarity=row["polarity"], magnitude=row["magnitude"],
            label=row["label"], text=row["text"], url=row["url"],
            topic=row["topic"], kind=row["kind"],
            meta=json.loads(row["meta"] or "{}"),
        )
