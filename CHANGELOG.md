# Changelog — trade-sentiment

## v0.3.0 (2026-09-26)

- **Point-in-time sentiment archive** (`trade_sentiment.archive`,
  stdlib sqlite3 — the suite pattern from trade-paper's ledger):
  `Archive` records every observation with BOTH `observed_at` (the
  as-of time the sentiment refers to: the source publication
  timestamp) and `recorded_at` (when it entered the archive: the fetch
  time), plus symbol/topic, source, scores, and raw reference text/URI.
  `query(symbol, as_of)` returns only observations with
  `recorded_at <= as_of` — the no-lookahead contract, enforced in SQL
  with no opt-out; `query_range(symbol, start, end)` replays what was
  knowable at `end`; `coverage()` states what the archive holds (None
  when empty — the honest gap); `prune(days)` drops old
  `observed_at` rows. A write with `observed_at > recorded_at` is
  rejected outright, so a provider that stamps the as-of time with the
  fetch time fails loudly at insert.
- `scan(..., archive=None)` — record-on-fetch: every scored mention
  archived with its source publication timestamp as `observed_at`.
  Default `None` keeps the live API backward compatible; archiving
  failures never kill the scan. CLI: `scan --archive DB`, plus a new
  `archive` subcommand (`query` / `coverage` / `prune`).
- **No-lookahead test** (`tests/test_archive.py::TestNoLookahead`):
  observations recorded after T are invisible at `as_of=T` (and
  visible once `as_of` passes their `recorded_at`); a provider fixture
  that publishes at T but is fetched at T+10d — while holding tempting
  future text — verifies the archived row carries the publication
  timestamp and the published text only.
- **Backfill investigation (bounded, honest gap)**: GDELT 2.0 probed
  as the keyless historical candidate. The DOC 2.0 query API
  (rate-refuses programmatic queries, three probes) is not cleanly
  reachable; the 2.0 bulk files are reachable back to 2015-02-18 but
  have no ticker-addressable query path without multi-GB downloads and
  noisy org-name matching — so **no backfill provider ships**.
  Documented in `docs/BACKFILL.md`: the archive covers deployment
  forward only, and no history is fabricated.
- Docs: new `docs/ARCHIVE.md` (schema, API, retention/coverage, and a
  `## The maths` note on point-in-time correctness / lookahead bias),
  new `docs/BACKFILL.md`; README gains the archive section with "The
  maths", the coverage gap statement, and CLI updates; new
  `examples/archive_example.py`.
- 21 new tests. 83 passed, 1 skipped (live-model FinBERT test, gated).

## v0.2.0 (2026-09-24)

- **FinBERT rescoring adapter** (`trade_sentiment.finbert`, optional):
  blends lexicon tone with ProsusAI/finbert — `FinBERTScorer` (batched
  inference, lazy `torch`/`transformers`, one shared load per scan),
  `logits_to_polarity` (softmax → P(positive) − P(negative)), and
  `rescore_with_finbert` / `finbert_rescorer`, mirroring the existing
  `rescore_with_llm` seam with plain `ScoredMention` lists in and out.
  Without the ML stack installed everything degrades to pure lexicon
  instead of raising; core stays stdlib-only.
- `pipeline.scan()` gains an optional fail-soft `rescorer` hook applied
  per symbol after lexicon scoring; CLI gets `scan --rescore finbert
  --blend` and `score --model finbert`.
- Docs: new `docs/FINBERT.md` (setup, usage, the maths, scaling,
  limitations), `docs/SCORING.md` + `docs/ARCHITECTURE.md` updated,
  README gains a FinBERT section with "The maths".
  `examples/finbert_example.py` runs without the ML stack via a fake
  scorer.
- 22 new tests (all torch-free; one live-model test gated behind
  `TRADE_SENTIMENT_FINBERT_LIVE=1`). 62 passed, 1 skipped.

## v0.1.0 (2026-09-23)

Initial release.

- Social/news sentiment engine: Reddit, StockTwits, and news-RSS chatter
  monitoring for any ticker, keyless and ToS-clean.
- Deterministic finance-tuned lexicon scorer (phrases, negation windows,
  intensifiers, emoji, caps/exclamation boosts) with per-score `hits`
  for auditability; LLM-rescoring seam included.
- Dual 0–10 output: `bullishness_10` (pure tone) and `conviction_10`
  (tone × volume), so strong-but-quiet sentiment reads differently from
  moderate-but-loud sentiment.
- History-free burst detection: the scan window is time-sliced and the
  latest slice is z-tested against earlier slices.
- Plain-English verdicts ("XYZ has a pop in sentiment of 9/10
  bullishness") plus JSON, agent-idea, and signal-overlay outputs.
- Interop adapters: `to_agent_ideas` (trade-agents `sentiment_scout`),
  `to_signal_overlay` (trade-strategies gate), `to_dashboard_rows`.
- CLI: `scan`, `score`, `sources`, `license`, `update-check`; `--config`
  works before or after the subcommand.
- License-key and update-check hooks (suite convention).
- 40 tests, all passing. Stdlib-only, no third-party dependencies.
