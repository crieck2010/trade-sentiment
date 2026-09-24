# Changelog — trade-sentiment

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
