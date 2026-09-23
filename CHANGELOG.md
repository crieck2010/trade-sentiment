# Changelog — trade-sentiment

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
