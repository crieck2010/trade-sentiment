# Architecture — trade-sentiment

## Design principles (suite-wide)

- **Pure-logic engine**: zero UI-framework imports anywhere under `src/`.
- **Stdlib-only core**: no third-party runtime dependencies. Optional
  siblings (`trade-agents`, dashboards) are integrated through *lazy*
  adapters, never hard imports.
- **Deterministic**: the same mentions always produce the same scores.
  No randomness, no hidden model weights, no network at score time.
- **Explainable**: every score carries the lexicon `hits` that produced it.

## Module map

```
src/trade_sentiment/
    models.py       # Mention, ScoredMention, SentimentWindow, SentimentPop
                    # + the 0-10 scale functions (bullishness_10, conviction_10)
    lexicon.py      # finance-tuned word/phrase/emoji scores, negations
    scoring.py      # deterministic scorer + LLM rescoring seam
    sources.py      # Source ABC + Reddit / StockTwits / News adapters
    aggregation.py  # windowing, baselines, burst detection, pop ranking
    pipeline.py     # scan(): threaded fetch → score → aggregate → pops
    adapters.py     # to_agent_ideas / to_signal_overlay / to_dashboard_rows
    cli.py          # scan / score / sources / license / update-check
    licensing.py    # license-key + update-check hooks (suite convention)
```

## Data flow

```
sources.fetch(symbol)            (thread pool, one job per symbol×source)
        │ Mention[]
        ▼
scoring.score_mentions()         (pure, ~thousands/sec)
        │ ScoredMention[]  (polarity, label, hits)
        ▼
aggregation.burst_window()       (time-sliced baseline, no history needed)
        │ SentimentWindow  (n, mean_polarity, volume_zscore, tone_shift,
        │                  bullishness_10, conviction_10)
        ▼
aggregation.detect_pops()        (volume OR tone clears its bar; ranked)
        │ SentimentPop[]   (verdict, drivers)
        ▼
adapters.*  →  trade-agents ideas / strategy overlay / dashboard rows
```

## Burst detection without history

A fresh scan has no stored baseline, so the trailing window is cut into
`n_slices` equal time slices. The latest slice is the "now" under test;
earlier slices form the baseline. `volume_zscore` is the z-score of the
latest slice's mention count against the earlier slices — chatter
concentrating *right now* spikes it. `tone_shift` is the latest slice's
mean polarity minus the baseline mean. A later version can persist
windows (SQLite) and use real multi-day baselines; the `aggregate()`
`baseline` parameter already accepts them.

## Scaling

| Concern | v0.1.0 answer | Future |
|---|---|---|
| Fetch latency | thread pool across symbol×source; per-source politeness delays | async + persistent cache |
| Scoring throughput | lexicon scan, single pass | unchanged — already cheap |
| History | in-memory per scan | SQLite window store for real baselines |
| Source count | 3 keyless | X (keyed), broker feeds via `Source` ABC |
| Rate limits | polite delays, fail-soft per source | per-source token buckets |

One bad source never kills a scan: `fetch_all` catches per-job exceptions
and that source contributes zero mentions.

## Failure semantics

- Network error / bad payload → source returns `[]`, scan continues.
- No mentions → window with `n_mentions=0`, polarity 0; never a pop.
- LLM advisor raises → that mention keeps its lexicon score.
- Unknown source name → `KeyError` naming the available sources.

## Sentiment spectrum

Polarity −1…+1 buckets: ≤−0.6 very bearish, ≤−0.2 bearish, <0.2 neutral,
<0.6 bullish, else very bullish. `bullishness_10 = (polarity+1)/2×10`
(linear tone). `conviction_10 = bullishness × (0.5 + min(n,100)/100)`
capped at 10 (tone × volume). See `docs/SCORING.md`.
