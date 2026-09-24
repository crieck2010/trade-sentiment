# trade-sentiment

Social/news sentiment engine for the **trade-suite**: it monitors public chatter
about a ticker, scores the language on a very-bearish → very-bullish spectrum,
and emits *sentiment pops* — plain-English verdicts like:

> **XYZ has a pop in sentiment of 9/10 bullishness** (conviction 5.2/10, very
> bullish) — 12 mentions in the window.

Those pops feed the research desk as trade ideas: the output is designed to be
an input for identifying equities (and other instruments) to trade on the basis
of sentiment, momentum, and conversational chatter.

Part of the [trade-suite](https://github.com/crieck2010/trade-suite) algorithmic
and agentic trading system. Research/backtesting/paper-trading only — never live
trading, never personalized investment advice.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-sentiment.git
```

Requires Python 3.10+. No third-party dependencies — the core is stdlib-only.

## Quick start

```bash
# score one piece of text
trade-sentiment score "Huge breakout, earnings beat, upgrading to strong buy"

# scan tickers for sentiment pops (verdicts, JSON, agent ideas, or overlay)
trade-sentiment scan AAPL NVDA
trade-sentiment scan AAPL NVDA --format json
trade-sentiment scan AAPL NVDA --format ideas > ideas.json
trade-sentiment scan AAPL NVDA --sources reddit stocktwits --window 12

# list sources
trade-sentiment sources
```

```python
from trade_sentiment import scan, score_text

sm = score_text("Short squeeze incoming, loading calls")
print(sm.polarity, sm.label.value)   # 0.62 very bullish

pops = scan(["AAPL", "NVDA"])
for pop in pops:
    print(pop.verdict)
```

## The two scores

Every pop reports **two numbers**, because a loud crowd and an intense few are
different signals:

| Score | Meaning | Example |
|---|---|---|
| `bullishness_10` | Pure tone, 0–10 (linear map of polarity) | 9.0 = very bullish language |
| `conviction_10` | Tone × chatter volume, 0–10 | 5.0 = strong tone, few mentions; 9.0 = moderate tone, huge crowd |

So *"9/10 bullishness with 5 mentions"* (conviction ~5) reads differently from
*"6/10 bullishness with 200 mentions"* (conviction ~9). See `docs/SCORING.md`
for the exact formulas.

## FinBERT rescoring (v0.2.0, optional)

The lexicon is fast and explainable but context-blind. The optional
FinBERT adapter blends its tone with
[FinBERT](https://huggingface.co/ProsusAI/finbert) — a BERT model
fine-tuned on finance professionals' labels — for negation scope,
slang, and finance-specific word senses the lexicon misses:

```bash
pip install torch transformers   # CPU torch is fine; ~440 MB model download on first use
trade-sentiment score --model finbert "earnings beat, guidance raised"
trade-sentiment scan AAPL NVDA --rescore finbert --blend 0.5
```

```python
from trade_sentiment import scan
from trade_sentiment.finbert import finbert_rescorer

# one shared model load across all symbols; lexicon fallback if the
# ML stack is absent
pops = scan(["AAPL", "NVDA"], rescorer=finbert_rescorer(blend=0.5))
```

**The maths.** *What you learn:* a context-aware tone per mention in
[−1, 1], blended with the auditable lexicon score — FinBERT moves the
tone, the lexicon keeps the receipt (`hits`, `magnitude` stay
lexicon-side). *Why it matters:* hand-written lexicons can't cover novel
phrasing ("priced in", "inverse Cramer") or negation scope ("not exactly
a blowout"); a model trained on labeled financial sentences
generalizes. The cost is opacity and speed — hence a blend, not a
replacement.

```
P(c)     = softmax(logits)_c              # FinBERT class probabilities
polarity = P(positive) − P(negative)      # ∈ [−1, 1]; neutral mass → 0
blended  = (1 − β) · lexicon + β · finbert   # β = 0.5 default
```

Keep `β ≤ 0.5` until benchmarked on your own labeled chatter. Full
detail: `docs/FINBERT.md`.

## Sources (v0.1.0)

| Source | Access | Notes |
|---|---|---|
| Reddit | keyless public JSON | finance subs: wallstreetbets, stocks, investing, … |
| StockTwits | keyless public API | symbol streams |
| News | keyless RSS | Google News headlines (milder tone than social) |

X/Twitter is intentionally **not** included in v0.1.0: the official API is paid
and scraping violates its ToS. The `Source` ABC is the seam — see
`docs/SOURCES.md` for how the X adapter will plug in once a key exists.

## How it works

1. **Fetch** — mentions pulled from every source in parallel (thread pool).
2. **Score** — finance-tuned lexicon: phrase-first matching, negation windows
   ("not bullish" → bearish), intensifiers, emoji, caps/exclamation boosts.
3. **Aggregate** — per-symbol rolling window; the window is time-sliced so a
   *burst* (chatter concentrating right now) is detectable with no history.
4. **Detect** — a pop needs volume *or* tone to clear its bar; ranked
   loudest-first.

Full detail: `docs/ARCHITECTURE.md`, `docs/SCORING.md`.

## Interoperability

- `trade_sentiment.adapters.to_agent_ideas(pops)` → idea dicts for
  **trade-agents** (consumed by the `sentiment_scout` researcher, v0.1.2+).
- `to_signal_overlay(pops)` → `{symbol: "LONG"|"SHORT"}` gate for
  **trade-strategies** signals.
- `to_dashboard_rows(pops)` → flat JSON rows for dashboard tables.
- CLI `--format ideas|overlay|json` for shell pipelines.

## CLI reference

```
trade-sentiment scan SYMBOLS... [--sources reddit stocktwits news]
                                [--window HOURS] [--min-mentions N]
                                [--limit N] [--format verdicts|json|ideas|overlay]
                                [--rescore finbert] [--blend 0.5]
trade-sentiment score [TEXT] [--model lexicon|finbert]   # or pipe via stdin
trade-sentiment sources
trade-sentiment license
trade-sentiment update-check
```

`--config` works both before and after the subcommand; config JSON keys:
`symbols`, `sources`, `window_hours`, `rescore`, `blend`, `model`.

## 3×-daily runner

```bash
# crontab: 10:00, 13:00, 15:30 America/New_York, weekdays
0 10,13 * * 1-5 trade-sentiment scan --config sentiment-config.json --format ideas >> ideas.jsonl
30 15 * * 1-5 trade-sentiment scan --config sentiment-config.json --format ideas >> ideas.jsonl
```

## Limitations (read these)

- Lexicon scoring is a heuristic, not a model: sarcasm, memes, and novel slang
  ("priced in", "inverse Cramer") can mislead. The LLM-rescoring seam and the
  optional FinBERT adapter exist for this reason.
- Social chatter skews bullish and noisy; treat pops as *candidates for
  research*, not signals.
- Free sources are rate-limited and delayed; Reddit/StockTwits can throttle.

## Development

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## License

MIT. See `LICENSE`.
