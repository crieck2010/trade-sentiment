# FinBERT rescoring — trade-sentiment

## What it is

`trade_sentiment.finbert` is an **optional** adapter that blends the
deterministic lexicon tone with [FinBERT](https://huggingface.co/ProsusAI/finbert)
(ProsusAI/finbert) — a BERT-base model fine-tuned on the Financial
PhraseBank (~4,800 sentences labeled positive/negative/neutral by
finance professionals).  It reads whole sentences with attention, so it
catches context the lexicon misses: negation scope, novel slang,
sarcasm-adjacent phrasing, and finance-specific word senses ("cut
costs" vs "cut guidance").

The core engine stays **stdlib-only and deterministic**.  FinBERT is a
seam on top, never a requirement: without `torch`/`transformers`
installed, every rescoring call degrades to pure lexicon instead of
raising.

## Setup

```bash
pip install torch transformers   # CPU torch is fine for batch scoring
```

First use downloads ~440 MB from Hugging Face (`ProsusAI/finbert`) and
caches it under `~/.cache/huggingface`.  No API key, no account.

## Usage

**Rescore lexicon output (the seam):**

```python
from trade_sentiment import finbert, scan, scoring

scored = scoring.score_mentions(mentions)          # lexicon, always works
blended = finbert.rescore_with_finbert(scored, blend=0.5)
```

**Whole-scan rescoring** — one model load shared across all symbols:

```python
from trade_sentiment import scan
from trade_sentiment.finbert import finbert_rescorer

pops = scan(["AAPL", "NVDA"], rescorer=finbert_rescorer(blend=0.5))
```

**Direct model scoring:**

```python
from trade_sentiment.finbert import FinBERTScorer

scorer = FinBERTScorer(device="cuda", batch_size=64).load()  # load once
polarities = scorer.polarity(["earnings beat", "guidance cut"])  # [-1, 1]
```

**CLI:**

```bash
trade-sentiment score --model finbert "earnings beat, guidance raised"
trade-sentiment scan AAPL NVDA --rescore finbert --blend 0.5
```

## The maths

**What you learn.**  For each mention, a context-aware tone in [−1, 1]
that understands negation scope and finance jargon, blended with the
auditable lexicon score.  The blend keeps the lexicon's `hits` and
`magnitude` — FinBERT moves the tone, the lexicon keeps the receipt.

**Why it matters.**  The lexicon is fast and explainable but
context-blind: "not exactly a blowout quarter" scores positive on
"blowout", and novel slang ("priced in", "inverse Cramer") is invisible
until hand-added.  FinBERT was trained on sentences finance
professionals labeled, so it generalizes to phrasing the lexicon has
never seen.  The cost is opacity and speed — hence the blend, not a
replacement.

**The maths.**

1. **Logits → probabilities.**  FinBERT's classification head emits one
   logit per class, `z = (z_pos, z_neg, z_neu)`.  Softmax:

   ```
   P(c) = exp(z_c) / Σ_j exp(z_j)
   ```

2. **Probabilities → polarity.**  Net sentiment, the aggregation the
   FinBERT authors use for sentence scoring:

   ```
   polarity = P(positive) − P(negative)     # ∈ [−1, 1]
   ```

   Neutral mass pushes the result toward 0.  (The checkpoint's head
   order is `(positive, negative, neutral)`; override with
   `label_order=` if you point the scorer at a different head.)

3. **Blend with the lexicon.**  Convex combination, so the result stays
   in [−1, 1]:

   ```
   blended = (1 − β) · lexicon + β · finbert      # β = 0.5 default
   ```

   `β = 0` is pure lexicon (the model is never loaded), `β = 1` is pure
   FinBERT.  Keep `β ≤ 0.5` until you have benchmarked the model on your
   own labeled chatter — the same guidance as the LLM seam.

## Scaling

| Concern | Answer |
|---|---|
| Model load | Once per `FinBERTScorer`; `finbert_rescorer()` builds one scorer shared across every symbol in a scan |
| Throughput | Batched forward passes (`batch_size`, default 32; raise on GPU) under `torch.no_grad()` |
| Device | `device="cuda"`/`"cpu"`/`None` (auto); CPU handles hundreds of mentions/sec in batches |
| Long texts | Tokenizer truncates to 512 tokens (FinBERT's window); score per-sentence for long posts if nuance matters |
| Failure mode | Fail-soft per batch/per mention — bad inputs keep lexicon scores; a missing ML stack keeps lexicon scores; nothing raises mid-scan |
| Threading | One scorer per worker process; safe to share for inference within a process |

For the 3×-daily runner shape, batch the day's mentions per symbol
through a single shared scorer — the model-load cost amortizes to
~zero and the lexicon still does the cheap first pass.

## Limitations (read these)

- **Opaque.**  FinBERT gives no `hits`; a blended score's explanation
  is the lexicon component only.  Treat the model as a tone advisor,
  not an auditor.
- **English financial text.**  Trained on English news/analyst prose;
  expect degradation on heavy slang, non-English chatter, and emoji.
- **Truncation.**  512-token window; long threads are head-truncated.
- **Not calibrated.**  Softmax probabilities are not confidences;
  `polarity` is a ranking signal.  Recalibrate `β` against your own
  labeled data before trusting absolute values.
- Social chatter skews bullish and mean-reverts fast; pops are research
  candidates, not entry signals — same as the lexicon pipeline.
