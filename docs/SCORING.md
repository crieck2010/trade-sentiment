# Scoring — trade-sentiment

## Method

Deterministic finance lexicon, no ML, no network:

1. **Phrase pass** — multi-word phrases ("short squeeze", "earnings beat",
   "to the moon") match first; they carry the most specific meaning.
2. **Token pass** — single tokens with two modifiers:
   - *negation window*: `not`/`never`/`no`… flips the next 3 tokens'
     polarity × −0.9 ("not bullish" ≈ bearish, slightly muted);
   - *intensifiers*: `very` ×1.4, `extremely` ×1.6, `so` ×1.3…
3. **Emoji pass** — 🚀 +0.8, 📉 −0.7, etc. (`lexicon.EMOJI`).
4. **Emphasis** — >60% caps ×1.15, 2+ exclamation marks ×1.1.
5. Polarity = mean of weighted hits, clipped to [−1, 1].
   Magnitude = min(1, hits/6) — how much sentiment language was found.

Every result keeps its `hits`, so any score is auditable to the words
that produced it.

## The 0–10 scales

**Bullishness** (pure tone):
```
bullishness_10 = (polarity + 1) / 2 × 10        # −1→0, 0→5, +1→10
```

**Conviction** (tone × volume):
```
conviction_10 = min(10, bullishness × (0.5 + min(n_mentions, 100) / 100))
```

The volume factor runs 0.5 (a lone voice) → 1.5 (100+ mentions):

| Tone | Mentions | Bullishness | Conviction | Reads as |
|---|---|---|---|---|
| 0.8 | 5 | 9.0 | 5.0 | strong but quiet |
| 0.8 | 100 | 9.0 | 10.0 | strong and loud |
| 0.2 | 200 | 6.0 | 9.0 | moderate but crowded |
| −0.8 | 8 | 1.0 | 0.6 | bearish whisper |

Both numbers are always reported together — that is the point. A
single blended score would hide whether the move is conviction or crowd.

## Pop criteria

A window becomes a pop when it clears the mention floor (`min_mentions`,
default 5) **and** either:

- `volume_zscore ≥ 2.0` — the latest time-slice's chatter is ≥2σ above the
  earlier slices (burst), **or**
- `|mean_polarity| ≥ 0.25` — the tone itself is decisively non-neutral.

Ranking: volume z-score, then |tone|, then conviction — loudest first.

## LLM rescoring seam

`scoring.rescore_with_llm(scored, advisor, blend=0.5)` blends any
callable `advisor(text) → polarity` with the lexicon score. Advisor
failures fall back to the lexicon per-mention. Use it to catch sarcasm
and novel slang the lexicon misses; keep `blend ≤ 0.5` until the advisor
is benchmarked.

## Limitations

- Heuristic, not a model: sarcasm ("great, another dilution 🙄"),
  memes, and ticker-as-word collisions ("BEAR", "BULL" as tickers) can
  misfire. Ticker symbols are matched as `$SYM`/whole-word at fetch time,
  but text like "this is a bear market" scores bearish for any symbol.
- Social chatter skews bullish and mean-reverts fast; pops are research
  candidates, not entry signals.
- Intensities are hand-set. Recalibrate against labeled data before
  trusting absolute values; relative ranking is the safer use.
