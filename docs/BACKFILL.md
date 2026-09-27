# Historical backfill: GDELT investigation (v0.3.0)

**Verdict: no clean backfill source found. The archive covers
observations from deployment forward only. No history is fabricated —
this gap is stated, not filled.**

## What was investigated (2026-09-26)

Candidate: **GDELT 2.0** — free, keyless, deep history, with tone fields
(`AvgTone` in the events export; the 7-field `V2Tone` block in GKG).
Two access paths were probed over plain HTTP:

1. **GDELT DOC 2.0 query API** (`api.gdeltproject.org/api/v2/doc/doc`) —
   the only keyless endpoint with direct text/ticker search and tone
   fields. Three probes (`ArtList` and `ToneChart` modes, ≥6s spacing)
   all returned the same rate-limit refusal, directing programmatic
   users to the ngrams dataset:
   > "Please limit requests to one every 5 seconds … All high-traffic
   > users should switch to our ngrams dataset"
   The DOC query API is **not cleanly reachable** for ticker-addressable
   historical queries.

2. **GDELT 2.0 bulk files** (`data.gdeltproject.org/gdeltv2/
   masterfilelist.txt`) — **reachable** (128 MB file list, coverage back
   to 2015-02-18, plain HTTP, no key). But: bulk GKG zips are ~5–15 MB
   per 15-minute slice (≈GBs per day), and there is no ticker-addressable
   query path. Mapping to tickers would require downloading whole daily
   GKG dumps and filtering on noisy organization-name mentions — heavy
   infrastructure for sparse, low-quality coverage. That is not a
   "minimal historical provider", so it was not built.

The other live sources (Reddit public search, StockTwits streams, Google
News RSS) are recent-only by design.

## Consequence for research

Track-1 sentiment×price strategy screening **cannot use real sentiment
history before the archive's first observation**. Options:

- **Wait for the archive to accumulate** (recommended): the 3×-daily
  runner with `--archive` builds point-in-time history from deployment
  forward, correctly timestamped from day one.
- **Revisit GDELT later**: the bulk path becomes viable if/when a
  ticker-addressable, low-cost query route exists (e.g. the ngrams
  dataset maturing, or a bounded company-name index). Re-running this
  investigation then is legitimate work; inventing history is not.
- **A keyed source** (X/Twitter API, a paid news archive) would also
  unlock backfill, at the cost of a key and a budget.

## What would change this doc

If a keyless historical text/sentiment source becomes cleanly reachable
— direct ticker/text query over plain HTTP, tone fields, stable enough
to build on — add a minimal provider plus a `backfill()` path in a
later bump, record its provenance in each archived row's `meta`
(`{"backfill": "gdelt-..."}`), and note the coverage start per symbol.
Until then: **the archive starts at deployment, and says so.**
