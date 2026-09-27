# Point-in-time sentiment archive (v0.3.0)

A local SQLite store (`trade_sentiment.archive.Archive`, stdlib
`sqlite3`, the suite pattern) recording every sentiment observation with
**both** timestamps, so sentiment×price research can be screened with no
lookahead bias.

## Schema

Table `observations`:

| Column | Meaning |
|---|---|
| `symbol` | upper-cased ticker |
| `topic` | optional free grouping key beyond symbol |
| `source` | `reddit` / `stocktwits` / `news` / … |
| `kind` | `mention` today; aggregate kinds later |
| `observed_at` | **the as-of time** the sentiment refers to (source publication timestamp). UTC ISO. |
| `recorded_at` | **when it entered the archive** (the fetch time). UTC ISO, always ≥ `observed_at`. |
| `polarity` | −1…1 tone, NULL if unscored |
| `magnitude` | 0…1, NULL if unscored |
| `label` | `very bearish` … `very bullish` |
| `text` | raw reference text (truncated to 2000 chars at record time) |
| `url` | raw reference URI |
| `meta` | JSON: scorer name, lexicon hits, mention id, author, engagement |

Indexes on `(symbol, recorded_at)` and `(symbol, observed_at)`.

## The no-lookahead contract

*What you learn:* the difference between "when a thing was said" and
"when you could have known it".

For every observation *i*:

- `observed_i` — the time the sentiment *refers to*.
- `recorded_i` — the time the archive *learned* it.

A screening decision made at decision time *D* may use **only**
observations with `recorded_i ≤ D`. Filtering on `observed_i ≤ D`
alone is not point-in-time safe: a headline published at 09:30 but
scraped at 16:00 can carry information — edits, engagement counts,
corrections — that only existed at 16:00.

`Archive.query(symbol, as_of)` enforces the `recorded_at <= as_of`
predicate in SQL and it **cannot be opted out of**: no flag, no
parameter, no back door. `recorded_at` is set once at insert and never
updated. `Archive.query_range(symbol, start, end)` is the replay
helper: it equals "everything knowable about `[start, end]` at decision
time `end`".

Two more guards:

1. **Write-time invariant.** `record()` raises `ValueError` if
   `observed_at > recorded_at`. A provider that stamps `observed_at`
   with fetch time — the classic silent miswiring — fails loudly at
   insert instead of poisoning the archive.
2. **Archiving never kills the scan.** In `scan(archive=...)`, a failed
   insert is swallowed; live pops still return.

*Why it matters:* backtests that mix publication time with scrape time
manufacture information from the future and then "discover" strategies
that were never tradable. The archive's job is to make that mistake
impossible by construction, not by convention.

## Coverage and retention

The archive accumulates observations **from deployment forward**.
There is no history before the first recorded observation, and none is
fabricated — see `docs/BACKFILL.md` for the (failed) historical-source
investigation and the honest gap statement.

- `archive.coverage(symbol=None)` returns min/max `observed_at` and row
  counts; it returns `None` on an empty archive (which is itself the
  gap statement).
- `archive.prune(days)` deletes observations with `observed_at` older
  than `days`. Pruning is by *observed* time, so after pruning,
  point-in-time replays with `as_of` earlier than the new coverage
  start are incomplete — `coverage()` will say so. Retention is a
  coverage statement, never a lookahead fix: pruning cannot repair a
  lookahead leak, and it cannot create one.

Practical note: the default database lives at
`~/.trade-sentiment/sentiment-archive.db`. Point it elsewhere with
`Archive(path)` or the CLI `--db` flag. Observations are cheap (a few
hundred bytes each); a 3×-daily scan of 20 symbols keeps roughly
20k–100k rows/month.

## API

```python
from trade_sentiment import Archive, scan

arc = Archive()  # ~/.trade-sentiment/sentiment-archive.db

# record-on-fetch: every scored mention archived, observed_at = the
# source publication timestamp, recorded_at = the fetch time
pops = scan(["AAPL", "NVDA"], archive=arc)

# point-in-time query: only observations recorded at or before as_of
obs = arc.query("AAPL", as_of="2026-09-25T16:00:00+00:00")

# replay a window as known at its end
obs = arc.query_range("AAPL", "2026-09-18T00:00:00+00:00",
                              "2026-09-25T00:00:00+00:00")

arc.coverage("AAPL")   # {'from_observed': ..., 'to_observed': ...,
                       #  'n_observations': N}
arc.prune(days=90)     # keep the last 90 days of observed sentiment
```

CLI:

```bash
trade-sentiment scan AAPL NVDA --archive ./sentiment.db
trade-sentiment archive query AAPL --as-of 2026-09-25T16:00:00+00:00 --db ./sentiment.db
trade-sentiment archive coverage AAPL --db ./sentiment.db
trade-sentiment archive prune --days 90 --db ./sentiment.db
```

## Limitations

- The archive is only as good as the sources: live feeds are
  rate-limited, delayed, and can throttle; gaps in the archive are
  real gaps, not interpolated.
- Social chatter skews bullish and noisy; archived pops are research
  inputs, not signals.
- No history before deployment (see `docs/BACKFILL.md`). Any strategy
  screening that needs pre-deployment sentiment history must say so
  explicitly and wait for a real historical source — never backfill
  by hand.
