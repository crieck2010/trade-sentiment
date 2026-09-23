# Sources — trade-sentiment

## v0.1.0 sources (all keyless, ToS-clean)

### Reddit (`reddit`)
Public search JSON across finance subs: `wallstreetbets`, `stocks`,
`investing`, `stockmarket`, `options`, `pennystocks`. Query: `$SYM OR SYM`,
sorted new. ~60 req/min; the adapter self-throttles to 1 req/sec.
Engagement = score + comment count. Needs a descriptive `User-Agent`
(public JSON blocks default-library UAs).

### StockTwits (`stocktwits`)
Public symbol streams: `/api/2/streams/symbol/{SYM}.json`. No key.
HTML in message bodies is stripped; `entities.sentiment` is currently
ignored (the lexicon re-scores raw text — single source of truth).

### News (`news`)
Google News RSS search for `{SYM} stock`. Headlines only. Tone is milder
than social — headlines rarely say "moon". Useful as a reality check
against social euphoria.

## Why no X/Twitter in v0.1.0

Two reasons: the official API is paid (Basic tier ≈ $100/mo), and
scraping X violates its terms of service. Shipping a scraper would be
both fragile and wrong. The `Source` ABC is the seam:

```python
from trade_sentiment.sources import Source, SOURCES

class XSource(Source):
    name = "x"
    rate_limit_note = "paid API tier; bearer token in X_BEARER_TOKEN"
    def fetch(self, symbol, limit=50):
        # GET https://api.x.com/2/tweets/search/recent?query=${symbol}
        ...

SOURCES["x"] = XSource   # then: trade-sentiment scan AAPL --sources x reddit
```

When a key exists, add the adapter, register it in `SOURCES`, add the
env-var to `docs`, and bump the minor version.

## Adding any source

1. Subclass `Source`, set `name` and `rate_limit_note`.
2. Implement `fetch(symbol, limit)` → `list[Mention]`; return `[]` on
   any failure (fail-soft is a contract).
3. Register in `SOURCES`.
4. Add a mocked-network test in `tests/test_sentiment.py`.

## Politeness & scaling

- Each adapter owns its rate-limit behavior (Reddit: 1 req/sec).
- `pipeline.fetch_all` runs symbol×source jobs in a thread pool
  (`max_workers=6` default); network is the bottleneck, scoring is not.
- For large universes: raise `limit_per_source` cautiously, or shard
  symbols across cron runs. A persistent mention cache (SQLite) is the
  planned v0.2 scaling step.
