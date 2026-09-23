"""Chatter sources.  Every adapter is keyless, stdlib-only, and polite.

v0.1.0 ships three sources:
- ``RedditSource`` — public Reddit search JSON (no key, ~60 req/min)
- ``StockTwitsSource`` — public symbol streams (no key)
- ``NewsSource`` — Google News RSS (no key)

The ``Source`` ABC is the seam for later adapters (X/Twitter once an API
key exists, broker social feeds, etc.).  See docs/SOURCES.md.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .models import Mention

_UA = {"User-Agent": "trade-sentiment/0.1.0 (research; contact: github.com/crieck2010)"}
_TIMEOUT = 15


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as res:
        return res.read()


class Source(ABC):
    """One place public chatter about tickers can be fetched from."""

    name: str = "base"
    rate_limit_note: str = ""

    @abstractmethod
    def fetch(self, symbol: str, limit: int = 50) -> list[Mention]:
        """Return up to ``limit`` recent mentions of ``symbol``."""
        raise NotImplementedError

    def _mention(self, **kwargs) -> Mention:
        kwargs.setdefault("timestamp", datetime.now(timezone.utc))
        return Mention(**kwargs)


class RedditSource(Source):
    """Reddit search across finance subs (public JSON, no key)."""

    name = "reddit"
    rate_limit_note = "public JSON, ~60 req/min; be polite (1 req/sec)"
    _last_call = 0.0

    SUBS = ("wallstreetbets", "stocks", "investing", "stockmarket",
            "options", "pennystocks")

    def fetch(self, symbol: str, limit: int = 50) -> list[Mention]:
        self._polite()
        q = urllib.parse.quote(f"${symbol} OR {symbol}")
        subs = "+".join(self.SUBS)
        url = (f"https://www.reddit.com/r/{subs}/search.json"
               f"?q={q}&sort=new&restrict_sr=on&limit={min(limit, 100)}")
        try:
            data = json.loads(_get(url))
        except Exception:
            return []
        out = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            text = f"{d.get('title', '')}\n{d.get('selftext', '')}".strip()
            if len(text) < 10:
                continue
            out.append(self._mention(
                id=f"reddit:{d.get('id', '')}", source=self.name,
                symbol=symbol.upper(), text=text[:2000],
                timestamp=datetime.fromtimestamp(
                    d.get("created_utc", 0), tz=timezone.utc),
                author=str(d.get("author", "")),
                engagement=int(d.get("score", 0) or 0)
                           + int(d.get("num_comments", 0) or 0),
                url=f"https://www.reddit.com{d.get('permalink', '')}",
            ))
        return out[:limit]

    def _polite(self):
        wait = 1.0 - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()


class StockTwitsSource(Source):
    """StockTwits public symbol streams (no key)."""

    name = "stocktwits"
    rate_limit_note = "public streams endpoint, no key required"

    def fetch(self, symbol: str, limit: int = 50) -> list[Mention]:
        url = (f"https://api.stocktwits.com/api/2/streams/symbol/"
               f"{urllib.parse.quote(symbol.upper())}.json?limit={min(limit, 30)}")
        try:
            data = json.loads(_get(url))
        except Exception:
            return []
        out = []
        for m in data.get("messages", []):
            body = html.unescape(re.sub(r"<[^>]+>", " ", m.get("body", ""))).strip()
            if len(body) < 4:
                continue
            sentiment = (m.get("entities") or {}).get("sentiment") or {}
            out.append(self._mention(
                id=f"stocktwits:{m.get('id', '')}", source=self.name,
                symbol=symbol.upper(), text=body[:2000],
                timestamp=_parse_st(m.get("created_at", "")),
                author=str((m.get("user") or {}).get("username", "")),
                engagement=int((m.get("likes") or {}).get("total", 0) or 0),
                url=f"https://stocktwits.com/{(m.get('user') or {}).get('username', '')}/post/{m.get('id', '')}",
            ))
        return out[:limit]


class NewsSource(Source):
    """Google News RSS for ticker headlines (no key)."""

    name = "news"
    rate_limit_note = "RSS, no key; headlines only (tone is milder than social)"

    def fetch(self, symbol: str, limit: int = 50) -> list[Mention]:
        q = urllib.parse.quote(f"{symbol} stock")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        try:
            root = ET.fromstring(_get(url))
        except Exception:
            return []
        out = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            title = html.unescape(re.sub(r" - [^-]+$", "", title))
            if len(title) < 10:
                continue
            out.append(self._mention(
                id=f"news:{hash(title) & 0xFFFFFFFF:X}", source=self.name,
                symbol=symbol.upper(), text=title[:500],
                timestamp=_parse_rss(item.findtext("pubDate") or ""),
                author=str(item.findtext("source") or ""),
                url=str(item.findtext("link") or ""),
            ))
            if len(out) >= limit:
                break
        return out


def _parse_st(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _parse_rss(value: str) -> datetime:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


SOURCES: dict[str, type[Source]] = {
    "reddit": RedditSource,
    "stocktwits": StockTwitsSource,
    "news": NewsSource,
}


def get_source(name: str) -> Source:
    """Instantiate a source by name; ``KeyError`` with a hint otherwise."""
    try:
        return SOURCES[name.lower()]()
    except KeyError:
        raise KeyError(
            f"unknown sentiment source {name!r}; "
            f"available: {', '.join(sorted(SOURCES))}"
        ) from None
