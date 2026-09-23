"""Finance-tuned sentiment lexicon.

Each entry maps a lowercase token/phrase to an intensity in (-1, 1):
positive = bullish language, negative = bearish language.  Intensities are
deliberate but heuristic — this is a transparent, auditable starting point,
not a trained model.  See docs/SCORING.md for the method and limits.

Multi-word phrases are matched before single tokens.
"""

from __future__ import annotations

# -- bullish ---------------------------------------------------------------
BULLISH: dict[str, float] = {
    # euphoria / momentum
    "moon": 0.9, "to the moon": 0.9, "lambo": 0.8, "rocket": 0.8,
    "squeeze": 0.8, "short squeeze": 0.9, "gamma squeeze": 0.85,
    "breakout": 0.7, "break out": 0.7, "all time high": 0.85, "ath": 0.85,
    "parabolic": 0.75, "rip": 0.6, "ripping": 0.65, "soar": 0.7,
    "skyrocket": 0.8, "explode": 0.7, "explode higher": 0.75,
    # conviction
    "bullish": 0.8, "very bullish": 0.9, "extremely bullish": 0.95,
    "buy": 0.5, "buying": 0.55, "buy the dip": 0.6, "btd": 0.6,
    "long": 0.5, "going long": 0.6, "calls": 0.45, "call buying": 0.6,
    "accumulate": 0.55, "accumulating": 0.55, "loading": 0.5,
    "loading up": 0.6, "adding": 0.45, "doubling down": 0.6,
    # fundamentals
    "beat": 0.6, "beats": 0.6, "earnings beat": 0.75, "beat estimates": 0.7,
    "upgrade": 0.65, "upgraded": 0.65, "price target raise": 0.7,
    "outperform": 0.6, "strong buy": 0.8, "buy rating": 0.7,
    "record revenue": 0.75, "record earnings": 0.75, "guidance raise": 0.7,
    "raised guidance": 0.7, "growth": 0.4, "growing": 0.4,
    # value
    "undervalued": 0.6, "cheap": 0.4, "oversold": 0.5, "discount": 0.4,
    "steal": 0.5, "bargain": 0.5, "support holding": 0.5, "bounce": 0.5,
    "reversal": 0.45, "bull flag": 0.55, "cup and handle": 0.55,
    "golden cross": 0.6, "higher highs": 0.55, "uptrend": 0.5,
    # misc bullish
    "winner": 0.5, "winning": 0.5, "profit": 0.45, "profits": 0.45,
    "gains": 0.5, "tendies": 0.55, "diamond hands": 0.6,
    "hold": 0.35, "holding": 0.35, "hodl": 0.4, "yolo": 0.5,
    "bull run": 0.7, "bull market": 0.65, "momentum": 0.45,
}

# -- bearish ---------------------------------------------------------------
BEARISH: dict[str, float] = {
    # panic / momentum
    "dump": -0.8, "dumping": -0.8, "crash": -0.9, "crashing": -0.9,
    "collapse": -0.85, "plunge": -0.8, "plunging": -0.8, "tank": -0.8,
    "tanking": -0.8, "nosedive": -0.8, "freefall": -0.85, "free fall": -0.85,
    "bloodbath": -0.9, "capitulation": -0.8, "meltdown": -0.85,
    "death spiral": -0.9, "all time low": -0.85, "atl": -0.85,
    # conviction
    "bearish": -0.8, "very bearish": -0.9, "extremely bearish": -0.95,
    "sell": -0.5, "selling": -0.55, "selloff": -0.7, "sell off": -0.7,
    "short": -0.5, "shorting": -0.6, "puts": -0.45, "put buying": -0.6,
    "dump it": -0.75, "get out": -0.6, "exit": -0.45, "paper hands": -0.6,
    # fundamentals
    "miss": -0.6, "misses": -0.6, "earnings miss": -0.75, "missed estimates": -0.7,
    "downgrade": -0.65, "downgraded": -0.65, "price target cut": -0.7,
    "underperform": -0.6, "sell rating": -0.8, "strong sell": -0.85,
    "guidance cut": -0.7, "cut guidance": -0.7, "lowered guidance": -0.7,
    "bankruptcy": -0.95, "bankrupt": -0.95, "chapter 11": -0.9,
    "dilution": -0.7, "dilutive": -0.7, "offering": -0.55,
    "restatement": -0.7, "accounting issues": -0.75, "fraud": -0.9,
    "sec investigation": -0.8, "lawsuit": -0.6, "layoffs": -0.55,
    # value / technical
    "overvalued": -0.6, "overbought": -0.5, "expensive": -0.4,
    "bubble": -0.7, "topping": -0.5, "breakdown": -0.7, "break down": -0.7,
    "death cross": -0.6, "lower lows": -0.55, "downtrend": -0.5,
    "resistance": -0.35, "rejected": -0.5, "failed breakout": -0.6,
    # misc bearish
    "loser": -0.5, "loss": -0.45, "losses": -0.45, "losing": -0.5,
    "rug pull": -0.9, "rugpull": -0.9, "scam": -0.85, "ponzi": -0.9,
    "dead": -0.6, "dead money": -0.65, "bagholder": -0.55, "bag holding": -0.55,
    "bear market": -0.65, "recession": -0.6,
}

# -- emoji -----------------------------------------------------------------
EMOJI: dict[str, float] = {
    "🚀": 0.8, "🌙": 0.7, "💎": 0.5, "🙌": 0.5, "📈": 0.7, "💰": 0.5,
    "🔥": 0.5, "✅": 0.4, "💪": 0.4, "🐂": 0.8, "⬆️": 0.4, "📊": 0.1,
    "📉": -0.7, "🔻": -0.6, "💩": -0.7, "🩸": -0.6, "⚠️": -0.4, "🐻": -0.8,
    "⬇️": -0.4, "❌": -0.4, "💀": -0.7, "🪦": -0.7,
}

# Words that flip the polarity of the following sentiment tokens.
NEGATIONS = frozenset({
    "not", "no", "never", "n't", "dont", "don't", "doesnt", "doesn't",
    "isnt", "isn't", "arent", "aren't", "wont", "won't", "cant", "can't",
    "hardly", "barely", "without", "lack", "lacks", "lacking",
})

# Intensifiers multiply the next sentiment token's weight.
INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "extremely": 1.6, "super": 1.4, "really": 1.3,
    "so": 1.3, "quite": 1.2, "pretty": 1.15, "highly": 1.4,
    "incredibly": 1.6, "massively": 1.5, "hugely": 1.5,
}


def combined() -> dict[str, float]:
    """All phrase/token scores in one dict, phrases first for matching."""
    merged = {**BEARISH, **BULLISH}
    # phrases (contain a space) sort before single tokens
    return dict(sorted(merged.items(), key=lambda kv: (" " not in kv[0], kv[0])))
