"""Deterministic lexicon scorer: text → polarity in [-1, 1].

Method: phrase-first lexicon matching with a short negation window,
intensifier multipliers, emoji scores, and caps/exclamation boosts.
No ML, no network, no dependencies — thousands of mentions per second.

An optional LLM rescoring seam (``rescore_with_llm``) lets a later
version or a caller blend in a model judgment without changing the API.
"""

from __future__ import annotations

import re
from typing import Callable

from . import lexicon as lex
from .models import Mention, ScoredMention, label_for

_TOKEN_RE = re.compile(r"[a-zA-Z']+|[^\w\s]", re.UNICODE)

LEXICON = lex.combined()
_PHRASES = [p for p in LEXICON if " " in p]


def _find_phrases(text_lower: str) -> list[tuple[str, float]]:
    """Match multi-word phrases; return (phrase, score) hits."""
    hits = []
    for phrase in _PHRASES:
        if phrase in text_lower:
            hits.append((phrase, LEXICON[phrase]))
    return hits


def score_text(text: str) -> ScoredMention | None:
    """Score raw text.  Returns a ScoredMention with a synthetic Mention."""
    mention = Mention(id="adhoc", source="adhoc", symbol="",
                      text=text, timestamp=_now())
    return _score_mention(mention)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _score_mention(mention: Mention) -> ScoredMention:
    text = mention.text or ""
    lowered = text.lower()

    hits: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    # 1. phrases first (they carry the most specific meaning)
    phrase_hits = _find_phrases(lowered)
    covered_tokens: set[str] = set()
    for phrase, score in phrase_hits:
        hits.append(phrase)
        weighted_sum += score
        weight_total += 1.0
        covered_tokens.update(phrase.split())

    # 2. token pass with negation + intensifiers; tokens already covered
    #    by a phrase ("short" in "short squeeze") don't double-count
    tokens = _TOKEN_RE.findall(lowered)
    negate = 0  # tokens remaining in the negation window
    multiplier = 1.0
    for tok in tokens:
        if tok in lex.NEGATIONS:
            negate = 3
            continue
        if tok in lex.INTENSIFIERS:
            multiplier = lex.INTENSIFIERS[tok]
            continue
        if tok in LEXICON and " " not in tok and tok not in covered_tokens:
            s = LEXICON[tok]
            if negate > 0:
                s = -s * 0.9  # "not bullish" ≈ bearish, slightly muted
            s *= multiplier
            hits.append(tok)
            weighted_sum += s
            weight_total += 1.0
            multiplier = 1.0
        if tok not in lex.INTENSIFIERS:
            # intensifier persists only to the next sentiment token
            pass
        negate = max(0, negate - 1)

    # 3. emoji
    for char in text:
        if char in lex.EMOJI:
            s = lex.EMOJI[char]
            hits.append(char)
            weighted_sum += s
            weight_total += 1.0

    if weight_total == 0:
        polarity, magnitude = 0.0, 0.0
    else:
        polarity = weighted_sum / weight_total
        # magnitude: how much sentiment language, saturating
        magnitude = min(1.0, weight_total / 6.0)

    # 4. emphasis boosts: ALL-CAPS and exclamation runs amplify
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.6 and polarity != 0:
        polarity *= 1.15
    if text.count("!") >= 2 and polarity != 0:
        polarity *= 1.1
    polarity = max(-1.0, min(1.0, polarity))

    return ScoredMention(
        mention=mention,
        polarity=round(polarity, 4),
        magnitude=round(magnitude, 4),
        label=label_for(polarity),
        hits=tuple(dict.fromkeys(hits)),  # de-duplicated, order kept
    )


def score_mentions(mentions: list[Mention]) -> list[ScoredMention]:
    """Score a batch of mentions."""
    return [_score_mention(m) for m in mentions]


def rescore_with_llm(
    scored: list[ScoredMention],
    advisor: Callable[[str], float],
    blend: float = 0.5,
) -> list[ScoredMention]:
    """Blend lexicon polarity with an LLM judgment.

    ``advisor`` takes mention text and returns a polarity in [-1, 1].
    ``blend`` is the LLM weight (0 = lexicon only, 1 = LLM only).
    Failures in the advisor fall back to the lexicon score for that
    mention — one bad call must not kill the batch.
    """
    out = []
    for sm in scored:
        try:
            llm_p = max(-1.0, min(1.0, float(advisor(sm.mention.text))))
            p = (1 - blend) * sm.polarity + blend * llm_p
        except Exception:
            p = sm.polarity
        out.append(ScoredMention(
            mention=sm.mention, polarity=round(p, 4),
            magnitude=sm.magnitude, label=label_for(p), hits=sm.hits,
        ))
    return out
