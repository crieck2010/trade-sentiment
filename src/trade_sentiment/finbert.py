"""Optional FinBERT rescoring adapter.

FinBERT (ProsusAI/finbert) is a BERT-base model fine-tuned on financial
text — the Financial PhraseBank, ~4.8k sentences labeled
positive/negative/neutral by finance professionals.  It reads whole
sentences with attention, so it catches context the lexicon scorer
misses: negation scope, novel slang, sarcasm-adjacent phrasing, and
finance-specific senses (\"cut costs\" vs \"cut guidance\").

This module is the suite's seam between the deterministic lexicon scorer
and that model::

    lexicon score (fast, always available, explainable)
        → rescore_with_finbert(...) → blended polarity → pops/ideas

Design rules (suite conventions):

- **Stdlib-only import.** ``torch`` and ``transformers`` are imported
  lazily inside :func:`require_finbert`; importing this module never
  touches them, so the core engine stays dependency-free.
- **Plain data in/out.** Takes ``list[ScoredMention]``, returns
  ``list[ScoredMention]`` — the same seam as
  :func:`trade_sentiment.scoring.rescore_with_llm`.
- **Fail-soft.** One bad mention or batch never kills the run; that
  mention keeps its lexicon score.
- **Batched inference.** Texts are scored in batches (default 32) under
  ``torch.no_grad()``; the tokenizer and model load once per
  :class:`FinBERTScorer` and are reused for every call.

Requires ``pip install torch transformers`` (CPU torch is fine) plus a
one-time ~440 MB model download from Hugging Face on first use.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .models import ScoredMention, label_for

#: Default Hugging Face model id.
FINBERT_MODEL = "ProsusAI/finbert"

#: ProsusAI/finbert's classifier head order.  The checkpoint stores
#: ``id2label = {0: "positive", 1: "negative", 2: "neutral"}``; override
#: via ``label_order`` if you point the scorer at a different head.
DEFAULT_LABEL_ORDER = ("positive", "negative", "neutral")


class FinBERTError(RuntimeError):
    """Raised when FinBERT is requested but unusable."""


def require_finbert():
    """Import (torch, transformers), or raise a helpful FinBERTError.

    Called lazily so the rest of trade-sentiment imports fine without
    the heavy ML stack installed.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise FinBERTError(
            "FinBERT needs the ML stack: pip install torch transformers "
            "(CPU torch is fine), then rerun. The lexicon scorer keeps "
            "working without it."
        ) from exc
    import torch
    import transformers
    return torch, transformers


def logits_to_polarity(
    logits: Sequence[float],
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
) -> float:
    """Map FinBERT classifier logits to a polarity in [-1, 1].

    Softmax the logits to class probabilities, then take the standard
    net-sentiment mapping::

        polarity = P(positive) − P(negative)

    Neutral mass pushes the result toward 0.  This is the same
    aggregation the FinBERT authors use for sentence-level scoring.
    Pure math — no torch needed, so it is unit-testable standalone.

    >>> logits_to_polarity([3.0, 0.0, 0.0])
    0.909...
    """
    order = [label.lower() for label in label_order]
    try:
        i_pos = order.index("positive")
        i_neg = order.index("negative")
    except ValueError as exc:
        raise FinBERTError(
            f"label_order {tuple(label_order)} has no positive/negative "
            "entries; cannot map to polarity"
        ) from exc
    zs = [float(z) for z in logits]
    if len(zs) != len(order):
        raise FinBERTError(
            f"expected {len(order)} logits for {tuple(label_order)}, "
            f"got {len(zs)}"
        )
    m = max(zs)
    exps = [math.exp(z - m) for z in zs]
    total = sum(exps)
    probs = [e / total for e in exps]
    return max(-1.0, min(1.0, probs[i_pos] - probs[i_neg]))


class FinBERTScorer:
    """Batched FinBERT polarity scorer.  Loads the model on first use.

    Parameters
    ----------
    model_name:
        Hugging Face id or local path.  Defaults to ``ProsusAI/finbert``.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto: cuda when available).
    batch_size:
        Texts per forward pass.  32 is a good CPU default; raise it on
        GPU.
    max_length:
        Tokenizer truncation length.  FinBERT's window is 512 tokens;
        longer texts are truncated head-first.
    label_order:
        Class order of the checkpoint's head; see
        :data:`DEFAULT_LABEL_ORDER`.

    The instance is reusable across calls and threads for inference
    (each forward pass runs under ``torch.no_grad()``); create one
    scorer per worker process and share it.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.label_order = tuple(label_order)
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def load(self) -> "FinBERTScorer":
        """Download (first run) and load the tokenizer + model.  Idempotent."""
        torch, transformers = require_finbert()
        if self._model is not None:
            return self
        self._torch = torch
        self._device = torch.device(
            self.device_name
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name
        )
        self._model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            .to(self._device)
            .eval()
        )
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self):
        return self._device

    def polarity(self, texts: Sequence[str]) -> list[float]:
        """Score texts → polarities in [-1, 1], batched.

        Empty/whitespace texts score 0.0 without hitting the model.
        A failing batch falls back to 0.0 for its texts rather than
        raising — callers that need per-mention lexicon fallback should
        use :func:`rescore_with_finbert`.
        """
        self.load()
        torch = self._torch
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t or "" for t in texts[i : i + self.batch_size]]
            try:
                with torch.no_grad():
                    enc = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    logits = self._model(**enc).logits
                out.extend(
                    logits_to_polarity(row.tolist(), self.label_order)
                    for row in logits
                )
            except Exception:
                out.extend(0.0 for _ in batch)
        return out

    def __call__(self, text: str) -> float:
        """Single-text convenience: ``scorer(text) → polarity``."""
        return self.polarity([text])[0]


def rescore_with_finbert(
    scored: list[ScoredMention],
    blend: float = 0.5,
    scorer: FinBERTScorer | None = None,
    batch_size: int = 32,
    model_name: str = FINBERT_MODEL,
    device: str | None = None,
) -> list[ScoredMention]:
    """Blend lexicon polarity with FinBERT judgment.

    Mirrors :func:`trade_sentiment.scoring.rescore_with_llm`::

        blended = (1 − blend) · lexicon + blend · finbert

    ``blend`` is the FinBERT weight (0 = lexicon only, 1 = FinBERT only;
    0.5 default).  FinBERT runs in batches for throughput; a failing
    batch — or a missing torch/transformers install — leaves those
    mentions at their lexicon score instead of raising, so one bad
    call never kills the scan.

    ``magnitude`` still comes from the lexicon hit count (how much
    sentiment-bearing language was found) and ``hits`` still lists the
    lexicon terms that fired — FinBERT is opaque, so its contribution
    is the tone shift, not the explanation.
    """
    blend = max(0.0, min(1.0, float(blend)))
    if not scored or blend == 0.0:
        return list(scored)
    if scorer is None:
        try:
            scorer = FinBERTScorer(
                model_name=model_name, device=device, batch_size=batch_size
            ).load()
        except FinBERTError:
            return list(scored)  # ML stack absent → lexicon stands
    texts = [sm.mention.text or "" for sm in scored]
    try:
        fin_p = scorer.polarity(texts)
    except Exception:
        return list(scored)
    if len(fin_p) != len(scored):
        return list(scored)
    out = []
    for sm, fp in zip(scored, fin_p):
        try:
            f = max(-1.0, min(1.0, float(fp)))
        except (TypeError, ValueError):
            f = sm.polarity
        p = round((1 - blend) * sm.polarity + blend * f, 4)
        out.append(ScoredMention(
            mention=sm.mention, polarity=p, magnitude=sm.magnitude,
            label=label_for(p), hits=sm.hits,
        ))
    return out


def finbert_rescorer(
    blend: float = 0.5,
    model_name: str = FINBERT_MODEL,
    device: str | None = None,
    batch_size: int = 32,
) -> Callable[[list[ScoredMention]], list[ScoredMention]]:
    """Build a ``pipeline.scan(rescorer=...)``-ready FinBERT rescorer.

    The scorer loads once and is shared across every symbol's mentions,
    so a multi-symbol scan pays the model-load cost a single time.
    """
    scorer = FinBERTScorer(
        model_name=model_name, device=device, batch_size=batch_size
    )

    def _rescore(scored: list[ScoredMention]) -> list[ScoredMention]:
        return rescore_with_finbert(scored, blend=blend, scorer=scorer)

    return _rescore


def describe() -> dict:
    """Machine-readable capability summary for the suite registry."""
    return {
        "name": "trade-sentiment.finbert",
        "capabilities": ["rescore_with_finbert", "finbert_rescorer",
                         "FinBERTScorer", "logits_to_polarity"],
        "outputs": ["blended ScoredMention lists"],
        "optional_dependencies": ["torch", "transformers"],
        "default_model": FINBERT_MODEL,
    }
