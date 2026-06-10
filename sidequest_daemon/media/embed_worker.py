"""Sentence-embedding worker for the renderer daemon (story 15-7).

Extracted from ``daemon.py`` (story 101-7) so ``daemon.py`` is socket
lifecycle + routing only. ``daemon.py`` re-exports ``EmbedWorker`` for
back-compat — existing consumers import it from
``sidequest_daemon.media.daemon``.
"""

from __future__ import annotations


class EmbedWorker:
    """Generates sentence embeddings via sentence-transformers (story 15-7).

    Uses all-MiniLM-L6-v2 for 384-dimensional embeddings — fast and
    good enough for lore fragment similarity search.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Story 37-23: pin to CPU. MPS is reserved for Z-Image renders —
            # running embed on CPU gives it an independent device so the
            # embed path never contends with in-flight image generation
            # and can never re-trigger the 2026-04-10 concurrent-MPS-session
            # deadlock that story 37-5 originally fixed by sharing a lock.
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a sentence embedding for the given text.

        Raises ValueError if text is empty — no silent fallbacks.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return [float(v) for v in embedding]
