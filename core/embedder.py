"""
core/embedder.py – SentenceTransformer-based text embedding with caching.
"""

from __future__ import annotations

import logging
import functools
from typing import List

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once
_model_singleton = None


def _get_model():
    """Return the cached SentenceTransformer model (load on first call)."""
    global _model_singleton
    if _model_singleton is None:
        try:
            from sentence_transformers import SentenceTransformer
            from config import EMBEDDING_MODEL
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _model_singleton = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise
    return _model_singleton


class Embedder:
    """Wraps SentenceTransformer to produce dense vector embeddings."""

    def __init__(self, model_name: str | None = None):
        """
        Initialise the embedder.

        Parameters
        ----------
        model_name:
            Override the default model from config (optional).
        """
        self._model_name = model_name
        # Model is loaded lazily on first use
        self._model = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self):
        """Lazy-load the model."""
        if self._model is None:
            if self._model_name:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            else:
                self._model = _get_model()
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []
        try:
            model = self._load()
            vectors = model.encode(texts, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.error("embed_texts failed: %s", exc)
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Parameters
        ----------
        query:
            The search query to embed.

        Returns
        -------
        A single embedding vector as a list of floats.
        """
        results = self.embed_texts([query])
        if not results:
            raise ValueError("Embedding returned empty result for query.")
        return results[0]


# Convenience module-level functions ----------------------------------------

def get_embedder() -> Embedder:
    """Return a module-level Embedder singleton."""
    global _embedder_singleton
    if "_embedder_singleton" not in globals() or _embedder_singleton is None:  # type: ignore[name-defined]
        globals()["_embedder_singleton"] = Embedder()
    return globals()["_embedder_singleton"]

