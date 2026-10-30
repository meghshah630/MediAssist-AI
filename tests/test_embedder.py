"""
tests/test_embedder.py – Tests for core.embedder module.
"""

from __future__ import annotations

import pytest


class TestEmbedder:
    """Tests for the Embedder class."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Return an Embedder instance."""
        from core.embedder import Embedder
        return Embedder()

    def test_embed_texts_returns_list(self, embedder, sample_texts):
        """embed_texts should return a list of the same length as input."""
        result = embedder.embed_texts(sample_texts)
        assert isinstance(result, list)
        assert len(result) == len(sample_texts)

    def test_embed_texts_shape(self, embedder, sample_texts):
        """Each embedding should be a list of floats with consistent dimension."""
        result = embedder.embed_texts(sample_texts)
        dim = len(result[0])
        assert dim > 0, "Embedding dimension should be positive."
        for vec in result:
            assert len(vec) == dim, "All embeddings should have the same dimension."

    def test_embed_texts_returns_floats(self, embedder, sample_texts):
        """Each element of each embedding should be a float."""
        result = embedder.embed_texts(sample_texts)
        for vec in result:
            for val in vec[:5]:  # Check first 5 values
                assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_embed_texts_empty_returns_empty(self, embedder):
        """embed_texts with empty list should return empty list."""
        result = embedder.embed_texts([])
        assert result == []

    def test_embed_query_returns_list(self, embedder):
        """embed_query should return a list of floats."""
        result = embedder.embed_query("What is hypertension?")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_embed_query_returns_floats(self, embedder):
        """Each element of the query embedding should be a float."""
        result = embedder.embed_query("What medications treat diabetes?")
        for val in result[:5]:
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_embed_query_same_dim_as_texts(self, embedder, sample_texts):
        """Query embedding should have same dimension as text embeddings."""
        texts_result = embedder.embed_texts(sample_texts[:1])
        query_result = embedder.embed_query("What is blood pressure?")
        assert len(texts_result[0]) == len(query_result)

    def test_embed_single_text(self, embedder):
        """embed_texts should work with a single text."""
        result = embedder.embed_texts(["Single medical text about hypertension."])
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_different_texts_different_embeddings(self, embedder):
        """Different texts should produce different embeddings."""
        texts = [
            "Hypertension is high blood pressure.",
            "Asthma is an airway disease.",
        ]
        result = embedder.embed_texts(texts)
        # Embeddings should differ (cosine similarity < 1.0)
        vec1 = result[0]
        vec2 = result[1]
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a ** 2 for a in vec1) ** 0.5
        mag2 = sum(b ** 2 for b in vec2) ** 0.5
        cosine_sim = dot / (mag1 * mag2 + 1e-9)
        assert cosine_sim < 0.999, "Different texts should not have identical embeddings."

