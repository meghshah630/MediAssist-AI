"""
tests/test_vector_store.py – Tests for core.vector_store using ChromaDB ephemeral client.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def vector_store():
    """Return a VectorStore backed by a fresh ephemeral ChromaDB client per test."""
    import chromadb
    import uuid
    from core.vector_store import VectorStore
    client = chromadb.EphemeralClient()
    # Unique collection name per test to avoid cross-test contamination
    return VectorStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


@pytest.fixture(scope="module")
def simple_embedder():
    """Return an Embedder for generating real embeddings in tests."""
    from core.embedder import Embedder
    return Embedder()


class TestVectorStore:
    """Tests for the VectorStore wrapper."""

    def test_initial_count_is_zero(self, vector_store):
        """A new collection should have zero documents."""
        assert vector_store.collection_count() == 0

    def test_add_documents_increases_count(self, vector_store, simple_embedder, sample_texts):
        """Adding documents should increase the collection count."""
        embeddings = simple_embedder.embed_texts(sample_texts[:3])
        metadatas = [{"source": f"doc_{i}.txt"} for i in range(3)]
        ids = [f"id_{i}" for i in range(3)]

        vector_store.add_documents(
            docs=sample_texts[:3],
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        assert vector_store.collection_count() == 3

    def test_add_empty_documents_noop(self, vector_store):
        """Adding empty list should not raise and count stays zero."""
        vector_store.add_documents([], [], [], [])
        assert vector_store.collection_count() == 0

    def test_query_returns_expected_structure(self, vector_store, simple_embedder, sample_texts):
        """query() should return a dict with ids, documents, metadatas, distances."""
        embeddings = simple_embedder.embed_texts(sample_texts)
        metadatas = [{"source": f"doc_{i}.txt"} for i in range(len(sample_texts))]
        ids = [f"q_id_{i}" for i in range(len(sample_texts))]

        vector_store.add_documents(
            docs=sample_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        query_embedding = simple_embedder.embed_query("What is hypertension?")
        results = vector_store.query(query_embedding=query_embedding, n_results=3)

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert len(results["documents"][0]) == 3

    def test_query_on_empty_collection_returns_empty(self, vector_store, simple_embedder):
        """Querying an empty collection should return empty results without error."""
        query_embedding = simple_embedder.embed_query("blood pressure")
        results = vector_store.query(query_embedding=query_embedding, n_results=5)
        assert results["documents"] == [[]]

    def test_query_n_results_capped_at_count(self, vector_store, simple_embedder, sample_texts):
        """n_results > collection size should return only available docs."""
        texts = sample_texts[:2]
        embeddings = simple_embedder.embed_texts(texts)
        metadatas = [{"source": "doc.txt"}] * 2
        ids = ["cap_id_0", "cap_id_1"]

        vector_store.add_documents(
            docs=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        query_emb = simple_embedder.embed_query("medical information")
        results = vector_store.query(query_embedding=query_emb, n_results=10)
        assert len(results["documents"][0]) <= 2

    def test_reset_collection_empties_store(self, vector_store, simple_embedder, sample_texts):
        """reset_collection() should empty the collection."""
        embeddings = simple_embedder.embed_texts(sample_texts[:2])
        vector_store.add_documents(
            docs=sample_texts[:2],
            embeddings=embeddings,
            metadatas=[{"source": "f.txt"}] * 2,
            ids=["r_id_0", "r_id_1"],
        )
        assert vector_store.collection_count() == 2

        vector_store.reset_collection()
        assert vector_store.collection_count() == 0

    def test_query_returns_relevant_result(self, vector_store, simple_embedder, sample_texts):
        """The most semantically similar document should be returned first."""
        embeddings = simple_embedder.embed_texts(sample_texts)
        metadatas = [{"source": f"doc_{i}.txt"} for i in range(len(sample_texts))]
        ids = [f"rel_id_{i}" for i in range(len(sample_texts))]

        vector_store.add_documents(
            docs=sample_texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        # Query about hypertension should return the hypertension document first
        query_emb = simple_embedder.embed_query("high blood pressure hypertension treatment")
        results = vector_store.query(query_embedding=query_emb, n_results=1)
        top_doc = results["documents"][0][0]
        assert "hypertension" in top_doc.lower() or "blood pressure" in top_doc.lower()

    def test_is_empty_initially_true(self, vector_store):
        """is_empty() should return True on an empty vector store."""
        assert vector_store.is_empty() is True

    def test_is_empty_false_after_add(self, vector_store, simple_embedder, sample_texts):
        """is_empty() should return False once documents are added."""
        embeddings = simple_embedder.embed_texts(sample_texts[:1])
        vector_store.add_documents(
            docs=sample_texts[:1],
            embeddings=embeddings,
            metadatas=[{"source": "test.txt"}],
            ids=["single_id"],
        )
        assert vector_store.is_empty() is False

    def test_stale_collection_recovery_on_query(self, vector_store, simple_embedder, sample_texts):
        """
        When the collection is deleted and recreated behind the vector_store's back,
        query() should catch the stale reference, refresh, and query the new collection without crashing.
        """
        # 1. Add docs to initial collection
        embeddings = simple_embedder.embed_texts(sample_texts[:2])
        vector_store.add_documents(
            docs=sample_texts[:2],
            embeddings=embeddings,
            metadatas=[{"source": "f1.txt"}, {"source": "f2.txt"}],
            ids=["id_1", "id_2"],
        )
        assert vector_store.collection_count() == 2

        # 2. Simulate another process/reset recreating the collection in ChromaDB
        client = vector_store._client
        col_name = vector_store._collection_name
        client.delete_collection(col_name)
        new_col = client.create_collection(col_name)
        # Add a new doc to the recreated collection
        new_emb = simple_embedder.embed_texts(["New recreated doc content"])
        new_col.add(documents=["New recreated doc content"], embeddings=new_emb, ids=["new_id"])

        # At this moment, vector_store._collection holds the old (deleted) collection object
        # Querying should transparently recover
        query_emb = simple_embedder.embed_query("doc content")
        results = vector_store.query(query_embedding=query_emb, n_results=1)
        assert "documents" in results
        assert len(results["documents"][0]) == 1
        assert results["documents"][0][0] == "New recreated doc content"

    def test_stale_collection_recovery_on_count(self, vector_store):
        """collection_count() should recover when cached collection is deleted."""
        client = vector_store._client
        col_name = vector_store._collection_name
        client.delete_collection(col_name)
        client.create_collection(col_name)

        # vector_store._collection is stale; collection_count should auto-refresh to 0
        count = vector_store.collection_count()
        assert count == 0
