"""
core/vector_store.py – ChromaDB persistent vector store wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """Wraps a ChromaDB collection with add / query / utility methods."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        client=None,
    ):
        """
        Initialise the vector store.

        Parameters
        ----------
        persist_directory:
            Path to the ChromaDB persistent storage directory.
        collection_name:
            Name of the collection to use.
        client:
            Optional pre-built ChromaDB client (used in tests for ephemeral
            in-memory clients).
        """
        try:
            import chromadb
            from config import CHROMA_DB_PATH, COLLECTION_NAME

            self._collection_name = collection_name or COLLECTION_NAME
            if client is not None:
                self._client = client
            else:
                path = persist_directory or CHROMA_DB_PATH
                self._client = chromadb.PersistentClient(path=path)

            self._collection = self._get_or_create_collection()
            logger.info(
                "VectorStore ready: collection='%s'", self._collection_name
            )
        except Exception as exc:
            logger.error("VectorStore initialisation failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self):
        """Get existing collection or create it."""
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error(
                "Failed to get/create collection '%s': %s",
                self._collection_name,
                exc,
            )
            raise

    def get_collection(self, force_refresh: bool = False):
        """
        Return the active ChromaDB collection reference.

        If force_refresh is True or the cached reference is stale/deleted,
        it automatically re-fetches or re-creates the collection.
        """
        if force_refresh or self._collection is None:
            self._collection = self._get_or_create_collection()
            return self._collection

        try:
            # Probe collection to ensure the internal collection ID is still valid
            _ = self._collection.count()
        except Exception as exc:
            logger.warning(
                "Cached collection reference for '%s' is stale or invalid (%s). Refreshing...",
                self._collection_name,
                exc,
            )
            self._collection = self._get_or_create_collection()

        return self._collection

    def refresh_collection(self) -> None:
        """Force re-binding to the current collection in ChromaDB."""
        self.get_collection(force_refresh=True)

    def is_empty(self) -> bool:
        """Return True if the collection is uninitialized or has 0 chunks."""
        return self.collection_count() == 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        docs: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents with pre-computed embeddings to the collection.

        Parameters
        ----------
        docs:       Raw text chunks.
        embeddings: Corresponding embedding vectors.
        metadatas:  Per-document metadata dicts.
        ids:        Unique string IDs for each document.
        """
        if not docs:
            logger.warning("add_documents called with empty docs list.")
            return

        def _do_add(collection):
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        try:
            col = self.get_collection()
            _do_add(col)
            logger.info("Added %d documents to collection '%s'.", len(docs), self._collection_name)
        except Exception as exc:
            if "does not exist" in str(exc).lower():
                logger.warning("add_documents encountered stale collection, retrying after refresh: %s", exc)
                col = self.get_collection(force_refresh=True)
                _do_add(col)
                logger.info("Added %d documents after collection refresh.", len(docs))
            else:
                logger.error("add_documents failed: %s", exc)
                raise

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection for nearest neighbours.

        Parameters
        ----------
        query_embedding:
            The query vector.
        n_results:
            Number of results to return.
        where:
            Optional metadata filter dict.

        Returns
        -------
        ChromaDB query results dict with keys:
        ``ids``, ``documents``, ``metadatas``, ``distances``.
        """
        def _do_query(collection):
            count = collection.count()
            if count == 0:
                logger.warning("Collection '%s' is empty – returning empty results.", self._collection_name)
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            k = min(n_results, count)
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = collection.query(**kwargs)
            logger.debug("Query returned %d results.", len(results.get("ids", [[]])[0]))
            return results

        try:
            col = self.get_collection()
            return _do_query(col)
        except Exception as exc:
            if "does not exist" in str(exc).lower():
                logger.warning("query encountered stale collection reference, refreshing and retrying: %s", exc)
                col = self.get_collection(force_refresh=True)
                return _do_query(col)
            logger.error("query failed: %s", exc)
            raise

    def collection_count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            col = self.get_collection()
            return col.count()
        except Exception as exc:
            if "does not exist" in str(exc).lower():
                logger.warning("collection_count encountered stale reference, refreshing: %s", exc)
                col = self.get_collection(force_refresh=True)
                return col.count()
            logger.error("collection_count failed: %s", exc)
            return 0

    def reset_collection(self) -> None:
        """Delete and re-create the collection (wipes all documents)."""
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._get_or_create_collection()
            logger.info("Collection '%s' reset successfully.", self._collection_name)
        except Exception as exc:
            logger.error("reset_collection failed: %s", exc)
            # Re-create/bind even if deletion failed (e.g. collection didn't exist)
            self._collection = self._get_or_create_collection()
            raise

