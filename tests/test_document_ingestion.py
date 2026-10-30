"""
tests/test_document_ingestion.py – Tests for core.document_ingestion module.
"""

from __future__ import annotations

import pytest
from pathlib import Path


class TestLoadDocument:
    """Tests for the load_document function."""

    def test_load_existing_file(self, sample_text_file):
        """Should return the file's content as a string."""
        from core.document_ingestion import load_document
        content = load_document(str(sample_text_file))
        assert isinstance(content, str)
        assert len(content) > 0

    def test_load_document_contains_text(self, sample_text_file):
        """Loaded document should contain expected text."""
        from core.document_ingestion import load_document
        content = load_document(str(sample_text_file))
        assert "Hypertension" in content

    def test_load_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent file."""
        from core.document_ingestion import load_document
        with pytest.raises(FileNotFoundError):
            load_document(str(tmp_path / "nonexistent.txt"))

    def test_load_document_returns_string(self, sample_text_file):
        """Return type should be str."""
        from core.document_ingestion import load_document
        result = load_document(str(sample_text_file))
        assert isinstance(result, str)


class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_extra_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        from core.document_ingestion import clean_text
        result = clean_text("Hello   world")
        assert "  " not in result

    def test_removes_extra_blank_lines(self):
        """Multiple consecutive blank lines should be collapsed."""
        from core.document_ingestion import clean_text
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        from core.document_ingestion import clean_text
        result = clean_text("   Hello World   ")
        assert result == "Hello World"

    def test_normalises_windows_line_endings(self):
        """Windows-style CRLF should be replaced with LF."""
        from core.document_ingestion import clean_text
        result = clean_text("Line 1\r\nLine 2\r\nLine 3")
        assert "\r" not in result

    def test_empty_string_returns_empty(self):
        """Empty string input should return empty string."""
        from core.document_ingestion import clean_text
        assert clean_text("") == ""

    def test_clean_text_preserves_content(self):
        """Important medical text should be preserved after cleaning."""
        from core.document_ingestion import clean_text
        text = "  Hypertension  treatment  includes  ACE  inhibitors.  "
        result = clean_text(text)
        assert "Hypertension" in result
        assert "ACE" in result
        assert "inhibitors" in result


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        from core.document_ingestion import chunk_text
        text = "This is a short medical text."
        result = chunk_text(text, chunk_size=512, overlap=64)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_returns_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        from core.document_ingestion import chunk_text
        words = ["word"] * 1200
        text = " ".join(words)
        result = chunk_text(text, chunk_size=512, overlap=64)
        assert len(result) > 1

    def test_chunks_are_strings(self):
        """All chunks should be strings."""
        from core.document_ingestion import chunk_text
        words = ["medical"] * 600
        text = " ".join(words)
        result = chunk_text(text, chunk_size=200, overlap=20)
        for chunk in result:
            assert isinstance(chunk, str)

    def test_empty_text_returns_empty_list(self):
        """Empty input should return an empty list."""
        from core.document_ingestion import chunk_text
        assert chunk_text("") == []

    def test_overlap_creates_shared_words(self):
        """Consecutive chunks should share words equal to the overlap."""
        from core.document_ingestion import chunk_text
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert len(result) >= 2

        # Verify overlap: last 20 words of chunk 0 should appear at start of chunk 1
        chunk0_words = result[0].split()
        chunk1_words = result[1].split()
        overlap_words = chunk0_words[-20:]
        assert chunk1_words[:20] == overlap_words

    def test_chunk_size_respected(self):
        """Each chunk should not exceed chunk_size + overlap words significantly."""
        from core.document_ingestion import chunk_text
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        result = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in result:
            word_count = len(chunk.split())
            assert word_count <= 110, f"Chunk too large: {word_count} words"

    def test_default_chunk_size_is_200(self):
        """Default chunk size is 200 words for better retrieval precision."""
        from core.document_ingestion import chunk_text
        # 300 words should produce at least 2 chunks at 200-word default
        words = ["word"] * 300
        text = " ".join(words)
        result = chunk_text(text)  # uses default chunk_size=200
        assert len(result) > 1, "Expected multiple chunks at default 200-word chunk size"


class TestIngestDirectory:
    """Tests for the ingest_directory function."""

    def test_ingest_returns_list_of_dicts(self, sample_docs_dir):
        """Should return a list of dictionaries."""
        from core.document_ingestion import ingest_directory
        result = ingest_directory(str(sample_docs_dir))
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_ingest_dicts_have_required_keys(self, sample_docs_dir):
        """Each dict should have 'text', 'source', 'chunk_id', and new metadata keys."""
        from core.document_ingestion import ingest_directory
        result = ingest_directory(str(sample_docs_dir))
        assert len(result) > 0
        for item in result:
            assert "text" in item
            assert "source" in item
            assert "chunk_id" in item
            # New metadata fields added in v2
            assert "topic" in item
            assert "category" in item
            assert "content_hash" in item

    def test_ingest_counts_match_files(self, sample_docs_dir):
        """Should process all .txt files in the directory."""
        from core.document_ingestion import ingest_directory
        result = ingest_directory(str(sample_docs_dir))
        sources = {item["source"] for item in result}
        assert "hypertension.txt" in sources
        assert "diabetes.txt" in sources
        assert "asthma.txt" in sources

    def test_ingest_missing_directory_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent directory."""
        from core.document_ingestion import ingest_directory
        with pytest.raises(FileNotFoundError):
            ingest_directory(str(tmp_path / "nonexistent_dir"))

    def test_ingest_empty_directory_returns_empty(self, tmp_path):
        """Empty directory with no .txt files should return empty list."""
        from core.document_ingestion import ingest_directory
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        result = ingest_directory(str(empty_dir))
        assert result == []

    def test_ingest_chunk_ids_are_unique(self, sample_docs_dir):
        """All chunk_ids should be unique."""
        from core.document_ingestion import ingest_directory
        result = ingest_directory(str(sample_docs_dir))
        ids = [item["chunk_id"] for item in result]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique."

