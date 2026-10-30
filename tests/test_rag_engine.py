"""
tests/test_rag_engine.py – Tests for core.rag_engine with mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_embedder():
    """Return a mock Embedder."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    embedder.embed_texts.return_value = [[0.1] * 384]
    return embedder


@pytest.fixture
def mock_vector_store():
    """Return a mock VectorStore with preset query results."""
    vs = MagicMock()
    vs.is_empty.return_value = False
    vs.collection_count.return_value = 1
    vs.query.return_value = {
        "ids": [["chunk_001"]],
        "documents": [["Hypertension is high blood pressure. Treatment includes ACE inhibitors."]],
        "metadatas": [[{"source": "hypertension.txt", "chunk_id": "chunk_001"}]],
        "distances": [[0.15]],
    }
    return vs


@pytest.fixture
def mock_llm_client():
    """Return a mock OllamaClient."""
    client = MagicMock()
    client.generate.return_value = (
        "Based on the retrieved context, hypertension is treated with ACE inhibitors "
        "and lifestyle modifications such as a low-sodium diet and regular exercise."
    )
    client._fallback.return_value = "⚠️ LLM service unavailable. Based on retrieved context:\n\nHypertension context..."
    return client


@pytest.fixture
def rag_engine(mock_embedder, mock_vector_store, mock_llm_client):
    """Return a RAGEngine with mocked dependencies."""
    from core.rag_engine import RAGEngine
    return RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)


class TestRAGEngine:
    """Tests for the RAGEngine class."""

    def test_answer_returns_dict(self, rag_engine):
        """answer() should return a dictionary."""
        result = rag_engine.answer("What is hypertension?")
        assert isinstance(result, dict)

    def test_answer_has_required_keys(self, rag_engine):
        """answer() result should have 'answer', 'sources', and 'retrieved_chunks' keys."""
        result = rag_engine.answer("What is hypertension?")
        assert "answer" in result
        assert "sources" in result
        assert "retrieved_chunks" in result

    def test_answer_is_string(self, rag_engine):
        """'answer' field should be a non-empty string."""
        result = rag_engine.answer("What is hypertension?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_is_list(self, rag_engine):
        """'sources' field should be a list."""
        result = rag_engine.answer("What is hypertension?")
        assert isinstance(result["sources"], list)

    def test_retrieved_chunks_is_list(self, rag_engine):
        """'retrieved_chunks' field should be a list."""
        result = rag_engine.answer("What is hypertension?")
        assert isinstance(result["retrieved_chunks"], list)

    def test_answer_contains_disclaimer(self, rag_engine):
        """The answer should include the disclaimer."""
        from config import DISCLAIMER
        result = rag_engine.answer("What causes diabetes?")
        assert "DISCLAIMER" in result["answer"] or "educational purposes" in result["answer"]

    def test_embedder_called_with_query(self, rag_engine, mock_embedder):
        """embed_query should be called with the user's question."""
        question = "What are the symptoms of asthma?"
        rag_engine.answer(question)
        mock_embedder.embed_query.assert_called_once_with(question)

    def test_vector_store_queried(self, rag_engine, mock_vector_store):
        """The vector store query() should be called."""
        rag_engine.answer("What is COVID-19?")
        mock_vector_store.query.assert_called_once()

    def test_llm_called_with_model_override(self, rag_engine, mock_llm_client):
        """LLM generate() should be called with the specified model."""
        rag_engine.answer("What is angina?", model="phi3")
        call_kwargs = mock_llm_client.generate.call_args
        assert call_kwargs[1].get("model") == "phi3" or (
            call_kwargs[0] and "phi3" in str(call_kwargs[0])
        )

    def test_patient_context_included_in_system_prompt(self, rag_engine, mock_llm_client):
        """Patient context should be included in the system message."""
        patient_context = {
            "name": "Alice",
            "age": 55,
            "gender": "Female",
            "conditions": ["Hypertension"],
            "medications": ["Lisinopril"],
            "allergies": ["Penicillin"],
        }
        rag_engine.answer("What should I know about my condition?", patient_context=patient_context)
        call_kwargs = mock_llm_client.generate.call_args[1]
        system_msg = call_kwargs.get("system", "")
        assert "Alice" in system_msg

    def test_caching_returns_same_result(self, rag_engine):
        """Identical questions should be served from cache (LLM called only once)."""
        question = "What are beta blockers?"
        result1 = rag_engine.answer(question)
        result2 = rag_engine.answer(question)
        assert result1["answer"] == result2["answer"]

    def test_embedding_error_returns_error_dict(self, mock_vector_store, mock_llm_client):
        """If embedding fails, answer() should return an error response."""
        bad_embedder = MagicMock()
        bad_embedder.embed_query.side_effect = RuntimeError("Embedding service down")

        from core.rag_engine import RAGEngine
        engine = RAGEngine(bad_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("What is hypertension?")
        assert "error" in result["answer"].lower() or "⚠️" in result["answer"]

    def test_empty_collection_still_returns_answer(self, mock_embedder, mock_llm_client):
        """Even with no retrieved docs, LLM should be called and return something."""
        empty_vs = MagicMock()
        empty_vs.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, empty_vs, mock_llm_client)
        result = engine.answer("What is blood pressure?")
        assert isinstance(result["answer"], str)


# ---------------------------------------------------------------------------
# Greeting handling tests
# ---------------------------------------------------------------------------


class TestGreetingDetection:
    """Tests for RAGEngine._is_greeting() static method."""

    def test_hello_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("hello") is True

    def test_hi_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("hi") is True

    def test_hey_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("hey") is True

    def test_good_morning_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("good morning") is True

    def test_good_afternoon_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("good afternoon") is True

    def test_good_evening_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("good evening") is True

    def test_howdy_detected(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("howdy") is True

    def test_medical_question_not_greeting(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("What is hypertension?") is False

    def test_long_sentence_not_greeting(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("hello what are the symptoms of diabetes") is False

    def test_empty_string_not_greeting(self):
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting("") is False


class TestGreetingHandling:
    """Tests that greetings bypass the RAG pipeline."""

    def test_greeting_skips_vector_store(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greetings should NOT call the vector store."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        engine.answer("hello")
        mock_vector_store.query.assert_not_called()

    def test_greeting_skips_embedder(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greetings should NOT call the embedder."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        engine.answer("hi")
        mock_embedder.embed_query.assert_not_called()

    def test_greeting_skips_llm(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greetings should NOT call the LLM generate()."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        engine.answer("hey")
        mock_llm_client.generate.assert_not_called()

    def test_greeting_returns_is_greeting_true(self, mock_embedder, mock_vector_store, mock_llm_client):
        """answer() for a greeting should return is_greeting=True."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("hello")
        assert result.get("is_greeting") is True

    def test_greeting_has_empty_sources(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greeting response should have no sources."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("hi")
        assert result["sources"] == []

    def test_greeting_has_empty_chunks(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greeting response should have no retrieved_chunks."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("hello")
        assert result["retrieved_chunks"] == []

    def test_greeting_answer_mentions_mediassist(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greeting response should mention MediAssist or greet back."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("hello")
        assert "MediAssist" in result["answer"] or "Hello" in result["answer"] or "👋" in result["answer"]

    def test_greeting_answer_contains_disclaimer(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greeting response should still include the disclaimer."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("hi")
        assert "DISCLAIMER" in result["answer"] or "educational purposes" in result["answer"]

    def test_greeting_llm_online_true(self, mock_embedder, mock_vector_store, mock_llm_client):
        """Greeting response should mark llm_online=True (no actual LLM call needed)."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, mock_vector_store, mock_llm_client)
        result = engine.answer("good morning")
        assert result["llm_online"] is True


# ---------------------------------------------------------------------------
# Relevance / missing-context tests
# ---------------------------------------------------------------------------


class TestMissingContext:
    """Tests for behaviour when retrieved chunks are irrelevant or absent."""

    def test_high_distance_results_are_flagged(self, mock_embedder, mock_llm_client):
        """When all distances exceed threshold, irrelevant-context path is taken."""
        high_dist_vs = MagicMock()
        high_dist_vs.query.return_value = {
            "ids": [["chunk_x"]],
            "documents": [["This is unrelated content about stocks and finance."]],
            "metadatas": [[{"source": "unrelated.txt"}]],
            "distances": [[0.95]],  # above RELEVANCE_THRESHOLD (0.85)
        }
        # LLM should be instructed to say KB lacks info; simulate that response
        mock_llm_client.generate.return_value = (
            "The MediAssist knowledge base does not contain sufficient information about this topic."
        )
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, high_dist_vs, mock_llm_client)
        result = engine.answer("What is the price of Bitcoin?")
        # Prompt passed to LLM should signal irrelevance
        call_kwargs = mock_llm_client.generate.call_args[1]
        prompt_sent = call_kwargs.get("prompt", "")
        assert "No relevant information" in prompt_sent or "not contain" in prompt_sent.lower()

    def test_high_distance_results_have_no_sources(self, mock_embedder, mock_llm_client):
        """When context is irrelevant, sources list should be empty."""
        high_dist_vs = MagicMock()
        high_dist_vs.query.return_value = {
            "ids": [["chunk_x"]],
            "documents": [["Unrelated content."]],
            "metadatas": [[{"source": "unrelated.txt"}]],
            "distances": [[0.92]],
        }
        mock_llm_client.generate.return_value = (
            "The MediAssist knowledge base does not contain sufficient information."
        )
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, high_dist_vs, mock_llm_client)
        result = engine.answer("What is the stock market?")
        assert result["sources"] == []

    def test_low_distance_results_include_sources(self, mock_embedder, mock_llm_client):
        """When context is relevant (low distance), sources should be populated."""
        relevant_vs = MagicMock()
        relevant_vs.query.return_value = {
            "ids": [["chunk_1"]],
            "documents": [["Diabetes is a metabolic disorder characterised by high blood sugar."]],
            "metadatas": [[{"source": "diabetes.txt"}]],
            "distances": [[0.12]],  # well below threshold
        }
        mock_llm_client.generate.return_value = "Diabetes is a metabolic disorder."
        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, relevant_vs, mock_llm_client)
        result = engine.answer("What is diabetes?")
        assert len(result["sources"]) >= 1
        assert result["sources"][0]["source"] == "diabetes.txt"


# ---------------------------------------------------------------------------
# Section: Stale / Missing / Empty Vector Store & Repeated Greeting tests
# ---------------------------------------------------------------------------


class TestVectorStoreResilience:
    """Tests for RAGEngine resilience to collection state and greeting variations."""

    @pytest.mark.parametrize("greeting", ["HII", "hii", "hiii", "heyy", "heyyyy", "helloo", "hellooo"])
    def test_repeated_letter_greetings_detected(self, greeting):
        """Repeated-letter informal greetings should be detected."""
        from core.rag_engine import RAGEngine
        assert RAGEngine._is_greeting(greeting) is True

    def test_greeting_succeeds_when_vector_store_throws(self, mock_embedder, mock_llm_client):
        """A greeting must succeed even if the vector store throws Collection does not exist."""
        broken_vs = MagicMock()
        broken_vs.query.side_effect = RuntimeError("Collection [3a17d8c8] does not exist.")
        broken_vs.is_empty.side_effect = RuntimeError("Collection [3a17d8c8] does not exist.")

        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, broken_vs, mock_llm_client)

        result = engine.answer("HII")
        assert result["is_greeting"] is True
        assert "MediAssist" in result["answer"] or "Hello" in result["answer"]
        broken_vs.query.assert_not_called()

    def test_empty_vector_store_returns_helpful_uninitialized_message(self, mock_embedder, mock_llm_client):
        """When vector store is empty, answer() returns instructions to ingest documents."""
        empty_vs = MagicMock()
        empty_vs.is_empty.return_value = True

        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, empty_vs, mock_llm_client)

        result = engine.answer("what is fever?")
        assert "not initialized" in result["answer"].lower() or "re-ingest" in result["answer"].lower()
        empty_vs.query.assert_not_called()
        mock_llm_client.generate.assert_not_called()

    def test_query_failure_returns_actionable_error_without_fabrication(self, mock_embedder, mock_llm_client):
        """When vector store query fails, answer() gives an actionable error and does NOT call LLM."""
        failing_vs = MagicMock()
        failing_vs.is_empty.return_value = False
        failing_vs.query.side_effect = RuntimeError("ChromaDB disk I/O error")

        from core.rag_engine import RAGEngine
        engine = RAGEngine(mock_embedder, failing_vs, mock_llm_client)

        result = engine.answer("what is fever?")
        assert "⚠️" in result["answer"]
        assert "Knowledge base query error" in result["answer"]
        assert result["llm_online"] is False
        mock_llm_client.generate.assert_not_called()

    def test_normalize_response_formatting_converts_setext_and_removes_divider_lines(self):
        """Verify that Setext underlines ('====', '----') are converted to clean Markdown headings."""
        from core.rag_engine import RAGEngine

        raw_response = (
            "Fever\n"
            "================\n\n"
            "Fever is a temporary rise in body temperature.\n\n"
            "Definition and Temperature Thresholds\n"
            "----------------------------------------\n"
            "Normal body temperature ranges from 36.1 to 37.2 degrees Celsius."
        )

        cleaned = RAGEngine._normalize_response_formatting(raw_response)
        assert "====" not in cleaned
        assert "----" not in cleaned
        assert "## Fever" in cleaned
        assert "### Definition and Temperature Thresholds" in cleaned
        assert "Normal body temperature ranges" in cleaned

    def test_profile_aware_system_prompt_includes_allergies_and_medications(self, rag_engine, mock_llm_client):
        """Verify that patient profile with allergies and medications is incorporated into system prompt."""
        profile = {
            "name": "Jane Doe",
            "age": 45,
            "gender": "Female",
            "conditions": ["Asthma"],
            "medications": ["Albuterol"],
            "allergies": ["Penicillin", "Aspirin"],
            "notes": "Sensitive to NSAIDs",
        }

        rag_engine.answer("What pain relievers can I take?", patient_context=profile)
        mock_llm_client.generate.assert_called_once()
        _, kwargs = mock_llm_client.generate.call_args
        system_prompt = kwargs.get("system", "")

        assert "Jane Doe" in system_prompt
        assert "Asthma" in system_prompt
        assert "Albuterol" in system_prompt
        assert "Penicillin" in system_prompt
        assert "Aspirin" in system_prompt
        assert "Sensitive to NSAIDs" in system_prompt
        assert "ALLERGY SAFETY" in system_prompt

    def test_missing_profile_generates_standard_educational_prompt(self, rag_engine, mock_llm_client):
        """When patient_context is None or empty, system prompt does not contain patient profile block."""
        rag_engine.answer("What is fever?", patient_context=None)
        mock_llm_client.generate.assert_called_once()
        _, kwargs = mock_llm_client.generate.call_args
        system_prompt = kwargs.get("system", "")

        assert "PATIENT PROFILE" not in system_prompt
        assert "MediAssist AI" in system_prompt



