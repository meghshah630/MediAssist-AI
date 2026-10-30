"""
core/rag_engine.py – Retrieval-Augmented Generation pipeline.

Pipeline flow:
  1. Greeting detection  → friendly response (no vector search)
  2. Embed query         → retrieve chunks from ChromaDB
  3. Relevance check     → if all distances > threshold, flag as "not in KB"
  4. Build prompt        → context + relevance signal + question
  5. Call Ollama LLM     → structured, grounded answer
  6. Return result dict  → answer, sources, chunks, online/missing flags
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

from config import DISCLAIMER, TOP_K_RESULTS

logger = logging.getLogger(__name__)

# Markers used to detect offline / model-missing fallback responses
_OFFLINE_MARKER = "⚠️ **Ollama LLM is currently offline"
_MODEL_MISSING_MARKER = "⚠️ **Model `"

# Cosine-distance threshold above which a result is considered irrelevant
_RELEVANCE_THRESHOLD = 0.85

# Greeting words (compared against lowercased, punctuation-stripped tokens)
_GREETING_WORDS: Set[str] = {
    "hi", "hii", "hiii", "hello", "helo", "helloo", "hey", "heyy", "heyyy",
    "howdy", "greetings", "sup", "hiya", "yo", "aloha", "namaste", "hola",
    "hie", "bonjour",
}
_GREETING_PHRASES = {
    "good morning", "good afternoon", "good evening", "good night", "good day",
    "what's up", "whats up", "how are you", "how do you do", "nice to meet you",
}


def _make_cache_key(question: str, patient_context: Optional[Dict], history_len: int) -> str:  # noqa: E501
    """Create a stable string key for caching."""
    import json
    payload = json.dumps(
        {"q": question, "pc": patient_context, "hl": history_len},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
      embed query → retrieve chunks → build prompt → call LLM → return answer
    """

    def __init__(self, embedder, vector_store, llm_client):
        """
        Parameters
        ----------
        embedder:      An ``Embedder`` instance (core.embedder).
        vector_store:  A ``VectorStore`` instance (core.vector_store).
        llm_client:    An ``OllamaClient`` instance (core.llm_client).
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("RAGEngine initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Greeting detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_greeting(question: str) -> bool:
        """Return True if the question is a casual greeting requiring no RAG."""
        cleaned = re.sub(r"[^\w\s]", "", question.lower().strip())
        words = cleaned.split()
        if not words:
            return False

        # Multi-word greeting phrases
        for phrase in _GREETING_PHRASES:
            if phrase in cleaned and len(words) <= 7:
                return True

        # Normalized words (collapse runs of repeating characters, e.g. "hiiii" -> "hi", "heyyyy" -> "hey")
        def _collapse(w: str) -> str:
            return re.sub(r"([a-z])\1+", r"\1", w)

        words_set = set(words)
        collapsed_set = {_collapse(w) for w in words}

        # Check exact tokens or collapsed tokens
        if (words_set & _GREETING_WORDS or collapsed_set & _GREETING_WORDS) and len(words) <= 5:
            return True

        return False

    def answer(
        self,
        question: str,
        patient_context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        top_k: int = TOP_K_RESULTS,
    ) -> Dict[str, Any]:
        """
        Answer a medical question using RAG.

        Parameters
        ----------
        question:
            The user's natural-language question.
        patient_context:
            Optional patient profile dict for personalised context.
        chat_history:
            Recent chat messages (list of {role, content} dicts).
        model:
            Ollama model name override.
        top_k:
            Number of document chunks to retrieve.

        Returns
        -------
        Dict with keys:
          - ``answer``:           The LLM-generated (or synthesised) response.
          - ``sources``:          List of unique source metadata dicts.
          - ``retrieved_chunks``: List of raw retrieved text chunks.
          - ``llm_online``:       bool – Ollama reachable AND model answered.
          - ``model_missing``:    bool – Ollama online but model not pulled.
          - ``is_greeting``:      bool – True when question was a greeting (no RAG used).
        """
        chat_history = chat_history or []

        # ── 0. Handle greetings before vector search ───────────────────────────
        if self._is_greeting(question):
            logger.debug("Greeting detected – skipping RAG pipeline.")
            greeting_answer = (
                "Hello! I am **MediAssist AI**, your medical information assistant.\n\n"
                "I can assist you with:\n"
                "- Symptoms and medical conditions\n"
                "- Evidence-based treatments and medications\n"
                "- First aid procedures and urgent care guidance\n"
                "- Preventive healthcare and lifestyle wellness\n\n"
                "What medical question can I assist you with today?\n\n"
                f"---\n\n{DISCLAIMER}"
            )
            return {
                "answer": greeting_answer,
                "sources": [],
                "retrieved_chunks": [],
                "llm_online": True,
                "model_missing": False,
                "is_greeting": True,
            }

        cache_key = _make_cache_key(question, patient_context, len(chat_history))

        # Only serve from cache if the previous result was a real LLM answer
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.get("llm_online", False):
                logger.debug("RAGEngine cache hit (LLM answer).")
                return cached
            # Don't cache offline/model-missing answers – state may have changed

        # ── 1. Embed query ─────────────────────────────────────────────────────
        try:
            query_embedding = self.embedder.embed_query(question)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return self._error_response("Embedding service error: " + str(exc))

        # ── 2. Retrieve ────────────────────────────────────────────────────────
        # Check if vector store is initialized and non-empty
        try:
            is_empty_fn = getattr(self.vector_store, "is_empty", None)
            if callable(is_empty_fn) and is_empty_fn() is True:
                logger.warning("Vector store is empty – knowledge base not initialized.")
                return {
                    "answer": (
                        "⚠️ **Knowledge base not initialized.**\n\n"
                        "The medical knowledge base currently contains no documents. "
                        "Please click **'Re-ingest Documents'** in the sidebar or under the "
                        "**Knowledge Base** tab to initialize the database.\n\n"
                        f"---\n\n{DISCLAIMER}"
                    ),
                    "sources": [],
                    "retrieved_chunks": [],
                    "llm_online": True,
                    "model_missing": False,
                    "is_greeting": False,
                }
        except Exception as exc:
            logger.warning("Could not check vector store empty state: %s", exc)

        try:
            results = self.vector_store.query(
                query_embedding=query_embedding, n_results=top_k
            )
        except Exception as exc:
            logger.error("Vector store query failed: %s", exc)
            return self._error_response(
                f"Knowledge base query error: {exc}. "
                "Please click 'Re-ingest Documents' in the Knowledge Base tab or restart the application."
            )

        chunks: List[str] = results.get("documents", [[]])[0] or []
        metadatas: List[Dict] = results.get("metadatas", [[]])[0] or []
        distances: List[float] = results.get("distances", [[]])[0] or []

        # ── 3. Relevance check ─────────────────────────────────────────────────
        # If every retrieved chunk is far from the query, we treat the question
        # as outside the knowledge base and instruct the LLM accordingly.
        context_irrelevant = bool(distances) and all(
            d > _RELEVANCE_THRESHOLD for d in distances
        )

        # ── 4. Build prompt ────────────────────────────────────────────────────
        # Include topic labels in context header so the LLM knows which
        # knowledge-base document each chunk comes from.
        top_chunks = chunks[:3]
        top_metas = metadatas[:3]
        context_parts = []
        for chunk, meta in zip(top_chunks, top_metas):
            topic = meta.get("topic", meta.get("source", "Medical Knowledge Base"))
            context_parts.append(f"[Source: {topic}]\n{chunk}")
        context_text = "\n\n---\n\n".join(context_parts)

        system_prompt = self._build_system_prompt(patient_context)
        full_prompt = self._build_prompt(
            question, context_text, chat_history, context_irrelevant=context_irrelevant
        )

        # ── 5. Generate via LLM ────────────────────────────────────────────────
        # Import directly so static fallback is always resolved from the current
        # module — not from a potentially stale cached instance.
        from core.llm_client import OllamaClient as _OllamaClient  # noqa: PLC0415

        try:
            resp_text = self.llm_client.generate(
                prompt=full_prompt,
                model=model,
                context=context_text,
                system=system_prompt,
                question=question,
            )
        except Exception as exc:
            logger.error("LLM generation raised unexpectedly: %s", exc)
            resp_text = _OllamaClient._offline_answer(question, context_text)

        # ── 6. Classify response ───────────────────────────────────────────────
        model_missing = resp_text.startswith(_MODEL_MISSING_MARKER)
        llm_online = (
            not resp_text.startswith(_OFFLINE_MARKER)
            and not model_missing
            and not resp_text.startswith("⚠️")
        )

        # ── 7. Normalize formatting & append disclaimer ───────────────────────
        # Convert Setext underlines ('====', '----') to clean Markdown headings
        # and remove raw divider rows so they are never spoken or rendered awkwardly.
        resp_text = self._normalize_response_formatting(resp_text)
        answer = resp_text + f"\n\n---\n\n{DISCLAIMER}"

        # ── 8. Build deduplicated sources list ─────────────────────────────────
        # Only surface sources when LLM produced a real answer.
        # Skip sources for irrelevant-context answers (KB doesn't have the info).
        sources: List[Dict] = []
        if llm_online and not context_irrelevant:
            seen_sources: set = set()
            for meta in metadatas:
                src = meta.get("source", "Unknown")
                if src not in seen_sources:
                    sources.append({
                        "source":   src,
                        "topic":    meta.get("topic", ""),
                        "category": meta.get("category", ""),
                        "metadata": meta,
                    })
                    seen_sources.add(src)

        result: Dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "llm_online": llm_online,
            "model_missing": model_missing,
            "is_greeting": False,
        }

        # Cache only successful LLM answers
        if llm_online:
            self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self, patient_context: Optional[Dict]) -> str:
        """Construct the system-level instruction for the LLM."""
        lines = [
            "You are MediAssist AI, a knowledgeable medical information assistant.",
            "Your role is to provide accurate, evidence-based medical information.",
            "",
            "IMPORTANT RULES:",
            "1. Answer ONLY based on the retrieved context provided below.",
            "2. If the context does not contain sufficient information, respond with:",
            "   \"The MediAssist knowledge base does not contain sufficient information",
            "   about this topic. Please consult a qualified healthcare provider.\"",
            "3. Never make up medical facts, dosages, or diagnoses not in the context.",
            "4. Always recommend consulting a qualified healthcare provider for personal",
            "   medical decisions.",
            "5. Be concise and direct – do not repeat the context verbatim.",
            "6. FORMAT your answer using Markdown:",
            "   - Start with a short **bold heading** summarising the topic.",
            "   - Use short paragraphs (2-4 sentences each).",
            "   - Use bullet lists ( - item ) for enumerations (symptoms, medications, steps).",
            "   - Bold (**term**) key medical terms on first use.",
            "7. If the retrieved context explicitly says there is no relevant information,",
            "   do NOT answer from general knowledge. State that the KB lacks the info.",
            "8. PATIENT PERSONALIZATION (when patient details are provided):",
            "   - Only reference profile fields that are directly relevant to the question.",
            "   - ALWAYS flag any retrieved medication or treatment that may conflict with",
            "     the patient's listed allergies or current medications.",
            "   - Do NOT mention profile fields that have no bearing on the answer.",
            "   - Never diagnose, prescribe, or recommend dosage changes based on the profile.",
            "   - Do not expose profile details in responses unrelated to the question.",
        ]

        if patient_context:
            name = patient_context.get("name", "Patient")
            age = patient_context.get("age", "")
            gender = patient_context.get("gender", "")
            conditions = patient_context.get("conditions", [])
            medications = patient_context.get("medications", [])
            allergies = patient_context.get("allergies", [])
            notes = patient_context.get("notes", "")

            profile_lines = [
                "",
                "PATIENT PROFILE (use only fields relevant to the current question):",
                f"  Name: {name}",
            ]
            if age:
                profile_lines.append(f"  Age: {age}")
            if gender:
                profile_lines.append(f"  Gender: {gender}")
            if conditions:
                profile_lines.append(f"  Known conditions: {', '.join(conditions)}")
            if medications:
                profile_lines.append(f"  Current medications: {', '.join(medications)}")
            if allergies:
                profile_lines.append(f"  Allergies / intolerances: {', '.join(allergies)}")
            if notes and notes.strip():
                profile_lines.append(f"  Additional notes: {notes.strip()}")
            profile_lines += [
                "",
                "  ALLERGY SAFETY: If the retrieved context mentions any substance the patient",
                "  is allergic to, add a clear warning at the top of your answer.",
                "  MEDICATION INTERACTIONS: If retrieved treatments may interact with current",
                "  medications, note this with a brief caution.",
            ]
            lines.extend(profile_lines)

        return "\n".join(lines)

    def _build_prompt(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        context_irrelevant: bool = False,
    ) -> str:
        """Construct the full user prompt including context and history."""
        parts: List[str] = []

        if history:
            parts.append("CONVERSATION HISTORY (for context only):")
            for msg in history[-6:]:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                # Strip disclaimer lines from history to save tokens
                content = content.split("---\n\n⚠️ DISCLAIMER")[0].strip()
                parts.append(f"  {role}: {content}")
            parts.append("")

        if context_irrelevant or not context:
            parts.append(
                "RETRIEVED MEDICAL CONTEXT: "
                "[No relevant information found in the knowledge base for this query]"
            )
            parts.append("")
            parts.append(
                "NOTE: The knowledge base does not contain relevant information for this "
                "question. You MUST respond that the MediAssist knowledge base does not "
                "contain sufficient information about this topic, and advise the user to "
                "consult a qualified healthcare provider. Do NOT answer from general knowledge."
            )
        else:
            parts.append("RETRIEVED MEDICAL CONTEXT:")
            parts.append(context)
            parts.append("")

        parts.append(f"USER QUESTION: {question}")
        parts.append("")
        if not context_irrelevant and context:
            parts.append(
                "Answer the USER QUESTION using ONLY the retrieved context above. "
                "Use clear Markdown formatting: bold heading, short paragraphs, bullet lists. "
                "If the context is insufficient to answer fully, say so explicitly."
            )
        else:
            parts.append(
                "Since no relevant context was found, state clearly that the MediAssist "
                "knowledge base does not contain sufficient information about this topic."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response_formatting(text: str) -> str:
        """
        Normalize response markdown formatting:
          - Converts Setext header underlines ('Header\\n=====' or 'Header\\n-----')
            to clean Markdown headings ('## Header' or '### Header').
          - Strips raw divider lines (e.g. '====', '----') so they are neither
            displayed as ugly ASCII dividers nor spoken as symbols.
          - Normalizes multiple blank lines.
        """
        if not text:
            return text
        # Convert Setext H1: Heading followed by === underline
        text = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*={2,}\s*$", r"## \1\n", text)
        # Convert Setext H2: Heading followed by --- underline
        text = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*-{2,}\s*$", r"### \1\n", text)
        # Strip standalone divider rows (e.g. '====', '----', '____', '****')
        text = re.sub(r"(?m)^\s*[-=_*~]{2,}\s*$", "", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _error_response(message: str) -> Dict[str, Any]:
        return {
            "answer": f"⚠️ Error: {message}\n\n{DISCLAIMER}",
            "sources": [],
            "retrieved_chunks": [],
            "llm_online": False,
            "model_missing": False,
        }
