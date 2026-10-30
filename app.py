"""
app.py – MediAssist AI: Main Streamlit Application Entry Point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import (
    APP_NAME,
    APP_VERSION,
    DISCLAIMER,
    CHROMA_DB_PATH,
    SQLITE_DB_PATH,
    SAMPLE_DOCS_PATH,
    COLLECTION_NAME,
    DEFAULT_MODEL,
    LOG_LEVEL,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Cache version – bump this string whenever core modules change ─────────────
# This forces Streamlit to rebuild all @st.cache_resource objects, preventing
# stale instances (e.g. old OllamaClient without updated error handling or
# VectorStore holding obsolete deleted collection IDs) from persisting across hot-reloads.
_CACHE_VERSION = "v1.1.0"

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Resource loading (cached) ─────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(_cache_ver: str = _CACHE_VERSION):
    """Load and cache the SentenceTransformer embedder."""
    from core.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Connecting to vector store…")
def get_vector_store(_cache_ver: str = _CACHE_VERSION):
    """Initialise and cache the ChromaDB vector store with verified collection reference."""
    from core.vector_store import VectorStore
    vs = VectorStore(persist_directory=CHROMA_DB_PATH, collection_name=COLLECTION_NAME)
    vs.refresh_collection()
    return vs


@st.cache_resource(show_spinner=False)
def get_llm_client(_cache_ver: str = _CACHE_VERSION):
    """Initialise and cache the Ollama LLM client."""
    from core.llm_client import OllamaClient
    return OllamaClient()


def get_rag_engine(embedder, vector_store, llm_client):
    """Return a RAGEngine; re-create if any dependency instance changed."""
    from core.rag_engine import RAGEngine
    existing = st.session_state.get("rag_engine")
    if (
        existing is None
        or existing.llm_client is not llm_client
        or existing.vector_store is not vector_store
        or existing.embedder is not embedder
    ):
        st.session_state["rag_engine"] = RAGEngine(embedder, vector_store, llm_client)
    return st.session_state["rag_engine"]


# ── DB init ───────────────────────────────────────────────────────────────────


def ensure_db():
    """Create database tables if they don't exist."""
    try:
        from db.database import init_db
        init_db(SQLITE_DB_PATH)
    except Exception as exc:
        logger.error("DB init failed: %s", exc)
        st.error(f"Database initialisation failed: {exc}")


# ── Ingest handler ────────────────────────────────────────────────────────────


def ingest_documents(embedder, vector_store):
    """Ingest all sample documents into the vector store."""
    from core.document_ingestion import ingest_directory
    from db.chat_repo import record_ingested_document

    try:
        with st.spinner("📄 Ingesting documents into knowledge base…"):
            chunks = ingest_directory(SAMPLE_DOCS_PATH)
            if not chunks:
                st.warning("No documents found to ingest.")
                return

            # Embed all chunks
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(texts)
            # Rich metadata: source, topic, category, chunk_index, content_hash
            metadatas = [
                {
                    "source":       c["source"],
                    "chunk_id":     c["chunk_id"],
                    "chunk_index":  c.get("chunk_index", 0),
                    "topic":        c.get("topic", "General Medical Information"),
                    "category":     c.get("category", "general"),
                    "content_hash": c.get("content_hash", ""),
                }
                for c in chunks
            ]
            ids = [c["chunk_id"] for c in chunks]

            # Reset and re-populate
            vector_store.reset_collection()
            vector_store.add_documents(
                docs=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            vector_store.refresh_collection()

            # Refresh active session RAG engine reference and invalidate query cache
            from core.rag_engine import RAGEngine
            st.session_state["rag_engine"] = RAGEngine(embedder, vector_store, get_llm_client(_cache_ver=_CACHE_VERSION))
            st.session_state["rag_engine"]._cache.clear()

            # Record per-file stats in DB
            from collections import Counter
            source_counts = Counter(c["source"] for c in chunks)
            for filename, count in source_counts.items():
                try:
                    record_ingested_document(filename, count, SQLITE_DB_PATH)
                except Exception:
                    pass

        total = vector_store.collection_count()
        st.success(f"✅ Ingested {len(chunks)} chunks from {len(source_counts)} file(s). Total in KB: {total}")
        logger.info("Ingestion complete: %d chunks across %d files.", len(chunks), len(source_counts))
    except Exception as exc:
        st.error(f"Ingestion failed: {exc}")
        logger.error("Ingest error: %s", exc)


# ── Session management ────────────────────────────────────────────────────────


def ensure_session(patient_id: Optional[int] = None) -> int:
    """Ensure an active chat session exists; return its ID."""
    from db.chat_repo import create_session, list_sessions, get_messages

    # If active_session_id is already set, return it
    if st.session_state.get("active_session_id") is not None:
        return st.session_state["active_session_id"]

    # Only on the initial browser load / restart, restore the latest session
    if not st.session_state.get("app_initialized", False):
        st.session_state["app_initialized"] = True
        existing = list_sessions(patient_id=patient_id, db_path=SQLITE_DB_PATH)
        if existing:
            latest = existing[0]
            st.session_state["active_session_id"] = latest.id
            msgs = get_messages(latest.id, db_path=SQLITE_DB_PATH)
            st.session_state["chat_messages"] = [
                {"role": m.role, "content": m.content, "sources": m.sources or []}
                for m in msgs
            ]
            st.session_state["session_title_set"] = bool(latest.title and latest.title != "New Chat")
            return latest.id

    # Otherwise create a new session
    session_id = create_session(
        session_name=f"Session {st.session_state.get('session_counter', 1)}",
        title="New Chat",
        patient_id=patient_id,
        db_path=SQLITE_DB_PATH,
    )
    st.session_state["active_session_id"] = session_id
    st.session_state["chat_messages"] = []
    st.session_state["session_counter"] = st.session_state.get("session_counter", 1) + 1
    st.session_state["session_title_set"] = False
    return session_id

    
def get_chat_messages() -> List[Dict[str, Any]]:
    """Return the in-memory chat messages for the current session."""
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    return st.session_state["chat_messages"]


def add_message(role: str, content: str, sources: Optional[List] = None):
    """Append a message to in-memory state and persist to DB."""
    msg = {"role": role, "content": content, "sources": sources or []}
    get_chat_messages().append(msg)
    try:
        from db.chat_repo import save_message, ChatMessage
        session_id = st.session_state.get("active_session_id")
        if session_id:
            save_message(
                ChatMessage(
                    session_id=session_id,
                    role=role,
                    content=content,
                    sources=sources or [],
                ),
                db_path=SQLITE_DB_PATH,
            )
    except Exception as exc:
        logger.warning("Failed to persist message: %s", exc)


# ── Tab renderers ─────────────────────────────────────────────────────────────


def render_chat_tab(rag_engine, patient_context: Dict, model: str):
    """Render Tab 1 – Chat."""
    from ui.chat_window import render_chat_window, render_thinking_indicator, render_message
    from ui.voice_controls import render_voice_input_controls

    ensure_session()

    # ── Active patient profile banner ─────────────────────────────────────
    active_profile = st.session_state.get("active_patient_context")
    if active_profile and active_profile.get("name"):
        col_prof_info, col_prof_clear = st.columns([5, 1])
        with col_prof_info:
            conds = ", ".join(active_profile.get("conditions", [])) or "None"
            meds = ", ".join(active_profile.get("medications", [])) or "None"
            allergies = ", ".join(active_profile.get("allergies", [])) or "None"
            st.caption(
                f"**Active Patient Profile:** {active_profile['name']} "
                f"(Age: {active_profile.get('age') or 'N/A'}, Gender: {active_profile.get('gender') or 'N/A'}) | "
                f"Conditions: {conds} | Meds: {meds} | Allergies: {allergies}"
            )
        with col_prof_clear:
            if st.button("Clear Profile", key="btn_clear_active_profile", use_container_width=True):
                st.session_state.pop("active_patient_context", None)
                st.rerun()

    # ── Knowledge Base empty warning ──────────────────────────────────────
    if hasattr(rag_engine.vector_store, "is_empty") and rag_engine.vector_store.is_empty():
        st.warning(
            "**Knowledge base is empty or uninitialized.**\n\n"
            "MediAssist AI has no medical documents in its vector database. "
            "Please click **'Re-ingest Documents'** under the **Knowledge Base** tab.",
        )

    # ── LLM status notice ─────────────────────────────────────────────────
    from core.llm_client import OllamaClient
    _llm = rag_engine.llm_client
    if not _llm.health_check():
        st.warning(
            "**Ollama LLM is offline or unreachable at `http://localhost:11434`.**\n\n"
            "MediAssist AI requires Ollama to generate medical answers.\n\n"
            "**To get started:**\n"
            "1. Open a terminal and run: `ollama serve`\n"
            "2. Pull the model: `ollama pull llama3.2`\n"
            "3. Refresh this page.",
        )
    elif model and not _llm.is_model_available(model):
        st.info(
            f"**Selected model `{model}` is not downloaded in Ollama.**\n\n"
            f"Please run the following command in PowerShell:\n\n"
            f'```powershell\n& "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" pull {model}\n```\n\n'
            f"Once downloaded, ask your medical question below.",
        )

    # Display existing messages with source citations and TTS speaker buttons
    messages = get_chat_messages()
    render_chat_window(messages)

    # ── Input row: text chat input + voice microphone controls ───────────
    col_input, col_mic = st.columns([8, 2])
    with col_input:
        user_input = st.chat_input("Ask a medical question…", key="chat_input")
    with col_mic:
        voice_text = render_voice_input_controls(key="chat_voice_in")

    query = user_input or voice_text
    if not query:
        return

    # Update session title from first user query
    if not st.session_state.get("session_title_set", False):
        from db.chat_repo import update_session_title
        curr_session_id = st.session_state.get("active_session_id")
        if curr_session_id:
            first_title = " ".join(query.strip().split())
            if len(first_title) > 42:
                first_title = first_title[:39] + "..."
            try:
                update_session_title(curr_session_id, first_title, db_path=SQLITE_DB_PATH)
                st.session_state["session_title_set"] = True
            except Exception as e:
                logger.warning("Could not update session title: %s", e)

    # User message
    add_message("user", query)
    render_message("user", query)

    # Generate answer
    with render_thinking_indicator():
        history = [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]
        # Use active_patient_context if available
        effective_patient_context = active_profile or patient_context or None
        result = rag_engine.answer(
            question=query,
            patient_context=effective_patient_context,
            chat_history=history,
            model=model,
        )

    answer = result.get("answer", "No answer generated.")
    sources = result.get("sources", [])
    is_greeting = result.get("is_greeting", False)

    # For greetings, don't attach sources (there are none)
    add_message("assistant", answer, [] if is_greeting else sources)
    render_message("assistant", answer, [] if is_greeting else sources, msg_idx=len(messages) - 1)
    # Refresh sidebar conversation list to show the new title
    st.rerun()


def render_profile_tab():
    """Render Tab 2 – Patient Profile."""
    from db.patient_repo import list_patients, create_patient, PatientProfile, update_patient, delete_patient

    st.header("Patient Profile Management")
    st.caption("Manage patient profiles stored locally. Set an active profile to personalize RAG responses.")

    # Active profile indicator
    active_profile = st.session_state.get("active_patient_context")
    if active_profile and active_profile.get("name"):
        col_act1, col_act2 = st.columns([5, 1])
        with col_act1:
            st.info(f"**Current Active Profile:** {active_profile['name']}")
        with col_act2:
            if st.button("Clear Active", key="tab_clear_active_profile", use_container_width=True):
                st.session_state.pop("active_patient_context", None)
                st.rerun()

    with st.expander("Add New Patient Profile", expanded=False):
        with st.form("add_patient_form"):
            name = st.text_input("Name *")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=150, value=0)
            with col2:
                gender = st.selectbox("Gender", ["", "Male", "Female", "Non-binary", "Prefer not to say"])
            conditions = st.text_input("Conditions (comma-separated)")
            medications = st.text_input("Medications (comma-separated)")
            allergies = st.text_input("Allergies (comma-separated)")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save Patient")

            if submitted and name:
                try:
                    p = PatientProfile(
                        name=name,
                        age=age if age > 0 else None,
                        gender=gender or None,
                        conditions=[c.strip() for c in conditions.split(",") if c.strip()],
                        medications=[m.strip() for m in medications.split(",") if m.strip()],
                        allergies=[a.strip() for a in allergies.split(",") if a.strip()],
                        notes=notes,
                    )
                    create_patient(p, db_path=SQLITE_DB_PATH)
                    st.success(f"Patient '{name}' saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

    st.subheader("Existing Patients")
    try:
        patients = list_patients(db_path=SQLITE_DB_PATH)
        if not patients:
            st.info("No patients recorded yet.")
        else:
            for pt in patients:
                is_active = (active_profile and active_profile.get("name") == pt.name)
                expander_label = f"{pt.name} (ID: {pt.id}){' — Active' if is_active else ''}"
                with st.expander(expander_label, expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Age", pt.age or "N/A")
                    cols[1].metric("Gender", pt.gender or "N/A")
                    cols[2].metric("Conditions", len(pt.conditions))
                    if pt.medications:
                        st.write(f"**Medications:** {', '.join(pt.medications)}")
                    if pt.allergies:
                        st.write(f"**Allergies:** {', '.join(pt.allergies)}")
                    if pt.notes:
                        st.write(f"**Notes:** {pt.notes}")

                    col_act, col_del = st.columns([3, 1])
                    with col_act:
                        if not is_active:
                            if st.button("Set as Active Profile", key=f"set_active_{pt.id}", use_container_width=True):
                                st.session_state["active_patient_context"] = pt.to_dict()
                                st.success(f"Active profile set to {pt.name}.")
                                st.rerun()
                        else:
                            st.caption("Currently active in Chat")

                    with col_del:
                        if st.button("Delete", key=f"del_patient_{pt.id}", use_container_width=True):
                            delete_patient(pt.id, db_path=SQLITE_DB_PATH)
                            if is_active:
                                st.session_state.pop("active_patient_context", None)
                            st.rerun()
    except Exception as exc:
        st.error(f"Failed to load patients: {exc}")


def render_knowledge_base_tab(embedder, vector_store):
    """Render Tab 3 – Knowledge Base."""
    from db.chat_repo import list_ingested_documents

    st.header("Knowledge Base")
    count = vector_store.collection_count()
    st.metric("Total Chunks in Vector Store", count)

    col_ingest, col_reset = st.columns(2)
    with col_ingest:
        if st.button("Re-ingest Documents", use_container_width=True):
            ingest_documents(embedder, vector_store)
    with col_reset:
        if st.button("Reset Knowledge Base", use_container_width=True):
            try:
                vector_store.reset_collection()
                vector_store.refresh_collection()
                from core.rag_engine import RAGEngine
                st.session_state["rag_engine"] = RAGEngine(embedder, vector_store, get_llm_client(_cache_ver=_CACHE_VERSION))
                st.warning("Knowledge base cleared.")
            except Exception as exc:
                st.error(f"Reset failed: {exc}")

    st.subheader("Ingested Documents")
    try:
        docs = list_ingested_documents(db_path=SQLITE_DB_PATH)
        if not docs:
            st.info("No documents ingested yet. Click 'Re-ingest Documents' to begin.")
        else:
            for doc in docs:
                st.markdown(
                    f"**{doc['filename']}** — {doc['chunk_count']} chunks — _{doc['ingested_at']}_"
                )
    except Exception as exc:
        st.error(f"Failed to load document records: {exc}")

    # Show available sample docs
    st.subheader("Available Sample Documents")
    sample_dir = Path(SAMPLE_DOCS_PATH)
    if sample_dir.exists():
        txt_files = list(sample_dir.glob("*.txt"))
        if txt_files:
            for f in sorted(txt_files):
                size = f.stat().st_size // 1024
                st.markdown(f"- `{f.name}` ({size} KB)")
        else:
            st.info("No .txt files found in sample_docs/")
    else:
        st.warning(f"Sample docs directory not found: {SAMPLE_DOCS_PATH}")


def render_history_tab():
    """Render Tab 4 – History."""
    from ui.history_panel import render_history_panel
    render_history_panel(db_path=SQLITE_DB_PATH)


# ── Main app ──────────────────────────────────────────────────────────────────


def main():
    """MediAssist AI Streamlit main function."""
    # DB init
    ensure_db()

    # Load core components
    try:
        embedder = get_embedder(_cache_ver=_CACHE_VERSION)
        vector_store = get_vector_store(_cache_ver=_CACHE_VERSION)
        llm_client = get_llm_client(_cache_ver=_CACHE_VERSION)
        rag_engine = get_rag_engine(embedder, vector_store, llm_client)
    except Exception as exc:
        st.error(f"Failed to initialise core components: {exc}")
        logger.critical("Core init failed: %s", exc, exc_info=True)
        st.stop()

    # Disclaimer banner
    st.warning(DISCLAIMER)

    # Sidebar
    from ui.sidebar import render_sidebar
    patient_context, selected_model, _ = render_sidebar(
        llm_client=llm_client,
        on_ingest=lambda: ingest_documents(embedder, vector_store),
    )

    # Main tabs
    tab_chat, tab_profile, tab_kb, tab_history = st.tabs(
        ["Chat", "Patient Profile", "Knowledge Base", "History"]
    )

    with tab_chat:
        render_chat_tab(rag_engine, patient_context, selected_model)

    with tab_profile:
        render_profile_tab()

    with tab_kb:
        render_knowledge_base_tab(embedder, vector_store)

    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()