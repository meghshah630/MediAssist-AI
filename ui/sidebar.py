"""
ui/sidebar.py – ChatGPT-style conversation navigator sidebar.

Provides:
  - App branding and version
  - LLM status and model selector
  - New Chat button
  - Conversation search
  - Grouped conversation list (Today / Previous 7 Days / Older)
  - Per-conversation actions: switch, rename, delete
  - Light footer disclaimer
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from config import APP_NAME, APP_VERSION, SUPPORTED_MODELS, DEFAULT_MODEL, SQLITE_DB_PATH

logger = logging.getLogger(__name__)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse a SQLite datetime string to a UTC datetime object."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _group_sessions(sessions) -> Dict[str, list]:
    """Group session objects into Today / Previous 7 Days / Older buckets."""
    now = datetime.now(tz=timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today_start - timedelta(days=7)

    groups: Dict[str, list] = {"Today": [], "Previous 7 Days": [], "Older": []}
    for s in sessions:
        ts = _parse_ts(s.created_at)
        if ts is None or ts >= today_start:
            groups["Today"].append(s)
        elif ts >= week_ago:
            groups["Previous 7 Days"].append(s)
        else:
            groups["Older"].append(s)
    return groups


def _session_display_title(session) -> str:
    """Return the best display title for a session."""
    return session.title.strip() if session.title and session.title.strip() else session.session_name


# ── Main sidebar renderer ─────────────────────────────────────────────────────

def render_sidebar(
    llm_client=None,
    on_ingest=None,
) -> Tuple[Dict[str, Any], str, Optional[int]]:
    """
    Render the ChatGPT-style conversation navigator sidebar.

    Parameters
    ----------
    llm_client:
        OllamaClient instance used for health check and model listing.
    on_ingest:
        Callback (kept for backward compatibility; no longer shown here).

    Returns
    -------
    Tuple of:
      - patient_context dict (always empty – profile is managed in the Profile tab)
      - selected_model str
      - selected_session_id int | None
    """
    from db.chat_repo import (
        list_sessions,
        search_sessions,
        rename_session,
        delete_session,
    )

    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            f"<h1 style='color:#2563EB;font-size:1.35rem;margin-bottom:0'>"
            f"MediAssist AI</h1>"
            f"<p style='color:gray;font-size:0.75rem;margin-top:2px'>v{APP_VERSION} &nbsp;·&nbsp; Educational use only</p>",
            unsafe_allow_html=True,
        )

        # ── New Chat button ───────────────────────────────────────────────
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            _start_new_chat()

        st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

        # ── LLM Status + Model selector ───────────────────────────────────
        with st.expander("Model Settings", expanded=False):
            available_models: List[str] = []
            is_healthy = False
            if llm_client is not None:
                is_healthy = llm_client.health_check()
                if is_healthy:
                    st.markdown("**Status:** Online `http://localhost:11434`")
                    available_models = llm_client.list_models()
                    if not available_models:
                        st.warning(
                            "No models installed. Run:\n"
                            '```\nollama pull llama3.2\n```'
                        )
                else:
                    st.markdown("**Status:** Offline")
                    st.markdown(
                        "Start Ollama: `ollama serve`\n\n"
                        "Pull a model: `ollama pull llama3.2`"
                    )
            else:
                st.markdown("**Status:** Not configured")

            display_options = list(dict.fromkeys(SUPPORTED_MODELS + available_models))
            selected_model = st.selectbox(
                "Model",
                options=display_options,
                index=0,
                help="Select the Ollama model to use.",
                key="sidebar_model_select",
            )
            if is_healthy and selected_model and not llm_client.is_model_available(selected_model):
                st.info(f"Model `{selected_model}` not downloaded.\n`ollama pull {selected_model}`")
            else:
            # Read back previously chosen model without re-rendering the expander widgets
                selected_model = st.session_state.get("sidebar_model_select", DEFAULT_MODEL) or DEFAULT_MODEL

        st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

        # ── Conversation search ────────────────────────────────────────────
        search_q = st.text_input(
            "Search conversations",
            placeholder="Search…",
            key="conv_search",
            label_visibility="collapsed",
        )

        # ── Conversation list ──────────────────────────────────────────────
        try:
            if search_q and search_q.strip():
                sessions = search_sessions(search_q.strip(), db_path=SQLITE_DB_PATH)
            else:
                sessions = list_sessions(db_path=SQLITE_DB_PATH)
        except Exception as exc:
            st.error(f"Failed to load conversations: {exc}")
            sessions = []

        active_id = st.session_state.get("active_session_id")

        if not sessions:
            st.caption("No conversations yet. Ask a question to start!")
        else:
            groups = _group_sessions(sessions)
            for group_label, group_sessions in groups.items():
                if not group_sessions:
                    continue
                st.markdown(
                    f"<p style='font-size:0.7rem;color:gray;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:10px 0 4px 0'>{group_label}</p>",
                    unsafe_allow_html=True,
                )
                for session in group_sessions:
                    _render_conversation_item(
                        session=session,
                        is_active=(session.id == active_id),
                        db_path=SQLITE_DB_PATH,
                    )

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.70rem;color:gray;text-align:center'>"
            "For educational purposes only.<br>Not a substitute for medical advice.</p>",
            unsafe_allow_html=True,
        )

    # patient_context is now managed via Profile tab only
    patient_context: Dict[str, Any] = st.session_state.get("active_patient_context", {})
    selected_session_id = st.session_state.get("active_session_id")
    return patient_context, selected_model, selected_session_id


# ── Conversation item renderer ────────────────────────────────────────────────

def _render_conversation_item(session, is_active: bool, db_path: str) -> None:
    """Render a single conversation item with click, rename, and delete actions."""
    from db.chat_repo import rename_session, delete_session

    session_id = session.id
    display = _session_display_title(session)
    rename_key = f"renaming_{session_id}"
    confirm_key = f"confirm_del_{session_id}"

    # ── Rename mode ───────────────────────────────────────────────────────
    if st.session_state.get(rename_key, False):
        new_title = st.text_input(
            "Rename",
            value=display,
            key=f"rename_input_{session_id}",
            label_visibility="collapsed",
        )
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("Save", key=f"save_rename_{session_id}", use_container_width=True):
                try:
                    rename_session(session_id, new_title, db_path=db_path)
                except Exception as exc:
                    st.error(f"Rename failed: {exc}")
                st.session_state.pop(rename_key, None)
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key=f"cancel_rename_{session_id}", use_container_width=True):
                st.session_state.pop(rename_key, None)
                st.rerun()
        return

    # ── Delete confirmation mode ──────────────────────────────────────────
    if st.session_state.get(confirm_key, False):
        st.warning(f"Delete **{display}**?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Delete", key=f"yes_del_{session_id}", use_container_width=True):
                try:
                    delete_session(session_id, db_path=db_path)
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
                # If the deleted session was active, clear state
                if st.session_state.get("active_session_id") == session_id:
                    _start_new_chat()
                else:
                    st.session_state.pop(confirm_key, None)
                st.rerun()
        with col_no:
            if st.button("Cancel", key=f"no_del_{session_id}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
        return

    # ── Normal row ────────────────────────────────────────────────────────
    label_truncated = display[:42] + "…" if len(display) > 42 else display

    col_title, col_actions = st.columns([5, 1.2])
    with col_title:
        if st.button(
            label_truncated,
            key=f"conv_select_{session_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            help=display,
        ):
            _switch_to_session(session_id)

    with col_actions:
        # Compact "…" menu using a popover (Streamlit ≥ 1.37)
        if hasattr(st, "popover"):
            with st.popover("···", help="Rename or delete"):
                if st.button("Rename", key=f"btn_rename_{session_id}"):
                    st.session_state[rename_key] = True
                    st.rerun()
                if st.button("Delete", key=f"btn_del_{session_id}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
        else:
            # Fallback for older Streamlit: just show delete
            if st.button("✕", key=f"btn_del_{session_id}", help="Delete conversation"):
                st.session_state[confirm_key] = True
                st.rerun()


# ── Session management helpers ────────────────────────────────────────────────

def _start_new_chat() -> None:
    """Create a new chat session immediately and switch to it."""
    from db.chat_repo import create_session, get_messages

    active_id = st.session_state.get("active_session_id")
    # If the current session is already empty and new, simply clear in-memory state and stay on it
    if active_id:
        msgs = get_messages(active_id, db_path=SQLITE_DB_PATH)
        if not msgs:
            st.session_state["chat_messages"] = []
            st.session_state["session_title_set"] = False
            st.rerun()
            return

    counter = st.session_state.get("session_counter", 1)
    new_id = create_session(
        session_name=f"Session {counter}",
        title="New Chat",
        db_path=SQLITE_DB_PATH,
    )
    st.session_state["active_session_id"] = new_id
    st.session_state["chat_messages"] = []
    st.session_state["session_title_set"] = False
    st.session_state["session_counter"] = counter + 1
    # Clear any pending rename/confirm states
    keys_to_clear = [k for k in st.session_state if k.startswith("renaming_") or k.startswith("confirm_del_")]
    for k in keys_to_clear:
        del st.session_state[k]
    st.rerun()


def _switch_to_session(session_id: int) -> None:
    """Switch to an existing chat session, loading its messages from the DB."""
    from db.chat_repo import get_messages

    try:
        messages = get_messages(session_id, db_path=SQLITE_DB_PATH)
        st.session_state["active_session_id"] = session_id
        # Convert ChatMessage objects to the dict format used by in-memory state
        st.session_state["chat_messages"] = [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources or [],
            }
            for m in messages
        ]
        st.session_state["session_title_set"] = True
        logger.info("Switched to session id=%d (%d messages loaded).", session_id, len(messages))
    except Exception as exc:
        st.error(f"Failed to load conversation: {exc}")
        logger.error("_switch_to_session failed for id=%d: %s", session_id, exc)
    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_csv(raw: str) -> List[str]:
    """Split a comma-separated string into a cleaned list."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
