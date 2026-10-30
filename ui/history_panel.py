"""
ui/history_panel.py – Past chat sessions browser panel.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_history_panel(db_path: Optional[str] = None) -> None:
    """
    Render the chat history browser panel (Tab 4 – History).

    Parameters
    ----------
    db_path:
        Optional SQLite DB path override.
    """
    st.header("Chat History")

    try:
        from db.chat_repo import list_sessions, get_messages, delete_session

        sessions = list_sessions(db_path=db_path)

        if not sessions:
            st.info("No chat sessions recorded yet. Start chatting to create history.")
            return

        st.caption(f"Found **{len(sessions)}** session(s).")

        for session in sessions:
            display_title = session.title if session.title else session.session_name
            col_name, col_date, col_del = st.columns([5, 2, 1])
            with col_name:
                if st.button(
                    display_title,
                    key=f"hist_open_{session.id}",
                    use_container_width=True,
                ):
                    st.session_state["active_session_id"] = session.id
                    st.session_state["view_session_messages"] = session.id
                    msgs = get_messages(session.id, db_path=db_path)
                    st.session_state["chat_messages"] = [
                        {"role": m.role, "content": m.content, "sources": m.sources or []}
                        for m in msgs
                    ]
                    st.session_state["session_title_set"] = bool(session.title and session.title != "New Chat")

            with col_date:
                st.caption(session.created_at or "")

            with col_del:
                if st.button("Delete", key=f"hist_del_{session.id}", help="Delete session"):
                    try:
                        delete_session(session.id, db_path=db_path)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

        # Show messages for the selected session
        view_id = st.session_state.get("view_session_messages")
        if view_id:
            st.markdown("---")
            msgs = get_messages(view_id, db_path=db_path)
            st.subheader(f"Session {view_id} – {len(msgs)} message(s)")
            for msg in msgs:
                with st.expander(
                    f"{msg.role.capitalize()} – {msg.timestamp}",
                    expanded=False,
                ):
                    st.markdown(msg.content)
                    if msg.sources:
                        st.caption(f"Sources: {', '.join(s.get('source', '?') for s in msg.sources)}")

    except Exception as exc:
        st.error(f"History panel error: {exc}")
        logger.error("History panel error: %s", exc)

