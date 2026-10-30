"""
ui/chat_window.py – Chat messages display with source citations and voice output controls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_chat_window(messages: List[Dict[str, Any]]) -> None:
    """
    Render a list of chat messages as styled bubbles with source citations
    and real-time browser-native Web Speech read-along players.

    Parameters
    ----------
    messages:
        List of dicts with keys ``role`` ('user'|'assistant') and ``content``.
        Optionally includes ``sources`` (list of source dicts).
    """
    from ui.voice_controls import render_read_along_player

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                if sources:
                    _render_sources(sources)
                # Real-time read-along player with word highlighting & native Web Speech
                render_read_along_player(msg_idx=i, text=content, key_prefix="chat_read_along")


def render_message(
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    msg_idx: Optional[int] = None,
) -> None:
    """
    Render a single chat message with optional sources and real-time read-along player.

    Parameters
    ----------
    role:     ``'user'`` or ``'assistant'``.
    content:  Message text (markdown supported).
    sources:  Optional list of source dicts.
    msg_idx:  Optional message index for unique component ID.
    """
    from ui.voice_controls import render_read_along_player

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            if sources:
                _render_sources(sources)
            if msg_idx is not None:
                render_read_along_player(msg_idx=msg_idx, text=content, key_prefix="single_read_along")


def render_chat_input(placeholder: str = "Ask a medical question…") -> Optional[str]:
    """
    Render the chat input box and return the user's input (or None).

    Parameters
    ----------
    placeholder:
        Hint text displayed in the input box.

    Returns
    -------
    The submitted query string, or ``None`` if nothing was submitted.
    """
    return st.chat_input(placeholder)


def render_thinking_indicator() -> Any:
    """Return a Streamlit spinner context manager for 'thinking' state."""
    return st.spinner("Generating response…")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _render_sources(sources: List[Dict]) -> None:
    """Render source citations as collapsible expanders."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, start=1):
            src_name = source.get("source", f"Source {i}")
            meta = source.get("metadata", {})
            topic = source.get("topic") or (meta.get("topic") if meta else None)
            header = f"**{i}. {topic}** (`{src_name}`)" if topic else f"**{i}. {src_name}**"
            st.markdown(header)
            if meta:
                detail_parts = []
                if "chunk_id" in meta:
                    detail_parts.append(f"Chunk: `{meta['chunk_id']}`")
                if "category" in meta and meta["category"]:
                    detail_parts.append(f"Category: *{meta['category']}*")
                if detail_parts:
                    st.caption(" | ".join(detail_parts))
            if i < len(sources):
                st.markdown("---")
