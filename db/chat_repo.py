"""
db/chat_repo.py – Chat session and message CRUD operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from db.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a chat session."""
    session_name: str
    patient_id: Optional[int] = None
    id: Optional[int] = None
    title: str = ""
    created_at: Optional[str] = None


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    session_id: int
    role: str   # 'user' | 'assistant' | 'system'
    content: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    id: Optional[int] = None
    timestamp: Optional[str] = None


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(
    session_name: str,
    patient_id: Optional[int] = None,
    db_path: str | None = None,
    title: str = "",
) -> int:
    """
    Create a new chat session.

    Parameters
    ----------
    session_name:
        Internal session name (e.g. "Session 1").
    patient_id:
        Optional patient association.
    title:
        Display title shown in sidebar (auto-generated from first user question).

    Returns
    -------
    The new session's integer ID.
    """
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO chat_sessions (patient_id, session_name, title) VALUES (?, ?, ?)",
                (patient_id, session_name, title),
            )
            session_id = cur.lastrowid
            logger.info("Created chat session id=%d name='%s'.", session_id, session_name)
            return session_id
    except Exception as exc:
        logger.error("create_session failed: %s", exc)
        raise


def list_sessions(
    patient_id: Optional[int] = None,
    db_path: str | None = None,
) -> List[ChatSession]:
    """List sessions, optionally filtered by patient."""
    try:
        with get_connection(db_path) as conn:
            if patient_id is not None:
                rows = conn.execute(
                    "SELECT * FROM chat_sessions WHERE patient_id = ? ORDER BY created_at DESC",
                    (patient_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chat_sessions ORDER BY created_at DESC"
                ).fetchall()
        return [_row_to_session(r) for r in rows]
    except Exception as exc:
        logger.error("list_sessions failed: %s", exc)
        raise


def delete_session(session_id: int, db_path: str | None = None) -> bool:
    """Delete a session and its messages (ON DELETE CASCADE)."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
            )
            return cur.rowcount > 0
    except Exception as exc:
        logger.error("delete_session failed: %s", exc)
        raise


def rename_session(session_id: int, new_title: str, db_path: str | None = None) -> bool:
    """
    Rename a chat session's display title.

    Parameters
    ----------
    session_id:
        ID of the session to rename.
    new_title:
        New display title (shown in sidebar). Truncated to 60 chars.

    Returns
    -------
    True if the session was found and updated.
    """
    new_title = (new_title or "").strip()[:60]
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ?",
                (new_title, session_id),
            )
            updated = cur.rowcount > 0
            if updated:
                logger.info("Renamed session id=%d to '%s'.", session_id, new_title)
            return updated
    except Exception as exc:
        logger.error("rename_session failed: %s", exc)
        raise


def update_session_title(session_id: int, title: str, db_path: str | None = None) -> bool:
    """
    Set the auto-generated display title for a session (from first user message).

    Parameters
    ----------
    session_id:
        Target session ID.
    title:
        Concise display title (truncated to 60 chars).

    Returns
    -------
    True if updated.
    """
    return rename_session(session_id, title, db_path)


def search_sessions(
    query: str,
    db_path: str | None = None,
) -> List["ChatSession"]:
    """
    Search chat sessions by title or session_name (case-insensitive substring match).

    Parameters
    ----------
    query:
        Search string. Empty/blank returns all sessions.

    Returns
    -------
    Matching sessions ordered by created_at DESC.
    """
    if not query or not query.strip():
        return list_sessions(db_path=db_path)
    pattern = f"%{query.strip()}%"
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_sessions
                WHERE title LIKE ? OR session_name LIKE ?
                ORDER BY created_at DESC
                """,
                (pattern, pattern),
            ).fetchall()
        return [_row_to_session(r) for r in rows]
    except Exception as exc:
        logger.error("search_sessions failed: %s", exc)
        raise


# ── Message CRUD ──────────────────────────────────────────────────────────────

def save_message(message: ChatMessage, db_path: str | None = None) -> int:
    """
    Persist a chat message.

    Returns
    -------
    The new message's integer ID.
    """
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, sources_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    message.session_id,
                    message.role,
                    message.content,
                    json.dumps(message.sources),
                ),
            )
            message_id = cur.lastrowid
            logger.debug(
                "Saved message id=%d role=%s session=%d.",
                message_id, message.role, message.session_id,
            )
            return message_id
    except Exception as exc:
        logger.error("save_message failed: %s", exc)
        raise


def get_messages(session_id: int, db_path: str | None = None) -> List[ChatMessage]:
    """Return all messages for a session, oldest first."""
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_message(r) for r in rows]
    except Exception as exc:
        logger.error("get_messages failed: %s", exc)
        raise


def get_recent_messages(
    session_id: int,
    limit: int = 20,
    db_path: str | None = None,
) -> List[ChatMessage]:
    """Return the most recent *limit* messages for a session."""
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [_row_to_message(r) for r in rows]
    except Exception as exc:
        logger.error("get_recent_messages failed: %s", exc)
        raise


# ── Ingested documents ────────────────────────────────────────────────────────

def record_ingested_document(
    filename: str,
    chunk_count: int,
    db_path: str | None = None,
) -> int:
    """Record that a document was ingested into the knowledge base."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO ingested_documents (filename, chunk_count) VALUES (?, ?)",
                (filename, chunk_count),
            )
            return cur.lastrowid
    except Exception as exc:
        logger.error("record_ingested_document failed: %s", exc)
        raise


def list_ingested_documents(db_path: str | None = None) -> List[Dict[str, Any]]:
    """Return all ingested document records."""
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM ingested_documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("list_ingested_documents failed: %s", exc)
        raise


# ── Internal helpers ──────────────────────────────────────────────────────────

def _row_to_session(row) -> ChatSession:
    # title column may not exist in older databases before migration
    try:
        title = row["title"] or ""
    except (IndexError, KeyError):
        title = ""
    return ChatSession(
        id=row["id"],
        patient_id=row["patient_id"],
        session_name=row["session_name"],
        title=title,
        created_at=row["created_at"],
    )


def _row_to_message(row) -> ChatMessage:
    sources = []
    try:
        raw = row["sources_json"]
        if raw:
            sources = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    return ChatMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        sources=sources,
        timestamp=row["timestamp"],
    )

