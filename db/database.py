"""
db/database.py – SQLite setup, schema migrations, and connection helpers.

Schema
------
patients(id, name, age, gender, conditions, medications, allergies, notes,
         created_at, updated_at)
chat_sessions(id, patient_id, session_name, title, created_at)
chat_messages(id, session_id, role, content, sources_json, timestamp)
ingested_documents(id, filename, chunk_count, ingested_at)

Schema Migrations
-----------------
v1.1.0: Added `title` column to chat_sessions (auto-generated from first user message).
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────────

_CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    age         INTEGER,
    gender      TEXT,
    conditions  TEXT    DEFAULT '[]',
    medications TEXT    DEFAULT '[]',
    allergies   TEXT    DEFAULT '[]',
    notes       TEXT    DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now')),
    updated_at  TEXT    DEFAULT (datetime('now'))
);
"""

_CREATE_CHAT_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER REFERENCES patients(id) ON DELETE SET NULL,
    session_name TEXT    NOT NULL DEFAULT 'Session',
    title        TEXT    NOT NULL DEFAULT '',
    created_at   TEXT    DEFAULT (datetime('now'))
);
"""

_CREATE_CHAT_MESSAGES = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role         TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content      TEXT    NOT NULL,
    sources_json TEXT    DEFAULT '[]',
    timestamp    TEXT    DEFAULT (datetime('now'))
);
"""

_CREATE_INGESTED_DOCS = """
CREATE TABLE IF NOT EXISTS ingested_documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    ingested_at TEXT    DEFAULT (datetime('now'))
);
"""

_ALL_DDL = [
    _CREATE_PATIENTS,
    _CREATE_CHAT_SESSIONS,
    _CREATE_CHAT_MESSAGES,
    _CREATE_INGESTED_DOCS,
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_db_path(db_path: str | None) -> str:
    """Return the effective DB path, ensuring parent dirs exist."""
    path = db_path or SQLITE_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(db_path: str | None = None) -> None:
    """
    Create all tables if they don't already exist, and apply schema migrations.

    Parameters
    ----------
    db_path:
        Optional path override (useful for tests using ``:memory:``).
    """
    resolved = _resolve_db_path(db_path)
    try:
        with sqlite3.connect(resolved) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            for ddl in _ALL_DDL:
                conn.execute(ddl)
            conn.commit()
            # ── Schema migrations ─────────────────────────────────────────
            # v1.1.0: Add `title` column to chat_sessions if it doesn't exist
            cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()}
            if "title" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN title TEXT NOT NULL DEFAULT ''")
                conn.commit()
                logger.info("Migration v1.1.0: added 'title' column to chat_sessions.")
        logger.info("Database initialised at '%s'.", resolved)
    except Exception as exc:
        logger.error("init_db failed: %s", exc)
        raise


@contextmanager
def get_connection(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection.

    Usage
    -----
    >>> with get_connection() as conn:
    ...     conn.execute("SELECT 1")

    Parameters
    ----------
    db_path:
        Optional path override.
    """
    resolved = _resolve_db_path(db_path)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

