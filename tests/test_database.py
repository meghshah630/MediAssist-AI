"""
tests/test_database.py – Tests for db.database module.
"""

from __future__ import annotations

import sqlite3
import pytest


class TestInitDB:
    """Tests for the init_db function."""

    def test_init_db_creates_tables(self, tmp_db_path):
        """init_db should create all four expected tables."""
        from db.database import init_db
        init_db(tmp_db_path)

        with sqlite3.connect(tmp_db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        expected = {"patients", "chat_sessions", "chat_messages", "ingested_documents"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_init_db_idempotent(self, tmp_db_path):
        """Calling init_db twice should not raise errors."""
        from db.database import init_db
        init_db(tmp_db_path)
        init_db(tmp_db_path)  # Second call should be safe

    def test_patients_table_schema(self, tmp_db_path):
        """patients table should have the required columns."""
        from db.database import init_db
        init_db(tmp_db_path)

        with sqlite3.connect(tmp_db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(patients)").fetchall()
            }

        required_cols = {"id", "name", "age", "gender", "conditions", "medications", "allergies", "notes"}
        assert required_cols.issubset(cols)

    def test_chat_sessions_table_schema(self, tmp_db_path):
        """chat_sessions table should have the required columns."""
        from db.database import init_db
        init_db(tmp_db_path)

        with sqlite3.connect(tmp_db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }

        required_cols = {"id", "patient_id", "session_name", "created_at"}
        assert required_cols.issubset(cols)

    def test_chat_messages_table_schema(self, tmp_db_path):
        """chat_messages table should have the required columns."""
        from db.database import init_db
        init_db(tmp_db_path)

        with sqlite3.connect(tmp_db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
            }

        required_cols = {"id", "session_id", "role", "content", "sources_json", "timestamp"}
        assert required_cols.issubset(cols)

    def test_ingested_documents_table_schema(self, tmp_db_path):
        """ingested_documents table should have the required columns."""
        from db.database import init_db
        init_db(tmp_db_path)

        with sqlite3.connect(tmp_db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(ingested_documents)").fetchall()
            }

        required_cols = {"id", "filename", "chunk_count", "ingested_at"}
        assert required_cols.issubset(cols)

    def test_init_db_creates_parent_dirs(self, tmp_path):
        """init_db should create parent directories if they don't exist."""
        from db.database import init_db
        nested_path = str(tmp_path / "nested" / "deep" / "mediassist.db")
        init_db(nested_path)

        with sqlite3.connect(nested_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        assert len(tables) >= 4


class TestGetConnection:
    """Tests for the get_connection context manager."""

    def test_get_connection_yields_connection(self, tmp_db_path):
        """get_connection should yield a sqlite3.Connection."""
        from db.database import init_db, get_connection
        init_db(tmp_db_path)

        with get_connection(tmp_db_path) as conn:
            assert isinstance(conn, sqlite3.Connection)

    def test_get_connection_row_factory(self, tmp_db_path):
        """Rows should be accessible by column name."""
        from db.database import init_db, get_connection
        init_db(tmp_db_path)

        with get_connection(tmp_db_path) as conn:
            conn.execute("INSERT INTO ingested_documents (filename, chunk_count) VALUES (?, ?)", ("test.txt", 5))

        with get_connection(tmp_db_path) as conn:
            row = conn.execute("SELECT * FROM ingested_documents").fetchone()
            assert row["filename"] == "test.txt"
            assert row["chunk_count"] == 5

    def test_get_connection_rollback_on_error(self, tmp_db_path):
        """An exception inside the context should trigger rollback."""
        from db.database import init_db, get_connection
        init_db(tmp_db_path)

        try:
            with get_connection(tmp_db_path) as conn:
                conn.execute(
                    "INSERT INTO ingested_documents (filename, chunk_count) VALUES (?, ?)",
                    ("rollback_test.txt", 10),
                )
                raise ValueError("Simulated error to test rollback")
        except ValueError:
            pass

        with get_connection(tmp_db_path) as conn:
            row = conn.execute(
                "SELECT * FROM ingested_documents WHERE filename = ?",
                ("rollback_test.txt",),
            ).fetchone()
        assert row is None

