"""
tests/test_chat_repo.py – Tests for db.chat_repo operations.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def db(initialized_db):
    """Return the path to an initialized temporary database."""
    return initialized_db


@pytest.fixture
def session_id(db):
    """Create a test session and return its ID."""
    from db.chat_repo import create_session
    return create_session("Test Session", db_path=db)


class TestCreateSession:
    """Tests for create_session."""

    def test_create_session_returns_id(self, db):
        """create_session should return a positive integer."""
        from db.chat_repo import create_session
        sid = create_session("My Session", db_path=db)
        assert isinstance(sid, int)
        assert sid > 0

    def test_create_session_with_patient_id(self, db):
        """create_session should accept a patient_id."""
        from db.patient_repo import PatientProfile, create_patient
        from db.chat_repo import create_session, list_sessions
        pid = create_patient(PatientProfile(name="Test Patient"), db_path=db)
        sid = create_session("Patient Session", patient_id=pid, db_path=db)
        sessions = list_sessions(patient_id=pid, db_path=db)
        assert any(s.id == sid for s in sessions)


class TestListSessions:
    """Tests for list_sessions."""

    def test_list_sessions_returns_list(self, db):
        """Should return a list of ChatSession objects."""
        from db.chat_repo import list_sessions
        result = list_sessions(db_path=db)
        assert isinstance(result, list)

    def test_list_sessions_includes_created(self, db, session_id):
        """Newly created session should appear in list."""
        from db.chat_repo import list_sessions
        sessions = list_sessions(db_path=db)
        ids = [s.id for s in sessions]
        assert session_id in ids


class TestDeleteSession:
    """Tests for delete_session."""

    def test_delete_existing_session(self, db, session_id):
        """delete_session should return True for existing session."""
        from db.chat_repo import delete_session, list_sessions
        result = delete_session(session_id, db_path=db)
        assert result is True
        sessions = list_sessions(db_path=db)
        assert not any(s.id == session_id for s in sessions)

    def test_delete_nonexistent_session_returns_false(self, db):
        """delete_session should return False for non-existent ID."""
        from db.chat_repo import delete_session
        result = delete_session(99999, db_path=db)
        assert result is False


class TestSaveMessage:
    """Tests for save_message."""

    def test_save_message_returns_id(self, db, session_id):
        """save_message should return a positive integer ID."""
        from db.chat_repo import save_message, ChatMessage
        msg = ChatMessage(session_id=session_id, role="user", content="Hello, how are you?")
        mid = save_message(msg, db_path=db)
        assert isinstance(mid, int)
        assert mid > 0

    def test_save_user_message(self, db, session_id):
        """Should persist a user message correctly."""
        from db.chat_repo import save_message, get_messages, ChatMessage
        msg = ChatMessage(session_id=session_id, role="user", content="What is hypertension?")
        save_message(msg, db_path=db)
        messages = get_messages(session_id, db_path=db)
        assert any(m.content == "What is hypertension?" for m in messages)

    def test_save_assistant_message_with_sources(self, db, session_id):
        """Should persist assistant message with sources."""
        from db.chat_repo import save_message, get_messages, ChatMessage
        sources = [{"source": "hypertension.txt", "chunk_id": "chunk_001"}]
        msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content="Hypertension is treated with ACE inhibitors.",
            sources=sources,
        )
        save_message(msg, db_path=db)
        messages = get_messages(session_id, db_path=db)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) > 0
        assert assistant_msgs[0].sources == sources


class TestGetMessages:
    """Tests for get_messages."""

    def test_get_messages_returns_list(self, db, session_id):
        """get_messages should return a list."""
        from db.chat_repo import get_messages
        result = get_messages(session_id, db_path=db)
        assert isinstance(result, list)

    def test_get_messages_in_order(self, db, session_id):
        """Messages should be returned in chronological order."""
        from db.chat_repo import save_message, get_messages, ChatMessage
        import time
        msgs = ["First message", "Second message", "Third message"]
        for content in msgs:
            save_message(ChatMessage(session_id=session_id, role="user", content=content), db_path=db)
            time.sleep(0.01)

        result = get_messages(session_id, db_path=db)
        contents = [m.content for m in result]
        for msg in msgs:
            assert msg in contents

    def test_get_messages_empty_session(self, db, session_id):
        """New session should have no messages."""
        from db.chat_repo import get_messages
        result = get_messages(session_id, db_path=db)
        assert result == []


class TestGetRecentMessages:
    """Tests for get_recent_messages."""

    def test_get_recent_messages_limit(self, db, session_id):
        """Should return at most 'limit' messages."""
        from db.chat_repo import save_message, get_recent_messages, ChatMessage
        for i in range(10):
            save_message(
                ChatMessage(session_id=session_id, role="user", content=f"Message {i}"),
                db_path=db,
            )
        result = get_recent_messages(session_id, limit=5, db_path=db)
        assert len(result) <= 5

    def test_get_recent_messages_returns_latest(self, db, session_id):
        """Should return the most recent messages (by id/rowid which is insertion order)."""
        from db.chat_repo import save_message, get_recent_messages, ChatMessage
        for i in range(8):
            save_message(
                ChatMessage(session_id=session_id, role="user", content=f"Msg {i}"),
                db_path=db,
            )
        result = get_recent_messages(session_id, limit=3, db_path=db)
        # Should return exactly 3 messages
        assert len(result) == 3
        # All returned messages should belong to the session
        for m in result:
            assert m.session_id == session_id


class TestIngestedDocuments:
    """Tests for record_ingested_document and list_ingested_documents."""

    def test_record_and_list_documents(self, db):
        """Should record and retrieve ingested document info."""
        from db.chat_repo import record_ingested_document, list_ingested_documents
        record_ingested_document("hypertension.txt", 10, db_path=db)
        docs = list_ingested_documents(db_path=db)
        assert any(d["filename"] == "hypertension.txt" for d in docs)

    def test_record_returns_id(self, db):
        """record_ingested_document should return a positive integer."""
        from db.chat_repo import record_ingested_document
        doc_id = record_ingested_document("diabetes.txt", 8, db_path=db)
        assert isinstance(doc_id, int)
        assert doc_id > 0


class TestSessionTitlesAndSearch:
    """Tests for session titles, rename_session, update_session_title, and search_sessions."""

    def test_create_session_with_title(self, db):
        """create_session should persist the title field."""
        from db.chat_repo import create_session, list_sessions
        sid = create_session("Session A", title="What is Asthma?", db_path=db)
        sessions = list_sessions(db_path=db)
        match = next((s for s in sessions if s.id == sid), None)
        assert match is not None
        assert match.title == "What is Asthma?"

    def test_rename_session(self, db, session_id):
        """rename_session should update the title."""
        from db.chat_repo import rename_session, list_sessions
        success = rename_session(session_id, "Renamed Medical Discussion", db_path=db)
        assert success is True
        sessions = list_sessions(db_path=db)
        match = next(s for s in sessions if s.id == session_id)
        assert match.title == "Renamed Medical Discussion"

    def test_update_session_title_truncates(self, db, session_id):
        """update_session_title should truncate very long strings."""
        from db.chat_repo import update_session_title, list_sessions
        long_title = "A" * 100
        success = update_session_title(session_id, long_title, db_path=db)
        assert success is True
        sessions = list_sessions(db_path=db)
        match = next(s for s in sessions if s.id == session_id)
        assert len(match.title) <= 60

    def test_search_sessions_matches_title_and_name(self, db):
        """search_sessions should match on substring of title or session_name."""
        from db.chat_repo import create_session, search_sessions
        s1 = create_session("Session 10", title="Cardiac arrest protocol", db_path=db)
        s2 = create_session("Diabetes Consultation", title="Blood sugar ranges", db_path=db)
        s3 = create_session("General Health", title="Nutrition guidance", db_path=db)

        # Search by title keyword
        cardiac_results = search_sessions("cardiac", db_path=db)
        assert any(s.id == s1 for s in cardiac_results)
        assert not any(s.id == s2 for s in cardiac_results)

        # Search by session name
        diab_results = search_sessions("Diabetes", db_path=db)
        assert any(s.id == s2 for s in diab_results)

        # Empty search returns all
        all_results = search_sessions("", db_path=db)
        ids = [s.id for s in all_results]
        assert s1 in ids and s2 in ids and s3 in ids

