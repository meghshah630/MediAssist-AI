"""
tests/conftest.py – Shared pytest fixtures for MediAssist AI tests.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

import pytest

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root path."""
    return ROOT


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """Return a path to a temporary SQLite DB file."""
    return str(tmp_path / "test_mediassist.db")


@pytest.fixture
def memory_db_path() -> str:
    """Return the SQLite in-memory connection string."""
    return ":memory:"


@pytest.fixture
def initialized_db(tmp_db_path):
    """Create an initialised temporary database and return its path."""
    from db.database import init_db
    init_db(tmp_db_path)
    return tmp_db_path


@pytest.fixture
def sample_texts():
    """Return a small list of sample medical text strings."""
    return [
        "Hypertension is a chronic condition characterised by elevated blood pressure.",
        "Type 2 diabetes involves insulin resistance and relative insulin deficiency.",
        "Asthma is an inflammatory airway disease causing reversible bronchoconstriction.",
        "COVID-19 is caused by the SARS-CoV-2 coronavirus.",
        "CBT is the gold-standard psychological treatment for anxiety disorders.",
    ]


@pytest.fixture
def sample_text_file(tmp_path) -> Path:
    """Create a temporary .txt file with sample content."""
    content = (
        "Hypertension Overview\n\n"
        "Hypertension, commonly known as high blood pressure, is a chronic medical condition "
        "in which the blood pressure in the arteries is persistently elevated. "
        "Normal blood pressure is below 120/80 mmHg. "
        "Treatment includes lifestyle modifications and medications such as ACE inhibitors, "
        "ARBs, calcium channel blockers, and diuretics.\n\n"
        "Symptoms include headaches, dizziness, and shortness of breath.\n"
    )
    doc = tmp_path / "hypertension_test.txt"
    doc.write_text(content, encoding="utf-8")
    return doc


@pytest.fixture
def sample_docs_dir(tmp_path) -> Path:
    """Create a temporary directory with multiple sample .txt files."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir()

    files = {
        "hypertension.txt": "Hypertension is high blood pressure. Treatment includes ACE inhibitors and lifestyle changes.",
        "diabetes.txt": "Type 2 diabetes is characterised by insulin resistance. Metformin is first-line treatment.",
        "asthma.txt": "Asthma causes reversible airway obstruction. Inhalers are the main treatment.",
    }
    for fname, content in files.items():
        (docs_dir / fname).write_text(content, encoding="utf-8")

    return docs_dir


@pytest.fixture
def ephemeral_chroma_client():
    """Return an ephemeral (in-memory) ChromaDB client for testing."""
    import chromadb
    return chromadb.EphemeralClient()

