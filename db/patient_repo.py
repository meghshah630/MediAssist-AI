"""
db/patient_repo.py – Patient CRUD operations with typed dataclasses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from db.database import get_connection, init_db

logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    """Represents a patient's medical profile."""
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    notes: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for passing to RAG engine, etc.)."""
        return asdict(self)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_patient(patient: PatientProfile, db_path: str | None = None) -> int:
    """
    Insert a new patient record.

    Returns
    -------
    The newly created patient's integer ID.
    """
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO patients (name, age, gender, conditions, medications,
                                      allergies, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient.name,
                    patient.age,
                    patient.gender,
                    json.dumps(patient.conditions),
                    json.dumps(patient.medications),
                    json.dumps(patient.allergies),
                    patient.notes,
                ),
            )
            patient_id = cur.lastrowid
            logger.info("Created patient id=%d name='%s'.", patient_id, patient.name)
            return patient_id
    except Exception as exc:
        logger.error("create_patient failed: %s", exc)
        raise


def get_patient(patient_id: int, db_path: str | None = None) -> Optional[PatientProfile]:
    """
    Fetch a patient by primary key.

    Returns
    -------
    ``PatientProfile`` or ``None`` if not found.
    """
    try:
        with get_connection(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_patient(row)
    except Exception as exc:
        logger.error("get_patient failed: %s", exc)
        raise


def list_patients(db_path: str | None = None) -> List[PatientProfile]:
    """Return all patients ordered by name."""
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY name"
            ).fetchall()
        return [_row_to_patient(r) for r in rows]
    except Exception as exc:
        logger.error("list_patients failed: %s", exc)
        raise


def update_patient(patient: PatientProfile, db_path: str | None = None) -> bool:
    """
    Update an existing patient record.

    Returns
    -------
    True if a row was updated, False if the patient was not found.
    """
    if patient.id is None:
        raise ValueError("PatientProfile.id must be set to update a record.")
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                """
                UPDATE patients
                SET name        = ?,
                    age         = ?,
                    gender      = ?,
                    conditions  = ?,
                    medications = ?,
                    allergies   = ?,
                    notes       = ?,
                    updated_at  = datetime('now')
                WHERE id = ?
                """,
                (
                    patient.name,
                    patient.age,
                    patient.gender,
                    json.dumps(patient.conditions),
                    json.dumps(patient.medications),
                    json.dumps(patient.allergies),
                    patient.notes,
                    patient.id,
                ),
            )
            updated = cur.rowcount > 0
            if updated:
                logger.info("Updated patient id=%d.", patient.id)
            else:
                logger.warning("update_patient: patient id=%d not found.", patient.id)
            return updated
    except Exception as exc:
        logger.error("update_patient failed: %s", exc)
        raise


def delete_patient(patient_id: int, db_path: str | None = None) -> bool:
    """
    Delete a patient by ID.

    Returns
    -------
    True if deleted, False if not found.
    """
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM patients WHERE id = ?", (patient_id,)
            )
            deleted = cur.rowcount > 0
            if deleted:
                logger.info("Deleted patient id=%d.", patient_id)
            return deleted
    except Exception as exc:
        logger.error("delete_patient failed: %s", exc)
        raise


# ── Internal helpers ─────────────────────────────────────────────────────────

def _row_to_patient(row) -> PatientProfile:
    """Convert a sqlite3.Row to a PatientProfile."""
    return PatientProfile(
        id=row["id"],
        name=row["name"],
        age=row["age"],
        gender=row["gender"],
        conditions=_safe_json_list(row["conditions"]),
        medications=_safe_json_list(row["medications"]),
        allergies=_safe_json_list(row["allergies"]),
        notes=row["notes"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _safe_json_list(value: str | None) -> List[str]:
    """Parse a JSON-encoded list, returning [] on failure."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

