"""
tests/test_patient_repo.py – Tests for db.patient_repo CRUD operations.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def db(initialized_db):
    """Return the path to an initialized temporary database."""
    return initialized_db


class TestCreatePatient:
    """Tests for create_patient."""

    def test_create_patient_returns_id(self, db):
        """create_patient should return a positive integer ID."""
        from db.patient_repo import PatientProfile, create_patient
        p = PatientProfile(name="Alice Smith", age=45, gender="Female")
        patient_id = create_patient(p, db_path=db)
        assert isinstance(patient_id, int)
        assert patient_id > 0

    def test_create_patient_with_full_data(self, db):
        """create_patient should persist all fields."""
        from db.patient_repo import PatientProfile, create_patient, get_patient
        p = PatientProfile(
            name="Bob Jones",
            age=60,
            gender="Male",
            conditions=["Hypertension", "Diabetes"],
            medications=["Metformin", "Lisinopril"],
            allergies=["Penicillin"],
            notes="Patient requires regular blood pressure monitoring.",
        )
        patient_id = create_patient(p, db_path=db)
        loaded = get_patient(patient_id, db_path=db)
        assert loaded is not None
        assert loaded.name == "Bob Jones"
        assert loaded.age == 60
        assert "Hypertension" in loaded.conditions
        assert "Metformin" in loaded.medications
        assert "Penicillin" in loaded.allergies

    def test_create_minimal_patient(self, db):
        """Should create a patient with only a name."""
        from db.patient_repo import PatientProfile, create_patient, get_patient
        p = PatientProfile(name="Charlie Brown")
        patient_id = create_patient(p, db_path=db)
        loaded = get_patient(patient_id, db_path=db)
        assert loaded.name == "Charlie Brown"
        assert loaded.conditions == []
        assert loaded.medications == []
        assert loaded.allergies == []


class TestGetPatient:
    """Tests for get_patient."""

    def test_get_existing_patient(self, db):
        """Should return a PatientProfile for an existing patient."""
        from db.patient_repo import PatientProfile, create_patient, get_patient
        p = PatientProfile(name="Diana Prince", age=35)
        pid = create_patient(p, db_path=db)
        result = get_patient(pid, db_path=db)
        assert result is not None
        assert result.name == "Diana Prince"
        assert result.id == pid

    def test_get_nonexistent_patient_returns_none(self, db):
        """Should return None for a non-existent patient ID."""
        from db.patient_repo import get_patient
        result = get_patient(99999, db_path=db)
        assert result is None


class TestListPatients:
    """Tests for list_patients."""

    def test_list_patients_returns_all(self, db):
        """Should return all created patients."""
        from db.patient_repo import PatientProfile, create_patient, list_patients
        create_patient(PatientProfile(name="Eve Adams"), db_path=db)
        create_patient(PatientProfile(name="Frank Castle"), db_path=db)
        patients = list_patients(db_path=db)
        names = [p.name for p in patients]
        assert "Eve Adams" in names
        assert "Frank Castle" in names

    def test_list_patients_returns_list(self, db):
        """Should return a list."""
        from db.patient_repo import list_patients
        result = list_patients(db_path=db)
        assert isinstance(result, list)

    def test_list_patients_empty_db(self, tmp_db_path):
        """Should return empty list when no patients exist."""
        from db.database import init_db
        from db.patient_repo import list_patients
        init_db(tmp_db_path)
        result = list_patients(db_path=tmp_db_path)
        assert result == []


class TestUpdatePatient:
    """Tests for update_patient."""

    def test_update_patient_modifies_record(self, db):
        """update_patient should persist the updated values."""
        from db.patient_repo import PatientProfile, create_patient, get_patient, update_patient
        p = PatientProfile(name="Grace Hopper", age=40)
        pid = create_patient(p, db_path=db)

        p.id = pid
        p.age = 41
        p.conditions = ["Asthma"]
        update_patient(p, db_path=db)

        updated = get_patient(pid, db_path=db)
        assert updated.age == 41
        assert "Asthma" in updated.conditions

    def test_update_returns_true_on_success(self, db):
        """update_patient should return True when a row is updated."""
        from db.patient_repo import PatientProfile, create_patient, update_patient
        p = PatientProfile(name="Henry Ford", age=50)
        pid = create_patient(p, db_path=db)
        p.id = pid
        p.name = "Henry Ford Sr."
        result = update_patient(p, db_path=db)
        assert result is True

    def test_update_returns_false_for_missing_patient(self, db):
        """update_patient should return False when patient doesn't exist."""
        from db.patient_repo import PatientProfile, update_patient
        p = PatientProfile(name="Ghost", id=99999)
        result = update_patient(p, db_path=db)
        assert result is False

    def test_update_without_id_raises(self, db):
        """update_patient without an ID should raise ValueError."""
        from db.patient_repo import PatientProfile, update_patient
        p = PatientProfile(name="No ID")
        with pytest.raises(ValueError):
            update_patient(p, db_path=db)


class TestDeletePatient:
    """Tests for delete_patient."""

    def test_delete_existing_patient(self, db):
        """delete_patient should return True and remove the record."""
        from db.patient_repo import PatientProfile, create_patient, delete_patient, get_patient
        p = PatientProfile(name="Isaac Newton")
        pid = create_patient(p, db_path=db)
        result = delete_patient(pid, db_path=db)
        assert result is True
        assert get_patient(pid, db_path=db) is None

    def test_delete_nonexistent_returns_false(self, db):
        """delete_patient should return False for non-existent ID."""
        from db.patient_repo import delete_patient
        result = delete_patient(99999, db_path=db)
        assert result is False


class TestPatientProfileDataclass:
    """Tests for the PatientProfile dataclass."""

    def test_to_dict_returns_dict(self):
        """to_dict() should return a plain dictionary."""
        from db.patient_repo import PatientProfile
        p = PatientProfile(name="Test Patient", age=30, conditions=["Diabetes"])
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "Test Patient"
        assert d["age"] == 30
        assert "Diabetes" in d["conditions"]

