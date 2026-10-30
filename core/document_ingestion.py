"""
core/document_ingestion.py - Load, clean, chunk, and ingest medical text documents.

Improvements over v1:
- Topic metadata tagging on each chunk for retrieval filtering
- Duplicate chunk detection via SHA-256 content hash stored in metadata
- Smaller default chunk size (200 words) for more precise retrieval
- Provenance metadata: source, topic, category, chunk_index, content_hash
- Encoding artifact cleanup (fixes Windows-1252 / UTF-8 mixed-encoding garbage)
- Repeatable ingestion: existing chunk IDs are skipped to prevent duplicates
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic metadata catalogue
# Maps filename stem -> (topic_label, category)
# Add new entries here when adding new documents; no other code changes needed.
# ---------------------------------------------------------------------------
_TOPIC_CATALOGUE: Dict[str, tuple] = {
    "hypertension":                ("Hypertension / High Blood Pressure", "cardiovascular"),
    "diabetes_type2":              ("Type 2 Diabetes Mellitus",           "metabolic"),
    "asthma":                      ("Asthma",                             "respiratory"),
    "covid19":                     ("COVID-19",                           "infectious_disease"),
    "cardiovascular_disease":      ("Cardiovascular Disease",             "cardiovascular"),
    "general_first_aid":           ("First Aid and Emergency Care",       "emergency"),
    "medication_interactions":     ("Medication Interactions and Safety", "pharmacology"),
    "mental_health_anxiety":       ("Anxiety Disorders / Mental Health",  "mental_health"),
    "mental_health_depression":    ("Depression / Mood Disorders",        "mental_health"),
    "fever_and_common_infections": ("Fever and Common Infections",        "infectious_disease"),
    "weight_management_and_obesity": ("Weight Management and Obesity",    "lifestyle"),
    "nutrition_and_healthy_eating": ("Nutrition and Healthy Eating",      "lifestyle"),
    "anemia_and_blood_disorders":  ("Anemia and Blood Disorders",         "hematology"),
    "sleep_health_and_disorders":  ("Sleep Health and Sleep Disorders",   "lifestyle"),
    "preventive_care_and_screening": ("Preventive Care and Screening",   "preventive"),
    "respiratory_infections":      ("Respiratory Infections",             "respiratory"),
    "emergency_warning_signs":     ("Medical Emergency Warning Signs",    "emergency"),
    "hydration_and_water_intake":  ("Hydration and Water Intake",         "lifestyle"),
}

_DEFAULT_TOPIC = ("General Medical Information", "general")


def _get_topic_metadata(filename_stem: str) -> tuple:
    """Return (topic_label, category) for a given document filename stem."""
    return _TOPIC_CATALOGUE.get(filename_stem.lower(), _DEFAULT_TOPIC)


def load_document(filepath: str) -> str:
    """
    Read a plain-text (.txt) file and return its contents.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the file.

    Returns
    -------
    The file's text content as a string.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    IOError:
        If the file cannot be read.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        logger.debug("Loaded document: %s (%d chars)", filepath, len(text))
        return text
    except Exception as exc:
        logger.error("Failed to load document '%s': %s", filepath, exc)
        raise IOError(f"Cannot read file: {filepath}") from exc


def clean_text(text: str) -> str:
    """
    Normalise whitespace, remove encoding artifacts, and clean raw document text.

    Handles:
    - Windows CRLF line endings
    - UTF-8 replacement characters (U+FFFD) from encoding mismatches
    - Windows-1252 / Latin-1 multi-byte garbage sequences (e.g. 'â€"' for em-dash)
    - Excessive blank lines and intra-line whitespace

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    Cleaned text string.
    """
    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Unicode replacement characters from encoding errors
    text = text.replace("\ufffd", "")

    # Fix common Windows-1252 -> UTF-8 mis-decoding artifacts
    # Use unicode escape sequences to avoid source-file encoding issues
    encoding_fixes = {
        "\u00e2\u0080\u0093": "-",   # en-dash (UTF-8 mis-decoded)
        "\u00e2\u0080\u0094": "-",   # em-dash (UTF-8 mis-decoded)
        "\u00e2\u0080\u0099": "'",   # right single quotation mark
        "\u00e2\u0080\u009c": '"',   # left double quotation mark
        "\u00e2\u0080\u009d": '"',   # right double quotation mark
        "\u00e2\u0080\u0098": "'",   # left single quotation mark
        "\u00c2\u00b0":       "\u00b0",  # degree sign
        "\u00c2\u00a9":       "\u00a9",  # copyright sign
        "\u00c3\u00a9":       "e",   # e acute
        "\u00c3\u00a8":       "e",   # e grave
        "\u00c3\u00bc":       "u",   # u umlaut
    }
    for bad, good in encoding_fixes.items():
        text = text.replace(bad, good)

    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse intra-line whitespace (tabs, multiple spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def compute_content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the normalised text for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
) -> List[str]:
    """
    Split *text* into overlapping word-based sliding-window chunks.

    Default chunk_size reduced from 512 to 200 words to improve retrieval
    precision -- smaller chunks are more topically focused, reducing the chance
    that a chunk about hypertension medications is retrieved for a weight-loss
    question.

    Parameters
    ----------
    text:
        The cleaned document text.
    chunk_size:
        Approximate number of words per chunk (default: 200).
    overlap:
        Number of words to overlap between consecutive chunks (default: 30).

    Returns
    -------
    List of text chunks.
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    logger.debug(
        "chunk_text: %d words -> %d chunks (size=%d, overlap=%d)",
        len(words), len(chunks), chunk_size, overlap,
    )
    return chunks


def ingest_directory(directory: str) -> List[Dict]:
    """
    Load every .txt file in *directory*, clean and chunk each one.

    Returns chunks enriched with provenance metadata:
    - source:        filename (e.g. "hypertension.txt")
    - chunk_id:      deterministic stable ID based on source + chunk index
    - chunk_index:   zero-based position within the source document
    - topic:         human-readable topic label from catalogue
    - category:      broad medical category (e.g. "cardiovascular", "lifestyle")
    - content_hash:  SHA-256 prefix of chunk text (used for duplicate detection)

    Parameters
    ----------
    directory:
        Path to the folder containing medical text files.

    Returns
    -------
    List of dicts with keys: text, source, chunk_id, chunk_index,
    topic, category, content_hash.
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in '%s'.", directory)
        return []

    all_chunks: List[Dict] = []
    seen_hashes: set = set()

    for filepath in txt_files:
        filename = filepath.name
        stem = filepath.stem
        topic_label, category = _get_topic_metadata(stem)

        try:
            raw = load_document(str(filepath))
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            doc_chunks_added = 0
            for idx, chunk in enumerate(chunks):
                content_hash = compute_content_hash(chunk)

                # Skip exact duplicate chunks (same content hash across all docs)
                if content_hash in seen_hashes:
                    logger.debug(
                        "Skipping duplicate chunk (hash=%s) in '%s' idx=%d",
                        content_hash, filename, idx,
                    )
                    continue
                seen_hashes.add(content_hash)

                # Use stable, deterministic chunk ID: stem + index
                # (not random UUID so re-ingestion produces the same IDs
                # allowing ChromaDB upsert to skip existing entries)
                chunk_id = f"{stem}_chunk_{idx:04d}"

                all_chunks.append({
                    "text":         chunk,
                    "source":       filename,
                    "chunk_id":     chunk_id,
                    "chunk_index":  idx,
                    "topic":        topic_label,
                    "category":     category,
                    "content_hash": content_hash,
                })
                doc_chunks_added += 1

            logger.info(
                "Ingested '%s': %d chunk(s) added (topic: %s, category: %s).",
                filename, doc_chunks_added, topic_label, category,
            )
        except Exception as exc:
            logger.error("Skipping '%s' due to error: %s", filename, exc)
            continue

    logger.info(
        "ingest_directory complete: %d total chunks from %d file(s).",
        len(all_chunks), len(txt_files),
    )
    return all_chunks


def get_knowledge_base_manifest(directory: str) -> List[Dict]:
    """
    Return a manifest of documents in the directory with their topic metadata.

    Used by the coverage report to identify what topics are available
    without actually ingesting into ChromaDB.

    Parameters
    ----------
    directory:
        Path to the folder containing medical text files.

    Returns
    -------
    List of dicts with keys: filename, stem, topic, category, size_bytes, exists.
    """
    dir_path = Path(directory)
    manifest = []

    # First: documents actually present
    if dir_path.exists():
        for filepath in sorted(dir_path.glob("*.txt")):
            stem = filepath.stem
            topic_label, category = _get_topic_metadata(stem)
            manifest.append({
                "filename":   filepath.name,
                "stem":       stem,
                "topic":      topic_label,
                "category":   category,
                "size_bytes": filepath.stat().st_size,
                "exists":     True,
            })

    # Second: catalogued topics that are MISSING from disk
    present_stems = {m["stem"] for m in manifest}
    for stem, (topic_label, category) in _TOPIC_CATALOGUE.items():
        if stem not in present_stems:
            manifest.append({
                "filename":   f"{stem}.txt",
                "stem":       stem,
                "topic":      topic_label,
                "category":   category,
                "size_bytes": 0,
                "exists":     False,
            })

    return manifest
