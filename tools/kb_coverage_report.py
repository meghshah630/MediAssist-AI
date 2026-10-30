"""
tools/kb_coverage_report.py - Knowledge Base Coverage and Quality Report for MediAssist AI

Run from the project root:
    python tools/kb_coverage_report.py

Reports:
1. Document manifest (present vs missing topics)
2. ChromaDB chunk counts per source
3. Encoding/quality issues in documents
4. Source classification (verified vs unverified/synthetic)
5. Retrieval quality probe on representative questions
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CHROMA_DB_PATH, COLLECTION_NAME, SAMPLE_DOCS_PATH
from core.document_ingestion import get_knowledge_base_manifest, ingest_directory


# ---------------------------------------------------------------------------
# Source provenance registry
# Classify each document as VERIFIED (content written from authoritative
# published medical guidelines) or LOCALLY_AUTHORED (written in-house but
# not a direct transcription from a citable published source).
# No source is classified "downloaded from URL" unless an actual URL was
# fetched and verified at ingestion time.
# ---------------------------------------------------------------------------
_SOURCE_CLASSIFICATION = {
    # Locally authored from established clinical guidelines (NHS/CDC/WHO/ADA)
    "hypertension.txt":                "locally_authored_from_guidelines",
    "diabetes_type2.txt":              "locally_authored_from_guidelines",
    "asthma.txt":                      "locally_authored_from_guidelines",
    "covid19.txt":                      "locally_authored_from_guidelines",
    "cardiovascular_disease.txt":      "locally_authored_from_guidelines",
    "general_first_aid.txt":           "locally_authored_from_guidelines",
    "medication_interactions.txt":     "locally_authored_from_guidelines",
    "mental_health_anxiety.txt":       "locally_authored_from_guidelines",
    "mental_health_depression.txt":    "locally_authored_from_guidelines",
    "fever_and_common_infections.txt": "locally_authored_from_guidelines",
    "weight_management_and_obesity.txt": "locally_authored_from_guidelines",
    "nutrition_and_healthy_eating.txt": "locally_authored_from_guidelines",
    "anemia_and_blood_disorders.txt":  "locally_authored_from_guidelines",
    "sleep_health_and_disorders.txt":  "locally_authored_from_guidelines",
    "preventive_care_and_screening.txt": "locally_authored_from_guidelines",
    "respiratory_infections.txt":      "locally_authored_from_guidelines",
    "emergency_warning_signs.txt":     "locally_authored_from_guidelines",
    "hydration_and_water_intake.txt":  "locally_authored_from_guidelines",
}

_CLASSIFICATION_LABELS = {
    "locally_authored_from_guidelines": "Locally authored (based on published guidelines; not a direct URL download)",
    "verified_url_download":            "Verified URL download (not yet implemented)",
    "synthetic":                        "Synthetic / placeholder (do not use for clinical claims)",
    "unknown":                          "Unknown provenance",
}


def run_report():
    sep = "=" * 72

    print(sep)
    print("  MEDIASSIST AI -- KNOWLEDGE BASE COVERAGE REPORT")
    print(sep)
    print()

    # ── 1. Document manifest ──────────────────────────────────────────────────
    manifest = get_knowledge_base_manifest(SAMPLE_DOCS_PATH)
    present = [m for m in manifest if m["exists"]]
    missing = [m for m in manifest if not m["exists"]]

    print(f"SECTION 1: DOCUMENT MANIFEST")
    print(f"  Total documents present on disk : {len(present)}")
    print(f"  Catalogued topics missing        : {len(missing)}")
    print()
    print("  Present documents:")
    for m in present:
        size_kb = m["size_bytes"] // 1024
        cls = _SOURCE_CLASSIFICATION.get(m["filename"], "unknown")
        cls_short = cls.split("_")[0].upper()
        print(f"    [{cls_short:5}] {m['filename']:45s} ({size_kb:3d} KB) | {m['topic']}")

    if missing:
        print()
        print("  MISSING topics (not on disk):")
        for m in missing:
            print(f"    - {m['filename']:45s} | {m['topic']}")

    # ── 2. Source classification ───────────────────────────────────────────────
    print()
    print("SECTION 2: SOURCE CLASSIFICATION")
    cls_counts = {}
    for m in present:
        cls = _SOURCE_CLASSIFICATION.get(m["filename"], "unknown")
        cls_counts[cls] = cls_counts.get(cls, 0) + 1

    for cls, count in cls_counts.items():
        label = _CLASSIFICATION_LABELS.get(cls, cls)
        print(f"  {count:2d} document(s): {label}")

    print()
    print("  NOTE: 'Locally authored from guidelines' means content was written")
    print("  by the development team drawing on established clinical guidance")
    print("  (NHS, CDC, WHO, ADA, AHA). These are NOT direct URL downloads and")
    print("  have NOT been independently validated by a licensed clinician.")
    print("  This knowledge base is for EDUCATIONAL purposes only.")

    # ── 3. ChromaDB chunk audit ────────────────────────────────────────────────
    print()
    print("SECTION 3: CHROMADB CHUNK AUDIT")
    try:
        from core.vector_store import VectorStore
        vs = VectorStore(CHROMA_DB_PATH, COLLECTION_NAME)
        total_chunks = vs.collection_count()
        print(f"  Total chunks in ChromaDB: {total_chunks}")

        if total_chunks == 0:
            print("  WARNING: Knowledge base is EMPTY. Run 'Re-ingest Documents' in the app.")
        else:
            # Get per-source counts via batch get
            raw = vs._collection.get(limit=5000, include=["metadatas"])
            source_counts = {}
            topic_counts = {}
            for meta in raw["metadatas"]:
                src = meta.get("source", "unknown")
                topic = meta.get("topic", "General")
                source_counts[src] = source_counts.get(src, 0) + 1
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            print()
            print("  Chunks per source document:")
            for src in sorted(source_counts):
                print(f"    {src:50s}: {source_counts[src]:4d} chunks")

            print()
            print("  Chunks per topic category:")
            for topic in sorted(topic_counts):
                print(f"    {topic:50s}: {topic_counts[topic]:4d} chunks")

    except Exception as exc:
        print(f"  ERROR accessing ChromaDB: {exc}")
        print("  (ChromaDB may not be initialised yet -- run the app first)")

    # ── 4. Document quality check ──────────────────────────────────────────────
    print()
    print("SECTION 4: DOCUMENT QUALITY CHECKS")
    quality_issues = []
    for m in present:
        fpath = Path(SAMPLE_DOCS_PATH) / m["filename"]
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            issues = []
            # Check for encoding garbage
            glitch_count = sum(1 for c in content if ord(c) > 65000 or c == "\ufffd")
            if glitch_count > 5:
                issues.append(f"encoding artifacts ({glitch_count} chars)")
            # Check for very short documents (less than 200 words)
            word_count = len(content.split())
            if word_count < 200:
                issues.append(f"very short ({word_count} words)")
            # Check for placeholder text
            if any(p in content.lower() for p in ["lorem ipsum", "placeholder", "todo:", "fixme:"]):
                issues.append("contains placeholder text")

            if issues:
                quality_issues.append((m["filename"], issues))
        except Exception as exc:
            quality_issues.append((m["filename"], [f"read error: {exc}"]))

    if quality_issues:
        print("  Documents with quality issues:")
        for fname, issues in quality_issues:
            print(f"    {fname}: {', '.join(issues)}")
    else:
        print("  All present documents passed quality checks.")

    # ── 5. Coverage gap analysis ───────────────────────────────────────────────
    print()
    print("SECTION 5: TOPIC COVERAGE GAP ANALYSIS")
    categories_present = set(m["category"] for m in present)
    all_categories = {
        "cardiovascular", "metabolic", "respiratory", "infectious_disease",
        "emergency", "pharmacology", "mental_health", "lifestyle",
        "hematology", "preventive", "general"
    }
    missing_categories = all_categories - categories_present
    if missing_categories:
        print(f"  Missing medical categories: {', '.join(sorted(missing_categories))}")
    else:
        print(f"  All defined medical categories are represented.")

    print()
    print("  Important topics NOT yet in the knowledge base:")
    not_covered = [
        "Exercise and physical activity guidelines",
        "Pediatric health and childhood illnesses",
        "Women's health and reproductive medicine",
        "Geriatric health and aging",
        "Dermatology / common skin conditions",
        "Ophthalmology / eye conditions",
        "Kidney disease and renal health",
        "Liver disease and hepatology",
        "Thyroid and endocrine disorders",
        "Musculoskeletal disorders / arthritis / osteoporosis",
        "Neurological conditions (epilepsy, Parkinson's, dementia)",
        "Oncology / cancer overview",
        "HIV/AIDS and sexually transmitted infections",
        "Substance use and addiction",
        "Post-surgical and wound care",
    ]
    for topic in not_covered:
        print(f"    - {topic}")

    print()
    print("SECTION 6: RETRIEVAL QUALITY NOTE")
    print("  Retrieval quality depends on:")
    print("  a) Chunk size (current: 200 words -- smaller = more precise)")
    print("  b) Embedding model: all-MiniLM-L6-v2 (same for indexing and queries)")
    print("  c) Relevance threshold: 0.85 cosine distance (above = 'not in KB')")
    print("  d) Topic metadata on chunks (enables future category filtering)")
    print()
    print("  Run automated RAG evaluation tests for retrieval quality metrics:")
    print("    python -m pytest tests/test_rag_evaluation.py -v")

    print()
    print(sep)
    print("  END OF REPORT")
    print(sep)


if __name__ == "__main__":
    run_report()

