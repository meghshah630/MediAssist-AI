"""
core/summarizer.py – Optional text summarization helper.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def summarize_chunks(
    chunks: List[str],
    llm_client=None,
    model: Optional[str] = None,
    max_length: int = 300,
) -> str:
    """
    Produce a concise summary of a list of text chunks.

    Parameters
    ----------
    chunks:
        List of text chunks to summarise.
    llm_client:
        Optional OllamaClient instance. If None, uses extractive summarisation.
    model:
        Model name override for the LLM.
    max_length:
        Approximate maximum word count for the summary.

    Returns
    -------
    A summary string.
    """
    if not chunks:
        return ""

    combined = "\n\n".join(chunks)

    # ── LLM-based summarisation ─────────────────────────────────────────────
    if llm_client is not None:
        prompt = (
            f"Summarise the following medical text in at most {max_length} words. "
            "Focus on key medical facts, treatments, and important warnings.\n\n"
            f"TEXT:\n{combined}\n\nSUMMARY:"
        )
        try:
            return llm_client.generate(prompt=prompt, model=model, context=combined)
        except Exception as exc:
            logger.warning("LLM summarisation failed, falling back to extractive: %s", exc)

    # ── Extractive fallback ─────────────────────────────────────────────────
    return _extractive_summary(combined, max_words=max_length)


def _extractive_summary(text: str, max_words: int = 300) -> str:
    """
    Simple extractive summariser: pick sentences up to *max_words* words.

    Parameters
    ----------
    text:
        Input text.
    max_words:
        Maximum number of words to include.

    Returns
    -------
    Concatenated top sentences.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected: List[str] = []
    word_count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if word_count + words > max_words:
            break
        selected.append(sentence)
        word_count += words

    summary = " ".join(selected)
    logger.debug("Extractive summary: %d words.", word_count)
    return summary

