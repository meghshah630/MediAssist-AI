"""
voice/tts.py – Text-to-speech with clean speech normalization, gTTS, and pyttsx3 offline fallback.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def clean_text_for_speech(text: str) -> str:
    """
    Clean medical assistant responses for natural, fluent spoken text.

    Removes:
      - Medical disclaimers (e.g. '⚠️ DISCLAIMER: MediAssist AI is for educational...')
      - Source citation blocks (e.g. '📖 Sources (3)', '1. hypertension.txt...')
      - Markdown formatting (headers, bold, italics, links, inline code, code blocks)
      - Bullet indicators and numbering prefixes
      - Emojis and special decorative Unicode symbols
      - Excessive whitespace

    Parameters
    ----------
    text:
        Raw markdown response string.

    Returns
    -------
    Clean, natural-sounding prose suitable for text-to-speech.
    """
    if not text:
        return "No spoken response available."

    cleaned = text

    # 1. Strip disclaimers at the end of the text
    disclaimer_markers = [
        "---\n\n⚠️ DISCLAIMER",
        "---\n\nDISCLAIMER",
        "⚠️ DISCLAIMER",
        "⚠️ **DISCLAIMER",
        "DISCLAIMER:",
        "⚠️ Disclaimer",
    ]
    for marker in disclaimer_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    # 2. Strip sources blocks
    source_markers = [
        "📖 Sources",
        "### Sources",
        "## Sources",
        "**Sources:**",
        "Sources:",
        "[Source:",
    ]
    for marker in source_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    # 3. Remove code blocks and strip inline code backticks
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # 4. Remove Markdown links: [anchor text](http://...) -> anchor text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # 5. Remove Markdown headers (#, ##, ###, etc.)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # 6. Remove bold and italics formatting (**bold**, *italic*, __bold__, _italic_)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)

    # 7. Remove list bullet markers (- , * , • , 1. , etc.)
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)

    # 8. Remove horizontal rules, underlines, and divider rows (===, ---, ___, ***, ~~~)
    cleaned = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*={2,}\s*$", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*-{2,}\s*$", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-=_*~]{2,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[-=_*~]{2,}", " ", cleaned)

    # 9. Remove emojis and decorative symbols
    # Unicode emoji planes and symbols
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    cleaned = re.sub(r"[\u2600-\u27bf\u2300-\u23ff\u2b50\u200d\u200c]", "", cleaned)

    # 10. Clean whitespace and empty lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned).strip()

    return cleaned or "No spoken response available."


def text_to_audio_bytes(
    text: str,
    lang: str = "en",
    clean: bool = True,
) -> Tuple[bytes, str]:
    """
    Convert text to audio bytes suitable for Streamlit's st.audio() widget.

    Attempts gTTS (online, MP3) first; if unavailable or network fails,
    transparently falls back to pyttsx3 (offline SAPI5/eSpeak, WAV).

    Parameters
    ----------
    text:
        The response text to synthesize.
    lang:
        Language code (default 'en').
    clean:
        Whether to clean disclaimers, citations, and markdown before synthesis.

    Returns
    -------
    Tuple of (audio_bytes, mime_type) where mime_type is 'audio/mp3' or 'audio/wav'.

    Raises
    ------
    RuntimeError:
        If both gTTS and pyttsx3 synthesis fail.
    """
    speech_text = clean_text_for_speech(text) if clean else text

    # Strategy A: Try gTTS (Google Text-to-Speech)
    try:
        from gtts import gTTS
        tts = gTTS(text=speech_text, lang=lang, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
        if audio_bytes:
            logger.debug("gTTS generated %d MP3 bytes.", len(audio_bytes))
            return audio_bytes, "audio/mp3"
    except Exception as gtts_err:
        logger.warning("gTTS synthesis unavailable or failed (%s). Falling back to pyttsx3...", gtts_err)

    # Strategy B: Offline fallback via pyttsx3 (SAPI5 on Windows, eSpeak on Linux, NSSpeechSynthesizer on macOS)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 0.95)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            engine.save_to_file(speech_text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            if wav_bytes:
                logger.debug("pyttsx3 generated %d WAV bytes.", len(wav_bytes))
                return wav_bytes, "audio/wav"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    except Exception as pyttsx3_err:
        logger.error("pyttsx3 offline synthesis also failed: %s", pyttsx3_err)

    raise RuntimeError("Text-to-speech synthesis failed using both gTTS and pyttsx3 engines.")


def speak(text: str, engine: str = "pyttsx3", clean: bool = True) -> None:
    """
    Speak text directly aloud using the system audio device.

    Parameters
    ----------
    text:
        Text to speak.
    engine:
        'pyttsx3' (offline) or 'gtts'.
    clean:
        Whether to strip disclaimers and markdown formatting.
    """
    speech_text = clean_text_for_speech(text) if clean else text

    if engine == "pyttsx3":
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 165)
            tts_engine.setProperty("volume", 0.9)
            tts_engine.say(speech_text)
            tts_engine.runAndWait()
            tts_engine.stop()
        except Exception as exc:
            logger.error("pyttsx3 direct speak failed: %s", exc)
            raise RuntimeError(f"pyttsx3 speak failed: {exc}") from exc
    elif engine == "gtts":
        try:
            audio_bytes, _ = text_to_audio_bytes(speech_text, clean=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                _play_audio_file(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as exc:
            logger.error("gtts direct speak failed: %s", exc)
            raise RuntimeError(f"gTTS speak failed: {exc}") from exc
    else:
        raise ValueError(f"Unknown TTS engine: '{engine}'. Choose 'pyttsx3' or 'gtts'.")


def stop_speech() -> None:
    """Stop any currently active pyttsx3 speech playback."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
    except Exception as exc:
        logger.debug("stop_speech encountered exception (benign): %s", exc)


def _play_audio_file(path: str) -> None:
    """Play an audio file using system player."""
    import sys
    if sys.platform == "win32":
        try:
            os.startfile(path)
        except Exception as exc:
            logger.debug("os.startfile failed: %s", exc)
