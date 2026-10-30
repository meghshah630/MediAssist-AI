"""
ui/voice_controls.py – Voice input (speech-to-text) and voice output (text-to-speech) UI controls.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_voice_input_controls(key: str = "voice_input_controls") -> Optional[str]:
    """
    Render voice input controls allowing users to record and transcribe medical questions.

    Uses Streamlit's native browser microphone recording (st.audio_input) with graceful
    fallback to direct hardware recording (record_and_transcribe). Handles permissions,
    missing devices, and speech errors with informative messages.

    Parameters
    ----------
    key:
        Unique key prefix for Streamlit widgets.

    Returns
    -------
    Transcribed query string if speech was successfully recorded and converted, else None.
    """
    from voice.stt import transcribe_audio_bytes, record_and_transcribe, MicrophoneNotAvailableError, TranscriptionError

    # We use a compact popover or expander for clear microphone controls
    with st.popover("Voice Input", help="Speak your medical question"):
        st.markdown("##### Voice Question Input")
        st.caption("Click the record button, speak your medical question, and it will be converted to text.")

        transcribed_text: Optional[str] = None

        # Method 1: Browser-native recording via st.audio_input (Streamlit 1.40+)
        if hasattr(st, "audio_input"):
            audio_data = st.audio_input("Record via browser microphone:", key=f"{key}_browser_mic")
            if audio_data is not None:
                # Check if this audio has already been transcribed in this session
                last_transcribed_key = f"{key}_last_audio_id"
                current_audio_bytes = audio_data.getvalue()

                if st.session_state.get(last_transcribed_key) != hash(current_audio_bytes):
                    try:
                        with st.spinner("Transcribing your question…"):
                            transcribed_text = transcribe_audio_bytes(current_audio_bytes)

                        st.session_state[last_transcribed_key] = hash(current_audio_bytes)
                        st.success(f"Heard: *{transcribed_text}*")
                        logger.info("Browser audio transcribed successfully: '%s'", transcribed_text)
                        return transcribed_text
                    except TranscriptionError as exc:
                        st.warning(str(exc))
                    except Exception as exc:
                        st.error(f"Voice recognition error: {exc}")
                        logger.error("Browser audio transcription error: %s", exc)

        # Method 2: Hardware direct recording fallback
        st.markdown("---")
        if st.button("Direct Microphone Capture", key=f"{key}_hw_btn", help="Record directly from local system mic"):
            try:
                with st.spinner("Listening… speak now (5s timeout)…"):
                    transcribed_text = record_and_transcribe(timeout=5, phrase_time_limit=15)

                st.success(f"Heard: *{transcribed_text}*")
                logger.info("Direct mic transcribed successfully: '%s'", transcribed_text)
                return transcribed_text
            except MicrophoneNotAvailableError as exc:
                st.warning(
                    f"**Microphone not detected or permission denied:**\n\n{exc}\n\n"
                    "Please ensure your microphone is plugged in and permissions are granted in browser/OS settings."
                )
            except TranscriptionError as exc:
                st.warning(f"**Could not transcribe speech:** {exc}")
            except Exception as exc:
                st.error(f"Voice input error: {exc}")
                logger.error("Direct microphone error: %s", exc)

    return None


def render_read_along_player(
    msg_idx: int,
    text: str,
    key_prefix: str = "read_along",
) -> None:
    """
    Render a real-time, browser-native Web Speech API (speechSynthesis) read-along player.

    Features:
      - Reads answer aloud using a natural English voice
      - Play, Pause, Resume, and Stop controls
      - Real-time word-by-word synchronized highlighting with smooth auto-scrolling
      - Keeps answer visible and readable, preserving headings, lists, and paragraphs
      - Skips source metadata and repeated medical disclaimers during speech
      - Dynamic English voice selector dropdown and speed control (0.85x to 1.4x)
      - Automatically stops previous speech when another answer starts playing
      - Handles unsupported browsers with an informative message
      - Zero audio file generation or download

    Parameters
    ----------
    msg_idx:
        Index of the message in the conversation.
    text:
        Full markdown text of the chatbot response.
    key_prefix:
        Prefix for component ID.
    """
    import streamlit.components.v1 as components
    from voice.read_along import generate_read_along_component_html

    component_html = generate_read_along_component_html(text, component_id=f"{key_prefix}_{msg_idx}")
    components.html(component_html, height=220, scrolling=False)


def render_message_tts_controls(
    msg_idx: int,
    text: str,
    key_prefix: str = "msg_tts",
) -> None:
    """Render read-along player (replaces old generated-audio-file player)."""
    render_read_along_player(msg_idx=msg_idx, text=text, key_prefix=key_prefix)


# ── Backward-compatible aliases ───────────────────────────────────────────────


def render_voice_input_button(key: str = "voice_input_btn") -> Optional[str]:
    """Backward-compatible alias for render_voice_input_controls."""
    return render_voice_input_controls(key=key)


def render_tts_button(
    text: str,
    button_label: str = "🔊",
    key: str = "tts_btn",
) -> None:
    """Backward-compatible wrapper rendering browser-native read-along player."""
    try:
        idx = abs(hash(key)) % 100000
    except Exception:
        idx = 0
    render_read_along_player(msg_idx=idx, text=text, key_prefix=key)

