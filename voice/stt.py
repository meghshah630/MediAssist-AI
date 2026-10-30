"""
voice/stt.py – Speech-to-text using SpeechRecognition with browser and hardware audio support.
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class MicrophoneNotAvailableError(Exception):
    """Raised when no microphone input device is found or permission is denied."""


class TranscriptionError(Exception):
    """Raised when audio could not be understood or transcribed."""


def transcribe_audio_bytes(
    audio_data: Union[bytes, io.BytesIO],
    lang: str = "en-US",
) -> str:
    """
    Transcribe audio bytes (such as from Streamlit's st.audio_input or an uploaded WAV)
    into text using the Google Web Speech API.

    Parameters
    ----------
    audio_data:
        Raw audio bytes or a BytesIO buffer containing WAV/PCM audio.
    lang:
        BCP-47 language tag (default 'en-US').

    Returns
    -------
    Transcribed text string.

    Raises
    ------
    TranscriptionError:
        If the audio is empty, cannot be understood, or the recognition service is unavailable.
    """
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise ImportError(
            "SpeechRecognition library is not installed. Run: pip install SpeechRecognition"
        ) from exc

    if isinstance(audio_data, bytes):
        if not audio_data:
            raise TranscriptionError("No audio data received. Please record your speech and try again.")
        audio_stream = io.BytesIO(audio_data)
    else:
        audio_stream = audio_data

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_stream) as source:
            recorded_audio = recognizer.record(source)
    except Exception as exc:
        logger.error("Failed to read audio data into Recognizer: %s", exc)
        raise TranscriptionError(f"Could not process audio data: {exc}") from exc

    if not recorded_audio or not recorded_audio.frame_data:
        raise TranscriptionError("Recorded audio was silent or empty. Please speak clearly into your microphone.")

    try:
        text = recognizer.recognize_google(recorded_audio, language=lang)
        logger.info("Transcribed audio bytes: '%s'", text)
        return text.strip()
    except sr.UnknownValueError as exc:
        logger.info("SpeechRecognition could not understand audio.")
        raise TranscriptionError(
            "Could not understand the audio. Please speak clearly and try again."
        ) from exc
    except sr.RequestError as exc:
        logger.error("Speech recognition service request failed: %s", exc)
        raise TranscriptionError(
            f"Speech recognition service error: {exc}. Please check your network connection."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected transcription error: %s", exc)
        raise TranscriptionError(f"Transcription failed: {exc}") from exc


def record_and_transcribe(
    timeout: int = 5,
    phrase_time_limit: int = 15,
    lang: str = "en-US",
) -> str:
    """
    Record audio from the default local hardware microphone and return the transcription.

    Parameters
    ----------
    timeout:
        Seconds to wait for speech to begin before timing out.
    phrase_time_limit:
        Maximum duration (seconds) to record once speech begins.
    lang:
        Language code for recognition (default 'en-US').

    Returns
    -------
    Transcribed text string.

    Raises
    ------
    MicrophoneNotAvailableError:
        If no working microphone is detected or access is denied.
    TranscriptionError:
        If speech timed out, was not understood, or transcription failed.
    """
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise ImportError(
            "SpeechRecognition library is not installed. Run: pip install SpeechRecognition"
        ) from exc

    recognizer = sr.Recognizer()

    try:
        mic = sr.Microphone()
    except (AttributeError, ImportError) as exc:
        # PyAudio not installed or microphone interface missing
        raise MicrophoneNotAvailableError(
            "Hardware microphone access requires PyAudio. "
            "Please use the in-browser recording button or install PyAudio."
        ) from exc
    except OSError as exc:
        raise MicrophoneNotAvailableError(
            f"Microphone access denied or device unavailable: {exc}. "
            "Please check Windows microphone privacy settings."
        ) from exc
    except Exception as exc:
        raise MicrophoneNotAvailableError(f"Microphone initialization failed: {exc}") from exc

    try:
        with mic as source:
            logger.info("Adjusting for ambient noise (1s)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Listening for speech (timeout=%ds)...", timeout)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
    except sr.WaitTimeoutError as exc:
        raise TranscriptionError(
            "No speech detected within the timeout period. Please click the mic and start speaking."
        ) from exc
    except (AttributeError, OSError) as exc:
        raise MicrophoneNotAvailableError(
            f"Microphone recording error: {exc}. Verify your microphone is plugged in and permitted."
        ) from exc
    except Exception as exc:
        raise MicrophoneNotAvailableError(f"Unexpected microphone error: {exc}") from exc

    try:
        text = recognizer.recognize_google(audio, language=lang)
        logger.info("Hardware mic transcribed: '%s'", text)
        return text.strip()
    except sr.UnknownValueError as exc:
        raise TranscriptionError(
            "Could not understand the audio. Please speak clearly and try again."
        ) from exc
    except sr.RequestError as exc:
        raise TranscriptionError(f"Transcription service error: {exc}") from exc
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc


def is_microphone_available() -> bool:
    """
    Check whether a hardware microphone input device is available on the system.

    Returns
    -------
    True if at least one microphone is detected, False otherwise.
    """
    try:
        import speech_recognition as sr
        mics = sr.Microphone.list_microphone_names()
        return len(mics) > 0
    except Exception as exc:
        logger.debug("Microphone availability check failed (e.g. PyAudio missing): %s", exc)
        return False
