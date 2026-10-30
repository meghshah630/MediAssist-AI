"""
tests/test_voice.py – Unit and integration tests for speech-to-text and text-to-speech features.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
import pytest

from voice.stt import (
    transcribe_audio_bytes,
    record_and_transcribe,
    is_microphone_available,
    MicrophoneNotAvailableError,
    TranscriptionError,
)
from voice.tts import (
    clean_text_for_speech,
    text_to_audio_bytes,
    speak,
    stop_speech,
)


# ---------------------------------------------------------------------------
# Section 1: clean_text_for_speech tests
# ---------------------------------------------------------------------------


class TestCleanTextForSpeech:
    """Tests for text normalization and disclaimer/sources stripping prior to speech."""

    def test_empty_string_returns_fallback(self):
        result = clean_text_for_speech("")
        assert result == "No spoken response available."

    def test_none_returns_fallback(self):
        result = clean_text_for_speech(None)
        assert result == "No spoken response available."

    def test_strips_disclaimer_block(self):
        raw = (
            "A fever is a body temperature of 38 degrees Celsius or higher.\n\n"
            "---\n\n"
            "⚠️ DISCLAIMER: MediAssist AI is for educational purposes only. "
            "It is NOT a substitute for professional medical advice."
        )
        cleaned = clean_text_for_speech(raw)
        assert "DISCLAIMER" not in cleaned
        assert "educational purposes" not in cleaned
        assert "38 degrees Celsius" in cleaned

    def test_strips_sources_block(self):
        raw = (
            "Hypertension is treated with ACE inhibitors and low-sodium diet.\n\n"
            "📖 Sources (2)\n"
            "1. hypertension.txt\n"
            "2. cardiovascular_disease.txt"
        )
        cleaned = clean_text_for_speech(raw)
        assert "📖 Sources" not in cleaned
        assert "hypertension.txt" not in cleaned
        assert "ACE inhibitors" in cleaned

    def test_strips_markdown_formatting(self):
        raw = (
            "### **Fever Overview**\n\n"
            "Common symptoms include:\n"
            "- *Chills* and shivering\n"
            "- **Muscle aches**\n"
            "- `Paracetamol` can be used\n\n"
            "[NHS Guidance](https://nhs.uk)"
        )
        cleaned = clean_text_for_speech(raw)
        assert "###" not in cleaned
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "`" not in cleaned
        assert "https://nhs.uk" not in cleaned
        assert "NHS Guidance" in cleaned
        assert "Chills and shivering" in cleaned
        assert "Paracetamol can be used" in cleaned

    def test_strips_emojis(self):
        raw = "🩺 Symptoms: 🤒 Fever and 💊 Medications like paracetamol."
        cleaned = clean_text_for_speech(raw)
        assert "🩺" not in cleaned
        assert "🤒" not in cleaned
        assert "💊" not in cleaned
        assert "Fever and" in cleaned
        assert "paracetamol" in cleaned


# ---------------------------------------------------------------------------
# Section 2: text_to_audio_bytes tests
# ---------------------------------------------------------------------------


class TestTextToAudioBytes:
    """Tests for TTS synthesis and online/offline fallback."""

    def test_text_to_audio_bytes_generates_valid_payload(self):
        """Generates audio bytes and returns appropriate mime type."""
        text = "Normal resting heart rate is between 60 and 100 beats per minute."
        audio_bytes, mime_type = text_to_audio_bytes(text, clean=True)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        assert mime_type in ["audio/mp3", "audio/wav"]

    @patch("gtts.gTTS")
    def test_offline_fallback_to_pyttsx3_when_gtts_fails(self, mock_gtts):
        """When gTTS raises an exception (e.g. offline), falls back to pyttsx3."""
        mock_gtts.side_effect = RuntimeError("Network connection unreachable")
        audio_bytes, mime_type = text_to_audio_bytes("Testing offline fallback voice synthesis.")
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        assert mime_type == "audio/wav"

    def test_clean_flag_strips_disclaimer(self):
        raw = (
            "Diabetes mellitus requires monitoring blood glucose levels.\n\n"
            "---\n\n⚠️ DISCLAIMER: Educational purposes only."
        )
        audio_bytes, _ = text_to_audio_bytes(raw, clean=True)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

    def test_stop_speech_does_not_crash(self):
        """Calling stop_speech() should safely succeed without exceptions."""
        stop_speech()


# ---------------------------------------------------------------------------
# Section 3: Speech-to-text (STT) tests
# ---------------------------------------------------------------------------


class TestSpeechToText:
    """Tests for speech recognition, audio byte transcription, and error handling."""

    def test_transcribe_empty_bytes_raises_transcription_error(self):
        """Empty audio payload should raise TranscriptionError with clear message."""
        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_audio_bytes(b"")
        assert "No audio data received" in str(exc_info.value)

    @patch("speech_recognition.Recognizer")
    @patch("speech_recognition.AudioFile")
    def test_transcribe_audio_bytes_success(self, mock_audio_file, mock_recognizer_cls):
        """Successful audio recognition returns transcribed string."""
        mock_recognizer = MagicMock()
        mock_recognizer.record.return_value = MagicMock(frame_data=b"fake_audio_frames")
        mock_recognizer.recognize_google.return_value = "what are the symptoms of asthma"
        mock_recognizer_cls.return_value = mock_recognizer

        result = transcribe_audio_bytes(b"RIFF....WAVE....")
        assert result == "what are the symptoms of asthma"
        mock_recognizer.recognize_google.assert_called_once()

    @patch("speech_recognition.Recognizer")
    @patch("speech_recognition.AudioFile")
    def test_transcribe_audio_unknown_value_error(self, mock_audio_file, mock_recognizer_cls):
        """Unintelligible speech raises user-friendly TranscriptionError."""
        import speech_recognition as sr
        mock_recognizer = MagicMock()
        mock_recognizer.record.return_value = MagicMock(frame_data=b"fake_audio_frames")
        mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()
        mock_recognizer_cls.return_value = mock_recognizer

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_audio_bytes(b"audio_bytes")
        assert "Could not understand the audio" in str(exc_info.value)

    @patch("speech_recognition.Recognizer")
    @patch("speech_recognition.AudioFile")
    def test_transcribe_audio_request_error(self, mock_audio_file, mock_recognizer_cls):
        """Recognition service error raises helpful message."""
        import speech_recognition as sr
        mock_recognizer = MagicMock()
        mock_recognizer.record.return_value = MagicMock(frame_data=b"fake_audio_frames")
        mock_recognizer.recognize_google.side_effect = sr.RequestError("API rate limit exceeded")
        mock_recognizer_cls.return_value = mock_recognizer

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_audio_bytes(b"audio_bytes")
        assert "Speech recognition service error" in str(exc_info.value)

    @patch("speech_recognition.Microphone")
    def test_record_and_transcribe_pyaudio_missing(self, mock_mic):
        """Missing PyAudio raises MicrophoneNotAvailableError with instructions."""
        mock_mic.side_effect = AttributeError("Could not find PyAudio; check installation")
        with pytest.raises(MicrophoneNotAvailableError) as exc_info:
            record_and_transcribe()
        assert "PyAudio" in str(exc_info.value) or "browser" in str(exc_info.value)

    @patch("speech_recognition.Microphone")
    def test_record_and_transcribe_permission_denied(self, mock_mic):
        """Microphone permission denied raises MicrophoneNotAvailableError."""
        mock_mic.side_effect = OSError("Access is denied")
        with pytest.raises(MicrophoneNotAvailableError) as exc_info:
            record_and_transcribe()
        assert "denied" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()

    def test_is_microphone_available_returns_bool_safely(self):
        """is_microphone_available() returns a bool without crashing."""
        result = is_microphone_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Section 4: Real-time Web Speech Read-Along component tests
# ---------------------------------------------------------------------------


class TestReadAlongComponent:
    """Tests for browser-native real-time Web Speech read-along parser and component generator."""

    def test_extract_spoken_content_strips_disclaimers_and_sources(self):
        from voice.read_along import extract_spoken_content

        raw = (
            "### Fever Management\n\n"
            "Rest and hydration are key.\n\n"
            "📖 Sources (2)\n"
            "1. fever.txt\n\n"
            "---\n\n"
            "⚠️ DISCLAIMER: Educational purposes only."
        )
        cleaned = extract_spoken_content(raw)
        assert "DISCLAIMER" not in cleaned
        assert "Sources" not in cleaned
        assert "Fever Management" in cleaned
        assert "Rest and hydration" in cleaned

    def test_prepare_read_along_data_preserves_headings_and_lists(self):
        from voice.read_along import prepare_read_along_data

        raw = (
            "### Symptoms\n"
            "- High temperature\n"
            "- Shivering\n\n"
            "Treat with paracetamol."
        )
        plain, html_body, word_count = prepare_read_along_data(raw)
        assert "Symptoms" in plain
        assert "High temperature" in plain
        assert "<h5" in html_body  # Heading preserved
        assert "<ul" in html_body  # List preserved
        assert "<li" in html_body
        assert "<p" in html_body   # Paragraph preserved
        assert word_count > 0

    def test_word_offsets_match_plain_speech_text(self):
        """Verify that every word span's data-start and data-end map exactly to plain text."""
        import re
        from voice.read_along import prepare_read_along_data

        raw = (
            "### **Diabetes Care**\n\n"
            "Type 2 diabetes involves insulin resistance.\n\n"
            "Key recommendations:\n"
            "- Regular physical activity\n"
            "- Balanced low-glycemic nutrition\n"
        )
        plain, html_body, _ = prepare_read_along_data(raw)

        # Extract all span tags with start and end
        pattern = re.compile(r'<span class="speech-word"[^>]*data-start="(\d+)" data-end="(\d+)">([^<]+)</span>')
        matches = pattern.findall(html_body)
        assert len(matches) > 0

        for start_str, end_str, word_text in matches:
            start = int(start_str)
            end = int(end_str)
            slice_text = plain[start:end]
            assert slice_text == word_text, f"Offset mismatch: plain[{start}:{end}]='{slice_text}' != '{word_text}'"

    def test_long_answer_processing(self):
        """Test read-along generation with a long multi-paragraph medical answer."""
        from voice.read_along import prepare_read_along_data, generate_read_along_component_html

        long_answer = (
            "### Hypertension and Cardiovascular Health\n\n"
            "Hypertension, commonly known as high blood pressure, is a major risk factor "
            "for coronary artery disease, stroke, and chronic kidney disease.\n\n"
            "Recommended lifestyle interventions:\n"
            "- Reduce dietary sodium to under 2000 milligrams daily\n"
            "- Engage in at least 150 minutes of moderate aerobic exercise per week\n"
            "- Maintain optimal body mass index between 18.5 and 24.9\n"
            "- Limit alcohol consumption and avoid smoking\n\n"
            "Pharmacological treatments include ACE inhibitors, Angiotensin Receptor Blockers, "
            "and Calcium Channel Blockers.\n\n"
            "📖 Sources (3)\n1. hypertension.txt\n\n"
            "---\n\n⚠️ DISCLAIMER: Educational purposes only."
        )

        plain, html_body, word_count = prepare_read_along_data(long_answer)
        assert word_count >= 50
        assert "DISCLAIMER" not in plain
        assert "Sources" not in plain

        html_component = generate_read_along_component_html(long_answer, "test_long")
        assert "speechSynthesis" in html_component
        assert "SpeechSynthesisUtterance" in html_component
        assert "btn-play" in html_component
        assert "btn-pause" in html_component
        assert "btn-resume" in html_component
        assert "btn-stop" in html_component
        assert "voice-select" in html_component
        assert "rate-select" in html_component
        assert "speaking-active" in html_component

    def test_empty_answer_fallback(self):
        from voice.read_along import prepare_read_along_data

        plain, html_body, count = prepare_read_along_data("")
        assert "No spoken answer available" in plain
        assert count > 0

    def test_setext_headings_and_divider_lines_omitted_from_speech(self):
        """
        Verify that markdown Setext header underlines (e.g. '====', '----')
        and divider lines are NOT spoken aloud as words.
        """
        from voice.read_along import prepare_read_along_data

        raw = (
            "Fever\n"
            "================\n\n"
            "Fever is a temporary rise in body temperature.\n\n"
            "Definition and Temperature Thresholds\n"
            "----------------------------------------\n"
            "Normal body temperature ranges from 36.1 to 37.2 degrees Celsius."
        )
        plain, html_body, count = prepare_read_along_data(raw)

        # Neither '====' nor '----' should be in the plain speech text
        assert "====" not in plain
        assert "----" not in plain
        assert "=" not in plain

        # The header text itself must be preserved
        assert "Fever" in plain
        assert "Definition and Temperature Thresholds" in plain

        # Check that Fever is structured as an h4 heading in the HTML
        assert "<h4 class='read-along-h4'>" in html_body
        assert "Fever</span></h4>" in html_body or "Fever" in html_body

        # And Definition is structured as a heading
        assert "<h5 class='read-along-h5'>" in html_body

