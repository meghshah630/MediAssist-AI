"""
tests/test_llm_client.py – Tests for core.llm_client with mocked HTTP requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def client():
    """Return an OllamaClient pointing at a dummy URL."""
    from core.llm_client import OllamaClient
    return OllamaClient(base_url="http://localhost:11434")


class TestHealthCheck:
    """Tests for OllamaClient.health_check()."""

    def test_health_check_returns_true_on_200(self, client):
        """health_check should return True when /api/tags returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("core.llm_client.requests.get", return_value=mock_response) as mock_get:
            result = client.health_check()
        assert result is True

    def test_health_check_returns_false_on_non_200(self, client):
        """health_check should return False when /api/tags returns non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("core.llm_client.requests.get", return_value=mock_response):
            result = client.health_check()
        assert result is False

    def test_health_check_returns_false_on_connection_error(self, client):
        """health_check should return False on connection error."""
        import requests as req
        with patch("core.llm_client.requests.get", side_effect=req.ConnectionError("refused")):
            result = client.health_check()
        assert result is False

    def test_health_check_returns_false_on_timeout(self, client):
        """health_check should return False on timeout."""
        import requests as req
        with patch("core.llm_client.requests.get", side_effect=req.Timeout("timeout")):
            result = client.health_check()
        assert result is False


class TestListModels:
    """Tests for OllamaClient.list_models()."""

    def test_list_models_returns_names(self, client):
        """list_models should parse and return model names."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "phi3"},
                {"name": "mistral"},
            ]
        }
        with patch("core.llm_client.requests.get", return_value=mock_response):
            result = client.list_models()
        assert "llama3.2" in result
        assert "phi3" in result
        assert "mistral" in result

    def test_list_models_returns_empty_on_error(self, client):
        """list_models should return empty list when Ollama is unavailable."""
        import requests as req
        with patch("core.llm_client.requests.get", side_effect=req.ConnectionError("refused")):
            result = client.list_models()
        assert result == []

    def test_list_models_returns_empty_on_non_200(self, client):
        """list_models should return empty list on non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch("core.llm_client.requests.get", return_value=mock_response):
            result = client.list_models()
        assert result == []


class TestGenerate:
    """Tests for OllamaClient.generate()."""

    def test_generate_returns_string_on_success(self, client):
        """generate should return the LLM response string on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Hypertension is treated with ACE inhibitors and lifestyle changes.",
            "done": True,
        }
        with patch("core.llm_client.requests.post", return_value=mock_response):
            result = client.generate(prompt="What is hypertension?", model="llama3.2")
        assert isinstance(result, str)
        assert "ACE inhibitors" in result

    def test_generate_fallback_on_connection_error(self, client):
        """generate should return fallback message on connection error."""
        import requests as req
        with patch("core.llm_client.requests.post", side_effect=req.ConnectionError("refused")):
            result = client.generate(
                prompt="What is diabetes?",
                context="Diabetes is a metabolic disorder.",
            )
        assert "⚠️" in result or "unavailable" in result.lower()

    def test_generate_fallback_on_timeout(self, client):
        """generate should return fallback message on timeout."""
        import requests as req
        with patch("core.llm_client.requests.post", side_effect=req.Timeout("timeout")):
            result = client.generate(prompt="Explain asthma.", context="Asthma context here.")
        assert "⚠️" in result or "unavailable" in result.lower()

    def test_generate_fallback_on_non_200(self, client):
        """generate should return fallback on non-200 HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        with patch("core.llm_client.requests.post", return_value=mock_response):
            result = client.generate(prompt="What is COVID-19?", context="COVID-19 context.")
        assert "⚠️" in result or "unavailable" in result.lower()

    def test_generate_with_system_prompt(self, client):
        """generate should pass system message in the payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer here.", "done": True}
        with patch("core.llm_client.requests.post", return_value=mock_response) as mock_post:
            client.generate(
                prompt="Question?",
                model="llama3.2",
                system="You are a medical assistant.",
            )
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("system") == "You are a medical assistant."

    def test_generate_fallback_contains_context(self, client):
        """Fallback message when Ollama is unavailable should inform the user clearly."""
        import requests as req
        context = "Hypertension is a serious chronic condition affecting blood pressure."
        with patch("core.llm_client.requests.post", side_effect=req.ConnectionError("refused")):
            result = client.generate(prompt="Question?", context=context)
        assert "⚠️" in result
        assert "offline" in result.lower() or "unreachable" in result.lower()

    def test_generate_uses_default_model(self, client):
        """generate should use DEFAULT_MODEL when no model is specified."""
        from config import DEFAULT_MODEL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response.", "done": True}
        with patch("core.llm_client.requests.post", return_value=mock_response) as mock_post:
            client.generate(prompt="Test question?")
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("model") == DEFAULT_MODEL

    def test_generate_stream_is_false(self, client):
        """generate payload should always set stream=False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer.", "done": True}
        with patch("core.llm_client.requests.post", return_value=mock_response) as mock_post:
            client.generate(prompt="Test?")
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("stream") is False


class TestFallback:
    """Tests for OllamaClient._fallback static method."""

    def test_fallback_with_context(self):
        """_fallback should return a clear offline message with instructions."""
        from core.llm_client import OllamaClient
        context = "Some retrieved medical information."
        result = OllamaClient._fallback(context)
        assert "⚠️" in result
        assert "offline" in result.lower() or "unreachable" in result.lower()

    def test_fallback_without_context(self):
        """_fallback with empty context should still return a meaningful message."""
        from core.llm_client import OllamaClient
        result = OllamaClient._fallback("")
        # Should mention Ollama being offline and give guidance
        assert "offline" in result.lower() or "unavailable" in result.lower()
        assert len(result) > 20

    def test_fallback_truncates_long_context(self):
        """_fallback with long context should produce a reasonable-length output."""
        from core.llm_client import OllamaClient
        long_context = "word " * 500  # ~2500 chars
        result = OllamaClient._fallback(long_context)
        # Result should not balloon to many times the context size
        assert len(result) < len(long_context) * 3


class TestModelMissingMessage:
    """Tests for OllamaClient._model_missing_message() static method."""

    def test_model_missing_contains_model_name(self):
        """_model_missing_message should mention the model name."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._model_missing_message("llama3.2")
        assert "llama3.2" in msg

    def test_model_missing_contains_pull_command(self):
        """_model_missing_message should include the pull command."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._model_missing_message("llama3.2")
        assert "pull" in msg.lower()

    def test_model_missing_starts_with_warning_emoji(self):
        """_model_missing_message should start with the warning marker."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._model_missing_message("phi3")
        assert msg.startswith("⚠️")

    def test_model_missing_different_models(self):
        """_model_missing_message should work with various model names."""
        from core.llm_client import OllamaClient
        for model in ["phi3", "mistral", "llama2"]:
            msg = OllamaClient._model_missing_message(model)
            assert model in msg
            assert "pull" in msg.lower()


class TestOfflineMessage:
    """Tests for OllamaClient._offline_message() static method."""

    def test_offline_message_mentions_port(self):
        """_offline_message should mention the default Ollama port."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._offline_message()
        assert "11434" in msg

    def test_offline_message_mentions_serve(self):
        """_offline_message should mention 'ollama serve' command."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._offline_message()
        assert "ollama serve" in msg or "serve" in msg.lower()

    def test_offline_message_starts_with_warning(self):
        """_offline_message should start with the warning emoji."""
        from core.llm_client import OllamaClient
        msg = OllamaClient._offline_message()
        assert msg.startswith("⚠️")

    def test_offline_answer_alias_matches_offline_message(self):
        """_offline_answer legacy alias should return same content as _offline_message."""
        from core.llm_client import OllamaClient
        assert OllamaClient._offline_answer() == OllamaClient._offline_message()

    def test_generate_model_missing_http_404(self, client):
        """generate() with HTTP 404 should return the model-missing message."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("core.llm_client.requests.post", return_value=mock_response):
            result = client.generate(prompt="Tell me about diabetes.", model="nonexistent-model")
        assert "⚠️" in result
        assert "pull" in result.lower() or "not installed" in result.lower()

    def test_generate_empty_response_returns_warning(self, client):
        """generate() should warn when Ollama returns an empty response string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "", "done": True}
        with patch("core.llm_client.requests.post", return_value=mock_response):
            result = client.generate(prompt="Question?", model="llama3.2")
        assert "⚠️" in result or "empty" in result.lower()

