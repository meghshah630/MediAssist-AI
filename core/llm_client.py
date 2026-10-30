"""
core/llm_client.py – Ollama REST API client with graceful error handling.

Error & Availability Handling Strategy:
----------------------------------------
1. **Ollama unreachable / offline** (connection refused or timeout):
   -> Return a clear error message informing the user that Ollama is offline,
      with setup/start instructions. NEVER dump unrelated document text as the answer.

2. **Model not installed / pulled** (HTTP 404 or missing from Ollama):
   -> Return a clear, actionable message stating that the selected model is missing
      and provide the exact command: `ollama pull <model>`.

3. **HTTP error / unexpected exception**:
   -> Return a clear error message describing the failure.

4. **Model available & responsive**:
   -> Send the prompt to Ollama's REST API and return the generated answer.

Raw chunks or unrelated document text are NEVER returned as an answer.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from config import OLLAMA_BASE_URL, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Timeout settings (seconds)
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 120


class OllamaClient:
    """Thin wrapper around the Ollama REST API with robust error handling."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        logger.info("OllamaClient initialised at %s", self.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the Ollama service is reachable and responding."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Unexpected error during health check: %s", exc)
            return False

    def list_models(self) -> List[str]:
        """
        Return locally-available model names.

        Returns an empty list when Ollama is unavailable or no models are pulled.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return [m for m in models if m]
        except (requests.ConnectionError, requests.Timeout):
            logger.debug("list_models: Ollama unreachable.")
            return []
        except Exception as exc:
            logger.warning("list_models error: %s", exc)
            return []

    def is_model_available(self, model: str) -> bool:
        """
        Return True if *model* is pulled and available on the local Ollama instance.

        Returns False when Ollama is offline or the model hasn't been pulled.
        """
        available = self.list_models()
        # Ollama model names can carry a tag, e.g. "llama3.2:latest"
        for m in available:
            if m == model or m.startswith(f"{model}:"):
                return True
        return False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        context: str = "",
        system: str = "",
        question: str = "",
    ) -> str:
        """
        Generate a completion from the LLM.

        Parameters
        ----------
        prompt:
            Full user prompt (context + question).
        model:
            Model name override (default: config.DEFAULT_MODEL).
        context:
            Retrieved medical text (for reference if needed).
        system:
            System message / instruction for the model.
        question:
            Bare user question.

        Returns
        -------
        LLM response string, or a clear user-facing error message when
        Ollama is offline or the model is missing.
        """
        model = model or DEFAULT_MODEL
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("response", "").strip()
                logger.debug("LLM response received (%d chars).", len(answer))
                if answer:
                    return answer
                return "⚠️ Ollama returned an empty response. Please try rephrasing your question."

            elif resp.status_code == 404:
                # Model not pulled yet — provide the exact pull command
                logger.warning("Model '%s' not found on Ollama (HTTP 404).", model)
                return self._model_missing_message(model)

            else:
                logger.warning(
                    "Ollama returned HTTP %d: %s", resp.status_code, resp.text[:200]
                )
                return (
                    f"⚠️ **Ollama Service Error (HTTP {resp.status_code})**\n\n"
                    f"Ollama responded with an error while generating with model `{model}`.\n"
                    f"Details: {resp.text[:200]}"
                )

        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("Ollama unavailable: %s", exc)
            return self._offline_message()
        except Exception as exc:
            logger.error("generate() unexpected error: %s", exc)
            return f"⚠️ **An unexpected error occurred while communicating with Ollama:**\n\n{exc}"

    # ------------------------------------------------------------------
    # Clear, actionable error messages
    # ------------------------------------------------------------------

    @staticmethod
    def _model_missing_message(model: str) -> str:
        """Return a clear error message when the requested model is not installed."""
        return (
            f"⚠️ **Model `{model}` is not installed in Ollama.**\n\n"
            f"Please run the following command in PowerShell:\n\n"
            f"```powershell\n"
            f'& "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" pull {model}\n'
            f"```\n\n"
            f"*(Or simply restart your terminal so standard `ollama pull {model}` is recognised)*\n\n"
            f"Once the download is complete, ask your question again."
        )

    @staticmethod
    def _offline_message() -> str:
        """Return a clear error message when Ollama is offline or unreachable."""
        return (
            "⚠️ **Ollama LLM is currently offline or unreachable at `http://localhost:11434`.**\n\n"
            "To enable medical question-answering with MediAssist AI, please ensure Ollama is running:\n\n"
            "1. **Start Ollama service** in your terminal:\n"
            "   ```bash\n"
            "   ollama serve\n"
            "   ```\n"
            "2. **Ensure your model is pulled**:\n"
            "   ```bash\n"
            "   ollama pull llama3.2\n"
            "   ```\n"
            "3. Refresh or retry your question once the service is running."
        )

    # Legacy aliases kept for backward compatibility with existing tests
    @staticmethod
    def _offline_answer(question: str = "", context: str = "") -> str:
        """Legacy helper now returning the clear offline message."""
        return OllamaClient._offline_message()

    @staticmethod
    def _model_missing_answer(model: str, question: str = "", context: str = "") -> str:
        """Legacy helper now returning the clear model missing message."""
        return OllamaClient._model_missing_message(model)

    @staticmethod
    def _fallback(context: str) -> str:
        """Legacy fallback helper."""
        return OllamaClient._offline_message()

