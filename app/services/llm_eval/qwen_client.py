"""

Thin HTTP client for the Ollama REST API.

Sends a prompt to Qwen-2.5:7b running locally via Ollama and returns
the raw text response.

Ollama must be running before any requests are made:
    ollama serve           (starts the daemon)
    ollama pull qwen2.5:7b (download once)

Endpoint: POST http://localhost:11434/api/generate

temperature=0
-------------
Set to 0 for fully deterministic output. With temperature=0 Ollama
uses greedy decoding — always picks the highest-probability token.
This means the same code + same question always produces the same
JSON scores. Critical for a fair evaluation system.

temperature=0.1 (old value) still had ~2-3 point variance between
runs. For a grading system that variance is unacceptable.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Configuration — all from environment variables

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))  # seconds


def check_health() -> bool:
    """
    Check if Ollama is reachable and the model is available.

    Returns True if healthy, False otherwise.
    Called at startup from main.py to fail fast if Ollama is not running.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check if our configured model is available
            # Ollama model names can be "qwen2.5:7b" or "qwen2.5:7b-instruct" etc.
            model_available = any(
                OLLAMA_MODEL.split(":")[0] in name
                for name in model_names
            )
            if not model_available:
                logger.warning(
                    "Ollama is running but model '%s' not found. "
                    "Available models: %s. "
                    "Run: ollama pull %s",
                    OLLAMA_MODEL, model_names, OLLAMA_MODEL,
                )
                return False
            logger.info(
                "Ollama health check passed — model=%s available", OLLAMA_MODEL
            )
            return True
    except Exception as exc:
        logger.warning(
            "Ollama health check failed — %s. "
            "LLM evaluation will fail until Ollama is running. "
            "Start with: ollama serve && ollama pull %s",
            exc, OLLAMA_MODEL,
        )
        return False


def generate(prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """
    Send a prompt to Ollama and return the raw text response.

    Args:
        prompt:  The full prompt string to send.
        timeout: Request timeout in seconds.

    Returns:
        Raw text string from the model.

    Raises:
        httpx.HTTPError:       If Ollama is unreachable or returns HTTP error.
        httpx.TimeoutException: If inference exceeds the timeout.
        RuntimeError:          If Ollama returns an empty or malformed response.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,     # fully deterministic — same input = same output
            "top_p": 1.0,         # with temperature=0, top_p has no effect
            "seed": 42,           # explicit seed for reproducibility
        },
    }

    logger.debug(
        "Sending prompt to Ollama — model=%s prompt_length=%d chars",
        OLLAMA_MODEL, len(prompt),
    )

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "").strip()

    if not raw_text:
        raise RuntimeError(
            f"Ollama returned an empty response for model={OLLAMA_MODEL}. "
            "Ensure the model is pulled and Ollama is running (`ollama serve`)."
        )

    logger.debug(
        "Ollama response received — length=%d chars", len(raw_text)
    )
    return raw_text