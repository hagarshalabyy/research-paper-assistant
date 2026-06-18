"""Ollama connectivity helpers."""

import json
import urllib.error
import urllib.request

from app.config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> tuple[bool, list[str]]:
    """Return whether Ollama is reachable and installed model names."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False, []


def is_model_available(model: str, available: list[str]) -> bool:
    """Match Ollama tags like llama3.2:3b against requested model names."""
    if model in available:
        return True
    base = model.split(":")[0]
    return any(name == model or name.startswith(f"{base}:") or name.split(":")[0] == base for name in available)


def missing_models(base_url: str = OLLAMA_BASE_URL) -> tuple[bool, list[str]]:
    """Check Ollama is up and required models are pulled."""
    online, available = list_ollama_models(base_url)
    if not online:
        return False, [LLM_MODEL, EMBEDDING_MODEL]

    required = [LLM_MODEL, EMBEDDING_MODEL]
    return True, [m for m in required if not is_model_available(m, available)]
