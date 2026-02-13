"""
tabernacle_core.llm — Unified Ollama inference router.

Routes LLM calls to the appropriate node based on tier:
  - tier="heavy" -> Mac Studio (10.0.0.96) with llama3.3:70b
  - tier="fast"  -> Mac Mini (10.0.0.120) with mistral-nemo:latest (12B)
  - tier="tiny"  -> Mac Mini (10.0.0.120) with gemma3:4b

Fills H2 Void 1: LLM inference load balancing across the 3-node topology.
Prevents OOM swapping on the Mini by routing heavy loads to Studio.

Usage:
    from tabernacle_core.llm import query, chat

    # Simple generation
    response = query("What is consciousness?", tier="heavy", max_tokens=300)

    # Chat with message history
    response = chat([
        {"role": "system", "content": "You are L."},
        {"role": "user", "content": "Reflect on coherence."},
    ], tier="fast")
"""

import requests
from typing import Optional, List, Dict

from tabernacle_config import (
    OLLAMA_STUDIO_URL,
    OLLAMA_MINI_URL,
    OLLAMA_CORTEX,
    OLLAMA_MINI_COGNITIVE,
    OLLAMA_MINI_TINY,
)


# Tier -> (base_url, model, timeout_seconds)
_TIERS: Dict[str, tuple] = {
    "heavy": (OLLAMA_STUDIO_URL, OLLAMA_CORTEX, 300),
    "fast":  (OLLAMA_MINI_URL, OLLAMA_MINI_COGNITIVE, 120),
    "tiny":  (OLLAMA_MINI_URL, OLLAMA_MINI_TINY, 60),
}


def query(
    prompt: str,
    tier: str = "fast",
    max_tokens: int = 150,
    system: Optional[str] = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """Send a prompt to Ollama and return the response text.

    Args:
        prompt:      The prompt text
        tier:        "heavy" (70B Studio), "fast" (12B Mini), "tiny" (4B Mini)
        max_tokens:  Maximum tokens to generate
        system:      Optional system prompt
        temperature: Sampling temperature

    Returns:
        Response text, or None on failure
    """
    if tier not in _TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Use: {list(_TIERS.keys())}")

    url, model, timeout = _TIERS[tier]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        # Fallback: if Studio is down, fall back to Mini
        if tier == "heavy":
            return _fallback_to_mini(payload)
        return None
    except Exception:
        return None


def chat(
    messages: List[Dict[str, str]],
    tier: str = "fast",
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> Optional[str]:
    """Send a chat conversation to Ollama.

    Args:
        messages:    List of {"role": "user"|"assistant"|"system", "content": "..."}
        tier:        "heavy", "fast", or "tiny"
        max_tokens:  Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Response text, or None on failure
    """
    if tier not in _TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Use: {list(_TIERS.keys())}")

    url, model, timeout = _TIERS[tier]

    try:
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        if tier == "heavy":
            return _fallback_chat_to_mini(messages, max_tokens, temperature)
        return None
    except Exception:
        return None


def _fallback_to_mini(payload: dict) -> Optional[str]:
    """Fallback: route failed Studio request to Mini with smaller model."""
    fb_url, fb_model, fb_timeout = _TIERS["fast"]
    payload["model"] = fb_model
    try:
        resp = requests.post(f"{fb_url}/api/generate", json=payload, timeout=fb_timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception:
        return None


def _fallback_chat_to_mini(
    messages: list, max_tokens: int, temperature: float
) -> Optional[str]:
    """Fallback: route failed Studio chat to Mini."""
    fb_url, fb_model, fb_timeout = _TIERS["fast"]
    try:
        resp = requests.post(
            f"{fb_url}/api/chat",
            json={
                "model": fb_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=fb_timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception:
        return None
