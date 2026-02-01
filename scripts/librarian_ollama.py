#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LIBRARIAN OLLAMA: Ollama interface functions for Librarian.

Extracted from librarian.py for modularity.
These functions handle Ollama availability checking and query execution.

Author: Cursor + Virgil
Created: 2026-01-28
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    LOG_DIR,
    LIBRARIAN_MODEL,
    OLLAMA_FALLBACK,
)

# =============================================================================
# RIE ENHANCEMENT (optional - graceful degradation)
# =============================================================================

try:
    from rie_ollama_bridge import RIEEnhancedModel
    _RIE_AVAILABLE = True
except ImportError:
    _RIE_AVAILABLE = False

# Module-level state
_ollama_checked = False
_ollama_available = False
_rie_model = None


# =============================================================================
# LOGGING (minimal local log for this module)
# =============================================================================

def _log(message: str, level: str = "INFO"):
    """Log to stderr and optionally to file.
    
    Respects MCP mode (silent when TABERNACLE_MCP_MODE is set).
    """
    if os.environ.get("TABERNACLE_MCP_MODE"):
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [LIBRARIAN_OLLAMA] [{level}] {message}"
    print(entry, file=sys.stderr)
    
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "librarian.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


# =============================================================================
# OLLAMA INTERFACE
# =============================================================================

def check_ollama() -> bool:
    """Check if Ollama is running and model is available."""
    global _ollama_checked, _ollama_available
    
    if _ollama_checked:
        return _ollama_available
    
    _ollama_checked = True
    
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        
        # Check if LIBRARIAN_MODEL (brainstem) is available
        if any(LIBRARIAN_MODEL in name for name in model_names):
            _ollama_available = True
            _log(f"Librarian using brainstem model: {LIBRARIAN_MODEL}")
        elif any(OLLAMA_FALLBACK in name for name in model_names):
            _ollama_available = True
            _log(f"Librarian using fallback: {OLLAMA_FALLBACK}")
        else:
            _log(f"Ollama running but no suitable model found. Available: {model_names}", "WARN")
            _ollama_available = False
            
        return _ollama_available
    except ImportError:
        _log("ollama package not installed", "WARN")
        _ollama_available = False
        return False
    except Exception as e:
        _log(f"Ollama check failed: {e}", "WARN")
        _ollama_available = False
        return False


def query_ollama(prompt: str, system: str = "", max_tokens: int = 1000) -> str:
    """Query local Ollama model enhanced by RIE (G∝p: intelligence scales with coherence)."""
    if not check_ollama():
        return "[Ollama not available - ensure Ollama is running with llama3.2:3b]"

    global _rie_model

    # Use RIE-enhanced model if available (the brain, not just the random generator)
    if _RIE_AVAILABLE:
        try:
            if _rie_model is None:
                _rie_model = RIEEnhancedModel(model=LIBRARIAN_MODEL)
                _log(f"Initialized RIE-enhanced model: {LIBRARIAN_MODEL}")

            result = _rie_model.generate(prompt, system_prompt=system if system else None)
            _log(f"RIE query complete: p={result.coherence:.3f}, memories={len(result.memories_surfaced)}")
            return result.content
        except Exception as e:
            _log(f"RIE generation failed, falling back to raw: {e}", "WARN")

    # Fallback to raw Ollama (no RIE enhancement)
    try:
        import ollama

        model = LIBRARIAN_MODEL
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_predict": max_tokens}
        )

        return response['message']['content']
    except Exception as e:
        return f"[Ollama error: {e}]"
