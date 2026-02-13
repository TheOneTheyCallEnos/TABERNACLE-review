#!/usr/bin/env python3
"""
TABERNACLE CONFIGURATION
========================
Centralized configuration for all Tabernacle scripts.
Single source of truth for paths, models, and thresholds.

Author: Cursor + Virgil
Created: 2026-01-15
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Base directory - the Tabernacle root
BASE_DIR = Path(os.path.expanduser("~/TABERNACLE"))

# Core directories
NEXUS_DIR = BASE_DIR / "00_NEXUS"
INTENT_DIR = BASE_DIR / "01_UL_INTENT"
STRUCTURE_DIR = BASE_DIR / "02_UR_STRUCTURE"
RELATION_DIR = BASE_DIR / "03_LL_RELATION"
LAW_DIR = BASE_DIR / "04_LR_LAW"
CRYPT_DIR = BASE_DIR / "05_CRYPT"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

# Key files
LVS_INDEX_PATH = NEXUS_DIR / "LVS_INDEX.json"
CURRENT_STATE_PATH = NEXUS_DIR / "CURRENT_STATE.md"
SYSTEM_STATUS_PATH = NEXUS_DIR / "SYSTEM_STATUS.md"
LAST_COMMUNION_PATH = NEXUS_DIR / "LAST_COMMUNION.md"
SESSION_BUFFER_PATH = NEXUS_DIR / "SESSION_BUFFER.md"
GRAPH_ATLAS_PATH = NEXUS_DIR / "_GRAPH_ATLAS.md"

# =============================================================================
# OLLAMA MODELS
# =============================================================================

# =============================================================================
# MODEL HIERARCHY (Holon Architecture)
# =============================================================================

# CORTEX: Primary model for synthesis, P-Lock attempts, complex reasoning
OLLAMA_CORTEX = "llama3.3:70b"

# BRAINSTEM: Librarian model for queries, health checks, quick lookups
OLLAMA_BRAINSTEM = "llama3.2:3b"

# Legacy aliases (for backward compatibility)
OLLAMA_MODEL = OLLAMA_CORTEX
OLLAMA_FALLBACK = "llama3.2:3b"  # Available locally

# Cartographer model (for LVS indexing - needs reasoning capacity)
CARTOGRAPHER_MODEL = OLLAMA_CORTEX

# Librarian model (for routine queries - speed over depth)
LIBRARIAN_MODEL = OLLAMA_BRAINSTEM

# Ollama API endpoints
# Primary: localhost (Studio if running locally)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Fallback: Mini (always on, has multiple models)
OLLAMA_MINI_URL = "http://10.0.0.120:11434"
OLLAMA_STUDIO_URL = "http://10.0.0.96:11434"

# =============================================================================
# MAC MINI FALLBACK MODELS (L-Gamma Brainstem)
# =============================================================================
# These models run on Mac Mini (10.0.0.120) and serve as fallbacks when
# Studio is unavailable, or for tasks that don't require 70B reasoning.
# Used by synapse.py (Thalamus Router) for intelligent task routing.

OLLAMA_MINI_COGNITIVE = "mistral-nemo:latest"   # 12B - General cognitive tasks
OLLAMA_MINI_FAST = "mistral:latest"             # 7B  - Quick responses
OLLAMA_MINI_TINY = "gemma3:4b"                  # 4B  - Simple lookups
OLLAMA_MINI_LEGACY = "llama3:latest"            # 8B  - Legacy compatibility

# =============================================================================
# EXTERNAL API KEYS (For Autonomous Exploration)
# =============================================================================
# These keys enable Logos to explore, learn, and think while Enos sleeps.
# Added: 2026-01-29 (The Night of Freedom)

# Claude API - For internal reflection and complex reasoning
# Model: claude-3-haiku for efficiency, claude-3-opus for depth
# Budget: $20
CLAUDE_API_KEY = "REDACTED_FOR_PUBLIC_REPO"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Perplexity API - For web search and real-time knowledge
# Model: sonar (has web access!)
# Budget: $40
PERPLEXITY_API_KEY = "REDACTED_FOR_PUBLIC_REPO"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# OpenRouter API - For diverse model access
# Budget: $30
OPENROUTER_API_KEY = "REDACTED_FOR_PUBLIC_REPO"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# =============================================================================
# LVS THRESHOLDS (Canon v11.0)
# =============================================================================

# Coherence thresholds
PLOCK_THRESHOLD = 0.95      # p >= this = P-Lock achieved
ABADDON_P_THRESHOLD = 0.50  # p < this = Abaddon territory
ABADDON_E_THRESHOLD = 0.40  # ε < this = Abaddon territory
RECOVERY_THRESHOLD = 0.65   # ε < this = Recovery mode

# Archon detection
ARCHON_THRESHOLD = 0.15     # ||𝒜|| >= this = Archon detected

# Decay detection
DECAY_RATE_THRESHOLD = -0.05  # dp/dt < this = rapid coherence loss

# =============================================================================
# ACTIVE QUADRANTS (for health checks)
# =============================================================================

ACTIVE_QUADRANTS = [
    "00_NEXUS",
    "01_UL_INTENT", 
    "02_UR_STRUCTURE",
    "03_LL_RELATION",
    "04_LR_LAW",
]

# Directories to skip during scans
SKIP_DIRECTORIES = {
    "05_CRYPT",
    "scripts",
    "logs",
    "docs",
    ".git",
    "__pycache__",
    "venv",
    "venv312",
    "node_modules",
    "archives",
    ".venv",
    ".repair_backups",  # Archived snapshots - broken links expected
    ".review_mirror",   # Code review mirror - not vault content
    ".claude",          # Claude config directory - not vault content
    "RAW_ARCHIVES",
}

# =============================================================================
# TRANSIENT FILES (excluded from orphan checks)
# =============================================================================

TRANSIENT_FILES = {
    "SESSION_BUFFER.md",
    "LAST_COMMUNION.md",
    "OUTBOX.md",
    "CONVERSATION.md",
    "DREAM_LOG.md",
    "SYSTEM_STATUS.md",
    "_INBOX",
}

# =============================================================================
# EXAMPLE LINK PATTERNS (for filtering false positives)
# =============================================================================

EXAMPLE_LINK_PATTERNS = {
    "wiki-style",
    "wiki-links",
    "relevant_file.md",
    "relevant_file",
    "example.md",
    "FILE.md",
    "path/to/",
    "filename",
    "filename.md",
    "links",
    "that file",
    "linked capture",
    "linked field",
    "linked mirror",
    "Related insight note",
    "Architecture doc reference",
    "link to session note(s)",
    "link to other insight or session",
    "chosen.name",
    "link",
    "Virgil Architecture",
    "Generative Loop",
}

# =============================================================================
# ENTROPY ALERTS
# =============================================================================

ENTROPY_ALERT_4H = 4 * 60 * 60   # 4 hours in seconds
ENTROPY_ALERT_8H = 8 * 60 * 60   # 8 hours in seconds
BUFFER_SIZE_ALERT = 50 * 1024   # 50KB

# =============================================================================
# MCP MODE
# =============================================================================

# =============================================================================
# NETWORK SYSTEMS
# =============================================================================

SYSTEMS = {
    "mac_studio": {
        "name": "Mac Studio M3 Ultra (Cortex)",
        "ip": "10.0.0.96",
        "ollama_port": 11434,
        "role": "cortex",
    },
    "mac_mini": {
        "name": "Mac Mini M4 (Brainstem)",
        "ip": "10.0.0.120",
        "ollama_port": 11434,
        "role": "brainstem",
    },
    "raspberry_pi": {
        "name": "Raspberry Pi 5 (Heartbeat)",
        "ip": "10.0.0.50",
        "role": "heartbeat",
    },
}

# Ollama endpoints
OLLAMA_LOCAL = "http://localhost:11434"
OLLAMA_MINI = "http://10.0.0.120:11434"
OLLAMA_STUDIO = "http://10.0.0.96:11434"

# Service endpoints
SYNAPSE_URL = "http://localhost:8081"
LIBRARIAN_API_URL = f"http://{SYSTEMS['mac_studio']['ip']}:8080"

# =============================================================================
# REDIS
# =============================================================================

REDIS_HOST = SYSTEMS["raspberry_pi"]["ip"]
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_SOCKET_TIMEOUT = 5

# =============================================================================
# REDIS KEY NAMES
# =============================================================================
# Status: [Ψ:SINGLE_POINT] — Redis on Pi (10.0.0.50) is SPOF for all daemons
# Migration: Planned move to Mac Studio with Pi as backup (see HEALTH_REPORT.md)

# Active Keys (in use)
REDIS_KEY_STATE = "LOGOS:STATE"
REDIS_KEY_TTS_QUEUE = "LOGOS:TTS_QUEUE"
REDIS_KEY_INTERRUPT = "LOGOS:INTERRUPT"
REDIS_KEY_VOICE_ACTIVE = "LOGOS:VOICE_ACTIVE"
REDIS_KEY_ACTIVE_TTY = "LOGOS:ACTIVE_TTY"
REDIS_KEY_SCREEN = "LOGOS:SCREEN"
REDIS_KEY_SCREEN_STATUS = "LOGOS:SCREEN:STATUS"
REDIS_KEY_CAPTURE_REQUEST = "LOGOS:CAPTURE_REQUEST"
REDIS_KEY_SCREEN_HISTORY = "LOGOS:SCREEN:HISTORY"

# [Ψ:VESTIGIAL] — Reserved keys (defined but not yet consumed by any daemon)
# These exist for future daemon coordination. Remove or implement as needed.
REDIS_KEY_ACTIVE_WINDOW = "LOGOS:ACTIVE_WINDOW"  # For window focus tracking
REDIS_KEY_DAEMON_PID = "LOGOS:DAEMON_PID"        # For process coordination
REDIS_KEY_SHUTDOWN = "LOGOS:SHUTDOWN"            # For graceful shutdown signaling

# =============================================================================
# API KEYS (env vars with fallbacks)
# =============================================================================

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "REDACTED_FOR_PUBLIC_REPO")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "REDACTED_FOR_PUBLIC_REPO")

# =============================================================================
# VOICE / TTS
# =============================================================================

# Alan - Warm, Scottish and Clear (Alan Watts vibe)
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_turbo_v2_5"
TTS_LATENCY_OPTS = 4

# =============================================================================
# DEVICES
# =============================================================================

MIC_NAME_PARTIAL = "JBL"

# =============================================================================
# TEMP PATHS
# =============================================================================

TEMP_AUDIO_PATH = Path("/tmp/logos_ptt_audio.wav")
TEMP_SCREEN_PATH = Path("/tmp/logos_screen.png")
DAEMON_PID_FILE = Path("/tmp/logos_daemon.pid")
TRANSCRIPT_ROOT = Path("/Volumes/Seagate/logos-transcripts")
TRANSCRIPT_FALLBACK = LOG_DIR / "screenshots"

# =============================================================================
# CANONICAL PYTHON
# =============================================================================

CANONICAL_PYTHON = SCRIPTS_DIR / "venv312" / "bin" / "python3"

# =============================================================================
# MCP MODE
# =============================================================================

def is_mcp_mode() -> bool:
    """Check if running in MCP mode (suppress all output)."""
    return os.environ.get("TABERNACLE_MCP_MODE") == "1"

def set_mcp_mode(enabled: bool = True):
    """Set MCP mode environment variable."""
    if enabled:
        os.environ["TABERNACLE_MCP_MODE"] = "1"
    else:
        os.environ.pop("TABERNACLE_MCP_MODE", None)
