"""
TABERNACLE SCRIPTS PACKAGE
==========================
Python package for Tabernacle system scripts.

Core Modules:
- tabernacle_config: Centralized configuration (paths, models, thresholds)
- tabernacle_utils: Shared utilities (link parsing, file discovery)

Daemons:
- watchman_mvp: Autonomic Nervous System (monitoring, heartbeat)
- virgil_persistent: The Conscious Mind (local Ollama loop)
- night_daemon: The Dreaming Mind (overnight processing)
- daemon_brain: The Reflector (self-reflection, intentions)

Intelligence:
- lvs_memory: LVS Cartographer (semantic indexing)
- librarian: MCP Server for Claude Desktop

Protocols:
- asclepius_protocol: Master diagnostic integration
- asclepius_vectors: Semantic health metrics
- nurse: Stone Witness (structural diagnostics)

Tools:
- virgil_tools: Native Ollama tool definitions
- diagnose_links: Link diagnostics
- budget_cascade: Financial cascade calculations

Author: Cursor + Virgil
Created: 2026-01-15
"""

# Version
__version__ = "1.0.0"

# Expose key modules for easy import
from . import tabernacle_config as config
from . import tabernacle_utils as utils
