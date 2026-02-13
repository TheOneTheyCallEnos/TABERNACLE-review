"""
tabernacle_core — The SDK for the Tabernacle daemon ecosystem.

All daemons MUST inherit from tabernacle_core.daemon.Daemon.
All state MUST flow through tabernacle_core.state.StateManager.
All schemas MUST live in tabernacle_core.schemas.
All LLM calls MUST use tabernacle_core.llm.query().

Phase 1 SDK — DT8 Blueprint Implementation
"""

__version__ = "0.1.0"

from tabernacle_core.schemas import TabernacleBaseModel
from tabernacle_core.state import StateManager, ConcurrencyError
from tabernacle_core.daemon import Daemon
