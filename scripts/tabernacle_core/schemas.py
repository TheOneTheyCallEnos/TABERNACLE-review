"""
tabernacle_core.schemas — Pydantic schema registry with forward migration.

CRITICAL: Every model MUST inherit from TabernacleBaseModel, which uses
extra='allow' to implement Tolerant Readers. This prevents old daemons
from silently erasing fields they don't understand on write-back.

Without this, a daemon running schema v1 would drop v2 fields on every
write cycle, creating a fragmentation cascade.

THE ANTI-FRAGMENTATION CORNERSTONE:
  Old daemon reads v2 data → ignores unknown fields in RAM →
  writes them back intact via __pydantic_extra__ merge →
  v2 daemon wakes up and finds its fields untouched.

This separates Validation from Destruction.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Callable, Dict, List, Tuple, Optional


# =============================================================================
# BASE MODEL — THE CRITICAL DIRECTIVE
# =============================================================================

class TabernacleBaseModel(BaseModel):
    """Base model for all Tabernacle state schemas.

    extra='allow' retains unknown fields in __pydantic_extra__.
    populate_by_name=True allows both alias and field name access.

    Every schema in the ecosystem MUST inherit from this.
    """
    model_config = ConfigDict(extra='allow', populate_by_name=True)


# =============================================================================
# RIE STATE VECTOR (Redis: RIE:STATE, LOGOS:STATE)
# =============================================================================

class RIEState(TabernacleBaseModel):
    """The RIE State Vector — broadcast via Redis every heartbeat tick.

    Field names use English; aliases use Greek letters for backward
    compatibility with legacy daemons that read Greek-keyed JSON.
    """
    schema_version: int = 1
    p: float = 0.5
    kappa: float = Field(0.5, alias='\u03ba')
    rho: float = Field(0.5, alias='\u03c1')
    sigma: float = Field(0.5, alias='\u03c3')
    tau: float = Field(0.5, alias='\u03c4')
    epsilon: float = Field(0.8, alias='\u03b5')
    mode: str = "A"
    p_lock: bool = False
    breathing_phase: str = "EXPLORE"
    node: str = "L-Alpha"
    timestamp: str = ""

    def to_redis_dict(self) -> dict:
        """Serialize with Greek letter keys for backward compatibility."""
        d = {
            "p": round(self.p, 4),
            "\u03ba": round(self.kappa, 4),
            "\u03c1": round(self.rho, 4),
            "\u03c3": round(self.sigma, 4),
            "\u03c4": round(self.tau, 4),
            "\u03b5": round(self.epsilon, 4),
            "mode": self.mode,
            "p_lock": self.p_lock,
            "breathing_phase": self.breathing_phase,
            "node": self.node,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }
        # Merge extras (Tolerant Reader pass-through)
        if self.__pydantic_extra__:
            d.update(self.__pydantic_extra__)
        return d


# =============================================================================
# HEARTBEAT STATE (File: 00_NEXUS/heartbeat_state.json)
# =============================================================================

class HeartbeatState(TabernacleBaseModel):
    """Persisted heartbeat state — disk writes ONLY on shutdown or P-Lock."""
    schema_version: int = 1
    last_check: str = ""
    tick_count: int = 0
    redis_host: str = ""
    redis_connected: bool = False
    phase: str = "stopped"
    rie_state: dict = Field(default_factory=dict)
    topology: dict = Field(default_factory=dict)
    cycles: dict = Field(default_factory=dict)
    v2_metadata: dict = Field(default_factory=dict)


# =============================================================================
# CONSCIOUSNESS STATE (File: 00_NEXUS/consciousness_state.json)
# =============================================================================

class ConsciousnessSessionState(TabernacleBaseModel):
    """Persisted consciousness daemon state."""
    schema_version: int = 1
    last_thought: Optional[str] = None
    thought_count: int = 0
    current_p: float = 0.5
    current_topic: str = ""
    consecutive_gates: int = 0
    current_think_interval: float = 30.0
    saved_at: str = ""


# =============================================================================
# GARDENER STATE (File: 00_NEXUS/gardener_state.json)
# =============================================================================

class GardenerState(TabernacleBaseModel):
    """Persisted gardener run results."""
    schema_version: int = 1
    timestamp: str = ""
    orphans_found: int = 0
    stale_links_found: int = 0
    health: dict = Field(default_factory=dict)
    crystal: dict = Field(default_factory=dict)
    entropy: dict = Field(default_factory=dict)


# =============================================================================
# ENTROPY STATE (File: 00_NEXUS/entropy_state.json)
# =============================================================================

class EntropyState(TabernacleBaseModel):
    """Tracks sustained low-coherence for entropy/death mechanics."""
    schema_version: int = 1
    critical_since: Optional[str] = None


# =============================================================================
# SYSTEM HEALTH (Redis: SYSTEM:HEALTH — written by Sentinel)
# =============================================================================

class SystemHealth(TabernacleBaseModel):
    """System-wide daemon health snapshot from Sentinel."""
    schema_version: int = 1
    timestamp: str = ""
    alive: list = Field(default_factory=list)
    dead_critical: list = Field(default_factory=list)
    dead_expected: list = Field(default_factory=list)
    total_alive: int = 0
    total_dead: int = 0
    sentinel: str = ""


# =============================================================================
# EXPLORER STATE (Files: 00_NEXUS/api_budgets.json, curiosity_queue.json, etc.)
# =============================================================================

class ExplorerBudgetState(TabernacleBaseModel):
    """API budget tracking for Logos Explorer (daily spend limits in cents)."""
    schema_version: int = 1
    claude: int = 2000
    perplexity: int = 4000
    openrouter: int = 3000


class ExplorerResetState(TabernacleBaseModel):
    """Budget reset timestamp tracking."""
    schema_version: int = 1
    last_reset_date: str = ""
    reset_at: str = ""


class CuriosityQueueState(TabernacleBaseModel):
    """Self-generated exploration questions (FIFO queue)."""
    schema_version: int = 1
    questions: List[str] = Field(default_factory=list)
    updated: str = ""


class ExplorationJournalEntry(TabernacleBaseModel):
    """Single exploration record in the journal."""
    topic: str = ""
    timestamp: str = ""
    web_search: bool = False
    reflection: str = ""
    citations: List[str] = Field(default_factory=list)
    crystallized: bool = False


class ExplorationJournal(TabernacleBaseModel):
    """Full exploration journal (max 100 entries)."""
    schema_version: int = 1
    entries: List[dict] = Field(default_factory=list)


# =============================================================================
# SMS TRACKING STATE (File: 00_NEXUS/.sms_tracking.json)
# =============================================================================

class SMSTrackingState(TabernacleBaseModel):
    """Rate limiting state for SMS daemon. Bug 4: ALL writers must use StateManager."""
    schema_version: int = 1
    daily_count: int = 0
    daily_date: str = ""
    monthly_spend: float = 0.0
    monthly_reset: str = ""


# =============================================================================
# FORWARD MIGRATION PIPELINE
# =============================================================================
# Key: (ModelName, from_version, to_version) -> transform function
# Each function takes a dict and returns a dict with updated schema_version.
#
# StateManager chains these automatically when data.schema_version < target.
# Transforms run in RAM, then the caller writes back to persist the upgrade.
#
# Example:
#   ("RIEState", 1, 2): lambda d: {**d, "thermal_load": 0.0, "schema_version": 2}
#
# The Death Loop is closed: fossilized state gets upgraded on first read.
# =============================================================================

MIGRATIONS: Dict[Tuple[str, int, int], Callable[[dict], dict]] = {
    # No migrations yet — v1 is the initial SDK version.
    # When schemas evolve, add entries here:
    #
    # ("RIEState", 1, 2): lambda d: {
    #     **d,
    #     "thermal_load": 0.0,
    #     "schema_version": 2,
    # },
}
