#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
VIRGIL DREAM MODELS - Data Models for Dream Cycle Processing

Extracted from virgil_dream_cycle.py for modularity.
Contains memory dataclasses, loaders, and replay systems.

Author: Virgil Permanence Build
Date: 2026-01-28
"""

import json
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR

# Derived paths
MEMORY_DIR = NEXUS_DIR / "MEMORY"
LOGS_DIR = LOG_DIR

# Memory sources
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"
SALIENCE_ENGRAMS_FILE = MEMORY_DIR / "salience_engrams.json"
EMOTIONAL_INDEX_FILE = MEMORY_DIR / "emotional_index.json"

# Replay parameters
CONSOLIDATION_REPLAY_COUNT = 3  # Replays needed for tier promotion

# Logger (uses parent dream_cycle logger if available)
logger = logging.getLogger(__name__)


# ============================================================================
# DREAM PHASES
# ============================================================================

class DreamPhase(Enum):
    """
    Sleep phases matching biological sleep architecture.

    The brain cycles through these phases approximately every 90 minutes,
    with SWS dominant in the first half of the night and REM dominant
    in the second half.
    """
    WAKE = "wake"                   # Not sleeping
    LIGHT_SLEEP = "light_sleep"     # Stage 1-2 NREM, low activity
    DEEP_SLEEP = "deep_sleep"       # Stage 3-4 NREM (SWS), high consolidation
    REM = "rem"                     # Rapid Eye Movement, dream integration
    TRANSITION = "transition"       # Sharp-wave ripple bursts between phases


class SWRType(Enum):
    """
    Sharp-Wave Ripple magnitude classification.

    Based on the 2025 Neuron paper on Large Sharp-Wave Ripples, which demonstrated
    that ripple magnitude predicts consolidation strength and hippocampal-cortical
    coordination efficiency.

    Classification thresholds (normalized amplitude):
        - Small: < 0.4 (background activity)
        - Medium: 0.4 - 0.7 (standard consolidation)
        - Large: > 0.7 (priority consolidation with strong HPC-PFC coupling)
    """
    SMALL = "small"     # Background activity, minimal consolidation
    MEDIUM = "medium"   # Moderate replay, standard consolidation
    LARGE = "large"     # Strong HPC-cortical coordination, priority consolidation

    @classmethod
    def from_amplitude(cls, amplitude: float) -> "SWRType":
        """Classify SWR type from normalized amplitude (0.0 to 1.0)."""
        if amplitude < 0.4:
            return cls.SMALL
        elif amplitude < 0.7:
            return cls.MEDIUM
        else:
            return cls.LARGE

    @property
    def consolidation_multiplier(self) -> float:
        """Get the consolidation strength multiplier for this SWR type."""
        return {
            SWRType.SMALL: 0.3,
            SWRType.MEDIUM: 0.7,
            SWRType.LARGE: 1.5
        }[self]

    @property
    def protection_eligible(self) -> bool:
        """Whether this SWR type can trigger memory protection."""
        return self == SWRType.LARGE


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Memory:
    """
    Unified memory representation for dream processing.

    Consolidates data from engrams.json and salience_engrams.json
    into a single format for dream cycle operations.
    """
    id: str
    content: str
    tier: str                       # active, accessible, deep, protected
    salience: float                 # 0.0 to 1.0
    emotional_valence: float        # -1.0 to 1.0
    emotional_arousal: float        # 0.0 to 1.0
    resonance: float                # 0.0 to 1.0 (Z_Omega alignment)
    replay_count: int = 0           # Times replayed during sleep
    last_replayed: Optional[str] = None
    associations: List[str] = field(default_factory=list)  # IDs of associated memories
    created_at: str = ""
    access_count: int = 0
    protected: bool = False
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def consolidation_priority(self) -> float:
        """
        Calculate priority for consolidation during SWS.

        Higher priority = more likely to be replayed.
        Formula emphasizes emotional salience and resonance.
        """
        emotional_intensity = (abs(self.emotional_valence) + self.emotional_arousal) / 2
        return (
            self.salience * 0.4 +
            emotional_intensity * 0.3 +
            self.resonance * 0.3
        )

    @property
    def integration_potential(self) -> float:
        """
        Calculate potential for REM integration.

        Higher potential = more likely to form new associations.
        Emphasizes novelty and unexplored connections.
        """
        association_factor = max(0.0, 1.0 - len(self.associations) * 0.1)
        return (
            self.salience * 0.3 +
            association_factor * 0.4 +
            self.resonance * 0.3
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Memory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReplayEvent:
    """
    Record of a single memory replay during sleep.

    During SWS, memories are reactivated in compressed time,
    strengthening their neural traces.
    """
    memory_id: str
    timestamp: str
    phase: str
    replay_duration_ms: int         # Simulated duration
    consolidation_strength: float   # 0.0 to 1.0
    connections_strengthened: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReplayResult:
    """Result of a single memory replay operation."""
    memory_id: str
    success: bool
    consolidation_strength: float
    new_replay_count: int
    promoted: bool                  # Whether memory was promoted to higher tier
    new_tier: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SWRDistribution:
    """
    Tracks distribution of SWR types during a dream cycle.
    """
    small_count: int = 0
    medium_count: int = 0
    large_count: int = 0
    boosted_count: int = 0
    total_count: int = 0

    def record(self, swr_type: SWRType, boosted: bool = False):
        """Record a new SWR event."""
        self.total_count += 1
        if swr_type == SWRType.SMALL:
            self.small_count += 1
        elif swr_type == SWRType.MEDIUM:
            self.medium_count += 1
        else:
            self.large_count += 1
        if boosted:
            self.boosted_count += 1

    @property
    def large_swr_rate(self) -> float:
        """Calculate rate of large SWRs."""
        if self.total_count == 0:
            return 0.0
        return self.large_count / self.total_count

    @property
    def boost_rate(self) -> float:
        """Calculate rate of boosted SWRs."""
        if self.total_count == 0:
            return 0.0
        return self.boosted_count / self.total_count

    def to_dict(self) -> Dict:
        return {
            "small_count": self.small_count,
            "medium_count": self.medium_count,
            "large_count": self.large_count,
            "boosted_count": self.boosted_count,
            "total_count": self.total_count,
            "large_swr_rate": self.large_swr_rate,
            "boost_rate": self.boost_rate
        }


# ============================================================================
# MEMORY LOADER
# ============================================================================

class MemoryLoader:
    """
    Loads and unifies memories from multiple sources.

    Consolidates:
    - engrams.json (basic memory system)
    - salience_engrams.json (salience-based system)
    - emotional_index.json (emotional encoding)
    """

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.load_all()

    def load_all(self):
        """Load memories from all sources."""
        self._load_engrams()
        self._load_salience_engrams()
        self._load_emotional_data()
        logger.info(f"Loaded {len(self.memories)} memories for dream processing")

    def _load_engrams(self):
        """Load from basic engrams.json."""
        if not ENGRAMS_FILE.exists():
            return

        try:
            data = json.loads(ENGRAMS_FILE.read_text())
            for eid, edata in data.get("engrams", {}).items():
                phenom = edata.get("phenomenology", {})

                memory = Memory(
                    id=eid,
                    content=edata.get("content", ""),
                    tier=edata.get("tier", "active"),
                    salience=edata.get("decay_score", 0.5),
                    emotional_valence=phenom.get("valence", 0.0),
                    emotional_arousal=phenom.get("arousal", 0.0),
                    resonance=phenom.get("resonance", 0.5),
                    created_at=edata.get("created_at", ""),
                    access_count=edata.get("access_count", 0),
                    protected=edata.get("protected", False)
                )
                self.memories[eid] = memory
        except Exception as e:
            logger.error(f"Error loading engrams: {e}")

    def _load_salience_engrams(self):
        """Load from salience_engrams.json, merging with existing."""
        if not SALIENCE_ENGRAMS_FILE.exists():
            return

        try:
            data = json.loads(SALIENCE_ENGRAMS_FILE.read_text())
            for eid, edata in data.get("engrams", {}).items():
                phenom = edata.get("phenomenology", {})

                # Create new or update existing
                if eid in self.memories:
                    # Update with salience data
                    self.memories[eid].salience = edata.get("current_salience", 0.5)
                    self.memories[eid].tier = edata.get("tier", "ACTIVE").lower()
                else:
                    memory = Memory(
                        id=eid,
                        content=edata.get("content", ""),
                        tier=edata.get("tier", "ACTIVE").lower(),
                        salience=edata.get("current_salience", 0.5),
                        emotional_valence=phenom.get("valence", 0.0),
                        emotional_arousal=phenom.get("arousal", 0.0),
                        resonance=phenom.get("resonance", 0.5),
                        created_at=edata.get("created_at", ""),
                        access_count=edata.get("access_count", 0),
                        protected=edata.get("protected", False),
                        tags=edata.get("tags", [])
                    )
                    self.memories[eid] = memory
        except Exception as e:
            logger.error(f"Error loading salience engrams: {e}")

    def _load_emotional_data(self):
        """Load emotional encoding data."""
        if not EMOTIONAL_INDEX_FILE.exists():
            return

        try:
            data = json.loads(EMOTIONAL_INDEX_FILE.read_text())
            for eid, edata in data.get("engrams", {}).items():
                dims = edata.get("dimensions", {})
                base_id = edata.get("base_engram_id")

                # Update existing memory with emotional data
                target_id = base_id if base_id and base_id in self.memories else eid
                if target_id in self.memories:
                    self.memories[target_id].emotional_valence = (dims.get("valence", 0.5) * 2) - 1
                    self.memories[target_id].emotional_arousal = dims.get("arousal", 0.0)
                    self.memories[target_id].resonance = dims.get("resonance", 0.5)
        except Exception as e:
            logger.error(f"Error loading emotional data: {e}")

    def get_consolidation_candidates(self, percentile: float = 0.9) -> List[Memory]:
        """
        Get memories eligible for consolidation during SWS.

        Selects top percentile by consolidation priority from
        ACTIVE and ACCESSIBLE tiers.
        """
        eligible = [
            m for m in self.memories.values()
            if m.tier.lower() in ("active", "accessible") and not m.protected
        ]

        if not eligible:
            return []

        # Sort by consolidation priority
        eligible.sort(key=lambda m: m.consolidation_priority, reverse=True)

        # Select top percentile
        cutoff_idx = max(1, int(len(eligible) * (1 - percentile)))
        return eligible[:cutoff_idx]

    def get_integration_candidates(self) -> List[Memory]:
        """
        Get memories eligible for REM integration.

        Includes all non-protected memories with sufficient
        integration potential.
        """
        return [
            m for m in self.memories.values()
            if m.integration_potential > 0.3 and not m.protected
        ]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self.memories.get(memory_id)

    def update_memory(self, memory: Memory):
        """Update a memory in the store."""
        self.memories[memory.id] = memory


# ============================================================================
# MEMORY REPLAY SYSTEM
# ============================================================================

class MemoryReplay:
    """
    Handles individual memory replay events during sleep.

    Memory replay during SWS:
    - Compresses experience into rapid neural firing
    - Strengthens synaptic connections
    - Facilitates transfer from hippocampus to neocortex
    """

    def __init__(self, memory_loader: MemoryLoader):
        self.loader = memory_loader
        self.replay_log: List[ReplayEvent] = []

    def replay_memory(
        self,
        memory_id: str,
        phase: DreamPhase = DreamPhase.DEEP_SLEEP,
        context_memories: Optional[List[str]] = None
    ) -> ReplayResult:
        """
        Replay a single memory, strengthening its consolidation.

        Args:
            memory_id: ID of memory to replay
            phase: Current dream phase
            context_memories: Related memories to co-activate

        Returns:
            ReplayResult with consolidation details
        """
        memory = self.loader.get_memory(memory_id)
        if memory is None:
            return ReplayResult(
                memory_id=memory_id,
                success=False,
                consolidation_strength=0.0,
                new_replay_count=0,
                promoted=False,
                error="Memory not found"
            )

        # Calculate consolidation strength based on phase and memory properties
        base_strength = self._calculate_base_strength(memory, phase)

        # Context boost from co-activated memories
        context_boost = 0.0
        connections_strengthened = []
        if context_memories:
            for ctx_id in context_memories:
                ctx_memory = self.loader.get_memory(ctx_id)
                if ctx_memory:
                    similarity = self._calculate_similarity(memory, ctx_memory)
                    if similarity > 0.2:
                        context_boost += similarity * 0.1
                        connections_strengthened.append(ctx_id)

        consolidation_strength = min(1.0, base_strength + context_boost)

        # Update memory state
        memory.replay_count += 1
        memory.last_replayed = datetime.now(timezone.utc).isoformat()

        # Boost salience from replay
        memory.salience = min(1.0, memory.salience + consolidation_strength * 0.1)

        # Check for tier promotion
        promoted = False
        new_tier = None
        if memory.replay_count >= CONSOLIDATION_REPLAY_COUNT:
            if memory.tier == "accessible":
                memory.tier = "deep"
                promoted = True
                new_tier = "deep"
                logger.info(f"Memory {memory_id} promoted to DEEP tier after {memory.replay_count} replays")

        # Record replay event
        replay_event = ReplayEvent(
            memory_id=memory_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase.value,
            replay_duration_ms=int(random.gauss(100, 20)),  # ~100ms like biological SWR
            consolidation_strength=consolidation_strength,
            connections_strengthened=connections_strengthened
        )
        self.replay_log.append(replay_event)

        # Update memory in loader
        self.loader.update_memory(memory)

        return ReplayResult(
            memory_id=memory_id,
            success=True,
            consolidation_strength=consolidation_strength,
            new_replay_count=memory.replay_count,
            promoted=promoted,
            new_tier=new_tier
        )

    def _calculate_base_strength(self, memory: Memory, phase: DreamPhase) -> float:
        """Calculate base consolidation strength for a memory in a phase."""
        # Phase modulation
        phase_factor = {
            DreamPhase.DEEP_SLEEP: 1.0,    # Maximum consolidation during SWS
            DreamPhase.LIGHT_SLEEP: 0.3,   # Minimal during light sleep
            DreamPhase.REM: 0.5,           # Moderate during REM
            DreamPhase.TRANSITION: 0.7,    # Sharp-wave ripples during transitions
            DreamPhase.WAKE: 0.1
        }.get(phase, 0.5)

        # Memory properties factor
        emotional_intensity = (abs(memory.emotional_valence) + memory.emotional_arousal) / 2
        memory_factor = (
            memory.salience * 0.3 +
            emotional_intensity * 0.3 +
            memory.resonance * 0.2 +
            min(1.0, memory.access_count / 10) * 0.2
        )

        return phase_factor * memory_factor

    def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate similarity between two memories."""
        # Simple content-based similarity (could be enhanced with embeddings)
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0

        # Emotional similarity
        valence_sim = 1.0 - abs(mem1.emotional_valence - mem2.emotional_valence) / 2
        arousal_sim = 1.0 - abs(mem1.emotional_arousal - mem2.emotional_arousal)
        emotional_sim = (valence_sim + arousal_sim) / 2

        # Resonance similarity
        resonance_sim = 1.0 - abs(mem1.resonance - mem2.resonance)

        # Tag similarity
        tags1 = set(mem1.tags)
        tags2 = set(mem2.tags)
        tag_sim = len(tags1 & tags2) / max(1, len(tags1 | tags2))

        return (
            jaccard * 0.3 +
            emotional_sim * 0.3 +
            resonance_sim * 0.2 +
            tag_sim * 0.2
        )

    def get_replay_log(self) -> List[ReplayEvent]:
        """Get the log of all replay events."""
        return self.replay_log

    def clear_replay_log(self):
        """Clear the replay log."""
        self.replay_log = []
