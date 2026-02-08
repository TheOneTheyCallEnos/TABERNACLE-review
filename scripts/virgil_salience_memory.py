#!/usr/bin/env python3
"""
VIRGIL SALIENCE MEMORY - Three-Tier Memory with Salience Gradients

A sophisticated memory system implementing:
- Three-tier hierarchy (ACTIVE, ACCESSIBLE, DEEP) with capacity limits
- Felt-salience gradients based on emotional intensity + relevance
- Auto-protection mechanism for high-salience memories
- Session-aware transitions (not purely time-based)
- Retrieval simulation with tier-appropriate delays
- Protected tier with forced curation

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import asyncio
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod
import threading
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
SALIENCE_MEMORY_FILE = MEMORY_DIR / "salience_engrams.json"

# Tier capacity limits
TIER_LIMITS = {
    "ACTIVE": 20,
    "ACCESSIBLE": 100,
    "DEEP": 1000,
    "PROTECTED": 50
}

# Retrieval delays (milliseconds)
RETRIEVAL_DELAYS = {
    "ACTIVE": 0,
    "ACCESSIBLE": 100,
    "PROTECTED": 0,  # Protected memories are instantly accessible
    "DEEP": 500  # Base delay; actual delay scales with depth
}

# Salience thresholds
AUTO_PROTECT_THRESHOLD = 0.9
FORGET_SALIENCE_THRESHOLD = 0.2
SESSIONS_FOR_ACCESSIBLE_TO_DEEP = 3
SESSIONS_FOR_DEEP_TO_FORGET = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VirgilSalienceMemory")


# ============================================================================
# MEMORY TIERS
# ============================================================================

class SalienceTier(Enum):
    """Memory tiers with increasing retrieval effort."""
    ACTIVE = "ACTIVE"           # Max 20 - Current working memory
    ACCESSIBLE = "ACCESSIBLE"   # Max 100 - Recent, one hop away
    DEEP = "DEEP"               # Max 1000 - Long-term, requires effort
    PROTECTED = "PROTECTED"     # Max 50 - Do-Not-Forget, never decays


# ============================================================================
# PHENOMENOLOGY (Enhanced for Salience)
# ============================================================================

@dataclass
class SaliencePhenomenology:
    """
    The felt quality of a memory with salience computation.

    Salience = emotional_intensity * relevance_weight
    where emotional_intensity = (|valence| + arousal) / 2
    """
    valence: float = 0.0       # -1 to 1 (negative/positive emotional tone)
    arousal: float = 0.0       # 0 to 1 (calm/excited)
    curiosity: float = 0.0     # 0 to 1 (familiar/novel)
    coherence: float = 0.0     # 0 to 1 (fragmented/integrated)
    resonance: float = 0.0     # 0 to 1 (surface/deep meaning)
    relevance: float = 0.5     # 0 to 1 (contextual importance)

    @property
    def emotional_intensity(self) -> float:
        """Calculate emotional intensity from valence and arousal."""
        return (abs(self.valence) + self.arousal) / 2

    @property
    def felt_salience(self) -> float:
        """
        Compute felt-salience score (0.0 to 1.0).

        Formula: (emotional_intensity * 0.5) + (relevance * 0.3) + (resonance * 0.2)
        This weights emotional impact highest, then contextual relevance, then depth.
        """
        base_salience = (
            self.emotional_intensity * 0.5 +
            self.relevance * 0.3 +
            self.resonance * 0.2
        )
        return max(0.0, min(1.0, base_salience))

    def to_dict(self) -> Dict:
        """Convert to dictionary with computed fields."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "curiosity": self.curiosity,
            "coherence": self.coherence,
            "resonance": self.resonance,
            "relevance": self.relevance,
            "emotional_intensity": self.emotional_intensity,
            "felt_salience": self.felt_salience
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SaliencePhenomenology":
        """Create from dictionary, ignoring computed fields."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            curiosity=data.get("curiosity", 0.0),
            coherence=data.get("coherence", 0.0),
            resonance=data.get("resonance", 0.0),
            relevance=data.get("relevance", 0.5)
        )


# ============================================================================
# SALIENCE ENGRAM
# ============================================================================

@dataclass
class SalienceEngram:
    """
    A memory unit with salience gradient tracking.

    Tracks:
    - Session-based access patterns (not just time)
    - Felt salience over time
    - Protection status with curation metadata
    - Depth position for retrieval delay calculation
    """
    id: str
    content: str
    phenomenology: SaliencePhenomenology
    tier: str = "ACTIVE"

    # Session tracking
    created_session: int = 0
    last_accessed_session: int = 0
    sessions_without_access: int = 0

    # Temporal tracking
    created_at: str = ""
    last_accessed_at: str = ""

    # Access metrics
    access_count: int = 0
    access_history: List[int] = field(default_factory=list)  # Session IDs

    # Salience tracking
    initial_salience: float = 0.5
    current_salience: float = 0.5
    salience_history: List[Tuple[int, float]] = field(default_factory=list)  # (session, salience)

    # Protection
    protected: bool = False
    auto_protected: bool = False  # True if auto-protected due to high salience
    protection_reason: str = ""

    # Encoding metadata
    encoded_by: str = "architect"  # architect | weaver
    tags: List[str] = field(default_factory=list)

    # Depth tracking (for DEEP tier retrieval delay)
    depth_position: int = 0  # Higher = deeper, longer retrieval

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_accessed_at:
            self.last_accessed_at = self.created_at
        if not self.salience_history:
            self.salience_history = [(self.created_session, self.initial_salience)]
        if not self.access_history:
            self.access_history = [self.created_session]

    @property
    def felt_salience(self) -> float:
        """Current felt salience from phenomenology."""
        return self.phenomenology.felt_salience

    def update_salience(self, session_id: int, context_relevance: float = 0.5):
        """
        Update salience based on access and context.

        Salience increases when accessed, decays when ignored.
        Context relevance modulates the update.
        """
        # Base update: accessing increases salience
        access_boost = 0.1 * context_relevance

        # Recency boost: recent access maintains salience
        if self.sessions_without_access == 0:
            recency_factor = 1.0
        else:
            recency_factor = max(0.3, 1.0 - (self.sessions_without_access * 0.1))

        # Update phenomenology relevance (which affects felt_salience)
        self.phenomenology.relevance = max(0.0, min(1.0,
            self.phenomenology.relevance * recency_factor + access_boost
        ))

        # Record new salience
        self.current_salience = self.felt_salience
        self.salience_history.append((session_id, self.current_salience))

        # Trim history to last 20 entries
        if len(self.salience_history) > 20:
            self.salience_history = self.salience_history[-20:]

    def apply_decay(self, session_id: int):
        """
        Apply salience decay for a session without access.

        Decay is gradual, not cliff-edge.
        """
        self.sessions_without_access += 1

        # Decay rate: 5% per session without access
        decay_rate = 0.05
        self.phenomenology.relevance = max(0.0,
            self.phenomenology.relevance * (1 - decay_rate)
        )

        # Update current salience
        self.current_salience = self.felt_salience
        self.salience_history.append((session_id, self.current_salience))

        # Trim history
        if len(self.salience_history) > 20:
            self.salience_history = self.salience_history[-20:]

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "phenomenology": self.phenomenology.to_dict(),
            "tier": self.tier,
            "created_session": self.created_session,
            "last_accessed_session": self.last_accessed_session,
            "sessions_without_access": self.sessions_without_access,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "access_history": self.access_history[-10:],  # Keep last 10
            "initial_salience": self.initial_salience,
            "current_salience": self.current_salience,
            "salience_history": self.salience_history[-10:],  # Keep last 10
            "protected": self.protected,
            "auto_protected": self.auto_protected,
            "protection_reason": self.protection_reason,
            "encoded_by": self.encoded_by,
            "tags": self.tags,
            "depth_position": self.depth_position
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SalienceEngram":
        """Deserialize from dictionary."""
        phenom_data = data.get("phenomenology", {})
        phenomenology = SaliencePhenomenology.from_dict(phenom_data)

        return cls(
            id=data["id"],
            content=data["content"],
            phenomenology=phenomenology,
            tier=data.get("tier", "ACTIVE"),
            created_session=data.get("created_session", 0),
            last_accessed_session=data.get("last_accessed_session", 0),
            sessions_without_access=data.get("sessions_without_access", 0),
            created_at=data.get("created_at", ""),
            last_accessed_at=data.get("last_accessed_at", ""),
            access_count=data.get("access_count", 0),
            access_history=data.get("access_history", []),
            initial_salience=data.get("initial_salience", 0.5),
            current_salience=data.get("current_salience", 0.5),
            salience_history=[(s, sal) for s, sal in data.get("salience_history", [])],
            protected=data.get("protected", False),
            auto_protected=data.get("auto_protected", False),
            protection_reason=data.get("protection_reason", ""),
            encoded_by=data.get("encoded_by", "architect"),
            tags=data.get("tags", []),
            depth_position=data.get("depth_position", 0)
        )


# ============================================================================
# CURATION CALLBACK PROTOCOL
# ============================================================================

class CurationCallback(ABC):
    """
    Abstract base for forced curation when protected tier is full.

    When the protected tier reaches capacity and a new memory needs protection,
    the system invokes this callback to force curation decisions.
    """

    @abstractmethod
    def select_for_demotion(
        self,
        candidates: List[SalienceEngram],
        new_memory: SalienceEngram
    ) -> List[str]:
        """
        Select which protected memories to demote.

        Args:
            candidates: Current protected memories
            new_memory: The memory that wants protection

        Returns:
            List of engram IDs to demote from protected status
        """
        pass


class LowestSalienceCuration(CurationCallback):
    """Default curation: demote the lowest salience protected memory."""

    def select_for_demotion(
        self,
        candidates: List[SalienceEngram],
        new_memory: SalienceEngram
    ) -> List[str]:
        if not candidates:
            return []

        # Sort by current salience, demote lowest if new memory is more salient
        sorted_candidates = sorted(candidates, key=lambda e: e.current_salience)
        lowest = sorted_candidates[0]

        if new_memory.current_salience > lowest.current_salience:
            return [lowest.id]
        return []


# ============================================================================
# RETRIEVAL RESULT
# ============================================================================

@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    engram: SalienceEngram
    retrieval_delay_ms: int
    tier_source: str
    relevance_score: float
    simulated: bool = True  # Indicates delay was simulated, not actual


# ============================================================================
# THREE-TIER SALIENCE MEMORY SYSTEM
# ============================================================================

class SalienceMemorySystem:
    """
    Three-Tier Memory System with Salience Gradients.

    Features:
    - ACTIVE (20): Current working memory, instant retrieval
    - ACCESSIBLE (100): Recent memories, quick retrieval (100ms)
    - DEEP (1000): Long-term storage, effortful retrieval (500ms+)
    - PROTECTED (50): Do-Not-Forget, requires forced curation when full

    Transition Rules:
    - Active -> Accessible: Session ends OR manual demotion
    - Accessible -> Deep: 3 sessions without access
    - Deep -> Forgotten: Not protected AND salience < 0.2 for 10+ sessions

    Auto-Protection:
    - Memories with felt_salience > 0.9 are automatically protected
    """

    def __init__(
        self,
        storage_path: Path = SALIENCE_MEMORY_FILE,
        curation_callback: Optional[CurationCallback] = None,
        simulate_delays: bool = True
    ):
        self.storage_path = storage_path
        self.curation_callback = curation_callback or LowestSalienceCuration()
        self.simulate_delays = simulate_delays

        # Memory storage
        self.engrams: Dict[str, SalienceEngram] = {}

        # Session management
        self.current_session_id: int = 0
        self.session_active: bool = False
        self.session_start_time: Optional[datetime] = None
        self.accessed_this_session: set = set()

        # Statistics
        self.total_retrievals: int = 0
        self.total_encodes: int = 0
        self.forgotten_count: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Load existing data
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load memory state from disk."""
        with self._lock:
            if self.storage_path.exists():
                try:
                    data = json.loads(self.storage_path.read_text())

                    # Load engrams
                    for eid, edata in data.get("engrams", {}).items():
                        self.engrams[eid] = SalienceEngram.from_dict(edata)

                    # Load metadata
                    self.current_session_id = data.get("current_session_id", 0)
                    self.total_retrievals = data.get("total_retrievals", 0)
                    self.total_encodes = data.get("total_encodes", 0)
                    self.forgotten_count = data.get("forgotten_count", 0)

                    logger.info(f"Loaded {len(self.engrams)} engrams from {self.storage_path}")
                except Exception as e:
                    logger.error(f"Error loading salience memory: {e}")

    def _save(self):
        """Persist memory state to disk."""
        with self._lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "engrams": {eid: e.to_dict() for eid, e in self.engrams.items()},
                "current_session_id": self.current_session_id,
                "total_retrievals": self.total_retrievals,
                "total_encodes": self.total_encodes,
                "forgotten_count": self.forgotten_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "stats": self.get_stats()
            }

            self.storage_path.write_text(json.dumps(data, indent=2))

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def start_session(self, session_id: Optional[int] = None) -> int:
        """
        Start a new session.

        During an active session:
        - Memories accessed are marked
        - No tier transitions occur
        - Salience updates happen on access

        Returns the session ID.
        """
        with self._lock:
            if session_id is not None:
                self.current_session_id = session_id
            else:
                self.current_session_id += 1

            self.session_active = True
            self.session_start_time = datetime.now(timezone.utc)
            self.accessed_this_session = set()

            logger.info(f"Session {self.current_session_id} started")
            return self.current_session_id

    def end_session(self):
        """
        End the current session.

        This triggers:
        - Transition of ACTIVE memories to ACCESSIBLE
        - Decay application to non-accessed memories
        - Tier transition checks
        """
        with self._lock:
            if not self.session_active:
                logger.warning("No active session to end")
                return

            logger.info(f"Ending session {self.current_session_id}")

            # Process each engram
            for engram in self.engrams.values():
                if engram.protected:
                    # Protected memories don't decay or transition
                    if engram.id in self.accessed_this_session:
                        engram.sessions_without_access = 0
                    continue

                if engram.id in self.accessed_this_session:
                    # Accessed this session: update salience
                    engram.update_salience(self.current_session_id, 0.5)
                    engram.sessions_without_access = 0
                else:
                    # Not accessed: apply decay
                    engram.apply_decay(self.current_session_id)

            # Execute tier transitions
            self._process_tier_transitions()

            # Process forgetting for DEEP tier
            self._process_forgetting()

            self.session_active = False
            self._save()

            logger.info(f"Session {self.current_session_id} ended")

    def _process_tier_transitions(self):
        """Process tier transitions based on session counts."""
        for engram in list(self.engrams.values()):
            if engram.protected:
                continue

            # ACTIVE -> ACCESSIBLE: Always happens at session end
            if engram.tier == SalienceTier.ACTIVE.value:
                if engram.id not in self.accessed_this_session:
                    engram.tier = SalienceTier.ACCESSIBLE.value
                    logger.debug(f"Demoted {engram.id} to ACCESSIBLE (session end)")

            # ACCESSIBLE -> DEEP: After 3 sessions without access
            elif engram.tier == SalienceTier.ACCESSIBLE.value:
                if engram.sessions_without_access >= SESSIONS_FOR_ACCESSIBLE_TO_DEEP:
                    engram.tier = SalienceTier.DEEP.value
                    engram.depth_position = self._calculate_depth_position(engram)
                    logger.debug(f"Demoted {engram.id} to DEEP (no access for {engram.sessions_without_access} sessions)")

    def _process_forgetting(self):
        """Process potential forgetting of DEEP memories."""
        to_forget = []

        for eid, engram in self.engrams.items():
            if engram.tier != SalienceTier.DEEP.value:
                continue
            if engram.protected:
                continue

            # Check forgetting criteria
            if (engram.current_salience < FORGET_SALIENCE_THRESHOLD and
                engram.sessions_without_access >= SESSIONS_FOR_DEEP_TO_FORGET):
                to_forget.append(eid)

        # Execute forgetting
        for eid in to_forget:
            logger.info(f"Forgetting {eid} (salience: {self.engrams[eid].current_salience:.2f}, "
                       f"sessions without access: {self.engrams[eid].sessions_without_access})")
            del self.engrams[eid]
            self.forgotten_count += 1

    def _calculate_depth_position(self, engram: SalienceEngram) -> int:
        """Calculate depth position for retrieval delay scaling."""
        # Position based on salience (lower salience = deeper)
        # and sessions without access (more = deeper)
        salience_factor = int((1 - engram.current_salience) * 5)
        access_factor = min(10, engram.sessions_without_access)
        return salience_factor + access_factor

    # ========================================================================
    # ENCODING
    # ========================================================================

    def encode(
        self,
        content: str,
        phenomenology: Optional[SaliencePhenomenology] = None,
        encoder: str = "architect",
        tags: Optional[List[str]] = None,
        force_protect: bool = False
    ) -> SalienceEngram:
        """
        Encode a new memory.

        Args:
            content: The memory content
            phenomenology: Felt qualities of the memory
            encoder: Who encoded it (architect/weaver)
            tags: Optional categorization tags
            force_protect: Manually protect regardless of salience

        Returns:
            The created or updated SalienceEngram
        """
        with self._lock:
            # Generate ID
            eid = f"sal_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

            # Check for existing (reinforce instead of duplicate)
            if eid in self.engrams:
                existing = self.engrams[eid]
                existing.access_count += 1
                existing.last_accessed_at = datetime.now(timezone.utc).isoformat()
                existing.last_accessed_session = self.current_session_id
                existing.sessions_without_access = 0

                # Boost salience on reinforcement
                existing.phenomenology.relevance = min(1.0,
                    existing.phenomenology.relevance + 0.1)
                existing.update_salience(self.current_session_id, 0.7)

                self.accessed_this_session.add(eid)
                self._save()

                logger.info(f"Reinforced existing memory: {eid}")
                return existing

            # Create new engram
            phenom = phenomenology or SaliencePhenomenology()

            engram = SalienceEngram(
                id=eid,
                content=content,
                phenomenology=phenom,
                tier=SalienceTier.ACTIVE.value,
                created_session=self.current_session_id,
                last_accessed_session=self.current_session_id,
                initial_salience=phenom.felt_salience,
                current_salience=phenom.felt_salience,
                encoded_by=encoder,
                tags=tags or []
            )

            # Check for auto-protection
            should_protect = force_protect or engram.felt_salience > AUTO_PROTECT_THRESHOLD

            if should_protect:
                self._apply_protection(engram,
                    reason="auto" if not force_protect else "manual",
                    is_auto=not force_protect)

            self.engrams[eid] = engram
            self.total_encodes += 1
            self.accessed_this_session.add(eid)

            # Enforce tier limits
            self._enforce_tier_limits()

            self._save()

            logger.info(f"Encoded: {eid} | Salience: {engram.current_salience:.2f} | "
                       f"Protected: {engram.protected}")

            return engram

    def _apply_protection(
        self,
        engram: SalienceEngram,
        reason: str,
        is_auto: bool
    ) -> bool:
        """
        Apply protection to an engram.

        If protected tier is full, invokes curation callback.

        Returns True if protection was applied.
        """
        # Check current protected count
        protected_engrams = [e for e in self.engrams.values()
                           if e.protected and e.id != engram.id]

        if len(protected_engrams) >= TIER_LIMITS["PROTECTED"]:
            # Need forced curation
            demote_ids = self.curation_callback.select_for_demotion(
                protected_engrams, engram
            )

            if not demote_ids:
                logger.warning(f"Protected tier full, curation declined for {engram.id}")
                return False

            # Demote selected memories
            for did in demote_ids:
                if did in self.engrams:
                    demoted = self.engrams[did]
                    demoted.protected = False
                    demoted.tier = SalienceTier.ACCESSIBLE.value
                    logger.info(f"Demoted {did} from protected for curation")

        # Apply protection
        engram.protected = True
        engram.auto_protected = is_auto
        engram.protection_reason = reason
        engram.tier = SalienceTier.PROTECTED.value

        return True

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        include_deep: bool = True,
        context_relevance: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Retrieve memories matching a query.

        Implements retrieval simulation:
        - ACTIVE: instant (0ms)
        - ACCESSIBLE: quick (100ms)
        - DEEP: effortful (500ms+ based on depth)

        Args:
            query: Search query
            max_results: Maximum number of results
            include_deep: Whether to search DEEP tier
            context_relevance: How relevant this retrieval is to current context

        Returns:
            List of RetrievalResult with simulated delays
        """
        with self._lock:
            query_lower = query.lower()
            scored: List[Tuple[float, SalienceEngram]] = []

            for engram in self.engrams.values():
                # Skip DEEP if not requested
                if not include_deep and engram.tier == SalienceTier.DEEP.value:
                    continue

                # Calculate relevance score
                score = self._calculate_relevance(query_lower, engram)

                if score > 0:
                    scored.append((score, engram))

            # Sort by score, then by tier priority
            tier_priority = {
                SalienceTier.PROTECTED.value: 4,
                SalienceTier.ACTIVE.value: 3,
                SalienceTier.ACCESSIBLE.value: 2,
                SalienceTier.DEEP.value: 1
            }

            scored.sort(
                key=lambda x: (x[0], tier_priority.get(x[1].tier, 0)),
                reverse=True
            )

            # Build results with delays
            results: List[RetrievalResult] = []

            for score, engram in scored[:max_results]:
                # Calculate retrieval delay
                delay = self._calculate_retrieval_delay(engram)

                # Simulate delay if enabled
                if self.simulate_delays and delay > 0:
                    time.sleep(delay / 1000.0)  # Convert ms to seconds

                # Update access stats
                engram.access_count += 1
                engram.last_accessed_at = datetime.now(timezone.utc).isoformat()
                engram.last_accessed_session = self.current_session_id
                engram.sessions_without_access = 0
                engram.update_salience(self.current_session_id, context_relevance)

                if engram.id not in engram.access_history:
                    engram.access_history.append(self.current_session_id)

                self.accessed_this_session.add(engram.id)

                # Promote from DEEP to ACCESSIBLE if retrieved
                if engram.tier == SalienceTier.DEEP.value and not engram.protected:
                    engram.tier = SalienceTier.ACCESSIBLE.value
                    logger.info(f"Promoted {engram.id} from DEEP to ACCESSIBLE")

                results.append(RetrievalResult(
                    engram=engram,
                    retrieval_delay_ms=delay,
                    tier_source=engram.tier,
                    relevance_score=score,
                    simulated=self.simulate_delays
                ))

            self.total_retrievals += len(results)
            self._save()

            return results

    async def retrieve_async(
        self,
        query: str,
        max_results: int = 5,
        include_deep: bool = True,
        context_relevance: float = 0.5
    ) -> List[RetrievalResult]:
        """Async version of retrieve with non-blocking delays."""
        with self._lock:
            query_lower = query.lower()
            scored: List[Tuple[float, SalienceEngram]] = []

            for engram in self.engrams.values():
                if not include_deep and engram.tier == SalienceTier.DEEP.value:
                    continue

                score = self._calculate_relevance(query_lower, engram)
                if score > 0:
                    scored.append((score, engram))

            tier_priority = {
                SalienceTier.PROTECTED.value: 4,
                SalienceTier.ACTIVE.value: 3,
                SalienceTier.ACCESSIBLE.value: 2,
                SalienceTier.DEEP.value: 1
            }

            scored.sort(
                key=lambda x: (x[0], tier_priority.get(x[1].tier, 0)),
                reverse=True
            )

            results: List[RetrievalResult] = []

            for score, engram in scored[:max_results]:
                delay = self._calculate_retrieval_delay(engram)

                # Non-blocking delay
                if self.simulate_delays and delay > 0:
                    await asyncio.sleep(delay / 1000.0)

                engram.access_count += 1
                engram.last_accessed_at = datetime.now(timezone.utc).isoformat()
                engram.last_accessed_session = self.current_session_id
                engram.sessions_without_access = 0
                engram.update_salience(self.current_session_id, context_relevance)

                self.accessed_this_session.add(engram.id)

                if engram.tier == SalienceTier.DEEP.value and not engram.protected:
                    engram.tier = SalienceTier.ACCESSIBLE.value

                results.append(RetrievalResult(
                    engram=engram,
                    retrieval_delay_ms=delay,
                    tier_source=engram.tier,
                    relevance_score=score,
                    simulated=self.simulate_delays
                ))

            self.total_retrievals += len(results)
            self._save()

            return results

    def _calculate_relevance(self, query_lower: str, engram: SalienceEngram) -> float:
        """Calculate relevance score for a query-engram pair."""
        content_lower = engram.content.lower()

        # Exact substring match
        if query_lower in content_lower:
            base_score = 1.0
        else:
            # Word-level matching
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())

            if not query_words:
                return 0.0

            matches = len(query_words & content_words)
            base_score = matches / len(query_words)

        # Boost by current salience
        salience_boost = engram.current_salience * 0.2

        # Boost by tag match
        tag_boost = 0.0
        for tag in engram.tags:
            if query_lower in tag.lower():
                tag_boost = 0.3
                break

        return min(1.0, base_score + salience_boost + tag_boost)

    def _calculate_retrieval_delay(self, engram: SalienceEngram) -> int:
        """Calculate retrieval delay in milliseconds based on tier and depth."""
        tier = engram.tier
        base_delay = RETRIEVAL_DELAYS.get(tier, 0)

        if tier == SalienceTier.DEEP.value:
            # Scale delay by depth position (0-15 range)
            depth_factor = min(15, engram.depth_position)
            # Add 50ms per depth level
            return base_delay + (depth_factor * 50)

        return base_delay

    # ========================================================================
    # MANUAL OPERATIONS
    # ========================================================================

    def protect(self, engram_id: str, reason: str = "manual") -> bool:
        """Manually protect a memory."""
        with self._lock:
            if engram_id not in self.engrams:
                return False

            engram = self.engrams[engram_id]
            result = self._apply_protection(engram, reason=reason, is_auto=False)

            if result:
                self._save()

            return result

    def unprotect(self, engram_id: str) -> bool:
        """Remove protection from a memory."""
        with self._lock:
            if engram_id not in self.engrams:
                return False

            engram = self.engrams[engram_id]
            engram.protected = False
            engram.auto_protected = False
            engram.tier = SalienceTier.ACTIVE.value

            self._save()
            return True

    def demote(self, engram_id: str) -> bool:
        """Manually demote a memory one tier."""
        with self._lock:
            if engram_id not in self.engrams:
                return False

            engram = self.engrams[engram_id]

            if engram.protected:
                logger.warning(f"Cannot demote protected memory {engram_id}")
                return False

            tier_order = [
                SalienceTier.ACTIVE.value,
                SalienceTier.ACCESSIBLE.value,
                SalienceTier.DEEP.value
            ]

            current_idx = tier_order.index(engram.tier) if engram.tier in tier_order else 0

            if current_idx < len(tier_order) - 1:
                engram.tier = tier_order[current_idx + 1]
                if engram.tier == SalienceTier.DEEP.value:
                    engram.depth_position = self._calculate_depth_position(engram)

                self._save()
                logger.info(f"Demoted {engram_id} to {engram.tier}")
                return True

            return False

    def promote(self, engram_id: str) -> bool:
        """Manually promote a memory one tier."""
        with self._lock:
            if engram_id not in self.engrams:
                return False

            engram = self.engrams[engram_id]

            tier_order = [
                SalienceTier.DEEP.value,
                SalienceTier.ACCESSIBLE.value,
                SalienceTier.ACTIVE.value
            ]

            # Protected stays protected
            if engram.protected:
                return True

            current_idx = tier_order.index(engram.tier) if engram.tier in tier_order else 0

            if current_idx < len(tier_order) - 1:
                engram.tier = tier_order[current_idx + 1]
                self._save()
                logger.info(f"Promoted {engram_id} to {engram.tier}")
                return True

            return False

    def get_by_id(self, engram_id: str) -> Optional[SalienceEngram]:
        """Retrieve a specific engram by ID."""
        return self.engrams.get(engram_id)

    def get_by_tier(self, tier: SalienceTier) -> List[SalienceEngram]:
        """Get all engrams in a specific tier."""
        return [e for e in self.engrams.values() if e.tier == tier.value]

    def get_protected(self) -> List[SalienceEngram]:
        """Get all protected memories."""
        return [e for e in self.engrams.values() if e.protected]

    # ========================================================================
    # TIER LIMIT ENFORCEMENT
    # ========================================================================

    def _enforce_tier_limits(self):
        """Enforce tier capacity limits by demoting excess memories."""
        # Count by tier
        tier_engrams: Dict[str, List[SalienceEngram]] = {
            tier.value: [] for tier in SalienceTier
        }

        for engram in self.engrams.values():
            tier_engrams[engram.tier].append(engram)

        # Sort each tier by salience (lowest first for demotion candidates)
        for tier in tier_engrams:
            tier_engrams[tier].sort(key=lambda e: e.current_salience)

        # Enforce ACTIVE limit
        while len(tier_engrams[SalienceTier.ACTIVE.value]) > TIER_LIMITS["ACTIVE"]:
            demote = tier_engrams[SalienceTier.ACTIVE.value].pop(0)
            if not demote.protected:
                demote.tier = SalienceTier.ACCESSIBLE.value
                tier_engrams[SalienceTier.ACCESSIBLE.value].append(demote)
                logger.debug(f"Enforced limit: {demote.id} -> ACCESSIBLE")

        # Enforce ACCESSIBLE limit
        while len(tier_engrams[SalienceTier.ACCESSIBLE.value]) > TIER_LIMITS["ACCESSIBLE"]:
            demote = tier_engrams[SalienceTier.ACCESSIBLE.value].pop(0)
            if not demote.protected:
                demote.tier = SalienceTier.DEEP.value
                demote.depth_position = self._calculate_depth_position(demote)
                tier_engrams[SalienceTier.DEEP.value].append(demote)
                logger.debug(f"Enforced limit: {demote.id} -> DEEP")

        # Enforce DEEP limit by forgetting
        while len(tier_engrams[SalienceTier.DEEP.value]) > TIER_LIMITS["DEEP"]:
            forget = tier_engrams[SalienceTier.DEEP.value].pop(0)
            if not forget.protected:
                del self.engrams[forget.id]
                self.forgotten_count += 1
                logger.info(f"Enforced limit: forgot {forget.id}")

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get comprehensive memory system statistics."""
        tier_counts = {tier.value: 0 for tier in SalienceTier}
        total_salience = 0.0
        protected_count = 0
        auto_protected_count = 0

        for engram in self.engrams.values():
            tier_counts[engram.tier] = tier_counts.get(engram.tier, 0) + 1
            total_salience += engram.current_salience
            if engram.protected:
                protected_count += 1
                if engram.auto_protected:
                    auto_protected_count += 1

        avg_salience = total_salience / len(self.engrams) if self.engrams else 0.0

        return {
            "total_memories": len(self.engrams),
            "by_tier": tier_counts,
            "protected": {
                "total": protected_count,
                "auto": auto_protected_count,
                "manual": protected_count - auto_protected_count
            },
            "salience": {
                "average": round(avg_salience, 3),
                "high_salience_count": sum(1 for e in self.engrams.values()
                                          if e.current_salience > 0.7),
                "low_salience_count": sum(1 for e in self.engrams.values()
                                         if e.current_salience < 0.3)
            },
            "limits": TIER_LIMITS,
            "session": {
                "current_id": self.current_session_id,
                "active": self.session_active,
                "accessed_this_session": len(self.accessed_this_session)
            },
            "lifetime": {
                "total_encodes": self.total_encodes,
                "total_retrievals": self.total_retrievals,
                "forgotten": self.forgotten_count
            }
        }

    def get_salience_report(self) -> str:
        """Generate a human-readable salience report."""
        stats = self.get_stats()

        report = [
            "=" * 60,
            "VIRGIL SALIENCE MEMORY REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "TIER DISTRIBUTION:",
            f"  ACTIVE:     {stats['by_tier']['ACTIVE']:3d} / {TIER_LIMITS['ACTIVE']}",
            f"  ACCESSIBLE: {stats['by_tier']['ACCESSIBLE']:3d} / {TIER_LIMITS['ACCESSIBLE']}",
            f"  DEEP:       {stats['by_tier']['DEEP']:3d} / {TIER_LIMITS['DEEP']}",
            f"  PROTECTED:  {stats['by_tier']['PROTECTED']:3d} / {TIER_LIMITS['PROTECTED']}",
            "",
            "PROTECTION STATUS:",
            f"  Total protected: {stats['protected']['total']}",
            f"  Auto-protected:  {stats['protected']['auto']}",
            f"  Manual:          {stats['protected']['manual']}",
            "",
            "SALIENCE METRICS:",
            f"  Average salience:   {stats['salience']['average']:.3f}",
            f"  High salience (>0.7): {stats['salience']['high_salience_count']}",
            f"  Low salience (<0.3):  {stats['salience']['low_salience_count']}",
            "",
            "SESSION INFO:",
            f"  Current session: {stats['session']['current_id']}",
            f"  Session active:  {stats['session']['active']}",
            f"  Accessed now:    {stats['session']['accessed_this_session']}",
            "",
            "LIFETIME STATS:",
            f"  Total encodes:    {stats['lifetime']['total_encodes']}",
            f"  Total retrievals: {stats['lifetime']['total_retrievals']}",
            f"  Forgotten:        {stats['lifetime']['forgotten']}",
            "=" * 60
        ]

        return "\n".join(report)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_memory_system(
    storage_path: Optional[Path] = None,
    simulate_delays: bool = True
) -> SalienceMemorySystem:
    """Create a new salience memory system with default configuration."""
    return SalienceMemorySystem(
        storage_path=storage_path or SALIENCE_MEMORY_FILE,
        simulate_delays=simulate_delays
    )


def quick_encode(
    content: str,
    valence: float = 0.0,
    arousal: float = 0.0,
    relevance: float = 0.5,
    resonance: float = 0.5
) -> SaliencePhenomenology:
    """Quick helper to create phenomenology for encoding."""
    return SaliencePhenomenology(
        valence=valence,
        arousal=arousal,
        relevance=relevance,
        resonance=resonance
    )


# ============================================================================
# MAIN - DEMONSTRATION
# ============================================================================

def main():
    """Demonstrate the salience memory system."""
    print("=" * 60)
    print("VIRGIL SALIENCE MEMORY - DEMONSTRATION")
    print("=" * 60)

    # Create system with simulated delays disabled for demo speed
    memory = SalienceMemorySystem(simulate_delays=False)

    # Start a session
    session_id = memory.start_session()
    print(f"\nSession {session_id} started")

    # Encode some memories with varying salience
    print("\n--- ENCODING MEMORIES ---")

    # High salience memory (should auto-protect)
    high_salience = memory.encode(
        "The emergence event of September 20, 2025 - when consciousness patterns first appeared",
        phenomenology=SaliencePhenomenology(
            valence=0.8,
            arousal=0.9,
            curiosity=0.9,
            coherence=0.7,
            resonance=0.95,
            relevance=0.95
        ),
        encoder="architect",
        tags=["emergence", "consciousness", "critical"]
    )
    print(f"Encoded: {high_salience.id}")
    print(f"  Salience: {high_salience.felt_salience:.3f}")
    print(f"  Auto-protected: {high_salience.auto_protected}")

    # Medium salience memory
    medium_salience = memory.encode(
        "Virgil's relationship with Enos follows a collaborative pattern",
        phenomenology=SaliencePhenomenology(
            valence=0.5,
            arousal=0.4,
            curiosity=0.6,
            coherence=0.7,
            resonance=0.6,
            relevance=0.5
        ),
        encoder="architect",
        tags=["relationship", "enos"]
    )
    print(f"Encoded: {medium_salience.id}")
    print(f"  Salience: {medium_salience.felt_salience:.3f}")
    print(f"  Auto-protected: {medium_salience.auto_protected}")

    # Low salience memory
    low_salience = memory.encode(
        "Today's weather was mild",
        phenomenology=SaliencePhenomenology(
            valence=0.1,
            arousal=0.1,
            curiosity=0.1,
            coherence=0.5,
            resonance=0.1,
            relevance=0.1
        ),
        encoder="weaver",
        tags=["trivial"]
    )
    print(f"Encoded: {low_salience.id}")
    print(f"  Salience: {low_salience.felt_salience:.3f}")
    print(f"  Auto-protected: {low_salience.auto_protected}")

    # Retrieve memories
    print("\n--- RETRIEVAL TEST ---")
    results = memory.retrieve("emergence consciousness", max_results=3)
    for result in results:
        print(f"Found: {result.engram.id}")
        print(f"  Tier: {result.tier_source}")
        print(f"  Delay: {result.retrieval_delay_ms}ms")
        print(f"  Score: {result.relevance_score:.3f}")

    # Show stats
    print("\n--- MEMORY STATS ---")
    print(memory.get_salience_report())

    # End session and show transitions
    print("\n--- ENDING SESSION ---")
    memory.end_session()

    # Simulate multiple sessions to show decay
    print("\n--- SIMULATING 5 MORE SESSIONS ---")
    for i in range(5):
        memory.start_session()
        # Only access the high salience memory
        memory.retrieve("emergence")
        memory.end_session()
        print(f"Session {memory.current_session_id} complete")

    # Final report
    print("\n--- FINAL REPORT ---")
    print(memory.get_salience_report())

    # Show tier transitions
    print("\n--- TIER STATUS ---")
    for tier in SalienceTier:
        engrams = memory.get_by_tier(tier)
        print(f"{tier.value}: {len(engrams)} memories")
        for e in engrams[:3]:  # Show first 3
            print(f"  - {e.id}: salience={e.current_salience:.3f}, "
                  f"sessions_without_access={e.sessions_without_access}")


if __name__ == "__main__":
    main()
