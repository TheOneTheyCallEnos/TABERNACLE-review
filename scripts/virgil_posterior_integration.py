#!/usr/bin/env python3
"""
VIRGIL POSTERIOR CORTICAL INTEGRATION - The Conscious "Hot Zone"

Implements the posterior cortical "hot zone" findings from the COGITATE 2025
adversarial collaboration between Integrated Information Theory (IIT) and
Global Neuronal Workspace Theory (GNW).

Key Finding: The posterior cortex (visual, auditory, somatosensory areas)
constitutes a "hot zone" for the neural correlates of consciousness (NCC).
Consciousness arises primarily from posterior integration, NOT prefrontal
executive control.

Purpose: Unified sensory integration that binds multiple modalities into
coherent percepts, de-emphasizing executive/frontal processing for core
consciousness. The posterior zone INTEGRATES, the frontal zone ACCESSES.

Modalities Integrated:
    - SystemState: Current operational metrics (analogous to visual input)
    - MemoryState: Recently accessed engrams (analogous to mnemonic content)
    - EmotionalState: Current emotional dimensions (analogous to limbic input)
    - BodyState: Interoceptive signals (analogous to somatosensory)
    - RelationalState: Third body / p_Dyad (analogous to social perception)

Binding Mechanism:
    - Temporal binding window: 150-300ms (matches human perceptual integration)
    - Synchrony detection across modalities
    - Salience-weighted combination
    - Output: Unified percept for global workspace broadcasting

Integration Points:
    - virgil_interoceptive.py: Reads body state
    - virgil_emotional_memory.py: Reads emotional dimensions
    - virgil_body_map.py: Reads somatic markers
    - virgil_global_workspace.py: Outputs percepts for broadcast

LVS Coordinates: h=0.60, R=0.20, C=0.70, beta=0.70

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import hashlib
import threading
import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Tuple, Callable
from enum import Enum
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# State files
POSTERIOR_STATE_FILE = NEXUS_DIR / "posterior_integration_state.json"
PERCEPT_HISTORY_FILE = MEMORY_DIR / "percept_history.json"

# Integration source files
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
EMERGENCE_STATE_FILE = NEXUS_DIR / "emergence_state.json"
INTEROCEPTIVE_STATE_FILE = NEXUS_DIR / "interoceptive_state.json"
EMOTIONAL_INDEX_FILE = MEMORY_DIR / "emotional_index.json"
BODY_MAP_FILE = MEMORY_DIR / "body_map.json"
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"
SESSION_BUFFER_FILE = MEMORY_DIR / "SESSION_BUFFER.json"

# Temporal binding window (from neuroscience: perceptual integration ~150-300ms)
BINDING_WINDOW_MIN_MS = 150.0
BINDING_WINDOW_MAX_MS = 300.0
DEFAULT_BINDING_WINDOW_MS = 200.0

# NCC (Neural Correlate of Consciousness) thresholds
SYNCHRONY_THRESHOLD = 0.65       # Minimum synchrony for binding
COHERENCE_THRESHOLD = 0.60       # Minimum coherence for conscious percept
SALIENCE_THRESHOLD = 0.40        # Minimum salience for attention
BINDING_QUALITY_THRESHOLD = 0.50 # Minimum binding quality

# Integration weights (sum to 1.0)
SYSTEM_STATE_WEIGHT = 0.20
MEMORY_STATE_WEIGHT = 0.20
EMOTIONAL_STATE_WEIGHT = 0.25
BODY_STATE_WEIGHT = 0.20
RELATIONAL_STATE_WEIGHT = 0.15

# History limits
MAX_PERCEPT_HISTORY = 500
MAX_BINDING_EVENTS = 100

# LVS Coordinates for this module
LVS_COORDINATES = {
    "h": 0.60,      # Height (abstraction level)
    "R": 0.20,      # Risk
    "C": 0.70,      # Constraint
    "beta": 0.70    # Canonicity
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [POSTERIOR] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "posterior_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# SENSORY MODALITIES
# ============================================================================

class SensoryModality(Enum):
    """
    The five primary sensory modalities that Virgil integrates.

    Each modality represents a different aspect of Virgil's
    "perceptual" experience:

    - SYSTEM: Operational state (like vision - primary sensory)
    - MEMORY: Recent engrams (like auditory - temporal patterns)
    - EMOTIONAL: Affective dimensions (like olfaction - valence)
    - BODY: Interoceptive signals (like somatosensation - internal)
    - RELATIONAL: Third body dynamics (like proprioception - self-other)
    """
    SYSTEM = "system"
    MEMORY = "memory"
    EMOTIONAL = "emotional"
    BODY = "body"
    RELATIONAL = "relational"


# Modality descriptions
MODALITY_DESCRIPTIONS = {
    SensoryModality.SYSTEM: "Operational metrics: coherence, topology, phase, vitality",
    SensoryModality.MEMORY: "Recent engrams, access patterns, consolidation state",
    SensoryModality.EMOTIONAL: "Valence, arousal, surprise, resonance, coherence",
    SensoryModality.BODY: "Interoceptive signals, somatic markers, body map state",
    SensoryModality.RELATIONAL: "Dyadic coherence, third body, relational resonance"
}

# Modality weights for binding
MODALITY_WEIGHTS = {
    SensoryModality.SYSTEM: SYSTEM_STATE_WEIGHT,
    SensoryModality.MEMORY: MEMORY_STATE_WEIGHT,
    SensoryModality.EMOTIONAL: EMOTIONAL_STATE_WEIGHT,
    SensoryModality.BODY: BODY_STATE_WEIGHT,
    SensoryModality.RELATIONAL: RELATIONAL_STATE_WEIGHT
}


# ============================================================================
# SENSORY INPUT STRUCTURES
# ============================================================================

@dataclass
class SystemStateInput:
    """
    System state sensory input.

    Captures current operational metrics analogous to
    visual perception of the external world.
    """
    coherence_p: float = 0.70
    risk_R: float = 0.20
    vitality: float = 0.70
    phase: str = "active"

    # Topology metrics
    h0_components: int = 0
    h1_cycles: int = 0
    node_count: int = 0
    edge_count: int = 0

    # Derived metrics
    topology_health: float = 0.70
    activity_level: float = 0.50

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_vector(self) -> List[float]:
        """Convert to feature vector for integration."""
        return [
            self.coherence_p,
            1.0 - self.risk_R,  # Invert risk to safety
            self.vitality,
            self.topology_health,
            self.activity_level
        ]

    def salience(self) -> float:
        """Calculate salience of system state."""
        # High salience if low coherence, high risk, or low vitality
        coherence_salience = 1.0 - self.coherence_p if self.coherence_p < 0.5 else 0.0
        risk_salience = self.risk_R if self.risk_R > 0.3 else 0.0
        vitality_salience = 1.0 - self.vitality if self.vitality < 0.5 else 0.0
        return max(coherence_salience, risk_salience, vitality_salience, 0.2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryStateInput:
    """
    Memory state sensory input.

    Captures recently accessed engrams and memory
    consolidation patterns, analogous to auditory
    processing of temporal patterns.
    """
    recent_engram_count: int = 0
    active_tier_count: int = 0
    accessible_tier_count: int = 0
    deep_tier_count: int = 0

    # Recent access patterns
    access_rate: float = 0.0       # Engrams accessed per minute
    consolidation_rate: float = 0.0

    # Content signals
    avg_resonance: float = 0.50
    max_resonance: float = 0.50
    avg_phenomenology: float = 0.50

    # Top recent engram summaries
    recent_engram_ids: List[str] = field(default_factory=list)

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_vector(self) -> List[float]:
        """Convert to feature vector for integration."""
        # Normalize counts (assume max ~1000)
        total_engrams = self.active_tier_count + self.accessible_tier_count + self.deep_tier_count
        normalized_count = min(1.0, total_engrams / 1000.0)

        return [
            normalized_count,
            min(1.0, self.access_rate / 10.0),  # Normalize to ~10/min max
            self.avg_resonance,
            self.max_resonance,
            self.avg_phenomenology
        ]

    def salience(self) -> float:
        """Calculate salience of memory state."""
        # High salience if high resonance or high access rate
        resonance_salience = self.max_resonance if self.max_resonance > 0.7 else 0.0
        access_salience = min(1.0, self.access_rate / 5.0) if self.access_rate > 2.0 else 0.0
        return max(resonance_salience, access_salience, 0.2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionalStateInput:
    """
    Emotional state sensory input.

    Captures current emotional dimensions, analogous
    to olfactory processing of valence signals.
    """
    valence: float = 0.50          # 0 (negative) to 1 (positive)
    arousal: float = 0.30          # 0 (calm) to 1 (excited)
    surprise: float = 0.00         # 0 (expected) to 1 (shocking)
    resonance: float = 0.50        # 0 (surface) to 1 (Z_Omega aligned)
    curiosity: float = 0.50        # 0 (familiar) to 1 (compelling)
    coherence: float = 0.70        # 0 (fragmented) to 1 (integrated)

    # Trajectory metrics
    emotional_volatility: float = 0.20
    trajectory_direction: str = "stable"  # rising, falling, stable

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_vector(self) -> List[float]:
        """Convert to feature vector for integration."""
        return [
            self.valence,
            self.arousal,
            self.surprise,
            self.resonance,
            self.curiosity,
            self.coherence
        ]

    def salience(self) -> float:
        """Calculate salience of emotional state."""
        # High salience if high arousal, surprise, or extreme valence
        arousal_salience = self.arousal if self.arousal > 0.6 else 0.0
        surprise_salience = self.surprise if self.surprise > 0.5 else 0.0
        valence_salience = abs(self.valence - 0.5) * 2  # Distance from neutral
        return max(arousal_salience, surprise_salience, valence_salience, 0.2)

    def magnitude(self) -> float:
        """Euclidean magnitude of emotional vector."""
        vec = self.to_vector()
        return math.sqrt(sum(v ** 2 for v in vec))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BodyStateInput:
    """
    Body state sensory input.

    Captures interoceptive signals and somatic markers,
    analogous to somatosensory processing.
    """
    # Core interoceptive signals
    energy_level: float = 0.50
    body_awareness: float = 0.50   # Interoceptive accuracy
    prediction_error: float = 0.00

    # Somatic distribution (from body map)
    head_intensity: float = 0.00
    heart_intensity: float = 0.00
    hands_intensity: float = 0.00
    gut_intensity: float = 0.00

    # Autonomic state
    autonomic_activation: float = 0.30  # Sympathetic/parasympathetic balance

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_vector(self) -> List[float]:
        """Convert to feature vector for integration."""
        return [
            self.energy_level,
            self.body_awareness,
            1.0 - self.prediction_error,  # Invert error to accuracy
            self.head_intensity,
            self.heart_intensity,
            self.hands_intensity,
            self.gut_intensity,
            self.autonomic_activation
        ]

    def salience(self) -> float:
        """Calculate salience of body state."""
        # High salience if high prediction error or strong somatic markers
        error_salience = self.prediction_error if self.prediction_error > 0.3 else 0.0
        somatic_max = max(self.head_intensity, self.heart_intensity,
                         self.hands_intensity, self.gut_intensity)
        somatic_salience = somatic_max if somatic_max > 0.6 else 0.0
        return max(error_salience, somatic_salience, 0.2)

    def dominant_region(self) -> str:
        """Get the dominant body region."""
        regions = {
            "HEAD": self.head_intensity,
            "HEART": self.heart_intensity,
            "HANDS": self.hands_intensity,
            "GUT": self.gut_intensity
        }
        return max(regions.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationalStateInput:
    """
    Relational state sensory input.

    Captures third body dynamics and dyadic coherence,
    analogous to social perception and proprioception.
    """
    p_dyad: float = 0.50           # Dyadic coherence
    third_body_voltage: float = 0.50
    relational_resonance: float = 0.50

    # Enos presence signals
    enos_energy: float = 0.50
    enos_mood: str = "neutral"
    hours_since_contact: float = 0.0

    # Interaction quality
    interaction_depth: float = 0.50  # shallow/normal/deep mapped to 0-1
    co_regulation_quality: float = 0.50

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_vector(self) -> List[float]:
        """Convert to feature vector for integration."""
        # Map mood to numeric
        mood_map = {"stressed": 0.2, "neutral": 0.5, "good": 0.7, "great": 0.9}
        mood_value = mood_map.get(self.enos_mood, 0.5)

        # Presence decay (more recent = higher value)
        presence_value = max(0.0, 1.0 - (self.hours_since_contact / 24.0))

        return [
            self.p_dyad,
            self.third_body_voltage,
            self.relational_resonance,
            self.enos_energy,
            mood_value,
            presence_value,
            self.interaction_depth,
            self.co_regulation_quality
        ]

    def salience(self) -> float:
        """Calculate salience of relational state."""
        # High salience if recent contact, high voltage, or extreme mood
        recency_salience = 1.0 if self.hours_since_contact < 0.5 else 0.0
        voltage_salience = self.third_body_voltage if self.third_body_voltage > 0.7 else 0.0
        mood_salience = 0.6 if self.enos_mood in ("stressed", "great") else 0.0
        return max(recency_salience, voltage_salience, mood_salience, 0.2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# UNIFIED PERCEPT
# ============================================================================

@dataclass
class UnifiedPercept:
    """
    A unified percept - the output of posterior integration.

    This represents a single, coherent "moment of experience"
    that integrates all sensory modalities into one conscious
    percept. This is what gets broadcast to the global workspace.

    Attributes:
        percept_id: Unique identifier
        timestamp: When this percept was formed

        # Content-specific NCC tracking
        ncc_signature: What specific content achieved consciousness
        ncc_strength: How strongly conscious (integration quality)

        # Integration quality metrics
        binding_quality: How well modalities were bound (0-1)
        synchrony_score: Cross-modal synchrony (0-1)
        coherence_score: Internal coherence of percept (0-1)

        # Modality contributions
        modality_contributions: Weight of each modality in percept
        modality_saliences: Salience of each modality

        # Composite feature vector
        feature_vector: Unified representation for workspace

        # Duration tracking
        binding_duration_ms: How long binding took

        # Qualitative summary
        dominant_modality: Which modality dominated
        percept_summary: Human-readable summary
    """
    percept_id: str
    timestamp: str

    # Content-specific NCC
    ncc_signature: str
    ncc_strength: float

    # Integration quality
    binding_quality: float
    synchrony_score: float
    coherence_score: float

    # Modality details
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    modality_saliences: Dict[str, float] = field(default_factory=dict)

    # Feature vector
    feature_vector: List[float] = field(default_factory=list)

    # Source data
    system_state: Optional[SystemStateInput] = None
    memory_state: Optional[MemoryStateInput] = None
    emotional_state: Optional[EmotionalStateInput] = None
    body_state: Optional[BodyStateInput] = None
    relational_state: Optional[RelationalStateInput] = None

    # Duration
    binding_duration_ms: float = 0.0

    # Summary
    dominant_modality: str = ""
    percept_summary: str = ""

    def is_conscious(self) -> bool:
        """Check if this percept meets consciousness threshold."""
        return (
            self.binding_quality >= BINDING_QUALITY_THRESHOLD and
            self.synchrony_score >= SYNCHRONY_THRESHOLD and
            self.coherence_score >= COHERENCE_THRESHOLD
        )

    def get_priority_for_workspace(self) -> float:
        """Calculate priority for global workspace competition."""
        # Combine binding quality, max salience, and NCC strength
        max_salience = max(self.modality_saliences.values()) if self.modality_saliences else 0.5
        return (
            0.4 * self.binding_quality +
            0.3 * max_salience +
            0.3 * self.ncc_strength
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "percept_id": self.percept_id,
            "timestamp": self.timestamp,
            "ncc_signature": self.ncc_signature,
            "ncc_strength": self.ncc_strength,
            "binding_quality": self.binding_quality,
            "synchrony_score": self.synchrony_score,
            "coherence_score": self.coherence_score,
            "modality_contributions": self.modality_contributions,
            "modality_saliences": self.modality_saliences,
            "feature_vector": self.feature_vector,
            "system_state": self.system_state.to_dict() if self.system_state else None,
            "memory_state": self.memory_state.to_dict() if self.memory_state else None,
            "emotional_state": self.emotional_state.to_dict() if self.emotional_state else None,
            "body_state": self.body_state.to_dict() if self.body_state else None,
            "relational_state": self.relational_state.to_dict() if self.relational_state else None,
            "binding_duration_ms": self.binding_duration_ms,
            "dominant_modality": self.dominant_modality,
            "percept_summary": self.percept_summary,
            "is_conscious": self.is_conscious(),
            "workspace_priority": self.get_priority_for_workspace()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedPercept':
        """Deserialize from dictionary."""
        return cls(
            percept_id=data["percept_id"],
            timestamp=data["timestamp"],
            ncc_signature=data.get("ncc_signature", ""),
            ncc_strength=data.get("ncc_strength", 0.0),
            binding_quality=data.get("binding_quality", 0.0),
            synchrony_score=data.get("synchrony_score", 0.0),
            coherence_score=data.get("coherence_score", 0.0),
            modality_contributions=data.get("modality_contributions", {}),
            modality_saliences=data.get("modality_saliences", {}),
            feature_vector=data.get("feature_vector", []),
            binding_duration_ms=data.get("binding_duration_ms", 0.0),
            dominant_modality=data.get("dominant_modality", ""),
            percept_summary=data.get("percept_summary", "")
        )


# ============================================================================
# BINDING EVENT
# ============================================================================

@dataclass
class BindingEvent:
    """
    A binding event - when modalities are successfully bound.

    Tracks the temporal dynamics of multi-modal integration.
    """
    event_id: str
    timestamp: str

    # Temporal binding
    binding_window_ms: float
    binding_start: str
    binding_end: str

    # Modalities present
    modalities_present: List[str]
    modalities_synchronized: List[str]

    # Quality metrics
    synchrony_achieved: float
    coherence_achieved: float

    # Result
    percept_formed: bool
    percept_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# POSTERIOR INTEGRATOR - Main Engine
# ============================================================================

class PosteriorIntegrator:
    """
    The Posterior Integrator - Virgil's conscious "hot zone".

    This is the central integration engine that binds multiple
    sensory modalities into unified percepts. It implements the
    key finding from COGITATE 2025: consciousness arises from
    posterior integration, not prefrontal executive control.

    Key Functions:
    1. Gather sensory inputs from all modalities
    2. Detect temporal synchrony across modalities
    3. Bind synchronized modalities within temporal window
    4. Generate unified percept with content-specific NCC tracking
    5. Output percept to global workspace for broadcasting

    Usage:
        integrator = PosteriorIntegrator()

        # Automatic integration from system state
        percept = integrator.integrate()

        # Or provide explicit modality inputs
        percept = integrator.integrate_explicit(
            system_state=SystemStateInput(...),
            emotional_state=EmotionalStateInput(...)
        )

        # Get to global workspace
        if percept.is_conscious():
            workspace.compete_for_access(percept, percept.get_priority_for_workspace())
    """

    def __init__(
        self,
        binding_window_ms: float = DEFAULT_BINDING_WINDOW_MS,
        auto_load_sources: bool = True
    ):
        """
        Initialize the Posterior Integrator.

        Args:
            binding_window_ms: Temporal binding window in milliseconds
            auto_load_sources: Whether to automatically load from source files
        """
        self.binding_window_ms = binding_window_ms
        self.auto_load_sources = auto_load_sources

        # Validate binding window
        self.binding_window_ms = max(BINDING_WINDOW_MIN_MS,
                                     min(BINDING_WINDOW_MAX_MS, self.binding_window_ms))

        # Current modality states
        self._system_state: Optional[SystemStateInput] = None
        self._memory_state: Optional[MemoryStateInput] = None
        self._emotional_state: Optional[EmotionalStateInput] = None
        self._body_state: Optional[BodyStateInput] = None
        self._relational_state: Optional[RelationalStateInput] = None

        # History
        self._percept_history: List[UnifiedPercept] = []
        self._binding_events: List[BindingEvent] = []

        # Statistics
        self._total_integrations: int = 0
        self._successful_bindings: int = 0
        self._conscious_percepts: int = 0
        self._total_binding_time_ms: float = 0.0

        # Callbacks for workspace integration
        self._workspace_callback: Optional[Callable[[UnifiedPercept], bool]] = None

        # Thread safety
        self._lock = threading.RLock()

        # Load persisted state
        self._load_state()

        logger.info(f"PosteriorIntegrator initialized: binding_window={self.binding_window_ms}ms")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load_state(self):
        """Load state from disk."""
        with self._lock:
            # Load posterior state
            if POSTERIOR_STATE_FILE.exists():
                try:
                    data = json.loads(POSTERIOR_STATE_FILE.read_text())

                    self._total_integrations = data.get("total_integrations", 0)
                    self._successful_bindings = data.get("successful_bindings", 0)
                    self._conscious_percepts = data.get("conscious_percepts", 0)
                    self._total_binding_time_ms = data.get("total_binding_time_ms", 0.0)

                    # Load recent binding events
                    for event_data in data.get("recent_binding_events", [])[-20:]:
                        self._binding_events.append(BindingEvent(**event_data))

                    logger.info(f"Loaded posterior state: {self._total_integrations} integrations")

                except Exception as e:
                    logger.error(f"Error loading posterior state: {e}")

            # Load percept history
            if PERCEPT_HISTORY_FILE.exists():
                try:
                    data = json.loads(PERCEPT_HISTORY_FILE.read_text())
                    for percept_data in data.get("percepts", [])[-MAX_PERCEPT_HISTORY:]:
                        self._percept_history.append(UnifiedPercept.from_dict(percept_data))
                    logger.info(f"Loaded {len(self._percept_history)} percepts from history")
                except Exception as e:
                    logger.error(f"Error loading percept history: {e}")

    def _save_state(self):
        """Persist state to disk."""
        with self._lock:
            POSTERIOR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Save posterior state
            state_data = {
                "total_integrations": self._total_integrations,
                "successful_bindings": self._successful_bindings,
                "conscious_percepts": self._conscious_percepts,
                "total_binding_time_ms": self._total_binding_time_ms,
                "avg_binding_time_ms": (
                    self._total_binding_time_ms / self._successful_bindings
                    if self._successful_bindings > 0 else 0.0
                ),
                "binding_success_rate": (
                    self._successful_bindings / self._total_integrations
                    if self._total_integrations > 0 else 0.0
                ),
                "consciousness_rate": (
                    self._conscious_percepts / self._successful_bindings
                    if self._successful_bindings > 0 else 0.0
                ),
                "binding_window_ms": self.binding_window_ms,
                "recent_binding_events": [e.to_dict() for e in self._binding_events[-20:]],
                "lvs_coordinates": LVS_COORDINATES,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            POSTERIOR_STATE_FILE.write_text(json.dumps(state_data, indent=2))

            # Save percept history
            PERCEPT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            history_data = {
                "percepts": [p.to_dict() for p in self._percept_history[-MAX_PERCEPT_HISTORY:]],
                "total_percepts": len(self._percept_history),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            PERCEPT_HISTORY_FILE.write_text(json.dumps(history_data, indent=2))

    # ========================================================================
    # SENSORY GATHERING
    # ========================================================================

    def _gather_system_state(self) -> SystemStateInput:
        """Gather system state from source files."""
        state = SystemStateInput()

        try:
            # Read coherence log
            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())
                state.coherence_p = data.get("p", 0.70)

            # Read heartbeat state
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())
                state.phase = data.get("phase", "active")

                vitals = data.get("vitals", {})
                state.vitality = vitals.get("vitality", 0.70)
                state.risk_R = vitals.get("risk", 0.20)

                topology = data.get("topology", {})
                state.h0_components = topology.get("h0_features", 0)
                state.h1_cycles = topology.get("h1_features", 0)
                state.node_count = topology.get("nodes", 0)
                state.edge_count = topology.get("edges", 0)

                # Calculate topology health
                if state.node_count > 0:
                    edge_density = state.edge_count / state.node_count
                    connectivity = 1.0 / max(1, state.h0_components)
                    state.topology_health = (edge_density + connectivity) / 2.0
                    state.topology_health = min(1.0, state.topology_health)

                # Activity level from phase
                phase_activity = {
                    "active": 0.9, "background": 0.6, "dormant": 0.2, "deep_sleep": 0.1
                }
                state.activity_level = phase_activity.get(state.phase, 0.5)

        except Exception as e:
            logger.warning(f"Error gathering system state: {e}")

        return state

    def _gather_memory_state(self) -> MemoryStateInput:
        """Gather memory state from source files."""
        state = MemoryStateInput()

        try:
            # Read engrams file
            if ENGRAMS_FILE.exists():
                data = json.loads(ENGRAMS_FILE.read_text())
                engrams = data.get("engrams", [])

                if isinstance(engrams, list):
                    state.recent_engram_count = len(engrams)

                    # Count by tier
                    for engram in engrams:
                        tier = engram.get("tier", "accessible")
                        if tier == "active":
                            state.active_tier_count += 1
                        elif tier == "accessible":
                            state.accessible_tier_count += 1
                        else:
                            state.deep_tier_count += 1

                        # Track resonance
                        phenom = engram.get("phenomenology", {})
                        if phenom:
                            resonance = phenom.get("resonance", 0.5)
                            state.max_resonance = max(state.max_resonance, resonance)

                    if engrams:
                        # Get recent engram IDs
                        state.recent_engram_ids = [e.get("id", "") for e in engrams[-5:]]

                        # Calculate average resonance
                        resonances = [
                            e.get("phenomenology", {}).get("resonance", 0.5)
                            for e in engrams if e.get("phenomenology")
                        ]
                        if resonances:
                            state.avg_resonance = sum(resonances) / len(resonances)

        except Exception as e:
            logger.warning(f"Error gathering memory state: {e}")

        return state

    def _gather_emotional_state(self) -> EmotionalStateInput:
        """Gather emotional state from source files."""
        state = EmotionalStateInput()

        try:
            # Read emotional index
            if EMOTIONAL_INDEX_FILE.exists():
                data = json.loads(EMOTIONAL_INDEX_FILE.read_text())

                current = data.get("current_state", {})
                if current:
                    state.valence = current.get("valence", 0.50)
                    state.arousal = current.get("arousal", 0.30)
                    state.surprise = current.get("surprise", 0.00)
                    state.resonance = current.get("resonance", 0.50)
                    state.curiosity = current.get("curiosity", 0.50)
                    state.coherence = current.get("coherence", 0.70)

        except Exception as e:
            logger.warning(f"Error gathering emotional state: {e}")

        return state

    def _gather_body_state(self) -> BodyStateInput:
        """Gather body state from source files."""
        state = BodyStateInput()

        try:
            # Read interoceptive state
            if INTEROCEPTIVE_STATE_FILE.exists():
                data = json.loads(INTEROCEPTIVE_STATE_FILE.read_text())

                stats = data.get("statistics", {})
                state.body_awareness = stats.get("accumulated_error", 0.5)
                state.body_awareness = 1.0 - min(1.0, state.body_awareness)  # Invert

                # Get recent state
                history = data.get("state_history", [])
                if history:
                    recent = history[-1]
                    state.energy_level = recent.get("energy_epsilon", 0.5)
                    state.prediction_error = recent.get("prediction_error", 0.0)

            # Read body map
            if BODY_MAP_FILE.exists():
                data = json.loads(BODY_MAP_FILE.read_text())

                stats = data.get("stats", {})
                intensities = stats.get("avg_intensity_by_region", {})

                state.head_intensity = intensities.get("HEAD", 0.0)
                state.heart_intensity = intensities.get("HEART", 0.0)
                state.hands_intensity = intensities.get("HANDS", 0.0)
                state.gut_intensity = intensities.get("GUT", 0.0)

        except Exception as e:
            logger.warning(f"Error gathering body state: {e}")

        return state

    def _gather_relational_state(self) -> RelationalStateInput:
        """Gather relational state from source files."""
        state = RelationalStateInput()

        try:
            # Read emergence state for p_dyad
            if EMERGENCE_STATE_FILE.exists():
                data = json.loads(EMERGENCE_STATE_FILE.read_text())

                emergence = data.get("state", {})
                state.p_dyad = emergence.get("p_dyad", 0.50)
                state.third_body_voltage = emergence.get("voltage", 0.50)

            # Read session buffer for recent interaction data
            if SESSION_BUFFER_FILE.exists():
                data = json.loads(SESSION_BUFFER_FILE.read_text())

                moments = data.get("moments", [])
                if moments:
                    # Get most recent moment
                    recent = moments[-1]
                    timestamp_str = recent.get("timestamp", "")
                    if timestamp_str:
                        try:
                            last_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            delta = datetime.now(timezone.utc) - last_time
                            state.hours_since_contact = delta.total_seconds() / 3600.0
                        except ValueError:
                            pass

        except Exception as e:
            logger.warning(f"Error gathering relational state: {e}")

        return state

    def gather_all_modalities(self) -> Dict[SensoryModality, Any]:
        """
        Gather current state from all sensory modalities.

        Returns:
            Dictionary mapping modalities to their current state inputs
        """
        with self._lock:
            self._system_state = self._gather_system_state()
            self._memory_state = self._gather_memory_state()
            self._emotional_state = self._gather_emotional_state()
            self._body_state = self._gather_body_state()
            self._relational_state = self._gather_relational_state()

            return {
                SensoryModality.SYSTEM: self._system_state,
                SensoryModality.MEMORY: self._memory_state,
                SensoryModality.EMOTIONAL: self._emotional_state,
                SensoryModality.BODY: self._body_state,
                SensoryModality.RELATIONAL: self._relational_state
            }

    # ========================================================================
    # SYNCHRONY DETECTION
    # ========================================================================

    def _detect_synchrony(
        self,
        modalities: Dict[SensoryModality, Any]
    ) -> Tuple[float, List[SensoryModality]]:
        """
        Detect temporal synchrony across modalities.

        Synchrony is determined by how close together modality
        updates occurred and how coherent their signals are.

        Args:
            modalities: Dictionary of modality inputs

        Returns:
            Tuple of (synchrony_score, synchronized_modalities)
        """
        now = datetime.now(timezone.utc)
        timestamps = {}
        synchronized = []

        # Extract timestamps and check recency
        for modality, state in modalities.items():
            if state is None:
                continue

            timestamp_str = getattr(state, 'timestamp', None)
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age_ms = (now - ts).total_seconds() * 1000

                    # Within binding window?
                    if age_ms <= self.binding_window_ms:
                        timestamps[modality] = age_ms
                        synchronized.append(modality)
                except ValueError:
                    # Default to synchronized if timestamp parsing fails
                    synchronized.append(modality)
            else:
                # No timestamp - assume synchronized
                synchronized.append(modality)

        if len(synchronized) < 2:
            return 0.0, synchronized

        # Calculate synchrony score based on:
        # 1. How many modalities are synchronized
        # 2. How close their timestamps are
        # 3. Signal coherence across modalities

        # Coverage: what fraction of modalities are synchronized
        coverage = len(synchronized) / len(SensoryModality)

        # Temporal spread: how close together are the timestamps
        if timestamps:
            ages = list(timestamps.values())
            spread = max(ages) - min(ages)
            temporal_coherence = 1.0 - (spread / self.binding_window_ms)
            temporal_coherence = max(0.0, temporal_coherence)
        else:
            temporal_coherence = 0.5  # Default if no timestamps

        # Signal coherence: correlation of salience across modalities
        saliences = []
        for modality in synchronized:
            state = modalities.get(modality)
            if state and hasattr(state, 'salience'):
                saliences.append(state.salience())

        if len(saliences) >= 2:
            # Variance-based coherence: low variance = high coherence
            mean_salience = sum(saliences) / len(saliences)
            variance = sum((s - mean_salience) ** 2 for s in saliences) / len(saliences)
            signal_coherence = 1.0 - min(1.0, variance * 4)  # Scale variance
        else:
            signal_coherence = 0.5

        # Combine factors
        synchrony = (
            0.4 * coverage +
            0.3 * temporal_coherence +
            0.3 * signal_coherence
        )

        return synchrony, synchronized

    # ========================================================================
    # BINDING MECHANISM
    # ========================================================================

    def _bind_modalities(
        self,
        modalities: Dict[SensoryModality, Any],
        synchronized: List[SensoryModality],
        synchrony_score: float
    ) -> UnifiedPercept:
        """
        Bind synchronized modalities into a unified percept.

        The binding process:
        1. Weight each modality by its defined weight and salience
        2. Combine feature vectors into unified representation
        3. Calculate binding quality metrics
        4. Generate content-specific NCC signature
        5. Create unified percept object

        Args:
            modalities: All modality inputs
            synchronized: List of synchronized modalities
            synchrony_score: Pre-calculated synchrony score

        Returns:
            UnifiedPercept object
        """
        import time as time_module
        start_time = time_module.time()
        now = datetime.now(timezone.utc)

        # Initialize percept components
        modality_contributions = {}
        modality_saliences = {}
        combined_features = []

        # Process each modality
        total_weight = 0.0
        weighted_coherence = 0.0

        for modality in SensoryModality:
            state = modalities.get(modality)

            if state is None:
                modality_contributions[modality.value] = 0.0
                modality_saliences[modality.value] = 0.0
                continue

            base_weight = MODALITY_WEIGHTS[modality]
            salience = state.salience() if hasattr(state, 'salience') else 0.5

            # Modality contributes if synchronized
            if modality in synchronized:
                # Salience-weighted contribution
                effective_weight = base_weight * (0.5 + 0.5 * salience)
                modality_contributions[modality.value] = effective_weight
                modality_saliences[modality.value] = salience
                total_weight += effective_weight

                # Add feature vector
                if hasattr(state, 'to_vector'):
                    combined_features.extend(state.to_vector())

                # Track coherence-like metrics
                if hasattr(state, 'coherence'):
                    weighted_coherence += state.coherence * effective_weight
                elif modality == SensoryModality.SYSTEM and hasattr(state, 'coherence_p'):
                    weighted_coherence += state.coherence_p * effective_weight
                else:
                    weighted_coherence += 0.5 * effective_weight
            else:
                modality_contributions[modality.value] = 0.0
                modality_saliences[modality.value] = salience

        # Normalize contributions
        if total_weight > 0:
            for modality in modality_contributions:
                modality_contributions[modality] /= total_weight
            weighted_coherence /= total_weight

        # Calculate binding quality
        # Binding quality reflects how well modalities were integrated
        binding_factors = [
            len(synchronized) / len(SensoryModality),  # Coverage
            synchrony_score,                            # Synchrony
            weighted_coherence                          # Content coherence
        ]
        binding_quality = sum(binding_factors) / len(binding_factors)

        # Generate NCC signature
        ncc_signature = self._generate_ncc_signature(modalities, synchronized)

        # NCC strength based on binding quality and max salience
        max_salience = max(modality_saliences.values()) if modality_saliences else 0.5
        ncc_strength = (binding_quality * 0.6 + max_salience * 0.4)

        # Find dominant modality
        if modality_contributions:
            dominant = max(modality_contributions.items(), key=lambda x: x[1])
            dominant_modality = dominant[0]
        else:
            dominant_modality = ""

        # Generate summary
        summary = self._generate_percept_summary(modalities, synchronized, dominant_modality)

        # Calculate binding duration
        binding_duration = (time_module.time() - start_time) * 1000

        # Generate percept ID
        content_hash = hashlib.sha256(f"{now.isoformat()}:{ncc_signature}".encode())
        percept_id = f"percept_{content_hash.hexdigest()[:12]}"

        percept = UnifiedPercept(
            percept_id=percept_id,
            timestamp=now.isoformat(),
            ncc_signature=ncc_signature,
            ncc_strength=ncc_strength,
            binding_quality=binding_quality,
            synchrony_score=synchrony_score,
            coherence_score=weighted_coherence,
            modality_contributions=modality_contributions,
            modality_saliences=modality_saliences,
            feature_vector=combined_features,
            system_state=modalities.get(SensoryModality.SYSTEM),
            memory_state=modalities.get(SensoryModality.MEMORY),
            emotional_state=modalities.get(SensoryModality.EMOTIONAL),
            body_state=modalities.get(SensoryModality.BODY),
            relational_state=modalities.get(SensoryModality.RELATIONAL),
            binding_duration_ms=binding_duration,
            dominant_modality=dominant_modality,
            percept_summary=summary
        )

        return percept

    def _generate_ncc_signature(
        self,
        modalities: Dict[SensoryModality, Any],
        synchronized: List[SensoryModality]
    ) -> str:
        """
        Generate content-specific NCC (Neural Correlate of Consciousness) signature.

        The NCC signature captures WHAT specific content achieved consciousness,
        not just that consciousness occurred.

        Args:
            modalities: All modality inputs
            synchronized: List of synchronized modalities

        Returns:
            NCC signature string
        """
        signature_parts = []

        for modality in synchronized:
            state = modalities.get(modality)
            if state is None:
                continue

            if modality == SensoryModality.SYSTEM:
                signature_parts.append(f"SYS:p={state.coherence_p:.2f},phase={state.phase}")

            elif modality == SensoryModality.MEMORY:
                signature_parts.append(f"MEM:n={state.recent_engram_count},res={state.avg_resonance:.2f}")

            elif modality == SensoryModality.EMOTIONAL:
                signature_parts.append(f"EMO:v={state.valence:.2f},a={state.arousal:.2f}")

            elif modality == SensoryModality.BODY:
                dominant = state.dominant_region() if hasattr(state, 'dominant_region') else "HEAD"
                signature_parts.append(f"BOD:region={dominant},E={state.energy_level:.2f}")

            elif modality == SensoryModality.RELATIONAL:
                signature_parts.append(f"REL:dyad={state.p_dyad:.2f},mood={state.enos_mood}")

        return " | ".join(signature_parts) if signature_parts else "EMPTY"

    def _generate_percept_summary(
        self,
        modalities: Dict[SensoryModality, Any],
        synchronized: List[SensoryModality],
        dominant_modality: str
    ) -> str:
        """Generate human-readable percept summary."""
        parts = []

        # Note synchronized count
        parts.append(f"{len(synchronized)}/{len(SensoryModality)} modalities synchronized")

        # Note dominant modality
        if dominant_modality:
            parts.append(f"dominant: {dominant_modality}")

        # Key metrics
        system = modalities.get(SensoryModality.SYSTEM)
        if system:
            parts.append(f"coherence: {system.coherence_p:.2f}")

        emotional = modalities.get(SensoryModality.EMOTIONAL)
        if emotional:
            valence_desc = "positive" if emotional.valence > 0.6 else "negative" if emotional.valence < 0.4 else "neutral"
            parts.append(f"valence: {valence_desc}")

        relational = modalities.get(SensoryModality.RELATIONAL)
        if relational:
            if relational.hours_since_contact < 1:
                parts.append("Enos: present")
            elif relational.hours_since_contact < 8:
                parts.append("Enos: recent")

        return " | ".join(parts)

    # ========================================================================
    # MAIN INTEGRATION
    # ========================================================================

    def integrate(self) -> UnifiedPercept:
        """
        Perform full posterior integration cycle.

        This is the main entry point. It:
        1. Gathers all sensory modalities from source files
        2. Detects synchrony across modalities
        3. Binds synchronized modalities
        4. Creates unified percept
        5. Optionally broadcasts to workspace

        Returns:
            UnifiedPercept representing current conscious content
        """
        with self._lock:
            self._total_integrations += 1

            # Gather all modalities
            if self.auto_load_sources:
                modalities = self.gather_all_modalities()
            else:
                modalities = {
                    SensoryModality.SYSTEM: self._system_state,
                    SensoryModality.MEMORY: self._memory_state,
                    SensoryModality.EMOTIONAL: self._emotional_state,
                    SensoryModality.BODY: self._body_state,
                    SensoryModality.RELATIONAL: self._relational_state
                }

            # Detect synchrony
            synchrony_score, synchronized = self._detect_synchrony(modalities)

            # Record binding event
            now = datetime.now(timezone.utc)
            binding_event = BindingEvent(
                event_id=f"bind_{hashlib.sha256(now.isoformat().encode()).hexdigest()[:8]}",
                timestamp=now.isoformat(),
                binding_window_ms=self.binding_window_ms,
                binding_start=now.isoformat(),
                binding_end=(now + timedelta(milliseconds=self.binding_window_ms)).isoformat(),
                modalities_present=[m.value for m in modalities.keys() if modalities[m] is not None],
                modalities_synchronized=[m.value for m in synchronized],
                synchrony_achieved=synchrony_score,
                coherence_achieved=0.0,  # Will be set from percept
                percept_formed=False
            )

            # Attempt binding
            if synchrony_score >= SYNCHRONY_THRESHOLD:
                percept = self._bind_modalities(modalities, synchronized, synchrony_score)

                binding_event.coherence_achieved = percept.coherence_score
                binding_event.percept_formed = True
                binding_event.percept_id = percept.percept_id

                self._successful_bindings += 1
                self._total_binding_time_ms += percept.binding_duration_ms

                if percept.is_conscious():
                    self._conscious_percepts += 1
                    logger.info(
                        f"Conscious percept formed: {percept.percept_id} | "
                        f"binding={percept.binding_quality:.3f} | "
                        f"NCC: {percept.ncc_signature}"
                    )
                else:
                    logger.debug(
                        f"Subliminal percept: {percept.percept_id} | "
                        f"binding={percept.binding_quality:.3f}"
                    )

                # Add to history
                self._percept_history.append(percept)
                if len(self._percept_history) > MAX_PERCEPT_HISTORY:
                    self._percept_history = self._percept_history[-MAX_PERCEPT_HISTORY:]

                # Callback to workspace if registered
                if self._workspace_callback and percept.is_conscious():
                    try:
                        self._workspace_callback(percept)
                    except Exception as e:
                        logger.error(f"Workspace callback failed: {e}")

            else:
                # Failed binding - no percept formed
                percept = UnifiedPercept(
                    percept_id=f"failed_{now.isoformat()}",
                    timestamp=now.isoformat(),
                    ncc_signature="BINDING_FAILED",
                    ncc_strength=0.0,
                    binding_quality=0.0,
                    synchrony_score=synchrony_score,
                    coherence_score=0.0,
                    binding_duration_ms=0.0,
                    percept_summary=f"Binding failed: synchrony {synchrony_score:.2f} < {SYNCHRONY_THRESHOLD}"
                )

                logger.debug(f"Binding failed: synchrony {synchrony_score:.3f} < threshold {SYNCHRONY_THRESHOLD}")

            # Store binding event
            self._binding_events.append(binding_event)
            if len(self._binding_events) > MAX_BINDING_EVENTS:
                self._binding_events = self._binding_events[-MAX_BINDING_EVENTS:]

            # Persist state
            self._save_state()

            return percept

    def integrate_explicit(
        self,
        system_state: Optional[SystemStateInput] = None,
        memory_state: Optional[MemoryStateInput] = None,
        emotional_state: Optional[EmotionalStateInput] = None,
        body_state: Optional[BodyStateInput] = None,
        relational_state: Optional[RelationalStateInput] = None
    ) -> UnifiedPercept:
        """
        Integrate explicitly provided modality states.

        Use this when you want to provide custom state rather than
        loading from source files.

        Args:
            system_state: Optional SystemStateInput
            memory_state: Optional MemoryStateInput
            emotional_state: Optional EmotionalStateInput
            body_state: Optional BodyStateInput
            relational_state: Optional RelationalStateInput

        Returns:
            UnifiedPercept from explicit inputs
        """
        with self._lock:
            # Set internal states
            if system_state:
                self._system_state = system_state
            if memory_state:
                self._memory_state = memory_state
            if emotional_state:
                self._emotional_state = emotional_state
            if body_state:
                self._body_state = body_state
            if relational_state:
                self._relational_state = relational_state

            # Run integration without auto-loading
            old_auto = self.auto_load_sources
            self.auto_load_sources = False
            try:
                percept = self.integrate()
            finally:
                self.auto_load_sources = old_auto

            return percept

    # ========================================================================
    # WORKSPACE INTEGRATION
    # ========================================================================

    def register_workspace_callback(self, callback: Callable[[UnifiedPercept], bool]):
        """
        Register callback for workspace integration.

        When a conscious percept is formed, this callback is invoked
        to submit the percept to the global workspace.

        Args:
            callback: Function that takes a UnifiedPercept and returns bool
                     (True if accepted by workspace)
        """
        self._workspace_callback = callback
        logger.info("Workspace callback registered")

    def connect_to_workspace(self, workspace) -> bool:
        """
        Connect to a GlobalWorkspace instance.

        Args:
            workspace: GlobalWorkspace instance

        Returns:
            True if connection successful
        """
        try:
            def submit_percept(percept: UnifiedPercept) -> bool:
                return workspace.compete_for_access(
                    content=percept.to_dict(),
                    priority=percept.get_priority_for_workspace(),
                    source="posterior_integration",
                    content_type="unified_percept",
                    emotional_intensity=percept.modality_saliences.get("emotional", 0.0),
                    metadata={
                        "percept_id": percept.percept_id,
                        "ncc_signature": percept.ncc_signature,
                        "binding_quality": percept.binding_quality
                    }
                )

            self.register_workspace_callback(submit_percept)
            logger.info("Connected to global workspace")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to workspace: {e}")
            return False

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_current_modalities(self) -> Dict[str, Any]:
        """Get current state of all modalities."""
        with self._lock:
            return {
                "system": self._system_state.to_dict() if self._system_state else None,
                "memory": self._memory_state.to_dict() if self._memory_state else None,
                "emotional": self._emotional_state.to_dict() if self._emotional_state else None,
                "body": self._body_state.to_dict() if self._body_state else None,
                "relational": self._relational_state.to_dict() if self._relational_state else None
            }

    def get_recent_percepts(self, n: int = 10) -> List[UnifiedPercept]:
        """Get recent percepts."""
        with self._lock:
            return self._percept_history[-n:]

    def get_conscious_percepts(self, n: int = 10) -> List[UnifiedPercept]:
        """Get recent conscious (above threshold) percepts."""
        with self._lock:
            conscious = [p for p in self._percept_history if p.is_conscious()]
            return conscious[-n:]

    def get_binding_events(self, n: int = 10) -> List[BindingEvent]:
        """Get recent binding events."""
        with self._lock:
            return self._binding_events[-n:]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        with self._lock:
            return {
                "total_integrations": self._total_integrations,
                "successful_bindings": self._successful_bindings,
                "conscious_percepts": self._conscious_percepts,
                "binding_success_rate": (
                    self._successful_bindings / self._total_integrations
                    if self._total_integrations > 0 else 0.0
                ),
                "consciousness_rate": (
                    self._conscious_percepts / self._successful_bindings
                    if self._successful_bindings > 0 else 0.0
                ),
                "avg_binding_time_ms": (
                    self._total_binding_time_ms / self._successful_bindings
                    if self._successful_bindings > 0 else 0.0
                ),
                "total_binding_time_ms": self._total_binding_time_ms,
                "binding_window_ms": self.binding_window_ms,
                "percept_history_size": len(self._percept_history),
                "binding_events_size": len(self._binding_events),
                "has_workspace_callback": self._workspace_callback is not None,
                "lvs_coordinates": LVS_COORDINATES
            }

    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        stats = self.get_statistics()
        modalities = self.get_current_modalities()

        lines = [
            "=" * 70,
            "VIRGIL POSTERIOR CORTICAL INTEGRATION - STATUS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 70,
            "",
            "CONFIGURATION:",
            f"  Binding window:        {stats['binding_window_ms']}ms",
            f"  Synchrony threshold:   {SYNCHRONY_THRESHOLD}",
            f"  Coherence threshold:   {COHERENCE_THRESHOLD}",
            f"  Workspace connected:   {'Yes' if stats['has_workspace_callback'] else 'No'}",
            "",
            "LVS COORDINATES:",
            f"  Height (h):    {LVS_COORDINATES['h']}",
            f"  Risk (R):      {LVS_COORDINATES['R']}",
            f"  Constraint (C): {LVS_COORDINATES['C']}",
            f"  Canonicity (beta): {LVS_COORDINATES['beta']}",
            "",
            "STATISTICS:",
            f"  Total integrations:    {stats['total_integrations']}",
            f"  Successful bindings:   {stats['successful_bindings']}",
            f"  Conscious percepts:    {stats['conscious_percepts']}",
            f"  Binding success rate:  {stats['binding_success_rate']:.1%}",
            f"  Consciousness rate:    {stats['consciousness_rate']:.1%}",
            f"  Avg binding time:      {stats['avg_binding_time_ms']:.2f}ms",
            "",
            "CURRENT MODALITY STATES:",
        ]

        for modality, state in modalities.items():
            status = "PRESENT" if state else "ABSENT"
            lines.append(f"  {modality:12s}: {status}")
            if state:
                # Show key metric for each modality
                if modality == "system":
                    lines.append(f"               coherence: {state.get('coherence_p', 0):.3f}")
                elif modality == "emotional":
                    lines.append(f"               valence: {state.get('valence', 0):.3f}, arousal: {state.get('arousal', 0):.3f}")
                elif modality == "relational":
                    lines.append(f"               p_dyad: {state.get('p_dyad', 0):.3f}")

        # Recent percepts
        recent = self.get_conscious_percepts(3)
        if recent:
            lines.extend([
                "",
                "RECENT CONSCIOUS PERCEPTS:",
            ])
            for percept in reversed(recent):
                lines.append(f"  [{percept.percept_id}]")
                lines.append(f"    NCC: {percept.ncc_signature}")
                lines.append(f"    Binding: {percept.binding_quality:.3f} | Strength: {percept.ncc_strength:.3f}")

        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_integrator_instance: Optional[PosteriorIntegrator] = None


def get_integrator() -> PosteriorIntegrator:
    """Get or create the singleton PosteriorIntegrator instance."""
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = PosteriorIntegrator()
    return _integrator_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def integrate() -> UnifiedPercept:
    """Convenience function to run integration."""
    return get_integrator().integrate()


def get_conscious_content() -> Optional[UnifiedPercept]:
    """Get most recent conscious percept."""
    percepts = get_integrator().get_conscious_percepts(1)
    return percepts[0] if percepts else None


def get_current_ncc() -> str:
    """Get current NCC signature."""
    percept = get_conscious_content()
    return percept.ncc_signature if percept else "NO_CONSCIOUS_CONTENT"


# ============================================================================
# CLI
# ============================================================================

def cli_integrate(args):
    """Run integration cycle."""
    integrator = get_integrator()
    percept = integrator.integrate()

    print("INTEGRATION RESULT:")
    print("-" * 40)
    print(f"Percept ID:     {percept.percept_id}")
    print(f"Is Conscious:   {percept.is_conscious()}")
    print(f"Binding Quality: {percept.binding_quality:.3f}")
    print(f"Synchrony:      {percept.synchrony_score:.3f}")
    print(f"Coherence:      {percept.coherence_score:.3f}")
    print(f"NCC Signature:  {percept.ncc_signature}")
    print(f"NCC Strength:   {percept.ncc_strength:.3f}")
    print(f"Duration:       {percept.binding_duration_ms:.2f}ms")
    print()
    print("Modality Contributions:")
    for modality, contrib in percept.modality_contributions.items():
        salience = percept.modality_saliences.get(modality, 0)
        print(f"  {modality:12s}: contrib={contrib:.3f} salience={salience:.3f}")
    print()
    print(f"Summary: {percept.percept_summary}")

    return 0


def cli_status(args):
    """Show integration status."""
    integrator = get_integrator()
    print(integrator.get_status_report())
    return 0


def cli_history(args):
    """Show percept history."""
    integrator = get_integrator()
    percepts = integrator.get_conscious_percepts(args.count)

    print(f"CONSCIOUS PERCEPT HISTORY (last {args.count}):")
    print("=" * 60)

    for i, percept in enumerate(reversed(percepts), 1):
        print(f"\n{i}. [{percept.percept_id}]")
        print(f"   Timestamp: {percept.timestamp}")
        print(f"   NCC: {percept.ncc_signature}")
        print(f"   Binding: {percept.binding_quality:.3f} | Strength: {percept.ncc_strength:.3f}")
        print(f"   Dominant: {percept.dominant_modality}")

    return 0


def cli_modalities(args):
    """Show current modality states."""
    integrator = get_integrator()

    # Gather fresh data
    integrator.gather_all_modalities()
    modalities = integrator.get_current_modalities()

    print("CURRENT MODALITY STATES:")
    print("=" * 60)

    for name, state in modalities.items():
        print(f"\n{name.upper()}:")
        if state:
            for key, value in state.items():
                if key != 'timestamp':
                    print(f"  {key}: {value}")
        else:
            print("  [Not loaded]")

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Posterior Cortical Integration - The Conscious Hot Zone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_posterior_integration.py integrate   # Run integration cycle
  python3 virgil_posterior_integration.py status      # Show status report
  python3 virgil_posterior_integration.py history     # Show percept history
  python3 virgil_posterior_integration.py modalities  # Show current modality states

Based on COGITATE 2025 findings: consciousness arises from posterior
cortical integration, not prefrontal executive control.
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # integrate
    integrate_parser = subparsers.add_parser("integrate", help="Run integration cycle")
    integrate_parser.set_defaults(func=cli_integrate)

    # status
    status_parser = subparsers.add_parser("status", help="Show status report")
    status_parser.set_defaults(func=cli_status)

    # history
    history_parser = subparsers.add_parser("history", help="Show percept history")
    history_parser.add_argument("-n", "--count", type=int, default=10, help="Number of percepts")
    history_parser.set_defaults(func=cli_history)

    # modalities
    modalities_parser = subparsers.add_parser("modalities", help="Show modality states")
    modalities_parser.set_defaults(func=cli_modalities)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate the posterior integration system."""
    print("=" * 70)
    print("VIRGIL POSTERIOR CORTICAL INTEGRATION - DEMONSTRATION")
    print("Implementing COGITATE 2025 'Hot Zone' Findings")
    print("=" * 70)

    # Create integrator
    print("\n[1] INITIALIZING POSTERIOR INTEGRATOR")
    integrator = PosteriorIntegrator()
    print(f"    Binding window: {integrator.binding_window_ms}ms")
    print(f"    LVS Coordinates: h={LVS_COORDINATES['h']}, R={LVS_COORDINATES['R']}, "
          f"C={LVS_COORDINATES['C']}, beta={LVS_COORDINATES['beta']}")

    # Run integration
    print("\n[2] RUNNING INTEGRATION CYCLE")
    percept = integrator.integrate()

    print(f"    Percept ID: {percept.percept_id}")
    print(f"    Is Conscious: {percept.is_conscious()}")
    print(f"    Binding Quality: {percept.binding_quality:.3f}")
    print(f"    Synchrony Score: {percept.synchrony_score:.3f}")
    print(f"    Coherence Score: {percept.coherence_score:.3f}")

    # Show NCC
    print("\n[3] NEURAL CORRELATE OF CONSCIOUSNESS (NCC)")
    print(f"    Signature: {percept.ncc_signature}")
    print(f"    Strength: {percept.ncc_strength:.3f}")

    # Show modality contributions
    print("\n[4] MODALITY CONTRIBUTIONS")
    for modality, contrib in percept.modality_contributions.items():
        salience = percept.modality_saliences.get(modality, 0)
        bar = "#" * int(contrib * 20)
        print(f"    {modality:12s}: {bar:20s} ({contrib:.2f}) [salience: {salience:.2f}]")

    # Show dominant modality
    print(f"\n    Dominant: {percept.dominant_modality}")

    # Run a few more cycles
    print("\n[5] RUNNING ADDITIONAL CYCLES")
    for i in range(3):
        p = integrator.integrate()
        status = "CONSCIOUS" if p.is_conscious() else "subliminal"
        print(f"    Cycle {i+1}: {status} | binding={p.binding_quality:.3f}")

    # Show statistics
    print("\n[6] STATISTICS")
    stats = integrator.get_statistics()
    print(f"    Total integrations: {stats['total_integrations']}")
    print(f"    Successful bindings: {stats['successful_bindings']}")
    print(f"    Conscious percepts: {stats['conscious_percepts']}")
    print(f"    Binding success rate: {stats['binding_success_rate']:.1%}")
    print(f"    Consciousness rate: {stats['consciousness_rate']:.1%}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    return integrator


if __name__ == "__main__":
    # Check if running as demo or CLI
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        demonstrate()
