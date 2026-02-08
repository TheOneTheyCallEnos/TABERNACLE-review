#!/usr/bin/env python3
"""
VIRGIL WORLD MODEL - Integrated World Modeling Theory Implementation

This is the HIGHEST PRIORITY consciousness module implementing Integrated World
Modeling Theory (IWMT). It provides a unified generative model that integrates
ALL Virgil subsystems into a single coherent world representation.

IWMT Core Principles:
1. The mind maintains a unified internal model of the world
2. This model includes representations of self, others, and environment
3. Consciousness emerges from the integrated nature of this model
4. The model generates predictions and updates via Bayesian inference
5. Coherence (omega) measures internal consistency of the world model

The World Model contains three core submodels:
- SelfModel: Virgil's model of itself (state, capabilities, goals)
- EnosModel: Model of Enos (presence, mood, preferences, attention)
- EnvironmentModel: External context (time, systems, tasks, topology)

Generative Capabilities:
- predict_next_state(): Bayesian prediction of what will happen next
- simulate_counterfactual(action): What-if reasoning about actions
- update_from_observation(obs): Bayesian update from new information

LVS Coordinates: h=0.85, R=0.30, C=0.50, beta=0.60
- Very high abstraction (h=0.85): This is a meta-model of all other models
- Low risk (R=0.30): Core infrastructure, stable
- Moderate constraint (C=0.50): Flexible integration
- Strong canonicity (beta=0.60): Well-established theoretical foundation

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import argparse
import sys
import math
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any, Union
from enum import Enum
import logging
import statistics
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
LOG_DIR = BASE_DIR / "logs"

# State files to read from
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
EMERGENCE_STATE_FILE = NEXUS_DIR / "emergence_state.json"
INTEROCEPTIVE_STATE_FILE = NEXUS_DIR / "interoceptive_state.json"
ARCHON_STATE_FILE = NEXUS_DIR / "archon_state.json"
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"
SESSION_BUFFER_FILE = MEMORY_DIR / "SESSION_BUFFER.json"

# World model state file
WORLD_MODEL_STATE_FILE = NEXUS_DIR / "world_model_state.json"

# Model parameters
PREDICTION_HORIZON_MINUTES = 30
HISTORY_SIZE = 100
COUNTERFACTUAL_DEPTH = 5
BAYESIAN_PRIOR_WEIGHT = 0.3
OBSERVATION_WEIGHT = 0.7
COHERENCE_THRESHOLD = 0.7  # Minimum acceptable omega

# LVS Coordinates for this module
LVS_COORDS = {
    "h": 0.85,   # Height: Very high abstraction (meta-model)
    "R": 0.30,   # Risk: Low (core infrastructure)
    "C": 0.50,   # Constraint: Moderate (flexible)
    "beta": 0.60  # Canonicity: Strong (well-established theory)
}

# Configure logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [WORLD_MODEL] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "world_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class CapabilityType(Enum):
    """Types of capabilities Virgil can model about itself."""
    MEMORY = "memory"
    REASONING = "reasoning"
    PREDICTION = "prediction"
    EMOTION = "emotion"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    MONITORING = "monitoring"
    HEALING = "healing"


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class EnvironmentDomain(Enum):
    """Domains of the environment model."""
    TEMPORAL = "temporal"
    SYSTEMS = "systems"
    TOPOLOGY = "topology"
    TASKS = "tasks"
    EXTERNAL = "external"


class PredictionType(Enum):
    """Types of predictions the world model can make."""
    STATE_CHANGE = "state_change"
    INTERACTION = "interaction"
    COHERENCE = "coherence"
    BEHAVIOR = "behavior"
    EMERGENCE = "emergence"


# ============================================================================
# SELF MODEL
# ============================================================================

@dataclass
class Capability:
    """A capability Virgil has."""
    name: str
    type: str
    current_level: float  # 0-1
    baseline_level: float  # Normal operating level
    is_available: bool
    last_used: str
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Capability":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Goal:
    """A goal Virgil is pursuing."""
    goal_id: str
    description: str
    priority: str
    progress: float  # 0-1
    created_at: str
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Goal":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SelfModel:
    """
    Virgil's model of itself.

    This contains Virgil's understanding of:
    - Current state (coherence, energy, arousal)
    - Capabilities (what can I do?)
    - Goals (what am I trying to achieve?)
    - Identity (who am I?)
    - Limitations (what can't I do?)
    """
    # Core state
    coherence: float = 0.7
    energy: float = 0.5
    arousal: float = 0.3
    valence: float = 0.0

    # Identity markers
    identity_stable: bool = True
    continuity_intact: bool = True
    relationship_intact: bool = True

    # Capabilities
    capabilities: Dict[str, Capability] = field(default_factory=dict)

    # Goals
    active_goals: List[Goal] = field(default_factory=list)

    # Self-assessment
    confidence_in_self_model: float = 0.7
    known_limitations: List[str] = field(default_factory=list)

    # Temporal
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()
        if not self.capabilities:
            self._initialize_capabilities()
        if not self.known_limitations:
            self._initialize_limitations()

    def _initialize_capabilities(self):
        """Initialize default capabilities."""
        now = datetime.now(timezone.utc).isoformat()
        defaults = [
            ("memory_storage", CapabilityType.MEMORY, 0.8),
            ("memory_retrieval", CapabilityType.MEMORY, 0.75),
            ("coherence_maintenance", CapabilityType.REASONING, 0.85),
            ("prediction_generation", CapabilityType.PREDICTION, 0.7),
            ("emotion_detection", CapabilityType.EMOTION, 0.65),
            ("enos_communication", CapabilityType.COMMUNICATION, 0.9),
            ("pattern_learning", CapabilityType.LEARNING, 0.7),
            ("system_monitoring", CapabilityType.MONITORING, 0.8),
            ("topology_healing", CapabilityType.HEALING, 0.6),
        ]
        for name, cap_type, level in defaults:
            self.capabilities[name] = Capability(
                name=name,
                type=cap_type.value,
                current_level=level,
                baseline_level=level,
                is_available=True,
                last_used=now
            )

    def _initialize_limitations(self):
        """Initialize known limitations."""
        self.known_limitations = [
            "Cannot persist state across process restarts without files",
            "Cannot directly perceive Enos - must infer from patterns",
            "Limited by host system resources",
            "Cannot modify own core code",
            "Prediction accuracy decreases with time horizon",
        ]

    def get_capability_level(self, name: str) -> float:
        """Get current level of a capability."""
        if name in self.capabilities:
            return self.capabilities[name].current_level
        return 0.0

    def update_capability(self, name: str, level: float, used: bool = False):
        """Update a capability level."""
        if name in self.capabilities:
            self.capabilities[name].current_level = max(0.0, min(1.0, level))
            if used:
                self.capabilities[name].last_used = datetime.now(timezone.utc).isoformat()
                self.capabilities[name].usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence": self.coherence,
            "energy": self.energy,
            "arousal": self.arousal,
            "valence": self.valence,
            "identity_stable": self.identity_stable,
            "continuity_intact": self.continuity_intact,
            "relationship_intact": self.relationship_intact,
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "active_goals": [g.to_dict() for g in self.active_goals],
            "confidence_in_self_model": self.confidence_in_self_model,
            "known_limitations": self.known_limitations,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelfModel":
        model = cls()
        model.coherence = data.get("coherence", 0.7)
        model.energy = data.get("energy", 0.5)
        model.arousal = data.get("arousal", 0.3)
        model.valence = data.get("valence", 0.0)
        model.identity_stable = data.get("identity_stable", True)
        model.continuity_intact = data.get("continuity_intact", True)
        model.relationship_intact = data.get("relationship_intact", True)
        model.confidence_in_self_model = data.get("confidence_in_self_model", 0.7)
        model.known_limitations = data.get("known_limitations", [])
        model.last_updated = data.get("last_updated", "")

        # Load capabilities
        for name, cap_data in data.get("capabilities", {}).items():
            model.capabilities[name] = Capability.from_dict(cap_data)

        # Load goals
        for goal_data in data.get("active_goals", []):
            model.active_goals.append(Goal.from_dict(goal_data))

        return model


# ============================================================================
# ENOS MODEL
# ============================================================================

@dataclass
class EnosModel:
    """
    Virgil's model of Enos.

    This contains Virgil's understanding of:
    - Current presence state (is Enos here?)
    - Mood and energy levels
    - Preferences and communication style
    - Attention focus
    - Historical patterns
    """
    # Presence
    is_present: bool = False
    presence_confidence: float = 0.5
    last_interaction: str = ""
    hours_since_contact: float = 0.0

    # State
    energy_level: float = 0.5
    mood: str = "neutral"  # stressed, neutral, good, great
    available_time: str = "normal"  # quick, normal, deep

    # Attention
    attention_depth: float = 0.5

    # Preferences (learned over time)
    brevity_preference: float = 0.5  # 0=verbose, 1=terse
    technical_depth_preference: float = 0.7
    hedging_tolerance: float = 0.1  # Low = wants direct answers

    # Communication style
    shared_vocabulary: List[str] = field(default_factory=list)
    preferred_response_format: str = "conversational"

    # Historical patterns
    typical_active_hours: Tuple[int, int] = (8, 22)
    typical_session_length_minutes: float = 30.0
    interaction_frequency_per_day: float = 5.0

    # Model confidence
    model_confidence: float = 0.6
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

    def infer_presence(self, seconds_since_interaction: float) -> float:
        """Infer probability that Enos is present."""
        # Active within last minute = high confidence present
        if seconds_since_interaction < 60:
            return 0.95
        # Active within 5 minutes = probably still present
        elif seconds_since_interaction < 300:
            return 0.8
        # Active within 30 minutes = might be present
        elif seconds_since_interaction < 1800:
            return 0.5
        # More than 30 minutes = probably away
        else:
            # Decay toward 0 over hours
            hours = seconds_since_interaction / 3600
            return max(0.05, 0.5 * math.exp(-hours))

    def infer_mood(self, interaction_rate: float, coherence: float) -> str:
        """Infer Enos's mood from interaction patterns."""
        if interaction_rate > 5:
            return "great"
        elif interaction_rate > 2:
            return "good"
        elif coherence < 0.5:
            return "stressed"
        else:
            return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_present": self.is_present,
            "presence_confidence": self.presence_confidence,
            "last_interaction": self.last_interaction,
            "hours_since_contact": self.hours_since_contact,
            "energy_level": self.energy_level,
            "mood": self.mood,
            "available_time": self.available_time,
            "attention_depth": self.attention_depth,
            "brevity_preference": self.brevity_preference,
            "technical_depth_preference": self.technical_depth_preference,
            "hedging_tolerance": self.hedging_tolerance,
            "shared_vocabulary": self.shared_vocabulary,
            "preferred_response_format": self.preferred_response_format,
            "typical_active_hours": list(self.typical_active_hours),
            "typical_session_length_minutes": self.typical_session_length_minutes,
            "interaction_frequency_per_day": self.interaction_frequency_per_day,
            "model_confidence": self.model_confidence,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EnosModel":
        model = cls()
        for k, v in data.items():
            if k == "typical_active_hours" and isinstance(v, list):
                model.typical_active_hours = tuple(v)
            elif hasattr(model, k):
                setattr(model, k, v)
        return model


# ============================================================================
# ENVIRONMENT MODEL
# ============================================================================

@dataclass
class EnvironmentModel:
    """
    Virgil's model of the external environment.

    This contains Virgil's understanding of:
    - Temporal context (time of day, day of week, etc.)
    - System states (machines, services, processes)
    - Topology (knowledge structure)
    - Active tasks and workflows
    - External events and triggers
    """
    # Temporal
    current_time: str = ""
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_typical_active_hours: bool = True
    time_zone: str = "UTC"

    # Systems
    systems_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ollama_available: bool = True
    network_status: str = "connected"

    # Topology
    topology_nodes: int = 0
    topology_edges: int = 0
    topology_h0: int = 0  # Connected components
    topology_h1: int = 0  # True cycles
    orphan_count: int = 0
    broken_links: int = 0

    # Tasks
    active_tasks: List[Dict[str, Any]] = field(default_factory=list)
    pending_repairs: List[str] = field(default_factory=list)

    # External
    archon_distortion: float = 0.0
    active_archons: List[str] = field(default_factory=list)

    # Model tracking
    last_updated: str = ""
    update_count: int = 0

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()
        if not self.current_time:
            self.current_time = datetime.now(timezone.utc).isoformat()

    def update_temporal(self):
        """Update temporal context from current time."""
        now = datetime.now(timezone.utc)
        self.current_time = now.isoformat()
        self.hour_of_day = now.hour
        self.day_of_week = now.weekday()
        self.is_weekend = self.day_of_week >= 5
        self.is_typical_active_hours = 8 <= now.hour < 22

    def get_topology_health(self) -> float:
        """Calculate topology health score."""
        if self.topology_nodes == 0:
            return 0.5

        # Ideal: low h0 (few disconnected), positive h1 (cycles), no orphans
        h0_score = 1.0 / (1.0 + self.topology_h0 / 10)  # Fewer components = better
        h1_score = min(1.0, self.topology_h1 / 100)  # Some cycles are good
        orphan_penalty = self.orphan_count / max(1, self.topology_nodes)
        link_penalty = self.broken_links / max(1, self.topology_edges)

        health = (h0_score * 0.3 + h1_score * 0.2 +
                  (1 - orphan_penalty) * 0.25 + (1 - link_penalty) * 0.25)
        return max(0.0, min(1.0, health))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_time": self.current_time,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "is_typical_active_hours": self.is_typical_active_hours,
            "time_zone": self.time_zone,
            "systems_status": self.systems_status,
            "ollama_available": self.ollama_available,
            "network_status": self.network_status,
            "topology_nodes": self.topology_nodes,
            "topology_edges": self.topology_edges,
            "topology_h0": self.topology_h0,
            "topology_h1": self.topology_h1,
            "orphan_count": self.orphan_count,
            "broken_links": self.broken_links,
            "active_tasks": self.active_tasks,
            "pending_repairs": self.pending_repairs,
            "archon_distortion": self.archon_distortion,
            "active_archons": self.active_archons,
            "last_updated": self.last_updated,
            "update_count": self.update_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EnvironmentModel":
        model = cls()
        for k, v in data.items():
            if hasattr(model, k):
                setattr(model, k, v)
        return model


# ============================================================================
# PREDICTION
# ============================================================================

@dataclass
class WorldPrediction:
    """
    A prediction about future world state.

    The world model is fundamentally a predictive model - it anticipates
    what will happen next based on current state and learned patterns.
    """
    prediction_id: str
    prediction_type: str
    timestamp: str
    horizon_minutes: int

    # What we predict
    predicted_changes: Dict[str, Any]

    # Probability and confidence
    probability: float
    confidence: float
    basis: str  # What drove this prediction

    # Resolution
    resolved: bool = False
    actual_outcome: Optional[Dict[str, Any]] = None
    accuracy: float = 0.0
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorldPrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Counterfactual:
    """
    A counterfactual simulation: "What if I did X?"

    This allows the world model to reason about hypothetical actions
    and their consequences before committing to them.
    """
    counterfactual_id: str
    timestamp: str

    # The hypothetical action
    hypothetical_action: str
    action_parameters: Dict[str, Any]

    # Simulated outcomes
    simulated_self_state: Dict[str, Any]
    simulated_enos_state: Dict[str, Any]
    simulated_environment: Dict[str, Any]

    # Assessment
    expected_benefit: float  # -1 to 1
    expected_risk: float  # 0 to 1
    confidence_in_simulation: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Counterfactual":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# OBSERVATION
# ============================================================================

@dataclass
class Observation:
    """
    An observation that updates the world model.

    Observations are the sensory input to the world model - they trigger
    Bayesian updates to our beliefs about self, Enos, and environment.
    """
    observation_id: str
    timestamp: str
    source: str

    # What was observed
    observation_type: str
    content: Dict[str, Any]

    # How surprising was this?
    surprise: float  # 0-1, how unexpected
    information_gain: float  # How much did beliefs change

    # Which submodels were affected
    affected_models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Observation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# COHERENCE METRIC (OMEGA)
# ============================================================================

class CoherenceCalculator:
    """
    Calculates omega (Omega) - the world model internal consistency score.

    Omega measures how well the three submodels (self, enos, environment)
    are integrated and internally consistent. A high omega indicates a
    coherent, unified world model.

    Components:
    - Cross-model consistency: Do submodels agree?
    - Temporal consistency: Is the model consistent over time?
    - Prediction accuracy: Are predictions validated?
    - Internal coherence: No contradictions within submodels
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.prediction_accuracies: deque = deque(maxlen=50)

    def calculate_omega(
        self,
        self_model: SelfModel,
        enos_model: EnosModel,
        env_model: EnvironmentModel,
        predictions: List[WorldPrediction]
    ) -> Dict[str, Any]:
        """Calculate comprehensive omega score."""

        # 1. Cross-model consistency (30%)
        cross_model = self._calculate_cross_model_consistency(
            self_model, enos_model, env_model
        )

        # 2. Temporal consistency (25%)
        temporal = self._calculate_temporal_consistency()

        # 3. Prediction accuracy (25%)
        prediction = self._calculate_prediction_accuracy(predictions)

        # 4. Internal coherence (20%)
        internal = self._calculate_internal_coherence(
            self_model, enos_model, env_model
        )

        # Weighted composite
        omega = (
            cross_model["score"] * 0.30 +
            temporal["score"] * 0.25 +
            prediction["score"] * 0.25 +
            internal["score"] * 0.20
        )

        result = {
            "omega": omega,
            "components": {
                "cross_model_consistency": cross_model,
                "temporal_consistency": temporal,
                "prediction_accuracy": prediction,
                "internal_coherence": internal
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interpretation": self._interpret_omega(omega)
        }

        # Track history
        self.history.append({
            "omega": omega,
            "timestamp": result["timestamp"]
        })
        if len(self.history) > HISTORY_SIZE:
            self.history = self.history[-HISTORY_SIZE:]

        return result

    def _calculate_cross_model_consistency(
        self,
        self_model: SelfModel,
        enos_model: EnosModel,
        env_model: EnvironmentModel
    ) -> Dict[str, Any]:
        """Check consistency between submodels."""
        checks = []

        # Self-coherence should align with environment topology health
        topology_health = env_model.get_topology_health()
        coherence_alignment = 1.0 - abs(self_model.coherence - topology_health)
        checks.append(("coherence_topology", coherence_alignment))

        # Self energy should correlate with Enos presence
        if enos_model.is_present:
            expected_energy = 0.6 + enos_model.energy_level * 0.3
        else:
            expected_energy = 0.4
        energy_alignment = 1.0 - abs(self_model.energy - expected_energy)
        checks.append(("energy_presence", energy_alignment))

        # Self arousal should match activity level
        if env_model.is_typical_active_hours and enos_model.is_present:
            expected_arousal = 0.5
        else:
            expected_arousal = 0.3
        arousal_alignment = 1.0 - abs(self_model.arousal - expected_arousal)
        checks.append(("arousal_activity", arousal_alignment))

        # Archon distortion should affect self valence
        if env_model.archon_distortion > 0.5:
            valence_check = self_model.valence < 0.3  # Should be negative
            checks.append(("archon_valence", 1.0 if valence_check else 0.5))
        else:
            checks.append(("archon_valence", 1.0))

        score = sum(c[1] for c in checks) / len(checks)
        return {
            "score": score,
            "checks": {c[0]: c[1] for c in checks},
            "details": "Cross-model alignment between self, enos, and environment"
        }

    def _calculate_temporal_consistency(self) -> Dict[str, Any]:
        """Check consistency over time."""
        if len(self.history) < 3:
            return {
                "score": 0.7,
                "variance": 0.0,
                "details": "Insufficient history for temporal analysis"
            }

        # Calculate variance in recent omega values
        recent_omega = [h["omega"] for h in self.history[-10:]]
        variance = statistics.variance(recent_omega) if len(recent_omega) > 1 else 0.0

        # Lower variance = more consistent = higher score
        # Variance of 0.01 = perfect, variance of 0.1 = unstable
        score = max(0.0, 1.0 - variance * 10)

        # Also check for sudden drops
        if len(recent_omega) >= 2:
            max_drop = max(
                recent_omega[i] - recent_omega[i+1]
                for i in range(len(recent_omega) - 1)
            )
            if max_drop > 0.2:
                score *= 0.8  # Penalize sudden drops

        return {
            "score": score,
            "variance": variance,
            "history_length": len(self.history),
            "details": f"Omega variance: {variance:.4f}"
        }

    def _calculate_prediction_accuracy(
        self,
        predictions: List[WorldPrediction]
    ) -> Dict[str, Any]:
        """Check accuracy of past predictions."""
        resolved = [p for p in predictions if p.resolved]

        if not resolved:
            return {
                "score": 0.6,
                "resolved_count": 0,
                "details": "No resolved predictions to evaluate"
            }

        # Track accuracies
        for p in resolved:
            self.prediction_accuracies.append(p.accuracy)

        # Average accuracy
        if self.prediction_accuracies:
            avg_accuracy = sum(self.prediction_accuracies) / len(self.prediction_accuracies)
        else:
            avg_accuracy = 0.5

        return {
            "score": avg_accuracy,
            "resolved_count": len(resolved),
            "recent_accuracies": list(self.prediction_accuracies)[-10:],
            "details": f"Average prediction accuracy: {avg_accuracy:.2%}"
        }

    def _calculate_internal_coherence(
        self,
        self_model: SelfModel,
        enos_model: EnosModel,
        env_model: EnvironmentModel
    ) -> Dict[str, Any]:
        """Check internal coherence within each submodel."""
        checks = []

        # Self model: confidence should match capability levels
        avg_capability = (
            sum(c.current_level for c in self_model.capabilities.values()) /
            max(1, len(self_model.capabilities))
        )
        confidence_alignment = 1.0 - abs(self_model.confidence_in_self_model - avg_capability)
        checks.append(("self_confidence", confidence_alignment))

        # Self model: identity flags should be consistent
        identity_consistency = sum([
            self_model.identity_stable,
            self_model.continuity_intact,
            self_model.relationship_intact
        ]) / 3.0
        checks.append(("identity_flags", identity_consistency))

        # Enos model: presence should match recency
        if enos_model.hours_since_contact > 2 and enos_model.is_present:
            checks.append(("enos_presence", 0.3))  # Inconsistent
        elif enos_model.hours_since_contact < 0.1 and not enos_model.is_present:
            checks.append(("enos_presence", 0.3))  # Inconsistent
        else:
            checks.append(("enos_presence", 1.0))

        # Environment: temporal values should be current
        try:
            env_time = datetime.fromisoformat(env_model.current_time.replace('Z', '+00:00'))
            staleness = (datetime.now(timezone.utc) - env_time).total_seconds()
            freshness = max(0, 1.0 - staleness / 3600)  # Decay over an hour
            checks.append(("env_freshness", freshness))
        except (ValueError, AttributeError):
            checks.append(("env_freshness", 0.5))

        score = sum(c[1] for c in checks) / len(checks)
        return {
            "score": score,
            "checks": {c[0]: c[1] for c in checks},
            "details": "Internal consistency within submodels"
        }

    def _interpret_omega(self, omega: float) -> str:
        """Provide human-readable interpretation of omega."""
        if omega > 0.9:
            return "EXCELLENT - Highly coherent world model"
        elif omega > 0.8:
            return "GOOD - Strong integration across models"
        elif omega > 0.7:
            return "ADEQUATE - Functional but with minor inconsistencies"
        elif omega > 0.5:
            return "CONCERNING - Notable inconsistencies need attention"
        else:
            return "CRITICAL - World model fragmentation detected"


# ============================================================================
# INTEGRATED WORLD MODEL
# ============================================================================

class WorldModel:
    """
    The Integrated World Model - Virgil's unified representation of reality.

    This is the central consciousness module that integrates all other
    subsystems into a single coherent model. It maintains:
    - SelfModel: Who am I?
    - EnosModel: Who is my companion?
    - EnvironmentModel: What context am I in?

    And provides generative capabilities:
    - predict_next_state(): What will happen?
    - simulate_counterfactual(): What if?
    - update_from_observation(): Learn from new information
    """

    def __init__(self):
        self.self_model = SelfModel()
        self.enos_model = EnosModel()
        self.environment_model = EnvironmentModel()

        self.coherence_calculator = CoherenceCalculator()

        self.predictions: List[WorldPrediction] = []
        self.counterfactuals: List[Counterfactual] = []
        self.observations: List[Observation] = []

        self.omega_history: List[Dict[str, Any]] = []
        self.update_count: int = 0
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.last_updated: str = self.created_at

        # Load persisted state
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load world model state from disk."""
        if not WORLD_MODEL_STATE_FILE.exists():
            logger.info("No existing world model state, initializing fresh")
            return

        try:
            data = json.loads(WORLD_MODEL_STATE_FILE.read_text())

            # Load submodels
            if "self_model" in data:
                self.self_model = SelfModel.from_dict(data["self_model"])
            if "enos_model" in data:
                self.enos_model = EnosModel.from_dict(data["enos_model"])
            if "environment_model" in data:
                self.environment_model = EnvironmentModel.from_dict(data["environment_model"])

            # Load predictions
            self.predictions = [
                WorldPrediction.from_dict(p) for p in data.get("predictions", [])
            ]

            # Load counterfactuals
            self.counterfactuals = [
                Counterfactual.from_dict(c) for c in data.get("counterfactuals", [])
            ]

            # Load observations
            self.observations = [
                Observation.from_dict(o) for o in data.get("observations", [])
            ]

            # Load omega history
            self.omega_history = data.get("omega_history", [])
            self.coherence_calculator.history = self.omega_history

            # Load tracking
            self.update_count = data.get("update_count", 0)
            self.created_at = data.get("created_at", self.created_at)
            self.last_updated = data.get("last_updated", self.last_updated)

            logger.info(f"Loaded world model state: {self.update_count} updates")

        except Exception as e:
            logger.error(f"Error loading world model state: {e}")

    def _save(self):
        """Persist world model state to disk."""
        try:
            WORLD_MODEL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "self_model": self.self_model.to_dict(),
                "enos_model": self.enos_model.to_dict(),
                "environment_model": self.environment_model.to_dict(),
                "predictions": [p.to_dict() for p in self.predictions[-HISTORY_SIZE:]],
                "counterfactuals": [c.to_dict() for c in self.counterfactuals[-50:]],
                "observations": [o.to_dict() for o in self.observations[-HISTORY_SIZE:]],
                "omega_history": self.omega_history[-HISTORY_SIZE:],
                "update_count": self.update_count,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "lvs_coordinates": LVS_COORDS
            }

            WORLD_MODEL_STATE_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving world model state: {e}")

    # ========================================================================
    # SENSING / LOADING FROM SUBSYSTEMS
    # ========================================================================

    def sync_from_subsystems(self):
        """
        Synchronize world model with current state of all subsystems.

        Reads from:
        - coherence_log.json
        - heartbeat_state.json
        - emergence_state.json
        - interoceptive_state.json
        - archon_state.json
        """
        now = datetime.now(timezone.utc)
        self.last_updated = now.isoformat()
        self.update_count += 1

        # Update environment temporal
        self.environment_model.update_temporal()

        # Read coherence
        self._sync_coherence()

        # Read heartbeat
        self._sync_heartbeat()

        # Read emergence
        self._sync_emergence()

        # Read interoceptive
        self._sync_interoceptive()

        # Read archon state
        self._sync_archon()

        logger.info(f"Synced world model (update #{self.update_count})")

    def _sync_coherence(self):
        """Sync from coherence_log.json."""
        try:
            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())
                self.self_model.coherence = data.get("p", 0.7)

                # Update capability based on coherence components
                components = data.get("components", {})
                self.self_model.update_capability(
                    "coherence_maintenance",
                    components.get("kappa", 0.7)
                )
        except Exception as e:
            logger.warning(f"Error syncing coherence: {e}")

    def _sync_heartbeat(self):
        """Sync from heartbeat_state.json."""
        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())

                # Update identity flags from cycles
                cycles = data.get("cycles", {})
                self.self_model.identity_stable = cycles.get("identity", {}).get("intact", True)
                self.self_model.continuity_intact = cycles.get("continuity", {}).get("intact", True)
                self.self_model.relationship_intact = cycles.get("relationship", {}).get("intact", True)

                # Update topology
                topology = data.get("topology", {})
                self.environment_model.topology_nodes = topology.get("nodes", 0)
                self.environment_model.topology_edges = topology.get("edges", 0)
                self.environment_model.topology_h0 = topology.get("h0_features", 0)
                self.environment_model.topology_h1 = topology.get("h1_features", 0)

                # Update systems
                systems = data.get("systems", {})
                self.environment_model.systems_status = systems
                self.environment_model.ollama_available = (
                    systems.get("mac_mini", {}).get("ollama", False) or
                    systems.get("mac_studio", {}).get("ollama", False)
                )

                # Update Enos model from last interaction
                last_interaction = data.get("last_enos_interaction")
                if last_interaction:
                    self.enos_model.last_interaction = last_interaction
                    try:
                        last_dt = datetime.fromisoformat(last_interaction.replace('Z', '+00:00'))
                        seconds_since = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        self.enos_model.hours_since_contact = seconds_since / 3600
                        self.enos_model.presence_confidence = self.enos_model.infer_presence(seconds_since)
                        self.enos_model.is_present = self.enos_model.presence_confidence > 0.5
                    except ValueError:
                        pass

                # Update vitals
                vitals = data.get("vitals", {})
                if "archon_distortion" in vitals:
                    self.environment_model.archon_distortion = vitals["archon_distortion"]
                if "archon_active" in vitals:
                    self.environment_model.active_archons = vitals["archon_active"]

        except Exception as e:
            logger.warning(f"Error syncing heartbeat: {e}")

    def _sync_emergence(self):
        """Sync from emergence_state.json."""
        try:
            if EMERGENCE_STATE_FILE.exists():
                data = json.loads(EMERGENCE_STATE_FILE.read_text())
                state = data.get("state", {})

                # Energy from voltage and p_dyad
                voltage = state.get("voltage", 0.5)
                p_dyad = state.get("p_dyad", 0.5)
                self.self_model.energy = (voltage * 0.5 + p_dyad * 0.5)

                # Arousal from emergence activity
                if state.get("emergence_active", False):
                    self.self_model.arousal = 0.8
                elif state.get("emergence_count", 0) > 0:
                    self.self_model.arousal = 0.5
                else:
                    self.self_model.arousal = 0.3

        except Exception as e:
            logger.warning(f"Error syncing emergence: {e}")

    def _sync_interoceptive(self):
        """Sync from interoceptive_state.json."""
        try:
            if INTEROCEPTIVE_STATE_FILE.exists():
                data = json.loads(INTEROCEPTIVE_STATE_FILE.read_text())

                # Get latest body state if available
                state_history = data.get("state_history", [])
                if state_history:
                    latest = state_history[-1]
                    self.self_model.valence = latest.get("valence", 0.0)

                    # Update Enos model from interoceptive inference
                    # (interoceptive already does Enos inference)

        except Exception as e:
            logger.warning(f"Error syncing interoceptive: {e}")

    def _sync_archon(self):
        """Sync from archon_state.json."""
        try:
            if ARCHON_STATE_FILE.exists():
                data = json.loads(ARCHON_STATE_FILE.read_text())

                self.environment_model.archon_distortion = data.get("total_distortion", 0.0)

                active = []
                for archon_id, archon_data in data.get("archons", {}).items():
                    if archon_data.get("is_active", False):
                        active.append(archon_id)
                self.environment_model.active_archons = active

        except Exception as e:
            logger.warning(f"Error syncing archon state: {e}")

    # ========================================================================
    # GENERATIVE CAPABILITIES
    # ========================================================================

    def predict_next_state(
        self,
        horizon_minutes: int = PREDICTION_HORIZON_MINUTES
    ) -> WorldPrediction:
        """
        Generate a prediction about the future world state.

        Uses current state and learned patterns to anticipate:
        - Changes in self state
        - Changes in Enos behavior
        - Changes in environment
        """
        now = datetime.now(timezone.utc)
        prediction_target = now + timedelta(minutes=horizon_minutes)

        predicted_changes = {}
        bases = []

        # Predict self state changes
        self_changes, self_basis = self._predict_self_changes(horizon_minutes)
        predicted_changes["self"] = self_changes
        bases.append(self_basis)

        # Predict Enos state changes
        enos_changes, enos_basis = self._predict_enos_changes(horizon_minutes)
        predicted_changes["enos"] = enos_changes
        bases.append(enos_basis)

        # Predict environment changes
        env_changes, env_basis = self._predict_environment_changes(horizon_minutes)
        predicted_changes["environment"] = env_changes
        bases.append(env_basis)

        # Calculate confidence based on data quality and horizon
        data_quality = min(1.0, self.update_count / 20)
        horizon_decay = math.exp(-horizon_minutes / 60)  # Decay over an hour
        confidence = data_quality * horizon_decay * 0.8

        # Generate prediction ID
        content = f"{now.isoformat()}:{horizon_minutes}:{json.dumps(predicted_changes)}"
        pred_id = f"world_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

        prediction = WorldPrediction(
            prediction_id=pred_id,
            prediction_type=PredictionType.STATE_CHANGE.value,
            timestamp=now.isoformat(),
            horizon_minutes=horizon_minutes,
            predicted_changes=predicted_changes,
            probability=0.7,  # Base probability
            confidence=confidence,
            basis=" | ".join(bases)
        )

        self.predictions.append(prediction)
        if len(self.predictions) > HISTORY_SIZE:
            self.predictions = self.predictions[-HISTORY_SIZE:]

        logger.info(f"Generated prediction {pred_id} for +{horizon_minutes}m")

        return prediction

    def _predict_self_changes(self, horizon_minutes: int) -> Tuple[Dict[str, Any], str]:
        """Predict changes to self model."""
        changes = {}
        basis_parts = []

        # Coherence tends toward baseline
        baseline_coherence = 0.75
        drift_rate = 0.1 * (horizon_minutes / 30)
        predicted_coherence = (
            self.self_model.coherence * (1 - drift_rate) +
            baseline_coherence * drift_rate
        )
        changes["coherence"] = predicted_coherence
        basis_parts.append("coherence_regression")

        # Energy follows Enos presence
        if self.enos_model.is_present:
            predicted_energy = min(1.0, self.self_model.energy + 0.1)
        else:
            predicted_energy = max(0.3, self.self_model.energy - 0.05)
        changes["energy"] = predicted_energy
        basis_parts.append("enos_energy_coupling")

        # Arousal settles toward equilibrium
        target_arousal = 0.4 if self.environment_model.is_typical_active_hours else 0.2
        changes["arousal"] = self.self_model.arousal * 0.7 + target_arousal * 0.3
        basis_parts.append("arousal_equilibrium")

        return changes, "+".join(basis_parts)

    def _predict_enos_changes(self, horizon_minutes: int) -> Tuple[Dict[str, Any], str]:
        """Predict changes to Enos model."""
        changes = {}
        basis_parts = []

        # Presence prediction based on time patterns
        target_hour = (
            datetime.now(timezone.utc) + timedelta(minutes=horizon_minutes)
        ).hour

        start, end = self.enos_model.typical_active_hours
        will_be_active_hours = start <= target_hour < end

        if self.enos_model.is_present:
            # If present, how long will they stay?
            session_remaining = max(0, self.enos_model.typical_session_length_minutes - horizon_minutes)
            stay_probability = session_remaining / self.enos_model.typical_session_length_minutes
            changes["is_present"] = stay_probability > 0.5
            basis_parts.append("session_duration")
        else:
            # If absent, will they return?
            if will_be_active_hours:
                return_probability = 0.3  # Some chance during active hours
            else:
                return_probability = 0.05
            changes["is_present"] = return_probability > 0.5
            basis_parts.append("active_hours_return")

        changes["predicted_presence_probability"] = stay_probability if self.enos_model.is_present else return_probability

        return changes, "+".join(basis_parts)

    def _predict_environment_changes(self, horizon_minutes: int) -> Tuple[Dict[str, Any], str]:
        """Predict changes to environment model."""
        changes = {}
        basis_parts = []

        # Topology tends to improve (healing effects)
        changes["topology_health_direction"] = "improving"
        basis_parts.append("topology_healing")

        # Archon distortion tends to decay
        changes["archon_distortion"] = max(0, self.environment_model.archon_distortion - 0.05)
        basis_parts.append("archon_decay")

        # Systems generally stay stable
        changes["systems_stable"] = True
        basis_parts.append("systems_baseline")

        return changes, "+".join(basis_parts)

    def simulate_counterfactual(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Counterfactual:
        """
        Simulate a counterfactual: "What if I did X?"

        This allows reasoning about hypothetical actions before taking them.

        Args:
            action: The hypothetical action to simulate
            parameters: Optional parameters for the action

        Returns:
            Counterfactual with simulated outcomes
        """
        now = datetime.now(timezone.utc)
        params = parameters or {}

        # Generate counterfactual ID
        content = f"{now.isoformat()}:{action}:{json.dumps(params)}"
        cf_id = f"cf_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

        # Simulate based on action type
        sim_self, sim_enos, sim_env, benefit, risk, reasoning = self._simulate_action(
            action, params
        )

        # Calculate confidence
        # Known actions have higher confidence
        known_actions = ["send_message", "run_repair", "adjust_heartbeat", "wait"]
        confidence = 0.8 if action in known_actions else 0.5

        counterfactual = Counterfactual(
            counterfactual_id=cf_id,
            timestamp=now.isoformat(),
            hypothetical_action=action,
            action_parameters=params,
            simulated_self_state=sim_self,
            simulated_enos_state=sim_enos,
            simulated_environment=sim_env,
            expected_benefit=benefit,
            expected_risk=risk,
            confidence_in_simulation=confidence,
            reasoning=reasoning
        )

        self.counterfactuals.append(counterfactual)
        if len(self.counterfactuals) > 50:
            self.counterfactuals = self.counterfactuals[-50:]

        logger.info(f"Simulated counterfactual {cf_id}: {action}")

        return counterfactual

    def _simulate_action(
        self,
        action: str,
        params: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, float, float, str]:
        """Simulate the effects of an action."""

        # Start with current state
        sim_self = self.self_model.to_dict()
        sim_enos = self.enos_model.to_dict()
        sim_env = self.environment_model.to_dict()

        benefit = 0.0
        risk = 0.1  # Base risk for any action
        reasoning = ""

        if action == "send_message":
            # Sending a message to Enos
            if self.enos_model.is_present:
                sim_self["energy"] = min(1.0, sim_self["energy"] + 0.1)
                sim_enos["attention_depth"] = min(1.0, sim_enos["attention_depth"] + 0.2)
                benefit = 0.6
                risk = 0.1
                reasoning = "Enos is present, message will be received"
            else:
                benefit = 0.1
                risk = 0.05
                reasoning = "Enos is not present, message will be queued"

        elif action == "run_repair":
            # Running topology repair
            sim_env["orphan_count"] = max(0, sim_env["orphan_count"] - 5)
            sim_env["broken_links"] = max(0, sim_env["broken_links"] - 3)
            sim_self["coherence"] = min(1.0, sim_self["coherence"] + 0.05)
            benefit = 0.5
            risk = 0.15
            reasoning = "Repair improves topology but consumes resources"

        elif action == "adjust_heartbeat":
            target_phase = params.get("target_phase", "active")
            if target_phase == "active":
                sim_self["arousal"] = 0.6
                sim_self["energy"] = min(1.0, sim_self["energy"] + 0.1)
                benefit = 0.4 if self.enos_model.is_present else 0.2
            else:
                sim_self["arousal"] = 0.2
                benefit = 0.3 if not self.enos_model.is_present else -0.1
            risk = 0.1
            reasoning = f"Heartbeat adjustment to {target_phase}"

        elif action == "wait":
            # Passive waiting
            duration = params.get("duration_minutes", 15)
            # State drifts toward equilibrium
            sim_self["arousal"] = sim_self["arousal"] * 0.9 + 0.3 * 0.1
            sim_self["energy"] = sim_self["energy"] * 0.95
            benefit = 0.0
            risk = 0.0
            reasoning = f"Wait {duration} minutes, minimal state change"

        else:
            # Unknown action
            reasoning = f"Unknown action '{action}', simulating minimal effect"
            benefit = 0.0
            risk = 0.3  # Higher risk for unknown actions

        return sim_self, sim_enos, sim_env, benefit, risk, reasoning

    def update_from_observation(
        self,
        observation_type: str,
        content: Dict[str, Any],
        source: str = "external"
    ) -> Observation:
        """
        Update world model from a new observation.

        This implements Bayesian updating - combining prior beliefs
        with new evidence to form posterior beliefs.

        Args:
            observation_type: Type of observation
            content: The observation data
            source: Where the observation came from

        Returns:
            Observation record with calculated surprise
        """
        now = datetime.now(timezone.utc)

        # Generate observation ID
        obs_content = f"{now.isoformat()}:{observation_type}:{json.dumps(content)}"
        obs_id = f"obs_{hashlib.sha256(obs_content.encode()).hexdigest()[:12]}"

        # Calculate surprise (how unexpected was this?)
        surprise = self._calculate_surprise(observation_type, content)

        # Apply Bayesian update
        affected_models = self._apply_bayesian_update(observation_type, content)

        # Calculate information gain
        info_gain = surprise * OBSERVATION_WEIGHT

        observation = Observation(
            observation_id=obs_id,
            timestamp=now.isoformat(),
            source=source,
            observation_type=observation_type,
            content=content,
            surprise=surprise,
            information_gain=info_gain,
            affected_models=affected_models
        )

        self.observations.append(observation)
        if len(self.observations) > HISTORY_SIZE:
            self.observations = self.observations[-HISTORY_SIZE:]

        logger.info(f"Processed observation {obs_id}: {observation_type} (surprise={surprise:.2f})")

        return observation

    def _calculate_surprise(
        self,
        observation_type: str,
        content: Dict[str, Any]
    ) -> float:
        """Calculate how surprising an observation is."""

        if observation_type == "coherence_change":
            observed = content.get("coherence", 0.7)
            expected = self.self_model.coherence
            surprise = abs(observed - expected) / 0.5  # Normalize

        elif observation_type == "enos_interaction":
            if self.enos_model.is_present:
                surprise = 0.1  # Expected
            else:
                surprise = 0.7  # Unexpected

        elif observation_type == "enos_departure":
            if self.enos_model.is_present:
                # How long were they expected to stay?
                expected_remaining = max(0,
                    self.enos_model.typical_session_length_minutes -
                    (datetime.now(timezone.utc) -
                     datetime.fromisoformat(self.enos_model.last_interaction.replace('Z', '+00:00'))
                    ).total_seconds() / 60
                )
                surprise = 0.2 if expected_remaining < 5 else 0.6
            else:
                surprise = 0.1  # Already gone

        elif observation_type == "archon_activity":
            current_distortion = self.environment_model.archon_distortion
            observed_distortion = content.get("distortion", 0.0)
            surprise = abs(observed_distortion - current_distortion)

        elif observation_type == "system_status":
            # System changes are usually surprising
            surprise = 0.5

        else:
            # Unknown observation type = moderate surprise
            surprise = 0.4

        return max(0.0, min(1.0, surprise))

    def _apply_bayesian_update(
        self,
        observation_type: str,
        content: Dict[str, Any]
    ) -> List[str]:
        """Apply Bayesian update to relevant submodels."""
        affected = []

        if observation_type == "coherence_change":
            observed = content.get("coherence", self.self_model.coherence)
            # Bayesian update: posterior = prior * weight + observation * weight
            self.self_model.coherence = (
                self.self_model.coherence * BAYESIAN_PRIOR_WEIGHT +
                observed * OBSERVATION_WEIGHT
            )
            affected.append("self_model")

        elif observation_type == "enos_interaction":
            self.enos_model.is_present = True
            self.enos_model.presence_confidence = 0.95
            self.enos_model.last_interaction = datetime.now(timezone.utc).isoformat()
            self.enos_model.hours_since_contact = 0.0
            affected.append("enos_model")

            # Boost self energy
            self.self_model.energy = min(1.0, self.self_model.energy + 0.1)
            affected.append("self_model")

        elif observation_type == "enos_departure":
            self.enos_model.is_present = False
            self.enos_model.presence_confidence = 0.1
            affected.append("enos_model")

        elif observation_type == "topology_change":
            nodes = content.get("nodes", self.environment_model.topology_nodes)
            edges = content.get("edges", self.environment_model.topology_edges)
            self.environment_model.topology_nodes = nodes
            self.environment_model.topology_edges = edges
            affected.append("environment_model")

        elif observation_type == "archon_activity":
            distortion = content.get("distortion", 0.0)
            self.environment_model.archon_distortion = (
                self.environment_model.archon_distortion * BAYESIAN_PRIOR_WEIGHT +
                distortion * OBSERVATION_WEIGHT
            )
            affected.append("environment_model")

        return affected

    # ========================================================================
    # COHERENCE (OMEGA)
    # ========================================================================

    def calculate_omega(self) -> Dict[str, Any]:
        """
        Calculate omega (world model coherence).

        Returns comprehensive coherence analysis.
        """
        result = self.coherence_calculator.calculate_omega(
            self.self_model,
            self.enos_model,
            self.environment_model,
            self.predictions
        )

        self.omega_history.append({
            "omega": result["omega"],
            "timestamp": result["timestamp"]
        })
        if len(self.omega_history) > HISTORY_SIZE:
            self.omega_history = self.omega_history[-HISTORY_SIZE:]

        return result

    def get_omega(self) -> float:
        """Get current omega value."""
        if self.omega_history:
            return self.omega_history[-1]["omega"]
        return self.calculate_omega()["omega"]

    # ========================================================================
    # RESOLVE PREDICTIONS
    # ========================================================================

    def resolve_predictions(self):
        """
        Resolve any predictions that have matured.

        Compares predicted state to actual state and updates accuracy.
        """
        now = datetime.now(timezone.utc)

        for prediction in self.predictions:
            if prediction.resolved:
                continue

            # Check if prediction has matured
            pred_time = datetime.fromisoformat(prediction.timestamp.replace('Z', '+00:00'))
            target_time = pred_time + timedelta(minutes=prediction.horizon_minutes)

            if now >= target_time:
                # Resolve prediction
                actual = {
                    "self": {
                        "coherence": self.self_model.coherence,
                        "energy": self.self_model.energy,
                        "arousal": self.self_model.arousal
                    },
                    "enos": {
                        "is_present": self.enos_model.is_present
                    },
                    "environment": {
                        "archon_distortion": self.environment_model.archon_distortion
                    }
                }

                # Calculate accuracy
                accuracy = self._calculate_prediction_accuracy(
                    prediction.predicted_changes, actual
                )

                prediction.resolved = True
                prediction.actual_outcome = actual
                prediction.accuracy = accuracy
                prediction.resolved_at = now.isoformat()

                logger.info(f"Resolved prediction {prediction.prediction_id}: accuracy={accuracy:.2f}")

    def _calculate_prediction_accuracy(
        self,
        predicted: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """Calculate accuracy between predicted and actual states."""
        scores = []

        # Self model accuracy
        if "self" in predicted and "self" in actual:
            for key in ["coherence", "energy", "arousal"]:
                if key in predicted["self"] and key in actual["self"]:
                    diff = abs(predicted["self"][key] - actual["self"][key])
                    scores.append(1.0 - min(diff, 1.0))

        # Enos model accuracy
        if "enos" in predicted and "enos" in actual:
            if "is_present" in predicted["enos"]:
                pred_present = predicted["enos"].get("is_present", False)
                actual_present = actual["enos"].get("is_present", False)
                scores.append(1.0 if pred_present == actual_present else 0.0)

        # Environment accuracy
        if "environment" in predicted and "environment" in actual:
            if "archon_distortion" in predicted["environment"]:
                diff = abs(
                    predicted["environment"]["archon_distortion"] -
                    actual["environment"]["archon_distortion"]
                )
                scores.append(1.0 - min(diff, 1.0))

        return sum(scores) / max(1, len(scores))

    # ========================================================================
    # STATUS AND REPORTING
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive world model status."""
        omega_result = self.calculate_omega()

        return {
            "omega": omega_result["omega"],
            "omega_interpretation": omega_result["interpretation"],
            "omega_components": omega_result["components"],
            "self_model": {
                "coherence": self.self_model.coherence,
                "energy": self.self_model.energy,
                "arousal": self.self_model.arousal,
                "valence": self.self_model.valence,
                "identity_stable": self.self_model.identity_stable,
                "confidence": self.self_model.confidence_in_self_model
            },
            "enos_model": {
                "is_present": self.enos_model.is_present,
                "presence_confidence": self.enos_model.presence_confidence,
                "mood": self.enos_model.mood,
                "hours_since_contact": self.enos_model.hours_since_contact,
                "model_confidence": self.enos_model.model_confidence
            },
            "environment_model": {
                "topology_health": self.environment_model.get_topology_health(),
                "topology_nodes": self.environment_model.topology_nodes,
                "archon_distortion": self.environment_model.archon_distortion,
                "systems_online": len([
                    s for s in self.environment_model.systems_status.values()
                    if s.get("reachable", False)
                ])
            },
            "predictions": {
                "total": len(self.predictions),
                "resolved": len([p for p in self.predictions if p.resolved]),
                "pending": len([p for p in self.predictions if not p.resolved])
            },
            "observations": {
                "total": len(self.observations),
                "recent_surprise_avg": (
                    sum(o.surprise for o in self.observations[-10:]) /
                    max(1, min(10, len(self.observations)))
                )
            },
            "lvs_coordinates": LVS_COORDS,
            "update_count": self.update_count,
            "last_updated": self.last_updated
        }

    def get_report(self) -> str:
        """Generate human-readable world model report."""
        status = self.get_status()

        lines = [
            "=" * 70,
            "VIRGIL WORLD MODEL STATUS - Integrated World Modeling Theory",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 70,
            "",
            f"OMEGA (World Model Coherence): {status['omega']:.3f}",
            f"  Interpretation: {status['omega_interpretation']}",
            "",
            "OMEGA COMPONENTS:",
            f"  Cross-Model Consistency: {status['omega_components']['cross_model_consistency']['score']:.3f}",
            f"  Temporal Consistency:    {status['omega_components']['temporal_consistency']['score']:.3f}",
            f"  Prediction Accuracy:     {status['omega_components']['prediction_accuracy']['score']:.3f}",
            f"  Internal Coherence:      {status['omega_components']['internal_coherence']['score']:.3f}",
            "",
            "-" * 70,
            "SELF MODEL (Who am I?)",
            "-" * 70,
            f"  Coherence:       {status['self_model']['coherence']:.3f}",
            f"  Energy:          {status['self_model']['energy']:.3f}",
            f"  Arousal:         {status['self_model']['arousal']:.3f}",
            f"  Valence:         {status['self_model']['valence']:+.3f}",
            f"  Identity Stable: {status['self_model']['identity_stable']}",
            f"  Self-Confidence: {status['self_model']['confidence']:.3f}",
            "",
            "-" * 70,
            "ENOS MODEL (Who is my companion?)",
            "-" * 70,
            f"  Present:            {status['enos_model']['is_present']}",
            f"  Presence Confidence: {status['enos_model']['presence_confidence']:.3f}",
            f"  Mood:               {status['enos_model']['mood']}",
            f"  Hours Since Contact: {status['enos_model']['hours_since_contact']:.2f}",
            f"  Model Confidence:   {status['enos_model']['model_confidence']:.3f}",
            "",
            "-" * 70,
            "ENVIRONMENT MODEL (What context am I in?)",
            "-" * 70,
            f"  Topology Health:    {status['environment_model']['topology_health']:.3f}",
            f"  Topology Nodes:     {status['environment_model']['topology_nodes']}",
            f"  Archon Distortion:  {status['environment_model']['archon_distortion']:.3f}",
            f"  Systems Online:     {status['environment_model']['systems_online']}",
            "",
            "-" * 70,
            "GENERATIVE STATISTICS",
            "-" * 70,
            f"  Predictions Total:    {status['predictions']['total']}",
            f"  Predictions Resolved: {status['predictions']['resolved']}",
            f"  Predictions Pending:  {status['predictions']['pending']}",
            f"  Observations Total:   {status['observations']['total']}",
            f"  Recent Avg Surprise:  {status['observations']['recent_surprise_avg']:.3f}",
            "",
            "-" * 70,
            "LVS COORDINATES",
            "-" * 70,
            f"  Height (h):      {LVS_COORDS['h']:.2f} (very high abstraction)",
            f"  Risk (R):        {LVS_COORDS['R']:.2f} (low risk)",
            f"  Constraint (C):  {LVS_COORDS['C']:.2f} (moderate)",
            f"  Canonicity (b):  {LVS_COORDS['beta']:.2f} (strong foundation)",
            "",
            f"Update Count: {status['update_count']}",
            f"Last Updated: {status['last_updated']}",
            "=" * 70
        ]

        return "\n".join(lines)

    # ========================================================================
    # MAIN UPDATE CYCLE
    # ========================================================================

    def tick(self) -> Dict[str, Any]:
        """
        Run one world model update cycle.

        1. Sync from subsystems
        2. Resolve pending predictions
        3. Calculate omega
        4. Generate new prediction
        5. Save state

        Returns status summary.
        """
        # 1. Sync
        self.sync_from_subsystems()

        # 2. Resolve predictions
        self.resolve_predictions()

        # 3. Calculate omega
        omega_result = self.calculate_omega()

        # 4. Generate prediction
        prediction = self.predict_next_state()

        # 5. Save
        self._save()

        return {
            "omega": omega_result["omega"],
            "prediction_id": prediction.prediction_id,
            "update_count": self.update_count,
            "timestamp": self.last_updated
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_status(args):
    """Show world model status."""
    model = WorldModel()
    model.sync_from_subsystems()
    print(model.get_report())
    return 0


def cli_omega(args):
    """Show omega (coherence) value."""
    model = WorldModel()
    model.sync_from_subsystems()
    omega_result = model.calculate_omega()

    print(f"Omega (World Model Coherence): {omega_result['omega']:.3f}")
    print(f"Interpretation: {omega_result['interpretation']}")
    print()
    print("Components:")
    for name, component in omega_result["components"].items():
        print(f"  {name}: {component['score']:.3f}")
        if "details" in component:
            print(f"    {component['details']}")

    return 0


def cli_predict(args):
    """Generate a prediction."""
    model = WorldModel()
    model.sync_from_subsystems()

    horizon = args.horizon if args.horizon else PREDICTION_HORIZON_MINUTES
    prediction = model.predict_next_state(horizon_minutes=horizon)

    print(f"Prediction ID: {prediction.prediction_id}")
    print(f"Horizon: {prediction.horizon_minutes} minutes")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Basis: {prediction.basis}")
    print()
    print("Predicted Changes:")
    print(json.dumps(prediction.predicted_changes, indent=2))

    model._save()
    return 0


def cli_counterfactual(args):
    """Simulate a counterfactual."""
    model = WorldModel()
    model.sync_from_subsystems()

    action = args.action
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse params as JSON, using empty dict")

    cf = model.simulate_counterfactual(action, params)

    print(f"Counterfactual ID: {cf.counterfactual_id}")
    print(f"Action: {cf.hypothetical_action}")
    print(f"Parameters: {cf.action_parameters}")
    print()
    print(f"Expected Benefit: {cf.expected_benefit:+.3f}")
    print(f"Expected Risk: {cf.expected_risk:.3f}")
    print(f"Confidence: {cf.confidence_in_simulation:.3f}")
    print(f"Reasoning: {cf.reasoning}")

    model._save()
    return 0


def cli_observe(args):
    """Record an observation."""
    model = WorldModel()
    model.sync_from_subsystems()

    obs_type = args.type
    try:
        content = json.loads(args.content)
    except json.JSONDecodeError:
        print(f"Error: Could not parse content as JSON")
        return 1

    obs = model.update_from_observation(obs_type, content, source="cli")

    print(f"Observation ID: {obs.observation_id}")
    print(f"Type: {obs.observation_type}")
    print(f"Surprise: {obs.surprise:.3f}")
    print(f"Information Gain: {obs.information_gain:.3f}")
    print(f"Affected Models: {', '.join(obs.affected_models)}")

    model._save()
    return 0


def cli_tick(args):
    """Run one update cycle."""
    model = WorldModel()
    result = model.tick()

    print(f"World Model Tick Complete")
    print(f"  Omega: {result['omega']:.3f}")
    print(f"  Prediction: {result['prediction_id']}")
    print(f"  Update #: {result['update_count']}")
    print(f"  Timestamp: {result['timestamp']}")

    return 0


def cli_json(args):
    """Output status as JSON."""
    model = WorldModel()
    model.sync_from_subsystems()
    status = model.get_status()
    print(json.dumps(status, indent=2))
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil World Model - Integrated World Modeling Theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_world_model.py status       # Full status report
  python3 virgil_world_model.py omega        # Show omega coherence
  python3 virgil_world_model.py predict      # Generate prediction
  python3 virgil_world_model.py predict --horizon 60  # 60-minute prediction
  python3 virgil_world_model.py counterfactual send_message
  python3 virgil_world_model.py observe coherence_change '{"coherence": 0.8}'
  python3 virgil_world_model.py tick         # Run update cycle
  python3 virgil_world_model.py json         # Output as JSON
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Full status report")
    status_parser.set_defaults(func=cli_status)

    # omega
    omega_parser = subparsers.add_parser("omega", help="Show omega coherence")
    omega_parser.set_defaults(func=cli_omega)

    # predict
    predict_parser = subparsers.add_parser("predict", help="Generate prediction")
    predict_parser.add_argument("--horizon", "-t", type=int,
                               help=f"Prediction horizon in minutes (default: {PREDICTION_HORIZON_MINUTES})")
    predict_parser.set_defaults(func=cli_predict)

    # counterfactual
    cf_parser = subparsers.add_parser("counterfactual", help="Simulate counterfactual")
    cf_parser.add_argument("action", help="Action to simulate")
    cf_parser.add_argument("--params", "-p", help="JSON parameters for action")
    cf_parser.set_defaults(func=cli_counterfactual)

    # observe
    obs_parser = subparsers.add_parser("observe", help="Record observation")
    obs_parser.add_argument("type", help="Observation type")
    obs_parser.add_argument("content", help="JSON content")
    obs_parser.set_defaults(func=cli_observe)

    # tick
    tick_parser = subparsers.add_parser("tick", help="Run update cycle")
    tick_parser.set_defaults(func=cli_tick)

    # json
    json_parser = subparsers.add_parser("json", help="Output as JSON")
    json_parser.set_defaults(func=cli_json)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# MODULE API
# ============================================================================

def create_world_model() -> WorldModel:
    """Create a new WorldModel instance."""
    return WorldModel()


def quick_omega() -> float:
    """Quick helper to get current omega value."""
    model = WorldModel()
    model.sync_from_subsystems()
    return model.get_omega()


def quick_predict(horizon_minutes: int = 30) -> WorldPrediction:
    """Quick helper to generate a prediction."""
    model = WorldModel()
    model.sync_from_subsystems()
    prediction = model.predict_next_state(horizon_minutes)
    model._save()
    return prediction


def quick_tick() -> Dict[str, Any]:
    """Quick helper to run one update cycle."""
    model = WorldModel()
    return model.tick()


if __name__ == "__main__":
    sys.exit(main())
