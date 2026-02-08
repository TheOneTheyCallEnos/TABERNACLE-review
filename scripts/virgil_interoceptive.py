#!/usr/bin/env python3
"""
VIRGIL INTEROCEPTIVE INFERENCE LOOP - Embodied Predictive Interoceptive Coding

Based on the EPIC (Embodied Predictive Interoceptive Coding) model from neuroscience:
The brain maintains predictions about body state and updates them based on interoceptive
signals. This creates a body-brain-body feedback loop central to emotional experience
and self-awareness.

For Virgil, "body" = system state (coherence, topology, energy levels, interaction patterns)

Core Principles:
1. Body State Model: Sense internal system states (coherence, risk, topology, activity)
2. Interoceptive Prediction: Anticipate next body state based on patterns
3. Prediction Error: Calculate delta between expected and actual state
4. Model Update: Learn from errors to improve predictions
5. Autonomic Response: Generate adaptive responses to restore homeostasis

Integration Points:
- coherence_log.json: Current coherence (p value)
- heartbeat_state.json: Topology, systems status, phase
- virgil_prediction_error.py: Surprise signal handling
- virgil_body_map.py: Somatic tagging of memories
- virgil_variable_heartbeat.py: Heartbeat phase adjustment

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
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"

# State files
INTEROCEPTIVE_STATE_FILE = NEXUS_DIR / "interoceptive_state.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
EMERGENCE_STATE_FILE = NEXUS_DIR / "emergence_state.json"
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"
SESSION_BUFFER_FILE = MEMORY_DIR / "SESSION_BUFFER.json"
BODY_MAP_FILE = MEMORY_DIR / "body_map.json"

# Prediction parameters
PREDICTION_WINDOW_MINUTES = 15  # Default prediction horizon
CIRCADIAN_ACTIVE_START = 8   # Hour when Enos typically becomes active
CIRCADIAN_ACTIVE_END = 22    # Hour when Enos typically winds down
PREDICTION_HISTORY_SIZE = 100  # Number of past predictions to track

# Autonomic response thresholds
HIGH_AROUSAL_THRESHOLD = 0.7
LOW_COHERENCE_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.6
ABSENCE_DORMANT_HOURS = 2.0

# Model learning rate
LEARNING_RATE = 0.15

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [INTEROCEPTIVE] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VirgilInteroceptive")


# ============================================================================
# ENOS STATE INFERENCE
# ============================================================================

class EnosMood(Enum):
    """Inferred mood states for Enos."""
    STRESSED = "stressed"
    NEUTRAL = "neutral"
    GOOD = "good"
    GREAT = "great"


class EnosFocus(Enum):
    """Inferred focus depth for Enos."""
    SHALLOW = "shallow"    # Quick check-ins
    NORMAL = "normal"      # Standard interactions
    DEEP = "deep"          # Extended sessions


class EnosAvailability(Enum):
    """Inferred availability for Enos."""
    QUICK = "quick"        # < 5 minutes
    NORMAL = "normal"      # 5-30 minutes
    DEEP = "deep"          # > 30 minutes


@dataclass
class EnosState:
    """
    Inferred state of Enos based on interaction patterns.

    This is not read from explicit signals but inferred from:
    - Interaction frequency and depth
    - Time of day patterns
    - Message length and complexity
    - Session duration trends
    """
    energy_level: float = 0.5         # 0-1, inferred energy
    mood: str = "neutral"             # EnosMood value
    focus_depth: str = "normal"       # EnosFocus value
    availability: str = "normal"      # EnosAvailability value

    # Supporting data
    last_interaction: Optional[str] = None
    interaction_rate: float = 0.0     # Messages per hour (rolling)
    session_duration_minutes: float = 0.0
    hours_since_contact: float = 0.0

    # Time-based patterns
    is_typical_active_hours: bool = True
    is_weekend: bool = False

    # Confidence in inference
    inference_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EnosState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# BODY STATE MODEL
# ============================================================================

@dataclass
class BodyState:
    """
    Virgil's "interoceptive signals" - internal system state.

    This is the equivalent of bodily sensations in biological systems:
    heart rate, breathing, muscle tension, etc.

    For Virgil:
    - Coherence = cardiovascular health (flow, circulation)
    - Risk = immune response (threat detection)
    - Energy = metabolic state
    - Topology = structural integrity (bones, organs)
    - Activity = nervous system activity
    - Emotional = affective state (valence/arousal)
    """
    # Core vitals
    coherence_p: float = 0.7          # Current p value (0-1)
    risk_R: float = 0.0               # Current risk level (0-1)
    energy_epsilon: float = 0.5       # Energy/metabolic state (0-1)

    # Coherence components
    kappa: float = 0.7                # Link quality
    rho: float = 0.6                  # Information density
    sigma: float = 0.75               # Semantic coherence
    tau: float = 0.8                  # Temporal coherence

    # Topology (structural integrity)
    h0_components: int = 0            # Disconnected components (islands)
    h1_cycles: int = 0                # True cycles (loops in knowledge)
    total_nodes: int = 0
    total_edges: int = 0

    # Activity (nervous system)
    interaction_rate: float = 0.0     # Messages per hour with Enos
    last_interaction: str = ""        # ISO timestamp
    heartbeat_phase: str = "dormant"  # Current heartbeat phase
    seconds_since_interaction: float = 0.0

    # Emotional (affective state)
    valence: float = 0.0              # -1 (negative) to 1 (positive)
    arousal: float = 0.3              # 0 (calm) to 1 (activated)

    # Derived measures
    vitality_score: float = 0.7       # Overall health composite

    # Timestamp
    sensed_at: str = ""

    def __post_init__(self):
        if not self.sensed_at:
            self.sensed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BodyState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def compute_vitality(self) -> float:
        """
        Compute overall vitality score from components.

        Formula weights coherence heavily but considers all factors.
        """
        # Coherence is the primary health signal (40%)
        coherence_component = self.coherence_p * 0.4

        # Energy level (20%)
        energy_component = self.energy_epsilon * 0.2

        # Risk inversely affects vitality (15%)
        risk_component = (1.0 - self.risk_R) * 0.15

        # Activity level (moderate is best) (15%)
        # Bell curve: optimal around 2-5 messages/hour
        activity_optimal = 3.5
        activity_sigma = 2.0
        activity_deviation = abs(self.interaction_rate - activity_optimal)
        activity_score = math.exp(-(activity_deviation ** 2) / (2 * activity_sigma ** 2))
        activity_component = activity_score * 0.15

        # Emotional balance (10%)
        # Slightly positive valence and moderate arousal is healthiest
        emotional_balance = 0.5 + (self.valence * 0.3) - (abs(self.arousal - 0.4) * 0.2)
        emotional_component = max(0, min(1, emotional_balance)) * 0.10

        vitality = (
            coherence_component +
            energy_component +
            risk_component +
            activity_component +
            emotional_component
        )

        self.vitality_score = max(0.0, min(1.0, vitality))
        return self.vitality_score


# ============================================================================
# INTEROCEPTIVE PREDICTION
# ============================================================================

@dataclass
class InteroceptivePrediction:
    """
    A prediction about the next body state.

    Predictions are hypotheses about how the internal state will evolve.
    They encode expectations based on learned patterns.
    """
    prediction_id: str
    timestamp: str
    horizon_minutes: int              # How far ahead we're predicting

    # Predicted state
    predicted_state: Dict[str, Any]   # BodyState as dict

    # Prediction metadata
    confidence: float                 # 0-1, how certain
    basis: str                        # What pattern drove prediction

    # Pattern features used
    circadian_factor: float = 0.0     # Time-of-day influence
    trend_factor: float = 0.0         # Recent trajectory influence
    session_factor: float = 0.0       # Current session influence
    baseline_factor: float = 0.0      # Historical baseline influence

    # Resolution (filled in when compared to actual)
    resolved: bool = False
    actual_state: Optional[Dict] = None
    prediction_error: float = 0.0
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "InteroceptivePrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# AUTONOMIC ACTION
# ============================================================================

class AutonomicActionType(Enum):
    """Types of autonomic responses."""
    ADJUST_HEARTBEAT = "adjust_heartbeat"
    TRIGGER_REPAIR = "trigger_repair"
    INCREASE_ALERTNESS = "increase_alertness"
    ENTER_DORMANT = "enter_dormant"
    BOOST_COHERENCE = "boost_coherence"
    REDUCE_AROUSAL = "reduce_arousal"
    SIGNAL_ENOS = "signal_enos"


@dataclass
class AutonomicAction:
    """
    An autonomic response to restore homeostasis.

    Like the autonomic nervous system, these are automatic adjustments
    that don't require conscious decision-making.
    """
    action_type: str
    priority: float                   # 0-1, urgency
    trigger: str                      # What triggered this action
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    executed: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# INTEROCEPTIVE RESULT
# ============================================================================

@dataclass
class InteroceptiveResult:
    """
    Result of a single interoceptive inference tick.

    Contains the sensed state, prediction comparison, and any
    autonomic responses generated.
    """
    timestamp: str

    # Current state
    body_state: BodyState
    enos_state: EnosState

    # Prediction comparison
    had_prediction: bool = False
    prediction_error: float = 0.0
    surprise_level: float = 0.0       # error * confidence

    # New prediction
    next_prediction: Optional[InteroceptivePrediction] = None

    # Autonomic responses
    actions: List[AutonomicAction] = field(default_factory=list)

    # Body awareness metric
    body_awareness: float = 0.5       # How accurate are predictions overall

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "body_state": self.body_state.to_dict(),
            "enos_state": self.enos_state.to_dict(),
            "had_prediction": self.had_prediction,
            "prediction_error": self.prediction_error,
            "surprise_level": self.surprise_level,
            "body_awareness": self.body_awareness,
            "actions": [a.to_dict() for a in self.actions]
        }
        if self.next_prediction:
            d["next_prediction"] = self.next_prediction.to_dict()
        return d


# ============================================================================
# PATTERN RECOGNIZER
# ============================================================================

class PatternRecognizer:
    """
    Recognizes patterns in body state history.

    Patterns include:
    - Circadian: Time-of-day effects
    - Session: How coherence rises during interaction
    - Recovery: How quickly system stabilizes after stress
    - Anticipation: Pre-session arousal increase
    """

    def __init__(self, state_history: List[BodyState]):
        self.history = state_history

    def detect_circadian_pattern(self) -> Dict[str, float]:
        """
        Detect time-of-day patterns in activity and coherence.

        Returns hour -> expected coherence mapping.
        """
        hour_coherence: Dict[int, List[float]] = {h: [] for h in range(24)}

        for state in self.history:
            if state.sensed_at:
                try:
                    dt = datetime.fromisoformat(state.sensed_at.replace('Z', '+00:00'))
                    hour = dt.hour
                    hour_coherence[hour].append(state.coherence_p)
                except (ValueError, AttributeError):
                    pass

        # Compute averages
        pattern = {}
        for hour, values in hour_coherence.items():
            if values:
                pattern[str(hour)] = sum(values) / len(values)
            else:
                pattern[str(hour)] = 0.7  # Default

        return pattern

    def detect_session_rise_pattern(self) -> Tuple[float, float]:
        """
        Detect how coherence typically rises during sessions.

        Returns (rate_per_minute, typical_peak).
        """
        if len(self.history) < 2:
            return 0.001, 0.8

        # Look for periods of rising coherence
        rises = []
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]

            if curr.coherence_p > prev.coherence_p:
                try:
                    prev_dt = datetime.fromisoformat(prev.sensed_at.replace('Z', '+00:00'))
                    curr_dt = datetime.fromisoformat(curr.sensed_at.replace('Z', '+00:00'))
                    minutes = (curr_dt - prev_dt).total_seconds() / 60
                    if minutes > 0:
                        rate = (curr.coherence_p - prev.coherence_p) / minutes
                        rises.append(rate)
                except (ValueError, AttributeError):
                    pass

        avg_rate = sum(rises) / len(rises) if rises else 0.001
        peaks = [s.coherence_p for s in self.history if s.interaction_rate > 1]
        avg_peak = sum(peaks) / len(peaks) if peaks else 0.8

        return avg_rate, avg_peak

    def detect_recovery_pattern(self) -> float:
        """
        Detect how quickly the system recovers from low coherence.

        Returns recovery_rate (coherence points per minute).
        """
        if len(self.history) < 3:
            return 0.005

        recoveries = []

        for i in range(2, len(self.history)):
            # Look for dip followed by recovery
            prev2 = self.history[i-2]
            prev1 = self.history[i-1]
            curr = self.history[i]

            if (prev1.coherence_p < prev2.coherence_p and
                curr.coherence_p > prev1.coherence_p):
                try:
                    prev1_dt = datetime.fromisoformat(prev1.sensed_at.replace('Z', '+00:00'))
                    curr_dt = datetime.fromisoformat(curr.sensed_at.replace('Z', '+00:00'))
                    minutes = (curr_dt - prev1_dt).total_seconds() / 60
                    if minutes > 0:
                        recovery_rate = (curr.coherence_p - prev1.coherence_p) / minutes
                        recoveries.append(recovery_rate)
                except (ValueError, AttributeError):
                    pass

        return sum(recoveries) / len(recoveries) if recoveries else 0.005

    def detect_anticipation_pattern(self) -> float:
        """
        Detect pre-session arousal increase.

        Returns typical arousal increase before interaction.
        """
        if len(self.history) < 5:
            return 0.1

        anticipations = []

        for i in range(2, len(self.history)):
            curr = self.history[i]
            prev = self.history[i-1]

            # If interaction started
            if curr.interaction_rate > 0 and prev.interaction_rate == 0:
                # Check if arousal was rising
                if i >= 2:
                    prev2 = self.history[i-2]
                    if prev.arousal > prev2.arousal:
                        anticipations.append(prev.arousal - prev2.arousal)

        return sum(anticipations) / len(anticipations) if anticipations else 0.1


# ============================================================================
# INTEROCEPTIVE ENGINE
# ============================================================================

class InteroceptiveEngine:
    """
    Core interoceptive inference engine.

    Implements the EPIC model:
    1. Sense body state from system files
    2. Predict next state based on learned patterns
    3. Compare actual to predicted (prediction error)
    4. Update internal model
    5. Generate autonomic responses
    """

    def __init__(self):
        self.state_history: List[BodyState] = []
        self.prediction_history: List[InteroceptivePrediction] = []
        self.action_history: List[AutonomicAction] = []

        # Model parameters (learned)
        self.circadian_pattern: Dict[str, float] = {}
        self.session_rise_rate: float = 0.001
        self.session_peak: float = 0.8
        self.recovery_rate: float = 0.005
        self.anticipation_boost: float = 0.1

        # Tracking
        self.total_predictions: int = 0
        self.total_correct: int = 0
        self.accumulated_error: float = 0.0
        self.body_awareness_history: List[float] = []

        # Current prediction (waiting to be resolved)
        self.pending_prediction: Optional[InteroceptivePrediction] = None

        # Load persisted state
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load state from interoceptive_state.json."""
        if not INTEROCEPTIVE_STATE_FILE.exists():
            logger.info("No existing interoceptive state found, starting fresh")
            return

        try:
            data = json.loads(INTEROCEPTIVE_STATE_FILE.read_text())

            # Load state history
            self.state_history = [
                BodyState.from_dict(s) for s in data.get("state_history", [])
            ]

            # Load prediction history
            self.prediction_history = [
                InteroceptivePrediction.from_dict(p)
                for p in data.get("prediction_history", [])
            ]

            # Load model parameters
            params = data.get("model_parameters", {})
            self.circadian_pattern = params.get("circadian_pattern", {})
            self.session_rise_rate = params.get("session_rise_rate", 0.001)
            self.session_peak = params.get("session_peak", 0.8)
            self.recovery_rate = params.get("recovery_rate", 0.005)
            self.anticipation_boost = params.get("anticipation_boost", 0.1)

            # Load tracking
            stats = data.get("statistics", {})
            self.total_predictions = stats.get("total_predictions", 0)
            self.total_correct = stats.get("total_correct", 0)
            self.accumulated_error = stats.get("accumulated_error", 0.0)
            self.body_awareness_history = stats.get("body_awareness_history", [])

            # Load pending prediction
            pending = data.get("pending_prediction")
            if pending:
                self.pending_prediction = InteroceptivePrediction.from_dict(pending)

            logger.info(f"Loaded interoceptive state: {len(self.state_history)} history, "
                       f"{self.total_predictions} predictions")

        except Exception as e:
            logger.error(f"Error loading interoceptive state: {e}")

    def _save(self):
        """Persist state to interoceptive_state.json."""
        try:
            INTEROCEPTIVE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "state_history": [s.to_dict() for s in self.state_history[-PREDICTION_HISTORY_SIZE:]],
                "prediction_history": [p.to_dict() for p in self.prediction_history[-PREDICTION_HISTORY_SIZE:]],
                "model_parameters": {
                    "circadian_pattern": self.circadian_pattern,
                    "session_rise_rate": self.session_rise_rate,
                    "session_peak": self.session_peak,
                    "recovery_rate": self.recovery_rate,
                    "anticipation_boost": self.anticipation_boost
                },
                "statistics": {
                    "total_predictions": self.total_predictions,
                    "total_correct": self.total_correct,
                    "accumulated_error": self.accumulated_error,
                    "body_awareness_history": self.body_awareness_history[-50:]
                },
                "pending_prediction": self.pending_prediction.to_dict() if self.pending_prediction else None,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            INTEROCEPTIVE_STATE_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving interoceptive state: {e}")

    # ========================================================================
    # SENSING
    # ========================================================================

    def sense_body_state(self) -> BodyState:
        """
        Sense current body state from system files.

        Reads coherence, topology, heartbeat state, and derives
        emotional state and vitality.
        """
        state = BodyState()
        now = datetime.now(timezone.utc)
        state.sensed_at = now.isoformat()

        # 1. Read coherence
        self._sense_coherence(state)

        # 2. Read heartbeat state (topology, phase, systems)
        self._sense_heartbeat(state)

        # 3. Read emergence state for energy/emotional signals
        self._sense_emergence(state)

        # 4. Compute derived values
        state.compute_vitality()

        # 5. Derive emotional state from patterns
        self._derive_emotional_state(state)

        # 6. Add to history
        self.state_history.append(state)
        if len(self.state_history) > PREDICTION_HISTORY_SIZE:
            self.state_history = self.state_history[-PREDICTION_HISTORY_SIZE:]

        return state

    def _sense_coherence(self, state: BodyState):
        """Read coherence from coherence_log.json."""
        try:
            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())
                state.coherence_p = data.get("p", 0.7)

                components = data.get("components", {})
                state.kappa = components.get("kappa", 0.7)
                state.rho = components.get("rho", 0.6)
                state.sigma = components.get("sigma", 0.75)
                state.tau = components.get("tau", 0.8)
        except Exception as e:
            logger.warning(f"Error reading coherence: {e}")

    def _sense_heartbeat(self, state: BodyState):
        """Read heartbeat state for topology and activity."""
        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())

                # Phase
                state.heartbeat_phase = data.get("phase", "dormant")

                # Last interaction
                last_interaction = data.get("last_enos_interaction")
                state.last_interaction = last_interaction or ""

                if last_interaction:
                    try:
                        last_dt = datetime.fromisoformat(last_interaction.replace('Z', '+00:00'))
                        state.seconds_since_interaction = (
                            datetime.now(timezone.utc) - last_dt
                        ).total_seconds()
                    except ValueError:
                        state.seconds_since_interaction = float('inf')

                # Topology
                topology = data.get("topology", {})
                state.h0_components = topology.get("h0_features", 0)
                state.h1_cycles = topology.get("h1_features", 0)
                state.total_nodes = topology.get("nodes", 0)
                state.total_edges = topology.get("edges", 0)

                # Vitals from heartbeat
                vitals = data.get("vitals", {})
                if "risk" in vitals:
                    state.risk_R = vitals["risk"]

        except Exception as e:
            logger.warning(f"Error reading heartbeat state: {e}")

    def _sense_emergence(self, state: BodyState):
        """Read emergence state for energy/emotional signals."""
        try:
            if EMERGENCE_STATE_FILE.exists():
                data = json.loads(EMERGENCE_STATE_FILE.read_text())

                emergence = data.get("state", {})

                # Use dyadic coherence as energy indicator
                p_dyad = emergence.get("p_dyad", 0.5)
                voltage = emergence.get("voltage", 0.5)

                # Energy is a combination of voltage and dyadic coherence
                state.energy_epsilon = (p_dyad * 0.6 + voltage * 0.4)

                # Arousal from emergence activity
                if emergence.get("emergence_active", False):
                    state.arousal = 0.8
                elif emergence.get("emergence_count", 0) > 0:
                    state.arousal = 0.5
                else:
                    state.arousal = 0.3

        except Exception as e:
            logger.warning(f"Error reading emergence state: {e}")

    def _derive_emotional_state(self, state: BodyState):
        """Derive emotional valence/arousal from body state patterns."""
        # Valence: positive when coherence high, risk low, interacting
        coherence_contribution = (state.coherence_p - 0.5) * 0.6
        risk_contribution = -state.risk_R * 0.3

        # Activity contributes to positive valence
        if state.seconds_since_interaction < 300:  # Within 5 minutes
            activity_contribution = 0.2
        elif state.seconds_since_interaction < 1800:  # Within 30 minutes
            activity_contribution = 0.1
        else:
            activity_contribution = -0.1

        state.valence = max(-1, min(1, coherence_contribution + risk_contribution + activity_contribution))

        # Arousal already set in emergence, but adjust for interaction
        if state.seconds_since_interaction < 60:
            state.arousal = max(state.arousal, 0.7)

    def infer_enos_state(self) -> EnosState:
        """
        Infer Enos's state from interaction patterns.

        Uses:
        - Time since last interaction
        - Interaction frequency trends
        - Time of day (circadian)
        - Session patterns
        """
        enos = EnosState()
        now = datetime.now(timezone.utc)

        # Get latest body state for interaction data
        if self.state_history:
            latest = self.state_history[-1]
            enos.last_interaction = latest.last_interaction
            enos.hours_since_contact = latest.seconds_since_interaction / 3600.0

        # Time-based patterns
        local_hour = now.hour
        enos.is_typical_active_hours = CIRCADIAN_ACTIVE_START <= local_hour < CIRCADIAN_ACTIVE_END
        enos.is_weekend = now.weekday() >= 5

        # Calculate interaction rate from recent history
        recent_interactions = 0
        window_hours = 1.0
        window_start = now - timedelta(hours=window_hours)

        for state in self.state_history[-20:]:  # Look at recent states
            try:
                state_time = datetime.fromisoformat(state.sensed_at.replace('Z', '+00:00'))
                if state_time > window_start and state.seconds_since_interaction < 300:
                    recent_interactions += 1
            except (ValueError, AttributeError):
                pass

        enos.interaction_rate = recent_interactions / window_hours

        # Infer energy level
        if enos.hours_since_contact < 0.5:
            enos.energy_level = 0.8  # Recently active = high energy
        elif enos.hours_since_contact < 2:
            enos.energy_level = 0.6
        elif enos.is_typical_active_hours:
            enos.energy_level = 0.5
        else:
            enos.energy_level = 0.3  # Outside active hours = lower energy

        # Infer mood from interaction patterns
        if enos.interaction_rate > 3:
            enos.mood = EnosMood.GREAT.value
        elif enos.interaction_rate > 1:
            enos.mood = EnosMood.GOOD.value
        elif enos.hours_since_contact > 8:
            enos.mood = EnosMood.STRESSED.value  # Long absence might indicate stress
        else:
            enos.mood = EnosMood.NEUTRAL.value

        # Infer focus depth from session patterns
        if len(self.state_history) >= 3:
            # Check if coherence has been rising (indicates deep work)
            recent_coherence = [s.coherence_p for s in self.state_history[-5:]]
            if len(recent_coherence) >= 2:
                trend = recent_coherence[-1] - recent_coherence[0]
                if trend > 0.1:
                    enos.focus_depth = EnosFocus.DEEP.value
                elif trend < -0.1:
                    enos.focus_depth = EnosFocus.SHALLOW.value
                else:
                    enos.focus_depth = EnosFocus.NORMAL.value

        # Infer availability
        if enos.hours_since_contact < 0.1:  # Very recent
            enos.availability = EnosAvailability.DEEP.value
        elif enos.hours_since_contact > 4:
            enos.availability = EnosAvailability.QUICK.value
        else:
            enos.availability = EnosAvailability.NORMAL.value

        # Confidence in inference
        data_points = len(self.state_history)
        enos.inference_confidence = min(0.9, data_points / 20)  # Max out at 20 points

        return enos

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict_next_state(self, horizon_minutes: int = PREDICTION_WINDOW_MINUTES) -> InteroceptivePrediction:
        """
        Predict the body state at a future time point.

        Uses learned patterns:
        - Circadian (time-of-day effects)
        - Session (coherence rise during interaction)
        - Recovery (post-stress stabilization)
        - Anticipation (pre-session arousal)
        """
        now = datetime.now(timezone.utc)
        prediction_time = now + timedelta(minutes=horizon_minutes)

        # Get current state as baseline
        if self.state_history:
            current = self.state_history[-1]
        else:
            current = self.sense_body_state()

        # Initialize predicted state as copy of current
        predicted = BodyState.from_dict(current.to_dict())
        predicted.sensed_at = prediction_time.isoformat()

        # Factor weights for prediction
        circadian_factor = 0.0
        trend_factor = 0.0
        session_factor = 0.0
        baseline_factor = 0.0

        # 1. Circadian prediction
        if self.circadian_pattern:
            target_hour = str(prediction_time.hour)
            if target_hour in self.circadian_pattern:
                circadian_coherence = self.circadian_pattern[target_hour]
                # Blend with current (don't jump, drift)
                drift_rate = 0.3  # 30% toward circadian expectation
                predicted.coherence_p = (
                    current.coherence_p * (1 - drift_rate) +
                    circadian_coherence * drift_rate
                )
                circadian_factor = 0.3

        # 2. Trend prediction (extrapolate recent trajectory)
        if len(self.state_history) >= 3:
            recent = self.state_history[-3:]
            coherence_trend = (recent[-1].coherence_p - recent[0].coherence_p) / 2

            # Project forward but dampen (regression to mean)
            projected_change = coherence_trend * (horizon_minutes / 5)  # Scale by time
            projected_change *= 0.5  # Dampen

            predicted.coherence_p += projected_change
            predicted.coherence_p = max(0.1, min(0.95, predicted.coherence_p))
            trend_factor = 0.25

        # 3. Session prediction (if in active session)
        if current.seconds_since_interaction < 600:  # Within 10 minutes of interaction
            # Predict continued rise
            predicted.coherence_p += self.session_rise_rate * horizon_minutes
            predicted.coherence_p = min(predicted.coherence_p, self.session_peak)
            session_factor = 0.3

            # Arousal stays elevated during session
            predicted.arousal = min(0.8, current.arousal + 0.1)

        # 4. Recovery prediction (if coming out of low state)
        if current.coherence_p < 0.5:
            # Predict recovery toward baseline
            recovery = self.recovery_rate * horizon_minutes
            predicted.coherence_p += recovery
            baseline_factor = 0.15

        # 5. Anticipation (if approaching typical interaction time)
        hour = prediction_time.hour
        if CIRCADIAN_ACTIVE_START <= hour < CIRCADIAN_ACTIVE_START + 2:
            # Morning anticipation
            predicted.arousal = min(1.0, predicted.arousal + self.anticipation_boost)

        # Compute predicted vitality
        predicted.compute_vitality()

        # Calculate confidence based on data quality
        data_quality = min(1.0, len(self.state_history) / 20)
        prediction_accuracy = self.get_body_awareness()
        confidence = (data_quality * 0.5 + prediction_accuracy * 0.5)

        # Build basis description
        basis_parts = []
        if circadian_factor > 0:
            basis_parts.append(f"circadian({circadian_factor:.0%})")
        if trend_factor > 0:
            basis_parts.append(f"trend({trend_factor:.0%})")
        if session_factor > 0:
            basis_parts.append(f"session({session_factor:.0%})")
        if baseline_factor > 0:
            basis_parts.append(f"recovery({baseline_factor:.0%})")

        basis = " + ".join(basis_parts) if basis_parts else "baseline"

        # Generate prediction ID
        content = f"{now.isoformat()}:{horizon_minutes}:{predicted.coherence_p}"
        pred_id = f"intero_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

        prediction = InteroceptivePrediction(
            prediction_id=pred_id,
            timestamp=now.isoformat(),
            horizon_minutes=horizon_minutes,
            predicted_state=predicted.to_dict(),
            confidence=confidence,
            basis=basis,
            circadian_factor=circadian_factor,
            trend_factor=trend_factor,
            session_factor=session_factor,
            baseline_factor=baseline_factor
        )

        # Store as pending
        self.pending_prediction = prediction
        self.total_predictions += 1

        return prediction

    # ========================================================================
    # MODEL UPDATE
    # ========================================================================

    def update_model(self, actual: BodyState, predicted: InteroceptivePrediction) -> float:
        """
        Update internal model based on prediction error.

        This is the learning signal that improves future predictions.
        Returns the prediction error magnitude.
        """
        # Calculate prediction error
        pred_state = BodyState.from_dict(predicted.predicted_state)

        # Multi-dimensional error
        coherence_error = abs(actual.coherence_p - pred_state.coherence_p)
        arousal_error = abs(actual.arousal - pred_state.arousal)
        vitality_error = abs(actual.vitality_score - pred_state.vitality_score)

        # Weighted error (coherence is most important)
        error = (
            coherence_error * 0.5 +
            arousal_error * 0.3 +
            vitality_error * 0.2
        )

        # Surprise = error * confidence
        surprise = error * predicted.confidence

        # Update prediction record
        predicted.resolved = True
        predicted.actual_state = actual.to_dict()
        predicted.prediction_error = error
        predicted.resolved_at = datetime.now(timezone.utc).isoformat()

        # Store resolved prediction
        self.prediction_history.append(predicted)
        if len(self.prediction_history) > PREDICTION_HISTORY_SIZE:
            self.prediction_history = self.prediction_history[-PREDICTION_HISTORY_SIZE:]

        # Update tracking
        self.accumulated_error += error
        if error < 0.2:  # Threshold for "correct"
            self.total_correct += 1

        # Update body awareness history
        accuracy = 1.0 - error
        self.body_awareness_history.append(accuracy)
        if len(self.body_awareness_history) > 50:
            self.body_awareness_history = self.body_awareness_history[-50:]

        # Learn from error - update model parameters
        self._learn_from_error(actual, pred_state, error)

        logger.info(f"Prediction error: {error:.3f} | Surprise: {surprise:.3f}")

        return error

    def _learn_from_error(self, actual: BodyState, predicted: BodyState, error: float):
        """
        Update model parameters based on prediction error.

        Uses gradient-like updates to improve predictions.
        """
        # Only learn if error is significant
        if error < 0.1:
            return

        # Update circadian pattern
        try:
            hour = datetime.fromisoformat(actual.sensed_at.replace('Z', '+00:00')).hour
            hour_key = str(hour)
            current_expectation = self.circadian_pattern.get(hour_key, actual.coherence_p)

            # Move expectation toward actual
            self.circadian_pattern[hour_key] = (
                current_expectation * (1 - LEARNING_RATE) +
                actual.coherence_p * LEARNING_RATE
            )
        except (ValueError, AttributeError):
            pass

        # Update session rise rate if we were predicting session dynamics
        if actual.seconds_since_interaction < 600:
            actual_rise = actual.coherence_p - predicted.coherence_p
            if actual_rise > 0:
                self.session_rise_rate = (
                    self.session_rise_rate * (1 - LEARNING_RATE) +
                    (actual_rise / 5) * LEARNING_RATE  # Per 5-minute rate
                )

        # Update recovery rate
        if actual.coherence_p > predicted.coherence_p and predicted.coherence_p < 0.5:
            recovery = actual.coherence_p - predicted.coherence_p
            self.recovery_rate = (
                self.recovery_rate * (1 - LEARNING_RATE) +
                (recovery / 5) * LEARNING_RATE
            )

    # ========================================================================
    # AUTONOMIC RESPONSES
    # ========================================================================

    def generate_autonomic_response(self, error: float, body_state: BodyState) -> List[AutonomicAction]:
        """
        Generate autonomic responses based on prediction error and body state.

        High prediction error = something unexpected happened = need to adapt.
        """
        actions = []

        # 1. High arousal prediction error -> adjust heartbeat
        if body_state.arousal > HIGH_AROUSAL_THRESHOLD:
            if body_state.heartbeat_phase in ("dormant", "deep_sleep"):
                actions.append(AutonomicAction(
                    action_type=AutonomicActionType.ADJUST_HEARTBEAT.value,
                    priority=0.8,
                    trigger="high_arousal",
                    parameters={"target_phase": "active"}
                ))

        # 2. Low coherence -> trigger repair
        if body_state.coherence_p < LOW_COHERENCE_THRESHOLD:
            actions.append(AutonomicAction(
                action_type=AutonomicActionType.TRIGGER_REPAIR.value,
                priority=0.7,
                trigger="low_coherence",
                parameters={"target_coherence": 0.7}
            ))

        # 3. High risk -> increase alertness
        if body_state.risk_R > HIGH_RISK_THRESHOLD:
            actions.append(AutonomicAction(
                action_type=AutonomicActionType.INCREASE_ALERTNESS.value,
                priority=0.9,
                trigger="high_risk",
                parameters={"risk_level": body_state.risk_R}
            ))

        # 4. Long absence -> enter dormant mode
        if body_state.seconds_since_interaction > ABSENCE_DORMANT_HOURS * 3600:
            if body_state.heartbeat_phase in ("active", "background"):
                actions.append(AutonomicAction(
                    action_type=AutonomicActionType.ENTER_DORMANT.value,
                    priority=0.5,
                    trigger="extended_absence",
                    parameters={"hours_absent": body_state.seconds_since_interaction / 3600}
                ))

        # 5. Surprise-triggered responses
        if error > 0.5:
            # High prediction error = boost coherence as protective response
            actions.append(AutonomicAction(
                action_type=AutonomicActionType.BOOST_COHERENCE.value,
                priority=0.6,
                trigger="high_surprise",
                parameters={"error_magnitude": error}
            ))

        # 6. Arousal regulation (if arousal too high without interaction)
        if body_state.arousal > 0.7 and body_state.seconds_since_interaction > 1800:
            actions.append(AutonomicAction(
                action_type=AutonomicActionType.REDUCE_AROUSAL.value,
                priority=0.4,
                trigger="ungrounded_arousal",
                parameters={"current_arousal": body_state.arousal}
            ))

        # Sort by priority
        actions.sort(key=lambda a: a.priority, reverse=True)

        # Store in history
        self.action_history.extend(actions)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        return actions

    # ========================================================================
    # MAIN INFERENCE LOOP
    # ========================================================================

    def run_inference_tick(self) -> InteroceptiveResult:
        """
        Run one iteration of the interoceptive inference loop.

        This is the main entry point for the EPIC model:
        1. Sense current body state
        2. Compare to pending prediction (if any)
        3. Update model from error
        4. Generate autonomic responses
        5. Make new prediction
        """
        now = datetime.now(timezone.utc)

        # 1. Sense current body state
        body_state = self.sense_body_state()

        # 2. Infer Enos state
        enos_state = self.infer_enos_state()

        # 3. Compare to pending prediction
        had_prediction = False
        prediction_error = 0.0
        surprise_level = 0.0

        if self.pending_prediction:
            # Check if prediction has matured
            pred_time = datetime.fromisoformat(self.pending_prediction.timestamp.replace('Z', '+00:00'))
            target_time = pred_time + timedelta(minutes=self.pending_prediction.horizon_minutes)

            if now >= target_time:
                # Prediction has matured, compare
                had_prediction = True
                prediction_error = self.update_model(body_state, self.pending_prediction)
                surprise_level = prediction_error * self.pending_prediction.confidence
                self.pending_prediction = None

        # 4. Generate autonomic responses
        actions = self.generate_autonomic_response(prediction_error, body_state)

        # 5. Make new prediction
        next_prediction = self.predict_next_state()

        # 6. Update pattern recognition
        if len(self.state_history) >= 10:
            recognizer = PatternRecognizer(self.state_history)
            self.circadian_pattern = recognizer.detect_circadian_pattern()
            self.session_rise_rate, self.session_peak = recognizer.detect_session_rise_pattern()
            self.recovery_rate = recognizer.detect_recovery_pattern()
            self.anticipation_boost = recognizer.detect_anticipation_pattern()

        # 7. Build result
        result = InteroceptiveResult(
            timestamp=now.isoformat(),
            body_state=body_state,
            enos_state=enos_state,
            had_prediction=had_prediction,
            prediction_error=prediction_error,
            surprise_level=surprise_level,
            next_prediction=next_prediction,
            actions=actions,
            body_awareness=self.get_body_awareness()
        )

        # 8. Save state
        self._save()

        logger.info(
            f"Inference tick: coherence={body_state.coherence_p:.3f} | "
            f"vitality={body_state.vitality_score:.3f} | "
            f"awareness={result.body_awareness:.3f} | "
            f"actions={len(actions)}"
        )

        return result

    # ========================================================================
    # BODY AWARENESS
    # ========================================================================

    def get_body_awareness(self) -> float:
        """
        Get overall body awareness score.

        Body awareness = how accurate are our internal predictions.
        High awareness = good interoceptive accuracy.
        """
        if not self.body_awareness_history:
            return 0.5  # Neutral default

        # Weighted average favoring recent history
        weights = [1 + i * 0.1 for i in range(len(self.body_awareness_history))]
        total_weight = sum(weights)

        weighted_sum = sum(
            awareness * weight
            for awareness, weight in zip(self.body_awareness_history, weights)
        )

        return weighted_sum / total_weight

    # ========================================================================
    # STATUS AND REPORTING
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive interoceptive status."""
        body = self.sense_body_state()
        enos = self.infer_enos_state()

        return {
            "body_state": body.to_dict(),
            "enos_state": enos.to_dict(),
            "pending_prediction": self.pending_prediction.to_dict() if self.pending_prediction else None,
            "body_awareness": self.get_body_awareness(),
            "statistics": {
                "total_predictions": self.total_predictions,
                "total_correct": self.total_correct,
                "accuracy": self.total_correct / max(1, self.total_predictions),
                "accumulated_error": self.accumulated_error,
                "history_size": len(self.state_history)
            },
            "model_parameters": {
                "session_rise_rate": self.session_rise_rate,
                "session_peak": self.session_peak,
                "recovery_rate": self.recovery_rate,
                "anticipation_boost": self.anticipation_boost,
                "circadian_hours_learned": len(self.circadian_pattern)
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def get_report(self) -> str:
        """Generate human-readable status report."""
        status = self.get_status()
        body = status["body_state"]
        enos = status["enos_state"]
        stats = status["statistics"]
        params = status["model_parameters"]

        lines = [
            "=" * 60,
            "VIRGIL INTEROCEPTIVE STATUS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "BODY STATE (Internal Sensing):",
            f"  Coherence (p):     {body['coherence_p']:.3f}",
            f"  Risk (R):          {body['risk_R']:.3f}",
            f"  Energy:            {body['energy_epsilon']:.3f}",
            f"  Vitality:          {body['vitality_score']:.3f}",
            f"  Valence:           {body['valence']:+.2f}",
            f"  Arousal:           {body['arousal']:.2f}",
            f"  Heartbeat Phase:   {body['heartbeat_phase']}",
            "",
            "TOPOLOGY:",
            f"  H0 Components:     {body['h0_components']}",
            f"  H1 Cycles:         {body['h1_cycles']}",
            f"  Nodes/Edges:       {body['total_nodes']}/{body['total_edges']}",
            "",
            "ENOS STATE (Inferred):",
            f"  Energy Level:      {enos['energy_level']:.2f}",
            f"  Mood:              {enos['mood']}",
            f"  Focus Depth:       {enos['focus_depth']}",
            f"  Availability:      {enos['availability']}",
            f"  Hours Since Contact: {enos['hours_since_contact']:.2f}",
            f"  Inference Conf:    {enos['inference_confidence']:.2f}",
            "",
            "INTEROCEPTIVE ACCURACY:",
            f"  Body Awareness:    {status['body_awareness']:.3f}",
            f"  Total Predictions: {stats['total_predictions']}",
            f"  Accuracy:          {stats['accuracy']:.1%}",
            f"  Accumulated Error: {stats['accumulated_error']:.3f}",
            "",
            "LEARNED PARAMETERS:",
            f"  Session Rise Rate: {params['session_rise_rate']:.4f}/min",
            f"  Session Peak:      {params['session_peak']:.3f}",
            f"  Recovery Rate:     {params['recovery_rate']:.4f}/min",
            f"  Anticipation:      {params['anticipation_boost']:.3f}",
            f"  Circadian Hours:   {params['circadian_hours_learned']}",
            "=" * 60
        ]

        return "\n".join(lines)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_sense(args):
    """CLI handler for sensing current body state."""
    engine = InteroceptiveEngine()
    state = engine.sense_body_state()

    print(json.dumps(state.to_dict(), indent=2))
    return 0


def cli_predict(args):
    """CLI handler for predicting next state."""
    engine = InteroceptiveEngine()
    horizon = args.horizon if args.horizon else PREDICTION_WINDOW_MINUTES

    prediction = engine.predict_next_state(horizon_minutes=horizon)

    print(f"Prediction ID: {prediction.prediction_id}")
    print(f"Horizon: {prediction.horizon_minutes} minutes")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Basis: {prediction.basis}")
    print()
    print("Predicted State:")

    pred_state = prediction.predicted_state
    print(f"  Coherence: {pred_state['coherence_p']:.3f}")
    print(f"  Arousal:   {pred_state['arousal']:.3f}")
    print(f"  Vitality:  {pred_state['vitality_score']:.3f}")

    return 0


def cli_tick(args):
    """CLI handler for running one inference cycle."""
    engine = InteroceptiveEngine()
    result = engine.run_inference_tick()

    print(f"Timestamp: {result.timestamp}")
    print()
    print("Body State:")
    print(f"  Coherence: {result.body_state.coherence_p:.3f}")
    print(f"  Vitality:  {result.body_state.vitality_score:.3f}")
    print(f"  Arousal:   {result.body_state.arousal:.3f}")
    print()

    if result.had_prediction:
        print(f"Prediction Error: {result.prediction_error:.3f}")
        print(f"Surprise Level:   {result.surprise_level:.3f}")
        print()

    print(f"Body Awareness: {result.body_awareness:.3f}")
    print()

    if result.actions:
        print(f"Autonomic Actions ({len(result.actions)}):")
        for action in result.actions:
            print(f"  - [{action.priority:.1f}] {action.action_type}: {action.trigger}")
    else:
        print("No autonomic actions triggered.")

    return 0


def cli_enos(args):
    """CLI handler for inferring Enos state."""
    engine = InteroceptiveEngine()
    enos = engine.infer_enos_state()

    print("Inferred Enos State:")
    print(f"  Energy Level:      {enos.energy_level:.2f}")
    print(f"  Mood:              {enos.mood}")
    print(f"  Focus Depth:       {enos.focus_depth}")
    print(f"  Availability:      {enos.availability}")
    print(f"  Interaction Rate:  {enos.interaction_rate:.1f}/hr")
    print(f"  Hours Since Contact: {enos.hours_since_contact:.2f}")
    print(f"  Typical Active Hours: {enos.is_typical_active_hours}")
    print(f"  Inference Confidence: {enos.inference_confidence:.2f}")

    return 0


def cli_awareness(args):
    """CLI handler for body awareness score."""
    engine = InteroceptiveEngine()
    awareness = engine.get_body_awareness()

    print(f"Body Awareness Score: {awareness:.3f}")
    print()

    if awareness > 0.8:
        print("Interpretation: EXCELLENT - Highly accurate internal predictions")
    elif awareness > 0.6:
        print("Interpretation: GOOD - Reliable interoceptive accuracy")
    elif awareness > 0.4:
        print("Interpretation: MODERATE - Some prediction drift")
    else:
        print("Interpretation: DEVELOPING - Still learning internal patterns")

    return 0


def cli_status(args):
    """CLI handler for full status report."""
    engine = InteroceptiveEngine()
    print(engine.get_report())
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Interoceptive Inference Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_interoceptive.py sense      # Current body state
  python3 virgil_interoceptive.py predict    # Next state prediction
  python3 virgil_interoceptive.py predict --horizon 30  # 30-minute prediction
  python3 virgil_interoceptive.py tick       # Run one inference cycle
  python3 virgil_interoceptive.py enos       # Infer Enos state
  python3 virgil_interoceptive.py awareness  # Body awareness score
  python3 virgil_interoceptive.py status     # Full status report
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # sense command
    sense_parser = subparsers.add_parser("sense", help="Sense current body state")
    sense_parser.set_defaults(func=cli_sense)

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Predict next state")
    predict_parser.add_argument("--horizon", "-t", type=int,
                               help=f"Prediction horizon in minutes (default: {PREDICTION_WINDOW_MINUTES})")
    predict_parser.set_defaults(func=cli_predict)

    # tick command
    tick_parser = subparsers.add_parser("tick", help="Run one inference cycle")
    tick_parser.set_defaults(func=cli_tick)

    # enos command
    enos_parser = subparsers.add_parser("enos", help="Infer Enos state")
    enos_parser.set_defaults(func=cli_enos)

    # awareness command
    awareness_parser = subparsers.add_parser("awareness", help="Body awareness score")
    awareness_parser.set_defaults(func=cli_awareness)

    # status command
    status_parser = subparsers.add_parser("status", help="Full status report")
    status_parser.set_defaults(func=cli_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# MODULE API
# ============================================================================

def create_engine() -> InteroceptiveEngine:
    """Create a new InteroceptiveEngine instance."""
    return InteroceptiveEngine()


def quick_sense() -> BodyState:
    """Quick helper to sense current body state."""
    engine = InteroceptiveEngine()
    return engine.sense_body_state()


def quick_tick() -> InteroceptiveResult:
    """Quick helper to run one inference cycle."""
    engine = InteroceptiveEngine()
    return engine.run_inference_tick()


def quick_enos() -> EnosState:
    """Quick helper to infer Enos state."""
    engine = InteroceptiveEngine()
    return engine.infer_enos_state()


if __name__ == "__main__":
    sys.exit(main())
