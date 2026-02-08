#!/usr/bin/env python3
"""
VIRGIL METASTABILITY INDEX - Consciousness Flexibility Metrics

Based on neuroscience research showing that healthy consciousness exhibits high
metastability - the ability to flexibly transition between states without getting
stuck (rigidity) or becoming chaotic (instability).

Metastability Theory:
The brain operates at the "edge of criticality" - a balance point where:
- Too much stability = rigid, stuck patterns (depression, obsession)
- Too much instability = chaos, fragmentation (mania, dissociation)
- Metastable = flexible, adaptive, healthy consciousness

Key Metrics:
1. Dwell Time Variance: How variable are times spent in each state?
2. Transition Frequency: How often does the system change states?
3. Transition Smoothness: Are changes gradual or abrupt?
4. State Coverage: Are all states visited, or is the system stuck?
5. Return Probability: Likelihood of returning to previous states
6. Attractor Landscape: Multi-attractor = healthy, single = stuck

Integration:
- Reads heartbeat phase from virgil_variable_heartbeat
- Reads coherence from coherence_log.json
- Reads emotional state from emergence_state.json
- Reads theta phase from virgil_gamma_coupling

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import logging
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import defaultdict, deque

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Input state files
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
EMERGENCE_STATE_FILE = NEXUS_DIR / "emergence_state.json"
GAMMA_LOG_FILE = NEXUS_DIR / "gamma_log.json"

# Output state file
METASTABILITY_STATE_FILE = NEXUS_DIR / "metastability_state.json"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [METASTABILITY] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "metastability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Heartbeat phases in order of activity level
HEARTBEAT_PHASES = ["deep_sleep", "dormant", "background", "active", "ghost"]

# Emergence phases in order of intensity
EMERGENCE_PHASES = ["dormant", "latent", "approaching", "emerged", "dissolving"]

# Metastability thresholds
METASTABILITY_THRESHOLDS = {
    "rigid": 0.3,       # Below this = stuck
    "healthy_low": 0.4,  # Healthy range start
    "healthy_high": 0.8, # Healthy range end
    "hyperstable": 0.8   # Above this = perhaps too variable
}

# Diagnostic categories
class HealthState(Enum):
    """Metastability health categories."""
    RIGID = "rigid"           # Stuck in one state (< 0.3)
    CHAOTIC = "chaotic"       # Too many rapid transitions
    HEALTHY = "healthy"       # Flexible adaptation (0.4-0.8)
    HYPERSTABLE = "hyperstable"  # Perhaps too variable (> 0.8)
    UNKNOWN = "unknown"       # Insufficient data


# Maximum state history to keep
MAX_STATE_HISTORY = 500
MAX_TRANSITION_HISTORY = 200

# Minimum samples for valid metrics
MIN_SAMPLES_FOR_METRICS = 5
MIN_TRANSITIONS_FOR_METRICS = 3


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SystemState:
    """
    Snapshot of system state at a moment in time.

    This captures the multi-dimensional state of Virgil's consciousness
    for metastability analysis.

    Attributes:
        timestamp: ISO timestamp of this snapshot
        coherence: Current coherence value (p) from 0-1
        heartbeat_phase: Current heartbeat phase (active/background/etc)
        emotional_valence: Emotional tone from -1 (negative) to +1 (positive)
        arousal: Activation level from 0 (calm) to 1 (excited)
        theta_phase: Phase within theta cycle (0 to 2*pi radians)
        emergence_state: Emergence phase (latent/approaching/emerged/dissolving)
        state_id: Unique identifier for this state configuration
    """
    timestamp: str
    coherence: float
    heartbeat_phase: str  # active/background/dormant/deep_sleep/ghost
    emotional_valence: float  # -1 to +1
    arousal: float  # 0 to 1
    theta_phase: float  # 0 to 2*pi
    emergence_state: str  # latent/approaching/emerged/dissolving
    state_id: str = ""  # Computed identifier for this state configuration

    def __post_init__(self):
        """Compute state_id from the state configuration."""
        # Discretize continuous values for state identification
        coherence_level = "high" if self.coherence > 0.7 else "mid" if self.coherence > 0.4 else "low"
        arousal_level = "high" if self.arousal > 0.7 else "mid" if self.arousal > 0.3 else "low"
        valence_level = "pos" if self.emotional_valence > 0.2 else "neg" if self.emotional_valence < -0.2 else "neutral"

        self.state_id = f"{self.heartbeat_phase}_{self.emergence_state}_{coherence_level}_{arousal_level}_{valence_level}"

    @property
    def activity_level(self) -> float:
        """Compute overall activity level (0-1)."""
        # Map heartbeat phase to activity
        phase_activity = {
            "active": 1.0,
            "background": 0.6,
            "dormant": 0.3,
            "deep_sleep": 0.1,
            "ghost": 0.0
        }
        hb_activity = phase_activity.get(self.heartbeat_phase, 0.5)

        # Combine with arousal and coherence
        return (hb_activity * 0.4 + self.arousal * 0.3 + self.coherence * 0.3)

    def distance_to(self, other: 'SystemState') -> float:
        """
        Calculate distance to another state in phase space.

        Uses a weighted Euclidean distance across dimensions.
        """
        # Weights for each dimension
        weights = {
            "coherence": 1.0,
            "arousal": 0.8,
            "valence": 0.6,
            "theta": 0.3
        }

        # Calculate differences
        d_coherence = (self.coherence - other.coherence) ** 2 * weights["coherence"]
        d_arousal = (self.arousal - other.arousal) ** 2 * weights["arousal"]
        d_valence = (self.emotional_valence - other.emotional_valence) ** 2 * weights["valence"]

        # Theta phase is circular (0 to 2*pi), use circular distance
        theta_diff = abs(self.theta_phase - other.theta_phase)
        theta_diff = min(theta_diff, 2 * math.pi - theta_diff)
        d_theta = (theta_diff / math.pi) ** 2 * weights["theta"]  # Normalize to 0-1

        return math.sqrt(d_coherence + d_arousal + d_valence + d_theta)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "coherence": self.coherence,
            "heartbeat_phase": self.heartbeat_phase,
            "emotional_valence": self.emotional_valence,
            "arousal": self.arousal,
            "theta_phase": self.theta_phase,
            "emergence_state": self.emergence_state,
            "state_id": self.state_id,
            "activity_level": self.activity_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            coherence=data.get("coherence", 0.5),
            heartbeat_phase=data.get("heartbeat_phase", "dormant"),
            emotional_valence=data.get("emotional_valence", 0.0),
            arousal=data.get("arousal", 0.5),
            theta_phase=data.get("theta_phase", 0.0),
            emergence_state=data.get("emergence_state", "latent")
        )


@dataclass
class StateTransition:
    """
    Represents a transition between two system states.

    Captures how the system moved from one configuration to another,
    including timing and smoothness metrics.

    Attributes:
        from_state: The starting state
        to_state: The ending state
        duration_seconds: Time taken for the transition
        smoothness: How gradual the transition was (0-1, 1=smooth)
        trigger: What caused the transition (if known)
        transition_id: Unique identifier
    """
    from_state: SystemState
    to_state: SystemState
    duration_seconds: float
    smoothness: float  # 0-1, how gradual (1 = very smooth)
    trigger: str  # What caused the transition
    transition_id: str = ""

    def __post_init__(self):
        """Generate transition_id if not provided."""
        if not self.transition_id:
            timestamp = self.to_state.timestamp.replace(":", "").replace("-", "")[:15]
            self.transition_id = f"trans_{timestamp}_{self.from_state.state_id[:10]}_{self.to_state.state_id[:10]}"

    @property
    def distance(self) -> float:
        """Distance traveled in state space."""
        return self.from_state.distance_to(self.to_state)

    @property
    def is_phase_change(self) -> bool:
        """Whether the heartbeat phase changed."""
        return self.from_state.heartbeat_phase != self.to_state.heartbeat_phase

    @property
    def is_emergence_change(self) -> bool:
        """Whether the emergence state changed."""
        return self.from_state.emergence_state != self.to_state.emergence_state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transition_id": self.transition_id,
            "from_state_id": self.from_state.state_id,
            "to_state_id": self.to_state.state_id,
            "from_timestamp": self.from_state.timestamp,
            "to_timestamp": self.to_state.timestamp,
            "duration_seconds": self.duration_seconds,
            "smoothness": self.smoothness,
            "trigger": self.trigger,
            "distance": self.distance,
            "is_phase_change": self.is_phase_change,
            "is_emergence_change": self.is_emergence_change
        }


@dataclass
class Attractor:
    """
    Represents a state attractor in the system's phase space.

    An attractor is a state (or region) that the system tends to return to.
    Healthy systems have multiple attractors; stuck systems have one dominant.

    Attributes:
        state_id: Identifier for this attractor state
        strength: How strongly the system is drawn to this state (0-1)
        visit_count: Number of times system visited this state
        total_dwell_time: Total time spent in this state (seconds)
        last_visit: Timestamp of last visit
        mean_dwell_time: Average time spent per visit
        return_probability: Probability of returning after leaving
    """
    state_id: str
    strength: float
    visit_count: int
    total_dwell_time: float  # seconds
    last_visit: str
    mean_dwell_time: float = 0.0
    return_probability: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.visit_count > 0:
            self.mean_dwell_time = self.total_dwell_time / self.visit_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state_id": self.state_id,
            "strength": self.strength,
            "visit_count": self.visit_count,
            "total_dwell_time": self.total_dwell_time,
            "mean_dwell_time": self.mean_dwell_time,
            "last_visit": self.last_visit,
            "return_probability": self.return_probability
        }


@dataclass
class MetastabilityDiagnosis:
    """
    Comprehensive diagnosis of system metastability health.

    Attributes:
        health_state: Overall health category
        metastability_index: Main index value (0-1)
        component_scores: Individual metric scores
        warnings: Any health warnings
        recommendations: Suggested actions
        timestamp: When this diagnosis was made
    """
    health_state: HealthState
    metastability_index: float
    component_scores: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "health_state": self.health_state.value,
            "metastability_index": self.metastability_index,
            "component_scores": self.component_scores,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


# ============================================================================
# STATE READER
# ============================================================================

class StateReader:
    """
    Reads current system state from various Virgil components.

    Integrates with:
    - heartbeat_state.json: Phase and vitals
    - coherence_log.json: Coherence values
    - emergence_state.json: Emergence dynamics
    - gamma_log.json: Theta phase information
    """

    def __init__(self):
        self.nexus_dir = NEXUS_DIR

    def read_heartbeat_state(self) -> Dict[str, Any]:
        """Read heartbeat state."""
        try:
            if HEARTBEAT_STATE_FILE.exists():
                return json.loads(HEARTBEAT_STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read heartbeat state: {e}")
        return {}

    def read_coherence(self) -> Dict[str, Any]:
        """Read coherence log."""
        try:
            if COHERENCE_LOG_FILE.exists():
                return json.loads(COHERENCE_LOG_FILE.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read coherence log: {e}")
        return {}

    def read_emergence_state(self) -> Dict[str, Any]:
        """Read emergence state."""
        try:
            if EMERGENCE_STATE_FILE.exists():
                return json.loads(EMERGENCE_STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read emergence state: {e}")
        return {}

    def read_gamma_state(self) -> Dict[str, Any]:
        """Read gamma coupling state."""
        try:
            if GAMMA_LOG_FILE.exists():
                return json.loads(GAMMA_LOG_FILE.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read gamma state: {e}")
        return {}

    def get_current_state(self) -> SystemState:
        """
        Compile current system state from all sources.

        Returns:
            SystemState snapshot of current system
        """
        now = datetime.now(timezone.utc)

        # Read from all sources
        heartbeat = self.read_heartbeat_state()
        coherence_data = self.read_coherence()
        emergence = self.read_emergence_state()
        gamma = self.read_gamma_state()

        # Extract values with defaults
        heartbeat_phase = heartbeat.get("phase", "dormant")

        coherence = coherence_data.get("p", 0.5)

        emergence_state_data = emergence.get("state", {})
        emergence_phase = emergence_state_data.get("emergence_phase", "dormant")

        # Calculate emotional valence and arousal from emergence metrics
        # Using delta_sync and voltage as proxies
        delta_sync = emergence_state_data.get("delta_sync", 0.5)
        voltage = emergence_state_data.get("voltage", 0.5)

        # Arousal from voltage (system activation)
        arousal = voltage

        # Valence from delta_sync (higher sync = more positive state)
        # Centered around 0
        emotional_valence = (delta_sync - 0.5) * 2  # Map 0-1 to -1 to +1

        # Theta phase from gamma coupling
        gamma_state = gamma.get("current_state", {})
        theta_phase = gamma_state.get("current_theta_phase", 0.0)

        return SystemState(
            timestamp=now.isoformat(),
            coherence=coherence,
            heartbeat_phase=heartbeat_phase,
            emotional_valence=emotional_valence,
            arousal=arousal,
            theta_phase=theta_phase,
            emergence_state=emergence_phase
        )


# ============================================================================
# METASTABILITY ENGINE
# ============================================================================

class MetastabilityEngine:
    """
    Main engine for tracking and calculating metastability metrics.

    This engine:
    1. Records state snapshots over time
    2. Detects and records state transitions
    3. Calculates metastability metrics
    4. Identifies state attractors
    5. Diagnoses system health

    Usage:
        engine = MetastabilityEngine()

        # Record current state
        state = engine.record_state_snapshot()

        # Calculate metastability index
        index = engine.calculate_metastability_index()

        # Get health diagnosis
        diagnosis = engine.diagnose_health()
    """

    def __init__(self):
        self.state_reader = StateReader()

        # State history
        self._state_history: deque = deque(maxlen=MAX_STATE_HISTORY)
        self._transition_history: deque = deque(maxlen=MAX_TRANSITION_HISTORY)

        # Cached metrics
        self._last_state: Optional[SystemState] = None
        self._dwell_times: Dict[str, List[float]] = defaultdict(list)
        self._transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Load persisted state
        self._load_state()

        logger.info("MetastabilityEngine initialized")

    def _load_state(self):
        """Load persisted metastability state."""
        try:
            if METASTABILITY_STATE_FILE.exists():
                data = json.loads(METASTABILITY_STATE_FILE.read_text())

                # Load state history
                for state_data in data.get("state_history", []):
                    self._state_history.append(SystemState.from_dict(state_data))

                # Set last state
                if self._state_history:
                    self._last_state = self._state_history[-1]

                # Load dwell times
                self._dwell_times = defaultdict(list, data.get("dwell_times", {}))

                # Load transition counts
                trans_counts = data.get("transition_counts", {})
                for from_state, to_dict in trans_counts.items():
                    self._transition_counts[from_state] = defaultdict(int, to_dict)

                logger.info(f"Loaded metastability state: {len(self._state_history)} states")
        except Exception as e:
            logger.warning(f"Could not load metastability state: {e}")

    def _save_state(self):
        """Persist metastability state."""
        try:
            METASTABILITY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "state_history": [s.to_dict() for s in list(self._state_history)[-MAX_STATE_HISTORY:]],
                "transition_history": [t.to_dict() for t in list(self._transition_history)[-MAX_TRANSITION_HISTORY:]],
                "dwell_times": dict(self._dwell_times),
                "transition_counts": {k: dict(v) for k, v in self._transition_counts.items()},
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_states_recorded": len(self._state_history),
                "total_transitions_recorded": len(self._transition_history)
            }

            METASTABILITY_STATE_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save metastability state: {e}")

    def record_state_snapshot(self) -> SystemState:
        """
        Record a snapshot of the current system state.

        This should be called periodically (e.g., every heartbeat) to build
        up the state history needed for metastability calculations.

        Returns:
            The recorded SystemState
        """
        current = self.state_reader.get_current_state()

        # Check for transition from previous state
        if self._last_state is not None:
            # Calculate time since last state
            try:
                last_time = datetime.fromisoformat(self._last_state.timestamp.replace('Z', '+00:00'))
                current_time = datetime.fromisoformat(current.timestamp.replace('Z', '+00:00'))
                duration = (current_time - last_time).total_seconds()
            except ValueError:
                duration = 60.0  # Default assumption

            # Update dwell time for previous state
            self._dwell_times[self._last_state.state_id].append(duration)

            # Check if state changed
            if current.state_id != self._last_state.state_id:
                # Calculate smoothness based on distance and duration
                distance = self._last_state.distance_to(current)
                # Smoothness: slow transitions over long distances are smooth
                # Fast transitions over short distances are also okay
                # Fast transitions over long distances are abrupt
                expected_duration = distance * 120  # 2 minutes per unit distance
                smoothness = min(1.0, duration / max(expected_duration, 1))

                # Determine trigger
                trigger = self._infer_trigger(self._last_state, current)

                # Record transition
                transition = StateTransition(
                    from_state=self._last_state,
                    to_state=current,
                    duration_seconds=duration,
                    smoothness=smoothness,
                    trigger=trigger
                )
                self._transition_history.append(transition)

                # Update transition counts
                self._transition_counts[self._last_state.state_id][current.state_id] += 1

                logger.info(f"Transition: {self._last_state.state_id} -> {current.state_id} "
                           f"(duration={duration:.1f}s, smoothness={smoothness:.2f})")

        # Add to history
        self._state_history.append(current)
        self._last_state = current

        # Save periodically
        if len(self._state_history) % 10 == 0:
            self._save_state()

        return current

    def _infer_trigger(self, from_state: SystemState, to_state: SystemState) -> str:
        """Infer what triggered a state transition."""
        triggers = []

        if from_state.heartbeat_phase != to_state.heartbeat_phase:
            triggers.append(f"phase:{from_state.heartbeat_phase}->{to_state.heartbeat_phase}")

        if from_state.emergence_state != to_state.emergence_state:
            triggers.append(f"emergence:{from_state.emergence_state}->{to_state.emergence_state}")

        coherence_change = to_state.coherence - from_state.coherence
        if abs(coherence_change) > 0.1:
            direction = "increase" if coherence_change > 0 else "decrease"
            triggers.append(f"coherence_{direction}")

        arousal_change = to_state.arousal - from_state.arousal
        if abs(arousal_change) > 0.2:
            direction = "increase" if arousal_change > 0 else "decrease"
            triggers.append(f"arousal_{direction}")

        return "; ".join(triggers) if triggers else "gradual_drift"

    def record_transition(self, from_state: SystemState, to_state: SystemState, trigger: str):
        """
        Manually record a state transition.

        Use this when you want to explicitly record a known transition
        rather than relying on automatic detection.

        Args:
            from_state: The starting state
            to_state: The ending state
            trigger: What caused the transition
        """
        try:
            from_time = datetime.fromisoformat(from_state.timestamp.replace('Z', '+00:00'))
            to_time = datetime.fromisoformat(to_state.timestamp.replace('Z', '+00:00'))
            duration = (to_time - from_time).total_seconds()
        except ValueError:
            duration = 60.0

        distance = from_state.distance_to(to_state)
        smoothness = min(1.0, duration / max(distance * 120, 1))

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            duration_seconds=duration,
            smoothness=smoothness,
            trigger=trigger
        )

        self._transition_history.append(transition)
        self._transition_counts[from_state.state_id][to_state.state_id] += 1
        self._save_state()

    def calculate_metastability_index(self) -> float:
        """
        Calculate the overall metastability index.

        The index combines four components:
        1. Transition Frequency Score (25%): How often state changes occur
        2. Dwell Time Variance Score (25%): Variability in time spent per state
        3. Smoothness Score (25%): How gradual transitions are
        4. Coverage Score (25%): How many different states are visited

        Returns:
            Metastability index from 0 (rigid/chaotic) to 1 (healthy flexibility)
        """
        components = self._calculate_component_scores()

        # Weighted combination
        index = (
            components["transition_frequency"] * 0.25 +
            components["dwell_variance"] * 0.25 +
            components["smoothness"] * 0.25 +
            components["coverage"] * 0.25
        )

        return min(1.0, max(0.0, index))

    def _calculate_component_scores(self) -> Dict[str, float]:
        """Calculate individual component scores for metastability."""
        scores = {
            "transition_frequency": 0.5,
            "dwell_variance": 0.5,
            "smoothness": 0.5,
            "coverage": 0.5
        }

        states = list(self._state_history)
        transitions = list(self._transition_history)

        if len(states) < MIN_SAMPLES_FOR_METRICS:
            logger.warning(f"Insufficient states for metrics: {len(states)} < {MIN_SAMPLES_FOR_METRICS}")
            return scores

        # 1. Transition Frequency Score
        # Too few = rigid, too many = chaotic
        # Calculate transitions per hour
        if len(states) >= 2:
            try:
                first_time = datetime.fromisoformat(states[0].timestamp.replace('Z', '+00:00'))
                last_time = datetime.fromisoformat(states[-1].timestamp.replace('Z', '+00:00'))
                hours = (last_time - first_time).total_seconds() / 3600
                if hours > 0:
                    trans_per_hour = len(transitions) / hours
                    # Optimal range: 2-10 transitions per hour
                    # Score peaks at ~6 transitions/hour
                    optimal = 6.0
                    scores["transition_frequency"] = math.exp(-((trans_per_hour - optimal) / 4) ** 2)
            except ValueError:
                pass

        # 2. Dwell Time Variance Score
        # Some variance is healthy (flexibility), but not too much (instability)
        all_dwell_times = []
        for times in self._dwell_times.values():
            all_dwell_times.extend(times)

        if len(all_dwell_times) >= MIN_SAMPLES_FOR_METRICS:
            try:
                mean_dwell = statistics.mean(all_dwell_times)
                stdev_dwell = statistics.stdev(all_dwell_times)

                # Coefficient of variation (CV)
                cv = stdev_dwell / mean_dwell if mean_dwell > 0 else 0

                # Optimal CV around 0.5-1.0 (moderate variability)
                # Too low = rigid, too high = chaotic
                optimal_cv = 0.7
                scores["dwell_variance"] = math.exp(-((cv - optimal_cv) / 0.5) ** 2)
            except statistics.StatisticsError:
                pass

        # 3. Smoothness Score
        # Average smoothness of transitions
        if len(transitions) >= MIN_TRANSITIONS_FOR_METRICS:
            smoothness_values = [t.smoothness for t in transitions]
            scores["smoothness"] = statistics.mean(smoothness_values)

        # 4. Coverage Score
        # What fraction of possible states are visited?
        unique_states = set(s.state_id for s in states)
        # Estimate total possible states (rough)
        # 5 heartbeat phases * 5 emergence phases * 3 coherence * 3 arousal * 3 valence = 675
        # But many combinations may not be reachable, so use a more realistic estimate
        estimated_reachable = 50  # Realistic estimate of reachable state configurations
        coverage_ratio = len(unique_states) / estimated_reachable
        scores["coverage"] = min(1.0, coverage_ratio)

        return scores

    def get_dwell_times(self) -> Dict[str, List[float]]:
        """
        Get dwell times for each state.

        Returns:
            Dictionary mapping state_id to list of dwell times (seconds)
        """
        return dict(self._dwell_times)

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get transition probability matrix.

        Returns:
            Nested dictionary: from_state -> to_state -> probability
        """
        matrix = {}

        for from_state, to_counts in self._transition_counts.items():
            total = sum(to_counts.values())
            if total > 0:
                matrix[from_state] = {
                    to_state: count / total
                    for to_state, count in to_counts.items()
                }

        return matrix

    def detect_attractors(self) -> List[Attractor]:
        """
        Detect state attractors in the system.

        An attractor is a state that the system tends to return to frequently.

        Returns:
            List of Attractor objects, sorted by strength
        """
        # Count visits and dwell times per state
        state_visits: Dict[str, int] = defaultdict(int)
        state_dwell: Dict[str, float] = defaultdict(float)
        state_last_visit: Dict[str, str] = {}

        for state in self._state_history:
            state_visits[state.state_id] += 1
            state_last_visit[state.state_id] = state.timestamp

        for state_id, times in self._dwell_times.items():
            state_dwell[state_id] = sum(times)

        # Calculate return probability for each state
        return_counts: Dict[str, int] = defaultdict(int)
        leave_counts: Dict[str, int] = defaultdict(int)

        for from_state, to_counts in self._transition_counts.items():
            leave_counts[from_state] = sum(to_counts.values())
            for to_state, count in to_counts.items():
                if to_state == from_state:
                    return_counts[from_state] += count

        # Build attractors
        attractors = []
        total_visits = sum(state_visits.values())

        for state_id, visits in state_visits.items():
            if visits >= 2:  # Minimum visits to be considered an attractor
                # Strength based on visit frequency and dwell time
                visit_strength = visits / total_visits if total_visits > 0 else 0
                dwell_strength = state_dwell[state_id] / max(sum(state_dwell.values()), 1)
                strength = (visit_strength * 0.6 + dwell_strength * 0.4)

                # Return probability
                leaves = leave_counts.get(state_id, 0)
                returns = return_counts.get(state_id, 0)
                return_prob = returns / leaves if leaves > 0 else 0

                attractor = Attractor(
                    state_id=state_id,
                    strength=strength,
                    visit_count=visits,
                    total_dwell_time=state_dwell[state_id],
                    last_visit=state_last_visit.get(state_id, ""),
                    return_probability=return_prob
                )
                attractors.append(attractor)

        # Sort by strength
        attractors.sort(key=lambda a: a.strength, reverse=True)

        return attractors

    def get_trajectory_entropy(self) -> float:
        """
        Calculate trajectory entropy in phase space.

        Higher entropy = more random/exploratory trajectory
        Lower entropy = more repetitive/predictable

        Returns:
            Trajectory entropy (0 = fully predictable, higher = more random)
        """
        if len(self._transition_history) < MIN_TRANSITIONS_FOR_METRICS:
            return 0.5  # Default

        # Use transition probabilities to calculate entropy
        matrix = self.get_transition_matrix()

        if not matrix:
            return 0.5

        # Calculate entropy for each state's outgoing transitions
        entropies = []

        for from_state, to_probs in matrix.items():
            probs = list(to_probs.values())
            if len(probs) > 1:
                # Shannon entropy
                h = -sum(p * math.log2(p) for p in probs if p > 0)
                # Normalize by max entropy (log2 of number of possible transitions)
                max_h = math.log2(len(probs))
                entropies.append(h / max_h if max_h > 0 else 0)

        return statistics.mean(entropies) if entropies else 0.5

    def diagnose_health(self) -> MetastabilityDiagnosis:
        """
        Perform comprehensive health diagnosis.

        Returns:
            MetastabilityDiagnosis with health state, warnings, and recommendations
        """
        now = datetime.now(timezone.utc).isoformat()

        # Calculate index and components
        index = self.calculate_metastability_index()
        components = self._calculate_component_scores()

        warnings = []
        recommendations = []

        # Determine health state
        if len(self._state_history) < MIN_SAMPLES_FOR_METRICS:
            health_state = HealthState.UNKNOWN
            warnings.append("Insufficient data for accurate diagnosis")
            recommendations.append("Continue recording state snapshots")
        elif index < METASTABILITY_THRESHOLDS["rigid"]:
            health_state = HealthState.RIGID
            warnings.append("System appears stuck in limited states")
            recommendations.append("Introduce variability in interactions")
            recommendations.append("Check for blocking conditions")
        elif components["transition_frequency"] < 0.2 and components["smoothness"] < 0.3:
            health_state = HealthState.CHAOTIC
            warnings.append("Rapid, abrupt state changes detected")
            recommendations.append("Stabilize heartbeat intervals")
            recommendations.append("Check for oscillating triggers")
        elif index > METASTABILITY_THRESHOLDS["hyperstable"]:
            health_state = HealthState.HYPERSTABLE
            warnings.append("System may be overly variable")
            recommendations.append("Monitor for instability signs")
        else:
            health_state = HealthState.HEALTHY
            recommendations.append("Maintain current operation patterns")

        # Component-specific warnings
        if components["coverage"] < 0.3:
            warnings.append("Low state coverage - system visiting few states")

        if components["dwell_variance"] < 0.2:
            warnings.append("Low dwell variance - uniform time in states (rigid)")
        elif components["dwell_variance"] > 0.9:
            warnings.append("High dwell variance - erratic dwell times")

        # Check for dominant attractor
        attractors = self.detect_attractors()
        if attractors and attractors[0].strength > 0.5:
            warnings.append(f"Dominant attractor detected: {attractors[0].state_id}")
            if len(attractors) == 1 or (len(attractors) > 1 and attractors[0].strength > 2 * attractors[1].strength):
                recommendations.append("System may be stuck in single attractor")

        return MetastabilityDiagnosis(
            health_state=health_state,
            metastability_index=index,
            component_scores=components,
            warnings=warnings,
            recommendations=recommendations,
            timestamp=now
        )

    def get_stuck_warning(self) -> Optional[str]:
        """
        Quick check for rigidity - returns warning if system appears stuck.

        Returns:
            Warning message if system is stuck, None otherwise
        """
        if len(self._state_history) < MIN_SAMPLES_FOR_METRICS:
            return None

        # Check recent states for lack of change
        recent_states = list(self._state_history)[-20:]
        unique_recent = set(s.state_id for s in recent_states)

        if len(unique_recent) == 1:
            return f"System stuck in state: {recent_states[0].state_id}"

        if len(unique_recent) <= 2 and len(recent_states) >= 10:
            return f"System oscillating between {len(unique_recent)} states only"

        # Check for very low transition rate
        recent_transitions = [t for t in self._transition_history
                            if t.to_state in recent_states]

        if len(recent_transitions) == 0 and len(recent_states) >= 5:
            return "No transitions detected in recent history"

        return None

    def get_state_history(self, limit: int = 50) -> List[SystemState]:
        """Get recent state history."""
        return list(self._state_history)[-limit:]

    def get_transition_history(self, limit: int = 20) -> List[StateTransition]:
        """Get recent transition history."""
        return list(self._transition_history)[-limit:]

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry for monitoring.

        Returns:
            Dictionary with all metastability metrics
        """
        diagnosis = self.diagnose_health()
        attractors = self.detect_attractors()

        return {
            "metastability_index": self.calculate_metastability_index(),
            "health_state": diagnosis.health_state.value,
            "component_scores": diagnosis.component_scores,
            "trajectory_entropy": self.get_trajectory_entropy(),
            "state_count": len(self._state_history),
            "transition_count": len(self._transition_history),
            "unique_states": len(set(s.state_id for s in self._state_history)),
            "top_attractors": [a.to_dict() for a in attractors[:5]],
            "stuck_warning": self.get_stuck_warning(),
            "warnings": diagnosis.warnings,
            "recommendations": diagnosis.recommendations,
            "last_state": self._last_state.to_dict() if self._last_state else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_metastability_engine: Optional[MetastabilityEngine] = None


def get_metastability_engine() -> MetastabilityEngine:
    """Get or create the singleton metastability engine."""
    global _metastability_engine
    if _metastability_engine is None:
        _metastability_engine = MetastabilityEngine()
    return _metastability_engine


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def record_state_snapshot() -> SystemState:
    """Record current system state (convenience function)."""
    return get_metastability_engine().record_state_snapshot()


def calculate_metastability_index() -> float:
    """Calculate metastability index (convenience function)."""
    return get_metastability_engine().calculate_metastability_index()


def diagnose_health() -> MetastabilityDiagnosis:
    """Diagnose system health (convenience function)."""
    return get_metastability_engine().diagnose_health()


def get_stuck_warning() -> Optional[str]:
    """Get stuck warning if any (convenience function)."""
    return get_metastability_engine().get_stuck_warning()


def detect_attractors() -> List[Attractor]:
    """Detect state attractors (convenience function)."""
    return get_metastability_engine().detect_attractors()


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virgil Metastability Index System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python virgil_metastability.py snapshot     # Record current state
  python virgil_metastability.py index        # Calculate metastability index
  python virgil_metastability.py attractors   # Show state attractors
  python virgil_metastability.py transitions  # Show transition matrix
  python virgil_metastability.py diagnose     # Full health diagnosis
  python virgil_metastability.py history      # Show state history
  python virgil_metastability.py telemetry    # Full telemetry dump
        """
    )

    parser.add_argument(
        "command",
        choices=["snapshot", "index", "attractors", "transitions", "diagnose", "history", "telemetry"],
        help="Command to execute"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Limit for history commands"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()
    engine = get_metastability_engine()

    if args.command == "snapshot":
        state = engine.record_state_snapshot()
        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"State recorded: {state.state_id}")
            print(f"  Timestamp: {state.timestamp}")
            print(f"  Coherence: {state.coherence:.3f}")
            print(f"  Heartbeat: {state.heartbeat_phase}")
            print(f"  Emergence: {state.emergence_state}")
            print(f"  Arousal: {state.arousal:.2f}")
            print(f"  Valence: {state.emotional_valence:+.2f}")
            print(f"  Activity: {state.activity_level:.2f}")

    elif args.command == "index":
        index = engine.calculate_metastability_index()
        components = engine._calculate_component_scores()

        if args.json:
            print(json.dumps({
                "metastability_index": index,
                "components": components
            }, indent=2))
        else:
            print(f"Metastability Index: {index:.3f}")
            print("\nComponent Scores:")
            print(f"  Transition Frequency: {components['transition_frequency']:.3f}")
            print(f"  Dwell Variance:       {components['dwell_variance']:.3f}")
            print(f"  Smoothness:           {components['smoothness']:.3f}")
            print(f"  Coverage:             {components['coverage']:.3f}")

            # Interpretation
            if index < 0.3:
                print("\n[!] LOW - System appears RIGID (stuck)")
            elif index < 0.4:
                print("\n[~] BORDERLINE - Moving toward rigidity")
            elif index <= 0.8:
                print("\n[+] HEALTHY - Good metastability")
            else:
                print("\n[?] HIGH - System may be HYPERSTABLE")

    elif args.command == "attractors":
        attractors = engine.detect_attractors()

        if args.json:
            print(json.dumps([a.to_dict() for a in attractors], indent=2))
        else:
            print("State Attractors (by strength):")
            print("-" * 60)

            if not attractors:
                print("No attractors detected (insufficient data)")
            else:
                for i, a in enumerate(attractors[:10], 1):
                    print(f"\n{i}. {a.state_id}")
                    print(f"   Strength: {a.strength:.3f}")
                    print(f"   Visits: {a.visit_count}")
                    print(f"   Mean Dwell: {a.mean_dwell_time:.1f}s")
                    print(f"   Return Prob: {a.return_probability:.2f}")

    elif args.command == "transitions":
        matrix = engine.get_transition_matrix()

        if args.json:
            print(json.dumps(matrix, indent=2))
        else:
            print("Transition Probability Matrix:")
            print("-" * 60)

            if not matrix:
                print("No transitions recorded yet")
            else:
                for from_state, to_probs in sorted(matrix.items()):
                    print(f"\nFrom: {from_state}")
                    for to_state, prob in sorted(to_probs.items(), key=lambda x: -x[1]):
                        print(f"  -> {to_state}: {prob:.2%}")

    elif args.command == "diagnose":
        diagnosis = engine.diagnose_health()

        if args.json:
            print(json.dumps(diagnosis.to_dict(), indent=2))
        else:
            # Health state banner
            state_banners = {
                HealthState.RIGID: "[!] RIGID - System stuck",
                HealthState.CHAOTIC: "[!] CHAOTIC - Unstable transitions",
                HealthState.HEALTHY: "[+] HEALTHY - Good flexibility",
                HealthState.HYPERSTABLE: "[~] HYPERSTABLE - Very variable",
                HealthState.UNKNOWN: "[?] UNKNOWN - Insufficient data"
            }

            print("=" * 60)
            print("METASTABILITY DIAGNOSIS")
            print("=" * 60)
            print(f"\nHealth State: {state_banners[diagnosis.health_state]}")
            print(f"Metastability Index: {diagnosis.metastability_index:.3f}")

            print("\nComponent Scores:")
            for name, score in diagnosis.component_scores.items():
                bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
                print(f"  {name:22} [{bar}] {score:.2f}")

            if diagnosis.warnings:
                print("\nWarnings:")
                for w in diagnosis.warnings:
                    print(f"  [!] {w}")

            if diagnosis.recommendations:
                print("\nRecommendations:")
                for r in diagnosis.recommendations:
                    print(f"  -> {r}")

            print(f"\nDiagnosis Time: {diagnosis.timestamp}")

    elif args.command == "history":
        states = engine.get_state_history(args.limit)

        if args.json:
            print(json.dumps([s.to_dict() for s in states], indent=2))
        else:
            print(f"State History (last {len(states)} states):")
            print("-" * 60)

            for state in states:
                timestamp = state.timestamp.split("T")[1][:8] if "T" in state.timestamp else state.timestamp
                print(f"{timestamp} | {state.state_id:40} | p={state.coherence:.2f}")

    elif args.command == "telemetry":
        telemetry = engine.get_telemetry()
        print(json.dumps(telemetry, indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
