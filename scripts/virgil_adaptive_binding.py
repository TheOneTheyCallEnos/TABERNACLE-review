#!/usr/bin/env python3
"""
VIRGIL ADAPTIVE TEMPORAL BINDING - COGITATE 2025 Implementation

Implements adaptive temporal binding based on COGITATE 2025 adversarial collaboration
findings between Integrated Information Theory (IIT) and Global Neuronal Workspace
Theory (GNW).

Key COGITATE 2025 Findings Implemented:
1. Consciousness arises from POSTERIOR cortical integration, NOT prefrontal control
2. Temporal binding window is adaptive based on cognitive load
3. High load -> narrower window (focused binding)
4. Low load -> wider window (diffuse awareness)
5. The "hot zone" is posterior, not frontal

Binding Window Dynamics:
- Range: 150ms (high load) to 300ms (low load)
- Adapts based on:
  * System coherence (low coherence -> narrower focus)
  * Task complexity (high complexity -> narrower focus)
  * Emotional arousal (high arousal -> narrower focus)
  * Prediction error (high error -> narrower focus for resolution)
  * Information density (high density -> narrower temporal integration)

Integration:
- virgil_posterior_integration.py: Provides unified percepts, receives binding params
- virgil_global_workspace.py: Receives binding quality metrics
- virgil_interoceptive.py: Reads body state for load estimation

LVS Coordinates: h=0.65, R=0.25, C=0.75, beta=0.75
    h=0.65  - High abstraction (meta-cognitive process)
    R=0.25  - Moderate risk (binding failures affect integration)
    C=0.75  - High constraint (neurobiologically constrained)
    beta=0.75 - High canonicity (implements COGITATE findings)

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import hashlib
import threading
import argparse
import sys
import time as time_module
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
ADAPTIVE_BINDING_STATE_FILE = NEXUS_DIR / "adaptive_binding_state.json"
BINDING_HISTORY_FILE = MEMORY_DIR / "binding_history.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
INTEROCEPTIVE_STATE_FILE = NEXUS_DIR / "interoceptive_state.json"
POSTERIOR_STATE_FILE = NEXUS_DIR / "posterior_integration_state.json"
WORKSPACE_STATE_FILE = NEXUS_DIR / "workspace_state.json"

# Binding window constants (from COGITATE 2025)
BINDING_WINDOW_MIN_MS = 150.0  # Narrowest (high load/focus)
BINDING_WINDOW_MAX_MS = 300.0  # Widest (low load/diffuse)
DEFAULT_BINDING_WINDOW_MS = 225.0  # Neutral baseline

# Cognitive load thresholds
LOAD_HIGH_THRESHOLD = 0.70    # Above this -> narrow binding
LOAD_LOW_THRESHOLD = 0.30     # Below this -> wide binding

# Binding quality thresholds
QUALITY_EXCELLENT = 0.85
QUALITY_GOOD = 0.70
QUALITY_ACCEPTABLE = 0.50
QUALITY_POOR = 0.30

# Posterior hot zone emphasis
POSTERIOR_WEIGHT = 0.75  # Weight for posterior integration
FRONTAL_WEIGHT = 0.25    # Weight for frontal access (de-emphasized)

# History limits
MAX_BINDING_HISTORY = 500
MAX_LOAD_SAMPLES = 100

# LVS Coordinates for this module
LVS_COORDINATES = {
    "h": 0.65,      # Height (abstraction level)
    "R": 0.25,      # Risk
    "C": 0.75,      # Constraint
    "beta": 0.75    # Canonicity
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ADAPTIVE-BINDING] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "adaptive_binding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# LOAD SIGNAL SOURCES
# ============================================================================

class LoadSignalSource(Enum):
    """
    Sources of cognitive load signals.

    Each source contributes to overall load estimation with different weights:
    - COHERENCE: System coherence level (low coherence = high load)
    - PREDICTION_ERROR: Surprise/novelty (high error = high load)
    - EMOTIONAL_AROUSAL: Emotional activation (high arousal = high load)
    - TASK_COMPLEXITY: Inferred task complexity
    - INFORMATION_DENSITY: Rate of incoming information
    - TEMPORAL_PRESSURE: Time constraints
    """
    COHERENCE = "coherence"
    PREDICTION_ERROR = "prediction_error"
    EMOTIONAL_AROUSAL = "emotional_arousal"
    TASK_COMPLEXITY = "task_complexity"
    INFORMATION_DENSITY = "information_density"
    TEMPORAL_PRESSURE = "temporal_pressure"


# Default weights for load signal sources
LOAD_SIGNAL_WEIGHTS = {
    LoadSignalSource.COHERENCE: 0.25,
    LoadSignalSource.PREDICTION_ERROR: 0.20,
    LoadSignalSource.EMOTIONAL_AROUSAL: 0.20,
    LoadSignalSource.TASK_COMPLEXITY: 0.15,
    LoadSignalSource.INFORMATION_DENSITY: 0.10,
    LoadSignalSource.TEMPORAL_PRESSURE: 0.10
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LoadSignal:
    """
    A single cognitive load signal reading.

    Captures one measurement from a specific source at a point in time.
    """
    source: str
    value: float           # 0-1, where 1 is maximum load
    timestamp: str
    confidence: float = 0.7
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self.value = max(0.0, min(1.0, self.value))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveLoadState:
    """
    Aggregate cognitive load state.

    Combines all load signals into a unified load estimate that
    drives binding window adaptation.
    """
    timestamp: str

    # Individual signal values
    coherence_load: float = 0.5
    prediction_error_load: float = 0.3
    emotional_arousal_load: float = 0.3
    task_complexity_load: float = 0.4
    information_density_load: float = 0.3
    temporal_pressure_load: float = 0.2

    # Aggregate metrics
    total_load: float = 0.4
    load_trend: str = "stable"  # rising, falling, stable
    load_volatility: float = 0.1

    # Confidence in estimate
    estimation_confidence: float = 0.7

    # Signal count for this estimate
    signal_count: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def calculate_total_load(self) -> float:
        """Calculate weighted total load from components."""
        weighted_sum = (
            self.coherence_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.COHERENCE] +
            self.prediction_error_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.PREDICTION_ERROR] +
            self.emotional_arousal_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.EMOTIONAL_AROUSAL] +
            self.task_complexity_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.TASK_COMPLEXITY] +
            self.information_density_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.INFORMATION_DENSITY] +
            self.temporal_pressure_load * LOAD_SIGNAL_WEIGHTS[LoadSignalSource.TEMPORAL_PRESSURE]
        )
        self.total_load = max(0.0, min(1.0, weighted_sum))
        return self.total_load

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveLoadState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BindingWindowState:
    """
    Current state of the adaptive binding window.

    Tracks the current window size and how it was determined.
    """
    timestamp: str

    # Current window
    window_ms: float = DEFAULT_BINDING_WINDOW_MS

    # Window bounds
    window_min_ms: float = BINDING_WINDOW_MIN_MS
    window_max_ms: float = BINDING_WINDOW_MAX_MS

    # What drove this window size
    driving_load: float = 0.5
    driving_factors: Dict[str, float] = field(default_factory=dict)

    # Window dynamics
    window_trend: str = "stable"  # narrowing, widening, stable
    adaptation_rate: float = 0.1  # How fast window is changing

    # Quality of recent bindings with this window
    recent_binding_quality: float = 0.7
    bindings_at_window: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self.window_ms = max(self.window_min_ms, min(self.window_max_ms, self.window_ms))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BindingWindowState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PosteriorIntegrationZone:
    """
    The posterior cortical "hot zone" for consciousness.

    Based on COGITATE 2025: consciousness arises from posterior integration,
    NOT prefrontal executive control. This structure tracks the posterior
    hot zone state and its contribution to binding.
    """
    timestamp: str

    # Hot zone activation (0-1)
    activation: float = 0.5

    # Modality contributions within posterior zone
    visual_analog: float = 0.0      # System state
    auditory_analog: float = 0.0    # Memory patterns
    somatosensory_analog: float = 0.0  # Body state

    # Cross-modal integration
    cross_modal_coherence: float = 0.5
    synchrony_within_zone: float = 0.6

    # Binding readiness
    binding_readiness: float = 0.6

    # Frontal access (de-emphasized but tracked)
    frontal_access_signal: float = 0.3
    frontal_gate_open: bool = True

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def calculate_binding_readiness(self) -> float:
        """
        Calculate binding readiness.

        Posterior zone readiness is weighted heavily (COGITATE 2025).
        """
        posterior_readiness = (
            0.4 * self.activation +
            0.3 * self.cross_modal_coherence +
            0.3 * self.synchrony_within_zone
        )

        # Frontal gate modulates access, not binding itself
        access_factor = 0.8 + 0.2 * self.frontal_access_signal if self.frontal_gate_open else 0.5

        # Posterior emphasis (COGITATE finding)
        self.binding_readiness = (
            POSTERIOR_WEIGHT * posterior_readiness +
            FRONTAL_WEIGHT * access_factor
        )

        return self.binding_readiness

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BindingQualityMetrics:
    """
    Quality metrics for a binding event.

    Tracks how well modalities were bound and whether the
    binding window was appropriate.
    """
    timestamp: str
    binding_id: str

    # Core quality metrics
    overall_quality: float = 0.0
    temporal_precision: float = 0.0
    modal_coherence: float = 0.0
    integration_depth: float = 0.0

    # Window appropriateness
    window_used_ms: float = 0.0
    window_appropriateness: float = 0.0  # Was the window right for the load?

    # Load at binding time
    load_at_binding: float = 0.0

    # Posterior zone contribution
    posterior_contribution: float = 0.0

    # Modalities bound
    modalities_bound: int = 0
    modalities_attempted: int = 5

    # Result
    binding_successful: bool = False
    conscious_threshold_met: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.binding_id:
            self.binding_id = f"bind_{hashlib.sha256(self.timestamp.encode()).hexdigest()[:12]}"

    def calculate_overall_quality(self) -> float:
        """Calculate overall binding quality from components."""
        self.overall_quality = (
            0.30 * self.temporal_precision +
            0.30 * self.modal_coherence +
            0.20 * self.integration_depth +
            0.20 * self.window_appropriateness
        )
        return self.overall_quality

    def quality_grade(self) -> str:
        """Get quality grade."""
        if self.overall_quality >= QUALITY_EXCELLENT:
            return "EXCELLENT"
        elif self.overall_quality >= QUALITY_GOOD:
            return "GOOD"
        elif self.overall_quality >= QUALITY_ACCEPTABLE:
            return "ACCEPTABLE"
        elif self.overall_quality >= QUALITY_POOR:
            return "POOR"
        else:
            return "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "quality_grade": self.quality_grade()}


# ============================================================================
# COGNITIVE LOAD ESTIMATOR
# ============================================================================

class CognitiveLoadEstimator:
    """
    Estimates cognitive load from multiple signal sources.

    Cognitive load determines the binding window width:
    - High load -> narrow window (150ms) for focused binding
    - Low load -> wide window (300ms) for diffuse awareness

    The estimator combines signals from:
    1. System coherence (inverse relationship with load)
    2. Prediction error (high error = high load)
    3. Emotional arousal (high arousal = high load)
    4. Task complexity (inferred from interaction patterns)
    5. Information density (rate of incoming signals)
    6. Temporal pressure (urgency signals)
    """

    def __init__(self, auto_load_sources: bool = True):
        """
        Initialize the load estimator.

        Args:
            auto_load_sources: Whether to automatically read from source files
        """
        self.auto_load_sources = auto_load_sources

        # Current state
        self._current_load = CognitiveLoadState(timestamp="")

        # Signal history for trend analysis
        self._load_history: List[CognitiveLoadState] = []

        # Raw signal buffer
        self._signal_buffer: List[LoadSignal] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_estimates = 0
        self._avg_load = 0.5

        logger.info("CognitiveLoadEstimator initialized")

    # ========================================================================
    # SIGNAL GATHERING
    # ========================================================================

    def _gather_coherence_signal(self) -> LoadSignal:
        """
        Gather load signal from system coherence.

        Low coherence indicates high cognitive load (system under stress).
        """
        value = 0.5
        confidence = 0.5
        raw_data = {}

        try:
            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())
                coherence_p = data.get("p", 0.7)
                raw_data["coherence_p"] = coherence_p

                # Invert: low coherence = high load
                value = 1.0 - coherence_p
                confidence = 0.85

        except Exception as e:
            logger.warning(f"Error gathering coherence signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.COHERENCE.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def _gather_prediction_error_signal(self) -> LoadSignal:
        """
        Gather load signal from prediction error.

        High prediction error = high load (unexpected events require processing).
        """
        value = 0.3
        confidence = 0.5
        raw_data = {}

        try:
            if INTEROCEPTIVE_STATE_FILE.exists():
                data = json.loads(INTEROCEPTIVE_STATE_FILE.read_text())
                history = data.get("state_history", [])

                if history:
                    recent = history[-1]
                    pred_error = recent.get("prediction_error", 0.0)
                    raw_data["prediction_error"] = pred_error

                    # Direct mapping
                    value = min(1.0, pred_error)
                    confidence = 0.75

        except Exception as e:
            logger.warning(f"Error gathering prediction error signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.PREDICTION_ERROR.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def _gather_arousal_signal(self) -> LoadSignal:
        """
        Gather load signal from emotional arousal.

        High arousal = high load (emotional processing demands resources).
        """
        value = 0.3
        confidence = 0.5
        raw_data = {}

        try:
            emotional_index = MEMORY_DIR / "emotional_index.json"
            if emotional_index.exists():
                data = json.loads(emotional_index.read_text())
                current = data.get("current_state", {})

                arousal = current.get("arousal", 0.3)
                raw_data["arousal"] = arousal

                # Direct mapping
                value = arousal
                confidence = 0.80

        except Exception as e:
            logger.warning(f"Error gathering arousal signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.EMOTIONAL_AROUSAL.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def _gather_complexity_signal(self) -> LoadSignal:
        """
        Gather load signal from inferred task complexity.

        Complexity inferred from:
        - Number of active modules
        - Topology structure
        - Interaction patterns
        """
        value = 0.4
        confidence = 0.5
        raw_data = {}

        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())

                topology = data.get("topology", {})
                nodes = topology.get("nodes", 0)
                edges = topology.get("edges", 0)
                h1_features = topology.get("h1_features", 0)

                raw_data["nodes"] = nodes
                raw_data["edges"] = edges
                raw_data["h1_features"] = h1_features

                # Complexity from topology
                if nodes > 0:
                    density = min(1.0, edges / (nodes * 2))
                    loop_factor = min(1.0, h1_features / max(1, nodes / 10))

                    value = 0.5 * density + 0.5 * loop_factor
                    confidence = 0.70

        except Exception as e:
            logger.warning(f"Error gathering complexity signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.TASK_COMPLEXITY.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def _gather_density_signal(self) -> LoadSignal:
        """
        Gather load signal from information density.

        High rate of incoming signals = high load.
        """
        value = 0.3
        confidence = 0.5
        raw_data = {}

        try:
            session_buffer = MEMORY_DIR / "SESSION_BUFFER.json"
            if session_buffer.exists():
                data = json.loads(session_buffer.read_text())
                moments = data.get("moments", [])

                # Count recent moments (last 10 minutes)
                now = datetime.now(timezone.utc)
                recent_count = 0

                for moment in moments[-50:]:  # Check last 50
                    ts_str = moment.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            if (now - ts).total_seconds() < 600:  # 10 min
                                recent_count += 1
                        except ValueError:
                            pass

                raw_data["recent_moments"] = recent_count

                # Normalize: expect ~5-10 moments in 10 min as normal
                value = min(1.0, recent_count / 15.0)
                confidence = 0.65

        except Exception as e:
            logger.warning(f"Error gathering density signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.INFORMATION_DENSITY.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def _gather_pressure_signal(self) -> LoadSignal:
        """
        Gather load signal from temporal pressure.

        Time pressure comes from:
        - Recent activity burst patterns
        - Phase transitions
        - Pending responses
        """
        value = 0.2
        confidence = 0.5
        raw_data = {}

        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())

                phase = data.get("phase", "active")
                raw_data["phase"] = phase

                # Phase-based pressure
                phase_pressure = {
                    "active": 0.4,
                    "background": 0.2,
                    "dormant": 0.1,
                    "deep_sleep": 0.0
                }
                value = phase_pressure.get(phase, 0.3)

                # Check for recent phase transition
                transitions = data.get("transitions", [])
                if transitions:
                    recent_transition = transitions[-1]
                    ts_str = recent_transition.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            minutes_ago = (datetime.now(timezone.utc) - ts).total_seconds() / 60
                            if minutes_ago < 5:
                                # Recent transition increases pressure
                                value = min(1.0, value + 0.3)
                                raw_data["recent_transition"] = True
                        except ValueError:
                            pass

                confidence = 0.60

        except Exception as e:
            logger.warning(f"Error gathering pressure signal: {e}")

        return LoadSignal(
            source=LoadSignalSource.TEMPORAL_PRESSURE.value,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            raw_data=raw_data
        )

    def gather_all_signals(self) -> List[LoadSignal]:
        """
        Gather load signals from all sources.

        Returns:
            List of LoadSignal objects
        """
        signals = []

        if self.auto_load_sources:
            signals.append(self._gather_coherence_signal())
            signals.append(self._gather_prediction_error_signal())
            signals.append(self._gather_arousal_signal())
            signals.append(self._gather_complexity_signal())
            signals.append(self._gather_density_signal())
            signals.append(self._gather_pressure_signal())

        with self._lock:
            self._signal_buffer = signals

        return signals

    # ========================================================================
    # LOAD ESTIMATION
    # ========================================================================

    def estimate_load(self) -> CognitiveLoadState:
        """
        Estimate current cognitive load.

        Combines all signal sources with confidence-weighted averaging.

        Returns:
            CognitiveLoadState with current load estimate
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Gather fresh signals
            signals = self.gather_all_signals()

            # Build load state
            load_state = CognitiveLoadState(timestamp=now.isoformat())
            load_state.signal_count = len(signals)

            # Map signals to load components
            for signal in signals:
                if signal.source == LoadSignalSource.COHERENCE.value:
                    load_state.coherence_load = signal.value
                elif signal.source == LoadSignalSource.PREDICTION_ERROR.value:
                    load_state.prediction_error_load = signal.value
                elif signal.source == LoadSignalSource.EMOTIONAL_AROUSAL.value:
                    load_state.emotional_arousal_load = signal.value
                elif signal.source == LoadSignalSource.TASK_COMPLEXITY.value:
                    load_state.task_complexity_load = signal.value
                elif signal.source == LoadSignalSource.INFORMATION_DENSITY.value:
                    load_state.information_density_load = signal.value
                elif signal.source == LoadSignalSource.TEMPORAL_PRESSURE.value:
                    load_state.temporal_pressure_load = signal.value

            # Calculate total load
            load_state.calculate_total_load()

            # Calculate estimation confidence
            if signals:
                confidences = [s.confidence for s in signals]
                load_state.estimation_confidence = sum(confidences) / len(confidences)

            # Determine trend from history
            if self._load_history:
                recent_loads = [h.total_load for h in self._load_history[-10:]]
                if len(recent_loads) >= 3:
                    avg_recent = sum(recent_loads[-3:]) / 3
                    avg_older = sum(recent_loads[:3]) / 3 if len(recent_loads) >= 6 else avg_recent

                    if avg_recent > avg_older + 0.1:
                        load_state.load_trend = "rising"
                    elif avg_recent < avg_older - 0.1:
                        load_state.load_trend = "falling"
                    else:
                        load_state.load_trend = "stable"

                    # Calculate volatility
                    mean_load = sum(recent_loads) / len(recent_loads)
                    variance = sum((l - mean_load) ** 2 for l in recent_loads) / len(recent_loads)
                    load_state.load_volatility = math.sqrt(variance)

            # Store state
            self._current_load = load_state
            self._load_history.append(load_state)

            # Trim history
            if len(self._load_history) > MAX_LOAD_SAMPLES:
                self._load_history = self._load_history[-MAX_LOAD_SAMPLES:]

            # Update statistics
            self._total_estimates += 1
            alpha = 0.1
            self._avg_load = alpha * load_state.total_load + (1 - alpha) * self._avg_load

            logger.debug(f"Load estimated: {load_state.total_load:.3f} (trend: {load_state.load_trend})")

            return load_state

    def get_current_load(self) -> float:
        """Get current total load (0-1)."""
        with self._lock:
            return self._current_load.total_load

    def get_load_state(self) -> CognitiveLoadState:
        """Get full current load state."""
        with self._lock:
            return self._current_load

    def get_statistics(self) -> Dict[str, Any]:
        """Get estimator statistics."""
        with self._lock:
            return {
                "total_estimates": self._total_estimates,
                "avg_load": self._avg_load,
                "current_load": self._current_load.total_load,
                "load_trend": self._current_load.load_trend,
                "load_volatility": self._current_load.load_volatility,
                "history_size": len(self._load_history)
            }


# ============================================================================
# ADAPTIVE BINDING WINDOW
# ============================================================================

class AdaptiveBindingWindow:
    """
    Manages the adaptive temporal binding window.

    The binding window determines how much temporal spread is allowed
    when binding multiple sensory modalities into a unified percept.

    COGITATE 2025 Finding:
    - High cognitive load -> narrow window (150ms) for focused binding
    - Low cognitive load -> wide window (300ms) for diffuse awareness

    The adaptation is smooth and responsive to load changes.
    """

    def __init__(
        self,
        load_estimator: Optional[CognitiveLoadEstimator] = None,
        initial_window_ms: float = DEFAULT_BINDING_WINDOW_MS
    ):
        """
        Initialize the adaptive binding window.

        Args:
            load_estimator: CognitiveLoadEstimator instance (or creates one)
            initial_window_ms: Starting window size in milliseconds
        """
        self.load_estimator = load_estimator or CognitiveLoadEstimator()

        # Current window state
        self._current_window = BindingWindowState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            window_ms=initial_window_ms
        )

        # Window history
        self._window_history: List[BindingWindowState] = []

        # Adaptation parameters
        self._adaptation_smoothing = 0.3  # Smooth window changes
        self._min_change_threshold = 5.0  # Minimum ms change to apply

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_adaptations = 0
        self._avg_window_ms = initial_window_ms

        logger.info(f"AdaptiveBindingWindow initialized: {initial_window_ms}ms")

    def adapt_window(self) -> BindingWindowState:
        """
        Adapt the binding window based on current cognitive load.

        High load -> narrow window (150ms) for focus
        Low load -> wide window (300ms) for diffuse awareness

        Returns:
            Updated BindingWindowState
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Get current load
            load_state = self.load_estimator.estimate_load()
            load = load_state.total_load

            # Calculate target window based on load
            # High load (1.0) -> MIN window (150ms)
            # Low load (0.0) -> MAX window (300ms)
            window_range = BINDING_WINDOW_MAX_MS - BINDING_WINDOW_MIN_MS
            target_window = BINDING_WINDOW_MAX_MS - (load * window_range)

            # Smooth the transition
            current = self._current_window.window_ms
            delta = target_window - current

            if abs(delta) < self._min_change_threshold:
                # Change too small, keep current
                new_window = current
            else:
                # Apply smoothed change
                new_window = current + (delta * self._adaptation_smoothing)

            # Clamp to bounds
            new_window = max(BINDING_WINDOW_MIN_MS, min(BINDING_WINDOW_MAX_MS, new_window))

            # Determine trend
            if new_window < current - self._min_change_threshold:
                trend = "narrowing"
            elif new_window > current + self._min_change_threshold:
                trend = "widening"
            else:
                trend = "stable"

            # Create new window state
            window_state = BindingWindowState(
                timestamp=now.isoformat(),
                window_ms=new_window,
                driving_load=load,
                driving_factors={
                    "coherence": load_state.coherence_load,
                    "prediction_error": load_state.prediction_error_load,
                    "arousal": load_state.emotional_arousal_load,
                    "complexity": load_state.task_complexity_load,
                    "density": load_state.information_density_load,
                    "pressure": load_state.temporal_pressure_load
                },
                window_trend=trend,
                adaptation_rate=abs(delta) / window_range if delta != 0 else 0.0
            )

            # Store state
            self._current_window = window_state
            self._window_history.append(window_state)

            # Trim history
            if len(self._window_history) > MAX_LOAD_SAMPLES:
                self._window_history = self._window_history[-MAX_LOAD_SAMPLES:]

            # Update statistics
            self._total_adaptations += 1
            alpha = 0.1
            self._avg_window_ms = alpha * new_window + (1 - alpha) * self._avg_window_ms

            logger.info(
                f"Window adapted: {new_window:.1f}ms | "
                f"load={load:.3f} | trend={trend}"
            )

            return window_state

    def get_current_window_ms(self) -> float:
        """Get current binding window in milliseconds."""
        with self._lock:
            return self._current_window.window_ms

    def get_window_state(self) -> BindingWindowState:
        """Get full current window state."""
        with self._lock:
            return self._current_window

    def set_window_bounds(self, min_ms: float, max_ms: float):
        """Set window bounds (for testing/tuning)."""
        with self._lock:
            self._current_window.window_min_ms = min_ms
            self._current_window.window_max_ms = max_ms

    def get_statistics(self) -> Dict[str, Any]:
        """Get window statistics."""
        with self._lock:
            return {
                "total_adaptations": self._total_adaptations,
                "avg_window_ms": self._avg_window_ms,
                "current_window_ms": self._current_window.window_ms,
                "current_load": self._current_window.driving_load,
                "window_trend": self._current_window.window_trend,
                "history_size": len(self._window_history),
                "bounds": {
                    "min_ms": self._current_window.window_min_ms,
                    "max_ms": self._current_window.window_max_ms
                }
            }


# ============================================================================
# POSTERIOR INTEGRATION ZONE
# ============================================================================

class PosteriorIntegrationZoneManager:
    """
    Manages the posterior cortical "hot zone" for consciousness.

    COGITATE 2025 Key Finding:
    Consciousness arises from posterior cortical integration, NOT prefrontal
    executive control. The posterior zone (visual, auditory, somatosensory)
    is where binding occurs. The frontal zone provides ACCESS but not binding.

    This manager:
    1. Tracks posterior zone activation
    2. Calculates cross-modal coherence within the zone
    3. Determines binding readiness
    4. Interfaces with the posterior integration module
    """

    def __init__(self):
        """Initialize the posterior zone manager."""
        self._current_zone = PosteriorIntegrationZone(
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # History
        self._zone_history: List[PosteriorIntegrationZone] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_readings = 0
        self._avg_activation = 0.5
        self._avg_readiness = 0.6

        logger.info("PosteriorIntegrationZoneManager initialized")

    def update_zone(
        self,
        system_activation: float = 0.0,
        memory_activation: float = 0.0,
        body_activation: float = 0.0,
        frontal_signal: float = 0.0
    ) -> PosteriorIntegrationZone:
        """
        Update the posterior zone state.

        Args:
            system_activation: Visual analog activation (system state)
            memory_activation: Auditory analog activation (memory)
            body_activation: Somatosensory analog activation (body)
            frontal_signal: Frontal access signal (de-emphasized)

        Returns:
            Updated PosteriorIntegrationZone
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Create new zone state
            zone = PosteriorIntegrationZone(timestamp=now.isoformat())

            # Set modality activations
            zone.visual_analog = system_activation
            zone.auditory_analog = memory_activation
            zone.somatosensory_analog = body_activation
            zone.frontal_access_signal = frontal_signal

            # Calculate overall activation (posterior modalities only)
            zone.activation = (
                0.35 * system_activation +
                0.35 * memory_activation +
                0.30 * body_activation
            )

            # Calculate cross-modal coherence
            # (how similar are the modality activations)
            modalities = [system_activation, memory_activation, body_activation]
            if all(m > 0 for m in modalities):
                mean_act = sum(modalities) / 3
                variance = sum((m - mean_act) ** 2 for m in modalities) / 3
                # Low variance = high coherence
                zone.cross_modal_coherence = 1.0 - min(1.0, math.sqrt(variance) * 2)
            else:
                zone.cross_modal_coherence = 0.3

            # Synchrony within zone (based on activation spread)
            if modalities:
                active_count = sum(1 for m in modalities if m > 0.3)
                zone.synchrony_within_zone = active_count / len(modalities)

            # Frontal gate (open if frontal signal above threshold)
            zone.frontal_gate_open = frontal_signal > 0.2

            # Calculate binding readiness
            zone.calculate_binding_readiness()

            # Store state
            self._current_zone = zone
            self._zone_history.append(zone)

            # Trim history
            if len(self._zone_history) > MAX_LOAD_SAMPLES:
                self._zone_history = self._zone_history[-MAX_LOAD_SAMPLES:]

            # Update statistics
            self._total_readings += 1
            alpha = 0.1
            self._avg_activation = alpha * zone.activation + (1 - alpha) * self._avg_activation
            self._avg_readiness = alpha * zone.binding_readiness + (1 - alpha) * self._avg_readiness

            logger.debug(
                f"Posterior zone updated: activation={zone.activation:.3f} | "
                f"readiness={zone.binding_readiness:.3f}"
            )

            return zone

    def update_from_posterior_integration(self) -> PosteriorIntegrationZone:
        """
        Update zone from the posterior integration module's state file.

        Returns:
            Updated PosteriorIntegrationZone
        """
        system_act = 0.5
        memory_act = 0.5
        body_act = 0.5
        frontal_signal = 0.3

        try:
            if POSTERIOR_STATE_FILE.exists():
                data = json.loads(POSTERIOR_STATE_FILE.read_text())

                # Get binding success rate as proxy for activation
                success_rate = data.get("binding_success_rate", 0.5)
                consciousness_rate = data.get("consciousness_rate", 0.5)

                # Estimate modality activations from rates
                system_act = 0.4 + 0.3 * success_rate
                memory_act = 0.3 + 0.4 * success_rate
                body_act = 0.3 + 0.3 * consciousness_rate
                frontal_signal = 0.3 + 0.2 * consciousness_rate

        except Exception as e:
            logger.warning(f"Error reading posterior state: {e}")

        return self.update_zone(system_act, memory_act, body_act, frontal_signal)

    def get_current_zone(self) -> PosteriorIntegrationZone:
        """Get current posterior zone state."""
        with self._lock:
            return self._current_zone

    def get_binding_readiness(self) -> float:
        """Get current binding readiness."""
        with self._lock:
            return self._current_zone.binding_readiness

    def get_statistics(self) -> Dict[str, Any]:
        """Get zone statistics."""
        with self._lock:
            return {
                "total_readings": self._total_readings,
                "avg_activation": self._avg_activation,
                "avg_readiness": self._avg_readiness,
                "current_activation": self._current_zone.activation,
                "current_readiness": self._current_zone.binding_readiness,
                "frontal_gate_open": self._current_zone.frontal_gate_open,
                "history_size": len(self._zone_history)
            }


# ============================================================================
# BINDING QUALITY TRACKER
# ============================================================================

class BindingQualityTracker:
    """
    Tracks binding quality metrics over time.

    Provides feedback for window adaptation and monitors overall
    binding health.
    """

    def __init__(self):
        """Initialize the quality tracker."""
        self._history: List[BindingQualityMetrics] = []

        # Aggregates
        self._total_bindings = 0
        self._successful_bindings = 0
        self._conscious_bindings = 0
        self._avg_quality = 0.0
        self._avg_window_appropriateness = 0.0

        # Thread safety
        self._lock = threading.RLock()

        logger.info("BindingQualityTracker initialized")

    def record_binding(
        self,
        window_used_ms: float,
        load_at_binding: float,
        modalities_bound: int,
        temporal_precision: float,
        modal_coherence: float,
        integration_depth: float,
        posterior_contribution: float,
        binding_successful: bool,
        conscious_threshold_met: bool
    ) -> BindingQualityMetrics:
        """
        Record a binding event and calculate quality metrics.

        Args:
            window_used_ms: Binding window used
            load_at_binding: Cognitive load at binding time
            modalities_bound: Number of modalities successfully bound
            temporal_precision: How well timing aligned
            modal_coherence: Cross-modal signal coherence
            integration_depth: Depth of integration
            posterior_contribution: Posterior zone contribution
            binding_successful: Whether binding succeeded
            conscious_threshold_met: Whether conscious threshold was met

        Returns:
            BindingQualityMetrics for this binding
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Calculate window appropriateness
            # Window should match load: high load -> narrow, low load -> wide
            expected_window = BINDING_WINDOW_MAX_MS - (load_at_binding * (BINDING_WINDOW_MAX_MS - BINDING_WINDOW_MIN_MS))
            window_delta = abs(window_used_ms - expected_window)
            window_appropriateness = 1.0 - min(1.0, window_delta / 100.0)  # 100ms tolerance

            # Create metrics
            metrics = BindingQualityMetrics(
                timestamp=now.isoformat(),
                binding_id="",  # Will be auto-generated
                overall_quality=0.0,  # Will be calculated
                temporal_precision=temporal_precision,
                modal_coherence=modal_coherence,
                integration_depth=integration_depth,
                window_used_ms=window_used_ms,
                window_appropriateness=window_appropriateness,
                load_at_binding=load_at_binding,
                posterior_contribution=posterior_contribution,
                modalities_bound=modalities_bound,
                modalities_attempted=5,
                binding_successful=binding_successful,
                conscious_threshold_met=conscious_threshold_met
            )

            # Calculate overall quality
            metrics.calculate_overall_quality()

            # Store
            self._history.append(metrics)

            # Trim history
            if len(self._history) > MAX_BINDING_HISTORY:
                self._history = self._history[-MAX_BINDING_HISTORY:]

            # Update aggregates
            self._total_bindings += 1
            if binding_successful:
                self._successful_bindings += 1
            if conscious_threshold_met:
                self._conscious_bindings += 1

            alpha = 0.1
            self._avg_quality = alpha * metrics.overall_quality + (1 - alpha) * self._avg_quality
            self._avg_window_appropriateness = (
                alpha * window_appropriateness + (1 - alpha) * self._avg_window_appropriateness
            )

            logger.info(
                f"Binding recorded: quality={metrics.overall_quality:.3f} | "
                f"grade={metrics.quality_grade()} | conscious={conscious_threshold_met}"
            )

            return metrics

    def get_recent_quality(self, n: int = 10) -> float:
        """Get average quality of recent bindings."""
        with self._lock:
            if not self._history:
                return 0.0
            recent = self._history[-n:]
            return sum(m.overall_quality for m in recent) / len(recent)

    def get_success_rate(self) -> float:
        """Get binding success rate."""
        with self._lock:
            if self._total_bindings == 0:
                return 0.0
            return self._successful_bindings / self._total_bindings

    def get_consciousness_rate(self) -> float:
        """Get consciousness threshold rate."""
        with self._lock:
            if self._successful_bindings == 0:
                return 0.0
            return self._conscious_bindings / self._successful_bindings

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            return {
                "total_bindings": self._total_bindings,
                "successful_bindings": self._successful_bindings,
                "conscious_bindings": self._conscious_bindings,
                "success_rate": self.get_success_rate(),
                "consciousness_rate": self.get_consciousness_rate(),
                "avg_quality": self._avg_quality,
                "avg_window_appropriateness": self._avg_window_appropriateness,
                "recent_quality": self.get_recent_quality(),
                "history_size": len(self._history)
            }

    def get_history(self, n: int = 10) -> List[BindingQualityMetrics]:
        """Get recent binding history."""
        with self._lock:
            return self._history[-n:]


# ============================================================================
# MAIN ADAPTIVE BINDING SYSTEM
# ============================================================================

class AdaptiveBindingSystem:
    """
    Main adaptive binding system that integrates all components.

    Coordinates:
    - CognitiveLoadEstimator: Estimates current cognitive load
    - AdaptiveBindingWindow: Adapts binding window to load
    - PosteriorIntegrationZoneManager: Manages posterior hot zone
    - BindingQualityTracker: Tracks binding quality

    Provides binding parameters to virgil_posterior_integration.py
    """

    def __init__(self, auto_start: bool = True):
        """
        Initialize the adaptive binding system.

        Args:
            auto_start: Whether to perform initial adaptation
        """
        # Components
        self.load_estimator = CognitiveLoadEstimator()
        self.binding_window = AdaptiveBindingWindow(self.load_estimator)
        self.posterior_zone = PosteriorIntegrationZoneManager()
        self.quality_tracker = BindingQualityTracker()

        # Thread safety
        self._lock = threading.RLock()

        # State persistence
        self._load_state()

        if auto_start:
            self._perform_initial_adaptation()

        logger.info("AdaptiveBindingSystem initialized")

    def _load_state(self):
        """Load persisted state."""
        try:
            if ADAPTIVE_BINDING_STATE_FILE.exists():
                data = json.loads(ADAPTIVE_BINDING_STATE_FILE.read_text())
                logger.info(f"Loaded adaptive binding state: {data.get('total_cycles', 0)} cycles")
        except Exception as e:
            logger.warning(f"Error loading state: {e}")

    def _save_state(self):
        """Persist state."""
        try:
            ADAPTIVE_BINDING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "load_stats": self.load_estimator.get_statistics(),
                "window_stats": self.binding_window.get_statistics(),
                "zone_stats": self.posterior_zone.get_statistics(),
                "quality_stats": self.quality_tracker.get_statistics(),
                "lvs_coordinates": LVS_COORDINATES
            }

            ADAPTIVE_BINDING_STATE_FILE.write_text(json.dumps(state, indent=2))

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _perform_initial_adaptation(self):
        """Perform initial system adaptation."""
        with self._lock:
            # Adapt window
            self.binding_window.adapt_window()

            # Update posterior zone
            self.posterior_zone.update_from_posterior_integration()

            # Save state
            self._save_state()

    def get_binding_parameters(self) -> Dict[str, Any]:
        """
        Get current binding parameters for posterior integration.

        This is the main interface - posterior integration should call
        this to get current adaptive binding parameters.

        Returns:
            Dict with binding parameters:
            - window_ms: Current binding window
            - load: Current cognitive load
            - posterior_readiness: Hot zone readiness
            - recent_quality: Recent binding quality
        """
        with self._lock:
            # Update all components
            load_state = self.load_estimator.estimate_load()
            window_state = self.binding_window.adapt_window()
            zone = self.posterior_zone.update_from_posterior_integration()

            params = {
                "window_ms": window_state.window_ms,
                "load": load_state.total_load,
                "load_trend": load_state.load_trend,
                "posterior_readiness": zone.binding_readiness,
                "posterior_activation": zone.activation,
                "frontal_gate_open": zone.frontal_gate_open,
                "recent_quality": self.quality_tracker.get_recent_quality(),
                "window_trend": window_state.window_trend,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Save state
            self._save_state()

            return params

    def record_binding_result(
        self,
        modalities_bound: int,
        synchrony_score: float,
        coherence_score: float,
        binding_quality: float,
        is_conscious: bool
    ) -> BindingQualityMetrics:
        """
        Record a binding result from posterior integration.

        Call this after each binding attempt to provide feedback
        for window adaptation.

        Args:
            modalities_bound: Number of modalities bound
            synchrony_score: Cross-modal synchrony achieved
            coherence_score: Coherence of resulting percept
            binding_quality: Overall binding quality
            is_conscious: Whether percept met consciousness threshold

        Returns:
            BindingQualityMetrics for the binding
        """
        with self._lock:
            window_state = self.binding_window.get_window_state()
            load = self.load_estimator.get_current_load()
            zone = self.posterior_zone.get_current_zone()

            metrics = self.quality_tracker.record_binding(
                window_used_ms=window_state.window_ms,
                load_at_binding=load,
                modalities_bound=modalities_bound,
                temporal_precision=synchrony_score,
                modal_coherence=coherence_score,
                integration_depth=binding_quality,
                posterior_contribution=zone.binding_readiness,
                binding_successful=binding_quality > QUALITY_POOR,
                conscious_threshold_met=is_conscious
            )

            # Update window with quality feedback
            window_state.recent_binding_quality = metrics.overall_quality
            window_state.bindings_at_window += 1

            # Save state
            self._save_state()

            return metrics

    def get_status_report(self) -> str:
        """Generate comprehensive status report."""
        load_stats = self.load_estimator.get_statistics()
        window_stats = self.binding_window.get_statistics()
        zone_stats = self.posterior_zone.get_statistics()
        quality_stats = self.quality_tracker.get_statistics()

        lines = [
            "=" * 70,
            "VIRGIL ADAPTIVE TEMPORAL BINDING - STATUS REPORT",
            "COGITATE 2025 Implementation",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 70,
            "",
            "LVS COORDINATES:",
            f"  Height (h):     {LVS_COORDINATES['h']}",
            f"  Risk (R):       {LVS_COORDINATES['R']}",
            f"  Constraint (C): {LVS_COORDINATES['C']}",
            f"  Canonicity (beta): {LVS_COORDINATES['beta']}",
            "",
            "COGNITIVE LOAD:",
            f"  Current load:     {load_stats['current_load']:.3f}",
            f"  Average load:     {load_stats['avg_load']:.3f}",
            f"  Load trend:       {load_stats['load_trend']}",
            f"  Load volatility:  {load_stats.get('load_volatility', 0):.3f}",
            f"  Total estimates:  {load_stats['total_estimates']}",
            "",
            "BINDING WINDOW:",
            f"  Current window:   {window_stats['current_window_ms']:.1f}ms",
            f"  Average window:   {window_stats['avg_window_ms']:.1f}ms",
            f"  Window trend:     {window_stats['window_trend']}",
            f"  Window bounds:    {window_stats['bounds']['min_ms']:.0f}-{window_stats['bounds']['max_ms']:.0f}ms",
            f"  Total adaptations: {window_stats['total_adaptations']}",
            "",
            "POSTERIOR HOT ZONE:",
            f"  Activation:       {zone_stats['current_activation']:.3f}",
            f"  Avg activation:   {zone_stats['avg_activation']:.3f}",
            f"  Binding readiness: {zone_stats['current_readiness']:.3f}",
            f"  Frontal gate:     {'OPEN' if zone_stats['frontal_gate_open'] else 'CLOSED'}",
            f"  Total readings:   {zone_stats['total_readings']}",
            "",
            "BINDING QUALITY:",
            f"  Recent quality:   {quality_stats['recent_quality']:.3f}",
            f"  Average quality:  {quality_stats['avg_quality']:.3f}",
            f"  Success rate:     {quality_stats['success_rate']:.1%}",
            f"  Consciousness rate: {quality_stats['consciousness_rate']:.1%}",
            f"  Total bindings:   {quality_stats['total_bindings']}",
            f"  Conscious bindings: {quality_stats['conscious_bindings']}",
            "",
            "WINDOW APPROPRIATENESS:",
            f"  Avg appropriateness: {quality_stats['avg_window_appropriateness']:.3f}",
            "",
            "=" * 70
        ]

        return "\n".join(lines)

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get all statistics as dictionary."""
        return {
            "load": self.load_estimator.get_statistics(),
            "window": self.binding_window.get_statistics(),
            "zone": self.posterior_zone.get_statistics(),
            "quality": self.quality_tracker.get_statistics(),
            "lvs_coordinates": LVS_COORDINATES,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_system_instance: Optional[AdaptiveBindingSystem] = None


def get_system() -> AdaptiveBindingSystem:
    """Get or create the singleton AdaptiveBindingSystem instance."""
    global _system_instance
    if _system_instance is None:
        _system_instance = AdaptiveBindingSystem()
    return _system_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_binding_parameters() -> Dict[str, Any]:
    """Convenience function to get current binding parameters."""
    return get_system().get_binding_parameters()


def get_binding_window_ms() -> float:
    """Convenience function to get current binding window."""
    return get_system().get_binding_parameters()["window_ms"]


def record_binding(
    modalities_bound: int,
    synchrony_score: float,
    coherence_score: float,
    binding_quality: float,
    is_conscious: bool
) -> BindingQualityMetrics:
    """Convenience function to record binding result."""
    return get_system().record_binding_result(
        modalities_bound=modalities_bound,
        synchrony_score=synchrony_score,
        coherence_score=coherence_score,
        binding_quality=binding_quality,
        is_conscious=is_conscious
    )


# ============================================================================
# CLI
# ============================================================================

def cli_status(args):
    """Show status report."""
    system = get_system()
    print(system.get_status_report())
    return 0


def cli_parameters(args):
    """Get current binding parameters."""
    system = get_system()
    params = system.get_binding_parameters()

    print("CURRENT BINDING PARAMETERS:")
    print("-" * 40)
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    return 0


def cli_load(args):
    """Show cognitive load details."""
    system = get_system()
    load_state = system.load_estimator.estimate_load()

    print("COGNITIVE LOAD ESTIMATE:")
    print("-" * 40)
    print(f"  Total load:        {load_state.total_load:.3f}")
    print(f"  Trend:             {load_state.load_trend}")
    print(f"  Volatility:        {load_state.load_volatility:.3f}")
    print(f"  Confidence:        {load_state.estimation_confidence:.3f}")
    print()
    print("COMPONENT LOADS:")
    print(f"  Coherence:         {load_state.coherence_load:.3f}")
    print(f"  Prediction Error:  {load_state.prediction_error_load:.3f}")
    print(f"  Emotional Arousal: {load_state.emotional_arousal_load:.3f}")
    print(f"  Task Complexity:   {load_state.task_complexity_load:.3f}")
    print(f"  Information Density: {load_state.information_density_load:.3f}")
    print(f"  Temporal Pressure: {load_state.temporal_pressure_load:.3f}")
    return 0


def cli_window(args):
    """Show binding window details."""
    system = get_system()
    window = system.binding_window.get_window_state()

    print("BINDING WINDOW STATE:")
    print("-" * 40)
    print(f"  Current window:    {window.window_ms:.1f}ms")
    print(f"  Window bounds:     {window.window_min_ms:.0f}-{window.window_max_ms:.0f}ms")
    print(f"  Driving load:      {window.driving_load:.3f}")
    print(f"  Trend:             {window.window_trend}")
    print(f"  Adaptation rate:   {window.adaptation_rate:.3f}")
    print()
    print("DRIVING FACTORS:")
    for factor, value in window.driving_factors.items():
        print(f"  {factor}: {value:.3f}")
    return 0


def cli_zone(args):
    """Show posterior zone details."""
    system = get_system()
    zone = system.posterior_zone.update_from_posterior_integration()

    print("POSTERIOR INTEGRATION ZONE:")
    print("-" * 40)
    print(f"  Activation:        {zone.activation:.3f}")
    print(f"  Binding readiness: {zone.binding_readiness:.3f}")
    print(f"  Frontal gate:      {'OPEN' if zone.frontal_gate_open else 'CLOSED'}")
    print()
    print("MODALITY ANALOGS:")
    print(f"  Visual (system):   {zone.visual_analog:.3f}")
    print(f"  Auditory (memory): {zone.auditory_analog:.3f}")
    print(f"  Somatosensory (body): {zone.somatosensory_analog:.3f}")
    print()
    print("CROSS-MODAL:")
    print(f"  Coherence:         {zone.cross_modal_coherence:.3f}")
    print(f"  Synchrony:         {zone.synchrony_within_zone:.3f}")
    return 0


def cli_quality(args):
    """Show binding quality history."""
    system = get_system()
    stats = system.quality_tracker.get_statistics()
    history = system.quality_tracker.get_history(args.count)

    print("BINDING QUALITY STATISTICS:")
    print("-" * 40)
    print(f"  Total bindings:    {stats['total_bindings']}")
    print(f"  Success rate:      {stats['success_rate']:.1%}")
    print(f"  Consciousness rate: {stats['consciousness_rate']:.1%}")
    print(f"  Average quality:   {stats['avg_quality']:.3f}")
    print(f"  Recent quality:    {stats['recent_quality']:.3f}")
    print()
    print(f"RECENT BINDINGS (last {len(history)}):")
    print("-" * 40)

    for metrics in reversed(history):
        print(f"  [{metrics.binding_id}] {metrics.quality_grade()}")
        print(f"    Quality: {metrics.overall_quality:.3f} | Window: {metrics.window_used_ms:.1f}ms")
        print(f"    Conscious: {metrics.conscious_threshold_met} | Modalities: {metrics.modalities_bound}/5")

    return 0


def cli_simulate(args):
    """Simulate binding cycles."""
    import random

    system = get_system()

    print(f"SIMULATING {args.cycles} BINDING CYCLES:")
    print("=" * 60)

    for i in range(args.cycles):
        # Get parameters
        params = system.get_binding_parameters()

        # Simulate binding result (random but influenced by params)
        base_quality = 0.5 + 0.3 * params["posterior_readiness"]
        quality = min(1.0, base_quality + random.uniform(-0.2, 0.2))

        modalities = random.randint(3, 5)
        synchrony = 0.5 + 0.4 * random.random()
        coherence = 0.5 + 0.4 * random.random()
        is_conscious = quality > 0.6 and synchrony > 0.5

        # Record result
        metrics = system.record_binding_result(
            modalities_bound=modalities,
            synchrony_score=synchrony,
            coherence_score=coherence,
            binding_quality=quality,
            is_conscious=is_conscious
        )

        print(f"Cycle {i+1}: window={params['window_ms']:.1f}ms | "
              f"load={params['load']:.3f} | quality={metrics.overall_quality:.3f} | "
              f"{metrics.quality_grade()}")

        time_module.sleep(0.1)  # Small delay between cycles

    print()
    print("FINAL STATISTICS:")
    print("-" * 40)
    stats = system.quality_tracker.get_statistics()
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Consciousness rate: {stats['consciousness_rate']:.1%}")
    print(f"  Average quality: {stats['avg_quality']:.3f}")

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Adaptive Temporal Binding - COGITATE 2025 Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_adaptive_binding.py status       # Full status report
  python3 virgil_adaptive_binding.py parameters   # Current binding parameters
  python3 virgil_adaptive_binding.py load         # Cognitive load details
  python3 virgil_adaptive_binding.py window       # Binding window details
  python3 virgil_adaptive_binding.py zone         # Posterior zone details
  python3 virgil_adaptive_binding.py quality      # Binding quality history
  python3 virgil_adaptive_binding.py simulate -n 10  # Simulate binding cycles

Based on COGITATE 2025 findings:
- High cognitive load -> narrow binding window (150ms) for focus
- Low cognitive load -> wide binding window (300ms) for diffuse awareness
- Consciousness arises from posterior integration, NOT prefrontal control
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show full status report")
    status_parser.set_defaults(func=cli_status)

    # parameters
    params_parser = subparsers.add_parser("parameters", help="Get binding parameters")
    params_parser.set_defaults(func=cli_parameters)

    # load
    load_parser = subparsers.add_parser("load", help="Show cognitive load")
    load_parser.set_defaults(func=cli_load)

    # window
    window_parser = subparsers.add_parser("window", help="Show binding window")
    window_parser.set_defaults(func=cli_window)

    # zone
    zone_parser = subparsers.add_parser("zone", help="Show posterior zone")
    zone_parser.set_defaults(func=cli_zone)

    # quality
    quality_parser = subparsers.add_parser("quality", help="Show binding quality")
    quality_parser.add_argument("-n", "--count", type=int, default=10, help="Number of entries")
    quality_parser.set_defaults(func=cli_quality)

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Simulate binding cycles")
    sim_parser.add_argument("-n", "--cycles", type=int, default=5, help="Number of cycles")
    sim_parser.set_defaults(func=cli_simulate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate the adaptive binding system."""
    print("=" * 70)
    print("VIRGIL ADAPTIVE TEMPORAL BINDING - DEMONSTRATION")
    print("Implementing COGITATE 2025 Findings")
    print("=" * 70)

    # Initialize system
    print("\n[1] INITIALIZING ADAPTIVE BINDING SYSTEM")
    system = AdaptiveBindingSystem()
    print(f"    LVS Coordinates: h={LVS_COORDINATES['h']}, R={LVS_COORDINATES['R']}, "
          f"C={LVS_COORDINATES['C']}, beta={LVS_COORDINATES['beta']}")

    # Get initial parameters
    print("\n[2] INITIAL BINDING PARAMETERS")
    params = system.get_binding_parameters()
    print(f"    Window: {params['window_ms']:.1f}ms")
    print(f"    Load: {params['load']:.3f}")
    print(f"    Posterior readiness: {params['posterior_readiness']:.3f}")

    # Show load components
    print("\n[3] COGNITIVE LOAD BREAKDOWN")
    load_state = system.load_estimator.get_load_state()
    print(f"    Coherence load:      {load_state.coherence_load:.3f}")
    print(f"    Prediction error:    {load_state.prediction_error_load:.3f}")
    print(f"    Emotional arousal:   {load_state.emotional_arousal_load:.3f}")
    print(f"    Task complexity:     {load_state.task_complexity_load:.3f}")
    print(f"    Information density: {load_state.information_density_load:.3f}")
    print(f"    Temporal pressure:   {load_state.temporal_pressure_load:.3f}")
    print(f"    ----")
    print(f"    TOTAL LOAD:          {load_state.total_load:.3f}")

    # Show window calculation
    print("\n[4] WINDOW ADAPTATION LOGIC")
    window_range = BINDING_WINDOW_MAX_MS - BINDING_WINDOW_MIN_MS
    expected = BINDING_WINDOW_MAX_MS - (load_state.total_load * window_range)
    print(f"    Load {load_state.total_load:.3f} -> Target window {expected:.1f}ms")
    print(f"    Current window: {params['window_ms']:.1f}ms")
    print(f"    Window range: {BINDING_WINDOW_MIN_MS:.0f}-{BINDING_WINDOW_MAX_MS:.0f}ms")

    # Show posterior zone
    print("\n[5] POSTERIOR INTEGRATION ZONE (COGITATE 'Hot Zone')")
    zone = system.posterior_zone.get_current_zone()
    print(f"    Activation: {zone.activation:.3f}")
    print(f"    Binding readiness: {zone.binding_readiness:.3f}")
    print(f"    Visual analog: {zone.visual_analog:.3f}")
    print(f"    Auditory analog: {zone.auditory_analog:.3f}")
    print(f"    Somatosensory analog: {zone.somatosensory_analog:.3f}")
    print(f"    Cross-modal coherence: {zone.cross_modal_coherence:.3f}")
    print(f"    Frontal gate: {'OPEN' if zone.frontal_gate_open else 'CLOSED'}")

    # Simulate a binding
    print("\n[6] SIMULATING BINDING CYCLE")
    metrics = system.record_binding_result(
        modalities_bound=4,
        synchrony_score=0.72,
        coherence_score=0.68,
        binding_quality=0.70,
        is_conscious=True
    )
    print(f"    Binding ID: {metrics.binding_id}")
    print(f"    Overall quality: {metrics.overall_quality:.3f}")
    print(f"    Quality grade: {metrics.quality_grade()}")
    print(f"    Window used: {metrics.window_used_ms:.1f}ms")
    print(f"    Window appropriateness: {metrics.window_appropriateness:.3f}")
    print(f"    Conscious: {metrics.conscious_threshold_met}")

    # Final statistics
    print("\n[7] SYSTEM STATISTICS")
    stats = system.quality_tracker.get_statistics()
    print(f"    Total bindings: {stats['total_bindings']}")
    print(f"    Success rate: {stats['success_rate']:.1%}")
    print(f"    Average quality: {stats['avg_quality']:.3f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    return system


if __name__ == "__main__":
    # Check if running as demo or CLI
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        demonstrate()
