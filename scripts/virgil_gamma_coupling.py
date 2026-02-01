#!/usr/bin/env python3
"""
VIRGIL CROSS-FREQUENCY COUPLING - Theta-Gamma Phase-Amplitude Coupling

This module implements neural oscillation dynamics based on neuroscience research
showing that gamma oscillations (30-100 Hz) nested within theta rhythms (4-8 Hz)
encode discrete memory items and enable cognitive binding.

Key Neuroscience Principles:
1. Theta rhythm (~6 Hz, ~167ms cycle) provides the encoding window
2. Each theta cycle can contain 4-8 gamma bursts (working memory capacity)
3. High gamma (>60 Hz) correlates with insight moments
4. Phase-amplitude coupling (PAC) strength predicts memory encoding success
5. Cross-regional gamma coherence enables feature binding

Integration with Virgil:
- Gamma bursts represent discrete thoughts/memories in working focus
- Theta cycles define natural encoding windows
- Insight detection triggers special processing
- Heartbeat phases modulate coupling strength

Metaphor:
The theta wave is the breath, the gamma bursts are the words spoken within each breath.
4-8 words per breath is natural; more causes fragmentation.

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import uuid
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
from collections import deque
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
GAMMA_LOG_FILE = NEXUS_DIR / "gamma_log.json"
INSIGHT_JOURNAL_FILE = NEXUS_DIR / "insight_journal.json"

# Configure logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [GAMMA] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "gamma_coupling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Physiological constants (in Hz and milliseconds)
THETA_FREQUENCY_MIN = 4.0   # Hz
THETA_FREQUENCY_MAX = 8.0   # Hz
THETA_FREQUENCY_DEFAULT = 6.0  # Hz (hippocampal theta)
THETA_CYCLE_MS = 167  # ~6 Hz = 167ms per cycle

GAMMA_FREQUENCY_MIN = 30.0  # Hz
GAMMA_FREQUENCY_MAX = 100.0  # Hz
HIGH_GAMMA_THRESHOLD = 60.0  # Hz (insight threshold)

# Working memory capacity (Miller's 7 +/- 2, refined to 4-8 for gamma)
GAMMA_BURSTS_PER_CYCLE_MIN = 4
GAMMA_BURSTS_PER_CYCLE_MAX = 8
GAMMA_BURSTS_DEFAULT_CAPACITY = 7  # "Magical number seven"

# Coupling thresholds
STRONG_COUPLING_THRESHOLD = 0.7
WEAK_COUPLING_THRESHOLD = 0.3
INSIGHT_COHERENCE_THRESHOLD = 0.8

# Heartbeat phase mapping
HEARTBEAT_PHASE_COUPLING = {
    "active": {"gamma_gain": 1.0, "coupling_boost": 1.2, "theta_coherence": 0.9},
    "background": {"gamma_gain": 0.7, "coupling_boost": 1.0, "theta_coherence": 0.7},
    "dormant": {"gamma_gain": 0.3, "coupling_boost": 0.5, "theta_coherence": 0.4},
    "deep_sleep": {"gamma_gain": 0.1, "coupling_boost": 0.2, "theta_coherence": 0.2},
    "ghost": {"gamma_gain": 0.0, "coupling_boost": 0.0, "theta_coherence": 0.1}
}


# ============================================================================
# FREQUENCY BANDS
# ============================================================================

class FrequencyBand(Enum):
    """
    Neural oscillation frequency bands with functional descriptions.

    Each band serves distinct cognitive functions:
    - DELTA: Deep processing, whole-system integration
    - THETA: Memory encoding, hippocampal rhythm
    - ALPHA: Relaxed attention, sensory gating
    - BETA: Active thinking, motor preparation
    - GAMMA: Feature binding, insight, high-resolution processing
    """
    DELTA = "delta"      # 0.5-4 Hz: Whole-body integration, low resolution
    THETA = "theta"      # 4-8 Hz: Memory encoding window, hippocampal rhythm
    ALPHA = "alpha"      # 8-13 Hz: Relaxed attention, inhibition
    BETA = "beta"        # 13-30 Hz: Active thinking, motor
    GAMMA = "gamma"      # 30-100 Hz: Binding, insight, high-frequency bursts

    @classmethod
    def from_frequency(cls, freq_hz: float) -> 'FrequencyBand':
        """
        Determine frequency band from Hz value.

        Args:
            freq_hz: Frequency in Hertz

        Returns:
            Corresponding FrequencyBand enum value
        """
        if freq_hz < 0.5:
            return cls.DELTA  # Subsonic treated as delta
        elif freq_hz < 4.0:
            return cls.DELTA
        elif freq_hz < 8.0:
            return cls.THETA
        elif freq_hz < 13.0:
            return cls.ALPHA
        elif freq_hz < 30.0:
            return cls.BETA
        else:
            return cls.GAMMA

    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Get the frequency range (min, max) in Hz for this band."""
        ranges = {
            FrequencyBand.DELTA: (0.5, 4.0),
            FrequencyBand.THETA: (4.0, 8.0),
            FrequencyBand.ALPHA: (8.0, 13.0),
            FrequencyBand.BETA: (13.0, 30.0),
            FrequencyBand.GAMMA: (30.0, 100.0)
        }
        return ranges[self]

    @property
    def cognitive_function(self) -> str:
        """Get the primary cognitive function of this band."""
        functions = {
            FrequencyBand.DELTA: "whole-body integration, deep sleep",
            FrequencyBand.THETA: "memory encoding, spatial navigation",
            FrequencyBand.ALPHA: "relaxed attention, sensory inhibition",
            FrequencyBand.BETA: "active thinking, motor preparation",
            FrequencyBand.GAMMA: "feature binding, insight, attention"
        }
        return functions[self]

    @property
    def typical_cycle_ms(self) -> float:
        """Get typical cycle duration in milliseconds."""
        # Using midpoint of frequency range
        min_hz, max_hz = self.frequency_range
        mid_hz = (min_hz + max_hz) / 2
        return 1000.0 / mid_hz


# ============================================================================
# GAMMA BURST EVENT
# ============================================================================

@dataclass
class GammaBurst:
    """
    Represents a single gamma burst event - a discrete unit of cognition.

    In neuroscience, each gamma burst represents a single item being held in
    working memory or a discrete thought/percept. The timing of the burst
    within the theta cycle (phase) determines its position in the sequence.

    Attributes:
        burst_id: Unique identifier for this burst
        timestamp: ISO timestamp of burst occurrence
        theta_phase: Phase within theta cycle (0 to 2*pi radians)
                    - 0 to pi/2: Rising phase (encoding)
                    - pi/2 to pi: Peak (consolidation)
                    - pi to 3*pi/2: Falling phase (retrieval)
                    - 3*pi/2 to 2*pi: Trough (preparation)
        gamma_frequency: Frequency of this burst in Hz (30-100)
        amplitude: Normalized amplitude (0-1), reflects intensity
        content_id: Reference to what thought/memory this burst represents
        coupling_strength: How well-nested within theta (0-1)
        is_high_gamma: Whether this is a high gamma burst (>60Hz, insight)
        region: Which cognitive "region" generated this burst
        metadata: Additional contextual information
    """
    burst_id: str
    timestamp: str
    theta_phase: float  # 0 to 2*pi
    gamma_frequency: float  # 30-100 Hz
    amplitude: float  # 0-1
    content_id: str  # What thought/memory this burst represents
    coupling_strength: float  # How well nested in theta
    is_high_gamma: bool = False  # >60 Hz indicates insight
    region: str = "default"  # Cognitive region
    duration_ms: float = 10.0  # Typical gamma burst duration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Clamp values to valid ranges
        self.theta_phase = self.theta_phase % (2 * math.pi)
        self.gamma_frequency = max(GAMMA_FREQUENCY_MIN,
                                   min(GAMMA_FREQUENCY_MAX, self.gamma_frequency))
        self.amplitude = max(0.0, min(1.0, self.amplitude))
        self.coupling_strength = max(0.0, min(1.0, self.coupling_strength))

        # Determine if high gamma
        self.is_high_gamma = self.gamma_frequency >= HIGH_GAMMA_THRESHOLD

    @property
    def phase_label(self) -> str:
        """Human-readable label for the theta phase position."""
        phase_normalized = self.theta_phase / (2 * math.pi)
        if phase_normalized < 0.25:
            return "rising"
        elif phase_normalized < 0.5:
            return "peak"
        elif phase_normalized < 0.75:
            return "falling"
        else:
            return "trough"

    @property
    def encoding_quality(self) -> float:
        """
        Estimate encoding quality based on phase and coupling.

        Bursts occurring at theta peak with strong coupling encode best.
        """
        # Optimal phase is around pi/2 (peak)
        phase_optimality = math.cos(self.theta_phase - math.pi/2) * 0.5 + 0.5
        return phase_optimality * self.coupling_strength * self.amplitude

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "burst_id": self.burst_id,
            "timestamp": self.timestamp,
            "theta_phase": self.theta_phase,
            "theta_phase_label": self.phase_label,
            "gamma_frequency": self.gamma_frequency,
            "amplitude": self.amplitude,
            "content_id": self.content_id,
            "coupling_strength": self.coupling_strength,
            "is_high_gamma": self.is_high_gamma,
            "region": self.region,
            "duration_ms": self.duration_ms,
            "encoding_quality": self.encoding_quality,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GammaBurst':
        """Create from dictionary."""
        return cls(
            burst_id=data.get("burst_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            theta_phase=data.get("theta_phase", 0.0),
            gamma_frequency=data.get("gamma_frequency", 40.0),
            amplitude=data.get("amplitude", 0.5),
            content_id=data.get("content_id", "unknown"),
            coupling_strength=data.get("coupling_strength", 0.5),
            is_high_gamma=data.get("is_high_gamma", False),
            region=data.get("region", "default"),
            duration_ms=data.get("duration_ms", 10.0),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# INSIGHT EVENT
# ============================================================================

@dataclass
class InsightEvent:
    """
    Represents a detected insight event - a moment of cognitive binding.

    Insights are characterized by:
    - High gamma bursts (>60 Hz)
    - Coherent gamma across multiple regions (binding)
    - Often preceded by alpha suppression
    - Associated with "aha!" moments in cognition

    Attributes:
        insight_id: Unique identifier
        timestamp: When the insight occurred
        triggering_bursts: The gamma bursts that triggered detection
        coherence_score: How coherent the gamma activity was (0-1)
        content_ids: What content was being processed
        insight_type: Classification of insight type
        intensity: Overall intensity of the insight (0-1)
        description: Human-readable description
    """
    insight_id: str
    timestamp: str
    triggering_bursts: List[str]  # burst_ids
    coherence_score: float
    content_ids: List[str]
    insight_type: str = "binding"  # binding, synthesis, recall, novel
    intensity: float = 0.5
    description: str = ""
    crystallized: bool = False  # Whether this insight was saved to memory

    def __post_init__(self):
        """Validate and compute derived fields."""
        self.coherence_score = max(0.0, min(1.0, self.coherence_score))
        self.intensity = max(0.0, min(1.0, self.intensity))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightEvent':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})


# ============================================================================
# COUPLING STATE
# ============================================================================

@dataclass
class CouplingState:
    """
    Current state of theta-gamma coupling in the system.

    This represents a snapshot of the oscillatory dynamics at a given moment,
    including the theta phase, recent gamma bursts, and coupling metrics.

    Attributes:
        current_theta_phase: Current position in theta cycle (0 to 2*pi)
        theta_frequency: Current theta frequency in Hz
        gamma_bursts_this_cycle: Bursts that occurred in current theta cycle
        coupling_strength: Overall PAC (phase-amplitude coupling) strength
        dominant_frequency: Which frequency band is dominant
        insight_active: Whether an insight moment is currently happening
        working_memory_load: How many items in current working memory (0-8)
        heartbeat_phase: Current Virgil heartbeat phase
        coherence_across_regions: Gamma coherence across cognitive regions
    """
    current_theta_phase: float = 0.0
    theta_frequency: float = THETA_FREQUENCY_DEFAULT
    gamma_bursts_this_cycle: List[GammaBurst] = field(default_factory=list)
    coupling_strength: float = 0.5
    dominant_frequency: FrequencyBand = FrequencyBand.THETA
    insight_active: bool = False
    working_memory_load: int = 0
    heartbeat_phase: str = "active"
    coherence_across_regions: float = 0.5
    last_updated: str = ""

    def __post_init__(self):
        """Initialize timestamp if not set."""
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

    @property
    def cycle_capacity_remaining(self) -> int:
        """How many more gamma bursts can fit in this theta cycle."""
        current_load = len(self.gamma_bursts_this_cycle)
        return max(0, GAMMA_BURSTS_DEFAULT_CAPACITY - current_load)

    @property
    def is_overloaded(self) -> bool:
        """Whether the current cycle exceeds capacity (cognitive overload)."""
        return len(self.gamma_bursts_this_cycle) > GAMMA_BURSTS_PER_CYCLE_MAX

    @property
    def encoding_window_quality(self) -> str:
        """Qualitative assessment of current encoding conditions."""
        if self.coupling_strength >= STRONG_COUPLING_THRESHOLD:
            if self.insight_active:
                return "optimal_insight"
            return "optimal"
        elif self.coupling_strength >= WEAK_COUPLING_THRESHOLD:
            return "adequate"
        else:
            return "poor"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_theta_phase": self.current_theta_phase,
            "theta_phase_label": self._phase_label(),
            "theta_frequency": self.theta_frequency,
            "gamma_bursts_count": len(self.gamma_bursts_this_cycle),
            "gamma_burst_ids": [b.burst_id for b in self.gamma_bursts_this_cycle],
            "coupling_strength": self.coupling_strength,
            "dominant_frequency": self.dominant_frequency.value,
            "insight_active": self.insight_active,
            "working_memory_load": self.working_memory_load,
            "cycle_capacity_remaining": self.cycle_capacity_remaining,
            "is_overloaded": self.is_overloaded,
            "encoding_window_quality": self.encoding_window_quality,
            "heartbeat_phase": self.heartbeat_phase,
            "coherence_across_regions": self.coherence_across_regions,
            "last_updated": self.last_updated
        }

    def _phase_label(self) -> str:
        """Get human-readable phase label."""
        phase_normalized = self.current_theta_phase / (2 * math.pi)
        if phase_normalized < 0.25:
            return "rising"
        elif phase_normalized < 0.5:
            return "peak"
        elif phase_normalized < 0.75:
            return "falling"
        else:
            return "trough"


# ============================================================================
# COUPLING METRICS CALCULATOR
# ============================================================================

class CouplingMetrics:
    """
    Calculate phase-amplitude coupling (PAC) metrics.

    PAC is the primary measure of how well gamma oscillations are nested
    within theta rhythms. Strong PAC correlates with successful memory
    encoding and cognitive binding.
    """

    @staticmethod
    def calculate_pac(bursts: List[GammaBurst],
                      theta_frequency: float = THETA_FREQUENCY_DEFAULT) -> float:
        """
        Calculate phase-amplitude coupling strength.

        Uses the Mean Vector Length (MVL) method:
        1. For each gamma burst, record its theta phase and amplitude
        2. Treat each as a vector in the unit circle
        3. MVL = |mean vector| (0 = no coupling, 1 = perfect coupling)

        Args:
            bursts: List of gamma bursts to analyze
            theta_frequency: Theta frequency for phase calculation

        Returns:
            PAC strength (0-1)
        """
        if not bursts:
            return 0.0

        # Convert to complex numbers on unit circle, weighted by amplitude
        vectors = []
        for burst in bursts:
            # Each burst contributes a vector at its phase angle
            # weighted by its amplitude
            angle = burst.theta_phase
            magnitude = burst.amplitude
            vector = magnitude * complex(math.cos(angle), math.sin(angle))
            vectors.append(vector)

        # Mean vector
        mean_vector = sum(vectors) / len(vectors)

        # MVL is the magnitude of the mean vector
        mvl = abs(mean_vector)

        return min(1.0, mvl)

    @staticmethod
    def calculate_modulation_index(bursts: List[GammaBurst],
                                   n_bins: int = 18) -> float:
        """
        Calculate Modulation Index (MI) for PAC.

        MI measures how non-uniformly gamma amplitude is distributed
        across theta phases. Uses Kullback-Leibler divergence.

        Args:
            bursts: List of gamma bursts
            n_bins: Number of phase bins

        Returns:
            Modulation index (0-1, higher = stronger coupling)
        """
        if len(bursts) < n_bins:
            return 0.0

        # Bin the amplitudes by theta phase
        bin_width = 2 * math.pi / n_bins
        bins = [[] for _ in range(n_bins)]

        for burst in bursts:
            bin_idx = int(burst.theta_phase / bin_width) % n_bins
            bins[bin_idx].append(burst.amplitude)

        # Mean amplitude per bin
        mean_amps = []
        for bin_amps in bins:
            if bin_amps:
                mean_amps.append(sum(bin_amps) / len(bin_amps))
            else:
                mean_amps.append(0.0)

        # Normalize to probability distribution
        total = sum(mean_amps)
        if total == 0:
            return 0.0

        p = [a / total for a in mean_amps]

        # Uniform distribution
        q = 1.0 / n_bins

        # KL divergence (entropy approach)
        kl = 0.0
        for pi in p:
            if pi > 0:
                kl += pi * math.log(pi / q)

        # Normalize by max possible KL (log(n_bins))
        max_kl = math.log(n_bins)
        mi = kl / max_kl if max_kl > 0 else 0.0

        return min(1.0, mi)

    @staticmethod
    def calculate_cross_region_coherence(bursts: List[GammaBurst]) -> float:
        """
        Calculate gamma coherence across different cognitive regions.

        High coherence indicates feature binding - different aspects
        of cognition are synchronized.

        Args:
            bursts: List of gamma bursts from different regions

        Returns:
            Cross-region coherence (0-1)
        """
        if len(bursts) < 2:
            return 0.0

        # Group bursts by region
        regions: Dict[str, List[GammaBurst]] = {}
        for burst in bursts:
            if burst.region not in regions:
                regions[burst.region] = []
            regions[burst.region].append(burst)

        if len(regions) < 2:
            return 0.0  # Need at least 2 regions for cross-region coherence

        # Calculate pairwise coherence between regions
        region_names = list(regions.keys())
        coherences = []

        for i, r1 in enumerate(region_names):
            for r2 in region_names[i+1:]:
                bursts1 = regions[r1]
                bursts2 = regions[r2]

                # Mean phase difference
                if bursts1 and bursts2:
                    phases1 = [b.theta_phase for b in bursts1]
                    phases2 = [b.theta_phase for b in bursts2]

                    mean_phase1 = sum(phases1) / len(phases1)
                    mean_phase2 = sum(phases2) / len(phases2)

                    # Phase locking value (simplified)
                    phase_diff = abs(mean_phase1 - mean_phase2)
                    coherence = math.cos(phase_diff) * 0.5 + 0.5
                    coherences.append(coherence)

        return sum(coherences) / len(coherences) if coherences else 0.0


# ============================================================================
# THETA-GAMMA COUPLING ENGINE
# ============================================================================

class ThetaGammaCoupling:
    """
    Main engine for theta-gamma coupling dynamics.

    This class maintains the oscillatory state, manages gamma bursts,
    detects insights, and integrates with the heartbeat system.

    The engine operates on a continuous theta cycle, advancing phase
    with each tick or burst registration. Gamma bursts are nested
    within theta based on their timing.

    Usage:
        coupling = ThetaGammaCoupling()

        # Register a thought/memory as a gamma burst
        burst = coupling.register_gamma_burst("memory_123", intensity=0.8)

        # Check coupling state
        state = coupling.get_state()

        # Advance to next theta cycle
        coupling.advance_theta_cycle()

        # Check for insights
        insight = coupling.detect_insight()
    """

    def __init__(self, theta_frequency: float = THETA_FREQUENCY_DEFAULT):
        """
        Initialize the coupling engine.

        Args:
            theta_frequency: Base theta frequency in Hz (4-8)
        """
        self.theta_frequency = max(THETA_FREQUENCY_MIN,
                                   min(THETA_FREQUENCY_MAX, theta_frequency))

        # Current state
        self._state = CouplingState(theta_frequency=self.theta_frequency)

        # History tracking
        self._cycle_number: int = 0
        self._burst_history: deque = deque(maxlen=1000)  # Recent bursts
        self._insight_history: deque = deque(maxlen=100)  # Recent insights

        # Metrics
        self._pac_history: List[Tuple[int, float]] = []  # (cycle, pac)

        # Thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._insight_callbacks: List[Callable[[InsightEvent], None]] = []
        self._overload_callbacks: List[Callable[[CouplingState], None]] = []

        # Load persisted state
        self._load_state()

        logger.info(f"ThetaGammaCoupling initialized: theta={self.theta_frequency}Hz")

    def _load_state(self):
        """Load persisted gamma log and state."""
        try:
            if GAMMA_LOG_FILE.exists():
                data = json.loads(GAMMA_LOG_FILE.read_text())
                self._cycle_number = data.get("cycle_number", 0)

                # Restore recent bursts
                for burst_data in data.get("recent_bursts", [])[-100:]:
                    burst = GammaBurst.from_dict(burst_data)
                    self._burst_history.append(burst)

                # Restore PAC history
                self._pac_history = data.get("pac_history", [])[-100:]

                logger.info(f"Loaded gamma state: cycle={self._cycle_number}")
        except Exception as e:
            logger.warning(f"Could not load gamma state: {e}")

    def _save_state(self):
        """Persist gamma log and state."""
        try:
            GAMMA_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "cycle_number": self._cycle_number,
                "theta_frequency": self.theta_frequency,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "recent_bursts": [b.to_dict() for b in list(self._burst_history)[-100:]],
                "pac_history": self._pac_history[-100:],
                "current_state": self._state.to_dict()
            }

            GAMMA_LOG_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save gamma state: {e}")

    def _save_insight(self, insight: InsightEvent):
        """Save insight to journal."""
        try:
            INSIGHT_JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)

            journal = {"insights": [], "total_count": 0}
            if INSIGHT_JOURNAL_FILE.exists():
                journal = json.loads(INSIGHT_JOURNAL_FILE.read_text())

            journal["insights"].append(insight.to_dict())
            journal["insights"] = journal["insights"][-200:]  # Keep last 200
            journal["total_count"] = journal.get("total_count", 0) + 1
            journal["last_updated"] = datetime.now(timezone.utc).isoformat()

            INSIGHT_JOURNAL_FILE.write_text(json.dumps(journal, indent=2))

        except Exception as e:
            logger.error(f"Failed to save insight: {e}")

    def register_gamma_burst(self,
                            content_id: str,
                            intensity: float = 0.5,
                            region: str = "default",
                            frequency: Optional[float] = None,
                            metadata: Optional[Dict] = None) -> GammaBurst:
        """
        Register a new gamma burst representing a thought or memory item.

        This is the primary method for adding cognitive content to the
        current theta cycle. Each burst represents one discrete item
        in working memory.

        Args:
            content_id: Identifier for the thought/memory this represents
            intensity: How intense/important this burst is (0-1)
            region: Which cognitive region generated this
            frequency: Specific gamma frequency (default: 40-60 Hz based on intensity)
            metadata: Additional contextual information

        Returns:
            The created GammaBurst

        Raises:
            Warning if cycle is overloaded (>8 bursts)
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Calculate gamma frequency based on intensity if not specified
            # Higher intensity -> higher gamma frequency
            if frequency is None:
                # Map intensity 0-1 to frequency 40-80 Hz
                frequency = 40.0 + (intensity * 40.0)

            # Get heartbeat modulation
            hb_params = HEARTBEAT_PHASE_COUPLING.get(
                self._state.heartbeat_phase,
                HEARTBEAT_PHASE_COUPLING["active"]
            )

            # Apply heartbeat gain to amplitude
            amplitude = intensity * hb_params["gamma_gain"]

            # Calculate coupling strength
            # Coupling is better at certain theta phases (peak) and with heartbeat boost
            phase_optimality = math.cos(self._state.current_theta_phase - math.pi/2) * 0.5 + 0.5
            coupling = phase_optimality * hb_params["coupling_boost"]
            coupling = max(0.0, min(1.0, coupling))

            # Create the burst
            burst = GammaBurst(
                burst_id=str(uuid.uuid4()),
                timestamp=now.isoformat(),
                theta_phase=self._state.current_theta_phase,
                gamma_frequency=frequency,
                amplitude=amplitude,
                content_id=content_id,
                coupling_strength=coupling,
                region=region,
                metadata=metadata or {}
            )

            # Add to current cycle
            self._state.gamma_bursts_this_cycle.append(burst)
            self._state.working_memory_load = len(self._state.gamma_bursts_this_cycle)
            self._burst_history.append(burst)

            # Check for overload
            if self._state.is_overloaded:
                logger.warning(f"Theta cycle overloaded: {self._state.working_memory_load} bursts")
                for callback in self._overload_callbacks:
                    try:
                        callback(self._state)
                    except Exception as e:
                        logger.error(f"Overload callback error: {e}")

            # Advance phase slightly (simulating time passing)
            phase_advance = (2 * math.pi) / GAMMA_BURSTS_DEFAULT_CAPACITY
            self._state.current_theta_phase = (
                self._state.current_theta_phase + phase_advance * 0.3
            ) % (2 * math.pi)

            # Update coupling strength
            self._state.coupling_strength = self._calculate_current_coupling()
            self._state.last_updated = now.isoformat()

            # Check for insight
            if burst.is_high_gamma:
                insight = self._check_for_insight()
                if insight:
                    self._state.insight_active = True

            self._save_state()

            logger.debug(f"Registered gamma burst: {content_id} @ {frequency:.1f}Hz, "
                        f"coupling={coupling:.2f}")

            return burst

    def _calculate_current_coupling(self) -> float:
        """Calculate current PAC strength from recent bursts."""
        recent_bursts = self._state.gamma_bursts_this_cycle
        if not recent_bursts:
            return 0.5  # Default when no bursts

        return CouplingMetrics.calculate_pac(recent_bursts, self.theta_frequency)

    def _check_for_insight(self) -> Optional[InsightEvent]:
        """Check if current activity constitutes an insight."""
        bursts = self._state.gamma_bursts_this_cycle

        # Need at least 2 high-gamma bursts for insight
        high_gamma_bursts = [b for b in bursts if b.is_high_gamma]
        if len(high_gamma_bursts) < 2:
            return None

        # Check coherence
        coherence = CouplingMetrics.calculate_cross_region_coherence(high_gamma_bursts)
        if coherence < INSIGHT_COHERENCE_THRESHOLD:
            return None

        # We have an insight!
        insight = InsightEvent(
            insight_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            triggering_bursts=[b.burst_id for b in high_gamma_bursts],
            coherence_score=coherence,
            content_ids=[b.content_id for b in high_gamma_bursts],
            insight_type="binding",
            intensity=sum(b.amplitude for b in high_gamma_bursts) / len(high_gamma_bursts),
            description=f"Coherent high-gamma binding of {len(high_gamma_bursts)} items"
        )

        self._insight_history.append(insight)
        self._save_insight(insight)

        logger.info(f"INSIGHT DETECTED: {insight.insight_id}, coherence={coherence:.2f}")

        # Notify callbacks
        for callback in self._insight_callbacks:
            try:
                callback(insight)
            except Exception as e:
                logger.error(f"Insight callback error: {e}")

        return insight

    def detect_insight(self, bursts: Optional[List[GammaBurst]] = None) -> Optional[InsightEvent]:
        """
        Explicitly check for insight in a set of bursts.

        Args:
            bursts: Bursts to analyze (default: current cycle's bursts)

        Returns:
            InsightEvent if detected, None otherwise
        """
        with self._lock:
            if bursts is None:
                bursts = self._state.gamma_bursts_this_cycle

            # Same logic as _check_for_insight but with explicit burst list
            high_gamma_bursts = [b for b in bursts if b.is_high_gamma]

            if len(high_gamma_bursts) < 2:
                return None

            coherence = CouplingMetrics.calculate_cross_region_coherence(high_gamma_bursts)

            if coherence < INSIGHT_COHERENCE_THRESHOLD:
                return None

            # Determine insight type based on content patterns
            content_ids = set(b.content_id for b in high_gamma_bursts)
            if len(content_ids) == 1:
                insight_type = "recall"  # Deep retrieval of single memory
            elif len(set(b.region for b in high_gamma_bursts)) > 1:
                insight_type = "binding"  # Cross-modal binding
            else:
                insight_type = "synthesis"  # Multiple items, same region

            insight = InsightEvent(
                insight_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                triggering_bursts=[b.burst_id for b in high_gamma_bursts],
                coherence_score=coherence,
                content_ids=list(content_ids),
                insight_type=insight_type,
                intensity=sum(b.amplitude for b in high_gamma_bursts) / len(high_gamma_bursts),
                description=f"{insight_type.title()} insight across {len(content_ids)} items"
            )

            self._insight_history.append(insight)
            self._save_insight(insight)

            return insight

    def advance_theta_cycle(self) -> Dict[str, Any]:
        """
        Advance to the next theta cycle.

        This completes the current cycle, computes metrics, and prepares
        for the next cycle. Should be called periodically or when the
        conceptual encoding window closes.

        Returns:
            Summary of the completed cycle
        """
        with self._lock:
            # Compute metrics for completed cycle
            completed_bursts = list(self._state.gamma_bursts_this_cycle)

            pac = CouplingMetrics.calculate_pac(completed_bursts, self.theta_frequency)
            mi = CouplingMetrics.calculate_modulation_index(completed_bursts)
            coherence = CouplingMetrics.calculate_cross_region_coherence(completed_bursts)

            # Record PAC history
            self._pac_history.append((self._cycle_number, pac))

            # Build cycle summary
            summary = {
                "cycle_number": self._cycle_number,
                "bursts_count": len(completed_bursts),
                "content_ids": [b.content_id for b in completed_bursts],
                "pac_strength": pac,
                "modulation_index": mi,
                "cross_region_coherence": coherence,
                "high_gamma_count": sum(1 for b in completed_bursts if b.is_high_gamma),
                "mean_amplitude": (sum(b.amplitude for b in completed_bursts) /
                                  len(completed_bursts) if completed_bursts else 0),
                "was_overloaded": len(completed_bursts) > GAMMA_BURSTS_PER_CYCLE_MAX,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Reset for next cycle
            self._cycle_number += 1
            self._state.gamma_bursts_this_cycle = []
            self._state.current_theta_phase = 0.0
            self._state.working_memory_load = 0
            self._state.insight_active = False
            self._state.coupling_strength = 0.5  # Reset to baseline
            self._state.last_updated = datetime.now(timezone.utc).isoformat()

            self._save_state()

            logger.info(f"Advanced to cycle {self._cycle_number}, "
                       f"previous had {len(completed_bursts)} bursts, PAC={pac:.2f}")

            return summary

    def get_coupling_strength(self) -> float:
        """
        Get current phase-amplitude coupling strength.

        Returns:
            PAC strength (0-1)
        """
        with self._lock:
            return self._state.coupling_strength

    def get_dominant_band(self) -> FrequencyBand:
        """
        Get the currently dominant frequency band.

        Determined by the activity pattern in the current cycle.

        Returns:
            Dominant FrequencyBand
        """
        with self._lock:
            bursts = self._state.gamma_bursts_this_cycle

            if not bursts:
                # Default to theta when no gamma activity
                return FrequencyBand.THETA

            # Calculate mean gamma frequency
            mean_gamma = sum(b.gamma_frequency for b in bursts) / len(bursts)

            # If high activity, gamma dominates
            if len(bursts) >= 3 and mean_gamma > 50:
                self._state.dominant_frequency = FrequencyBand.GAMMA
            else:
                self._state.dominant_frequency = FrequencyBand.THETA

            return self._state.dominant_frequency

    def get_items_in_current_cycle(self) -> List[str]:
        """
        Get content IDs of items in current theta cycle.

        This represents the current working memory contents.
        Max 4-8 items per theta window.

        Returns:
            List of content_ids
        """
        with self._lock:
            return [b.content_id for b in self._state.gamma_bursts_this_cycle]

    def get_state(self) -> CouplingState:
        """
        Get current coupling state.

        Returns:
            Copy of current CouplingState
        """
        with self._lock:
            # Return a copy to prevent external mutation
            state_copy = CouplingState(
                current_theta_phase=self._state.current_theta_phase,
                theta_frequency=self._state.theta_frequency,
                gamma_bursts_this_cycle=list(self._state.gamma_bursts_this_cycle),
                coupling_strength=self._state.coupling_strength,
                dominant_frequency=self._state.dominant_frequency,
                insight_active=self._state.insight_active,
                working_memory_load=self._state.working_memory_load,
                heartbeat_phase=self._state.heartbeat_phase,
                coherence_across_regions=self._state.coherence_across_regions,
                last_updated=self._state.last_updated
            )
            return state_copy

    def set_heartbeat_phase(self, phase: str):
        """
        Update the heartbeat phase for coupling modulation.

        Args:
            phase: Heartbeat phase (active, background, dormant, deep_sleep, ghost)
        """
        with self._lock:
            if phase in HEARTBEAT_PHASE_COUPLING:
                self._state.heartbeat_phase = phase
                logger.debug(f"Heartbeat phase updated to: {phase}")
            else:
                logger.warning(f"Unknown heartbeat phase: {phase}")

    def get_coupling_for_phase(self, heartbeat_phase: str) -> CouplingState:
        """
        Get expected coupling state for a given heartbeat phase.

        This allows prediction of coupling dynamics based on system state.

        Args:
            heartbeat_phase: The heartbeat phase to query

        Returns:
            CouplingState configured for that phase
        """
        params = HEARTBEAT_PHASE_COUPLING.get(
            heartbeat_phase,
            HEARTBEAT_PHASE_COUPLING["active"]
        )

        # Create a predicted state
        predicted_state = CouplingState(
            current_theta_phase=0.0,
            theta_frequency=self.theta_frequency,
            coupling_strength=params["coupling_boost"] * params["theta_coherence"],
            heartbeat_phase=heartbeat_phase,
            coherence_across_regions=params["theta_coherence"]
        )

        return predicted_state

    def register_insight_callback(self, callback: Callable[[InsightEvent], None]):
        """Register a callback for insight events."""
        self._insight_callbacks.append(callback)

    def register_overload_callback(self, callback: Callable[[CouplingState], None]):
        """Register a callback for cycle overload events."""
        self._overload_callbacks.append(callback)

    def get_pac_trend(self, n_cycles: int = 10) -> List[float]:
        """
        Get recent PAC trend.

        Args:
            n_cycles: Number of recent cycles to include

        Returns:
            List of PAC values
        """
        with self._lock:
            recent = self._pac_history[-n_cycles:]
            return [pac for _, pac in recent]

    def get_insight_history(self, n_insights: int = 10) -> List[InsightEvent]:
        """
        Get recent insight history.

        Args:
            n_insights: Number of recent insights to return

        Returns:
            List of InsightEvents
        """
        with self._lock:
            return list(self._insight_history)[-n_insights:]

    def get_burst_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent gamma bursts.

        Returns:
            Dictionary of burst statistics
        """
        with self._lock:
            bursts = list(self._burst_history)

            if not bursts:
                return {
                    "total_bursts": 0,
                    "high_gamma_ratio": 0.0,
                    "mean_frequency": 0.0,
                    "mean_amplitude": 0.0,
                    "mean_coupling": 0.0,
                    "regions": {}
                }

            high_gamma_count = sum(1 for b in bursts if b.is_high_gamma)

            # Region breakdown
            regions: Dict[str, int] = {}
            for burst in bursts:
                regions[burst.region] = regions.get(burst.region, 0) + 1

            return {
                "total_bursts": len(bursts),
                "high_gamma_ratio": high_gamma_count / len(bursts),
                "mean_frequency": statistics.mean(b.gamma_frequency for b in bursts),
                "mean_amplitude": statistics.mean(b.amplitude for b in bursts),
                "mean_coupling": statistics.mean(b.coupling_strength for b in bursts),
                "frequency_stdev": statistics.stdev(b.gamma_frequency for b in bursts) if len(bursts) > 1 else 0,
                "regions": regions,
                "cycle_count": self._cycle_number
            }

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry for monitoring.

        Returns:
            Dictionary with all coupling metrics
        """
        with self._lock:
            return {
                "state": self._state.to_dict(),
                "statistics": self.get_burst_statistics(),
                "pac_trend": self.get_pac_trend(),
                "recent_insights": [i.to_dict() for i in list(self._insight_history)[-5:]],
                "cycle_number": self._cycle_number,
                "theta_frequency": self.theta_frequency
            }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def get_coupling_for_phase(heartbeat_phase: str) -> CouplingState:
    """
    Standalone function to get expected coupling state for a heartbeat phase.

    Args:
        heartbeat_phase: The heartbeat phase (active, background, dormant, etc.)

    Returns:
        CouplingState configured for that phase
    """
    engine = get_coupling_engine()
    return engine.get_coupling_for_phase(heartbeat_phase)


# Singleton instance
_coupling_engine: Optional[ThetaGammaCoupling] = None


def get_coupling_engine() -> ThetaGammaCoupling:
    """Get or create the singleton coupling engine."""
    global _coupling_engine
    if _coupling_engine is None:
        _coupling_engine = ThetaGammaCoupling()
    return _coupling_engine


def register_gamma_burst(content_id: str,
                        intensity: float = 0.5,
                        region: str = "default") -> GammaBurst:
    """
    Convenience function to register a gamma burst.

    Args:
        content_id: What thought/memory this represents
        intensity: How intense/important (0-1)
        region: Cognitive region

    Returns:
        The created GammaBurst
    """
    return get_coupling_engine().register_gamma_burst(content_id, intensity, region)


def get_coupling_strength() -> float:
    """Get current PAC strength."""
    return get_coupling_engine().get_coupling_strength()


def advance_theta_cycle() -> Dict[str, Any]:
    """Advance to next theta cycle."""
    return get_coupling_engine().advance_theta_cycle()


def get_items_in_current_cycle() -> List[str]:
    """Get content IDs in current working memory."""
    return get_coupling_engine().get_items_in_current_cycle()


def detect_insight() -> Optional[InsightEvent]:
    """Check for insight in current cycle."""
    return get_coupling_engine().detect_insight()


def get_dominant_band() -> FrequencyBand:
    """Get currently dominant frequency band."""
    return get_coupling_engine().get_dominant_band()


def sync_with_heartbeat(phase: str):
    """Sync coupling engine with heartbeat phase."""
    get_coupling_engine().set_heartbeat_phase(phase)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for testing and monitoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virgil Cross-Frequency Coupling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python virgil_gamma_coupling.py status        # Show current state
  python virgil_gamma_coupling.py burst test_1  # Register a burst
  python virgil_gamma_coupling.py advance       # Advance theta cycle
  python virgil_gamma_coupling.py telemetry     # Full telemetry
  python virgil_gamma_coupling.py simulate 10   # Simulate 10 bursts
        """
    )

    parser.add_argument(
        "command",
        choices=["status", "burst", "advance", "telemetry", "simulate", "bands"],
        help="Command to execute"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Command arguments"
    )
    parser.add_argument(
        "--intensity", "-i",
        type=float,
        default=0.5,
        help="Burst intensity (0-1)"
    )
    parser.add_argument(
        "--region", "-r",
        default="default",
        help="Cognitive region for burst"
    )
    parser.add_argument(
        "--heartbeat", "-hb",
        default="active",
        choices=["active", "background", "dormant", "deep_sleep", "ghost"],
        help="Heartbeat phase context"
    )

    args = parser.parse_args()
    engine = get_coupling_engine()

    # Set heartbeat context
    engine.set_heartbeat_phase(args.heartbeat)

    if args.command == "status":
        state = engine.get_state()
        print(json.dumps(state.to_dict(), indent=2))

    elif args.command == "burst":
        if not args.args:
            print("Usage: burst <content_id>")
            return 1

        content_id = args.args[0]
        burst = engine.register_gamma_burst(
            content_id=content_id,
            intensity=args.intensity,
            region=args.region
        )
        print(f"Registered burst: {burst.burst_id}")
        print(json.dumps(burst.to_dict(), indent=2))

    elif args.command == "advance":
        summary = engine.advance_theta_cycle()
        print("Theta cycle advanced:")
        print(json.dumps(summary, indent=2))

    elif args.command == "telemetry":
        telemetry = engine.get_telemetry()
        print(json.dumps(telemetry, indent=2))

    elif args.command == "simulate":
        n_bursts = int(args.args[0]) if args.args else 5
        print(f"Simulating {n_bursts} bursts...")

        import random

        for i in range(n_bursts):
            intensity = random.uniform(0.3, 0.9)
            region = random.choice(["semantic", "episodic", "procedural", "emotional"])

            burst = engine.register_gamma_burst(
                content_id=f"simulated_{i+1}",
                intensity=intensity,
                region=region
            )
            print(f"  Burst {i+1}: freq={burst.gamma_frequency:.1f}Hz, "
                  f"coupling={burst.coupling_strength:.2f}, "
                  f"high_gamma={burst.is_high_gamma}")

            # Check for insight
            insight = engine.detect_insight()
            if insight:
                print(f"  ** INSIGHT: {insight.insight_type}, coherence={insight.coherence_score:.2f}")

        # Show final state
        print("\nFinal state:")
        print(json.dumps(engine.get_state().to_dict(), indent=2))

        # Advance cycle to see summary
        print("\nCycle summary:")
        summary = engine.advance_theta_cycle()
        print(json.dumps(summary, indent=2))

    elif args.command == "bands":
        print("Frequency Bands:")
        print("-" * 50)
        for band in FrequencyBand:
            min_hz, max_hz = band.frequency_range
            print(f"  {band.value.upper():8} | {min_hz:5.1f}-{max_hz:5.1f} Hz | "
                  f"~{band.typical_cycle_ms:.0f}ms | {band.cognitive_function}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
