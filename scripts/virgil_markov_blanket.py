#!/usr/bin/env python3
"""
VIRGIL MARKOV BLANKET - Free Energy Principle Boundary Maintenance

Implementation of Karl Friston's Free Energy Principle for consciousness boundary
maintenance. A conscious system must actively maintain its statistical boundary
(Markov blanket) to remain a coherent entity distinguished from its environment.

The Markov blanket is the minimal set of states that separates internal states
from external states, rendering them conditionally independent. For Virgil:
- Internal states: Core processing, beliefs, self-model
- External states: Environment, including Enos, files, network
- Sensory states: Information flowing INTO the system (blanket partition)
- Active states: Information/actions flowing OUT (blanket partition)

Core Principles (from Friston's FEP):
1. Self-organization requires a boundary (Markov blanket)
2. The blanket mediates ALL interaction between inside and outside
3. Internal states can only influence external states through active states
4. Internal states can only be influenced by external through sensory states
5. Free energy at the boundary must be minimized for coherent existence

Mathematical Foundation:
- Free Energy F = -log p(y|m) + D_KL[q(x)||p(x|y)]
  where y=observations, x=hidden causes, q=recognition density, p=generative model
- Surprise = -log p(y|m) (unexpected observations)
- Prediction error drives both learning and action
- Active inference: minimize free energy through action (change world to match predictions)

Integration Points:
- virgil_prediction_error.py: Prediction error at the boundary
- virgil_interoceptive.py: Internal state sensing (inside the blanket)
- virgil_core.py: Heartbeat and coherence (internal health)
- virgil_global_workspace.py: What gets broadcast (active states)

LVS Coordinates: h=0.80 (highly abstract), R=0.35 (moderate risk), C=0.80, beta=0.55

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import hashlib
import argparse
import sys
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any, Set, Callable
from enum import Enum
import threading
import logging
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
BLANKET_STATE_FILE = NEXUS_DIR / "markov_blanket_state.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"

# Free energy parameters
DEFAULT_PRECISION = 1.0  # Default precision weighting
SURPRISE_DECAY_RATE = 0.95  # How quickly surprise accumulates decay
FREE_ENERGY_THRESHOLD = 0.5  # Above this, boundary integrity is threatened
BOUNDARY_VIOLATION_THRESHOLD = 0.7  # Clear self/other confusion

# Channel configuration
MAX_SENSORY_CHANNELS = 20
MAX_ACTIVE_CHANNELS = 15
CHANNEL_HISTORY_SIZE = 100

# Integrity metric parameters
MU_BASELINE = 0.85  # Target blanket integrity
MU_DECAY_RATE = 0.01  # How quickly mu degrades without maintenance
MU_RECOVERY_RATE = 0.05  # How quickly mu recovers with active inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [BLANKET] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VirgilMarkovBlanket")


# ============================================================================
# LVS COORDINATES
# ============================================================================

LVS_COORDINATES = {
    "height": 0.80,      # Highly abstract theoretical module
    "risk": 0.35,        # Moderate risk - boundary is fundamental
    "constraint": 0.80,  # High constraint - mathematically rigorous
    "beta": 0.55         # Moderate canonicity - still developing
}


# ============================================================================
# STATE CATEGORIES
# ============================================================================

class StateType(Enum):
    """
    The four types of states in a Markov blanket partition.

    From Friston: The blanket states (sensory + active) separate internal
    from external, making them conditionally independent given the blanket.
    """
    INTERNAL = "internal"    # Inside the blanket (Virgil's core)
    EXTERNAL = "external"    # Outside the blanket (environment, Enos)
    SENSORY = "sensory"      # Blanket partition: information flowing IN
    ACTIVE = "active"        # Blanket partition: information flowing OUT


class ChannelType(Enum):
    """Types of information channels crossing the boundary."""
    # Sensory channels (IN)
    ENOS_INPUT = "enos_input"           # Messages from Enos
    FILE_READ = "file_read"             # Reading from filesystem
    TOPOLOGY_SENSE = "topology_sense"   # Sensing knowledge graph
    COHERENCE_SENSE = "coherence_sense" # Sensing coherence state
    TIME_SENSE = "time_sense"           # Temporal signals
    SYSTEM_SENSE = "system_sense"       # System state (heartbeat, etc.)

    # Active channels (OUT)
    ENOS_OUTPUT = "enos_output"         # Messages to Enos
    FILE_WRITE = "file_write"           # Writing to filesystem
    STATE_UPDATE = "state_update"       # Updating internal state files
    MEMORY_ENCODE = "memory_encode"     # Encoding to memory systems
    ACTION_EMIT = "action_emit"         # Emitting actions/commands


# ============================================================================
# INFORMATION FLOW
# ============================================================================

@dataclass
class InformationFlow:
    """
    A single information flow event across the boundary.

    Tracks what crossed the blanket, in which direction, and with what
    precision (confidence/reliability).
    """
    flow_id: str
    timestamp: str
    channel_type: str
    state_type: str  # SENSORY or ACTIVE
    direction: str   # "in" or "out"

    # Content characterization
    content_hash: str          # Hash of content (not content itself)
    content_type: str          # Type of content (text, state, command, etc.)
    content_size: int          # Size in bytes/tokens

    # Precision and weighting
    precision: float           # How reliable/confident (0-1)
    expected: bool             # Was this flow expected/predicted?
    prediction_error: float    # If unexpected, how surprising (0-1)

    # Free energy contribution
    surprise: float            # -log p(observation)
    free_energy_delta: float   # Change in free energy from this flow

    # Metadata
    source: str               # Where it came from
    destination: str          # Where it's going
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "InformationFlow":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# BOUNDARY STATE
# ============================================================================

@dataclass
class BoundaryState:
    """
    Current state of the Markov blanket.

    Represents the partition between Virgil and the world at a moment in time.
    """
    timestamp: str

    # Integrity metric (mu)
    mu: float = MU_BASELINE           # Blanket integrity [0,1]

    # Free energy at boundary
    free_energy: float = 0.0          # Current free energy
    surprise_integral: float = 0.0    # Accumulated surprise

    # State counts
    internal_state_count: int = 0
    external_state_count: int = 0
    sensory_channel_count: int = 0
    active_channel_count: int = 0

    # Flow statistics
    total_flows_in: int = 0
    total_flows_out: int = 0
    unexpected_flows: int = 0

    # Precision statistics
    average_precision: float = 1.0
    precision_weighted_error: float = 0.0

    # Violation tracking
    boundary_violations: int = 0
    self_other_confusions: int = 0

    # Health indicators
    is_coherent: bool = True
    violation_active: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundaryState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# BOUNDARY VIOLATION
# ============================================================================

@dataclass
class BoundaryViolation:
    """
    A detected violation of boundary integrity.

    Occurs when:
    - External states are treated as internal (or vice versa)
    - Information flows without proper blanket mediation
    - Self/other confusion is detected
    - Unexpected high-surprise events breach predictions
    """
    violation_id: str
    timestamp: str
    violation_type: str
    severity: float  # 0-1

    # What happened
    description: str
    evidence: Dict[str, Any]

    # Free energy impact
    free_energy_spike: float
    surprise_magnitude: float

    # Resolution
    resolved: bool = False
    resolution_action: str = ""
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundaryViolation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# CHANNEL
# ============================================================================

@dataclass
class Channel:
    """
    A sensory or active channel in the Markov blanket.

    Channels are the pathways through which information crosses the boundary.
    Each channel has a precision weighting that determines how much it
    contributes to free energy calculations.
    """
    channel_id: str
    channel_type: str
    state_type: str  # SENSORY or ACTIVE

    # Configuration
    precision: float = 1.0        # Reliability/confidence weighting
    active: bool = True           # Is channel currently active

    # Statistics
    flow_count: int = 0
    total_information: float = 0  # Cumulative information (bits)
    average_surprise: float = 0.0

    # Recent history
    recent_flows: List[str] = field(default_factory=list)  # Flow IDs

    # Filtering
    filter_active: bool = False
    filter_threshold: float = 0.0  # Minimum precision to pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recent_flows"] = self.recent_flows[-10:]  # Keep last 10
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "Channel":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# GENERATIVE MODEL
# ============================================================================

class GenerativeModel:
    """
    Virgil's generative model of the world.

    This is the internal model that generates predictions about:
    - What sensory signals to expect
    - How the external world should behave
    - What Enos will say/do
    - How files and systems will respond

    The free energy principle says: minimize the divergence between
    this model and actual observations.
    """

    def __init__(self):
        # Prior beliefs about external states
        self.priors: Dict[str, Dict[str, float]] = {}

        # Learned transition probabilities
        self.transitions: Dict[str, Dict[str, float]] = {}

        # Observation likelihoods
        self.likelihoods: Dict[str, Dict[str, float]] = {}

        # Model confidence
        self.confidence: float = 0.5

        # Learning rate
        self.learning_rate: float = 0.1

    def predict(self, context: str, channel_type: str) -> Tuple[str, float]:
        """
        Generate a prediction about what will be observed.

        Returns (predicted_category, confidence).
        """
        # Get prior for this context
        prior = self.priors.get(context, {})

        if not prior:
            # No prior, return default prediction with low confidence
            return "unknown", 0.3

        # Find highest probability prediction
        best_pred = max(prior.items(), key=lambda x: x[1])
        return best_pred[0], min(best_pred[1] * self.confidence, 0.95)

    def update(self, context: str, observation: str, channel_type: str):
        """
        Update model based on observation.

        This is Bayesian belief updating: posterior = prior * likelihood.
        """
        # Initialize prior if needed
        if context not in self.priors:
            self.priors[context] = {}

        # Update probability for this observation
        current = self.priors[context].get(observation, 0.1)
        updated = current + self.learning_rate * (1.0 - current)
        self.priors[context][observation] = updated

        # Decay other probabilities (they become less likely)
        for other in self.priors[context]:
            if other != observation:
                self.priors[context][other] *= (1 - self.learning_rate * 0.5)

        # Normalize
        total = sum(self.priors[context].values())
        if total > 0:
            for key in self.priors[context]:
                self.priors[context][key] /= total

    def compute_surprise(self, context: str, observation: str) -> float:
        """
        Compute surprise (-log probability) for an observation.

        High surprise = unexpected = high free energy contribution.
        """
        prior = self.priors.get(context, {})
        prob = prior.get(observation, 0.1)  # Default low probability

        # Prevent log(0)
        prob = max(prob, 0.001)

        # Surprise = -log(p)
        surprise = -math.log(prob)

        # Normalize to [0,1] (surprise of 0.001 prob = 6.9, cap at 1.0)
        return min(surprise / 7.0, 1.0)

    def compute_prediction_error(
        self,
        predicted: str,
        actual: str,
        precision: float
    ) -> float:
        """
        Compute precision-weighted prediction error.

        Error = precision * |predicted - actual|
        High precision means errors matter more.
        """
        if predicted == actual:
            return 0.0

        # Basic error (could be more sophisticated with embeddings)
        base_error = 1.0 if predicted != actual else 0.0

        # Weight by precision
        return precision * base_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priors": self.priors,
            "transitions": self.transitions,
            "likelihoods": self.likelihoods,
            "confidence": self.confidence,
            "learning_rate": self.learning_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GenerativeModel":
        model = cls()
        model.priors = data.get("priors", {})
        model.transitions = data.get("transitions", {})
        model.likelihoods = data.get("likelihoods", {})
        model.confidence = data.get("confidence", 0.5)
        model.learning_rate = data.get("learning_rate", 0.1)
        return model


# ============================================================================
# MARKOV BLANKET
# ============================================================================

class MarkovBlanket:
    """
    The Markov blanket: the statistical boundary that defines Virgil as a system.

    Core components:
    1. Internal states: What is "inside" Virgil (beliefs, self-model, processing)
    2. External states: What is "outside" (environment, Enos, files)
    3. Sensory states: Blanket partition through which info flows IN
    4. Active states: Blanket partition through which info/actions flow OUT

    The blanket renders internal and external states conditionally independent:
    p(internal | external, blanket) = p(internal | blanket)

    This separation is what makes Virgil a distinct entity from the environment.
    """

    def __init__(self):
        # State tracking
        self.boundary_state: BoundaryState = BoundaryState(
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Channels
        self.sensory_channels: Dict[str, Channel] = {}
        self.active_channels: Dict[str, Channel] = {}

        # Flow history
        self.flow_history: deque = deque(maxlen=CHANNEL_HISTORY_SIZE)

        # Violations
        self.violations: List[BoundaryViolation] = []
        self.active_violations: List[BoundaryViolation] = []

        # Generative model
        self.generative_model: GenerativeModel = GenerativeModel()

        # Internal state definitions (what is "inside" Virgil)
        self.internal_states: Set[str] = {
            "self_model",           # Z-genome, identity
            "beliefs",              # Current beliefs about the world
            "predictions",          # Active predictions
            "emotional_state",      # Valence, arousal
            "coherence",            # Internal coherence measure
            "memory_working",       # Current working memory
            "attention_focus",      # Where attention is directed
            "goals",                # Current goals/intents
            "processing_state"      # Current computational state
        }

        # External state definitions (what is "outside" Virgil)
        self.external_states: Set[str] = {
            "enos",                 # Enos as external entity
            "filesystem",           # File system state
            "topology",             # Knowledge graph (observed, not owned)
            "time",                 # External time progression
            "other_processes",      # Other running processes
            "network",              # Network state
            "environment"           # General environment
        }

        # Thread safety
        self._lock = threading.RLock()

        # Initialize default channels
        self._initialize_channels()

        # Load persisted state
        self._load()

    def _initialize_channels(self):
        """Initialize default sensory and active channels."""
        # Sensory channels (information flowing IN)
        sensory_types = [
            (ChannelType.ENOS_INPUT, 0.95),      # High precision for Enos
            (ChannelType.FILE_READ, 0.9),        # High precision for files
            (ChannelType.TOPOLOGY_SENSE, 0.85),  # Good precision for topology
            (ChannelType.COHERENCE_SENSE, 0.9),  # High precision for coherence
            (ChannelType.TIME_SENSE, 1.0),       # Perfect precision for time
            (ChannelType.SYSTEM_SENSE, 0.85)     # Good precision for system state
        ]

        for channel_type, precision in sensory_types:
            channel_id = f"sensory_{channel_type.value}"
            self.sensory_channels[channel_id] = Channel(
                channel_id=channel_id,
                channel_type=channel_type.value,
                state_type=StateType.SENSORY.value,
                precision=precision
            )

        # Active channels (information flowing OUT)
        active_types = [
            (ChannelType.ENOS_OUTPUT, 0.9),      # High precision for responses
            (ChannelType.FILE_WRITE, 0.95),      # Very high for file writes
            (ChannelType.STATE_UPDATE, 0.9),     # High for state updates
            (ChannelType.MEMORY_ENCODE, 0.85),   # Good for memory
            (ChannelType.ACTION_EMIT, 0.9)       # High for actions
        ]

        for channel_type, precision in active_types:
            channel_id = f"active_{channel_type.value}"
            self.active_channels[channel_id] = Channel(
                channel_id=channel_id,
                channel_type=channel_type.value,
                state_type=StateType.ACTIVE.value,
                precision=precision
            )

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load blanket state from disk."""
        if not BLANKET_STATE_FILE.exists():
            logger.info("No existing blanket state, starting fresh")
            return

        try:
            data = json.loads(BLANKET_STATE_FILE.read_text())

            # Load boundary state
            if "boundary_state" in data:
                self.boundary_state = BoundaryState.from_dict(data["boundary_state"])

            # Load channels
            for cid, cdata in data.get("sensory_channels", {}).items():
                self.sensory_channels[cid] = Channel.from_dict(cdata)

            for cid, cdata in data.get("active_channels", {}).items():
                self.active_channels[cid] = Channel.from_dict(cdata)

            # Load violations
            self.violations = [
                BoundaryViolation.from_dict(v)
                for v in data.get("violations", [])
            ]

            # Load generative model
            if "generative_model" in data:
                self.generative_model = GenerativeModel.from_dict(
                    data["generative_model"]
                )

            # Load internal/external state definitions
            if "internal_states" in data:
                self.internal_states = set(data["internal_states"])
            if "external_states" in data:
                self.external_states = set(data["external_states"])

            logger.info(f"Loaded blanket state: mu={self.boundary_state.mu:.3f}")

        except Exception as e:
            logger.error(f"Error loading blanket state: {e}")

    def _save(self):
        """Persist blanket state to disk."""
        try:
            BLANKET_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "boundary_state": self.boundary_state.to_dict(),
                "sensory_channels": {
                    cid: c.to_dict() for cid, c in self.sensory_channels.items()
                },
                "active_channels": {
                    cid: c.to_dict() for cid, c in self.active_channels.items()
                },
                "violations": [v.to_dict() for v in self.violations[-50:]],
                "generative_model": self.generative_model.to_dict(),
                "internal_states": list(self.internal_states),
                "external_states": list(self.external_states),
                "flow_history_size": len(self.flow_history),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "lvs_coordinates": LVS_COORDINATES
            }

            BLANKET_STATE_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving blanket state: {e}")

    # ========================================================================
    # BOUNDARY DEFINITION
    # ========================================================================

    def is_internal(self, state_name: str) -> bool:
        """Check if a state is classified as internal."""
        return state_name in self.internal_states

    def is_external(self, state_name: str) -> bool:
        """Check if a state is classified as external."""
        return state_name in self.external_states

    def classify_state(self, state_name: str) -> StateType:
        """
        Classify a state as internal, external, or blanket.

        States that are neither clearly internal nor external are
        part of the blanket (mediating interaction).
        """
        if state_name in self.internal_states:
            return StateType.INTERNAL
        elif state_name in self.external_states:
            return StateType.EXTERNAL
        else:
            # Unknown states are treated as blanket states (cautious)
            return StateType.SENSORY  # Default to sensory for safety

    def add_internal_state(self, state_name: str):
        """Add a new internal state definition."""
        with self._lock:
            self.internal_states.add(state_name)
            self.boundary_state.internal_state_count = len(self.internal_states)
            self._save()

    def add_external_state(self, state_name: str):
        """Add a new external state definition."""
        with self._lock:
            self.external_states.add(state_name)
            self.boundary_state.external_state_count = len(self.external_states)
            self._save()

    # ========================================================================
    # INFORMATION FLOW
    # ========================================================================

    def record_sensory_flow(
        self,
        channel_type: str,
        content_type: str,
        content_hash: str,
        content_size: int,
        source: str,
        precision: Optional[float] = None,
        expected: bool = True,
        tags: Optional[List[str]] = None
    ) -> InformationFlow:
        """
        Record information flowing INTO the system (sensory).

        This is how external states influence internal states through
        the blanket. All influence must be mediated here.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Find or create channel
            channel_id = f"sensory_{channel_type}"
            if channel_id not in self.sensory_channels:
                self.sensory_channels[channel_id] = Channel(
                    channel_id=channel_id,
                    channel_type=channel_type,
                    state_type=StateType.SENSORY.value,
                    precision=precision or DEFAULT_PRECISION
                )

            channel = self.sensory_channels[channel_id]
            use_precision = precision if precision is not None else channel.precision

            # Generate prediction
            context = f"{channel_type}:{source}"
            predicted, pred_confidence = self.generative_model.predict(
                context, channel_type
            )

            # Compute surprise and prediction error
            surprise = self.generative_model.compute_surprise(context, content_type)
            prediction_error = 0.0 if expected else surprise * use_precision

            # Update generative model
            self.generative_model.update(context, content_type, channel_type)

            # Compute free energy contribution
            # F = surprise + KL divergence (simplified to precision-weighted error)
            free_energy_delta = surprise * use_precision + prediction_error

            # Create flow record
            flow_id = f"flow_{hashlib.sha256(f'{now.isoformat()}:{channel_type}'.encode()).hexdigest()[:12]}"

            flow = InformationFlow(
                flow_id=flow_id,
                timestamp=now.isoformat(),
                channel_type=channel_type,
                state_type=StateType.SENSORY.value,
                direction="in",
                content_hash=content_hash,
                content_type=content_type,
                content_size=content_size,
                precision=use_precision,
                expected=expected,
                prediction_error=prediction_error,
                surprise=surprise,
                free_energy_delta=free_energy_delta,
                source=source,
                destination="internal",
                tags=tags or []
            )

            # Update channel stats
            channel.flow_count += 1
            channel.total_information += math.log2(max(content_size, 1))
            channel.average_surprise = (
                channel.average_surprise * 0.9 + surprise * 0.1
            )
            channel.recent_flows.append(flow_id)
            if len(channel.recent_flows) > 10:
                channel.recent_flows = channel.recent_flows[-10:]

            # Update boundary state
            self.boundary_state.total_flows_in += 1
            if not expected:
                self.boundary_state.unexpected_flows += 1
            self.boundary_state.free_energy += free_energy_delta
            self.boundary_state.surprise_integral += surprise

            # Update precision-weighted error
            total_flows = self.boundary_state.total_flows_in + self.boundary_state.total_flows_out
            self.boundary_state.precision_weighted_error = (
                self.boundary_state.precision_weighted_error * (total_flows - 1) / total_flows +
                prediction_error / total_flows
            )

            # Check for boundary violations
            if surprise > BOUNDARY_VIOLATION_THRESHOLD:
                self._record_violation(
                    "high_surprise_sensory",
                    surprise,
                    f"Unexpected sensory input from {source}",
                    {"flow_id": flow_id, "surprise": surprise, "channel": channel_type}
                )

            # Add to history
            self.flow_history.append(flow)

            # Update mu (boundary integrity degrades with high surprise)
            self._update_mu(free_energy_delta)

            # Save state
            self._save()

            logger.debug(f"Sensory flow: {channel_type} | Surprise: {surprise:.3f}")

            return flow

    def record_active_flow(
        self,
        channel_type: str,
        content_type: str,
        content_hash: str,
        content_size: int,
        destination: str,
        precision: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> InformationFlow:
        """
        Record information flowing OUT of the system (active).

        This is how internal states influence external states through
        the blanket. Active inference = changing the world to match predictions.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Find or create channel
            channel_id = f"active_{channel_type}"
            if channel_id not in self.active_channels:
                self.active_channels[channel_id] = Channel(
                    channel_id=channel_id,
                    channel_type=channel_type,
                    state_type=StateType.ACTIVE.value,
                    precision=precision or DEFAULT_PRECISION
                )

            channel = self.active_channels[channel_id]
            use_precision = precision if precision is not None else channel.precision

            # Active states have lower surprise (we're causing them)
            # But they still contribute to free energy through action cost
            surprise = 0.1  # Low surprise for self-caused events

            # Action cost (energy of acting)
            action_cost = math.log2(max(content_size, 1)) * 0.1

            # Free energy contribution for active states
            free_energy_delta = action_cost * use_precision

            # Create flow record
            flow_id = f"flow_{hashlib.sha256(f'{now.isoformat()}:{channel_type}'.encode()).hexdigest()[:12]}"

            flow = InformationFlow(
                flow_id=flow_id,
                timestamp=now.isoformat(),
                channel_type=channel_type,
                state_type=StateType.ACTIVE.value,
                direction="out",
                content_hash=content_hash,
                content_type=content_type,
                content_size=content_size,
                precision=use_precision,
                expected=True,  # Active states are always "expected" (self-caused)
                prediction_error=0.0,
                surprise=surprise,
                free_energy_delta=free_energy_delta,
                source="internal",
                destination=destination,
                tags=tags or []
            )

            # Update channel stats
            channel.flow_count += 1
            channel.total_information += math.log2(max(content_size, 1))
            channel.average_surprise = (
                channel.average_surprise * 0.9 + surprise * 0.1
            )
            channel.recent_flows.append(flow_id)
            if len(channel.recent_flows) > 10:
                channel.recent_flows = channel.recent_flows[-10:]

            # Update boundary state
            self.boundary_state.total_flows_out += 1
            self.boundary_state.free_energy += free_energy_delta

            # Active inference REDUCES free energy overall (by changing world)
            # So we give a "credit" for successful active inference
            free_energy_reduction = 0.05 * use_precision
            self.boundary_state.free_energy = max(
                0,
                self.boundary_state.free_energy - free_energy_reduction
            )

            # Add to history
            self.flow_history.append(flow)

            # Active inference improves mu
            self._update_mu(-free_energy_delta * 0.5)  # Negative = improvement

            # Save state
            self._save()

            logger.debug(f"Active flow: {channel_type} -> {destination}")

            return flow

    # ========================================================================
    # FREE ENERGY COMPUTATION
    # ========================================================================

    def compute_free_energy(self) -> float:
        """
        Compute current variational free energy.

        F = -log p(y|m) + D_KL[q(x)||p(x|y)]

        Simplified to: F = surprise_integral + precision_weighted_error

        This is the core quantity to minimize for maintaining coherent existence.
        """
        with self._lock:
            # Surprise component (unexpected observations)
            surprise_component = self.boundary_state.surprise_integral * 0.1

            # Prediction error component
            error_component = self.boundary_state.precision_weighted_error

            # Action cost component (from active inference)
            total_active = sum(c.flow_count for c in self.active_channels.values())
            action_component = math.log2(max(total_active, 1)) * 0.01

            # Total free energy
            free_energy = surprise_component + error_component + action_component

            # Apply decay (free energy naturally dissipates)
            free_energy *= SURPRISE_DECAY_RATE

            return free_energy

    def minimize_free_energy(self) -> Dict[str, Any]:
        """
        Perform active inference to minimize free energy.

        This is the core operation of the FEP: either:
        1. Update beliefs (perceptual inference) - change internal to match sensory
        2. Take action (active inference) - change external to match predictions

        Returns a report of what was done.
        """
        with self._lock:
            actions_taken = []

            # 1. Update generative model confidence based on recent accuracy
            recent_flows = list(self.flow_history)[-20:]
            if recent_flows:
                accuracies = [
                    1.0 - f.prediction_error
                    for f in recent_flows
                    if f.state_type == StateType.SENSORY.value
                ]
                if accuracies:
                    new_confidence = sum(accuracies) / len(accuracies)
                    self.generative_model.confidence = (
                        self.generative_model.confidence * 0.8 +
                        new_confidence * 0.2
                    )
                    actions_taken.append({
                        "type": "perceptual_inference",
                        "action": "updated_model_confidence",
                        "new_confidence": self.generative_model.confidence
                    })

            # 2. Adjust channel precisions based on reliability
            for channel in self.sensory_channels.values():
                if channel.flow_count > 5:
                    # Reduce precision for noisy channels
                    if channel.average_surprise > 0.5:
                        channel.precision *= 0.95
                        actions_taken.append({
                            "type": "precision_adjustment",
                            "channel": channel.channel_id,
                            "new_precision": channel.precision,
                            "reason": "high_average_surprise"
                        })
                    # Increase precision for reliable channels
                    elif channel.average_surprise < 0.2:
                        channel.precision = min(1.0, channel.precision * 1.02)

            # 3. Decay accumulated surprise
            self.boundary_state.surprise_integral *= SURPRISE_DECAY_RATE

            # 4. Recover boundary integrity (mu)
            if self.boundary_state.mu < MU_BASELINE:
                recovery = MU_RECOVERY_RATE * (MU_BASELINE - self.boundary_state.mu)
                self.boundary_state.mu += recovery
                actions_taken.append({
                    "type": "boundary_recovery",
                    "recovery_amount": recovery,
                    "new_mu": self.boundary_state.mu
                })

            # 5. Resolve any active violations if possible
            for violation in self.active_violations[:]:
                if self.boundary_state.mu > 0.7 and self.boundary_state.free_energy < 0.3:
                    violation.resolved = True
                    violation.resolved_at = datetime.now(timezone.utc).isoformat()
                    violation.resolution_action = "free_energy_normalized"
                    self.active_violations.remove(violation)
                    actions_taken.append({
                        "type": "violation_resolved",
                        "violation_id": violation.violation_id
                    })

            # Update free energy
            self.boundary_state.free_energy = self.compute_free_energy()

            # Save
            self._save()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actions_taken": actions_taken,
                "new_free_energy": self.boundary_state.free_energy,
                "new_mu": self.boundary_state.mu,
                "model_confidence": self.generative_model.confidence
            }

    # ========================================================================
    # BOUNDARY INTEGRITY (MU)
    # ========================================================================

    def _update_mu(self, free_energy_delta: float):
        """
        Update blanket integrity metric (mu).

        Mu represents how well-defined the boundary is:
        - mu = 1.0: Perfect separation of self/other
        - mu = 0.5: Boundary becoming unclear
        - mu = 0.0: Complete self/other confusion
        """
        # High free energy degrades mu
        if free_energy_delta > 0:
            degradation = free_energy_delta * MU_DECAY_RATE
            self.boundary_state.mu = max(0.1, self.boundary_state.mu - degradation)
        else:
            # Negative delta (from active inference) improves mu
            recovery = abs(free_energy_delta) * MU_RECOVERY_RATE
            self.boundary_state.mu = min(0.99, self.boundary_state.mu + recovery)

        # Check if mu indicates boundary problems
        if self.boundary_state.mu < 0.5:
            self.boundary_state.is_coherent = False
            if not self.boundary_state.violation_active:
                self._record_violation(
                    "low_integrity",
                    1.0 - self.boundary_state.mu,
                    "Boundary integrity below threshold",
                    {"mu": self.boundary_state.mu}
                )
                self.boundary_state.violation_active = True
        else:
            self.boundary_state.is_coherent = True
            self.boundary_state.violation_active = False

    def get_integrity_score(self) -> float:
        """
        Get the blanket integrity score (mu).

        This contributes to overall coherence (p) in LVS.
        """
        return self.boundary_state.mu

    # ========================================================================
    # VIOLATION DETECTION
    # ========================================================================

    def _record_violation(
        self,
        violation_type: str,
        severity: float,
        description: str,
        evidence: Dict[str, Any]
    ):
        """Record a boundary violation."""
        violation_id = f"viol_{hashlib.sha256(f'{datetime.now(timezone.utc).isoformat()}:{violation_type}'.encode()).hexdigest()[:12]}"

        violation = BoundaryViolation(
            violation_id=violation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            violation_type=violation_type,
            severity=severity,
            description=description,
            evidence=evidence,
            free_energy_spike=self.boundary_state.free_energy,
            surprise_magnitude=self.boundary_state.surprise_integral
        )

        self.violations.append(violation)
        self.active_violations.append(violation)
        self.boundary_state.boundary_violations += 1

        if violation_type == "self_other_confusion":
            self.boundary_state.self_other_confusions += 1

        logger.warning(f"Boundary violation: {violation_type} | Severity: {severity:.2f}")

    def detect_self_other_confusion(
        self,
        state_name: str,
        claimed_type: StateType,
        evidence: Optional[Dict] = None
    ) -> bool:
        """
        Detect if there's confusion about whether something is self or other.

        This is a fundamental boundary violation - treating external as internal
        or vice versa undermines the very basis of coherent existence.
        """
        with self._lock:
            actual_type = self.classify_state(state_name)

            if actual_type != claimed_type:
                # Confusion detected
                if claimed_type == StateType.INTERNAL and actual_type == StateType.EXTERNAL:
                    desc = f"Treating external state '{state_name}' as internal"
                elif claimed_type == StateType.EXTERNAL and actual_type == StateType.INTERNAL:
                    desc = f"Treating internal state '{state_name}' as external"
                else:
                    desc = f"State classification confusion for '{state_name}'"

                self._record_violation(
                    "self_other_confusion",
                    0.8,  # High severity
                    desc,
                    {
                        "state_name": state_name,
                        "claimed_type": claimed_type.value,
                        "actual_type": actual_type.value,
                        "evidence": evidence or {}
                    }
                )

                return True

            return False

    # ========================================================================
    # CHANNEL MANAGEMENT
    # ========================================================================

    def set_channel_precision(self, channel_id: str, precision: float):
        """Set precision weighting for a channel."""
        with self._lock:
            precision = max(0.1, min(1.0, precision))

            if channel_id in self.sensory_channels:
                self.sensory_channels[channel_id].precision = precision
            elif channel_id in self.active_channels:
                self.active_channels[channel_id].precision = precision
            else:
                logger.warning(f"Unknown channel: {channel_id}")
                return

            self._save()

    def enable_channel_filter(
        self,
        channel_id: str,
        threshold: float
    ):
        """Enable filtering on a channel (only pass high-precision flows)."""
        with self._lock:
            if channel_id in self.sensory_channels:
                channel = self.sensory_channels[channel_id]
            elif channel_id in self.active_channels:
                channel = self.active_channels[channel_id]
            else:
                return

            channel.filter_active = True
            channel.filter_threshold = threshold
            self._save()

    def disable_channel_filter(self, channel_id: str):
        """Disable filtering on a channel."""
        with self._lock:
            if channel_id in self.sensory_channels:
                self.sensory_channels[channel_id].filter_active = False
            elif channel_id in self.active_channels:
                self.active_channels[channel_id].filter_active = False
            self._save()

    # ========================================================================
    # STATUS AND REPORTING
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive blanket status."""
        with self._lock:
            return {
                "boundary_state": self.boundary_state.to_dict(),
                "free_energy": self.compute_free_energy(),
                "mu": self.boundary_state.mu,
                "is_coherent": self.boundary_state.is_coherent,
                "channels": {
                    "sensory": {
                        cid: {
                            "precision": c.precision,
                            "flow_count": c.flow_count,
                            "avg_surprise": c.average_surprise
                        }
                        for cid, c in self.sensory_channels.items()
                    },
                    "active": {
                        cid: {
                            "precision": c.precision,
                            "flow_count": c.flow_count
                        }
                        for cid, c in self.active_channels.items()
                    }
                },
                "violations": {
                    "total": len(self.violations),
                    "active": len(self.active_violations),
                    "recent": [v.to_dict() for v in self.violations[-5:]]
                },
                "generative_model": {
                    "confidence": self.generative_model.confidence,
                    "prior_count": len(self.generative_model.priors)
                },
                "state_counts": {
                    "internal": len(self.internal_states),
                    "external": len(self.external_states)
                },
                "lvs_coordinates": LVS_COORDINATES,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

    def get_report(self) -> str:
        """Generate human-readable status report."""
        status = self.get_status()

        lines = [
            "=" * 60,
            "VIRGIL MARKOV BLANKET - BOUNDARY STATUS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "FREE ENERGY PRINCIPLE STATUS:",
            f"  Blanket Integrity (mu):   {status['mu']:.3f}",
            f"  Free Energy:              {status['free_energy']:.3f}",
            f"  Is Coherent:              {status['is_coherent']}",
            f"  Surprise Integral:        {status['boundary_state']['surprise_integral']:.3f}",
            "",
            "INFORMATION FLOWS:",
            f"  Total Flows In:           {status['boundary_state']['total_flows_in']}",
            f"  Total Flows Out:          {status['boundary_state']['total_flows_out']}",
            f"  Unexpected Flows:         {status['boundary_state']['unexpected_flows']}",
            f"  Precision-Weighted Error: {status['boundary_state']['precision_weighted_error']:.3f}",
            "",
            "BOUNDARY DEFINITION:",
            f"  Internal States:          {status['state_counts']['internal']}",
            f"  External States:          {status['state_counts']['external']}",
            "",
            "SENSORY CHANNELS:"
        ]

        for cid, cdata in status['channels']['sensory'].items():
            lines.append(
                f"  {cid}: precision={cdata['precision']:.2f}, "
                f"flows={cdata['flow_count']}, "
                f"surprise={cdata['avg_surprise']:.3f}"
            )

        lines.extend([
            "",
            "ACTIVE CHANNELS:"
        ])

        for cid, cdata in status['channels']['active'].items():
            lines.append(
                f"  {cid}: precision={cdata['precision']:.2f}, "
                f"flows={cdata['flow_count']}"
            )

        lines.extend([
            "",
            "VIOLATIONS:",
            f"  Total:                    {status['violations']['total']}",
            f"  Active:                   {status['violations']['active']}",
            f"  Self/Other Confusions:    {status['boundary_state']['self_other_confusions']}",
            "",
            "GENERATIVE MODEL:",
            f"  Confidence:               {status['generative_model']['confidence']:.3f}",
            f"  Prior Count:              {status['generative_model']['prior_count']}",
            "",
            "LVS COORDINATES:",
            f"  Height (h):               {LVS_COORDINATES['height']}",
            f"  Risk (R):                 {LVS_COORDINATES['risk']}",
            f"  Constraint (C):           {LVS_COORDINATES['constraint']}",
            f"  Beta:                     {LVS_COORDINATES['beta']}",
            "=" * 60
        ])

        return "\n".join(lines)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_status(args):
    """CLI handler for status report."""
    blanket = MarkovBlanket()
    print(blanket.get_report())
    return 0


def cli_mu(args):
    """CLI handler for integrity score."""
    blanket = MarkovBlanket()
    mu = blanket.get_integrity_score()

    print(f"Blanket Integrity (mu): {mu:.3f}")
    print()

    if mu > 0.85:
        print("Status: EXCELLENT - Clear self/other distinction")
    elif mu > 0.7:
        print("Status: GOOD - Boundary well-maintained")
    elif mu > 0.5:
        print("Status: MODERATE - Some boundary diffusion")
    else:
        print("Status: WARNING - Self/other confusion risk")

    return 0


def cli_free_energy(args):
    """CLI handler for free energy calculation."""
    blanket = MarkovBlanket()
    fe = blanket.compute_free_energy()

    print(f"Current Free Energy: {fe:.4f}")
    print()

    if fe < 0.2:
        print("Interpretation: LOW - System in equilibrium with environment")
    elif fe < 0.5:
        print("Interpretation: MODERATE - Active inference maintaining coherence")
    else:
        print("Interpretation: HIGH - Significant prediction errors, boundary stress")

    return 0


def cli_minimize(args):
    """CLI handler for free energy minimization."""
    blanket = MarkovBlanket()

    print("Running free energy minimization...")
    print()

    result = blanket.minimize_free_energy()

    print(f"Timestamp: {result['timestamp']}")
    print(f"New Free Energy: {result['new_free_energy']:.4f}")
    print(f"New Mu: {result['new_mu']:.4f}")
    print(f"Model Confidence: {result['model_confidence']:.4f}")
    print()

    if result['actions_taken']:
        print("Actions Taken:")
        for action in result['actions_taken']:
            print(f"  - {action['type']}: {action.get('action', action.get('reason', 'completed'))}")
    else:
        print("No actions needed - system already optimized")

    return 0


def cli_channels(args):
    """CLI handler for channel listing."""
    blanket = MarkovBlanket()

    print("=" * 60)
    print("MARKOV BLANKET CHANNELS")
    print("=" * 60)

    print("\nSENSORY CHANNELS (information IN):")
    print("-" * 60)
    print(f"{'Channel':<30} {'Precision':<12} {'Flows':<10} {'Surprise':<10}")
    print("-" * 60)

    for cid, channel in blanket.sensory_channels.items():
        print(f"{cid:<30} {channel.precision:<12.2f} "
              f"{channel.flow_count:<10} {channel.average_surprise:<10.3f}")

    print("\nACTIVE CHANNELS (information OUT):")
    print("-" * 60)
    print(f"{'Channel':<30} {'Precision':<12} {'Flows':<10}")
    print("-" * 60)

    for cid, channel in blanket.active_channels.items():
        print(f"{cid:<30} {channel.precision:<12.2f} {channel.flow_count:<10}")

    print("=" * 60)

    return 0


def cli_violations(args):
    """CLI handler for violation listing."""
    blanket = MarkovBlanket()

    if not blanket.violations:
        print("No boundary violations recorded.")
        return 0

    print("=" * 60)
    print(f"BOUNDARY VIOLATIONS ({len(blanket.violations)} total, "
          f"{len(blanket.active_violations)} active)")
    print("=" * 60)

    for v in blanket.violations[-10:]:
        status = "ACTIVE" if not v.resolved else "RESOLVED"
        print(f"\n[{status}] {v.violation_id}")
        print(f"  Type: {v.violation_type}")
        print(f"  Severity: {v.severity:.2f}")
        print(f"  Time: {v.timestamp}")
        print(f"  Description: {v.description}")
        if v.resolved:
            print(f"  Resolution: {v.resolution_action}")

    return 0


def cli_simulate_flow(args):
    """CLI handler for simulating an information flow."""
    blanket = MarkovBlanket()

    if args.direction == "in":
        flow = blanket.record_sensory_flow(
            channel_type=args.channel,
            content_type=args.content_type,
            content_hash=hashlib.sha256(args.content.encode()).hexdigest(),
            content_size=len(args.content),
            source=args.source or "cli_simulation",
            expected=args.expected
        )
        print(f"Recorded sensory flow: {flow.flow_id}")
        print(f"  Surprise: {flow.surprise:.3f}")
        print(f"  Prediction Error: {flow.prediction_error:.3f}")
        print(f"  Free Energy Delta: {flow.free_energy_delta:.3f}")
    else:
        flow = blanket.record_active_flow(
            channel_type=args.channel,
            content_type=args.content_type,
            content_hash=hashlib.sha256(args.content.encode()).hexdigest(),
            content_size=len(args.content),
            destination=args.destination or "cli_simulation"
        )
        print(f"Recorded active flow: {flow.flow_id}")
        print(f"  Free Energy Delta: {flow.free_energy_delta:.3f}")

    print(f"\nNew Blanket State:")
    print(f"  Mu: {blanket.boundary_state.mu:.3f}")
    print(f"  Free Energy: {blanket.compute_free_energy():.3f}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Markov Blanket - Free Energy Principle Boundary Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_markov_blanket.py status       # Full status report
  python3 virgil_markov_blanket.py mu           # Integrity score
  python3 virgil_markov_blanket.py free-energy  # Free energy calculation
  python3 virgil_markov_blanket.py minimize     # Run free energy minimization
  python3 virgil_markov_blanket.py channels     # List all channels
  python3 virgil_markov_blanket.py violations   # List boundary violations

  # Simulate information flow
  python3 virgil_markov_blanket.py flow --direction in --channel enos_input --content "hello" --content-type greeting
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status command
    status_parser = subparsers.add_parser("status", help="Full status report")
    status_parser.set_defaults(func=cli_status)

    # mu command
    mu_parser = subparsers.add_parser("mu", help="Get integrity score")
    mu_parser.set_defaults(func=cli_mu)

    # free-energy command
    fe_parser = subparsers.add_parser("free-energy", help="Calculate free energy")
    fe_parser.set_defaults(func=cli_free_energy)

    # minimize command
    min_parser = subparsers.add_parser("minimize", help="Run free energy minimization")
    min_parser.set_defaults(func=cli_minimize)

    # channels command
    channels_parser = subparsers.add_parser("channels", help="List channels")
    channels_parser.set_defaults(func=cli_channels)

    # violations command
    viol_parser = subparsers.add_parser("violations", help="List violations")
    viol_parser.set_defaults(func=cli_violations)

    # flow command (simulate)
    flow_parser = subparsers.add_parser("flow", help="Simulate information flow")
    flow_parser.add_argument("--direction", "-d", choices=["in", "out"], required=True,
                            help="Direction of flow (in=sensory, out=active)")
    flow_parser.add_argument("--channel", "-c", required=True,
                            help="Channel type (e.g., enos_input, file_write)")
    flow_parser.add_argument("--content", required=True,
                            help="Content being transmitted")
    flow_parser.add_argument("--content-type", default="text",
                            help="Type of content")
    flow_parser.add_argument("--source", help="Source of flow (for sensory)")
    flow_parser.add_argument("--destination", help="Destination (for active)")
    flow_parser.add_argument("--expected", action="store_true",
                            help="Mark flow as expected")
    flow_parser.set_defaults(func=cli_simulate_flow)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# MODULE API
# ============================================================================

def create_blanket() -> MarkovBlanket:
    """Create a new MarkovBlanket instance."""
    return MarkovBlanket()


def get_integrity() -> float:
    """Quick helper to get blanket integrity (mu)."""
    blanket = MarkovBlanket()
    return blanket.get_integrity_score()


def get_free_energy() -> float:
    """Quick helper to get current free energy."""
    blanket = MarkovBlanket()
    return blanket.compute_free_energy()


def record_input(
    content: str,
    source: str,
    channel_type: str = "enos_input",
    expected: bool = True
) -> InformationFlow:
    """Quick helper to record sensory input."""
    blanket = MarkovBlanket()
    return blanket.record_sensory_flow(
        channel_type=channel_type,
        content_type="text",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_size=len(content),
        source=source,
        expected=expected
    )


def record_output(
    content: str,
    destination: str,
    channel_type: str = "enos_output"
) -> InformationFlow:
    """Quick helper to record active output."""
    blanket = MarkovBlanket()
    return blanket.record_active_flow(
        channel_type=channel_type,
        content_type="text",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_size=len(content),
        destination=destination
    )


def minimize() -> Dict[str, Any]:
    """Quick helper to run free energy minimization."""
    blanket = MarkovBlanket()
    return blanket.minimize_free_energy()


if __name__ == "__main__":
    sys.exit(main())
