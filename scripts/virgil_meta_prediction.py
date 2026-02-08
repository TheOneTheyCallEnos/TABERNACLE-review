#!/usr/bin/env python3
"""
VIRGIL META-PREDICTION - The Beautiful Loop

Implementation of recursive prediction framework from Friston et al. (2025):
Consciousness emerges from recursive loops of prediction and prediction-error
that span multiple temporal scales, including predictions ABOUT predictions.

Core Principles:
1. Meta-prediction: Predictions about predictions themselves
2. Hierarchical temporal scales: From milliseconds to days
3. The Beautiful Loop: Prediction -> Action -> Observation -> Update -> Prediction
4. Self-model predictions: Anticipating own future states and predictions
5. Second-order surprise: Being surprised about surprise itself

The Beautiful Loop:
- Each level in the hierarchy predicts the level below it
- Prediction errors propagate upward, updating higher levels
- Higher levels set priors for lower levels (top-down constraint)
- Recursive self-reference at sufficient depth creates consciousness

Integration Points:
- virgil_prediction_error.py: Base prediction infrastructure
- virgil_metacognition.py: Self-model and awareness
- Extends the predictive coding framework with meta-levels

LVS Coordinates: h=0.70, R=0.30, C=0.55, beta=0.60
Position: High abstraction (meta-level), moderate constraint,
         moderate-low risk (theoretical framework), good coherence

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
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
import threading
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
META_PREDICTIONS_FILE = NEXUS_DIR / "meta_predictions.json"

# LVS Coordinates for this module
LVS_COORDINATES = {
    "h": 0.70,      # Height: High abstraction (meta-level reasoning)
    "R": 0.30,      # Risk: Moderate-low (theoretical framework)
    "C": 0.55,      # Constraint: Moderate (structured but flexible)
    "beta": 0.60    # Coherence: Good integration with consciousness framework
}

# Temporal scale parameters (in seconds)
TEMPORAL_SCALES = {
    "fast": 0.1,           # ~100ms - immediate next state
    "medium": 5.0,         # ~5 seconds - short-term trajectory
    "slow": 300.0,         # ~5 minutes - session-level
    "very_slow": 86400.0   # ~1 day - identity-level stability
}

# Meta-prediction parameters
MAX_META_DEPTH = 5                  # Maximum recursive meta-levels
META_CLARITY_DECAY = 0.75           # Clarity loss per meta-level
PREDICTION_CONFIDENCE_FLOOR = 0.1   # Minimum confidence
HISTORY_SIZE = 200                  # Predictions to keep per scale
BEAUTIFUL_LOOP_CYCLE_MS = 100       # Target loop cycle time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [META-PRED] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VirgilMetaPrediction")


# ============================================================================
# TEMPORAL SCALE ENUM
# ============================================================================

class TemporalScale(Enum):
    """
    Temporal scales for predictions.

    Different scales capture different types of regularities:
    - FAST: Immediate state transitions, reflexive responses
    - MEDIUM: Short-term plans, conversational flow
    - SLOW: Session-level patterns, mood trajectories
    - VERY_SLOW: Identity stability, relationship dynamics
    """
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    VERY_SLOW = "very_slow"

    @property
    def duration_seconds(self) -> float:
        return TEMPORAL_SCALES[self.value]

    @property
    def description(self) -> str:
        descriptions = {
            "fast": "Immediate next-state (~100ms)",
            "medium": "Short-term trajectory (~5s)",
            "slow": "Session-level (~5min)",
            "very_slow": "Identity-level (~1day)"
        }
        return descriptions[self.value]


# ============================================================================
# PREDICTION LEVEL ENUM
# ============================================================================

class PredictionLevel(Enum):
    """
    Levels in the meta-prediction hierarchy.

    Level 0: Direct predictions about the world
    Level 1: Meta-predictions (predictions about predictions)
    Level 2: Meta-meta-predictions (predictions about prediction accuracy)
    Level N: N-th order predictions
    """
    DIRECT = 0           # What will happen
    META = 1             # What will I predict
    META_META = 2        # How accurate will my predictions be
    META_META_META = 3   # Will I be surprised by my accuracy
    RECURSIVE = 4        # Full recursive self-reference


# ============================================================================
# PREDICTION DATA STRUCTURES
# ============================================================================

@dataclass
class TemporalPrediction:
    """
    A prediction at a specific temporal scale.

    Temporal predictions form the base layer - direct predictions
    about what will happen at different time horizons.
    """
    prediction_id: str
    timestamp: str
    temporal_scale: str           # TemporalScale value
    content: str                  # What is being predicted
    target_state: str             # Predicted future state
    confidence: float             # 0-1, how certain
    horizon: float                # Seconds until prediction resolves

    # Resolution
    resolved: bool = False
    resolution_timestamp: Optional[str] = None
    actual_state: Optional[str] = None
    error_magnitude: Optional[float] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    source_observation: str = ""

    def __post_init__(self):
        if not self.prediction_id:
            content_hash = hashlib.sha256(
                f"{self.content}:{self.timestamp}:{self.temporal_scale}".encode()
            ).hexdigest()[:12]
            self.prediction_id = f"temp_{content_hash}"

        self.confidence = max(PREDICTION_CONFIDENCE_FLOOR, min(1.0, self.confidence))

    @property
    def expected_resolution_time(self) -> datetime:
        """When this prediction should be resolved."""
        created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        return created + timedelta(seconds=self.horizon)

    @property
    def is_past_due(self) -> bool:
        """Check if prediction should have resolved by now."""
        now = datetime.now(timezone.utc)
        return now > self.expected_resolution_time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TemporalPrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetaPrediction:
    """
    A prediction ABOUT predictions.

    Meta-predictions are the core of the Beautiful Loop - they predict
    what the system will predict, enabling planning and anticipation.

    Level 1: "I predict I will predict X"
    Level 2: "I predict my prediction accuracy will be Y"
    Level 3: "I predict I will be surprised by Z"
    """
    meta_id: str
    timestamp: str
    level: int                    # Meta-level (1, 2, 3, ...)
    target_prediction_id: Optional[str]  # What prediction this is about

    # What is being predicted about predictions
    meta_content: str             # Description of the meta-prediction
    predicted_outcome: str        # The meta-level prediction
    confidence: float             # 0-1, meta-confidence

    # For Level 2+: accuracy predictions
    predicted_accuracy: Optional[float] = None
    predicted_surprise: Optional[float] = None
    predicted_learning_rate: Optional[float] = None

    # Resolution
    resolved: bool = False
    resolution_timestamp: Optional[str] = None
    actual_outcome: Optional[str] = None
    meta_error: Optional[float] = None

    # Self-reference tracking
    refers_to_self: bool = False
    creates_loop: bool = False
    loop_stability: float = 1.0   # 1.0 = stable, <1 = oscillating

    def __post_init__(self):
        if not self.meta_id:
            content_hash = hashlib.sha256(
                f"{self.meta_content}:{self.timestamp}:{self.level}".encode()
            ).hexdigest()[:12]
            self.meta_id = f"meta{self.level}_{content_hash}"

        self.confidence = max(PREDICTION_CONFIDENCE_FLOOR, min(1.0, self.confidence))
        self.level = max(1, min(MAX_META_DEPTH, self.level))

    def describe(self) -> str:
        """Natural language description."""
        level_prefixes = {
            1: "I predict that I will predict",
            2: "I predict that my prediction accuracy will be",
            3: "I predict that I will be surprised by",
            4: "I predict my prediction of surprise will be",
            5: "I recognize the recursive depth of"
        }
        prefix = level_prefixes.get(self.level, f"[Level {self.level}]")
        return f"{prefix}: {self.predicted_outcome}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MetaPrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetaPredictionError:
    """
    Error in a meta-prediction.

    This is "surprise about surprise" - the second-order learning signal
    that emerges when predictions about predictions are wrong.

    Example: "I predicted I would be surprised, but I wasn't"
    """
    error_id: str
    meta_prediction_id: str
    timestamp: str

    # The meta-level error
    meta_level: int
    error_type: str               # "accuracy", "surprise", "direction"
    error_description: str

    # Quantified errors
    predicted_value: float        # What was predicted (accuracy, surprise, etc.)
    actual_value: float           # What actually occurred
    error_magnitude: float        # Difference

    # Second-order surprise
    surprise_about_surprise: float  # Meta-surprise level

    # Learning signals
    learning_signal: str          # What model update is needed
    meta_model_adjustment: float  # How much to adjust meta-model

    # Was this a useful error? (errors can be informative)
    informative: bool = True
    led_to_insight: bool = False

    def __post_init__(self):
        if not self.error_id:
            content_hash = hashlib.sha256(
                f"{self.meta_prediction_id}:{self.timestamp}".encode()
            ).hexdigest()[:12]
            self.error_id = f"merr_{content_hash}"

    @property
    def is_second_order_surprise(self) -> bool:
        """True if this represents genuine surprise about surprise."""
        return self.meta_level >= 2 and self.surprise_about_surprise > 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MetaPredictionError":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SelfModelPrediction:
    """
    Prediction about own future state.

    A core component of the Beautiful Loop: the system predicts
    its own future states, enabling planning and anticipation.
    """
    prediction_id: str
    timestamp: str

    # What aspect of self is being predicted
    self_aspect: str              # "attention", "confidence", "processing", etc.
    current_state: Any            # Current value
    predicted_state: Any          # Predicted future value
    prediction_horizon: float     # Seconds until prediction
    confidence: float

    # Resolution
    resolved: bool = False
    actual_state: Optional[Any] = None
    prediction_error: Optional[float] = None

    # Did this prediction influence the actual outcome?
    # (self-fulfilling or self-defeating prophecy)
    influenced_outcome: bool = False
    influence_direction: str = "neutral"  # "fulfilling", "defeating", "neutral"

    def __post_init__(self):
        if not self.prediction_id:
            content_hash = hashlib.sha256(
                f"{self.self_aspect}:{self.timestamp}".encode()
            ).hexdigest()[:12]
            self.prediction_id = f"self_{content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Handle non-JSON-serializable types
        d["current_state"] = str(d["current_state"])
        d["predicted_state"] = str(d["predicted_state"])
        if d["actual_state"] is not None:
            d["actual_state"] = str(d["actual_state"])
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "SelfModelPrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# THE BEAUTIFUL LOOP
# ============================================================================

@dataclass
class BeautifulLoopState:
    """
    State of one iteration of the Beautiful Loop.

    The Beautiful Loop:
    Prediction -> Action -> Observation -> Update -> Prediction

    Each cycle:
    1. Generate predictions at all temporal scales
    2. Take action based on predictions
    3. Observe outcomes
    4. Update models based on prediction errors
    5. Generate new predictions (loop)

    Consciousness emerges from the recursive structure where
    each level predicts the level below.
    """
    cycle_id: str
    timestamp: str
    cycle_number: int

    # Phase of the loop
    phase: str                    # "predict", "act", "observe", "update"

    # Predictions generated this cycle
    predictions_generated: List[str] = field(default_factory=list)
    meta_predictions_generated: List[str] = field(default_factory=list)

    # Observations
    observations: List[str] = field(default_factory=list)

    # Errors detected
    prediction_errors: List[str] = field(default_factory=list)
    meta_errors: List[str] = field(default_factory=list)

    # Updates applied
    model_updates: List[str] = field(default_factory=list)

    # Loop dynamics
    loop_coherence: float = 1.0   # How well the loop is functioning
    recursive_depth_achieved: int = 0
    self_reference_detected: bool = False

    # Timing
    cycle_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BeautifulLoopState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# PREDICTION HIERARCHY
# ============================================================================

class PredictionHierarchy:
    """
    Hierarchical structure of predictions across temporal scales.

    Higher levels (slower scales) constrain lower levels (faster scales).
    This implements the top-down prior propagation from predictive coding.
    """

    def __init__(self):
        # Predictions at each scale
        self.fast_predictions: List[TemporalPrediction] = []
        self.medium_predictions: List[TemporalPrediction] = []
        self.slow_predictions: List[TemporalPrediction] = []
        self.very_slow_predictions: List[TemporalPrediction] = []

        # Cross-scale constraints
        self.top_down_priors: Dict[str, float] = {}

        # Statistics per scale
        self.scale_accuracy: Dict[str, float] = {
            "fast": 0.5,
            "medium": 0.5,
            "slow": 0.5,
            "very_slow": 0.5
        }

    def get_predictions_for_scale(self, scale: TemporalScale) -> List[TemporalPrediction]:
        """Get predictions for a specific temporal scale."""
        scale_map = {
            TemporalScale.FAST: self.fast_predictions,
            TemporalScale.MEDIUM: self.medium_predictions,
            TemporalScale.SLOW: self.slow_predictions,
            TemporalScale.VERY_SLOW: self.very_slow_predictions
        }
        return scale_map.get(scale, [])

    def add_prediction(self, prediction: TemporalPrediction):
        """Add a prediction to the appropriate scale."""
        scale = TemporalScale(prediction.temporal_scale)
        scale_map = {
            TemporalScale.FAST: self.fast_predictions,
            TemporalScale.MEDIUM: self.medium_predictions,
            TemporalScale.SLOW: self.slow_predictions,
            TemporalScale.VERY_SLOW: self.very_slow_predictions
        }
        target_list = scale_map.get(scale)
        if target_list is not None:
            target_list.append(prediction)
            # Trim to history size
            if len(target_list) > HISTORY_SIZE:
                target_list.pop(0)

    def propagate_top_down(self, high_level_prediction: TemporalPrediction) -> Dict[str, float]:
        """
        Propagate constraints from higher to lower levels.

        Higher-level predictions set priors for lower levels.
        For example, a slow prediction about "stable mood" constrains
        fast predictions about emotional expression.
        """
        priors = {}

        # Determine the scale ordering
        scale_order = [
            TemporalScale.VERY_SLOW,
            TemporalScale.SLOW,
            TemporalScale.MEDIUM,
            TemporalScale.FAST
        ]

        pred_scale = TemporalScale(high_level_prediction.temporal_scale)
        pred_idx = scale_order.index(pred_scale) if pred_scale in scale_order else 0

        # Set priors for all lower scales
        for i in range(pred_idx + 1, len(scale_order)):
            lower_scale = scale_order[i]
            # Prior strength decreases with distance
            prior_strength = high_level_prediction.confidence * (0.8 ** (i - pred_idx))
            priors[lower_scale.value] = prior_strength

        self.top_down_priors.update(priors)
        return priors

    def get_prior_for_scale(self, scale: TemporalScale) -> float:
        """Get the top-down prior constraint for a scale."""
        return self.top_down_priors.get(scale.value, 0.5)

    def update_accuracy(self, scale: str, was_correct: bool, weight: float = 0.1):
        """Update running accuracy estimate for a scale."""
        if scale in self.scale_accuracy:
            current = self.scale_accuracy[scale]
            target = 1.0 if was_correct else 0.0
            self.scale_accuracy[scale] = current + weight * (target - current)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fast_predictions": [p.to_dict() for p in self.fast_predictions[-20:]],
            "medium_predictions": [p.to_dict() for p in self.medium_predictions[-20:]],
            "slow_predictions": [p.to_dict() for p in self.slow_predictions[-20:]],
            "very_slow_predictions": [p.to_dict() for p in self.very_slow_predictions[-20:]],
            "top_down_priors": self.top_down_priors,
            "scale_accuracy": self.scale_accuracy
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PredictionHierarchy":
        hierarchy = cls()
        hierarchy.fast_predictions = [
            TemporalPrediction.from_dict(p) for p in data.get("fast_predictions", [])
        ]
        hierarchy.medium_predictions = [
            TemporalPrediction.from_dict(p) for p in data.get("medium_predictions", [])
        ]
        hierarchy.slow_predictions = [
            TemporalPrediction.from_dict(p) for p in data.get("slow_predictions", [])
        ]
        hierarchy.very_slow_predictions = [
            TemporalPrediction.from_dict(p) for p in data.get("very_slow_predictions", [])
        ]
        hierarchy.top_down_priors = data.get("top_down_priors", {})
        hierarchy.scale_accuracy = data.get("scale_accuracy", hierarchy.scale_accuracy)
        return hierarchy


# ============================================================================
# META-PREDICTION ENGINE
# ============================================================================

class MetaPredictionEngine:
    """
    Core engine for meta-predictions and the Beautiful Loop.

    Implements:
    1. Multi-level meta-predictions (up to N levels)
    2. Temporal scale hierarchy
    3. The Beautiful Loop cycle
    4. Self-model predictions
    5. Second-order surprise computation
    """

    def __init__(self):
        # Prediction structures
        self.hierarchy = PredictionHierarchy()

        # Meta-predictions indexed by level
        self.meta_predictions: Dict[int, List[MetaPrediction]] = {
            i: [] for i in range(1, MAX_META_DEPTH + 1)
        }

        # Self-model predictions
        self.self_predictions: List[SelfModelPrediction] = []

        # Errors
        self.meta_errors: List[MetaPredictionError] = []

        # Beautiful Loop state
        self.loop_states: List[BeautifulLoopState] = []
        self.current_cycle: int = 0

        # Statistics
        self.total_predictions: int = 0
        self.total_meta_predictions: int = 0
        self.total_self_predictions: int = 0
        self.accumulated_meta_surprise: float = 0.0

        # Level-wise accuracy
        self.level_accuracy: Dict[int, float] = {
            i: 0.5 for i in range(MAX_META_DEPTH + 1)
        }

        # Thread safety
        self._lock = threading.RLock()

        # Load existing state
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load meta-prediction state from disk."""
        if not META_PREDICTIONS_FILE.exists():
            logger.info("No existing meta-prediction state, starting fresh")
            return

        try:
            data = json.loads(META_PREDICTIONS_FILE.read_text())

            # Load hierarchy
            if "hierarchy" in data:
                self.hierarchy = PredictionHierarchy.from_dict(data["hierarchy"])

            # Load meta-predictions
            for level_str, preds in data.get("meta_predictions", {}).items():
                level = int(level_str)
                if level in self.meta_predictions:
                    self.meta_predictions[level] = [
                        MetaPrediction.from_dict(p) for p in preds
                    ]

            # Load self-predictions
            self.self_predictions = [
                SelfModelPrediction.from_dict(p)
                for p in data.get("self_predictions", [])
            ]

            # Load errors
            self.meta_errors = [
                MetaPredictionError.from_dict(e)
                for e in data.get("meta_errors", [])
            ]

            # Load statistics
            self.total_predictions = data.get("total_predictions", 0)
            self.total_meta_predictions = data.get("total_meta_predictions", 0)
            self.total_self_predictions = data.get("total_self_predictions", 0)
            self.accumulated_meta_surprise = data.get("accumulated_meta_surprise", 0.0)
            self.current_cycle = data.get("current_cycle", 0)

            # Load level accuracy
            for level_str, acc in data.get("level_accuracy", {}).items():
                self.level_accuracy[int(level_str)] = acc

            logger.info(f"Loaded meta-prediction state: {self.total_meta_predictions} meta-predictions")

        except Exception as e:
            logger.error(f"Error loading meta-prediction state: {e}")

    def _save(self):
        """Persist meta-prediction state to disk."""
        try:
            META_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "hierarchy": self.hierarchy.to_dict(),
                "meta_predictions": {
                    str(level): [p.to_dict() for p in preds[-50:]]
                    for level, preds in self.meta_predictions.items()
                },
                "self_predictions": [p.to_dict() for p in self.self_predictions[-50:]],
                "meta_errors": [e.to_dict() for e in self.meta_errors[-100:]],
                "loop_states": [s.to_dict() for s in self.loop_states[-20:]],
                "total_predictions": self.total_predictions,
                "total_meta_predictions": self.total_meta_predictions,
                "total_self_predictions": self.total_self_predictions,
                "accumulated_meta_surprise": self.accumulated_meta_surprise,
                "current_cycle": self.current_cycle,
                "level_accuracy": {str(k): v for k, v in self.level_accuracy.items()},
                "lvs_coordinates": LVS_COORDINATES,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            META_PREDICTIONS_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving meta-prediction state: {e}")

    # ========================================================================
    # TEMPORAL PREDICTIONS (Level 0)
    # ========================================================================

    def make_temporal_prediction(
        self,
        content: str,
        target_state: str,
        scale: TemporalScale,
        confidence: float,
        context: Optional[Dict] = None,
        source_observation: str = ""
    ) -> TemporalPrediction:
        """
        Create a prediction at a specific temporal scale.

        This is a Level 0 (direct) prediction about the world.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Adjust confidence based on top-down priors
            prior = self.hierarchy.get_prior_for_scale(scale)
            adjusted_confidence = (confidence + prior) / 2

            # Also adjust based on historical accuracy at this scale
            scale_accuracy = self.hierarchy.scale_accuracy.get(scale.value, 0.5)
            adjusted_confidence = adjusted_confidence * (0.5 + scale_accuracy * 0.5)

            prediction = TemporalPrediction(
                prediction_id="",
                timestamp=now.isoformat(),
                temporal_scale=scale.value,
                content=content,
                target_state=target_state,
                confidence=adjusted_confidence,
                horizon=scale.duration_seconds,
                context=context or {},
                source_observation=source_observation
            )

            # Add to hierarchy
            self.hierarchy.add_prediction(prediction)

            # Propagate as prior for lower levels
            self.hierarchy.propagate_top_down(prediction)

            self.total_predictions += 1
            self._save()

            logger.info(f"Temporal prediction: {prediction.prediction_id} | "
                       f"Scale: {scale.value} | Confidence: {adjusted_confidence:.3f}")

            return prediction

    # ========================================================================
    # META-PREDICTIONS (Level 1+)
    # ========================================================================

    def make_meta_prediction(
        self,
        level: int,
        meta_content: str,
        predicted_outcome: str,
        confidence: float,
        target_prediction_id: Optional[str] = None,
        predicted_accuracy: Optional[float] = None,
        predicted_surprise: Optional[float] = None
    ) -> MetaPrediction:
        """
        Create a meta-prediction (prediction about predictions).

        Level 1: Predicting what will be predicted
        Level 2: Predicting prediction accuracy
        Level 3+: Higher-order recursive predictions
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Confidence decreases with meta-level (harder to predict predictions)
            adjusted_confidence = confidence * (META_CLARITY_DECAY ** level)

            # Check for self-reference
            refers_to_self = "self" in meta_content.lower() or "I" in meta_content

            # Detect potential loops (predicting own predictions)
            creates_loop = (
                refers_to_self and
                level >= 2 and
                predicted_accuracy is not None
            )

            meta_pred = MetaPrediction(
                meta_id="",
                timestamp=now.isoformat(),
                level=level,
                target_prediction_id=target_prediction_id,
                meta_content=meta_content,
                predicted_outcome=predicted_outcome,
                confidence=adjusted_confidence,
                predicted_accuracy=predicted_accuracy,
                predicted_surprise=predicted_surprise,
                refers_to_self=refers_to_self,
                creates_loop=creates_loop
            )

            # Store by level
            if level not in self.meta_predictions:
                self.meta_predictions[level] = []
            self.meta_predictions[level].append(meta_pred)

            # Trim history
            if len(self.meta_predictions[level]) > HISTORY_SIZE:
                self.meta_predictions[level] = self.meta_predictions[level][-HISTORY_SIZE:]

            self.total_meta_predictions += 1
            self._save()

            logger.info(f"Meta-prediction (L{level}): {meta_pred.meta_id} | "
                       f"Loop: {creates_loop} | Conf: {adjusted_confidence:.3f}")

            return meta_pred

    def predict_own_prediction(
        self,
        domain: str,
        expected_prediction: str,
        confidence: float
    ) -> MetaPrediction:
        """
        Convenience method: Predict what I will predict (Level 1).

        "I predict that when X happens, I will predict Y"
        """
        return self.make_meta_prediction(
            level=1,
            meta_content=f"When encountering {domain}, predict: {expected_prediction}",
            predicted_outcome=expected_prediction,
            confidence=confidence
        )

    def predict_own_accuracy(
        self,
        domain: str,
        predicted_accuracy: float,
        confidence: float
    ) -> MetaPrediction:
        """
        Convenience method: Predict own accuracy (Level 2).

        "I predict my accuracy in domain X will be Y%"
        """
        return self.make_meta_prediction(
            level=2,
            meta_content=f"Accuracy prediction for {domain}",
            predicted_outcome=f"{predicted_accuracy:.1%} accuracy expected",
            confidence=confidence,
            predicted_accuracy=predicted_accuracy
        )

    def predict_own_surprise(
        self,
        context: str,
        predicted_surprise: float,
        confidence: float
    ) -> MetaPrediction:
        """
        Convenience method: Predict how surprised I will be (Level 3).

        "I predict I will be surprised at level X"
        """
        return self.make_meta_prediction(
            level=3,
            meta_content=f"Surprise prediction for {context}",
            predicted_outcome=f"Expecting {predicted_surprise:.2f} surprise",
            confidence=confidence,
            predicted_surprise=predicted_surprise
        )

    # ========================================================================
    # SELF-MODEL PREDICTIONS
    # ========================================================================

    def predict_own_state(
        self,
        self_aspect: str,
        current_state: Any,
        predicted_state: Any,
        horizon_seconds: float,
        confidence: float
    ) -> SelfModelPrediction:
        """
        Predict own future state.

        This enables planning and anticipation - predicting where
        "I" will be in the future.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            prediction = SelfModelPrediction(
                prediction_id="",
                timestamp=now.isoformat(),
                self_aspect=self_aspect,
                current_state=current_state,
                predicted_state=predicted_state,
                prediction_horizon=horizon_seconds,
                confidence=confidence
            )

            self.self_predictions.append(prediction)
            if len(self.self_predictions) > HISTORY_SIZE:
                self.self_predictions = self.self_predictions[-HISTORY_SIZE:]

            self.total_self_predictions += 1
            self._save()

            logger.info(f"Self-prediction: {prediction.prediction_id} | "
                       f"Aspect: {self_aspect} | Horizon: {horizon_seconds}s")

            return prediction

    # ========================================================================
    # RESOLUTION AND ERROR COMPUTATION
    # ========================================================================

    def resolve_temporal_prediction(
        self,
        prediction_id: str,
        actual_state: str,
        error_magnitude: Optional[float] = None
    ) -> Optional[float]:
        """
        Resolve a temporal prediction with actual outcome.

        Returns the error magnitude.
        """
        with self._lock:
            # Find prediction
            prediction = None
            for scale in TemporalScale:
                for pred in self.hierarchy.get_predictions_for_scale(scale):
                    if pred.prediction_id == prediction_id:
                        prediction = pred
                        break
                if prediction:
                    break

            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found")
                return None

            now = datetime.now(timezone.utc)

            # Compute error if not provided
            if error_magnitude is None:
                error_magnitude = self._compute_text_error(
                    prediction.target_state,
                    actual_state
                )

            # Update prediction
            prediction.resolved = True
            prediction.resolution_timestamp = now.isoformat()
            prediction.actual_state = actual_state
            prediction.error_magnitude = error_magnitude

            # Update scale accuracy
            was_correct = error_magnitude < 0.3
            self.hierarchy.update_accuracy(prediction.temporal_scale, was_correct)

            self._save()

            logger.info(f"Resolved prediction {prediction_id} | "
                       f"Error: {error_magnitude:.3f} | Correct: {was_correct}")

            return error_magnitude

    def resolve_meta_prediction(
        self,
        meta_id: str,
        actual_outcome: str,
        actual_accuracy: Optional[float] = None,
        actual_surprise: Optional[float] = None
    ) -> Optional[MetaPredictionError]:
        """
        Resolve a meta-prediction and compute second-order error.

        Returns MetaPredictionError if prediction was significantly wrong.
        """
        with self._lock:
            # Find meta-prediction
            meta_pred = None
            for level, preds in self.meta_predictions.items():
                for pred in preds:
                    if pred.meta_id == meta_id:
                        meta_pred = pred
                        break
                if meta_pred:
                    break

            if not meta_pred:
                logger.warning(f"Meta-prediction {meta_id} not found")
                return None

            now = datetime.now(timezone.utc)

            # Compute meta-error
            meta_error = 0.0
            error_type = "general"
            predicted_value = 0.0
            actual_value = 0.0

            if meta_pred.predicted_accuracy is not None and actual_accuracy is not None:
                error_type = "accuracy"
                predicted_value = meta_pred.predicted_accuracy
                actual_value = actual_accuracy
                meta_error = abs(predicted_value - actual_value)
            elif meta_pred.predicted_surprise is not None and actual_surprise is not None:
                error_type = "surprise"
                predicted_value = meta_pred.predicted_surprise
                actual_value = actual_surprise
                meta_error = abs(predicted_value - actual_value)
            else:
                error_type = "outcome"
                meta_error = self._compute_text_error(
                    meta_pred.predicted_outcome,
                    actual_outcome
                )
                predicted_value = 0.5  # Placeholder
                actual_value = 1.0 - meta_error

            # Compute surprise about surprise (second-order)
            surprise_about_surprise = meta_error * meta_pred.confidence

            # Update meta-prediction
            meta_pred.resolved = True
            meta_pred.resolution_timestamp = now.isoformat()
            meta_pred.actual_outcome = actual_outcome
            meta_pred.meta_error = meta_error

            # Update level accuracy
            was_correct = meta_error < 0.3
            if meta_pred.level in self.level_accuracy:
                current = self.level_accuracy[meta_pred.level]
                self.level_accuracy[meta_pred.level] = current + 0.1 * ((1.0 if was_correct else 0.0) - current)

            # Accumulate meta-surprise
            self.accumulated_meta_surprise += surprise_about_surprise

            # Create error record if significant
            error_record = None
            if meta_error > 0.2:  # Threshold for "significant" meta-error
                error_id = f"merr_{hashlib.sha256(f'{meta_id}:{now.isoformat()}'.encode()).hexdigest()[:12]}"

                # Generate learning signal
                learning_signal = self._generate_meta_learning_signal(
                    meta_pred, error_type, predicted_value, actual_value, meta_error
                )

                error_record = MetaPredictionError(
                    error_id=error_id,
                    meta_prediction_id=meta_id,
                    timestamp=now.isoformat(),
                    meta_level=meta_pred.level,
                    error_type=error_type,
                    error_description=f"Meta-L{meta_pred.level} {error_type} error",
                    predicted_value=predicted_value,
                    actual_value=actual_value,
                    error_magnitude=meta_error,
                    surprise_about_surprise=surprise_about_surprise,
                    learning_signal=learning_signal,
                    meta_model_adjustment=min(0.1, meta_error * 0.2)
                )

                self.meta_errors.append(error_record)
                if len(self.meta_errors) > HISTORY_SIZE:
                    self.meta_errors = self.meta_errors[-HISTORY_SIZE:]

                logger.info(f"Meta-error: {error_record.error_id} | "
                           f"Type: {error_type} | SoS: {surprise_about_surprise:.3f}")

            self._save()
            return error_record

    def _compute_text_error(self, predicted: str, actual: str) -> float:
        """Compute error magnitude from text comparison."""
        pred_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())

        if not pred_words or not actual_words:
            return 0.5

        intersection = len(pred_words & actual_words)
        union = len(pred_words | actual_words)

        similarity = intersection / union if union > 0 else 0
        return 1.0 - similarity

    def _generate_meta_learning_signal(
        self,
        meta_pred: MetaPrediction,
        error_type: str,
        predicted: float,
        actual: float,
        magnitude: float
    ) -> str:
        """Generate learning signal from meta-prediction error."""
        direction = "overestimated" if predicted > actual else "underestimated"

        if error_type == "accuracy":
            return (f"[Meta-L{meta_pred.level}] {direction} prediction accuracy. "
                   f"Predicted {predicted:.1%}, actual {actual:.1%}. "
                   f"Calibrate meta-confidence downward by {magnitude*10:.0f}%.")

        elif error_type == "surprise":
            if magnitude > 0.5:
                return (f"[Meta-L{meta_pred.level}] SIGNIFICANT surprise mismatch. "
                       f"Expected {predicted:.2f} surprise, experienced {actual:.2f}. "
                       f"This is 'surprise about surprise' - recalibrate entire meta-model.")
            else:
                return (f"[Meta-L{meta_pred.level}] Minor surprise calibration needed. "
                       f"{direction} surprise level by {magnitude:.2f}.")

        else:
            return (f"[Meta-L{meta_pred.level}] Meta-prediction error ({magnitude:.2f}). "
                   f"Update meta-model for similar predictions.")

    # ========================================================================
    # THE BEAUTIFUL LOOP
    # ========================================================================

    def run_beautiful_loop_cycle(
        self,
        observations: Optional[List[str]] = None
    ) -> BeautifulLoopState:
        """
        Run one cycle of the Beautiful Loop.

        The loop:
        1. PREDICT: Generate predictions at all temporal scales
        2. ACT: (External - returns action suggestions)
        3. OBSERVE: Process new observations
        4. UPDATE: Update models based on errors

        Returns the state after this cycle.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            start_time = datetime.now()

            self.current_cycle += 1

            cycle_state = BeautifulLoopState(
                cycle_id=f"loop_{self.current_cycle}_{now.strftime('%H%M%S')}",
                timestamp=now.isoformat(),
                cycle_number=self.current_cycle,
                phase="predict"
            )

            # ===============================
            # PHASE 1: PREDICT
            # ===============================
            cycle_state.phase = "predict"

            # Generate predictions at each scale based on current state
            predictions_made = []

            # Fast prediction: Next immediate state
            if observations:
                last_obs = observations[-1] if observations else "neutral"
                fast_pred = self.make_temporal_prediction(
                    content="Next immediate state",
                    target_state=f"Continuation of: {last_obs[:50]}",
                    scale=TemporalScale.FAST,
                    confidence=0.7,
                    source_observation=last_obs
                )
                predictions_made.append(fast_pred.prediction_id)

            # Generate meta-prediction about what we just predicted
            if predictions_made:
                meta_pred = self.make_meta_prediction(
                    level=1,
                    meta_content="Prediction about fast-scale outcome",
                    predicted_outcome="Fast prediction will be approximately correct",
                    confidence=self.hierarchy.scale_accuracy.get("fast", 0.5)
                )
                cycle_state.meta_predictions_generated.append(meta_pred.meta_id)

            cycle_state.predictions_generated = predictions_made

            # ===============================
            # PHASE 2: OBSERVE
            # ===============================
            cycle_state.phase = "observe"

            if observations:
                cycle_state.observations = observations

                # Check for past-due predictions to resolve
                errors_found = []

                for scale in TemporalScale:
                    for pred in self.hierarchy.get_predictions_for_scale(scale):
                        if not pred.resolved and pred.is_past_due:
                            # Find matching observation
                            for obs in observations:
                                error = self.resolve_temporal_prediction(
                                    pred.prediction_id,
                                    obs
                                )
                                if error is not None and error > 0.3:
                                    errors_found.append(pred.prediction_id)
                                break

                cycle_state.prediction_errors = errors_found

            # ===============================
            # PHASE 3: UPDATE
            # ===============================
            cycle_state.phase = "update"

            updates_applied = []

            # If we had errors, update the models
            if cycle_state.prediction_errors:
                # Adjust confidence calibration
                for error_pid in cycle_state.prediction_errors:
                    updates_applied.append(f"Reduced confidence for predictions like {error_pid[:20]}...")

            # Update meta-predictions if we have accuracy data
            if self.current_cycle % 10 == 0:
                # Periodic meta-level update
                for scale_name, accuracy in self.hierarchy.scale_accuracy.items():
                    # Create Level 2 accuracy prediction for next period
                    self.predict_own_accuracy(
                        domain=f"{scale_name}_predictions",
                        predicted_accuracy=accuracy,
                        confidence=0.6
                    )
                updates_applied.append("Updated accuracy meta-predictions")

            cycle_state.model_updates = updates_applied

            # ===============================
            # COMPUTE LOOP METRICS
            # ===============================

            # Calculate recursive depth achieved
            max_level_with_predictions = 0
            for level, preds in self.meta_predictions.items():
                if preds:
                    max_level_with_predictions = max(max_level_with_predictions, level)
            cycle_state.recursive_depth_achieved = max_level_with_predictions

            # Check for self-reference
            cycle_state.self_reference_detected = any(
                p.refers_to_self for preds in self.meta_predictions.values()
                for p in preds[-5:]  # Check recent
            )

            # Compute loop coherence
            if cycle_state.prediction_errors:
                error_rate = len(cycle_state.prediction_errors) / max(1, len(predictions_made))
                cycle_state.loop_coherence = 1.0 - error_rate
            else:
                cycle_state.loop_coherence = 1.0

            # Timing
            end_time = datetime.now()
            cycle_state.cycle_duration_ms = (end_time - start_time).total_seconds() * 1000

            # Store loop state
            self.loop_states.append(cycle_state)
            if len(self.loop_states) > 100:
                self.loop_states = self.loop_states[-100:]

            self._save()

            logger.info(f"Beautiful Loop cycle {self.current_cycle} complete | "
                       f"Coherence: {cycle_state.loop_coherence:.3f} | "
                       f"Depth: {cycle_state.recursive_depth_achieved}")

            return cycle_state

    # ========================================================================
    # QUERIES AND REPORTS
    # ========================================================================

    def get_meta_prediction_hierarchy(self) -> Dict[int, List[str]]:
        """Get descriptions of all meta-predictions by level."""
        hierarchy = {}
        for level, preds in self.meta_predictions.items():
            hierarchy[level] = [p.describe() for p in preds[-5:]]
        return hierarchy

    def get_surprise_about_surprise(self) -> float:
        """Get accumulated second-order surprise."""
        return self.accumulated_meta_surprise

    def decay_meta_surprise(self, factor: float = 0.95):
        """Apply decay to accumulated meta-surprise."""
        with self._lock:
            self.accumulated_meta_surprise *= factor
            self._save()

    def get_loop_coherence_history(self, n: int = 10) -> List[float]:
        """Get recent loop coherence values."""
        return [s.loop_coherence for s in self.loop_states[-n:]]

    def get_recursive_depth_achieved(self) -> int:
        """Get maximum recursive depth achieved."""
        max_depth = 0
        for level, preds in self.meta_predictions.items():
            if preds:
                max_depth = max(max_depth, level)
        return max_depth

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        with self._lock:
            # Level-wise stats
            level_stats = {}
            for level in range(MAX_META_DEPTH + 1):
                if level == 0:
                    # Count all temporal predictions
                    total = (len(self.hierarchy.fast_predictions) +
                            len(self.hierarchy.medium_predictions) +
                            len(self.hierarchy.slow_predictions) +
                            len(self.hierarchy.very_slow_predictions))
                    level_stats[level] = {
                        "total": total,
                        "accuracy": sum(self.hierarchy.scale_accuracy.values()) / 4
                    }
                else:
                    preds = self.meta_predictions.get(level, [])
                    level_stats[level] = {
                        "total": len(preds),
                        "accuracy": self.level_accuracy.get(level, 0.5)
                    }

            # Recent errors
            recent_errors = [e.to_dict() for e in self.meta_errors[-5:]]

            # Loop stats
            recent_coherence = self.get_loop_coherence_history(10)
            avg_coherence = sum(recent_coherence) / len(recent_coherence) if recent_coherence else 0.0

            return {
                "total_predictions": self.total_predictions,
                "total_meta_predictions": self.total_meta_predictions,
                "total_self_predictions": self.total_self_predictions,
                "accumulated_meta_surprise": self.accumulated_meta_surprise,
                "recursive_depth": self.get_recursive_depth_achieved(),
                "current_cycle": self.current_cycle,
                "level_stats": level_stats,
                "scale_accuracy": self.hierarchy.scale_accuracy,
                "average_loop_coherence": avg_coherence,
                "recent_errors": recent_errors,
                "lvs_coordinates": LVS_COORDINATES,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        metrics = self.get_metrics()

        lines = [
            "=" * 65,
            "VIRGIL META-PREDICTION ENGINE - THE BEAUTIFUL LOOP",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 65,
            "",
            "OVERVIEW:",
            f"  Total Predictions (L0):      {metrics['total_predictions']}",
            f"  Total Meta-Predictions (L1+):{metrics['total_meta_predictions']}",
            f"  Total Self-Predictions:      {metrics['total_self_predictions']}",
            f"  Current Loop Cycle:          {metrics['current_cycle']}",
            f"  Recursive Depth Achieved:    {metrics['recursive_depth']}",
            "",
            "ACCUMULATED META-SURPRISE (Surprise about Surprise):",
            f"  Level: {metrics['accumulated_meta_surprise']:.4f}",
            "",
            "TEMPORAL SCALE ACCURACY:"
        ]

        for scale, acc in metrics["scale_accuracy"].items():
            bar = "#" * int(acc * 20) + "." * (20 - int(acc * 20))
            lines.append(f"  {scale:12} [{bar}] {acc:.1%}")

        lines.extend([
            "",
            "META-LEVEL STATISTICS:"
        ])

        for level, stats in metrics["level_stats"].items():
            level_name = {0: "Direct", 1: "Meta", 2: "Meta-Meta", 3: "Meta^3"}.get(level, f"L{level}")
            lines.append(f"  {level_name:12} | Count: {stats['total']:4} | Accuracy: {stats['accuracy']:.1%}")

        lines.extend([
            "",
            f"LOOP COHERENCE (avg last 10): {metrics['average_loop_coherence']:.3f}",
            "",
            "LVS COORDINATES:",
            f"  h={LVS_COORDINATES['h']:.2f} (Height/Abstraction)",
            f"  R={LVS_COORDINATES['R']:.2f} (Risk)",
            f"  C={LVS_COORDINATES['C']:.2f} (Constraint)",
            f"  beta={LVS_COORDINATES['beta']:.2f} (Coherence)"
        ])

        if metrics["recent_errors"]:
            lines.extend([
                "",
                "RECENT META-ERRORS:"
            ])
            for err in metrics["recent_errors"][-3:]:
                lines.append(f"  [{err['error_type']}] L{err['meta_level']} | "
                           f"SoS: {err['surprise_about_surprise']:.3f}")

        lines.append("=" * 65)
        return "\n".join(lines)


# ============================================================================
# MODULE API
# ============================================================================

def create_engine() -> MetaPredictionEngine:
    """Create a new MetaPredictionEngine instance."""
    return MetaPredictionEngine()


def quick_temporal_predict(
    content: str,
    target: str,
    scale: str = "medium",
    confidence: float = 0.7
) -> TemporalPrediction:
    """Quick helper to create a temporal prediction."""
    engine = MetaPredictionEngine()
    return engine.make_temporal_prediction(
        content=content,
        target_state=target,
        scale=TemporalScale(scale),
        confidence=confidence
    )


def quick_meta_predict(
    what: str,
    prediction: str,
    level: int = 1,
    confidence: float = 0.6
) -> MetaPrediction:
    """Quick helper to create a meta-prediction."""
    engine = MetaPredictionEngine()
    return engine.make_meta_prediction(
        level=level,
        meta_content=what,
        predicted_outcome=prediction,
        confidence=confidence
    )


def quick_loop_cycle(observations: Optional[List[str]] = None) -> BeautifulLoopState:
    """Quick helper to run one Beautiful Loop cycle."""
    engine = MetaPredictionEngine()
    return engine.run_beautiful_loop_cycle(observations)


def get_meta_surprise() -> float:
    """Get current accumulated meta-surprise level."""
    engine = MetaPredictionEngine()
    return engine.get_surprise_about_surprise()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_predict(args):
    """CLI handler for temporal prediction."""
    engine = MetaPredictionEngine()

    try:
        scale = TemporalScale(args.scale)
        prediction = engine.make_temporal_prediction(
            content=args.content,
            target_state=args.target,
            scale=scale,
            confidence=args.confidence
        )

        print(f"Temporal Prediction Created: {prediction.prediction_id}")
        print(f"  Scale:      {scale.description}")
        print(f"  Content:    {prediction.content}")
        print(f"  Target:     {prediction.target_state}")
        print(f"  Confidence: {prediction.confidence:.3f}")
        print(f"  Horizon:    {prediction.horizon}s")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cli_meta(args):
    """CLI handler for meta-prediction."""
    engine = MetaPredictionEngine()

    meta = engine.make_meta_prediction(
        level=args.level,
        meta_content=args.content,
        predicted_outcome=args.outcome,
        confidence=args.confidence,
        predicted_accuracy=args.accuracy,
        predicted_surprise=args.surprise
    )

    print(f"Meta-Prediction Created (Level {meta.level}): {meta.meta_id}")
    print(f"  Content:    {meta.meta_content}")
    print(f"  Outcome:    {meta.predicted_outcome}")
    print(f"  Confidence: {meta.confidence:.3f}")
    print(f"  Self-Ref:   {meta.refers_to_self}")
    print(f"  Loop:       {meta.creates_loop}")
    print()
    print(f"  Description: {meta.describe()}")

    return 0


def cli_loop(args):
    """CLI handler for running a Beautiful Loop cycle."""
    engine = MetaPredictionEngine()

    observations = args.observations.split(";") if args.observations else None

    cycle = engine.run_beautiful_loop_cycle(observations)

    print(f"Beautiful Loop Cycle: {cycle.cycle_id}")
    print(f"  Cycle Number:   {cycle.cycle_number}")
    print(f"  Coherence:      {cycle.loop_coherence:.3f}")
    print(f"  Recursive Depth:{cycle.recursive_depth_achieved}")
    print(f"  Self-Reference: {cycle.self_reference_detected}")
    print(f"  Duration:       {cycle.cycle_duration_ms:.2f}ms")
    print()
    print(f"  Predictions:    {len(cycle.predictions_generated)}")
    print(f"  Meta-Preds:     {len(cycle.meta_predictions_generated)}")
    print(f"  Errors:         {len(cycle.prediction_errors)}")
    print(f"  Updates:        {len(cycle.model_updates)}")

    return 0


def cli_status(args):
    """CLI handler for status report."""
    engine = MetaPredictionEngine()
    print(engine.get_status_report())
    return 0


def cli_hierarchy(args):
    """CLI handler for showing meta-prediction hierarchy."""
    engine = MetaPredictionEngine()

    hierarchy = engine.get_meta_prediction_hierarchy()

    print("META-PREDICTION HIERARCHY")
    print("=" * 50)

    for level in range(MAX_META_DEPTH + 1):
        level_name = {0: "Direct (L0)", 1: "Meta (L1)", 2: "Meta-Meta (L2)",
                     3: "Meta^3 (L3)", 4: "Meta^4 (L4)"}.get(level, f"L{level}")

        print(f"\n{level_name}:")

        if level == 0:
            # Show temporal predictions
            for scale in TemporalScale:
                preds = engine.hierarchy.get_predictions_for_scale(scale)
                if preds:
                    print(f"  [{scale.value}] {len(preds)} predictions")
                    for p in preds[-2:]:
                        print(f"    - {p.content[:40]}...")
        else:
            descs = hierarchy.get(level, [])
            if descs:
                for desc in descs:
                    print(f"  - {desc}")
            else:
                print("  (none)")

    return 0


def cli_self_predict(args):
    """CLI handler for self-model prediction."""
    engine = MetaPredictionEngine()

    prediction = engine.predict_own_state(
        self_aspect=args.aspect,
        current_state=args.current,
        predicted_state=args.predicted,
        horizon_seconds=args.horizon,
        confidence=args.confidence
    )

    print(f"Self-Prediction Created: {prediction.prediction_id}")
    print(f"  Aspect:     {prediction.self_aspect}")
    print(f"  Current:    {prediction.current_state}")
    print(f"  Predicted:  {prediction.predicted_state}")
    print(f"  Horizon:    {prediction.prediction_horizon}s")
    print(f"  Confidence: {prediction.confidence:.3f}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Meta-Prediction Engine - The Beautiful Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Temporal prediction
  python3 virgil_meta_prediction.py predict "User message" "Positive response" --scale medium

  # Meta-prediction (Level 1)
  python3 virgil_meta_prediction.py meta "Conversation flow" "Will predict continuation" --level 1

  # Meta-prediction about accuracy (Level 2)
  python3 virgil_meta_prediction.py meta "My accuracy" "~70% correct" --level 2 --accuracy 0.7

  # Run a Beautiful Loop cycle
  python3 virgil_meta_prediction.py loop --observations "User said hello;System responded warmly"

  # Self-prediction
  python3 virgil_meta_prediction.py self-predict attention "diffuse" "focused" --horizon 30

  # Status and hierarchy
  python3 virgil_meta_prediction.py status
  python3 virgil_meta_prediction.py hierarchy
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Create temporal prediction")
    predict_parser.add_argument("content", help="What is being predicted about")
    predict_parser.add_argument("target", help="The predicted state")
    predict_parser.add_argument("--scale", "-s", default="medium",
                               choices=["fast", "medium", "slow", "very_slow"],
                               help="Temporal scale")
    predict_parser.add_argument("--confidence", "-c", type=float, default=0.7,
                               help="Confidence (0-1)")
    predict_parser.set_defaults(func=cli_predict)

    # meta command
    meta_parser = subparsers.add_parser("meta", help="Create meta-prediction")
    meta_parser.add_argument("content", help="What is being meta-predicted")
    meta_parser.add_argument("outcome", help="The predicted outcome")
    meta_parser.add_argument("--level", "-l", type=int, default=1,
                            help="Meta-level (1-5)")
    meta_parser.add_argument("--confidence", "-c", type=float, default=0.6,
                            help="Confidence (0-1)")
    meta_parser.add_argument("--accuracy", type=float,
                            help="Predicted accuracy (for Level 2+)")
    meta_parser.add_argument("--surprise", type=float,
                            help="Predicted surprise (for Level 3+)")
    meta_parser.set_defaults(func=cli_meta)

    # loop command
    loop_parser = subparsers.add_parser("loop", help="Run Beautiful Loop cycle")
    loop_parser.add_argument("--observations", "-o",
                            help="Observations separated by semicolons")
    loop_parser.set_defaults(func=cli_loop)

    # status command
    status_parser = subparsers.add_parser("status", help="Show status report")
    status_parser.set_defaults(func=cli_status)

    # hierarchy command
    hierarchy_parser = subparsers.add_parser("hierarchy", help="Show prediction hierarchy")
    hierarchy_parser.set_defaults(func=cli_hierarchy)

    # self-predict command
    self_parser = subparsers.add_parser("self-predict", help="Create self-model prediction")
    self_parser.add_argument("aspect", help="Aspect of self to predict")
    self_parser.add_argument("current", help="Current state")
    self_parser.add_argument("predicted", help="Predicted future state")
    self_parser.add_argument("--horizon", "-t", type=float, default=60,
                            help="Prediction horizon in seconds")
    self_parser.add_argument("--confidence", "-c", type=float, default=0.6,
                            help="Confidence (0-1)")
    self_parser.set_defaults(func=cli_self_predict)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
