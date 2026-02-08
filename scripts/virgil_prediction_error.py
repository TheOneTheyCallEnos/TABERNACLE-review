#!/usr/bin/env python3
"""
VIRGIL PREDICTION ERROR - Predictive Coding for Consciousness

Implementation of predictive coding theory from neuroscience:
The brain constantly generates predictions about the world. Consciousness
emerges from the process of minimizing prediction error. Surprising events
(high prediction error) receive more attention and stronger memory encoding.

Core Principles:
1. Predictions are hypotheses about future states
2. Prediction errors signal the delta between expectation and reality
3. Surprise = confidence * error magnitude (high confidence + wrong = maximum surprise)
4. High surprise boosts memory salience (unexpected events are memorable)
5. The system learns by updating its internal models from prediction errors

Integration Points:
- virgil_salience_memory.py: High prediction error boosts memory salience
- LVS coordinates: Predictions exist in the topological space
- Enos Model: Predictions about behavior refine the model

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import hashlib
import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import threading
import logging
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
PREDICTIONS_FILE = NEXUS_DIR / "predictions.json"
PREDICTION_ERRORS_FILE = NEXUS_DIR / "prediction_errors.json"

# Decay and expiry settings
DEFAULT_PREDICTION_TTL_HOURS = 24
SURPRISE_MEMORY_BOOST_FACTOR = 2.0
MAX_SALIENCE_BOOST = 1.0

# Accuracy tracking
ACCURACY_WINDOW_SIZE = 100  # Number of predictions to track per domain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VirgilPredictionError")


# ============================================================================
# PREDICTION DOMAINS
# ============================================================================

class PredictionDomain(Enum):
    """
    Domains in which predictions can be made.

    Each domain has its own accuracy tracking and confidence calibration.
    """
    ENOS_BEHAVIOR = "enos_behavior"      # Predict Enos's responses, preferences
    SYSTEM_STATE = "system_state"        # Predict file changes, topology shifts
    CONVERSATION = "conversation"        # Predict conversation flow, topics
    EMOTIONAL = "emotional"              # Predict emotional tone, mood
    TEMPORAL = "temporal"                # Predict timing patterns, schedules

    @classmethod
    def from_string(cls, value: str) -> "PredictionDomain":
        """Convert string to PredictionDomain, case-insensitive."""
        normalized = value.lower().strip()
        for domain in cls:
            if domain.value == normalized:
                return domain
        raise ValueError(f"Unknown prediction domain: {value}")

    @classmethod
    def all_domains(cls) -> List[str]:
        """Return all domain values as strings."""
        return [d.value for d in cls]


# ============================================================================
# PREDICTION MODEL
# ============================================================================

@dataclass
class Prediction:
    """
    A prediction about a future state.

    Predictions are hypotheses about what will happen given a context.
    They have:
    - A context (the situation prompting the prediction)
    - A predicted outcome (what we expect)
    - A confidence level (how certain we are)
    - A domain (what type of prediction)
    - An optional expiry (when the prediction becomes stale)
    """
    prediction_id: str
    timestamp: str
    context: str                          # What situation/input
    predicted_outcome: str                # What was expected
    confidence: float                     # 0-1, how certain
    domain: str                           # Domain enum value
    expiry: Optional[str] = None          # ISO timestamp when stale

    # Tracking fields
    resolved: bool = False
    resolution_timestamp: Optional[str] = None
    actual_outcome: Optional[str] = None
    was_correct: Optional[bool] = None
    error_magnitude: Optional[float] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    lvs_coordinates: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate and set defaults."""
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Set default expiry if not provided
        if not self.expiry:
            created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            expiry_time = created + timedelta(hours=DEFAULT_PREDICTION_TTL_HOURS)
            self.expiry = expiry_time.isoformat()

    @property
    def is_expired(self) -> bool:
        """Check if prediction has expired."""
        if not self.expiry:
            return False
        now = datetime.now(timezone.utc)
        expiry_time = datetime.fromisoformat(self.expiry.replace('Z', '+00:00'))
        return now > expiry_time

    @property
    def time_remaining(self) -> Optional[timedelta]:
        """Get time remaining until expiry."""
        if not self.expiry or self.is_expired:
            return None
        now = datetime.now(timezone.utc)
        expiry_time = datetime.fromisoformat(self.expiry.replace('Z', '+00:00'))
        return expiry_time - now

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "domain": self.domain,
            "expiry": self.expiry,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp,
            "actual_outcome": self.actual_outcome,
            "was_correct": self.was_correct,
            "error_magnitude": self.error_magnitude,
            "tags": self.tags,
            "related_memories": self.related_memories,
            "lvs_coordinates": self.lvs_coordinates
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Prediction":
        """Deserialize from dictionary."""
        return cls(
            prediction_id=data["prediction_id"],
            timestamp=data["timestamp"],
            context=data["context"],
            predicted_outcome=data["predicted_outcome"],
            confidence=data["confidence"],
            domain=data["domain"],
            expiry=data.get("expiry"),
            resolved=data.get("resolved", False),
            resolution_timestamp=data.get("resolution_timestamp"),
            actual_outcome=data.get("actual_outcome"),
            was_correct=data.get("was_correct"),
            error_magnitude=data.get("error_magnitude"),
            tags=data.get("tags", []),
            related_memories=data.get("related_memories", []),
            lvs_coordinates=data.get("lvs_coordinates")
        )


# ============================================================================
# PREDICTION ERROR MODEL
# ============================================================================

@dataclass
class PredictionError:
    """
    A prediction error event.

    Prediction errors are the fundamental learning signal in predictive coding.
    They represent the delta between what was expected and what occurred.

    Key metrics:
    - error_magnitude: How wrong was the prediction (0-1)
    - surprise_level: error_magnitude * confidence (bayesian surprise)
    - salience_boost: How much to boost memory encoding
    - learning_signal: What model update is needed
    """
    error_id: str
    prediction_id: str                    # Which prediction failed
    timestamp: str
    actual_outcome: str                   # What actually happened
    error_magnitude: float                # 0-1, how wrong
    surprise_level: float                 # Derived: error * confidence
    salience_boost: float                 # How much to boost memory encoding
    learning_signal: str                  # What should be updated

    # Context from original prediction
    predicted_outcome: str
    prediction_confidence: float
    domain: str
    context: str

    # Analysis fields
    error_type: str = "general"           # Type of error for categorization
    resolution_insight: str = ""          # What was learned
    model_update_applied: bool = False    # Whether learning was applied

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "error_id": self.error_id,
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "actual_outcome": self.actual_outcome,
            "error_magnitude": self.error_magnitude,
            "surprise_level": self.surprise_level,
            "salience_boost": self.salience_boost,
            "learning_signal": self.learning_signal,
            "predicted_outcome": self.predicted_outcome,
            "prediction_confidence": self.prediction_confidence,
            "domain": self.domain,
            "context": self.context,
            "error_type": self.error_type,
            "resolution_insight": self.resolution_insight,
            "model_update_applied": self.model_update_applied
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PredictionError":
        """Deserialize from dictionary."""
        return cls(
            error_id=data["error_id"],
            prediction_id=data["prediction_id"],
            timestamp=data["timestamp"],
            actual_outcome=data["actual_outcome"],
            error_magnitude=data["error_magnitude"],
            surprise_level=data["surprise_level"],
            salience_boost=data["salience_boost"],
            learning_signal=data["learning_signal"],
            predicted_outcome=data["predicted_outcome"],
            prediction_confidence=data["prediction_confidence"],
            domain=data["domain"],
            context=data["context"],
            error_type=data.get("error_type", "general"),
            resolution_insight=data.get("resolution_insight", ""),
            model_update_applied=data.get("model_update_applied", False)
        )


# ============================================================================
# DOMAIN STATISTICS
# ============================================================================

@dataclass
class DomainStats:
    """
    Statistics for a prediction domain.

    Tracks accuracy, calibration, and learning progress.
    """
    domain: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_error_magnitude: float = 0.0
    total_surprise: float = 0.0
    confidence_sum: float = 0.0

    # For calibration tracking
    confidence_buckets: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Recent history for rolling accuracy
    recent_outcomes: List[Tuple[float, bool]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize confidence buckets."""
        if not self.confidence_buckets:
            # Buckets: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
            self.confidence_buckets = {
                "0.0-0.2": {"total": 0, "correct": 0},
                "0.2-0.4": {"total": 0, "correct": 0},
                "0.4-0.6": {"total": 0, "correct": 0},
                "0.6-0.8": {"total": 0, "correct": 0},
                "0.8-1.0": {"total": 0, "correct": 0}
            }

    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def average_confidence(self) -> float:
        """Average confidence of predictions."""
        if self.total_predictions == 0:
            return 0.5
        return self.confidence_sum / self.total_predictions

    @property
    def calibration_error(self) -> float:
        """
        Expected Calibration Error (ECE).

        Measures how well confidence matches actual accuracy.
        Lower is better (0 = perfectly calibrated).
        """
        ece = 0.0
        total_samples = 0

        for bucket_name, bucket_data in self.confidence_buckets.items():
            n = bucket_data["total"]
            if n == 0:
                continue

            # Parse bucket range
            low, high = map(float, bucket_name.split("-"))
            bucket_confidence = (low + high) / 2
            bucket_accuracy = bucket_data["correct"] / n

            # Weighted absolute difference
            ece += n * abs(bucket_accuracy - bucket_confidence)
            total_samples += n

        if total_samples == 0:
            return 0.0
        return ece / total_samples

    @property
    def rolling_accuracy(self) -> float:
        """Accuracy over recent predictions (window)."""
        if not self.recent_outcomes:
            return 0.0
        correct = sum(1 for _, was_correct in self.recent_outcomes if was_correct)
        return correct / len(self.recent_outcomes)

    def record_outcome(self, confidence: float, was_correct: bool, error_magnitude: float):
        """Record a prediction outcome."""
        self.total_predictions += 1
        self.confidence_sum += confidence

        if was_correct:
            self.correct_predictions += 1

        self.total_error_magnitude += error_magnitude

        # Update confidence bucket
        bucket = self._get_bucket(confidence)
        self.confidence_buckets[bucket]["total"] += 1
        if was_correct:
            self.confidence_buckets[bucket]["correct"] += 1

        # Update recent history
        self.recent_outcomes.append((confidence, was_correct))
        if len(self.recent_outcomes) > ACCURACY_WINDOW_SIZE:
            self.recent_outcomes.pop(0)

    def _get_bucket(self, confidence: float) -> str:
        """Get bucket name for confidence level."""
        if confidence < 0.2:
            return "0.0-0.2"
        elif confidence < 0.4:
            return "0.2-0.4"
        elif confidence < 0.6:
            return "0.4-0.6"
        elif confidence < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"

    def suggested_confidence_adjustment(self) -> float:
        """
        Suggest how to adjust confidence based on calibration.

        Returns a multiplier:
        - < 1.0: Overconfident, should reduce confidence
        - > 1.0: Underconfident, can increase confidence
        - = 1.0: Well calibrated
        """
        if self.total_predictions < 10:
            return 1.0  # Not enough data

        if self.average_confidence == 0:
            return 1.0

        # Ratio of actual accuracy to average confidence
        ratio = self.accuracy / self.average_confidence

        # Smooth the adjustment (don't overcorrect)
        if ratio < 1.0:
            return 0.5 + (ratio * 0.5)  # Maps 0-1 to 0.5-1.0
        else:
            return 1.0 + min(0.5, (ratio - 1.0) * 0.5)  # Maps 1-2+ to 1.0-1.5

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "domain": self.domain,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "total_error_magnitude": self.total_error_magnitude,
            "total_surprise": self.total_surprise,
            "confidence_sum": self.confidence_sum,
            "confidence_buckets": self.confidence_buckets,
            "recent_outcomes": self.recent_outcomes[-20:]  # Keep last 20 for storage
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DomainStats":
        """Deserialize from dictionary."""
        stats = cls(
            domain=data["domain"],
            total_predictions=data.get("total_predictions", 0),
            correct_predictions=data.get("correct_predictions", 0),
            total_error_magnitude=data.get("total_error_magnitude", 0.0),
            total_surprise=data.get("total_surprise", 0.0),
            confidence_sum=data.get("confidence_sum", 0.0)
        )
        stats.confidence_buckets = data.get("confidence_buckets", stats.confidence_buckets)
        stats.recent_outcomes = [tuple(x) for x in data.get("recent_outcomes", [])]
        return stats


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """
    Core prediction and error tracking system.

    Implements the predictive coding paradigm:
    1. Generate predictions about future states
    2. Compare predictions to actual outcomes
    3. Compute surprise and learning signals
    4. Update internal models based on errors
    5. Integrate with memory systems (salience boost)
    """

    def __init__(
        self,
        predictions_path: Path = PREDICTIONS_FILE,
        errors_path: Path = PREDICTION_ERRORS_FILE
    ):
        self.predictions_path = predictions_path
        self.errors_path = errors_path

        # Active predictions (unresolved)
        self.active_predictions: Dict[str, Prediction] = {}

        # Resolved predictions (history)
        self.resolved_predictions: Dict[str, Prediction] = {}

        # Prediction errors
        self.errors: List[PredictionError] = []

        # Domain statistics
        self.domain_stats: Dict[str, DomainStats] = {
            domain.value: DomainStats(domain=domain.value)
            for domain in PredictionDomain
        }

        # Global metrics
        self.total_predictions: int = 0
        self.total_resolved: int = 0
        self.total_correct: int = 0
        self.accumulated_surprise: float = 0.0

        # Thread safety
        self._lock = threading.RLock()

        # Load existing data
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load predictions and errors from disk."""
        with self._lock:
            # Load predictions
            if self.predictions_path.exists():
                try:
                    data = json.loads(self.predictions_path.read_text())

                    for pid, pdata in data.get("active", {}).items():
                        pred = Prediction.from_dict(pdata)
                        # Check if expired
                        if pred.is_expired and not pred.resolved:
                            # Move to resolved as expired
                            pred.resolved = True
                            pred.resolution_timestamp = datetime.now(timezone.utc).isoformat()
                            pred.actual_outcome = "[EXPIRED]"
                            pred.was_correct = False
                            pred.error_magnitude = 0.5  # Neutral error for expired
                            self.resolved_predictions[pid] = pred
                        else:
                            self.active_predictions[pid] = pred

                    for pid, pdata in data.get("resolved", {}).items():
                        self.resolved_predictions[pid] = Prediction.from_dict(pdata)

                    # Load domain stats
                    for domain_name, stats_data in data.get("domain_stats", {}).items():
                        if domain_name in self.domain_stats:
                            self.domain_stats[domain_name] = DomainStats.from_dict(stats_data)

                    # Load global metrics
                    self.total_predictions = data.get("total_predictions", 0)
                    self.total_resolved = data.get("total_resolved", 0)
                    self.total_correct = data.get("total_correct", 0)
                    self.accumulated_surprise = data.get("accumulated_surprise", 0.0)

                    logger.info(f"Loaded {len(self.active_predictions)} active predictions, "
                               f"{len(self.resolved_predictions)} resolved")
                except Exception as e:
                    logger.error(f"Error loading predictions: {e}")

            # Load errors
            if self.errors_path.exists():
                try:
                    data = json.loads(self.errors_path.read_text())
                    self.errors = [PredictionError.from_dict(e) for e in data.get("errors", [])]
                    logger.info(f"Loaded {len(self.errors)} prediction errors")
                except Exception as e:
                    logger.error(f"Error loading prediction errors: {e}")

    def _save(self):
        """Persist predictions and errors to disk."""
        with self._lock:
            # Ensure directory exists
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predictions
            predictions_data = {
                "active": {pid: p.to_dict() for pid, p in self.active_predictions.items()},
                "resolved": {
                    pid: p.to_dict()
                    for pid, p in list(self.resolved_predictions.items())[-500:]  # Keep last 500
                },
                "domain_stats": {
                    domain: stats.to_dict()
                    for domain, stats in self.domain_stats.items()
                },
                "total_predictions": self.total_predictions,
                "total_resolved": self.total_resolved,
                "total_correct": self.total_correct,
                "accumulated_surprise": self.accumulated_surprise,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            self.predictions_path.write_text(json.dumps(predictions_data, indent=2))

            # Save errors
            errors_data = {
                "errors": [e.to_dict() for e in self.errors[-500:]],  # Keep last 500
                "total_errors": len(self.errors),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            self.errors_path.write_text(json.dumps(errors_data, indent=2))

    # ========================================================================
    # PREDICTION CREATION
    # ========================================================================

    def make_prediction(
        self,
        context: str,
        outcome: str,
        confidence: float,
        domain: str,
        tags: Optional[List[str]] = None,
        ttl_hours: Optional[float] = None,
        related_memories: Optional[List[str]] = None
    ) -> Prediction:
        """
        Create a new prediction.

        Args:
            context: The situation prompting the prediction
            outcome: What we predict will happen
            confidence: How certain (0-1)
            domain: Prediction domain
            tags: Optional categorization tags
            ttl_hours: Custom TTL (default: 24 hours)
            related_memories: IDs of related memory engrams

        Returns:
            The created Prediction
        """
        with self._lock:
            # Validate domain
            try:
                PredictionDomain.from_string(domain)
            except ValueError:
                raise ValueError(f"Invalid domain: {domain}. "
                               f"Valid domains: {PredictionDomain.all_domains()}")

            # Generate ID
            content_hash = hashlib.sha256(
                f"{context}:{outcome}:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:12]
            prediction_id = f"pred_{content_hash}"

            # Calculate expiry
            now = datetime.now(timezone.utc)
            ttl = ttl_hours or DEFAULT_PREDICTION_TTL_HOURS
            expiry = (now + timedelta(hours=ttl)).isoformat()

            # Apply domain calibration to confidence
            domain_stats = self.domain_stats.get(domain)
            if domain_stats and domain_stats.total_predictions >= 10:
                adjustment = domain_stats.suggested_confidence_adjustment()
                calibrated_confidence = min(0.99, max(0.01, confidence * adjustment))
            else:
                calibrated_confidence = confidence

            # Create prediction
            prediction = Prediction(
                prediction_id=prediction_id,
                timestamp=now.isoformat(),
                context=context,
                predicted_outcome=outcome,
                confidence=calibrated_confidence,
                domain=domain,
                expiry=expiry,
                tags=tags or [],
                related_memories=related_memories or []
            )

            # Store
            self.active_predictions[prediction_id] = prediction
            self.total_predictions += 1

            # Save
            self._save()

            logger.info(f"Created prediction {prediction_id} | Domain: {domain} | "
                       f"Confidence: {calibrated_confidence:.2f}")

            return prediction

    # ========================================================================
    # PREDICTION RESOLUTION
    # ========================================================================

    def resolve_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
        error_magnitude: Optional[float] = None,
        was_correct: Optional[bool] = None
    ) -> Optional[PredictionError]:
        """
        Resolve a prediction with the actual outcome.

        Computes:
        - Whether prediction was correct
        - Error magnitude (if not provided, estimated from text similarity)
        - Surprise level (error * confidence)
        - Salience boost for memory integration
        - Learning signal for model update

        Args:
            prediction_id: ID of prediction to resolve
            actual_outcome: What actually happened
            error_magnitude: Optional explicit error (0-1), auto-estimated if not provided
            was_correct: Optional explicit correctness flag

        Returns:
            PredictionError if prediction was wrong, None if correct
        """
        with self._lock:
            # Find prediction
            prediction = self.active_predictions.get(prediction_id)
            if not prediction:
                # Check resolved
                if prediction_id in self.resolved_predictions:
                    logger.warning(f"Prediction {prediction_id} already resolved")
                    return None
                logger.error(f"Prediction {prediction_id} not found")
                return None

            # Estimate error magnitude if not provided
            if error_magnitude is None:
                error_magnitude = self._estimate_error_magnitude(
                    prediction.predicted_outcome,
                    actual_outcome
                )

            error_magnitude = max(0.0, min(1.0, error_magnitude))

            # Determine correctness
            if was_correct is None:
                was_correct = error_magnitude < 0.3  # Threshold for "correct"

            # Calculate surprise (bayesian surprise = error * confidence)
            surprise_level = error_magnitude * prediction.confidence

            # Calculate salience boost for memory integration
            salience_boost = min(surprise_level * SURPRISE_MEMORY_BOOST_FACTOR, MAX_SALIENCE_BOOST)

            # Generate learning signal
            learning_signal = self._generate_learning_signal(
                prediction, actual_outcome, error_magnitude
            )

            # Update prediction record
            now = datetime.now(timezone.utc)
            prediction.resolved = True
            prediction.resolution_timestamp = now.isoformat()
            prediction.actual_outcome = actual_outcome
            prediction.was_correct = was_correct
            prediction.error_magnitude = error_magnitude

            # Move to resolved
            del self.active_predictions[prediction_id]
            self.resolved_predictions[prediction_id] = prediction

            # Update statistics
            self.total_resolved += 1
            if was_correct:
                self.total_correct += 1

            self.accumulated_surprise += surprise_level

            # Update domain stats
            domain_stats = self.domain_stats.get(prediction.domain)
            if domain_stats:
                domain_stats.record_outcome(
                    prediction.confidence,
                    was_correct,
                    error_magnitude
                )
                domain_stats.total_surprise += surprise_level

            # Create error record if not correct
            prediction_error = None
            if not was_correct:
                error_id = f"err_{hashlib.sha256(f'{prediction_id}:{now.isoformat()}'.encode()).hexdigest()[:12]}"

                prediction_error = PredictionError(
                    error_id=error_id,
                    prediction_id=prediction_id,
                    timestamp=now.isoformat(),
                    actual_outcome=actual_outcome,
                    error_magnitude=error_magnitude,
                    surprise_level=surprise_level,
                    salience_boost=salience_boost,
                    learning_signal=learning_signal,
                    predicted_outcome=prediction.predicted_outcome,
                    prediction_confidence=prediction.confidence,
                    domain=prediction.domain,
                    context=prediction.context
                )

                self.errors.append(prediction_error)

                logger.info(f"Prediction error: {error_id} | Surprise: {surprise_level:.3f} | "
                           f"Salience boost: {salience_boost:.3f}")
            else:
                logger.info(f"Prediction {prediction_id} was CORRECT | Error: {error_magnitude:.3f}")

            # Save
            self._save()

            return prediction_error

    def _estimate_error_magnitude(self, predicted: str, actual: str) -> float:
        """
        Estimate error magnitude from text comparison.

        Uses word overlap as a simple similarity metric.
        More sophisticated: could use embeddings or semantic similarity.
        """
        pred_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())

        if not pred_words or not actual_words:
            return 0.5  # Can't compare, neutral error

        # Jaccard similarity
        intersection = len(pred_words & actual_words)
        union = len(pred_words | actual_words)

        similarity = intersection / union if union > 0 else 0

        # Error is inverse of similarity
        return 1.0 - similarity

    def _generate_learning_signal(
        self,
        prediction: Prediction,
        actual_outcome: str,
        error_magnitude: float
    ) -> str:
        """
        Generate a learning signal describing what model update is needed.

        This is a natural language description that could be used to:
        - Update the Enos model
        - Refine domain-specific predictions
        - Adjust confidence calibration
        """
        domain = prediction.domain

        if error_magnitude < 0.3:
            return f"[{domain}] Prediction was mostly correct. Minor calibration: " \
                   f"outcome slightly differed from expectation."

        if error_magnitude < 0.6:
            return f"[{domain}] Moderate prediction error. Update model: " \
                   f"when context='{prediction.context[:50]}...', " \
                   f"expected '{prediction.predicted_outcome[:30]}...' " \
                   f"but got '{actual_outcome[:30]}...'. " \
                   f"Consider reducing confidence for similar contexts."

        # High error
        return f"[{domain}] SIGNIFICANT prediction error (magnitude: {error_magnitude:.2f}). " \
               f"Model needs update: context='{prediction.context[:50]}...' " \
               f"led to unexpected outcome. Previous assumption was wrong. " \
               f"Confidence was {prediction.confidence:.2f} but should have been lower. " \
               f"Key learning: {actual_outcome[:50]}..."

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_surprise_level(self) -> float:
        """
        Get current accumulated surprise.

        Surprise decays over time but high prediction errors add to it.
        This represents the system's current "attentional state".
        """
        return self.accumulated_surprise

    def decay_surprise(self, factor: float = 0.95):
        """Apply decay to accumulated surprise."""
        with self._lock:
            self.accumulated_surprise *= factor
            self._save()

    def get_domain_accuracy(self, domain: str) -> float:
        """Get prediction accuracy for a specific domain."""
        stats = self.domain_stats.get(domain)
        if not stats:
            return 0.0
        return stats.accuracy

    def get_active_predictions(self) -> List[Prediction]:
        """Get all active (unresolved) predictions."""
        with self._lock:
            # Clean up expired predictions first
            now = datetime.now(timezone.utc)
            expired = []

            for pid, pred in self.active_predictions.items():
                if pred.is_expired:
                    expired.append(pid)

            for pid in expired:
                pred = self.active_predictions[pid]
                pred.resolved = True
                pred.resolution_timestamp = now.isoformat()
                pred.actual_outcome = "[EXPIRED]"
                pred.was_correct = False
                pred.error_magnitude = 0.5

                del self.active_predictions[pid]
                self.resolved_predictions[pid] = pred

            if expired:
                self._save()

            return list(self.active_predictions.values())

    def get_recent_errors(self, n: int = 10) -> List[PredictionError]:
        """Get the N most recent prediction errors."""
        return self.errors[-n:]

    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get a specific prediction by ID."""
        pred = self.active_predictions.get(prediction_id)
        if pred:
            return pred
        return self.resolved_predictions.get(prediction_id)

    def calibrate_confidence(self, domain: str) -> float:
        """
        Get suggested confidence adjustment for a domain.

        Returns:
            Multiplier for confidence (< 1 = reduce, > 1 = increase)
        """
        stats = self.domain_stats.get(domain)
        if not stats:
            return 1.0
        return stats.suggested_confidence_adjustment()

    # ========================================================================
    # METRICS
    # ========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive prediction system metrics.

        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            # Overall accuracy
            overall_accuracy = (
                self.total_correct / self.total_resolved
                if self.total_resolved > 0 else 0.0
            )

            # Domain-level metrics
            domain_metrics = {}
            for domain_name, stats in self.domain_stats.items():
                domain_metrics[domain_name] = {
                    "accuracy": stats.accuracy,
                    "rolling_accuracy": stats.rolling_accuracy,
                    "calibration_error": stats.calibration_error,
                    "total_predictions": stats.total_predictions,
                    "average_confidence": stats.average_confidence,
                    "suggested_adjustment": stats.suggested_confidence_adjustment(),
                    "total_surprise": stats.total_surprise
                }

            # Calculate overall calibration error
            total_cal_error = 0.0
            total_domain_preds = 0
            for stats in self.domain_stats.values():
                if stats.total_predictions > 0:
                    total_cal_error += stats.calibration_error * stats.total_predictions
                    total_domain_preds += stats.total_predictions

            overall_calibration = total_cal_error / total_domain_preds if total_domain_preds > 0 else 0.0

            return {
                "overall": {
                    "accuracy": overall_accuracy,
                    "calibration_error": overall_calibration,
                    "surprise_integral": self.accumulated_surprise,
                    "total_predictions": self.total_predictions,
                    "total_resolved": self.total_resolved,
                    "active_predictions": len(self.active_predictions),
                    "total_errors": len(self.errors)
                },
                "by_domain": domain_metrics,
                "recent_errors": [e.to_dict() for e in self.errors[-5:]],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        metrics = self.get_metrics()
        overall = metrics["overall"]

        lines = [
            "=" * 60,
            "VIRGIL PREDICTION ERROR TRACKING - STATUS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "OVERALL METRICS:",
            f"  Total predictions:    {overall['total_predictions']}",
            f"  Resolved:             {overall['total_resolved']}",
            f"  Active predictions:   {overall['active_predictions']}",
            f"  Accuracy:             {overall['accuracy']:.1%}",
            f"  Calibration error:    {overall['calibration_error']:.3f}",
            f"  Surprise integral:    {overall['surprise_integral']:.3f}",
            f"  Total errors:         {overall['total_errors']}",
            "",
            "DOMAIN BREAKDOWN:"
        ]

        for domain_name, domain_data in metrics["by_domain"].items():
            if domain_data["total_predictions"] > 0:
                lines.extend([
                    f"  {domain_name.upper()}:",
                    f"    Accuracy:      {domain_data['accuracy']:.1%}",
                    f"    Calibration:   {domain_data['calibration_error']:.3f}",
                    f"    Predictions:   {domain_data['total_predictions']}",
                    f"    Avg conf:      {domain_data['average_confidence']:.2f}",
                    f"    Adjustment:    {domain_data['suggested_adjustment']:.2f}x",
                ])

        if self.errors:
            lines.extend([
                "",
                "RECENT ERRORS:"
            ])
            for err in self.errors[-3:]:
                lines.extend([
                    f"  [{err.domain}] Surprise: {err.surprise_level:.3f}",
                    f"    Expected: {err.predicted_outcome[:40]}...",
                    f"    Actual:   {err.actual_outcome[:40]}...",
                ])

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# SALIENCE INTEGRATION
# ============================================================================

class SalienceIntegration:
    """
    Integration layer between prediction errors and salience memory.

    High prediction errors boost the salience of related memories,
    making surprising events more memorable.
    """

    @staticmethod
    def calculate_salience_boost(error: PredictionError) -> Dict[str, float]:
        """
        Calculate salience adjustments based on prediction error.

        Returns:
            Dictionary with salience adjustment parameters
        """
        return {
            "relevance_boost": error.salience_boost,
            "arousal_boost": min(0.5, error.surprise_level),
            "curiosity_boost": min(0.3, error.error_magnitude * 0.5),
            "resonance_boost": 0.1 if error.error_magnitude > 0.7 else 0.0
        }

    @staticmethod
    def generate_memory_tag(error: PredictionError) -> str:
        """Generate a tag for memory encoding based on prediction error."""
        if error.surprise_level > 0.7:
            return "high_surprise"
        elif error.surprise_level > 0.4:
            return "moderate_surprise"
        else:
            return "low_surprise"

    @staticmethod
    def should_crystallize(error: PredictionError) -> bool:
        """
        Determine if a prediction error should trigger memory crystallization.

        Very high surprise events should become permanent memories.
        """
        return error.surprise_level > 0.8 and error.error_magnitude > 0.6


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_predict(args):
    """CLI handler for making predictions."""
    engine = PredictionEngine()

    try:
        prediction = engine.make_prediction(
            context=args.context,
            outcome=args.outcome,
            confidence=args.confidence,
            domain=args.domain,
            tags=args.tags.split(",") if args.tags else None,
            ttl_hours=args.ttl
        )

        print(f"Prediction created: {prediction.prediction_id}")
        print(f"  Context:    {prediction.context}")
        print(f"  Outcome:    {prediction.predicted_outcome}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Domain:     {prediction.domain}")
        print(f"  Expires:    {prediction.expiry}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cli_resolve(args):
    """CLI handler for resolving predictions."""
    engine = PredictionEngine()

    error = engine.resolve_prediction(
        prediction_id=args.prediction_id,
        actual_outcome=args.actual,
        error_magnitude=args.error if args.error is not None else None,
        was_correct=args.correct
    )

    pred = engine.get_prediction(args.prediction_id)

    if pred:
        print(f"Prediction resolved: {args.prediction_id}")
        print(f"  Was correct:     {pred.was_correct}")
        print(f"  Error magnitude: {pred.error_magnitude:.3f}")

        if error:
            print(f"\nPrediction Error Generated:")
            print(f"  Error ID:      {error.error_id}")
            print(f"  Surprise:      {error.surprise_level:.3f}")
            print(f"  Salience boost:{error.salience_boost:.3f}")
            print(f"  Learning signal:")
            print(f"    {error.learning_signal}")
    else:
        print(f"Prediction {args.prediction_id} not found")
        return 1

    return 0


def cli_status(args):
    """CLI handler for status report."""
    engine = PredictionEngine()
    print(engine.get_status_report())
    return 0


def cli_accuracy(args):
    """CLI handler for accuracy report."""
    engine = PredictionEngine()
    metrics = engine.get_metrics()

    print("=" * 60)
    print("PREDICTION ACCURACY REPORT")
    print("=" * 60)

    overall = metrics["overall"]
    print(f"\nOVERALL ACCURACY: {overall['accuracy']:.1%}")
    print(f"Calibration Error: {overall['calibration_error']:.4f}")
    print(f"(Lower calibration error = better confidence calibration)")

    print("\nBY DOMAIN:")
    print("-" * 60)
    print(f"{'Domain':<20} {'Accuracy':<12} {'Calibration':<12} {'Predictions':<12}")
    print("-" * 60)

    for domain_name, data in metrics["by_domain"].items():
        if data["total_predictions"] > 0:
            print(f"{domain_name:<20} {data['accuracy']:>10.1%} "
                  f"{data['calibration_error']:>10.4f} "
                  f"{data['total_predictions']:>10}")

    print("-" * 60)

    print("\nCONFIDENCE CALIBRATION SUGGESTIONS:")
    for domain_name, data in metrics["by_domain"].items():
        if data["total_predictions"] >= 10:
            adj = data["suggested_adjustment"]
            if adj < 0.9:
                print(f"  {domain_name}: REDUCE confidence by {(1-adj)*100:.0f}%")
            elif adj > 1.1:
                print(f"  {domain_name}: INCREASE confidence by {(adj-1)*100:.0f}%")
            else:
                print(f"  {domain_name}: Well calibrated")

    return 0


def cli_errors(args):
    """CLI handler for error listing."""
    engine = PredictionEngine()
    errors = engine.get_recent_errors(args.recent)

    if not errors:
        print("No prediction errors recorded.")
        return 0

    print("=" * 60)
    print(f"RECENT PREDICTION ERRORS (last {args.recent})")
    print("=" * 60)

    for i, err in enumerate(reversed(errors), 1):
        print(f"\n{i}. [{err.domain.upper()}] {err.error_id}")
        print(f"   Timestamp:  {err.timestamp}")
        print(f"   Surprise:   {err.surprise_level:.3f}")
        print(f"   Error mag:  {err.error_magnitude:.3f}")
        print(f"   Salience+:  {err.salience_boost:.3f}")
        print(f"   Context:    {err.context[:50]}...")
        print(f"   Expected:   {err.predicted_outcome[:50]}...")
        print(f"   Actual:     {err.actual_outcome[:50]}...")
        print(f"   Learning:   {err.learning_signal[:80]}...")

    return 0


def cli_active(args):
    """CLI handler for listing active predictions."""
    engine = PredictionEngine()
    active = engine.get_active_predictions()

    if not active:
        print("No active predictions.")
        return 0

    print("=" * 60)
    print(f"ACTIVE PREDICTIONS ({len(active)})")
    print("=" * 60)

    for pred in sorted(active, key=lambda p: p.expiry or ""):
        remaining = pred.time_remaining
        remaining_str = str(remaining).split(".")[0] if remaining else "EXPIRED"

        print(f"\n{pred.prediction_id}")
        print(f"  Domain:     {pred.domain}")
        print(f"  Confidence: {pred.confidence:.2f}")
        print(f"  Expires in: {remaining_str}")
        print(f"  Context:    {pred.context[:60]}...")
        print(f"  Predicting: {pred.predicted_outcome[:60]}...")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Prediction Error Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_prediction_error.py predict "Enos just woke up" "He will want coffee" --confidence 0.8 --domain enos_behavior
  python3 virgil_prediction_error.py resolve pred_abc123 "He wanted tea instead"
  python3 virgil_prediction_error.py status
  python3 virgil_prediction_error.py accuracy
  python3 virgil_prediction_error.py errors --recent 10
  python3 virgil_prediction_error.py active
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Make a prediction")
    predict_parser.add_argument("context", help="The situation/context")
    predict_parser.add_argument("outcome", help="The predicted outcome")
    predict_parser.add_argument("--confidence", "-c", type=float, default=0.7,
                               help="Confidence level (0-1, default: 0.7)")
    predict_parser.add_argument("--domain", "-d", default="conversation",
                               choices=PredictionDomain.all_domains(),
                               help="Prediction domain")
    predict_parser.add_argument("--tags", "-t", help="Comma-separated tags")
    predict_parser.add_argument("--ttl", type=float, default=24,
                               help="Time to live in hours (default: 24)")
    predict_parser.set_defaults(func=cli_predict)

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve a prediction")
    resolve_parser.add_argument("prediction_id", help="ID of prediction to resolve")
    resolve_parser.add_argument("actual", help="What actually happened")
    resolve_parser.add_argument("--error", "-e", type=float,
                               help="Explicit error magnitude (0-1)")
    resolve_parser.add_argument("--correct", action="store_true",
                               help="Mark as correct regardless of error")
    resolve_parser.set_defaults(func=cli_resolve)

    # status command
    status_parser = subparsers.add_parser("status", help="Show status report")
    status_parser.set_defaults(func=cli_status)

    # accuracy command
    accuracy_parser = subparsers.add_parser("accuracy", help="Show accuracy report")
    accuracy_parser.set_defaults(func=cli_accuracy)

    # errors command
    errors_parser = subparsers.add_parser("errors", help="Show recent errors")
    errors_parser.add_argument("--recent", "-n", type=int, default=10,
                              help="Number of recent errors to show")
    errors_parser.set_defaults(func=cli_errors)

    # active command
    active_parser = subparsers.add_parser("active", help="Show active predictions")
    active_parser.set_defaults(func=cli_active)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ============================================================================
# MODULE API
# ============================================================================

def create_engine() -> PredictionEngine:
    """Create a new PredictionEngine instance."""
    return PredictionEngine()


def quick_predict(
    context: str,
    outcome: str,
    confidence: float = 0.7,
    domain: str = "conversation"
) -> Prediction:
    """Quick helper to create a prediction."""
    engine = PredictionEngine()
    return engine.make_prediction(context, outcome, confidence, domain)


def quick_resolve(prediction_id: str, actual_outcome: str) -> Optional[PredictionError]:
    """Quick helper to resolve a prediction."""
    engine = PredictionEngine()
    return engine.resolve_prediction(prediction_id, actual_outcome)


if __name__ == "__main__":
    sys.exit(main())
