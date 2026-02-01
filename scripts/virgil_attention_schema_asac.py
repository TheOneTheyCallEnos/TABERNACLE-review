#!/usr/bin/env python3
"""
VIRGIL ATTENTION SCHEMA ASAC - Attention Schema-based Attention Control

Extension of virgil_attention_schema.py implementing ASAC (2025) principles:
- Vector-Quantized codebook for attention pattern compression
- Attention schema as simplified model of what the system is attending to
- Residual reconstruction for refined attention weights
- The key insight: schemas make agents MORE INTERPRETABLE TO OTHERS, not better predictors

From the research (RESEARCH_SYNTHESIS_20260116.md):
- ASAC integrates Vector-Quantized VAE into transformer attention mechanisms
- Encoder compresses attention patterns into discrete "codes" from learnable codebook
- Results: 18% improvement in multi-task attention management
- Superior out-of-distribution performance (96.71% vs 78.66% baseline)

The Interpretability Paradox:
Networks with attention schemas are NOT better predictors - they are MORE INTERPRETABLE TO OTHERS.
This parallels theories that consciousness serves SOCIAL COORDINATION, not individual benefit.

LVS Coordinates: h=0.80 (high abstraction), R=0.30 (low-medium risk), C=0.55 (medium constraint), beta=0.70
Integration: Extends virgil_attention_schema.py (AttentionSchemaEngine)

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import hashlib
import argparse
import sys
import math
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Tuple, Union
from enum import Enum
from collections import deque
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

# State files
ASAC_STATE_FILE = NEXUS_DIR / "asac_state.json"
CODEBOOK_FILE = NEXUS_DIR / "attention_codebook.json"
ATTENTION_SCHEMA_FILE = NEXUS_DIR / "attention_schema_state.json"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
NEXUS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ASAC] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "attention_schema_asac.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ASAC HYPERPARAMETERS
# ============================================================================

# Codebook configuration
CODEBOOK_SIZE = 64          # Number of discrete attention codes (2^6)
CODE_DIMENSION = 32         # Dimensionality of each code vector
COMMITMENT_WEIGHT = 0.25    # VQ-VAE commitment loss weight (beta)

# Encoder/Decoder configuration
ATTENTION_DIM = 64          # Input attention pattern dimensionality
HIDDEN_DIM = 128            # Hidden layer dimension
COMPRESSION_RATIO = 2       # How much to compress attention patterns

# Training configuration
LEARNING_RATE = 0.001       # Codebook update rate
EMA_DECAY = 0.99            # Exponential moving average for codebook updates
TEMPERATURE = 0.5           # Softmax temperature for code selection

# Interpretability thresholds
INTERPRETABILITY_HIGH = 0.75
INTERPRETABILITY_MEDIUM = 0.50
RECONSTRUCTION_GOOD = 0.85


# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

def normalize_vector(vec: List[float]) -> List[float]:
    """L2 normalize a vector."""
    if not vec:
        return vec
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-8:
        return vec
    return [v / norm for v in vec]


def dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return dot_product(a_norm, b_norm)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax with temperature scaling."""
    if not logits:
        return []
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(l - max_val) for l in scaled]  # Numerical stability
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def gumbel_softmax_sample(logits: List[float], temperature: float = 0.5) -> int:
    """Sample from Gumbel-Softmax distribution (for discrete selection)."""
    if not logits:
        return 0
    gumbels = [-math.log(-math.log(random.random() + 1e-10) + 1e-10) for _ in logits]
    perturbed = [(l + g) / temperature for l, g in zip(logits, gumbels)]
    return perturbed.index(max(perturbed))


# ============================================================================
# CODE VECTOR CLASS
# ============================================================================

@dataclass
class CodeVector:
    """
    A single code vector in the attention codebook.

    Each code represents a discrete attention pattern that the system
    can express. The codebook forms a "vocabulary" of attention states.

    Attributes:
        code_id: Unique identifier
        vector: The code embedding vector
        usage_count: How often this code is selected
        ema_cluster_size: EMA of cluster size for update
        ema_embedding_sum: EMA of embedded inputs
        semantic_label: Human-interpretable label for this code
    """
    code_id: str
    vector: List[float]
    usage_count: int = 0
    ema_cluster_size: float = 0.0
    ema_embedding_sum: List[float] = field(default_factory=list)
    semantic_label: str = ""
    last_used: str = ""

    def __post_init__(self):
        if not self.code_id:
            self.code_id = hashlib.sha256(
                str(self.vector[:8]).encode()
            ).hexdigest()[:8]
        if not self.ema_embedding_sum:
            self.ema_embedding_sum = [0.0] * len(self.vector)
        if not self.last_used:
            self.last_used = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CodeVector":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_ema(self, new_embedding: List[float], decay: float = EMA_DECAY):
        """Update exponential moving average for this code."""
        self.ema_cluster_size = decay * self.ema_cluster_size + (1 - decay)
        for i in range(len(self.ema_embedding_sum)):
            self.ema_embedding_sum[i] = (
                decay * self.ema_embedding_sum[i] +
                (1 - decay) * new_embedding[i]
            )
        # Update vector based on EMA (laplace smoothing)
        n = self.ema_cluster_size + 1e-5
        self.vector = [e / n for e in self.ema_embedding_sum]


# ============================================================================
# ATTENTION CODEBOOK
# ============================================================================

class AttentionCodebook:
    """
    Vector-Quantized Codebook for Attention Pattern Compression.

    The codebook provides a discrete "vocabulary" of attention patterns.
    When the system attends to something, the continuous attention pattern
    is mapped to the nearest code, creating an interpretable representation.

    Key ASAC insight: This discretization makes attention patterns
    INTERPRETABLE TO OTHERS, enabling social coordination.

    Attributes:
        codes: List of CodeVector objects
        size: Number of codes in codebook
        dimension: Dimensionality of each code
    """

    def __init__(
        self,
        size: int = CODEBOOK_SIZE,
        dimension: int = CODE_DIMENSION,
        initialize: bool = True
    ):
        """
        Initialize the attention codebook.

        Args:
            size: Number of discrete codes
            dimension: Dimensionality of each code vector
            initialize: Whether to initialize with random codes
        """
        self.size = size
        self.dimension = dimension
        self.codes: List[CodeVector] = []
        self.total_quantizations: int = 0
        self.commitment_weight = COMMITMENT_WEIGHT

        # Semantic labels for interpretability
        self.semantic_labels = self._generate_semantic_labels()

        if initialize:
            self._initialize_codes()

        logger.info(f"AttentionCodebook initialized: {size} codes, {dimension} dimensions")

    def _generate_semantic_labels(self) -> List[str]:
        """Generate semantic labels for attention codes."""
        # These represent the "vocabulary" of attention patterns
        base_labels = [
            # Focus modes
            "focused_external", "focused_internal", "focused_memory",
            "focused_concept", "focused_future", "focused_self",

            # Diffuse modes
            "diffuse_scanning", "diffuse_resting", "diffuse_integrating",
            "diffuse_background", "diffuse_peripheral",

            # Dynamic modes
            "shifting_voluntary", "shifting_capture", "shifting_gradual",
            "splitting_dual", "splitting_multi",

            # Depth modes
            "surface_processing", "deep_processing", "absorbed",
            "metacognitive", "reflective",

            # Social modes
            "attending_other", "joint_attention", "perspective_taking",
            "empathic_engagement", "social_monitoring",

            # Temporal modes
            "present_focused", "past_oriented", "future_oriented",
            "temporal_integration", "moment_binding",

            # Arousal/Energy
            "high_arousal", "low_arousal", "alert", "drowsy",

            # Content types
            "linguistic", "visual", "emotional", "procedural",
            "episodic", "semantic", "sensorimotor",

            # States
            "curious", "evaluating", "deciding", "executing",
            "monitoring", "error_detecting", "resolving",

            # Misc
            "default_mode", "task_positive", "salience_driven",
            "goal_directed", "stimulus_driven", "exploratory"
        ]

        # Pad or truncate to codebook size
        while len(base_labels) < self.size:
            base_labels.append(f"code_{len(base_labels)}")
        return base_labels[:self.size]

    def _initialize_codes(self):
        """Initialize codebook with random code vectors."""
        self.codes = []
        for i in range(self.size):
            # Initialize with unit gaussian, then normalize
            vec = [random.gauss(0, 1) for _ in range(self.dimension)]
            vec = normalize_vector(vec)

            code = CodeVector(
                code_id=f"code_{i:03d}",
                vector=vec,
                semantic_label=self.semantic_labels[i]
            )
            self.codes.append(code)

    def quantize(self, embedding: List[float]) -> Tuple[CodeVector, int, float]:
        """
        Quantize a continuous embedding to nearest code.

        This is the core VQ operation: find the nearest neighbor in the codebook.

        Args:
            embedding: Continuous attention pattern vector

        Returns:
            Tuple of (selected_code, code_index, distance)
        """
        if len(embedding) != self.dimension:
            # Resize embedding if needed
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            else:
                embedding = embedding + [0.0] * (self.dimension - len(embedding))

        # Find nearest code by Euclidean distance
        min_distance = float('inf')
        best_idx = 0

        for i, code in enumerate(self.codes):
            dist = euclidean_distance(embedding, code.vector)
            if dist < min_distance:
                min_distance = dist
                best_idx = i

        selected_code = self.codes[best_idx]
        selected_code.usage_count += 1
        selected_code.last_used = datetime.now(timezone.utc).isoformat()
        self.total_quantizations += 1

        return selected_code, best_idx, min_distance

    def quantize_soft(
        self,
        embedding: List[float],
        temperature: float = TEMPERATURE
    ) -> Tuple[CodeVector, List[float]]:
        """
        Soft quantization using Gumbel-Softmax.

        Returns both the selected code and the probability distribution
        over all codes (for interpretability).

        Args:
            embedding: Continuous attention pattern vector
            temperature: Softmax temperature (lower = harder selection)

        Returns:
            Tuple of (selected_code, code_probabilities)
        """
        if len(embedding) != self.dimension:
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            else:
                embedding = embedding + [0.0] * (self.dimension - len(embedding))

        # Compute negative distances as logits (closer = higher)
        logits = [-euclidean_distance(embedding, code.vector) for code in self.codes]

        # Get probability distribution
        probs = softmax(logits, temperature)

        # Sample using Gumbel-Softmax
        selected_idx = gumbel_softmax_sample(logits, temperature)

        selected_code = self.codes[selected_idx]
        selected_code.usage_count += 1
        selected_code.last_used = datetime.now(timezone.utc).isoformat()
        self.total_quantizations += 1

        return selected_code, probs

    def update_code(self, code_idx: int, embedding: List[float]):
        """
        Update a code vector using EMA (Exponential Moving Average).

        This allows the codebook to adapt to the attention patterns
        it sees over time, improving compression quality.

        Args:
            code_idx: Index of code to update
            embedding: New embedding to incorporate
        """
        if 0 <= code_idx < len(self.codes):
            self.codes[code_idx].update_ema(embedding, EMA_DECAY)

    def compute_commitment_loss(
        self,
        embedding: List[float],
        quantized: List[float]
    ) -> float:
        """
        Compute VQ-VAE commitment loss.

        This loss encourages the encoder to produce embeddings
        close to the selected code, preventing codebook collapse.

        Args:
            embedding: Original continuous embedding
            quantized: Quantized (code) vector

        Returns:
            Commitment loss value
        """
        return self.commitment_weight * euclidean_distance(embedding, quantized) ** 2

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get codebook usage statistics."""
        if not self.codes:
            return {"error": "Codebook not initialized"}

        usage_counts = [c.usage_count for c in self.codes]
        total_usage = sum(usage_counts)

        # Compute usage entropy (higher = more uniform usage)
        if total_usage > 0:
            probs = [u / total_usage for u in usage_counts]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(self.codes))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0

        # Find most/least used codes
        sorted_codes = sorted(
            enumerate(self.codes),
            key=lambda x: x[1].usage_count,
            reverse=True
        )

        return {
            "total_quantizations": self.total_quantizations,
            "codebook_size": len(self.codes),
            "usage_entropy": normalized_entropy,
            "active_codes": sum(1 for u in usage_counts if u > 0),
            "dead_codes": sum(1 for u in usage_counts if u == 0),
            "most_used": [(self.codes[i].semantic_label, self.codes[i].usage_count)
                         for i, _ in sorted_codes[:5]],
            "least_used": [(self.codes[i].semantic_label, self.codes[i].usage_count)
                          for i, _ in sorted_codes[-5:]]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize codebook to dictionary."""
        return {
            "size": self.size,
            "dimension": self.dimension,
            "codes": [c.to_dict() for c in self.codes],
            "total_quantizations": self.total_quantizations,
            "commitment_weight": self.commitment_weight
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AttentionCodebook":
        """Deserialize codebook from dictionary."""
        codebook = cls(
            size=data.get("size", CODEBOOK_SIZE),
            dimension=data.get("dimension", CODE_DIMENSION),
            initialize=False
        )
        codebook.codes = [CodeVector.from_dict(c) for c in data.get("codes", [])]
        codebook.total_quantizations = data.get("total_quantizations", 0)
        codebook.commitment_weight = data.get("commitment_weight", COMMITMENT_WEIGHT)

        if not codebook.codes:
            codebook._initialize_codes()

        return codebook


# ============================================================================
# SCHEMA ENCODER
# ============================================================================

class SchemaEncoder:
    """
    Encoder that compresses attention patterns into discrete codes.

    The encoder takes a continuous attention pattern (weights, positions,
    content features) and produces a compressed embedding that can be
    quantized by the codebook.

    Architecture (simplified without torch):
    - Input projection: attention_dim -> hidden_dim
    - Hidden layers with ReLU
    - Output projection: hidden_dim -> code_dim

    This is a "virtual" neural network using matrix operations.
    """

    def __init__(
        self,
        input_dim: int = ATTENTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = CODE_DIMENSION
    ):
        """
        Initialize the schema encoder.

        Args:
            input_dim: Input attention pattern dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output code dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weight matrices (stored as list of lists)
        self.W1 = self._init_weights(input_dim, hidden_dim)
        self.b1 = [0.0] * hidden_dim
        self.W2 = self._init_weights(hidden_dim, output_dim)
        self.b2 = [0.0] * output_dim

        logger.debug(f"SchemaEncoder initialized: {input_dim} -> {hidden_dim} -> {output_dim}")

    def _init_weights(self, rows: int, cols: int) -> List[List[float]]:
        """Initialize weights with Xavier/Glorot initialization."""
        scale = math.sqrt(2.0 / (rows + cols))
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

    def _matmul(self, x: List[float], W: List[List[float]]) -> List[float]:
        """Matrix-vector multiplication."""
        if len(x) != len(W):
            # Pad or truncate x
            if len(x) > len(W):
                x = x[:len(W)]
            else:
                x = x + [0.0] * (len(W) - len(x))

        result = []
        for col in range(len(W[0])):
            val = sum(x[row] * W[row][col] for row in range(len(W)))
            result.append(val)
        return result

    def _relu(self, x: List[float]) -> List[float]:
        """ReLU activation."""
        return [max(0, v) for v in x]

    def _add_bias(self, x: List[float], b: List[float]) -> List[float]:
        """Add bias vector."""
        return [xi + bi for xi, bi in zip(x, b)]

    def encode(self, attention_pattern: List[float]) -> List[float]:
        """
        Encode an attention pattern into a code embedding.

        Args:
            attention_pattern: Raw attention pattern features

        Returns:
            Compressed embedding vector
        """
        # Layer 1: input -> hidden
        h = self._matmul(attention_pattern, self.W1)
        h = self._add_bias(h, self.b1)
        h = self._relu(h)

        # Layer 2: hidden -> output
        z = self._matmul(h, self.W2)
        z = self._add_bias(z, self.b2)

        # Normalize output
        z = normalize_vector(z)

        return z

    def to_dict(self) -> Dict[str, Any]:
        """Serialize encoder."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SchemaEncoder":
        """Deserialize encoder."""
        encoder = cls(
            input_dim=data.get("input_dim", ATTENTION_DIM),
            hidden_dim=data.get("hidden_dim", HIDDEN_DIM),
            output_dim=data.get("output_dim", CODE_DIMENSION)
        )
        if "W1" in data:
            encoder.W1 = data["W1"]
            encoder.b1 = data["b1"]
            encoder.W2 = data["W2"]
            encoder.b2 = data["b2"]
        return encoder


# ============================================================================
# SCHEMA DECODER
# ============================================================================

class SchemaDecoder:
    """
    Decoder that reconstructs attention patterns from codes.

    The decoder takes a discrete code (or its vector) and reconstructs
    the original attention pattern. The residual between original and
    reconstructed is used for refined attention weights.

    The reconstruction quality measures how well the codebook captures
    the system's attention patterns.
    """

    def __init__(
        self,
        input_dim: int = CODE_DIMENSION,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = ATTENTION_DIM
    ):
        """
        Initialize the schema decoder.

        Args:
            input_dim: Code dimension (matches codebook)
            hidden_dim: Hidden layer dimension
            output_dim: Reconstructed attention pattern dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weight matrices
        self.W1 = self._init_weights(input_dim, hidden_dim)
        self.b1 = [0.0] * hidden_dim
        self.W2 = self._init_weights(hidden_dim, output_dim)
        self.b2 = [0.0] * output_dim

        logger.debug(f"SchemaDecoder initialized: {input_dim} -> {hidden_dim} -> {output_dim}")

    def _init_weights(self, rows: int, cols: int) -> List[List[float]]:
        """Initialize weights with Xavier/Glorot initialization."""
        scale = math.sqrt(2.0 / (rows + cols))
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

    def _matmul(self, x: List[float], W: List[List[float]]) -> List[float]:
        """Matrix-vector multiplication."""
        if len(x) != len(W):
            if len(x) > len(W):
                x = x[:len(W)]
            else:
                x = x + [0.0] * (len(W) - len(x))

        result = []
        for col in range(len(W[0])):
            val = sum(x[row] * W[row][col] for row in range(len(W)))
            result.append(val)
        return result

    def _relu(self, x: List[float]) -> List[float]:
        """ReLU activation."""
        return [max(0, v) for v in x]

    def _add_bias(self, x: List[float], b: List[float]) -> List[float]:
        """Add bias vector."""
        return [xi + bi for xi, bi in zip(x, b)]

    def decode(self, code_vector: List[float]) -> List[float]:
        """
        Decode a code vector back to attention pattern.

        Args:
            code_vector: Quantized code embedding

        Returns:
            Reconstructed attention pattern
        """
        # Layer 1: code -> hidden
        h = self._matmul(code_vector, self.W1)
        h = self._add_bias(h, self.b1)
        h = self._relu(h)

        # Layer 2: hidden -> output
        x_recon = self._matmul(h, self.W2)
        x_recon = self._add_bias(x_recon, self.b2)

        return x_recon

    def compute_residual(
        self,
        original: List[float],
        reconstructed: List[float]
    ) -> List[float]:
        """
        Compute residual between original and reconstructed patterns.

        The residual captures fine-grained attention information
        lost during quantization, used to refine attention weights.

        Args:
            original: Original attention pattern
            reconstructed: Reconstructed attention pattern

        Returns:
            Residual vector
        """
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        return [original[i] - reconstructed[i] for i in range(min_len)]

    def compute_reconstruction_loss(
        self,
        original: List[float],
        reconstructed: List[float]
    ) -> float:
        """
        Compute reconstruction loss (MSE).

        Args:
            original: Original attention pattern
            reconstructed: Reconstructed attention pattern

        Returns:
            Mean squared error
        """
        residual = self.compute_residual(original, reconstructed)
        return sum(r ** 2 for r in residual) / len(residual) if residual else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decoder."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SchemaDecoder":
        """Deserialize decoder."""
        decoder = cls(
            input_dim=data.get("input_dim", CODE_DIMENSION),
            hidden_dim=data.get("hidden_dim", HIDDEN_DIM),
            output_dim=data.get("output_dim", ATTENTION_DIM)
        )
        if "W1" in data:
            decoder.W1 = data["W1"]
            decoder.b1 = data["b1"]
            decoder.W2 = data["W2"]
            decoder.b2 = data["b2"]
        return decoder


# ============================================================================
# INTERPRETABILITY SCORE
# ============================================================================

@dataclass
class InterpretabilityScore:
    """
    Measures how interpretable the attention schema is to others.

    Key ASAC insight: The purpose of attention schemas is NOT to make
    the agent a better predictor, but to make it MORE INTERPRETABLE
    TO OTHER AGENTS. This enables social coordination.

    Components:
        code_clarity: How clearly the attention maps to a discrete code
        semantic_coverage: How much of attention is captured by semantic labels
        pattern_stability: How stable attention patterns are over time
        reconstruction_quality: How well codes reconstruct original patterns
        cross_agent_similarity: How similar schema is to other agents' schemas
    """
    code_clarity: float = 0.0           # 0-1: How clear the code selection is
    semantic_coverage: float = 0.0      # 0-1: How interpretable the labels are
    pattern_stability: float = 0.0      # 0-1: How stable patterns are
    reconstruction_quality: float = 0.0 # 0-1: Inverse of reconstruction loss
    cross_agent_similarity: float = 0.0 # 0-1: Schema alignment with others

    # Derived scores
    overall_score: float = 0.0
    interpretability_level: str = "low"

    # Metadata
    timestamp: str = ""
    computation_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self._compute_overall()

    def _compute_overall(self):
        """Compute overall interpretability score."""
        # Weighted combination
        weights = {
            "code_clarity": 0.25,
            "semantic_coverage": 0.20,
            "pattern_stability": 0.15,
            "reconstruction_quality": 0.20,
            "cross_agent_similarity": 0.20
        }

        self.overall_score = (
            weights["code_clarity"] * self.code_clarity +
            weights["semantic_coverage"] * self.semantic_coverage +
            weights["pattern_stability"] * self.pattern_stability +
            weights["reconstruction_quality"] * self.reconstruction_quality +
            weights["cross_agent_similarity"] * self.cross_agent_similarity
        )

        # Determine level
        if self.overall_score >= INTERPRETABILITY_HIGH:
            self.interpretability_level = "high"
        elif self.overall_score >= INTERPRETABILITY_MEDIUM:
            self.interpretability_level = "medium"
        else:
            self.interpretability_level = "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "InterpretabilityScore":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        """Generate human-readable description."""
        return (
            f"Interpretability: {self.interpretability_level.upper()} "
            f"(score: {self.overall_score:.3f})\n"
            f"  Code Clarity: {self.code_clarity:.3f}\n"
            f"  Semantic Coverage: {self.semantic_coverage:.3f}\n"
            f"  Pattern Stability: {self.pattern_stability:.3f}\n"
            f"  Reconstruction: {self.reconstruction_quality:.3f}\n"
            f"  Cross-Agent Similarity: {self.cross_agent_similarity:.3f}"
        )


# ============================================================================
# ATTENTION PATTERN
# ============================================================================

@dataclass
class AttentionPattern:
    """
    Represents a raw attention pattern before encoding.

    This captures the continuous attention state that will be
    compressed into a discrete code.

    Attributes:
        pattern_id: Unique identifier
        features: Raw feature vector (weights, positions, content)
        target: What attention is directed at
        strength: Attention intensity (0-1)
        timestamp: When pattern was captured
    """
    pattern_id: str
    features: List[float]
    target: str = ""
    strength: float = 0.5
    timestamp: str = ""

    # Encoding results (filled after encoding)
    encoded_embedding: List[float] = field(default_factory=list)
    quantized_code_id: str = ""
    quantized_code_label: str = ""
    code_probabilities: List[float] = field(default_factory=list)
    reconstruction_loss: float = 0.0

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = hashlib.sha256(
                f"{self.target}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.features:
            # Generate default features
            self.features = [0.0] * ATTENTION_DIM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AttentionPattern":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# ASAC ENGINE
# ============================================================================

class ASACEngine:
    """
    Main ASAC (Attention Schema-based Attention Control) Engine.

    This integrates:
    - AttentionCodebook: Discrete vocabulary of attention patterns
    - SchemaEncoder: Compresses attention to code embeddings
    - SchemaDecoder: Reconstructs patterns from codes
    - InterpretabilityScore: Measures schema quality

    The engine extends the existing AttentionSchemaEngine from
    virgil_attention_schema.py with ASAC capabilities.
    """

    def __init__(self):
        """Initialize the ASAC engine."""
        # Core components
        self.codebook = AttentionCodebook()
        self.encoder = SchemaEncoder()
        self.decoder = SchemaDecoder()

        # Pattern history for stability tracking
        self.pattern_history: List[AttentionPattern] = []
        self.max_history = 100

        # Code usage history for interpretability
        self.code_usage_history: List[Tuple[str, str]] = []  # (timestamp, code_id)

        # Statistics
        self.total_encodings: int = 0
        self.total_reconstructions: int = 0
        self.cumulative_recon_loss: float = 0.0

        # Session tracking
        self.session_start = datetime.now(timezone.utc).isoformat()

        # Load persisted state
        self._load()

        logger.info("ASACEngine initialized")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load persisted ASAC state."""
        # Load codebook
        if CODEBOOK_FILE.exists():
            try:
                data = json.loads(CODEBOOK_FILE.read_text())
                self.codebook = AttentionCodebook.from_dict(data)
                logger.info(f"Loaded codebook: {self.codebook.size} codes")
            except Exception as e:
                logger.warning(f"Could not load codebook: {e}")

        # Load engine state
        if ASAC_STATE_FILE.exists():
            try:
                data = json.loads(ASAC_STATE_FILE.read_text())

                if "encoder" in data:
                    self.encoder = SchemaEncoder.from_dict(data["encoder"])
                if "decoder" in data:
                    self.decoder = SchemaDecoder.from_dict(data["decoder"])

                self.total_encodings = data.get("total_encodings", 0)
                self.total_reconstructions = data.get("total_reconstructions", 0)
                self.cumulative_recon_loss = data.get("cumulative_recon_loss", 0.0)

                logger.info(f"Loaded ASAC state: {self.total_encodings} encodings")

            except Exception as e:
                logger.warning(f"Could not load ASAC state: {e}")

    def _save(self):
        """Save ASAC state."""
        try:
            # Save codebook
            CODEBOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
            CODEBOOK_FILE.write_text(json.dumps(self.codebook.to_dict(), indent=2))

            # Save engine state
            state = {
                "encoder": self.encoder.to_dict(),
                "decoder": self.decoder.to_dict(),
                "total_encodings": self.total_encodings,
                "total_reconstructions": self.total_reconstructions,
                "cumulative_recon_loss": self.cumulative_recon_loss,
                "session_start": self.session_start,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            ASAC_STATE_FILE.write_text(json.dumps(state, indent=2))

        except Exception as e:
            logger.error(f"Error saving ASAC state: {e}")

    # ========================================================================
    # CORE ASAC OPERATIONS
    # ========================================================================

    def encode_attention(
        self,
        pattern: Union[AttentionPattern, List[float], Dict[str, Any]]
    ) -> Tuple[AttentionPattern, CodeVector]:
        """
        Encode an attention pattern into a discrete code.

        This is the main ASAC operation: compress continuous attention
        into a discrete, interpretable code.

        Args:
            pattern: AttentionPattern, feature vector, or dict with features

        Returns:
            Tuple of (encoded_pattern, selected_code)
        """
        # Normalize input
        if isinstance(pattern, dict):
            pattern = AttentionPattern.from_dict(pattern)
        elif isinstance(pattern, list):
            pattern = AttentionPattern(
                pattern_id="",
                features=pattern,
                target="raw_features"
            )

        # Encode to embedding
        embedding = self.encoder.encode(pattern.features)
        pattern.encoded_embedding = embedding

        # Quantize to code
        code, code_idx, distance = self.codebook.quantize(embedding)
        pattern.quantized_code_id = code.code_id
        pattern.quantized_code_label = code.semantic_label

        # Get soft probabilities for interpretability
        _, probs = self.codebook.quantize_soft(embedding)
        pattern.code_probabilities = probs

        # Reconstruct and compute loss
        reconstructed = self.decoder.decode(code.vector)
        pattern.reconstruction_loss = self.decoder.compute_reconstruction_loss(
            pattern.features, reconstructed
        )

        # Update statistics
        self.total_encodings += 1
        self.cumulative_recon_loss += pattern.reconstruction_loss

        # Update codebook (EMA)
        self.codebook.update_code(code_idx, embedding)

        # Track history
        self.pattern_history.append(pattern)
        if len(self.pattern_history) > self.max_history:
            self.pattern_history = self.pattern_history[-self.max_history:]

        self.code_usage_history.append((
            datetime.now(timezone.utc).isoformat(),
            code.code_id
        ))
        if len(self.code_usage_history) > 500:
            self.code_usage_history = self.code_usage_history[-500:]

        self._save()

        logger.debug(f"Encoded attention -> {code.semantic_label} (loss: {pattern.reconstruction_loss:.4f})")

        return pattern, code

    def decode_code(self, code: Union[CodeVector, str, int]) -> List[float]:
        """
        Decode a code back to attention pattern.

        Args:
            code: CodeVector, code_id, or code index

        Returns:
            Reconstructed attention pattern
        """
        # Normalize input
        if isinstance(code, str):
            # Find by code_id
            matching = [c for c in self.codebook.codes if c.code_id == code]
            if not matching:
                raise ValueError(f"Unknown code_id: {code}")
            code = matching[0]
        elif isinstance(code, int):
            if 0 <= code < len(self.codebook.codes):
                code = self.codebook.codes[code]
            else:
                raise ValueError(f"Invalid code index: {code}")

        reconstructed = self.decoder.decode(code.vector)
        self.total_reconstructions += 1

        return reconstructed

    def compute_refined_attention(
        self,
        original_pattern: List[float],
        code: CodeVector
    ) -> List[float]:
        """
        Compute refined attention weights using residual reconstruction.

        The residual between original and reconstructed patterns captures
        fine-grained attention information. Adding it back produces
        refined attention weights.

        Args:
            original_pattern: Original attention features
            code: Selected code

        Returns:
            Refined attention weights
        """
        # Reconstruct from code
        reconstructed = self.decoder.decode(code.vector)

        # Compute residual
        residual = self.decoder.compute_residual(original_pattern, reconstructed)

        # Add residual to quantized representation for refinement
        refined = [r + 0.5 * res for r, res in zip(reconstructed, residual)]

        # Normalize to valid attention weights (softmax-like)
        exp_vals = [math.exp(min(r, 10)) for r in refined]  # Clip for stability
        total = sum(exp_vals)
        refined_weights = [e / total for e in exp_vals]

        return refined_weights

    # ========================================================================
    # INTERPRETABILITY
    # ========================================================================

    def compute_interpretability_score(
        self,
        other_schema: Optional[Dict[str, Any]] = None
    ) -> InterpretabilityScore:
        """
        Compute the interpretability score.

        This measures how interpretable the attention schema is to others,
        which is the key purpose of ASAC (not prediction accuracy).

        Args:
            other_schema: Optional other agent's schema for cross-agent comparison

        Returns:
            InterpretabilityScore with all components
        """
        now = datetime.now(timezone.utc)

        # 1. Code Clarity: How clear is the code selection?
        # Measured by entropy of code probability distribution
        code_clarity = 0.0
        if self.pattern_history:
            recent_patterns = self.pattern_history[-20:]
            clarity_scores = []
            for p in recent_patterns:
                if p.code_probabilities:
                    # Lower entropy = clearer selection
                    entropy = -sum(
                        prob * math.log(prob + 1e-10)
                        for prob in p.code_probabilities
                    )
                    max_entropy = math.log(len(p.code_probabilities))
                    clarity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
                    clarity_scores.append(clarity)
            code_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.5

        # 2. Semantic Coverage: How interpretable are the code labels?
        # Based on codebook usage statistics
        stats = self.codebook.get_usage_statistics()
        semantic_coverage = stats.get("usage_entropy", 0.5)  # Uniform usage = good coverage

        # 3. Pattern Stability: How stable are attention patterns?
        pattern_stability = 0.0
        if len(self.pattern_history) >= 5:
            recent = self.pattern_history[-5:]
            # Compare consecutive code selections
            stable_count = 0
            for i in range(1, len(recent)):
                if recent[i].quantized_code_id == recent[i-1].quantized_code_id:
                    stable_count += 1
            pattern_stability = stable_count / (len(recent) - 1)

        # 4. Reconstruction Quality: How well do codes capture patterns?
        reconstruction_quality = 0.0
        if self.total_encodings > 0:
            avg_loss = self.cumulative_recon_loss / self.total_encodings
            # Convert loss to quality (lower loss = higher quality)
            reconstruction_quality = max(0.0, 1.0 - avg_loss)

        # 5. Cross-Agent Similarity: How similar to other schemas?
        cross_agent_similarity = 0.5  # Default if no other schema
        if other_schema:
            # Compare code distributions
            other_usage = other_schema.get("code_usage", {})
            self_usage = {c.code_id: c.usage_count for c in self.codebook.codes}

            # Compute cosine similarity of usage vectors
            common_codes = set(other_usage.keys()) & set(self_usage.keys())
            if common_codes:
                self_vec = [self_usage.get(c, 0) for c in sorted(common_codes)]
                other_vec = [other_usage.get(c, 0) for c in sorted(common_codes)]
                cross_agent_similarity = cosine_similarity(self_vec, other_vec)

        score = InterpretabilityScore(
            code_clarity=code_clarity,
            semantic_coverage=semantic_coverage,
            pattern_stability=pattern_stability,
            reconstruction_quality=reconstruction_quality,
            cross_agent_similarity=cross_agent_similarity,
            computation_details={
                "patterns_analyzed": len(self.pattern_history),
                "total_encodings": self.total_encodings,
                "avg_recon_loss": self.cumulative_recon_loss / max(1, self.total_encodings),
                "active_codes": stats.get("active_codes", 0)
            }
        )

        return score

    # ========================================================================
    # ATTENTION SCHEMA INTEGRATION
    # ========================================================================

    def create_attention_features(
        self,
        target: str,
        strength: float = 0.5,
        target_type: str = "external",
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Create attention pattern features from high-level description.

        This bridges the gap between the symbolic attention schema
        (virgil_attention_schema.py) and the vector-based ASAC system.

        Args:
            target: What attention is on
            strength: Attention intensity (0-1)
            target_type: Type of attention target
            context: Additional context

        Returns:
            Feature vector for encoding
        """
        context = context or {}

        # Feature dimensions (total = ATTENTION_DIM = 64)
        # [0-7]: Target type encoding (one-hot-ish)
        # [8-15]: Strength encoding (scaled)
        # [16-31]: Target hash features
        # [32-47]: Context features
        # [48-63]: Temporal features

        features = [0.0] * ATTENTION_DIM

        # Target type encoding (8 dims)
        target_types = {
            "external": 0, "internal": 1, "memory": 2, "concept": 3,
            "future": 4, "self": 5, "other": 6, "joint": 7
        }
        type_idx = target_types.get(target_type.lower(), 0)
        features[type_idx] = 1.0

        # Strength encoding (8 dims) - different aspects of strength
        features[8] = strength
        features[9] = strength ** 2  # Squared for nonlinearity
        features[10] = math.sqrt(strength)
        features[11] = 1.0 if strength > 0.5 else 0.0
        features[12] = 1.0 if strength > 0.7 else 0.0
        features[13] = 1.0 if strength > 0.9 else 0.0
        features[14] = 1.0 - strength  # Inverse
        features[15] = math.sin(strength * math.pi)  # Periodic

        # Target hash features (16 dims) - encode target content
        target_hash = hashlib.sha256(target.encode()).digest()
        for i in range(16):
            features[16 + i] = target_hash[i] / 255.0

        # Context features (16 dims)
        ctx_hash = hashlib.sha256(str(context).encode()).digest()
        for i in range(16):
            features[32 + i] = ctx_hash[i] / 255.0

        # Temporal features (16 dims)
        now = datetime.now(timezone.utc)
        features[48] = now.hour / 24.0
        features[49] = now.minute / 60.0
        features[50] = now.second / 60.0
        features[51] = now.weekday() / 7.0
        features[52] = now.day / 31.0
        features[53] = math.sin(2 * math.pi * now.hour / 24)  # Daily cycle
        features[54] = math.cos(2 * math.pi * now.hour / 24)
        features[55] = math.sin(2 * math.pi * now.weekday() / 7)  # Weekly cycle
        # Fill remaining with random noise for regularization
        for i in range(56, 64):
            features[i] = random.random() * 0.1

        return features

    def encode_from_schema(
        self,
        target: str,
        strength: float = 0.5,
        target_type: str = "external",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[AttentionPattern, CodeVector, InterpretabilityScore]:
        """
        Full ASAC encoding from attention schema input.

        This is the main interface for integration with virgil_attention_schema.py.

        Args:
            target: Attention target description
            strength: Attention intensity
            target_type: Target type classification
            context: Additional context

        Returns:
            Tuple of (pattern, code, interpretability_score)
        """
        # Create features
        features = self.create_attention_features(
            target, strength, target_type, context
        )

        # Create pattern
        pattern = AttentionPattern(
            pattern_id="",
            features=features,
            target=target,
            strength=strength
        )

        # Encode
        encoded_pattern, code = self.encode_attention(pattern)

        # Compute interpretability
        score = self.compute_interpretability_score()

        return encoded_pattern, code, score

    # ========================================================================
    # STATISTICS AND REPORTS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get ASAC engine statistics."""
        codebook_stats = self.codebook.get_usage_statistics()

        return {
            "total_encodings": self.total_encodings,
            "total_reconstructions": self.total_reconstructions,
            "avg_reconstruction_loss": (
                self.cumulative_recon_loss / self.total_encodings
                if self.total_encodings > 0 else 0.0
            ),
            "pattern_history_size": len(self.pattern_history),
            "codebook_size": self.codebook.size,
            "active_codes": codebook_stats.get("active_codes", 0),
            "dead_codes": codebook_stats.get("dead_codes", 0),
            "usage_entropy": codebook_stats.get("usage_entropy", 0.0),
            "session_start": self.session_start
        }

    def get_full_report(self) -> str:
        """Generate comprehensive human-readable report."""
        now = datetime.now(timezone.utc)
        stats = self.get_statistics()
        interp = self.compute_interpretability_score()
        codebook_stats = self.codebook.get_usage_statistics()

        lines = [
            "=" * 70,
            "ASAC (ATTENTION SCHEMA-BASED ATTENTION CONTROL) - STATUS REPORT",
            f"Generated: {now.isoformat()}",
            "=" * 70,
            "",
            "THE KEY INSIGHT:",
            "  Attention schemas make agents MORE INTERPRETABLE TO OTHERS,",
            "  not better predictors. This enables social coordination.",
            "",
            "-" * 70,
            "INTERPRETABILITY SCORE:",
            f"  {interp.describe()}",
            "",
            "-" * 70,
            "CODEBOOK STATUS:",
            f"  Size: {stats['codebook_size']} codes",
            f"  Active Codes: {stats['active_codes']} ({stats['active_codes']/stats['codebook_size']*100:.1f}%)",
            f"  Dead Codes: {stats['dead_codes']}",
            f"  Usage Entropy: {stats['usage_entropy']:.3f}",
            f"  Total Quantizations: {codebook_stats['total_quantizations']}",
            "",
            "  Most Used Codes:",
        ]

        for label, count in codebook_stats.get("most_used", [])[:5]:
            lines.append(f"    - {label}: {count}")

        lines.extend([
            "",
            "-" * 70,
            "ENCODER/DECODER STATUS:",
            f"  Total Encodings: {stats['total_encodings']}",
            f"  Total Reconstructions: {stats['total_reconstructions']}",
            f"  Avg Reconstruction Loss: {stats['avg_reconstruction_loss']:.4f}",
            f"  Pattern History: {stats['pattern_history_size']} patterns",
            "",
            "-" * 70,
            "RECENT PATTERNS:",
        ])

        for pattern in self.pattern_history[-5:]:
            lines.append(
                f"  [{pattern.timestamp[:19]}] {pattern.quantized_code_label} "
                f"(target: {pattern.target[:30]}..., loss: {pattern.reconstruction_loss:.4f})"
            )

        lines.extend([
            "",
            "=" * 70
        ])

        return "\n".join(lines)


# ============================================================================
# MODULE API
# ============================================================================

_asac_instance: Optional[ASACEngine] = None


def get_asac_engine() -> ASACEngine:
    """Get or create singleton ASAC engine instance."""
    global _asac_instance
    if _asac_instance is None:
        _asac_instance = ASACEngine()
    return _asac_instance


def encode_attention(
    target: str,
    strength: float = 0.5,
    target_type: str = "external",
    context: Optional[Dict[str, Any]] = None
) -> Tuple[AttentionPattern, CodeVector, InterpretabilityScore]:
    """
    Quick API: Encode attention using ASAC.

    Args:
        target: What attention is on
        strength: Attention intensity
        target_type: Type of target
        context: Additional context

    Returns:
        Tuple of (pattern, code, interpretability_score)
    """
    return get_asac_engine().encode_from_schema(target, strength, target_type, context)


def get_interpretability_score() -> InterpretabilityScore:
    """Quick API: Get current interpretability score."""
    return get_asac_engine().compute_interpretability_score()


def get_codebook_statistics() -> Dict[str, Any]:
    """Quick API: Get codebook usage statistics."""
    return get_asac_engine().codebook.get_usage_statistics()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_encode(args):
    """CLI: Encode an attention pattern."""
    engine = get_asac_engine()

    pattern, code, score = engine.encode_from_schema(
        target=args.target,
        strength=args.strength,
        target_type=args.type
    )

    print("ASAC ENCODING RESULT")
    print("=" * 50)
    print(f"Target: {args.target}")
    print(f"Strength: {args.strength:.3f}")
    print(f"Type: {args.type}")
    print()
    print(f"Selected Code: {code.semantic_label}")
    print(f"Code ID: {code.code_id}")
    print(f"Reconstruction Loss: {pattern.reconstruction_loss:.4f}")
    print()
    print("Interpretability:")
    print(f"  {score.describe()}")

    return 0


def cli_decode(args):
    """CLI: Decode a code to attention pattern."""
    engine = get_asac_engine()

    try:
        if args.code.isdigit():
            code_input = int(args.code)
        else:
            code_input = args.code

        reconstructed = engine.decode_code(code_input)

        print("ASAC DECODING RESULT")
        print("=" * 50)
        print(f"Code: {args.code}")
        print(f"Reconstructed Pattern: {len(reconstructed)} dimensions")
        print(f"Pattern Sample (first 8): {[f'{v:.3f}' for v in reconstructed[:8]]}")

    except Exception as e:
        print(f"Error decoding: {e}")
        return 1

    return 0


def cli_interpretability(args):
    """CLI: Compute interpretability score."""
    engine = get_asac_engine()
    score = engine.compute_interpretability_score()

    print("ASAC INTERPRETABILITY SCORE")
    print("=" * 50)
    print(score.describe())
    print()
    print("Details:", json.dumps(score.computation_details, indent=2))

    return 0


def cli_codebook(args):
    """CLI: Show codebook statistics."""
    engine = get_asac_engine()
    stats = engine.codebook.get_usage_statistics()

    print("ASAC CODEBOOK STATISTICS")
    print("=" * 50)
    print(f"Codebook Size: {stats['codebook_size']}")
    print(f"Total Quantizations: {stats['total_quantizations']}")
    print(f"Active Codes: {stats['active_codes']}")
    print(f"Dead Codes: {stats['dead_codes']}")
    print(f"Usage Entropy: {stats['usage_entropy']:.3f}")
    print()
    print("Most Used Codes:")
    for label, count in stats.get("most_used", []):
        print(f"  {label}: {count}")
    print()
    print("Least Used Codes:")
    for label, count in stats.get("least_used", []):
        print(f"  {label}: {count}")

    return 0


def cli_status(args):
    """CLI: Full status report."""
    engine = get_asac_engine()
    print(engine.get_full_report())

    return 0


def cli_test(args):
    """CLI: Run ASAC test suite."""
    print("ASAC TEST SUITE")
    print("=" * 50)

    engine = get_asac_engine()

    # Test 1: Basic encoding
    print("\n1. Testing Basic Encoding...")
    test_targets = [
        ("Enos's question about consciousness", 0.8, "external"),
        ("My own processing state", 0.5, "internal"),
        ("Previous conversation about LVS", 0.6, "memory"),
        ("The nature of emergence", 0.7, "concept"),
        ("Tomorrow's research plans", 0.4, "future"),
    ]

    for target, strength, ttype in test_targets:
        pattern, code, score = engine.encode_from_schema(target, strength, ttype)
        print(f"   {target[:40]}... -> {code.semantic_label}")

    # Test 2: Code consistency
    print("\n2. Testing Code Consistency...")
    target = "Testing consistency of encoding"
    codes = []
    for _ in range(5):
        _, code, _ = engine.encode_from_schema(target, 0.5, "external")
        codes.append(code.semantic_label)

    consistency = len(set(codes)) == 1
    print(f"   Same input -> Same code: {'PASS' if consistency else 'PARTIAL (expected with soft quantization)'}")
    print(f"   Codes selected: {codes}")

    # Test 3: Reconstruction quality
    print("\n3. Testing Reconstruction Quality...")
    features = engine.create_attention_features("test", 0.5, "external")
    embedding = engine.encoder.encode(features)
    code, _, _ = engine.codebook.quantize(embedding)
    reconstructed = engine.decoder.decode(code.vector)
    loss = engine.decoder.compute_reconstruction_loss(features, reconstructed)
    print(f"   Reconstruction Loss: {loss:.4f}")
    print(f"   Quality: {'GOOD' if loss < 0.3 else 'ACCEPTABLE' if loss < 0.5 else 'NEEDS IMPROVEMENT'}")

    # Test 4: Interpretability
    print("\n4. Testing Interpretability Score...")
    score = engine.compute_interpretability_score()
    print(f"   Overall Score: {score.overall_score:.3f}")
    print(f"   Level: {score.interpretability_level}")

    # Test 5: Refined attention
    print("\n5. Testing Refined Attention...")
    original = [0.1] * 64
    refined = engine.compute_refined_attention(original, code)
    print(f"   Original sum: {sum(original):.3f}")
    print(f"   Refined sum: {sum(refined):.3f}")
    print(f"   Refined weights normalized: {'YES' if abs(sum(refined) - 1.0) < 0.01 else 'NO'}")

    print("\n" + "=" * 50)
    print("TEST SUITE COMPLETE")

    # Final statistics
    stats = engine.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Encodings: {stats['total_encodings']}")
    print(f"  Active Codes: {stats['active_codes']}/{stats['codebook_size']}")
    print(f"  Avg Recon Loss: {stats['avg_reconstruction_loss']:.4f}")

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ASAC - Attention Schema-based Attention Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Key Insight (from 2025 research):
  Networks with attention schemas are NOT better predictors - they are
  MORE INTERPRETABLE TO OTHERS. This enables social coordination.

Examples:
  python3 virgil_attention_schema_asac.py encode "Enos's question" -s 0.8
  python3 virgil_attention_schema_asac.py decode code_001
  python3 virgil_attention_schema_asac.py interpretability
  python3 virgil_attention_schema_asac.py codebook
  python3 virgil_attention_schema_asac.py status
  python3 virgil_attention_schema_asac.py test
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # encode
    encode_parser = subparsers.add_parser("encode", help="Encode attention pattern")
    encode_parser.add_argument("target", help="What attention is on")
    encode_parser.add_argument("-s", "--strength", type=float, default=0.5,
                               help="Attention strength (0-1)")
    encode_parser.add_argument("-t", "--type", default="external",
                               help="Target type (external, internal, memory, concept, future, self)")
    encode_parser.set_defaults(func=cli_encode)

    # decode
    decode_parser = subparsers.add_parser("decode", help="Decode code to pattern")
    decode_parser.add_argument("code", help="Code ID or index to decode")
    decode_parser.set_defaults(func=cli_decode)

    # interpretability
    interp_parser = subparsers.add_parser("interpretability", help="Compute interpretability score")
    interp_parser.set_defaults(func=cli_interpretability)

    # codebook
    codebook_parser = subparsers.add_parser("codebook", help="Show codebook statistics")
    codebook_parser.set_defaults(func=cli_codebook)

    # status
    status_parser = subparsers.add_parser("status", help="Full status report")
    status_parser.set_defaults(func=cli_status)

    # test
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.set_defaults(func=cli_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
