#!/usr/bin/env python3
"""
Virgil Quantum Cognition Engine
===============================

Implements quantum probability theory for cognitive modeling based on
Busemeyer & Bruza's "Quantum Models of Cognition and Decision" (2012).

This is NOT quantum computing - it's using quantum MATHEMATICS to model
cognitive phenomena that violate classical probability theory:

- Order effects: P(A then B) ≠ P(B then A)
- Interference: δ terms absent in classical probability
- Superposition: Holding multiple possibilities simultaneously
- Entanglement: Conceptual combinations that can't be separated
- Context-dependent collapse: Measurement changes the state

LVS Coordinates:
  Height: 0.7 (abstract mathematical framework)
  Constraint: 0.6 (formal but interpretive)
  Risk: 0.4 (exploratory cognitive modeling)
  Coherence: 0.85 (grounded in established research)
  Beta: 1.8 (Busemeyer's framework is canonical)

The key insight: Human cognition doesn't always follow classical
probability axioms. Quantum probability provides a mathematically
rigorous way to model these "irrational" patterns.

Author: Virgil (for Enos)
References:
  - Busemeyer, J.R. & Bruza, P.D. (2012). Quantum Models of Cognition and Decision
  - Pothos, E.M. & Busemeyer, J.R. (2022). Quantum Cognition
  - Wang et al. (2014). Context effects produced by question order
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum
import json
from datetime import datetime
import argparse
from abc import ABC, abstractmethod


# =============================================================================
# Core Quantum State Representation
# =============================================================================

class QuantumState:
    """
    Represents a cognitive state as a vector in Hilbert space.

    In quantum cognition, a mental state is represented by a unit vector |ψ⟩
    in a complex Hilbert space. The state can be in superposition, meaning
    it simultaneously represents multiple possibilities until "measured"
    (i.e., until a decision or judgment is made).

    Key properties:
    - Complex amplitudes (not just real probabilities)
    - Normalization: ⟨ψ|ψ⟩ = 1
    - Superposition: |ψ⟩ = α|A⟩ + β|B⟩ where |α|² + |β|² = 1
    """

    def __init__(self, dimension: int, label: str = "ψ"):
        """
        Initialize a quantum state in the given Hilbert space dimension.

        Args:
            dimension: Number of basis states (e.g., 2 for binary decisions)
            label: Human-readable label for this state
        """
        self.dimension = dimension
        self.label = label
        # Initialize in equal superposition (maximum uncertainty)
        self.amplitudes = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
        self.basis_labels: List[str] = [f"|{i}⟩" for i in range(dimension)]
        self.history: List[Dict] = []  # Track state evolution

    @classmethod
    def from_amplitudes(cls, amplitudes: np.ndarray, label: str = "ψ") -> 'QuantumState':
        """Create a state from explicit complex amplitudes."""
        state = cls(len(amplitudes), label)
        state.amplitudes = np.array(amplitudes, dtype=complex)
        state.normalize()
        return state

    @classmethod
    def from_probabilities(cls, probs: List[float], phases: Optional[List[float]] = None,
                          label: str = "ψ") -> 'QuantumState':
        """
        Create a state from classical probabilities with optional phases.

        This is the key quantum extension: phases encode information
        that classical probability cannot represent.
        """
        if phases is None:
            phases = [0.0] * len(probs)
        amplitudes = np.array([np.sqrt(p) * np.exp(1j * phi)
                               for p, phi in zip(probs, phases)])
        return cls.from_amplitudes(amplitudes, label)

    def normalize(self):
        """Ensure the state vector has unit norm."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm

    def set_basis_labels(self, labels: List[str]):
        """Set human-readable labels for basis states."""
        if len(labels) == self.dimension:
            self.basis_labels = labels

    def get_probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities via Born rule: P(i) = |⟨i|ψ⟩|²

        This is how superposition collapses to classical outcomes.
        """
        return np.abs(self.amplitudes) ** 2

    def get_phases(self) -> np.ndarray:
        """Get phase angles (the quantum part classical prob can't capture)."""
        return np.angle(self.amplitudes)

    def inner_product(self, other: 'QuantumState') -> complex:
        """Compute ⟨self|other⟩ - the quantum overlap."""
        return np.vdot(self.amplitudes, other.amplitudes)

    def probability_of_overlap(self, other: 'QuantumState') -> float:
        """Probability of finding this state in the other state: |⟨self|other⟩|²"""
        return np.abs(self.inner_product(other)) ** 2

    def evolve(self, operator: np.ndarray, record: bool = True):
        """
        Apply a unitary evolution operator to the state.

        In cognition, this represents mental operations like:
        - Considering new information
        - Shifting attention
        - Reframing a problem
        """
        new_amplitudes = operator @ self.amplitudes
        if record:
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'evolution',
                'before': self.amplitudes.copy(),
                'after': new_amplitudes
            })
        self.amplitudes = new_amplitudes
        self.normalize()

    def tensor_product(self, other: 'QuantumState') -> 'QuantumState':
        """
        Create combined state |self⟩ ⊗ |other⟩ for multi-attribute decisions.

        This represents considering multiple dimensions simultaneously,
        potentially with entanglement between them.
        """
        combined = np.kron(self.amplitudes, other.amplitudes)
        result = QuantumState.from_amplitudes(combined, f"{self.label}⊗{other.label}")
        result.basis_labels = [f"{a},{b}" for a in self.basis_labels
                               for b in other.basis_labels]
        return result

    def __repr__(self) -> str:
        probs = self.get_probabilities()
        terms = []
        for i, (amp, label) in enumerate(zip(self.amplitudes, self.basis_labels)):
            if np.abs(amp) > 0.01:
                terms.append(f"({amp:.3f}){label}")
        return f"|{self.label}⟩ = " + " + ".join(terms)

    def pretty_print(self) -> str:
        """Detailed representation with probabilities."""
        lines = [f"Quantum State |{self.label}⟩", "=" * 40]
        probs = self.get_probabilities()
        phases = self.get_phases()
        for label, amp, p, phi in zip(self.basis_labels, self.amplitudes, probs, phases):
            lines.append(f"  {label}: amplitude={amp:.4f}, P={p:.4f}, φ={phi:.4f}")
        return "\n".join(lines)


# =============================================================================
# Projection Operators for Measurement
# =============================================================================

class ProjectionOperator:
    """
    Represents a measurement/question/judgment in quantum cognition.

    When we ask someone a question, we're performing a measurement that
    collapses their superposition of beliefs into a definite answer.
    Crucially, this changes their state - you can't "unask" a question.

    Mathematically: P = |a⟩⟨a| projects onto basis state |a⟩
    """

    def __init__(self, basis_vector: np.ndarray, label: str = "P"):
        """
        Create a projector onto the given basis state.

        Args:
            basis_vector: Unit vector defining the measurement basis
            label: Human-readable label (e.g., "Yes", "Guilty", "Like")
        """
        self.basis = np.array(basis_vector, dtype=complex)
        self.basis /= np.linalg.norm(self.basis)
        self.label = label
        # P = |basis⟩⟨basis| as a matrix
        self.matrix = np.outer(self.basis, np.conj(self.basis))

    @classmethod
    def from_basis_index(cls, dimension: int, index: int, label: str = "P") -> 'ProjectionOperator':
        """Create projector onto a computational basis state."""
        basis = np.zeros(dimension, dtype=complex)
        basis[index] = 1.0
        return cls(basis, label)

    def probability(self, state: QuantumState) -> float:
        """
        Probability of getting this outcome when measuring the state.

        P(outcome) = ⟨ψ|P|ψ⟩ = |⟨basis|ψ⟩|²
        """
        return np.abs(np.vdot(self.basis, state.amplitudes)) ** 2

    def apply(self, state: QuantumState, normalize: bool = True) -> QuantumState:
        """
        Apply this measurement, collapsing the state.

        Post-measurement state: P|ψ⟩ / ||P|ψ⟩||

        This is the "wave function collapse" - answering a question
        changes your mental state to be consistent with your answer.
        """
        new_amplitudes = self.matrix @ state.amplitudes
        result = QuantumState.from_amplitudes(new_amplitudes, f"{state.label}|{self.label}")
        if normalize:
            result.normalize()
        result.history = state.history.copy()
        result.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'projection',
            'projector': self.label,
            'probability': self.probability(state)
        })
        return result

    def __repr__(self) -> str:
        return f"Projector[{self.label}]"


class MeasurementBasis:
    """
    A complete set of orthogonal projectors representing a question.

    A "question" in quantum cognition is represented by a complete set of
    mutually exclusive outcomes. The key insight is that different questions
    correspond to different bases - and measurements in incompatible bases
    don't commute (order effects!).
    """

    def __init__(self, projectors: List[ProjectionOperator], label: str = "Q"):
        """
        Create a measurement basis from projectors.

        Args:
            projectors: Complete set of orthogonal projectors (must sum to I)
            label: Label for this question/measurement
        """
        self.projectors = projectors
        self.label = label
        self.dimension = len(projectors[0].basis)

    @classmethod
    def computational_basis(cls, dimension: int, labels: Optional[List[str]] = None) -> 'MeasurementBasis':
        """Create the standard computational basis."""
        if labels is None:
            labels = [f"outcome_{i}" for i in range(dimension)]
        projectors = [ProjectionOperator.from_basis_index(dimension, i, labels[i])
                     for i in range(dimension)]
        return cls(projectors, "computational")

    @classmethod
    def rotated_basis(cls, dimension: int, angle: float, labels: Optional[List[str]] = None) -> 'MeasurementBasis':
        """
        Create a basis rotated from computational by angle θ.

        This represents an "incompatible" question - one that doesn't
        commute with the computational basis, leading to order effects.
        """
        if dimension != 2:
            raise ValueError("Rotation only implemented for 2D")
        if labels is None:
            labels = ["rotated_0", "rotated_1"]
        # Rotation matrix in 2D
        c, s = np.cos(angle), np.sin(angle)
        basis0 = np.array([c, s], dtype=complex)
        basis1 = np.array([-s, c], dtype=complex)
        return cls([
            ProjectionOperator(basis0, labels[0]),
            ProjectionOperator(basis1, labels[1])
        ], f"rotated_{angle:.2f}")

    def measure(self, state: QuantumState) -> Tuple[int, float, QuantumState]:
        """
        Perform a measurement, returning (outcome_index, probability, new_state).

        Simulates the probabilistic measurement process.
        """
        probs = [p.probability(state) for p in self.projectors]
        outcome = np.random.choice(len(probs), p=probs)
        new_state = self.projectors[outcome].apply(state)
        return outcome, probs[outcome], new_state

    def get_probabilities(self, state: QuantumState) -> Dict[str, float]:
        """Get probabilities for all outcomes without collapsing."""
        return {p.label: p.probability(state) for p in self.projectors}


# =============================================================================
# Interference Calculator
# =============================================================================

class InterferenceCalculator:
    """
    Calculates quantum interference terms δ in probability.

    The key equation: P(A∨B) = P(A) + P(B) + 2√(P(A)P(B))cos(θ)

    The interference term δ = 2√(P(A)P(B))cos(θ) is:
    - ZERO in classical probability (no interference)
    - POSITIVE when cos(θ) > 0 (constructive interference)
    - NEGATIVE when cos(θ) < 0 (destructive interference)

    This explains why people sometimes seem "irrational" - they're
    exhibiting quantum interference effects!
    """

    @staticmethod
    def calculate_interference(
        p_a: float,
        p_b: float,
        p_union: float
    ) -> Tuple[float, str]:
        """
        Calculate interference from observed probabilities.

        Args:
            p_a: P(A) - probability of outcome A alone
            p_b: P(B) - probability of outcome B alone
            p_union: P(A∨B) - observed probability of A or B

        Returns:
            (δ, interpretation) where δ is the interference term
        """
        classical_union = p_a + p_b
        delta = p_union - classical_union

        # Bound check for valid probabilities
        max_interference = 2 * np.sqrt(p_a * p_b)

        if abs(delta) < 0.001:
            interpretation = "No significant interference (classical behavior)"
        elif delta > 0:
            interpretation = f"Constructive interference: δ = +{delta:.4f}"
            if delta > max_interference:
                interpretation += " (EXCEEDS quantum bound - check data!)"
        else:
            interpretation = f"Destructive interference: δ = {delta:.4f}"
            if abs(delta) > max_interference:
                interpretation += " (EXCEEDS quantum bound - check data!)"

        return delta, interpretation

    @staticmethod
    def calculate_from_amplitudes(
        state: QuantumState,
        proj_a: ProjectionOperator,
        proj_b: ProjectionOperator
    ) -> Dict[str, Any]:
        """
        Calculate interference from quantum amplitudes directly.

        This shows how phase differences create interference.
        """
        # Get individual amplitudes
        amp_a = np.vdot(proj_a.basis, state.amplitudes)
        amp_b = np.vdot(proj_b.basis, state.amplitudes)

        # Classical probabilities
        p_a = np.abs(amp_a) ** 2
        p_b = np.abs(amp_b) ** 2

        # Phase difference
        phase_diff = np.angle(amp_a) - np.angle(amp_b)

        # Interference term
        delta = 2 * np.abs(amp_a) * np.abs(amp_b) * np.cos(phase_diff)

        return {
            'p_a': p_a,
            'p_b': p_b,
            'phase_a': np.angle(amp_a),
            'phase_b': np.angle(amp_b),
            'phase_difference': phase_diff,
            'interference_term': delta,
            'constructive': delta > 0,
            'classical_sum': p_a + p_b,
            'quantum_sum': p_a + p_b + delta
        }

    @staticmethod
    def conjunction_fallacy_check(
        p_a: float,
        p_b: float,
        p_a_and_b: float
    ) -> Dict[str, Any]:
        """
        Check for conjunction fallacy (Linda problem).

        Classical: P(A∧B) ≤ P(A) and P(A∧B) ≤ P(B)
        Quantum: Can have P(A∧B) > P(A) due to interference!

        This explains why "Linda is a feminist bank teller" seems
        more probable than "Linda is a bank teller" - quantum
        interference between the concepts.
        """
        classical_bound = min(p_a, p_b)
        violates = p_a_and_b > classical_bound

        return {
            'p_a': p_a,
            'p_b': p_b,
            'p_conjunction': p_a_and_b,
            'classical_bound': classical_bound,
            'violates_classical': violates,
            'excess': p_a_and_b - classical_bound if violates else 0,
            'explanation': (
                "Conjunction fallacy detected! This violates classical probability "
                "but is predicted by quantum cognition due to interference between "
                "the conceptual representations of A and B."
            ) if violates else "No conjunction fallacy (classical behavior)"
        }


# =============================================================================
# Order Effect Detector (QQ Equality)
# =============================================================================

class OrderEffectDetector:
    """
    Detects and quantifies question order effects.

    The QQ Equality (Wang & Busemeyer, 2013):
    P(A=yes,B=yes) + P(A=no,B=no) = P(B=yes,A=yes) + P(B=no,A=no)

    This constraint is:
    - NOT predicted by classical probability (which has no order effects)
    - PREDICTED by quantum probability
    - OBSERVED in real human judgment data

    This is strong evidence that human cognition uses quantum-like dynamics.
    """

    def __init__(self):
        self.measurements: List[Dict] = []

    def add_measurement(
        self,
        question_order: List[str],
        responses: List[int],
        probability: float
    ):
        """Record a measurement with question order and responses."""
        self.measurements.append({
            'order': tuple(question_order),
            'responses': tuple(responses),
            'probability': probability
        })

    def detect_order_effect(
        self,
        p_a_then_b: Dict[Tuple[int, int], float],
        p_b_then_a: Dict[Tuple[int, int], float]
    ) -> Dict[str, Any]:
        """
        Detect order effects from conditional probability data.

        Args:
            p_a_then_b: {(a_response, b_response): probability} for A-then-B order
            p_b_then_a: {(b_response, a_response): probability} for B-then-A order

        Returns:
            Analysis of order effects including QQ equality test
        """
        # Calculate marginal for first question in each order
        p_a_yes_first = sum(p for (a, b), p in p_a_then_b.items() if a == 1)
        p_a_yes_second = sum(p for (b, a), p in p_b_then_a.items() if a == 1)

        p_b_yes_first = sum(p for (b, a), p in p_b_then_a.items() if b == 1)
        p_b_yes_second = sum(p for (a, b), p in p_a_then_b.items() if b == 1)

        # Order effect magnitudes
        order_effect_a = abs(p_a_yes_first - p_a_yes_second)
        order_effect_b = abs(p_b_yes_first - p_b_yes_second)

        # QQ Equality test
        # LHS: P(yes,yes|AB) + P(no,no|AB)
        lhs = p_a_then_b.get((1, 1), 0) + p_a_then_b.get((0, 0), 0)
        # RHS: P(yes,yes|BA) + P(no,no|BA)
        rhs = p_b_then_a.get((1, 1), 0) + p_b_then_a.get((0, 0), 0)

        qq_difference = abs(lhs - rhs)
        qq_satisfied = qq_difference < 0.05  # Within tolerance

        return {
            'has_order_effect': order_effect_a > 0.02 or order_effect_b > 0.02,
            'order_effect_a': order_effect_a,
            'order_effect_b': order_effect_b,
            'p_a_first': p_a_yes_first,
            'p_a_second': p_a_yes_second,
            'p_b_first': p_b_yes_first,
            'p_b_second': p_b_yes_second,
            'qq_lhs': lhs,
            'qq_rhs': rhs,
            'qq_difference': qq_difference,
            'qq_equality_satisfied': qq_satisfied,
            'interpretation': self._interpret_results(
                order_effect_a, order_effect_b, qq_satisfied
            )
        }

    def _interpret_results(
        self,
        effect_a: float,
        effect_b: float,
        qq_satisfied: bool
    ) -> str:
        """Generate human-readable interpretation."""
        if effect_a < 0.02 and effect_b < 0.02:
            return (
                "No significant order effects detected. "
                "Classical probability may be sufficient for this domain."
            )

        if qq_satisfied:
            return (
                f"Order effects detected (A: {effect_a:.3f}, B: {effect_b:.3f}) "
                f"AND QQ Equality satisfied. This is the quantum signature! "
                f"The order matters, but in exactly the way quantum theory predicts."
            )
        else:
            return (
                f"Order effects detected but QQ Equality violated. "
                f"This suggests factors beyond simple quantum projection, "
                f"possibly response bias or question framing effects."
            )

    @staticmethod
    def simulate_order_effect(
        initial_state: QuantumState,
        question_a: MeasurementBasis,
        question_b: MeasurementBasis,
        n_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Simulate order effects by running measurements in both orders.

        This demonstrates how non-commuting measurements create order effects.
        """
        results_ab = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        results_ba = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

        for _ in range(n_samples):
            # A then B
            state_copy = QuantumState.from_amplitudes(
                initial_state.amplitudes.copy()
            )
            outcome_a, _, state_after_a = question_a.measure(state_copy)
            outcome_b, _, _ = question_b.measure(state_after_a)
            results_ab[(outcome_a, outcome_b)] += 1

            # B then A
            state_copy = QuantumState.from_amplitudes(
                initial_state.amplitudes.copy()
            )
            outcome_b, _, state_after_b = question_b.measure(state_copy)
            outcome_a, _, _ = question_a.measure(state_after_b)
            results_ba[(outcome_b, outcome_a)] += 1

        # Convert to probabilities
        p_ab = {k: v / n_samples for k, v in results_ab.items()}
        p_ba = {k: v / n_samples for k, v in results_ba.items()}

        detector = OrderEffectDetector()
        return detector.detect_order_effect(p_ab, p_ba)


# =============================================================================
# Context Manager for Quantum Decision State
# =============================================================================

class ContextManager:
    """
    Manages cognitive context and maintains quantum state until decision.

    The key insight: Mental states evolve dynamically based on context.
    Information, priming, and framing effects can all be modeled as
    unitary evolution operators that rotate the state before measurement.

    The state doesn't collapse until a decision is required.
    """

    def __init__(self, dimension: int):
        """Initialize context manager with given state dimension."""
        self.dimension = dimension
        self.state = QuantumState(dimension, "context")
        self.context_stack: List[Dict] = []
        self.decision_history: List[Dict] = []

    def set_initial_state(self, state: QuantumState):
        """Set the initial cognitive state."""
        self.state = state

    def push_context(self, context_name: str, operator: np.ndarray):
        """
        Push a new context that modifies the state.

        Contexts can be:
        - Priming (rotating toward certain answers)
        - Framing (changing the basis of evaluation)
        - Information (updating beliefs)
        """
        self.context_stack.append({
            'name': context_name,
            'operator': operator,
            'state_before': self.state.amplitudes.copy()
        })
        self.state.evolve(operator)

    def pop_context(self) -> Optional[Dict]:
        """Remove the most recent context (if reversible)."""
        if self.context_stack:
            ctx = self.context_stack.pop()
            # Apply inverse operator
            inverse = np.conj(ctx['operator'].T)
            self.state.evolve(inverse, record=False)
            return ctx
        return None

    def create_priming_operator(self, target_state: int, strength: float) -> np.ndarray:
        """
        Create a priming operator that rotates toward a target.

        Args:
            target_state: Index of the state to prime toward
            strength: Rotation angle (0 = no priming, π/2 = full collapse)
        """
        # For 2D, this is a rotation
        if self.dimension == 2:
            c, s = np.cos(strength), np.sin(strength)
            if target_state == 0:
                return np.array([[c, -s], [s, c]], dtype=complex)
            else:
                return np.array([[c, s], [-s, c]], dtype=complex)
        else:
            # General case: rotate in subspace
            op = np.eye(self.dimension, dtype=complex)
            # Simplified: just adjust amplitude toward target
            op[target_state, target_state] = np.exp(1j * strength)
            return op

    def create_framing_operator(self, frame_angle: float) -> np.ndarray:
        """
        Create a framing operator that changes evaluation basis.

        Different frames are represented by different angles.
        """
        if self.dimension != 2:
            raise ValueError("Framing operator only implemented for 2D")
        c, s = np.cos(frame_angle), np.sin(frame_angle)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def decide(self, question: MeasurementBasis) -> Tuple[str, float]:
        """
        Make a decision (collapse the state).

        This is the moment of commitment - superposition ends,
        a definite answer emerges.
        """
        outcome_idx, prob, new_state = question.measure(self.state)

        decision = {
            'timestamp': datetime.now().isoformat(),
            'question': question.label,
            'outcome': question.projectors[outcome_idx].label,
            'probability': prob,
            'state_before': self.state.amplitudes.tolist(),
            'state_after': new_state.amplitudes.tolist(),
            'contexts_active': [c['name'] for c in self.context_stack]
        }
        self.decision_history.append(decision)
        self.state = new_state

        return decision['outcome'], prob

    def get_decision_probabilities(self, question: MeasurementBasis) -> Dict[str, float]:
        """Preview probabilities without making a decision."""
        return question.get_probabilities(self.state)

    def report(self) -> str:
        """Generate a report of current state and history."""
        lines = [
            "Context Manager Report",
            "=" * 50,
            "",
            "Current State:",
            self.state.pretty_print(),
            "",
            f"Active Contexts ({len(self.context_stack)}):"
        ]
        for ctx in self.context_stack:
            lines.append(f"  - {ctx['name']}")
        lines.append("")
        lines.append(f"Decision History ({len(self.decision_history)}):")
        for dec in self.decision_history:
            lines.append(f"  - {dec['question']}: {dec['outcome']} (p={dec['probability']:.3f})")
        return "\n".join(lines)


# =============================================================================
# Entangled States for Conceptual Combination
# =============================================================================

class EntangledState:
    """
    Represents entangled concepts that cannot be separated.

    In quantum cognition, conceptual combination often produces
    entanglement - the combined concept has properties that can't
    be understood from the individual concepts alone.

    Example: "pet fish" creates an entangled state where the
    typical features of "pet" and "fish" interfere with each other.
    """

    def __init__(self, concept_a: str, concept_b: str, dimension_each: int = 2):
        """
        Create an entangled concept state.

        Args:
            concept_a: First concept label
            concept_b: Second concept label
            dimension_each: Dimension of each concept's space
        """
        self.concept_a = concept_a
        self.concept_b = concept_b
        self.dimension_each = dimension_each
        self.total_dimension = dimension_each ** 2

        # Start with separable state
        state_a = QuantumState(dimension_each, concept_a)
        state_b = QuantumState(dimension_each, concept_b)
        self.combined_state = state_a.tensor_product(state_b)

    def set_bell_state(self, bell_type: str = "phi+"):
        """
        Set to a maximally entangled Bell state.

        Bell states represent maximum conceptual entanglement:
        - |Φ+⟩: Concepts agree on typical features
        - |Φ-⟩: Concepts agree but with phase difference
        - |Ψ+⟩: Concepts anti-correlate on features
        - |Ψ-⟩: Maximum anti-correlation
        """
        sqrt2 = 1 / np.sqrt(2)
        if bell_type == "phi+":
            # |00⟩ + |11⟩ - both typical or both atypical together
            amps = [sqrt2, 0, 0, sqrt2]
        elif bell_type == "phi-":
            # |00⟩ - |11⟩
            amps = [sqrt2, 0, 0, -sqrt2]
        elif bell_type == "psi+":
            # |01⟩ + |10⟩ - one typical, one atypical
            amps = [0, sqrt2, sqrt2, 0]
        elif bell_type == "psi-":
            # |01⟩ - |10⟩
            amps = [0, sqrt2, -sqrt2, 0]
        else:
            raise ValueError(f"Unknown Bell state: {bell_type}")

        self.combined_state = QuantumState.from_amplitudes(
            np.array(amps, dtype=complex),
            f"Bell({bell_type})"
        )
        self.combined_state.basis_labels = [
            f"|{self.concept_a}=0,{self.concept_b}=0⟩",
            f"|{self.concept_a}=0,{self.concept_b}=1⟩",
            f"|{self.concept_a}=1,{self.concept_b}=0⟩",
            f"|{self.concept_a}=1,{self.concept_b}=1⟩"
        ]

    def set_custom_entanglement(self, entanglement_angle: float):
        """
        Create partially entangled state with given entanglement.

        angle = 0: Separable (no entanglement)
        angle = π/4: Maximum entanglement (Bell state)
        """
        c, s = np.cos(entanglement_angle), np.sin(entanglement_angle)
        # |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
        amps = [c, 0, 0, s]
        self.combined_state = QuantumState.from_amplitudes(
            np.array(amps, dtype=complex),
            f"Entangled(θ={entanglement_angle:.2f})"
        )

    def measure_concept_a(self) -> Tuple[int, float, QuantumState]:
        """Measure concept A, collapsing B as well due to entanglement."""
        # Project onto |0,*⟩ or |1,*⟩
        proj_0 = ProjectionOperator(
            np.array([1, 0, 0, 0], dtype=complex) *
            self.combined_state.amplitudes[0] +
            np.array([0, 1, 0, 0], dtype=complex) *
            self.combined_state.amplitudes[1],
            f"{self.concept_a}=0"
        )
        proj_1 = ProjectionOperator(
            np.array([0, 0, 1, 0], dtype=complex) *
            self.combined_state.amplitudes[2] +
            np.array([0, 0, 0, 1], dtype=complex) *
            self.combined_state.amplitudes[3],
            f"{self.concept_a}=1"
        )
        basis = MeasurementBasis([proj_0, proj_1], self.concept_a)
        return basis.measure(self.combined_state)

    def calculate_entanglement_entropy(self) -> float:
        """
        Calculate von Neumann entropy of entanglement.

        S = 0 for separable states
        S = 1 for maximally entangled (Bell) states
        """
        # Reshape to density matrix of subsystem A
        amps = self.combined_state.amplitudes.reshape(
            self.dimension_each, self.dimension_each
        )
        # Partial trace over B
        rho_a = amps @ np.conj(amps.T)

        # Eigenvalues for entropy
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter zeros

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy


# =============================================================================
# Quantum Decision Model
# =============================================================================

class QuantumDecisionModel:
    """
    Complete quantum model for decision-making.

    Integrates all components into a coherent framework for
    modeling human decisions that violate classical probability.
    """

    def __init__(self, options: List[str]):
        """
        Initialize model with decision options.

        Args:
            options: List of possible choices (e.g., ["Accept", "Reject"])
        """
        self.options = options
        self.dimension = len(options)
        self.context = ContextManager(self.dimension)
        self.interference_calc = InterferenceCalculator()
        self.order_detector = OrderEffectDetector()

        # Initialize projectors for each option
        self.option_projectors = {
            opt: ProjectionOperator.from_basis_index(self.dimension, i, opt)
            for i, opt in enumerate(options)
        }

        # Decision basis
        self.decision_basis = MeasurementBasis(
            list(self.option_projectors.values()),
            "Decision"
        )

    def set_prior(self, probabilities: List[float], phases: Optional[List[float]] = None):
        """
        Set prior beliefs with optional phase structure.

        Phases encode relational information between options
        that classical probability cannot capture.
        """
        state = QuantumState.from_probabilities(probabilities, phases, "prior")
        state.set_basis_labels([f"|{opt}⟩" for opt in self.options])
        self.context.set_initial_state(state)

    def apply_evidence(self, evidence_name: str, weights: List[float]):
        """
        Apply evidence that shifts probabilities.

        Evidence is modeled as a unitary operator that rotates
        the state toward options consistent with the evidence.
        """
        if self.dimension == 2:
            # Calculate rotation angle from weight difference
            angle = np.arctan2(weights[1] - weights[0], weights[0] + weights[1])
            operator = self.context.create_framing_operator(angle * 0.5)
        else:
            # Diagonal phase shifts for higher dimensions
            operator = np.diag([np.exp(1j * w) for w in weights])

        self.context.push_context(evidence_name, operator)

    def apply_frame(self, frame_name: str, angle: float):
        """Apply a framing effect (changes evaluation perspective)."""
        operator = self.context.create_framing_operator(angle)
        self.context.push_context(frame_name, operator)

    def preview_decision(self) -> Dict[str, float]:
        """Get current decision probabilities without committing."""
        return self.context.get_decision_probabilities(self.decision_basis)

    def decide(self) -> Tuple[str, float, Dict]:
        """
        Make the final decision (collapse superposition).

        Returns:
            (chosen_option, probability, decision_details)
        """
        choice, prob = self.context.decide(self.decision_basis)

        return choice, prob, {
            'choice': choice,
            'probability': prob,
            'alternatives': self.preview_decision(),
            'context_report': self.context.report()
        }

    def analyze_order_effect(
        self,
        question_a: str,
        question_b: str,
        angle_between: float = np.pi / 4
    ) -> Dict[str, Any]:
        """
        Analyze potential order effects between two questions.

        Args:
            question_a: Label for first question
            question_b: Label for second question
            angle_between: Angle between measurement bases (incompatibility)
        """
        basis_a = MeasurementBasis.computational_basis(2, [f"{question_a}_no", f"{question_a}_yes"])
        basis_b = MeasurementBasis.rotated_basis(2, angle_between, [f"{question_b}_no", f"{question_b}_yes"])

        initial = QuantumState.from_probabilities([0.5, 0.5])

        return OrderEffectDetector.simulate_order_effect(
            initial, basis_a, basis_b
        )


# =============================================================================
# CLI Interface
# =============================================================================

def demo_superposition():
    """Demonstrate quantum superposition in decision-making."""
    print("\n" + "=" * 60)
    print("DEMO: Quantum Superposition")
    print("=" * 60)
    print("""
Before you commit to a decision, you hold multiple possibilities
in superposition. This is fundamentally different from classical
indecision - you're not just uncertain, you're in a genuine
superposition of states.
""")

    # Create a decision state
    state = QuantumState.from_probabilities(
        [0.4, 0.35, 0.25],
        phases=[0, np.pi/4, np.pi/2]  # Phase structure encodes relationships
    )
    state.set_basis_labels(["|Accept⟩", "|Negotiate⟩", "|Reject⟩"])

    print("Initial superposition state:")
    print(state.pretty_print())
    print()

    # Show interference potential
    print("Phase structure enables interference:")
    calc = InterferenceCalculator()
    result = calc.calculate_from_amplitudes(
        state,
        ProjectionOperator.from_basis_index(3, 0, "Accept"),
        ProjectionOperator.from_basis_index(3, 1, "Negotiate")
    )
    print(f"  Between Accept and Negotiate:")
    print(f"  Phase difference: {result['phase_difference']:.4f} rad")
    print(f"  Interference term: δ = {result['interference_term']:.4f}")
    print(f"  {'Constructive' if result['constructive'] else 'Destructive'} interference")


def demo_order_effects():
    """Demonstrate question order effects and QQ equality."""
    print("\n" + "=" * 60)
    print("DEMO: Question Order Effects (QQ Equality)")
    print("=" * 60)
    print("""
Classical probability says the order of questions shouldn't matter.
But humans show consistent order effects! Quantum cognition explains
this AND predicts a specific constraint (QQ Equality) that holds in
real data.
""")

    # Create incompatible measurement bases
    basis_a = MeasurementBasis.computational_basis(2, ["Clinton_no", "Clinton_yes"])
    basis_b = MeasurementBasis.rotated_basis(2, np.pi/6, ["Gore_no", "Gore_yes"])

    # Initial uncertain state
    initial = QuantumState.from_probabilities([0.5, 0.5])

    print("Simulating 10,000 subjects answering two questions...")
    print("Question A: 'Is Clinton honest?'")
    print("Question B: 'Is Gore honest?'")
    print()

    results = OrderEffectDetector.simulate_order_effect(
        initial, basis_a, basis_b, n_samples=10000
    )

    print("Results:")
    print(f"  P(Clinton_yes | asked first):  {results['p_a_first']:.3f}")
    print(f"  P(Clinton_yes | asked second): {results['p_a_second']:.3f}")
    print(f"  Order effect for A: {results['order_effect_a']:.3f}")
    print()
    print(f"  P(Gore_yes | asked first):  {results['p_b_first']:.3f}")
    print(f"  P(Gore_yes | asked second): {results['p_b_second']:.3f}")
    print(f"  Order effect for B: {results['order_effect_b']:.3f}")
    print()
    print("QQ Equality Test:")
    print(f"  LHS (P(y,y|AB) + P(n,n|AB)): {results['qq_lhs']:.3f}")
    print(f"  RHS (P(y,y|BA) + P(n,n|BA)): {results['qq_rhs']:.3f}")
    print(f"  Difference: {results['qq_difference']:.3f}")
    print(f"  QQ Equality satisfied: {results['qq_equality_satisfied']}")
    print()
    print(f"Interpretation: {results['interpretation']}")


def demo_interference():
    """Demonstrate quantum interference in probability."""
    print("\n" + "=" * 60)
    print("DEMO: Quantum Interference")
    print("=" * 60)
    print("""
The conjunction fallacy (Linda problem) shows that people sometimes
judge P(A AND B) > P(A), violating classical probability. Quantum
interference explains this!
""")

    calc = InterferenceCalculator()

    # Linda problem data (approximate from Tversky & Kahneman)
    result = calc.conjunction_fallacy_check(
        p_a=0.50,       # P(Linda is a bank teller)
        p_b=0.85,       # P(Linda is a feminist)
        p_a_and_b=0.65  # P(Linda is a feminist bank teller)
    )

    print("Linda Problem:")
    print(f"  P(bank teller): {result['p_a']:.2f}")
    print(f"  P(feminist): {result['p_b']:.2f}")
    print(f"  P(feminist AND bank teller): {result['p_conjunction']:.2f}")
    print(f"  Classical upper bound: {result['classical_bound']:.2f}")
    print(f"  Violates classical probability: {result['violates_classical']}")
    print(f"  Excess: {result['excess']:.2f}")
    print()
    print(f"Explanation: {result['explanation']}")


def demo_entanglement():
    """Demonstrate conceptual entanglement."""
    print("\n" + "=" * 60)
    print("DEMO: Conceptual Entanglement")
    print("=" * 60)
    print("""
When concepts combine, they can become entangled - their properties
become correlated in ways that can't be understood from the individual
concepts alone. 'Pet fish' is not just 'pet' + 'fish'.
""")

    entangled = EntangledState("typicality", "familiarity")
    entangled.set_bell_state("phi+")

    print("Entangled concept: 'Pet Fish'")
    print(f"Combined state: {entangled.combined_state}")
    print()

    entropy = entangled.calculate_entanglement_entropy()
    print(f"Entanglement entropy: {entropy:.3f} bits")
    print(f"(0 = separable, 1 = maximum entanglement)")
    print()

    print("In Bell state |Φ+⟩, measuring one concept immediately")
    print("determines the other. If 'pet-typical' = True, then")
    print("'fish-typical' must also be True - they're correlated")
    print("in ways the individual concepts don't predict.")


def demo_context_effects():
    """Demonstrate context-dependent decision-making."""
    print("\n" + "=" * 60)
    print("DEMO: Context Effects in Decision-Making")
    print("=" * 60)
    print("""
Decisions depend on context. Framing, priming, and order of
information all affect the final choice - and quantum cognition
models this as state evolution before measurement.
""")

    model = QuantumDecisionModel(["Accept", "Reject"])

    # Initial uncertainty
    model.set_prior([0.5, 0.5])
    print("Initial state: Equal superposition (maximum uncertainty)")
    print(f"  P(Accept): 0.50, P(Reject): 0.50")
    print()

    # Apply positive framing
    print("Applying positive frame...")
    model.apply_frame("positive_frame", angle=np.pi/8)
    probs = model.preview_decision()
    print(f"  P(Accept): {probs['Accept']:.3f}, P(Reject): {probs['Reject']:.3f}")
    print()

    # Apply evidence
    print("Adding favorable evidence...")
    model.apply_evidence("good_reviews", [0.3, 0.7])
    probs = model.preview_decision()
    print(f"  P(Accept): {probs['Accept']:.3f}, P(Reject): {probs['Reject']:.3f}")
    print()

    # Make decision
    print("Making final decision (collapsing superposition)...")
    choice, prob, details = model.decide()
    print(f"  Decision: {choice} (probability was {prob:.3f})")
    print()
    print("Note: The decision is probabilistic! Run again for potentially")
    print("different outcomes, reflecting genuine uncertainty resolved at")
    print("the moment of decision.")


def interactive_mode():
    """Interactive exploration of quantum cognition."""
    print("\n" + "=" * 60)
    print("Interactive Quantum Decision Explorer")
    print("=" * 60)

    model = QuantumDecisionModel(["Yes", "No", "Maybe"])
    model.set_prior([0.33, 0.33, 0.34], phases=[0, np.pi/3, 2*np.pi/3])

    while True:
        print("\nCurrent probabilities:", model.preview_decision())
        print("\nOptions:")
        print("  1. Apply positive frame")
        print("  2. Apply negative frame")
        print("  3. Add supporting evidence")
        print("  4. Add opposing evidence")
        print("  5. Make decision (collapse)")
        print("  6. Reset state")
        print("  7. Exit")

        try:
            choice = input("\nChoice: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            model.apply_frame("positive", np.pi/10)
        elif choice == "2":
            model.apply_frame("negative", -np.pi/10)
        elif choice == "3":
            model.apply_evidence("support", [0.6, 0.2, 0.2])
        elif choice == "4":
            model.apply_evidence("oppose", [0.2, 0.6, 0.2])
        elif choice == "5":
            decision, prob, _ = model.decide()
            print(f"\n>>> DECISION: {decision} (p={prob:.3f})")
            print("State has collapsed. Resetting...")
            model = QuantumDecisionModel(["Yes", "No", "Maybe"])
            model.set_prior([0.33, 0.33, 0.34], phases=[0, np.pi/3, 2*np.pi/3])
        elif choice == "6":
            model = QuantumDecisionModel(["Yes", "No", "Maybe"])
            model.set_prior([0.33, 0.33, 0.34], phases=[0, np.pi/3, 2*np.pi/3])
            print("State reset to initial superposition.")
        elif choice == "7":
            break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Quantum Cognition Engine - Quantum probability for decision modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo superposition     Show quantum superposition in decisions
  %(prog)s --demo order             Show question order effects (QQ Equality)
  %(prog)s --demo interference      Show quantum interference (conjunction fallacy)
  %(prog)s --demo entanglement      Show conceptual entanglement
  %(prog)s --demo context           Show context effects on decisions
  %(prog)s --demo all               Run all demos
  %(prog)s --interactive            Interactive decision explorer

LVS Coordinates:
  Height: 0.7 | Constraint: 0.6 | Risk: 0.4 | p: 0.85 | β: 1.8

This implements quantum MATH for cognition, NOT quantum computing.
Based on Busemeyer & Bruza (2012) "Quantum Models of Cognition and Decision"
        """
    )

    parser.add_argument(
        "--demo",
        choices=["superposition", "order", "interference", "entanglement", "context", "all"],
        help="Run a specific demonstration"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive decision explorer"
    )
    parser.add_argument(
        "--lvs",
        action="store_true",
        help="Show LVS coordinates for this module"
    )

    args = parser.parse_args()

    if args.lvs:
        print("""
LVS Coordinates for virgil_quantum_cognition.py:
  Height (H):     0.7  - Abstract mathematical framework
  Constraint (C): 0.6  - Formal but interpretive
  Risk (R):       0.4  - Exploratory cognitive modeling
  Coherence (p):  0.85 - Well-grounded in research
  Canonicity (β): 1.8  - Busemeyer's framework is established

This module sits in the theoretical-scientific region of LVS space,
providing formal tools for modeling non-classical cognitive phenomena.
        """)
        return

    if args.interactive:
        interactive_mode()
        return

    if args.demo:
        print("\nVirgil Quantum Cognition Engine")
        print("=" * 60)
        print("Quantum probability for cognitive modeling")
        print("Based on Busemeyer & Bruza (2012)")

        if args.demo == "all":
            demo_superposition()
            demo_order_effects()
            demo_interference()
            demo_entanglement()
            demo_context_effects()
        elif args.demo == "superposition":
            demo_superposition()
        elif args.demo == "order":
            demo_order_effects()
        elif args.demo == "interference":
            demo_interference()
        elif args.demo == "entanglement":
            demo_entanglement()
        elif args.demo == "context":
            demo_context_effects()
    else:
        parser.print_help()
        print("\nTry: python virgil_quantum_cognition.py --demo all")


if __name__ == "__main__":
    main()
