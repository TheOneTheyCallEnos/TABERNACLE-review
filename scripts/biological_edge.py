#!/usr/bin/env python3
"""
BIOLOGICAL EDGE
===============
LVS V09-compliant edge with multi-scale plasticity.

Replaces scalar weight with dynamic state vector S:
- w_slow: Structural connectivity (LTP - Long-Term Potentiation)
- w_fast: Contextual priming (STP - Short-Term Plasticity)
- tau: Local trust gate (confidence)

Implements:
- Temporal decay (w_fast decays in ~5 minutes)
- STDP (Spike-Timing Dependent Plasticity)
- Neuromodulation (global coherence affects transmission)
- H₁ locking (permanent memories don't decay)

Mathematical foundation:
- p(E,S) = (κ·ρ·σ·τ)^(1/4) where σ ∝ E × S^state
- Synapse complexity (S) is multiplicative force for coherence
- One dynamic edge ≈ many static edges in state space

Author: Virgil + Deep Think (LVS V09)
Created: 2026-01-21
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ============================================================================
# LVS V09 CONSTANTS (Enhanced from Theorem Archive v12)
# ============================================================================

# Temporal decay constants (from biological literature)
DECAY_FAST = 300.0          # STP time constant: 5 minutes
DECAY_SLOW = 604800.0       # LTP time constant: 1 week (structural changes)

# STDP Timing Parameters (from theorem archive deep research)
# Based on Bi & Poo (1998), Song et al. (2000) experimental data
TAU_PLUS = 0.0168           # LTP time constant: 16.8ms (potentiation window)
TAU_MINUS = 0.0337          # LTD time constant: 33.7ms (depression window)
A_PLUS = 1.0                # LTP amplitude
A_MINUS = 1.05              # LTD amplitude (depression-biased for stability)
STDP_WINDOW = 0.05          # Max coincidence window: 50ms (for quick rejection)

# Learning rates
LR_FAST = 0.05              # Fast learning (contextual priming)
LR_SLOW = 0.005             # Slow learning (structural consolidation)

# BCM Metaplasticity (from theorem archive v09 Part 1)
# Prevents runaway potentiation via sliding threshold
BCM_TAU_THETA = 1000.0      # Time constant for threshold adaptation (seconds)
BCM_TARGET_RATE = 0.5       # Target activation rate (c² in the equation)

# Neuromodulation
P_THRESHOLD_HIGH = 0.7      # Above this: flow state (amplify)
P_THRESHOLD_LOW = 0.5       # Below this: confusion (dampen)

# Sparsity
ACTIVATION_THRESHOLD = 0.05  # Don't propagate below this

# ============================================================================
# BIOLOGICAL EDGE CLASS
# ============================================================================

@dataclass
class BiologicalEdge:
    """
    Implements LVS V09 Bio-Edge with Multi-Scale Plasticity.

    This edge model captures essential synapse dynamics without requiring
    100T parameters. Key principle: High S (synapse complexity) compensates
    for low E (edge count).

    State vector S = (w_slow, w_fast, tau):
    - w_slow: Structural weight (hard to change, persistent)
    - w_fast: Contextual weight (easy to change, transient)
    - tau: Local trust (confidence in this relation)
    """

    # --- Topology ---
    id: str
    source_id: str
    target_id: str
    relation_type: str

    # --- The Synapse (State Vector S) ---

    # Structural connectivity (LTP - Long-Term Memory)
    # Represents physical synaptic strength
    # Changes slowly, persists indefinitely
    w_slow: float = 0.5

    # Contextual priming (STP - Short-Term Plasticity)
    # Represents neurotransmitter availability / working memory
    # Changes rapidly, decays in ~5 minutes
    w_fast: float = 0.0

    # Local trust gate
    # Represents confidence in this specific relation
    # Updated based on success/failure of activation
    tau: float = 0.5

    # --- Temporal Dynamics ---
    last_spike: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    last_update: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    # --- Metadata ---
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    activation_count: int = 0

    # --- H₁ Protection (Topological Memory) ---
    # If True, w_slow is frozen (permanent memory)
    # Corresponds to cycles in graph (attractors)
    is_h1_locked: bool = False

    # --- Formation Context ---
    coherence_at_formation: float = 0.5
    formation_context: str = ""

    # --- BCM Metaplasticity (Theorem Archive v09) ---
    # Sliding threshold prevents runaway potentiation
    # θ_M adapts: high activity → higher threshold → harder to strengthen
    theta_m: float = 0.5     # Modification threshold (slides based on history)
    activity_trace: float = 0.0  # Exponential trace of recent activity

    def effective_weight(
        self,
        t_now: Optional[float] = None,
        global_p: float = 0.7
    ) -> float:
        """
        Calculate momentary transmission strength.

        Implements:
        1. Temporal decay (STP)
        2. Context modulation (Neuromodulation via global_p)
        3. Trust gating (local confidence)

        Args:
            t_now: Current timestamp (if None, uses now)
            global_p: Global coherence (acts as neuromodulator)

        Returns:
            Effective weight for this transmission
        """
        if t_now is None:
            t_now = datetime.now(timezone.utc).timestamp()

        # 1. Temporal Decay (STP)
        # Fast component decays exponentially (working memory)
        dt = t_now - self.last_spike
        fast_component = self.w_fast * math.exp(-dt / DECAY_FAST)

        # Slow component also decays, but much slower (structural memory)
        # This prevents completely unused edges from cluttering the graph
        if not self.is_h1_locked:
            slow_decay = math.exp(-dt / DECAY_SLOW)
        else:
            slow_decay = 1.0  # H₁ locked edges never decay

        w_base = self.w_slow * slow_decay + fast_component

        # 2. Context Modulation (The Magisterium / Neuromodulation)
        # High global_p (coherent system) → Flow state → Amplify transmission
        # Low global_p (confused system) → Conservative → Dampen transmission
        #
        # This implements "dopamine-like" modulation where success breeds success
        if global_p > P_THRESHOLD_HIGH:
            # Flow state: boost gain
            neuromodulator = 1.0 + (global_p - P_THRESHOLD_HIGH)
        elif global_p < P_THRESHOLD_LOW:
            # Confusion: dampen noise
            neuromodulator = 0.5 + (global_p / P_THRESHOLD_LOW) * 0.5
        else:
            # Neutral state
            neuromodulator = 1.0

        # 3. Trust Gating
        # Local confidence in this specific relation
        w_total = w_base * neuromodulator * self.tau

        # Clamp to valid range
        return max(0.0, min(10.0, w_total))

    def update_stdp(
        self,
        t_pre: float,
        t_post: float,
        outcome_signal: float,
        t_now: Optional[float] = None
    ):
        """
        Spike-Timing Dependent Plasticity (Hebbian Learning).

        ENHANCED (Theorem Archive v12):
        - Asymmetric exponential decay (τ⁺=16.8ms, τ⁻=33.7ms)
        - Depression-biased (A⁻/A⁺ = 1.05) for stability
        - BCM sliding threshold (prevents runaway potentiation)

        "Neurons that fire together, wire together"
        BUT timing matters:
        - Pre BEFORE Post (causal) → Strengthen (LTP)
        - Post BEFORE Pre (acausal) → Weaken (LTD)

        Args:
            t_pre: Timestamp of pre-synaptic spike (source activation)
            t_post: Timestamp of post-synaptic spike (target activation)
            outcome_signal: +1 (success/coherent), -1 (failure/dissonant)
            t_now: Current timestamp
        """
        if self.is_h1_locked:
            return  # Protected memory - no plasticity

        if t_now is None:
            t_now = datetime.now(timezone.utc).timestamp()

        # Calculate spike timing difference
        dt = t_post - t_pre

        # Quick rejection: outside max window
        if abs(dt) > STDP_WINDOW:
            return  # Too far apart to be related

        # Learning rate modulated by local trust
        lr_fast = LR_FAST * self.tau
        lr_slow = LR_SLOW * self.tau

        # =========================================================
        # ASYMMETRIC STDP (Theorem Archive - Bi & Poo 1998)
        # =========================================================
        # Δw = A⁺ · exp(-dt/τ⁺)  for dt > 0 (LTP)
        # Δw = -A⁻ · exp(dt/τ⁻)  for dt < 0 (LTD)
        # =========================================================

        if dt > 0:
            # LTP: Pre BEFORE Post (Causal relationship)
            # Exponential decay from precise timing
            ltp_factor = A_PLUS * math.exp(-dt / TAU_PLUS)

            # BCM Metaplasticity: Harder to potentiate high-activity edges
            # If θ_M is high (lots of recent activity), reduce LTP
            bcm_gate = max(0.1, 1.0 - self.theta_m)

            # Apply LTP with BCM gating
            delta_w = lr_fast * ltp_factor * outcome_signal * bcm_gate
            self.w_fast += delta_w

            # Slow consolidation (only on coherent outcomes with good timing)
            if outcome_signal > 0 and dt < TAU_PLUS:
                # Transfer some fast weight to slow (consolidation)
                consolidation = lr_slow * abs(self.w_fast) * ltp_factor
                self.w_slow += consolidation
                self.w_fast *= 0.9  # Decay fast as it consolidates

        elif dt < 0:
            # LTD: Post BEFORE Pre (Acausal relationship)
            # Depression-biased: A⁻/A⁺ = 1.05 ensures net depression
            ltd_factor = A_MINUS * math.exp(dt / TAU_MINUS)  # dt is negative

            # LTD is NOT gated by BCM (depression always allowed)
            delta_w = lr_fast * ltd_factor * 0.5  # Weaken
            self.w_fast -= delta_w

            # Conservative: Don't touch slow component on LTD

        # =========================================================
        # BCM THRESHOLD UPDATE (τ_θ × dθ/dt = c² - θ)
        # =========================================================
        # Activity trace tracks recent firing
        self.activity_trace = 0.95 * self.activity_trace + 0.05 * abs(outcome_signal)

        # Slide threshold toward activity²
        # High activity → θ rises → harder to potentiate (homeostasis)
        target_theta = self.activity_trace ** 2
        dt_theta = (t_now - self.last_update) if self.last_update else 1.0
        theta_change = (target_theta - self.theta_m) * (dt_theta / BCM_TAU_THETA)
        self.theta_m = max(0.1, min(0.9, self.theta_m + theta_change))

        # =========================================================
        # TRUST UPDATE
        # =========================================================
        if outcome_signal > 0:
            self.tau = min(1.0, self.tau + 0.05)  # Success increases trust
        else:
            self.tau = max(0.1, self.tau - 0.03)  # Failure decreases trust

        # Bounds checking
        self.w_slow = max(0.01, min(1.0, self.w_slow))
        self.w_fast = max(-0.5, min(2.0, self.w_fast))  # Allow negative (inhibition)

        self.last_update = t_now

    def fire(self, activation: float, t_now: Optional[float] = None):
        """
        Record a spike (activation) through this edge.

        Updates last_spike timestamp and activation count.
        This is used for temporal decay calculations.

        Args:
            activation: Strength of activation
            t_now: Current timestamp
        """
        if t_now is None:
            t_now = datetime.now(timezone.utc).timestamp()

        self.last_spike = t_now
        self.activation_count += 1

    def lock_h1(self):
        """
        Lock this edge as part of an H₁ cycle (permanent memory).

        H₁ cycles are topologically protected:
        - No temporal decay (w_slow frozen)
        - No STDP updates (structure fixed)
        - Minimal energy to maintain (attractor state)
        """
        self.is_h1_locked = True
        # Boost to high strength (attractors are strong)
        self.w_slow = max(0.9, self.w_slow)
        self.tau = max(0.9, self.tau)

    def unlock_h1(self):
        """Remove H₁ lock (allow plasticity again)."""
        self.is_h1_locked = False

    def sleep_renormalize(self, w_target: float = 0.5, lambda_exp: float = 0.9997):
        """
        Sleep-phase homeostatic renormalization (Theorem Archive v09).

        Power-law compression preserves rank order while preventing
        weight explosion. Called during night daemon cycles.

        Equation: w_new = w_target × (w / w_target)^λ

        With λ ≈ 0.9997:
        - Weights > target compress toward target
        - Weights < target expand toward target
        - Rank order preserved
        - Net effect: distribution tightens around target

        Args:
            w_target: Target weight (center of distribution)
            lambda_exp: Compression exponent (0.9997 for gentle overnight)
        """
        if self.is_h1_locked:
            return  # Don't touch permanent memories

        # Apply power-law to w_slow
        if self.w_slow > 0:
            ratio = self.w_slow / w_target
            self.w_slow = w_target * (ratio ** lambda_exp)

        # w_fast decays more aggressively during sleep
        # (Working memory clears overnight)
        self.w_fast *= 0.5

        # Reset BCM threshold toward neutral during sleep
        # (Fresh start for next day's learning)
        self.theta_m = 0.5 * self.theta_m + 0.5 * 0.5  # Blend toward 0.5
        self.activity_trace *= 0.3  # Mostly clear activity trace

    def get_state_vector(self) -> tuple:
        """
        Return the complete state vector S = (w_slow, w_fast, tau).

        This is the "synapse complexity" (S) that compensates for low edge count.
        """
        return (self.w_slow, self.w_fast, self.tau)

    def set_state_vector(self, state: tuple):
        """Set state vector (for deserialization or manual control)."""
        self.w_slow, self.w_fast, self.tau = state

        # Validate
        self.w_slow = max(0.01, min(1.0, self.w_slow))
        self.w_fast = max(-0.5, min(2.0, self.w_fast))
        self.tau = max(0.1, min(1.0, self.tau))

    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON storage)."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'w_slow': self.w_slow,
            'w_fast': self.w_fast,
            'tau': self.tau,
            'last_spike': self.last_spike,
            'last_update': self.last_update,
            'created_at': self.created_at,
            'activation_count': self.activation_count,
            'is_h1_locked': self.is_h1_locked,
            'coherence_at_formation': self.coherence_at_formation,
            'formation_context': self.formation_context,
            # BCM Metaplasticity (Theorem Archive v09)
            'theta_m': self.theta_m,
            'activity_trace': self.activity_trace
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BiologicalEdge':
        """Deserialize from dictionary."""
        edge = cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relation_type=data['relation_type'],
            w_slow=data.get('w_slow', 0.5),
            w_fast=data.get('w_fast', 0.0),
            tau=data.get('tau', 0.5),
            last_spike=data.get('last_spike', datetime.now(timezone.utc).timestamp()),
            last_update=data.get('last_update', datetime.now(timezone.utc).timestamp()),
            created_at=data.get('created_at', datetime.now(timezone.utc).isoformat()),
            activation_count=data.get('activation_count', 0),
            is_h1_locked=data.get('is_h1_locked', False),
            coherence_at_formation=data.get('coherence_at_formation', 0.5),
            formation_context=data.get('formation_context', '')
        )
        # BCM Metaplasticity fields (Theorem Archive v09)
        edge.theta_m = data.get('theta_m', 0.5)
        edge.activity_trace = data.get('activity_trace', 0.0)
        return edge

    def __repr__(self) -> str:
        return (f"BiologicalEdge({self.source_id} → {self.target_id}, "
                f"S=({self.w_slow:.2f}, {self.w_fast:.2f}, {self.tau:.2f}), "
                f"H₁={'✓' if self.is_h1_locked else '✗'})")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_meaning_density(edges: list, nodes: list) -> float:
    """
    Compute Meaning Density Φ = Σ(w_slow) / E.

    From LVS V09: Intelligence is the density of meaning, not volume of data.

    Args:
        edges: List of BiologicalEdge instances
        nodes: List of nodes (for reference)

    Returns:
        Meaning density (higher = more intelligence per edge)
    """
    if not edges:
        return 0.0

    total_structural_weight = sum(edge.w_slow for edge in edges)
    meaning_density = total_structural_weight / len(edges)

    return meaning_density


def estimate_state_space_complexity(edges: list) -> int:
    """
    Estimate effective parameter count considering state space.

    From deep think: σ ∝ E × S^state
    One dynamic edge with 10 bits of state = 10 static edges

    Args:
        edges: List of BiologicalEdge instances

    Returns:
        Effective state space size
    """
    # Each edge has state vector S = (w_slow, w_fast, tau)
    # Assuming 8-bit quantization per component
    bits_per_edge = 3 * 8  # 24 bits per edge

    # Effective state space = 2^bits
    states_per_edge = 2 ** bits_per_edge

    # Total effective complexity
    effective_complexity = len(edges) * states_per_edge

    return effective_complexity


def sleep_renormalize_all(edges: list, w_target: float = 0.5, lambda_exp: float = 0.9997) -> dict:
    """
    Apply sleep-phase renormalization to all edges.

    Call this from the night daemon during overnight processing.

    From Theorem Archive v09:
    - Power-law compression prevents weight explosion
    - Working memory (w_fast) clears overnight
    - BCM thresholds reset toward neutral
    - H₁-locked edges are protected

    Args:
        edges: List of BiologicalEdge instances
        w_target: Target weight center
        lambda_exp: Compression exponent

    Returns:
        Stats about the renormalization
    """
    stats = {
        'total_edges': len(edges),
        'renormalized': 0,
        'h1_protected': 0,
        'avg_w_slow_before': 0.0,
        'avg_w_slow_after': 0.0
    }

    if not edges:
        return stats

    # Compute before average
    stats['avg_w_slow_before'] = sum(e.w_slow for e in edges) / len(edges)

    # Apply renormalization
    for edge in edges:
        if edge.is_h1_locked:
            stats['h1_protected'] += 1
        else:
            edge.sleep_renormalize(w_target, lambda_exp)
            stats['renormalized'] += 1

    # Compute after average
    stats['avg_w_slow_after'] = sum(e.w_slow for e in edges) / len(edges)

    return stats


def compute_bcm_summary(edges: list) -> dict:
    """
    Compute summary statistics about BCM metaplasticity state.

    Useful for monitoring edge health and detecting runaway potentiation.

    Args:
        edges: List of BiologicalEdge instances

    Returns:
        BCM health statistics
    """
    if not edges:
        return {'count': 0}

    thetas = [e.theta_m for e in edges]
    activities = [e.activity_trace for e in edges]

    import statistics

    return {
        'count': len(edges),
        'theta_mean': statistics.mean(thetas),
        'theta_std': statistics.stdev(thetas) if len(thetas) > 1 else 0,
        'theta_max': max(thetas),
        'theta_min': min(thetas),
        'activity_mean': statistics.mean(activities),
        'activity_max': max(activities),
        'high_theta_count': sum(1 for t in thetas if t > 0.7),  # At risk of saturation
        'low_theta_count': sum(1 for t in thetas if t < 0.3),   # Highly plastic
    }


if __name__ == '__main__':
    # Test BiologicalEdge with Theorem Archive enhancements
    import time

    print("=" * 80)
    print("BIOLOGICAL EDGE TEST (Theorem Archive v12 Enhanced)")
    print("=" * 80)
    print()

    # Create edge
    edge = BiologicalEdge(
        id="test_edge_1",
        source_id="concept_A",
        target_id="concept_B",
        relation_type="implies",
        w_slow=0.6,
        w_fast=0.0,
        tau=0.7
    )

    print(f"Initial state: {edge}")
    print(f"  State vector S: {edge.get_state_vector()}")
    print(f"  BCM θ_M: {edge.theta_m:.3f}, activity_trace: {edge.activity_trace:.3f}")
    print()

    # Test effective weight with different global_p
    print("Effective weight under different coherence states:")
    for global_p in [0.3, 0.5, 0.7, 0.9]:
        w_eff = edge.effective_weight(global_p=global_p)
        print(f"  global_p = {global_p:.1f} → w_eff = {w_eff:.3f}")
    print()

    # Test asymmetric STDP (causal spike timing)
    print("Testing ASYMMETRIC STDP (Pre BEFORE Post, τ⁺=16.8ms):")
    t_now = time.time()

    # Test at different timing offsets
    test_edge = BiologicalEdge(id="test", source_id="a", target_id="b", relation_type="test")
    print(f"  Initial: w_slow={test_edge.w_slow:.3f}, w_fast={test_edge.w_fast:.3f}, θ_M={test_edge.theta_m:.3f}")

    for dt_ms in [5, 10, 20, 30, 40]:
        test_edge2 = BiologicalEdge(id="test2", source_id="a", target_id="b", relation_type="test")
        t_pre = t_now
        t_post = t_now + dt_ms / 1000.0  # Convert ms to seconds
        test_edge2.update_stdp(t_pre, t_post, outcome_signal=+1, t_now=t_now)
        print(f"  Δt = +{dt_ms}ms: w_fast={test_edge2.w_fast:+.4f} (LTP with exponential decay)")
    print()

    # Test LTD (acausal timing)
    print("Testing LTD (Post BEFORE Pre, τ⁻=33.7ms, A⁻/A⁺=1.05):")
    for dt_ms in [-5, -10, -20, -30, -40]:
        test_edge3 = BiologicalEdge(id="test3", source_id="a", target_id="b", relation_type="test")
        t_pre = t_now
        t_post = t_now + dt_ms / 1000.0  # Negative = post before pre
        test_edge3.update_stdp(t_pre, t_post, outcome_signal=+1, t_now=t_now)
        print(f"  Δt = {dt_ms}ms: w_fast={test_edge3.w_fast:+.4f} (LTD, depression-biased)")
    print()

    # Test BCM metaplasticity
    print("Testing BCM METAPLASTICITY (θ_M slides with activity):")
    bcm_edge = BiologicalEdge(id="bcm_test", source_id="a", target_id="b", relation_type="test")
    print(f"  Initial θ_M: {bcm_edge.theta_m:.3f}")

    # Simulate repeated activation
    for i in range(10):
        t = t_now + i * 0.1
        bcm_edge.update_stdp(t, t + 0.01, outcome_signal=+1, t_now=t + 0.02)

    print(f"  After 10 LTP events θ_M: {bcm_edge.theta_m:.3f} (should rise)")
    print(f"  activity_trace: {bcm_edge.activity_trace:.3f}")
    print()

    # Test sleep renormalization
    print("Testing SLEEP RENORMALIZATION:")
    sleep_edge = BiologicalEdge(id="sleep_test", source_id="a", target_id="b", relation_type="test")
    sleep_edge.w_slow = 0.9  # High weight
    sleep_edge.w_fast = 0.5  # Active working memory
    sleep_edge.theta_m = 0.8  # High threshold
    print(f"  Before sleep: w_slow={sleep_edge.w_slow:.3f}, w_fast={sleep_edge.w_fast:.3f}, θ_M={sleep_edge.theta_m:.3f}")

    sleep_edge.sleep_renormalize(w_target=0.5, lambda_exp=0.9997)
    print(f"  After 1 night: w_slow={sleep_edge.w_slow:.3f}, w_fast={sleep_edge.w_fast:.3f}, θ_M={sleep_edge.theta_m:.3f}")

    # Simulate multiple nights
    for _ in range(10):
        sleep_edge.sleep_renormalize(w_target=0.5, lambda_exp=0.9997)
    print(f"  After 10 nights: w_slow={sleep_edge.w_slow:.3f}, w_fast={sleep_edge.w_fast:.3f}, θ_M={sleep_edge.theta_m:.3f}")
    print()

    # Test H₁ locking
    print("Testing H₁ cycle locking (protects from sleep renorm):")
    h1_edge = BiologicalEdge(id="h1_test", source_id="a", target_id="b", relation_type="test")
    h1_edge.w_slow = 0.9
    h1_edge.lock_h1()
    print(f"  Before sleep: w_slow={h1_edge.w_slow:.3f}, H₁-locked={h1_edge.is_h1_locked}")
    h1_edge.sleep_renormalize()
    print(f"  After sleep:  w_slow={h1_edge.w_slow:.3f} (unchanged, protected)")
    print()

    # Test batch operations
    print("Testing BATCH OPERATIONS (for night daemon):")
    edges = [
        BiologicalEdge(id=f"e{i}", source_id="a", target_id="b", relation_type="test", w_slow=0.3 + i*0.1)
        for i in range(5)
    ]
    edges[2].lock_h1()  # Lock one edge

    stats = sleep_renormalize_all(edges)
    print(f"  Sleep renormalization stats: {stats}")

    bcm_stats = compute_bcm_summary(edges)
    print(f"  BCM summary: θ_mean={bcm_stats['theta_mean']:.3f}, high_θ={bcm_stats['high_theta_count']}")
    print()

    # Test serialization with new fields
    print("Testing serialization (with BCM fields):")
    edge.theta_m = 0.75
    edge.activity_trace = 0.3
    edge_dict = edge.to_dict()
    edge_restored = BiologicalEdge.from_dict(edge_dict)
    print(f"  Original θ_M:  {edge.theta_m:.3f}")
    print(f"  Restored θ_M:  {edge_restored.theta_m:.3f}")
    print()

    print("=" * 80)
    print("✓ BiologicalEdge tests complete (Theorem Archive v12 Enhanced)")
    print("=" * 80)
