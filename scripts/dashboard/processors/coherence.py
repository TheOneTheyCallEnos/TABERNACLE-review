"""
CPGI Coherence Aggregator
=========================
Computes global coherence p via geometric mean of four CPGI components.

Canon: LVS_v12_FINAL_SYNTHESIS.md Section 3.1
Formula: p = (κ · ρ · σ · τ)^(1/4)

CRITICAL: Zero-Bottleneck Property (LVS_MATHEMATICS.md Section 1.1)
"If ANY component → 0, then p → 0 (multiplicative collapse)"

ARCHITECTURE NOTE: This processor DELEGATES to holarchy_engine.py for
the actual computation. The engine is the single source of truth.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict

# Import canonical functions from the engine
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from holarchy_engine import (
        compute_global_coherence as engine_compute_coherence,
        evaluate_coherence_state as engine_evaluate_state,
        CONSTANTS as ENGINE_CONSTANTS,
    )
    CANON_AVAILABLE = True
except ImportError:
    CANON_AVAILABLE = False
    engine_compute_coherence = None
    engine_evaluate_state = None
    ENGINE_CONSTANTS = None

from dashboard.config import DashboardConfig


class CoherenceAggregator:
    """
    Aggregates layer metrics into global coherence p via CPGI geometric mean.
    
    Canon: LVS_v12_FINAL_SYNTHESIS.md Section 3.1
    
    The four CPGI components:
    - κ (Kappa): Continuity via Anchor-and-Expand
    - ρ (Rho): Precision via inverse error variance
    - σ (Sigma): Structure via edge-of-chaos parabola
    - τ (Tau): Trust-gating with coherence-dependent dampening
    """

    def __init__(self, config: DashboardConfig) -> None:
        self._config = config

    def aggregate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute global coherence p = (κ · ρ · σ · τ)^(1/4)
        
        Canon: LVS_v12_FINAL_SYNTHESIS.md Section 3.1
        
        Implements Zero-Bottleneck Property:
        If ANY component → 0, then p → 0 (multiplicative collapse)
        
        DELEGATES to holarchy_engine for computation when available.
        """
        # Extract CPGI components from LVS bridge payload
        lvs_data = payload.get("lvs", {})
        coherence_data = lvs_data.get("coherence", {})
        
        # Get the four CPGI components (default to 0.5 if missing)
        kappa = float(coherence_data.get("kappa", coherence_data.get("κ", 0.5)))
        rho = float(coherence_data.get("rho", coherence_data.get("ρ", 0.5)))
        sigma = float(coherence_data.get("sigma", coherence_data.get("σ", 0.5)))
        tau = float(coherence_data.get("tau", coherence_data.get("τ", 0.5)))
        
        # DELEGATE to canonical engine if available
        if CANON_AVAILABLE and engine_compute_coherence:
            cpgi = engine_compute_coherence(kappa, rho, sigma, tau)
            state = engine_evaluate_state(cpgi).value
        else:
            # Fallback: inline computation (same formula, for resilience)
            product = kappa * rho * sigma * tau
            
            # Zero-Bottleneck Property: If ANY component → 0, p → 0
            if product <= 0:
                cpgi = 0.0
            else:
                cpgi = math.pow(product, 0.25)
            
            # Clamp to [0, 1]
            cpgi = max(0.0, min(1.0, cpgi))
            
            # Evaluate coherence state based on thresholds
            state = self._evaluate_state(cpgi)
        
        return {
            "layer": payload.get("layer"),
            "layer_id": payload.get("layer_id"),
            "timestamp": payload.get("timestamp"),
            "cpgi": cpgi,
            "state": state,
            "canon_delegated": CANON_AVAILABLE,
            "components": {
                "kappa": kappa,
                "rho": rho,
                "sigma": sigma,
                "tau": tau,
            }
        }
    
    def _evaluate_state(self, p: float) -> str:
        """
        Evaluate system state from coherence value.
        
        Canon: LVS_MATHEMATICS.md Section 1.3
        """
        if p < self._config.p_abaddon:
            return "ABADDON"
        elif p < self._config.p_crit:
            return "UNSTABLE"
        elif p < self._config.p_trust:
            return "DAMPENED"
        elif p < self._config.p_dyad:
            return "STABLE"
        elif p < self._config.p_lock:
            return "DYAD_READY"
        else:
            return "P_LOCK"
