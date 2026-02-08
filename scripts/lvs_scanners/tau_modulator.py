"""
LOGOS LVS MODULE: TAU MODULATOR
===============================
LVS Component: τ (Tau) - Trust & Plasticity
Theory Ref: 04_LR_LAW/CANON/LVS_MATHEMATICS.md

Purpose:
  Implements the "Trust Dynamics" that gate system evolution.
  High Coherence (p) -> Open Gate (High Plasticity)
  Low Coherence (p) -> Closed Gate (Safe Mode)

  Bio-Mimicry: Acts like Neuromodulation (Dopamine/Serotonin gating).

LVS Trust Dynamics:
  - P-LOCK (p ≥ 0.95): Flow state, exploratory permitted
  - NORMAL (p ≥ 0.80): Standard operations
  - CAUTIOUS (p ≥ 0.70): Extra validation required
  - CONSERVATIVE (p ≥ 0.50): Critical fixes only
  - ABADDON (p < 0.50): Emergency shutdown

Author: Gemini 2.5 Pro (LVS Review)
Date: 2026-01-29
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("logos.tau")


class TauModulator:
    """
    Trust Dynamics Gate for LVS Coherence Protocol.

    Modulates system plasticity (τ) based on current coherence (p).
    When coherence is high, the system is open to change.
    When coherence is low, the system becomes conservative.

    This mirrors biological neuromodulation:
    - High coherence = Dopamine flow = Exploratory behavior
    - Low coherence = Norepinephrine spike = Risk aversion
    """

    def __init__(self, base_risk_tolerance: float = 1.0):
        """
        Initialize with optional personality parameter.

        Args:
            base_risk_tolerance: 1.0 = Standard, >1.0 = Risk-seeking, <1.0 = Risk-averse
        """
        self.base_risk_tolerance = base_risk_tolerance

    def modulate(self, p_current: float, tau_raw: float) -> Dict[str, Any]:
        """
        Calculates the effective trust and system mode based on current coherence (p).

        Args:
            p_current: Current system coherence [0, 1]
            tau_raw: Raw trust score before modulation [0, 1]

        Returns:
            Dict with tau_raw, tau_modulated, mode, max_changes_allowed, recommendation
        """
        mode = "UNKNOWN"
        tau_effective = 0.0
        max_changes = 0
        recommendation = ""
        gate_status = "LOCKED"

        # LVS Trust Dynamics Curve
        if p_current >= 0.95:
            # P-LOCK: Flow State - Peak coherence achieved
            mode = "P-LOCK"
            tau_effective = min(tau_raw * 1.2 * self.base_risk_tolerance, 1.0)
            max_changes = 999  # Unlimited
            gate_status = "OPEN"
            recommendation = "Exploratory Optimization Permitted"

        elif p_current >= 0.80:
            # NORMAL: Healthy Function
            mode = "NORMAL"
            tau_effective = tau_raw * self.base_risk_tolerance
            max_changes = 10
            gate_status = "OPEN"
            recommendation = "Standard Operations"

        elif p_current >= 0.70:
            # CAUTIOUS: Mild Incoherence
            mode = "CAUTIOUS"
            tau_effective = tau_raw * 0.7 * self.base_risk_tolerance
            max_changes = 5
            gate_status = "RESTRICTED"
            recommendation = "Require Extra Validation"

        elif p_current >= 0.50:
            # CONSERVATIVE: System Instability
            mode = "CONSERVATIVE"
            tau_effective = tau_raw * 0.3 * self.base_risk_tolerance
            max_changes = 2
            gate_status = "RESTRICTED"
            recommendation = "Critical Fixes Only"

        else:
            # ABADDON: System Collapse
            mode = "ABADDON"
            tau_effective = 0.0
            max_changes = 0
            gate_status = "LOCKED"
            recommendation = "EMERGENCY SHUTDOWN / ROLLBACK"

        return {
            "tau_raw": round(tau_raw, 3),
            "tau_modulated": round(tau_effective, 3),
            "mode": mode,
            "gate_status": gate_status,
            "max_changes_allowed": max_changes,
            "recommendation": recommendation
        }


def modulate(p_current: float, tau_raw: float) -> Dict[str, Any]:
    """Module-level convenience function."""
    return TauModulator().modulate(p_current, tau_raw)


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    modulator = TauModulator()

    # Test different coherence levels
    test_cases = [0.98, 0.85, 0.75, 0.60, 0.40]
    tau_raw = 0.85

    for p in test_cases:
        result = modulator.modulate(p, tau_raw)
        print(f"p={p:.2f}: {result['mode']:12s} τ={result['tau_modulated']:.3f} max_changes={result['max_changes_allowed']}")
