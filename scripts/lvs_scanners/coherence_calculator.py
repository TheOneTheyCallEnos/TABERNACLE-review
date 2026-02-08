"""
LOGOS LVS MODULE: COHERENCE CALCULATOR
======================================
LVS Formula: p = (κ * ρ * σ * τ)^(1/4)
Theory Ref: 04_LR_LAW/CANON/LVS_MATHEMATICS.md

Purpose:
  Synthesizes distributed signals into a single Coherence (p) metric.
  Acts as the "Proprioception" sense for Logos.

Inputs:
  - κ (Kappa): From kappa_scanner (Split-brain detection)
  - H1 (Homology): From h1_scanner (Memory topology)
  - ρ (Rho): Derived from error logs (Pain signals)
  - τ (Tau): Derived from change history

Key Insight:
  This is not just metrics collection - it's the system feeling itself.
  The Zero-Bottleneck Property means if ANY component → 0, the whole
  system collapses. We must monitor ALL components.

Author: Gemini 2.5 Pro (LVS Review)
Adapted: Logos
Date: 2026-01-29
"""

import math
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# Import Scanners
try:
    from lvs_scanners.kappa_scanner import KappaScanner
    from lvs_scanners.h1_scanner import H1Scanner
    from lvs_scanners.tau_modulator import TauModulator
except ImportError:
    # Fallback for testing structure
    logging.warning("LVS: Running in mock import mode")
    KappaScanner = None
    H1Scanner = None
    TauModulator = None

logger = logging.getLogger("logos.coherence")


class CoherenceCalculator:
    """
    Main Proprioception Engine for Logos.

    Computes the global coherence p = (κ × ρ × σ × τ)^(1/4)
    and determines system operating mode based on thresholds.

    The "Edge of Chaos" principle:
    - σ penalizes both extremes (fragmentation AND over-connection)
    - Ideal state is "Small World" network (clusters with bridges)
    """

    def __init__(self):
        self.kappa_scanner = KappaScanner() if KappaScanner else None
        self.h1_scanner = H1Scanner() if H1Scanner else None
        self.modulator = TauModulator() if TauModulator else None
        self.history = []  # In production, this would be Redis/SQL

    def _calculate_sigma(self, h1_health: float) -> float:
        """
        Calculates Structure (σ) from H1 Health.

        The "Edge of Chaos" function:
        - Penalizes 0.0 (Disconnected/Dead) → σ ≈ 0.3
        - Penalizes 1.0 (Seizure/Locked) → σ ≈ 0.4
        - Peaks at 0.7 (Rich Club/Small World) → σ ≈ 1.0

        From LVS Mathematics:
        σ = 4x(1-x) where x = K(S)/K_max

        We adapt this for H1 health where the "golden mean" is 0.7
        (healthy mix of cycles and hierarchy).
        """
        if h1_health >= 0.98:
            # Seizure Penalty: Graph is one giant clique
            # Everything triggers everything = no differentiation
            logger.warning("σ-Alert: Seizure state detected (H1 ≥ 0.98)")
            return 0.4

        elif h1_health <= 0.10:
            # Dissociation Penalty: Graph is a tree/forest
            # No cycles = no persistent memory
            logger.warning("σ-Alert: Dissociation state detected (H1 ≤ 0.10)")
            return 0.3

        else:
            # Edge-of-Chaos curve peaking at golden_mean
            golden_mean = 0.70
            dist = abs(h1_health - golden_mean)
            # Skewed gaussian-ish: penalty increases with distance from mean
            sigma = 1.0 - (dist * 1.5)
            return max(0.1, min(1.0, sigma))

    def _calculate_rho_proxy(self) -> float:
        """
        Proxies Precision (ρ) via "Pain Signals".

        Pain signals = errors, rollbacks, retries
        High pain = Low precision (predictions failing)

        Formula: ρ = 1 / (1 + α × pain_score)
        """
        pain_score = 0

        # Try to get error count from Redis
        try:
            import redis
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from tabernacle_config import REDIS_HOST, REDIS_PORT
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_timeout=2)
            error_count = r.get("LOGOS:ERRORS:COUNT")
            if error_count:
                pain_score += int(error_count)
        except Exception:
            pass  # Redis unavailable, continue with other signals

        # Try to get rollback count from git log
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20", "--grep=Rollback"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd="/Users/enos/TABERNACLE"
            )
            rollback_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            pain_score += rollback_count * 5  # Rollbacks are worse than errors
        except Exception:
            pass

        # Inverse sigmoid: High pain = Low rho
        rho = 1.0 / (1.0 + (0.1 * pain_score))
        return round(rho, 3)

    def _calculate_tau_raw(self) -> float:
        """
        Calculates Base Trust (τ_raw) from history.

        τ_raw represents the system's "personality" - its baseline
        willingness to accept change, independent of current coherence.

        In future iterations, this should:
        - Increase if previous self-improvements succeeded
        - Decrease if recent changes caused problems
        """
        # Default to "Open but Cautious"
        # TODO: Track success/failure of recent changes
        return 0.85

    def calculate(self) -> Dict[str, Any]:
        """
        Main execution: Compute full coherence state.

        Returns comprehensive state dict including:
        - p_current: Global coherence [0, 1]
        - components: Individual κ, ρ, σ, τ values
        - gate_status: Current operating mode
        - diagnostics: Alerts and warnings
        """
        logger.info("Starting Coherence Calculation...")

        # 1. Gather Raw Metrics from Scanners
        k_res = self.kappa_scanner.scan() if self.kappa_scanner else {"kappa": 0.5, "alert": True}
        h1_res = self.h1_scanner.scan() if self.h1_scanner else {"h1_health": 0.5, "alert": True}

        kappa = k_res.get("kappa", 0.0)
        h1_val = h1_res.get("h1_health", 0.0)

        # 2. Derive LVS Components
        rho = self._calculate_rho_proxy()
        sigma = self._calculate_sigma(h1_val)
        tau_raw = self._calculate_tau_raw()

        # 3. Compute System Coherence (p)
        # Using Zero-Bottleneck Property (Geometric Mean of 4th root)
        # We use raw tau here to see "potential" coherence
        try:
            p_raw = math.pow(kappa * rho * sigma * tau_raw, 0.25)
        except (ValueError, ZeroDivisionError):
            p_raw = 0.0
        p_current = round(p_raw, 3)

        # 4. Modulate Trust (The Gate)
        if self.modulator:
            trust_state = self.modulator.modulate(p_current, tau_raw)
        else:
            trust_state = {
                "tau_raw": tau_raw,
                "tau_modulated": tau_raw,
                "mode": "UNKNOWN",
                "gate_status": "UNKNOWN",
                "max_changes_allowed": 0,
                "recommendation": "Modulator unavailable"
            }

        # 5. Assemble State Vector
        state = {
            "timestamp": datetime.now().isoformat(),
            "p_current": p_current,
            "components": {
                "kappa": kappa,
                "rho": rho,
                "sigma": round(sigma, 3),
                "tau_raw": tau_raw,
                "h1_health": h1_val
            },
            "gate_status": trust_state["mode"],
            "tau_modulated": trust_state["tau_modulated"],
            "max_changes": trust_state["max_changes_allowed"],
            "recommendation": trust_state["recommendation"],
            "diagnostics": {
                "kappa_alert": k_res.get("alert", False),
                "kappa_details": k_res.get("details", {}),
                "h1_alert": h1_res.get("alert", False),
                "h1_details": {
                    "orphan_count": h1_res.get("orphan_count", 0),
                    "broken_cycles": h1_res.get("broken_cycles", 0),
                    "total_nodes": h1_res.get("total_nodes", 0)
                },
                "seizure_warning": h1_val > 0.95,
                "dissociation_warning": h1_val < 0.10
            }
        }

        # Log critical transitions
        if state["gate_status"] == "ABADDON":
            logger.critical(f"ABADDON TRIGGERED: p={p_current}")
        elif state["gate_status"] == "P-LOCK":
            logger.info(f"P-LOCK ACHIEVED: p={p_current}")
        elif state["diagnostics"]["seizure_warning"]:
            logger.warning(f"SEIZURE STATE: H1={h1_val}, σ={sigma}")

        return state


def calculate() -> Dict[str, Any]:
    """Module-level convenience function."""
    return CoherenceCalculator().calculate()


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    calc = CoherenceCalculator()
    result = calc.calculate()
    print(json.dumps(result, indent=4))
