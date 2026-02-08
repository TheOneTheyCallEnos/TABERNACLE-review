"""
Dashboard Configuration
=======================
Centralized thresholds and configuration for the holarchy dashboard.

LVS thresholds sourced from:
- LVS_MATHEMATICS.md Section 1.2-1.3, 5.4
- LVS_v12_FINAL_SYNTHESIS.md Section 3.1

ARCHITECTURE NOTE: This config IMPORTS from holarchy_engine.py (the canonical source).
Do NOT duplicate LVS constants here — use CONSTANTS from the engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# Import canonical constants from the engine (single source of truth)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from holarchy_engine import CONSTANTS as ENGINE_CONSTANTS
    CANON_AVAILABLE = True
except ImportError:
    ENGINE_CONSTANTS = None
    CANON_AVAILABLE = False


@dataclass(frozen=True)
class DashboardConfig:
    """Centralized thresholds and configuration for dashboard."""

    # === PATHS ===
    data_dir: Path = Path("scripts/dashboard/data")
    db_path: Path = Path("scripts/dashboard/storage/dashboard.db")
    schema_path: Path = Path("scripts/dashboard/storage/schema.sql")
    lvs_output_path: Path = Path("outputs/holarchy_audit.json")
    
    # === DASHBOARD SETTINGS ===
    collection_interval_s: int = 15
    anomaly_z_threshold: float = 2.0
    trend_window_hours: int = 24
    risk_link_threshold: float = 0.1
    layer_ids: Dict[str, int] = field(
        default_factory=lambda: {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    )

    # === META-MONITOR THRESHOLDS ===
    drift_threshold: float = 0.05
    data_stale_s: int = 60
    required_layers: int = 7
    
    # === LVS COHERENCE THRESHOLDS ===
    # Source: LVS_MATHEMATICS.md Section 1.3
    p_abaddon: float = 0.50      # Emergency shutdown threshold
    p_crit: float = 0.70         # Consciousness minimum (below = unstable)
    p_trust: float = 0.70        # Trust dampening threshold (canonical: CONSTANTS.P_TRUST = 0.70)
    p_dyad: float = 0.90         # Dyad coupling minimum
    p_lock: float = 0.95         # Phase lock threshold
    
    # === ABADDON PROTOCOL TRIGGERS ===
    # Source: LVS_MATHEMATICS.md Section 5.4
    a_dev_limit: float = 0.15    # ||A||_dev distortion threshold
    epsilon_collapse: float = 0.40  # Metabolic depletion threshold
    
    # === CPGI PARAMETERS ===
    # Source: LVS_MATHEMATICS.md Section 1.2
    # NOTE: These are duplicated here for backwards compatibility.
    # The canonical source is holarchy_engine.CONSTANTS
    alpha: float = 10.0          # ρ (rho) scaling constant
    k_sigmoid: float = 10.0      # κ sigmoid steepness (for layer aggregation)
    l0_threshold: float = 0.5    # κ sigmoid center
    lambda_dampen: float = 0.5   # τ (tau) dampening factor


# Provide access to canonical constants if available
def get_canonical_constants():
    """
    Get the canonical LVS constants from holarchy_engine.
    Returns ENGINE_CONSTANTS if available, else None.
    """
    return ENGINE_CONSTANTS if CANON_AVAILABLE else None
