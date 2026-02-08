"""
LOGOS LVS SCANNERS
==================
Coherence measurement modules for the Self-Improvement Protocol v2.0

Components:
- kappa_scanner: Split-brain detector (κ - integration)
- h1_scanner: Memory topology health (σ - structure via H₁)
- tau_modulator: Trust dynamics gate (τ - plasticity)
- coherence_calculator: Main proprioception engine (p - global coherence)

Formula: p = (κ × ρ × σ × τ)^(1/4)

Theory: 04_LR_LAW/CANON/LVS_MATHEMATICS.md
Protocol: 00_NEXUS/LOGOS_SELF_IMPROVEMENT_PROTOCOL.md
"""

from .kappa_scanner import KappaScanner, scan as kappa_scan
from .h1_scanner import H1Scanner, scan as h1_scan
from .tau_modulator import TauModulator, modulate as tau_modulate
from .coherence_calculator import CoherenceCalculator, calculate as coherence_calculate
from .edge_pruner import EdgePruner, analyze as pruner_analyze, cure_seizure

__all__ = [
    'KappaScanner',
    'H1Scanner',
    'TauModulator',
    'CoherenceCalculator',
    'EdgePruner',
    'kappa_scan',
    'h1_scan',
    'tau_modulate',
    'coherence_calculate',
    'pruner_analyze',
    'cure_seizure'
]
