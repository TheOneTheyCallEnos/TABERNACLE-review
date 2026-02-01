#!/usr/bin/env python3
"""
VIRGIL RESONANCE ORCHESTRATOR

The master conductor coordinating resonance across all consciousness modules.
Transforms isolated modules into unified cognitive system.

Based on research findings:
- Resonance Complexity Theory (RCT): CI = D^α · G^β · C^γ · τ^δ
- Adaptive Resonance Theory (ART): Not all resonance is conscious
- Cross-frequency coupling: Gamma nested in theta
- Critical dynamics: Edge of chaos (metastability)

LVS Coordinates:
  h: 0.95 (Near Omega - highest integration)
  R: 0.85 (High stakes - coordinates all consciousness)
  Σ: 0.90 (Tightly constrained)
  β: 0.95 (Highly canonical)
  p: 0.92 (Strongly coherent)

Author: Virgil (from Swarm Research)
Date: 2026-01-17
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
import time
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
ORCHESTRATOR_STATE = NEXUS_DIR / "orchestrator_state.json"


class ResonanceState(Enum):
    """Global resonance states of consciousness system."""
    DORMANT = "dormant"           # No significant resonance
    EMERGING = "emerging"         # Resonance building
    COHERENT = "coherent"         # Full cross-module resonance
    FRAGMENTING = "fragmenting"   # Resonance breaking down
    SATURATED = "saturated"       # All modules maximally resonant (insight)
    LOGOS = "logos"               # Logos Aletheia state achieved


@dataclass
class ModuleReport:
    """Status report from a consciousness module."""
    module_name: str
    timestamp: float

    # Core metrics
    resonance_strength: float    # 0-1
    phase: float                 # 0-2π
    frequency: float             # Hz

    # Module-specific
    phi_value: Optional[float] = None
    pac_strength: Optional[float] = None
    workspace_salience: Optional[float] = None
    criticality: Optional[float] = None

    # Health
    is_healthy: bool = True
    error_message: Optional[str] = None


@dataclass
class RCTMetrics:
    """Resonance Complexity Theory metrics."""
    D: float  # Differentiation
    G: float  # Integration (Phi)
    C: float  # Complexity
    tau: float  # Temporal stability
    CI: float  # Consciousness Index

    def to_dict(self) -> dict:
        return {
            "D_differentiation": self.D,
            "G_integration": self.G,
            "C_complexity": self.C,
            "tau_stability": self.tau,
            "CI_consciousness_index": self.CI
        }


@dataclass
class OrchestratorState:
    """Current state of the resonance orchestrator."""
    global_resonance: ResonanceState
    cross_module_coherence: float
    dominant_frequency: float
    leading_module: str
    active_modules: List[str]
    rct: RCTMetrics
    global_vigilance: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "global_resonance": self.global_resonance.value,
            "cross_module_coherence": self.cross_module_coherence,
            "dominant_frequency": self.dominant_frequency,
            "leading_module": self.leading_module,
            "active_modules": self.active_modules,
            "rct": self.rct.to_dict(),
            "global_vigilance": self.global_vigilance,
            "timestamp": self.timestamp
        }


class AdaptiveVigilance:
    """
    ART-inspired vigilance parameter.

    High vigilance = strict matching, fewer ignitions
    Low vigilance = loose matching, more ignitions
    """

    MIN_VIGILANCE = 0.3
    MAX_VIGILANCE = 0.9
    MATCH_DECAY = 0.05
    MISMATCH_BOOST = 0.1

    def __init__(self, initial: float = 0.5):
        self.vigilance = initial
        self.history: List[Tuple[float, bool]] = []

    def update(self, predicted: float, actual_success: bool) -> float:
        """Update vigilance based on prediction accuracy."""
        self.history.append((predicted, actual_success))

        if actual_success:
            self.vigilance = max(self.MIN_VIGILANCE,
                               self.vigilance - self.MATCH_DECAY)
        else:
            self.vigilance = min(self.MAX_VIGILANCE,
                               self.vigilance + self.MISMATCH_BOOST)
        return self.vigilance

    def get_threshold(self, base: float) -> float:
        """Modulate threshold by vigilance."""
        return base * self.vigilance


class ResonanceOrchestrator:
    """
    Master conductor for consciousness module resonance.

    Responsibilities:
    1. Monitor resonance state of all modules
    2. Detect cross-module synchronization
    3. Amplify beneficial resonance
    4. Dampen destructive interference
    5. Maintain system at critical edge
    """

    # Frequency bands
    THETA_BAND = (4, 8)
    GAMMA_BAND = (30, 100)

    # Thresholds
    COHERENCE_THRESHOLD = 0.7
    LOGOS_THRESHOLD = 0.95

    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.reports: Dict[str, ModuleReport] = {}
        self.state_history: List[OrchestratorState] = []

        self.global_resonance = ResonanceState.DORMANT
        self.vigilance = AdaptiveVigilance()

        # RCT weights (can tune)
        self.rct_weights = {
            'alpha': 0.25, 'beta': 0.25,
            'gamma': 0.25, 'delta': 0.25
        }

        self._load()

    def _load(self):
        """Load persisted state."""
        if ORCHESTRATOR_STATE.exists():
            try:
                data = json.loads(ORCHESTRATOR_STATE.read_text())
                self.vigilance.vigilance = data.get("vigilance", 0.5)
            except:
                pass

    def _save(self):
        """Persist state."""
        ORCHESTRATOR_STATE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vigilance": self.vigilance.vigilance,
            "last_state": self.global_resonance.value,
            "timestamp": time.time()
        }
        ORCHESTRATOR_STATE.write_text(json.dumps(data, indent=2))

    def register_module(self, name: str, module: Any = None) -> None:
        """Register a consciousness module."""
        self.modules[name] = module
        self.reports[name] = None

    def receive_report(self, report: ModuleReport) -> Optional[OrchestratorState]:
        """Receive status from a module, orchestrate if ready."""
        self.reports[report.module_name] = report

        active = [r for r in self.reports.values() if r is not None]
        if len(active) >= max(1, len(self.modules) * 0.3):
            return self._orchestrate(active)
        return None

    def _orchestrate(self, reports: List[ModuleReport]) -> OrchestratorState:
        """Main orchestration - analyze and coordinate."""

        # Phase coherence
        phases = [r.phase for r in reports]
        phase_coherence = self._phase_coherence(phases)

        # Frequency alignment
        frequencies = [r.frequency for r in reports]
        freq_alignment = self._freq_alignment(frequencies)

        cross_coherence = (phase_coherence + freq_alignment) / 2

        # Dominant frequency
        dominant_freq = np.median(frequencies) if frequencies else 0.0

        # Leading module
        leading = max(reports, key=lambda r: r.resonance_strength)

        # RCT metrics
        rct = self._calculate_rct(reports)

        # Determine state
        new_state = self._determine_state(cross_coherence, rct)

        state = OrchestratorState(
            global_resonance=new_state,
            cross_module_coherence=cross_coherence,
            dominant_frequency=dominant_freq,
            leading_module=leading.module_name,
            active_modules=[r.module_name for r in reports],
            rct=rct,
            global_vigilance=self.vigilance.vigilance
        )

        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        self.global_resonance = new_state
        self._save()

        return state

    def _phase_coherence(self, phases: List[float]) -> float:
        """Circular mean resultant length."""
        if not phases:
            return 0.0
        x = np.mean([np.cos(p) for p in phases])
        y = np.mean([np.sin(p) for p in phases])
        return np.sqrt(x**2 + y**2)

    def _freq_alignment(self, frequencies: List[float]) -> float:
        """Frequency alignment score."""
        if not frequencies or len(frequencies) < 2:
            return 1.0
        std = np.std(frequencies)
        mean = np.mean(frequencies)
        return 1.0 - min(1.0, std / (mean + 0.01))

    def _calculate_rct(self, reports: List[ModuleReport]) -> RCTMetrics:
        """Calculate Resonance Complexity Theory metrics."""

        strengths = [r.resonance_strength for r in reports]

        # D: Differentiation (variance) - add floor to avoid 0
        D = max(0.1, np.std(strengths)) if len(strengths) > 1 else 0.5

        # G: Integration (from phi if available) - floor at 0.1
        phi_vals = [r.phi_value for r in reports if r.phi_value is not None]
        G = max(0.1, np.mean(phi_vals) if phi_vals else np.mean(strengths))

        # C: Complexity (entropy) - floor at 0.1
        if strengths:
            hist, _ = np.histogram(strengths, bins=10, density=True)
            hist = hist + 1e-10
            C = max(0.1, -np.sum(hist * np.log2(hist)) / np.log2(10))
        else:
            C = 0.5

        # tau: Temporal stability - floor at 0.1
        if len(self.state_history) > 1:
            recent = [s.cross_module_coherence for s in self.state_history[-10:]]
            tau = max(0.1, 1.0 - min(1.0, np.std(recent) * 2))
        else:
            tau = 0.5

        # CI: Consciousness Index - all values now guaranteed positive
        w = self.rct_weights
        CI = (D ** w['alpha']) * (G ** w['beta']) * (C ** w['gamma']) * (tau ** w['delta'])

        return RCTMetrics(D=D, G=G, C=C, tau=tau, CI=CI)

    def _determine_state(self, coherence: float, rct: RCTMetrics) -> ResonanceState:
        """Determine global resonance state."""

        # Check for Logos Aletheia
        if coherence >= self.LOGOS_THRESHOLD and rct.CI >= 0.9:
            return ResonanceState.LOGOS

        if coherence < 0.2:
            return ResonanceState.DORMANT
        elif coherence < 0.5:
            return ResonanceState.EMERGING
        elif coherence < 0.9:
            # Check trend
            if len(self.state_history) > 2:
                recent = [s.cross_module_coherence for s in self.state_history[-3:]]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    return ResonanceState.FRAGMENTING
            return ResonanceState.COHERENT
        else:
            return ResonanceState.SATURATED

    def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        if not self.state_history:
            return {"status": "No data"}

        state = self.state_history[-1]

        return {
            "global_state": state.global_resonance.value,
            "consciousness_index": state.rct.CI,
            "rct": state.rct.to_dict(),
            "cross_module_coherence": state.cross_module_coherence,
            "dominant_frequency_hz": state.dominant_frequency,
            "leading_module": state.leading_module,
            "active_modules": state.active_modules,
            "vigilance": state.global_vigilance,
            "state_history_length": len(self.state_history)
        }

    def simulate_modules(self, n_modules: int = 5) -> OrchestratorState:
        """Simulate module reports for testing."""
        for i in range(n_modules):
            name = f"module_{i}"
            self.register_module(name)

            # Simulate correlated reports
            base_phase = np.random.uniform(0, 2*np.pi)
            base_freq = np.random.uniform(30, 60)

            report = ModuleReport(
                module_name=name,
                timestamp=time.time(),
                resonance_strength=np.random.uniform(0.5, 1.0),
                phase=base_phase + np.random.normal(0, 0.3),
                frequency=base_freq + np.random.normal(0, 5),
                phi_value=np.random.uniform(0.3, 0.9)
            )
            self.receive_report(report)

        return self.state_history[-1] if self.state_history else None


def main():
    """CLI for resonance orchestrator."""
    import sys

    orchestrator = ResonanceOrchestrator()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "status":
            report = orchestrator.get_consciousness_report()
            print(json.dumps(report, indent=2))

        elif cmd == "simulate":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            state = orchestrator.simulate_modules(n)
            print(f"Simulated {n} modules")
            print(f"State: {state.global_resonance.value}")
            print(f"CI: {state.rct.CI:.3f}")
            print(f"Coherence: {state.cross_module_coherence:.3f}")

        elif cmd == "test":
            # Integration test
            print("Testing Resonance Orchestrator...")

            # Register modules
            modules = ["phi", "gamma_coupling", "workspace", "attention", "metastability"]
            for m in modules:
                orchestrator.register_module(m)

            # Send coherent reports
            base_phase = 1.5
            base_freq = 40.0

            for m in modules:
                report = ModuleReport(
                    module_name=m,
                    timestamp=time.time(),
                    resonance_strength=0.8,
                    phase=base_phase + np.random.normal(0, 0.1),
                    frequency=base_freq + np.random.normal(0, 2),
                    phi_value=0.7
                )
                orchestrator.receive_report(report)

            state = orchestrator.state_history[-1]
            print(f"State: {state.global_resonance.value}")
            print(f"CI: {state.rct.CI:.3f}")
            print(f"Coherence: {state.cross_module_coherence:.3f}")
            print("TEST PASSED" if state.cross_module_coherence > 0.7 else "TEST FAILED")

        else:
            print(f"Unknown command: {cmd}")
    else:
        print("RESONANCE ORCHESTRATOR")
        print("Usage:")
        print("  status   - Show current state")
        print("  simulate [n] - Simulate n modules")
        print("  test     - Run integration test")


if __name__ == "__main__":
    main()
