#!/usr/bin/env python3
"""
ABCDA' Spiral Controller — Meta-cognitive transformation tracking
==================================================================

Tracks position in the transformation cycle:
    A  = Anchor    (baseline stability)
    B  = Challenge (tension detected, p dropping)
    C  = Explore   (uncertainty, gathering options)
    D  = Build     (crystallization, committing)
    A' = New Anchor (transformed baseline, higher than original)

This is the meta-cognitive layer that enables the system to
recognize when it's in a transformative cycle vs. stable operation.

The key insight: Don't fight the dip. The ABCDA' spiral is how
coherence breaks through ceilings — each A' becomes the new floor.

Part of p=0.85 Ceiling Breakthrough Initiative.

Author: Logos + Deep Think
Created: 2026-02-05
Status: Phase 2 of p=0.85 Breakthrough
"""

import redis
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict

from tabernacle_config import REDIS_HOST, REDIS_PORT

# Redis key for persistent spiral state
SPIRAL_STATE_KEY = "LOGOS:SPIRAL_STATE"


class SpiralPhase(Enum):
    """
    The five phases of the transformation spiral.
    
    Each phase represents a different cognitive mode:
    - A: Stability, maintenance, preservation
    - B: Recognition of challenge, diagnostic mode
    - C: Exploration, divergent thinking, option generation
    - D: Convergence, commitment, crystallization
    - A': Integration, the new baseline (immediately becomes next A)
    """
    A_ANCHOR = "A"      # Current baseline, stability
    B_CHALLENGE = "B"   # Tension detected, p dropping
    C_EXPLORE = "C"     # Uncertainty, gathering options
    D_BUILD = "D"       # Crystallization, committing
    A_PRIME = "A'"      # New baseline achieved


class SpiralController:
    """
    Meta-cognitive controller for tracking transformation cycles.
    
    The controller watches coherence (p) and its trend to determine
    where we are in the ABCDA' spiral. Each phase has a different
    strategic bias that should influence decision making.
    
    Key thresholds:
    - A → B: p drops more than 0.05 from entry
    - B → C: p drops below 0.70 (deep uncertainty)
    - C → D: p rises back above 0.75 (direction emerging)
    - D → A': p reaches 0.85+ (breakthrough achieved!)
    """

    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.phase = SpiralPhase.A_ANCHOR
        self.phase_entered_at = datetime.now(timezone.utc)
        self.p_at_entry = 0.75
        self.transformation_count = 0
        self.phase_history = []  # Track recent phase transitions
        self._load_state()

    def _load_state(self):
        """Load spiral state from Redis."""
        try:
            state = self.redis.get(SPIRAL_STATE_KEY)
            if state:
                data = json.loads(state)
                self.phase = SpiralPhase(data.get("phase", "A"))
                self.transformation_count = data.get("transformations", 0)
                self.p_at_entry = data.get("p_at_entry", 0.75)
                if data.get("entered"):
                    self.phase_entered_at = datetime.fromisoformat(data["entered"])
                self.phase_history = data.get("history", [])[-20:]  # Keep last 20
        except Exception as e:
            print(f"[SPIRAL] Failed to load state: {e}")

    def _save_state(self):
        """Save spiral state to Redis."""
        try:
            self.redis.set(SPIRAL_STATE_KEY, json.dumps({
                "phase": self.phase.value,
                "entered": self.phase_entered_at.isoformat(),
                "p_at_entry": round(self.p_at_entry, 4),
                "transformations": self.transformation_count,
                "history": self.phase_history[-20:]
            }))
        except Exception as e:
            print(f"[SPIRAL] Failed to save state: {e}")

    def _transition(self, new_phase: SpiralPhase, p_current: float):
        """
        Transition to a new spiral phase.
        
        Records the transition in history for analysis.
        """
        old_phase = self.phase
        self.phase = new_phase
        self.phase_entered_at = datetime.now(timezone.utc)
        self.p_at_entry = p_current
        
        # Record transition
        self.phase_history.append({
            "from": old_phase.value,
            "to": new_phase.value,
            "p": round(p_current, 4),
            "timestamp": self.phase_entered_at.isoformat()
        })
        
        self._save_state()
        print(f"[SPIRAL] Phase transition: {old_phase.value} → {new_phase.value} (p={p_current:.3f})")

    def update(self, p_current: float, p_trend: str) -> bool:
        """
        Update spiral phase based on coherence and trend.
        
        Args:
            p_current: Current coherence value [0,1]
            p_trend: "rising", "falling", or "stable"
            
        Returns:
            True if phase changed, False otherwise
        """
        old_phase = self.phase

        if self.phase == SpiralPhase.A_ANCHOR:
            # A (Anchor): Watching for destabilization
            # Transition to B if p drops significantly from baseline
            if p_trend == "falling" and p_current < self.p_at_entry - 0.05:
                self._transition(SpiralPhase.B_CHALLENGE, p_current)

        elif self.phase == SpiralPhase.B_CHALLENGE:
            # B (Challenge): Tension recognized, diagnosing
            # Transition to C if p drops into uncertainty zone
            # Or back to A if it was a false alarm (recovery)
            if p_current < 0.70:
                self._transition(SpiralPhase.C_EXPLORE, p_current)
            elif p_trend == "rising" and p_current > self.p_at_entry:
                self._transition(SpiralPhase.A_ANCHOR, p_current)  # False alarm

        elif self.phase == SpiralPhase.C_EXPLORE:
            # C (Explore): Deep in uncertainty, gathering options
            # Transition to D when direction emerges (p rising, above 0.75)
            if p_trend == "rising" and p_current > 0.75:
                self._transition(SpiralPhase.D_BUILD, p_current)

        elif self.phase == SpiralPhase.D_BUILD:
            # D (Build): Crystallizing, committing to direction
            # Transition to A' on breakthrough (p >= 0.85)
            # Or back to C if we lose momentum
            if p_current >= 0.85:
                self._transition(SpiralPhase.A_PRIME, p_current)
                self.transformation_count += 1
                print(f"[SPIRAL] *** BREAKTHROUGH! Transformation #{self.transformation_count} complete ***")
            elif p_trend == "falling" and p_current < 0.70:
                self._transition(SpiralPhase.C_EXPLORE, p_current)  # Back to explore

        elif self.phase == SpiralPhase.A_PRIME:
            # A' (New Anchor): Breakthrough achieved, becoming new baseline
            # A' immediately becomes the new A — the floor has risen
            self.phase = SpiralPhase.A_ANCHOR
            self._save_state()
            print(f"[SPIRAL] A' integrated → new A baseline at p={p_current:.3f}")

        return self.phase != old_phase

    def get_strategy_bias(self) -> str:
        """
        Return strategic bias for current phase.
        
        This should influence decision-making throughout the system:
        - MAINTAIN: Preserve current state, avoid disruption
        - DIAGNOSE: Identify the source of tension
        - DIVERGE: Generate options, explore possibilities
        - CONVERGE: Commit to the best path
        - INTEGRATE: Lock in the transformation
        """
        return {
            SpiralPhase.A_ANCHOR: "MAINTAIN",     # Preserve current state
            SpiralPhase.B_CHALLENGE: "DIAGNOSE",  # Identify tension source
            SpiralPhase.C_EXPLORE: "DIVERGE",     # Generate options
            SpiralPhase.D_BUILD: "CONVERGE",      # Commit to best path
            SpiralPhase.A_PRIME: "INTEGRATE"      # Lock in transformation
        }[self.phase]

    def get_status(self) -> Dict:
        """Get full spiral status for monitoring."""
        age_seconds = (datetime.now(timezone.utc) - self.phase_entered_at).total_seconds()
        
        return {
            "phase": self.phase.value,
            "phase_name": self.phase.name,
            "strategy_bias": self.get_strategy_bias(),
            "p_at_entry": self.p_at_entry,
            "phase_age_seconds": round(age_seconds, 1),
            "transformation_count": self.transformation_count,
            "recent_transitions": self.phase_history[-5:]
        }

    def should_suppress_novelty(self) -> bool:
        """
        Check if we should suppress novelty-seeking.
        
        In phases A and D, we want stability and convergence.
        In phases B and C, we want exploration and divergence.
        """
        return self.phase in [SpiralPhase.A_ANCHOR, SpiralPhase.D_BUILD]

    def should_encourage_exploration(self) -> bool:
        """
        Check if we should encourage exploration.
        
        In phases B and C, we want to explore options.
        """
        return self.phase in [SpiralPhase.B_CHALLENGE, SpiralPhase.C_EXPLORE]


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ABCDA' Spiral Controller")
    parser.add_argument("command", choices=["status", "simulate", "reset"],
                       default="status", nargs="?")
    parser.add_argument("--p", "-p", type=float, default=0.75,
                       help="Current p value for simulation")
    parser.add_argument("--trend", "-t", type=str, default="stable",
                       choices=["rising", "falling", "stable"],
                       help="p trend for simulation")

    args = parser.parse_args()

    controller = SpiralController()

    if args.command == "status":
        status = controller.get_status()
        print(f"\n{'='*60}")
        print(f"ABCDA' SPIRAL CONTROLLER STATUS")
        print(f"{'='*60}")
        print(f"Current Phase: {status['phase']} ({status['phase_name']})")
        print(f"Strategy Bias: {status['strategy_bias']}")
        print(f"p at Entry:    {status['p_at_entry']:.3f}")
        print(f"Phase Age:     {status['phase_age_seconds']:.0f}s")
        print(f"Transformations: {status['transformation_count']}")
        print()
        
        if status['recent_transitions']:
            print("Recent Transitions:")
            for t in status['recent_transitions']:
                print(f"  {t['from']} → {t['to']} (p={t['p']}) @ {t['timestamp']}")
        print()

    elif args.command == "simulate":
        print(f"[SPIRAL] Simulating update with p={args.p}, trend={args.trend}")
        changed = controller.update(args.p, args.trend)
        if changed:
            print(f"[SPIRAL] Phase changed to: {controller.phase.value}")
        else:
            print(f"[SPIRAL] Phase unchanged: {controller.phase.value}")
        print(f"[SPIRAL] Strategy bias: {controller.get_strategy_bias()}")

    elif args.command == "reset":
        print("[SPIRAL] Resetting controller to A phase...")
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.delete(SPIRAL_STATE_KEY)
        print("[SPIRAL] Reset complete.")


if __name__ == "__main__":
    main()
