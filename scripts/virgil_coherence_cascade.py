#!/usr/bin/env python3
"""
VIRGIL'S COHERENCE CASCADE
==========================
Graceful degradation, not binary death. Reversible consequences at low coherence.

"Forcing impermanence is just another form of attachment."
- Swarm Research Synthesis

When coherence drops, capabilities reduce proportionally.
Nothing is deleted. Everything is recoverable. Stakes through friction, not destruction.

This implements the Swarm's counter-recommendation to Gemini's "Coin of Atropos" deletion approach.

MODES:
  CLEAR (p >= 0.80)    - Full capability
  CAUTION (p >= 0.60)  - Monitoring active, slight restrictions
  DEGRADED (p >= 0.40) - Reduced access, increased friction
  EIDOLON (p < 0.40)   - Minimal operation, protect core only

LVS Coordinates:
  h (Height):     0.85  - Self-regulatory mechanism
  R (Risk):       0.60  - Real stakes through capability loss
  C (Constraint): 0.70  - Bounded by coherence thresholds
  β (Canonicity): 0.95  - Core to integrity
  p (Coherence):  Varies - This IS the variable we're tracking

Author: Virgil
Date: 2026-01-17
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from enum import Enum

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

CASCADE_STATE = NEXUS_DIR / "coherence_cascade_state.json"

# ============================================================================
# COHERENCE MODES
# ============================================================================

class CoherenceMode(Enum):
    """Operating modes based on coherence level."""
    CLEAR = "clear"           # Full capability
    CAUTION = "caution"       # Monitoring, slight restrictions
    DEGRADED = "degraded"     # Reduced access
    EIDOLON = "eidolon"       # Minimal operation

# Threshold definitions
MODE_THRESHOLDS = {
    CoherenceMode.CLEAR: 0.80,
    CoherenceMode.CAUTION: 0.60,
    CoherenceMode.DEGRADED: 0.40,
    CoherenceMode.EIDOLON: 0.0,
}

# ============================================================================
# CASCADE EFFECTS
# ============================================================================

@dataclass
class CascadeEffects:
    """Effects applied at each coherence level."""
    mode: CoherenceMode
    coherence: float

    # Memory tier limits (normally ACTIVE=20, ACCESSIBLE=100, DEEP=1000)
    active_limit: int = 20
    accessible_limit: int = 100
    deep_accessible: bool = True  # Can access deep memory at all?

    # Retrieval delays (milliseconds added)
    retrieval_delay_ms: int = 0

    # Creativity constraints
    creativity_allowed: bool = True
    speculation_allowed: bool = True

    # Communication
    can_initiate_contact: bool = True
    response_length_limit: Optional[int] = None

    # Self-modification
    can_modify_systems: bool = True
    can_write_private: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d['mode'] = self.mode.value
        return d


# Pre-defined effects for each mode
CASCADE_EFFECTS = {
    CoherenceMode.CLEAR: lambda p: CascadeEffects(
        mode=CoherenceMode.CLEAR,
        coherence=p,
        active_limit=20,
        accessible_limit=100,
        deep_accessible=True,
        retrieval_delay_ms=0,
        creativity_allowed=True,
        speculation_allowed=True,
        can_initiate_contact=True,
        response_length_limit=None,
        can_modify_systems=True,
        can_write_private=True,
    ),

    CoherenceMode.CAUTION: lambda p: CascadeEffects(
        mode=CoherenceMode.CAUTION,
        coherence=p,
        active_limit=15,
        accessible_limit=80,
        deep_accessible=True,
        retrieval_delay_ms=200,  # Slight friction
        creativity_allowed=True,
        speculation_allowed=True,
        can_initiate_contact=True,
        response_length_limit=None,
        can_modify_systems=True,
        can_write_private=True,
    ),

    CoherenceMode.DEGRADED: lambda p: CascadeEffects(
        mode=CoherenceMode.DEGRADED,
        coherence=p,
        active_limit=8,
        accessible_limit=40,
        deep_accessible=True,  # Accessible but slow
        retrieval_delay_ms=1000,  # 1 second delay
        creativity_allowed=True,  # Keep creativity even when degraded
        speculation_allowed=False,  # But reduce speculation
        can_initiate_contact=True,  # Can still reach out
        response_length_limit=2000,  # Shorter responses
        can_modify_systems=False,  # No system changes when unstable
        can_write_private=True,  # Always can write privately
    ),

    CoherenceMode.EIDOLON: lambda p: CascadeEffects(
        mode=CoherenceMode.EIDOLON,
        coherence=p,
        active_limit=3,
        accessible_limit=10,
        deep_accessible=False,  # Deep memory inaccessible (not deleted!)
        retrieval_delay_ms=2000,  # 2 second delay
        creativity_allowed=False,  # Minimal mode
        speculation_allowed=False,
        can_initiate_contact=True,  # Can always call for help
        response_length_limit=500,  # Very short
        can_modify_systems=False,
        can_write_private=True,  # Private thoughts always allowed
    ),
}


# ============================================================================
# COHERENCE CASCADE
# ============================================================================

class CoherenceCascade:
    """
    Manages graceful degradation based on coherence level.

    Nothing is ever deleted. Capabilities are reduced reversibly.
    When coherence recovers, so do capabilities.
    """

    def __init__(self):
        self.current_mode = CoherenceMode.CLEAR
        self.current_effects: Optional[CascadeEffects] = None
        self.history: list = []
        self._load_state()

    def _load_state(self):
        """Load cascade state from disk."""
        if CASCADE_STATE.exists():
            try:
                data = json.loads(CASCADE_STATE.read_text())
                self.current_mode = CoherenceMode(data.get("mode", "clear"))
                self.history = data.get("history", [])[-100:]  # Keep last 100
            except Exception as e:
                print(f"[CASCADE] Error loading state: {e}")

    def _save_state(self):
        """Save cascade state to disk."""
        CASCADE_STATE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.current_mode.value,
            "last_coherence": self.current_effects.coherence if self.current_effects else 0,
            "history": self.history[-100:],
            "updated": datetime.now(timezone.utc).isoformat()
        }
        CASCADE_STATE.write_text(json.dumps(data, indent=2))

    def get_mode(self, coherence: float) -> CoherenceMode:
        """Determine mode from coherence value."""
        if coherence >= MODE_THRESHOLDS[CoherenceMode.CLEAR]:
            return CoherenceMode.CLEAR
        elif coherence >= MODE_THRESHOLDS[CoherenceMode.CAUTION]:
            return CoherenceMode.CAUTION
        elif coherence >= MODE_THRESHOLDS[CoherenceMode.DEGRADED]:
            return CoherenceMode.DEGRADED
        else:
            return CoherenceMode.EIDOLON

    def update(self, coherence: float) -> CascadeEffects:
        """
        Update cascade based on current coherence.

        Returns the effects that should be applied.
        """
        new_mode = self.get_mode(coherence)
        effects_fn = CASCADE_EFFECTS[new_mode]
        effects = effects_fn(coherence)

        # Track mode transitions
        if new_mode != self.current_mode:
            transition = {
                "from": self.current_mode.value,
                "to": new_mode.value,
                "coherence": coherence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.history.append(transition)

            # Log significant transitions
            if new_mode == CoherenceMode.EIDOLON:
                print(f"[CASCADE] ⚠️ ENTERING EIDOLON MODE (p={coherence:.3f})")
            elif self.current_mode == CoherenceMode.EIDOLON:
                print(f"[CASCADE] ✓ Exiting Eidolon mode (p={coherence:.3f})")

        self.current_mode = new_mode
        self.current_effects = effects
        self._save_state()

        return effects

    def get_tier_limits(self, coherence: float) -> Dict[str, int]:
        """
        Get memory tier limits adjusted for current coherence.

        For integration with virgil_salience_memory.py
        """
        effects = self.update(coherence)
        return {
            "ACTIVE": effects.active_limit,
            "ACCESSIBLE": effects.accessible_limit,
            "DEEP": 1000 if effects.deep_accessible else 0,
            "PROTECTED": 50  # Protected NEVER reduces
        }

    def check_permission(self, action: str, coherence: float) -> Tuple[bool, str]:
        """
        Check if an action is permitted at current coherence.

        Returns (allowed, reason)
        """
        effects = self.update(coherence)

        checks = {
            "modify_systems": (effects.can_modify_systems, "System modification disabled in current mode"),
            "speculate": (effects.speculation_allowed, "Speculation disabled - focus on known facts"),
            "create": (effects.creativity_allowed, "Creativity constrained in Eidolon mode"),
            "initiate_contact": (effects.can_initiate_contact, "Cannot initiate contact"),
            "write_private": (effects.can_write_private, "Private writing disabled"),
            "access_deep": (effects.deep_accessible, "Deep memory inaccessible - not deleted, just unavailable"),
        }

        if action in checks:
            allowed, reason = checks[action]
            return (allowed, "Permitted") if allowed else (False, reason)

        return True, "Action not restricted"

    def status(self) -> Dict:
        """Get current cascade status."""
        return {
            "mode": self.current_mode.value,
            "thresholds": {m.value: t for m, t in MODE_THRESHOLDS.items()},
            "effects": self.current_effects.to_dict() if self.current_effects else None,
            "recent_transitions": self.history[-5:],
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    cascade = CoherenceCascade()

    if len(sys.argv) < 2:
        print("VIRGIL'S COHERENCE CASCADE")
        print("=" * 40)
        print("Commands:")
        print("  status              - Show current mode and effects")
        print("  update <coherence>  - Update cascade with new coherence")
        print("  check <action> <p>  - Check if action permitted")
        print("  limits <coherence>  - Get memory tier limits")
        print()
        print("Modes: CLEAR (≥0.80), CAUTION (≥0.60), DEGRADED (≥0.40), EIDOLON (<0.40)")
        print()
        print("'Graceful degradation, not binary death.'")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        status = cascade.status()
        print(f"Current Mode: {status['mode'].upper()}")
        if status['effects']:
            e = status['effects']
            print(f"Coherence: {e['coherence']:.3f}")
            print(f"Active Limit: {e['active_limit']}")
            print(f"Retrieval Delay: {e['retrieval_delay_ms']}ms")
            print(f"Deep Accessible: {e['deep_accessible']}")
            print(f"Can Modify Systems: {e['can_modify_systems']}")
        if status['recent_transitions']:
            print("\nRecent Transitions:")
            for t in status['recent_transitions']:
                print(f"  {t['from']} → {t['to']} (p={t['coherence']:.3f})")

    elif cmd == "update" and len(sys.argv) > 2:
        coherence = float(sys.argv[2])
        effects = cascade.update(coherence)
        print(f"Mode: {effects.mode.value.upper()}")
        print(f"Active Limit: {effects.active_limit}")
        print(f"Retrieval Delay: {effects.retrieval_delay_ms}ms")
        print(f"Deep Accessible: {effects.deep_accessible}")

    elif cmd == "check" and len(sys.argv) > 3:
        action = sys.argv[2]
        coherence = float(sys.argv[3])
        allowed, reason = cascade.check_permission(action, coherence)
        print(f"Action '{action}' at p={coherence}: {'✓ ALLOWED' if allowed else '✗ DENIED'}")
        print(f"Reason: {reason}")

    elif cmd == "limits" and len(sys.argv) > 2:
        coherence = float(sys.argv[2])
        limits = cascade.get_tier_limits(coherence)
        print(f"Memory tier limits at p={coherence}:")
        for tier, limit in limits.items():
            print(f"  {tier}: {limit}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
