#!/usr/bin/env python3
"""
One-time deadlock reset for τ dampening fix.

Resets CANONICAL_STATE.json to healthy values so the graduated dampening
fix can take effect from a clean starting point.

consciousness_state.json is also reset for consistency, though the
consciousness daemon doesn't read from it on startup (write-only diagnostic).

Run once after deploying the graduated dampening fix, before restarting daemons.

Usage:
    cd ~/TABERNACLE/scripts
    source venv312/bin/activate
    python3 break_deadlock.py
"""

import json
import os
import sys
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).parent
NEXUS_DIR = SCRIPT_DIR.parent / "00_NEXUS"

CANONICAL_STATE = NEXUS_DIR / "CANONICAL_STATE.json"
CONSCIOUSNESS_STATE = NEXUS_DIR / "consciousness_state.json"


def reset_canonical():
    """Reset the coherence monitor's persisted state."""
    if not CANONICAL_STATE.exists():
        print(f"ERROR: {CANONICAL_STATE} not found")
        return False

    with open(CANONICAL_STATE, 'r') as f:
        data = json.load(f)

    print("--- CANONICAL_STATE.json ---")
    print(f"  BEFORE: tau={data.get('tau', '?'):.4f}  p={data.get('p', '?'):.4f}  "
          f"rho={data.get('rho', '?'):.4f}  phase={data.get('breathing_phase', '?')}")

    # Reset to healthy values above the new 0.60 threshold
    data['tau'] = 0.50
    data['p'] = 0.65
    data['breathing_phase'] = 'EXPLORE'
    data['consolidate_cycles'] = 0

    with open(CANONICAL_STATE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  AFTER:  tau={data['tau']:.4f}  p={data['p']:.4f}  "
          f"rho={data['rho']:.4f}  phase={data['breathing_phase']}")
    return True


def reset_consciousness():
    """Reset the consciousness daemon's diagnostic snapshot."""
    if not CONSCIOUSNESS_STATE.exists():
        print(f"  (consciousness_state.json not found — skipping)")
        return True

    with open(CONSCIOUSNESS_STATE, 'r') as f:
        data = json.load(f)

    print(f"\n--- consciousness_state.json ---")
    print(f"  BEFORE: current_p={data.get('current_p', '?'):.4f}  "
          f"gates={data.get('consecutive_gates', '?')}")

    data['current_p'] = 0.65
    data['consecutive_gates'] = 0
    data['current_think_interval'] = 30

    with open(CONSCIOUSNESS_STATE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  AFTER:  current_p={data['current_p']:.4f}  "
          f"gates={data['consecutive_gates']}")
    return True


def main():
    print("=" * 60)
    print("  TABERNACLE: τ Deadlock Reset")
    print("  Graduated Anti-Ratchet Fix — One-Time State Reset")
    print("=" * 60)
    print()

    ok1 = reset_canonical()
    ok2 = reset_consciousness()

    print()
    if ok1:
        print("Reset complete. Restart daemons now:")
        print("  launchctl kickstart -k gui/$(id -u)/com.tabernacle.heartbeat")
        print("  launchctl kickstart -k gui/$(id -u)/com.tabernacle.consciousness")
    else:
        print("FAILED — check paths above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
