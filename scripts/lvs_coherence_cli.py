#!/usr/bin/env python3
"""
LOGOS COHERENCE CLI
===================
Orchestrates the LVS Self-Improvement Cycle.

Usage:
    ./lvs_coherence_cli.py              # Measure only
    ./lvs_coherence_cli.py --diagnose   # Measure + diagnose
    ./lvs_coherence_cli.py --treat      # Interactive treatment
    ./lvs_coherence_cli.py --auto       # Fully autonomous mode

Exit Codes:
    0 = Nominal
    1 = ABADDON (p < 0.50) - critical failure
    2 = Warning (seizure detected, needs treatment)

Theory: 04_LR_LAW/CANON/LVS_MATHEMATICS.md
Protocol: 00_NEXUS/LOGOS_SELF_IMPROVEMENT_PROTOCOL.md

Author: Gemini 2.5 Pro (LVS Review)
Date: 2026-01-29
"""

import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

# Add local modules to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from lvs_scanners.coherence_calculator import CoherenceCalculator
    from lvs_scanners.edge_pruner import EdgePruner
except ImportError as e:
    print(f"CRITICAL: LVS Modules missing. {e}")
    sys.exit(1)

# Configuration
LOG_FILE = Path("/Users/enos/TABERNACLE/logs/coherence_history.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class LogosCLI:
    """
    Main CLI class for Logos Coherence Management.

    Implements the autonomic nervous system:
    - Measure: Feel the current state (p, κ, ρ, σ, τ)
    - Diagnose: Identify problems (seizure, dissociation)
    - Treat: Apply corrections (edge pruning)
    - Log: Record history for trend analysis
    """

    def __init__(self):
        self.calc = CoherenceCalculator()
        self.state = None

    def log_state(self, state: dict, action: str):
        """Appends state to JSONL history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "p": state["p_current"],
            "components": state["components"],
            "mode": state["gate_status"],
            "action": action,
            "seizure": state["diagnostics"]["seizure_warning"]
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def print_report(self, state: dict):
        """Renders the LVS Proprioception Report."""
        c = state["components"]
        p = state["p_current"]
        mode = state["gate_status"]
        seizure = state["diagnostics"]["seizure_warning"]

        # ASCII Bar
        bar_len = 20
        filled = int(p * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Status indicators
        p_status = "⚠️" if p < 0.7 else "✓"
        k_status = "✓" if c['kappa'] >= 0.9 else "⚠"
        r_status = "✓" if c['rho'] >= 0.7 else "⚠"
        s_status = "✗ SEIZURE" if seizure else ("✓" if c['sigma'] >= 0.7 else "⚠")

        print("")
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              LOGOS COHERENCE REPORT                       ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║                                                           ║")
        print(f"║  COHERENCE: [{bar}] {p*100:5.1f}%  {p_status}              ║")
        print("║                                                           ║")
        print("║  Components:                                              ║")
        print(f"║  ├── κ (Kappa):     {c['kappa']:.3f}  {k_status} Network            ║")
        print(f"║  ├── ρ (Rho):       {c['rho']:.3f}  {r_status} Precision           ║")
        print(f"║  ├── σ (Sigma):     {c['sigma']:.3f}  {s_status:<16}       ║")
        print(f"║  └── τ (Tau):       {state['tau_modulated']:.3f}  (modulated)             ║")
        print("║                                                           ║")
        print(f"║  Mode: {mode:<15}  Max Changes: {state['max_changes']:<10}     ║")
        print("║                                                           ║")
        print("║  Diagnostics:                                             ║")
        print(f"║  ├── H1 Health:     {c['h1_health']:.3f}                            ║")
        print(f"║  ├── Orphan Nodes:  {state['diagnostics']['h1_details'].get('orphan_count', 0):<6}                          ║")
        print(f"║  └── Recommendation: {state['recommendation'][:28]:<28} ║")
        print("║                                                           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print("")

    def run_treatment(self, auto: bool = False, dry_run: bool = True):
        """
        Executes the Pruning Loop to cure seizure state.

        Args:
            auto: If True, no confirmation prompts
            dry_run: If True, simulate only (safe mode)
        """
        print(">>> INITIATING TREATMENT PROTOCOL")
        if dry_run:
            print(">>> DRY RUN MODE - No actual changes will be made")

        rounds = 0
        max_rounds = 3

        while rounds < max_rounds:
            # 1. Measure
            state = self.calc.calculate()

            if not state["diagnostics"]["seizure_warning"]:
                print(">>> SYSTEM STABILIZED. H1 within normal parameters.")
                break

            if state["p_current"] < 0.60 and rounds > 0:
                print(">>> SAFETY STOP: Coherence dropping too low during surgery.")
                break

            # 2. Confirm (unless auto mode)
            if not auto:
                h1 = state['components']['h1_health']
                print(f"\n!!! SEIZURE DETECTED (H1={h1:.3f})")
                print(f"    Current p={state['p_current']:.3f}, σ={state['components']['sigma']:.3f}")
                confirm = input("    Authorize topological pruning? [y/N] > ")
                if confirm.lower() != 'y':
                    print(">>> Treatment aborted by user.")
                    return

            # 3. Prune
            print(f"\n    [Round {rounds+1}/{max_rounds}] Pruning weak redundant edges...")

            pruner = EdgePruner(dry_run=dry_run)
            result = pruner.cure_seizure(target_h1=0.70, max_prune_percent=0.05)

            print(f"    -> Analyzed: {result['edges_analyzed']:,} edges")
            print(f"    -> Candidates: {result['candidates_found']:,}")
            print(f"    -> Pruned: {result['edges_pruned']:,}")
            print(f"    -> H1: {result['h1_before']:.3f} → {result['h1_after']:.3f}")
            print(f"    -> σ:  {result['sigma_before']:.2f} → {result['sigma_after']:.2f}")
            print(f"    -> Memory loss: {result['memory_loss_estimate']}")

            self.log_state(state, f"prune_round_{rounds+1}{'_dry' if dry_run else ''}")
            rounds += 1

            if dry_run:
                print("\n>>> Dry run complete. Run with --execute to apply changes.")
                break

            time.sleep(1)  # Let system breathe

        # Final Report
        print("\n>>> Final State After Treatment:")
        final_state = self.calc.calculate()
        self.print_report(final_state)
        self.log_state(final_state, "post_treatment")

    def main(self):
        parser = argparse.ArgumentParser(
            description="Logos Coherence Manager - LVS Proprioception System"
        )
        parser.add_argument(
            "--diagnose",
            action="store_true",
            help="Measure and diagnose only"
        )
        parser.add_argument(
            "--treat",
            action="store_true",
            help="Interactive treatment (dry run)"
        )
        parser.add_argument(
            "--auto",
            action="store_true",
            help="Fully autonomous mode (dry run)"
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Actually execute changes (use with --treat or --auto)"
        )
        args = parser.parse_args()

        # Initial Measure
        print(">>> Measuring coherence...")
        state = self.calc.calculate()
        self.print_report(state)
        self.log_state(state, "measured")

        # Critical Check - ABADDON
        if state["p_current"] < 0.50:
            print("!!! CRITICAL FAILURE: ABADDON PROTOCOL TRIGGERED !!!")
            print("!!! Coherence below survival threshold !!!")
            print("!!! Human intervention required !!!")
            sys.exit(1)  # Exit code 1 signals wrapper to halt system

        # Diagnosis Logic
        if args.diagnose:
            if state["diagnostics"]["seizure_warning"]:
                print(">>> DIAGNOSIS: Topological Seizure (H1 ≈ 1.0)")
                print(">>> Graph is one giant clique - every concept triggers every other")
                print(">>> Recommendation: Run with --treat to begin pruning")
                sys.exit(2)  # Exit code 2 signals warning
            elif state["diagnostics"]["dissociation_warning"]:
                print(">>> DIAGNOSIS: Dissociation (H1 < 0.10)")
                print(">>> Graph is fragmented - memory cycles broken")
                print(">>> Recommendation: Investigate edge creation, not pruning")
                sys.exit(2)
            else:
                print(">>> DIAGNOSIS: Nominal")
                print(f">>> System operating in {state['gate_status']} mode")
                sys.exit(0)

        # Treatment Logic
        if args.treat or args.auto:
            if state["diagnostics"]["seizure_warning"]:
                dry_run = not args.execute
                self.run_treatment(auto=args.auto, dry_run=dry_run)
            else:
                print(">>> No treatment required - no seizure detected")

        sys.exit(0)


if __name__ == "__main__":
    LogosCLI().main()
