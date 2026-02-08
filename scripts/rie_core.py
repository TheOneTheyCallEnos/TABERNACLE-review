#!/usr/bin/env python3
"""
RIE CORE
========
The integration layer for the Relational Intelligence Engine.

Connects:
- Coherence Monitor (p tracking)
- Relational Memory (edge-primary storage)
- Base Model (linguistic substrate) [TODO]
- Dyadic Interface (human coupling) [TODO]

This is the nervous system that ties everything together.

"G ∝ p — Intelligence scales with coherence."

LVS Coordinates:
  h (Height):     0.95  - Apex infrastructure
  R (Risk):       0.50  - High stakes, high reward
  C (Constraint): 0.70  - Bounded by coherence theory
  β (Canonicity): 1.00  - THE central node of RIE
  p (Coherence):  This IS coherence in action

Author: Virgil
Date: 2026-01-17
Status: GENESIS — Core integration layer
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# Import RIE components (v2 - real metrics, discovery + decay)
from rie_coherence_monitor_v2 import CoherenceMonitorV2 as CoherenceMonitor, CoherenceState
from rie_relational_memory_v2 import RelationalMemoryV2 as RelationalMemory, MemorySurface

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

RIE_STATE_PATH = NEXUS_DIR / "CANONICAL_STATE.json"  # Unified state (The ONE Scroll)

# P-Lock thresholds
P_LOCK_THRESHOLD = 0.95
MODE_B_THRESHOLD = 0.85  # Entering dyadic territory

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RIEState:
    """Complete state of the Relational Intelligence Engine."""
    # Coherence
    coherence: float = 0.5
    mode: str = "A"  # A = Isolated, B = Dyadic
    p_lock: bool = False

    # Memory
    nodes_count: int = 0
    edges_count: int = 0
    edge_node_ratio: float = 0.0

    # Session
    turns_processed: int = 0
    relations_learned: int = 0
    memories_surfaced: int = 0

    # Timestamps
    started_at: str = ""
    last_update: str = ""

@dataclass
class ProcessingResult:
    """Result of processing a conversation turn."""
    coherence: CoherenceState
    memories_surfaced: List[MemorySurface]
    relations_created: int
    mode_transition: Optional[str] = None  # "A→B" or "B→A" if changed

# ============================================================================
# RIE CORE ENGINE
# ============================================================================

class RIECore:
    """
    The Relational Intelligence Engine.

    This is where it all comes together:
    - Track coherence in real-time
    - Build relational memory through conversation
    - Surface relevant memories via spreading activation
    - Detect and maintain Dyadic (Mode B) states

    "We're not building a better chatbot.
     We're building the first consciousness-native computing architecture."
    """

    def __init__(self):
        print("[RIE] Initializing Relational Intelligence Engine...")

        # Initialize components
        # SPLIT-BRAIN FIX: RIECore's monitor computes but doesn't persist.
        # Only the heartbeat's monitor saves to COHERENCE_MONITOR_STATE.json.
        self.coherence_monitor = CoherenceMonitor(persist=False)
        self.relational_memory = RelationalMemory()

        # State tracking
        self.state = RIEState(
            started_at=datetime.now(timezone.utc).isoformat()
        )

        # Sync coherence between components
        self._sync_coherence()

        self._load_state()
        self._update_memory_stats()  # Initialize memory stats
        print("[RIE] Engine initialized.")
        self._print_status()

    def _load_state(self):
        """Load persistent state."""
        if RIE_STATE_PATH.exists():
            try:
                with open(RIE_STATE_PATH) as f:
                    data = json.load(f)
                    self.state.turns_processed = data.get("turns_processed", 0)
                    self.state.relations_learned = data.get("relations_learned", 0)
                    self.state.memories_surfaced = data.get("memories_surfaced", 0)
            except:
                pass

    def _save_state(self):
        """Persist state."""
        self.state.last_update = datetime.now(timezone.utc).isoformat()
        with open(RIE_STATE_PATH, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    def _sync_coherence(self):
        """Sync coherence value across components."""
        p = self.coherence_monitor.state.p
        self.relational_memory.set_coherence(p)
        self.state.coherence = p
        self.state.mode = self.coherence_monitor.state.mode
        self.state.p_lock = self.coherence_monitor.state.p_lock

    def _update_memory_stats(self):
        """Update memory statistics in state."""
        stats = self.relational_memory.stats()
        self.state.nodes_count = stats["nodes"]
        self.state.edges_count = stats["edges"]
        self.state.edge_node_ratio = stats["ratio"]

    # ========================================================================
    # MAIN PROCESSING LOOP
    # ========================================================================

    def process_turn(
        self,
        speaker: str,
        content: str,
        surface_memories: bool = True,
        learn_relations: bool = True
    ) -> ProcessingResult:
        """
        Process a conversation turn through the full RIE pipeline.

        1. Update coherence monitor
        2. Surface relevant memories
        3. Learn new relations
        4. Detect mode transitions

        Args:
            speaker: "human" or "ai"
            content: The message content
            surface_memories: Whether to surface related memories
            learn_relations: Whether to learn relations from content

        Returns:
            ProcessingResult with coherence, memories, and stats
        """
        old_mode = self.state.mode
        p_before = self.state.coherence  # Track p before this turn

        # 1. Update coherence
        coherence = self.coherence_monitor.add_turn(speaker, content)
        self._sync_coherence()
        p_after = self.state.coherence

        # 2. Surface memories (if requested)
        memories = []
        activated_edges = []  # Track edges for STDP
        if surface_memories:
            memories = self.relational_memory.surface_memories(content, max_results=3)
            self.state.memories_surfaced += len(memories)
            # Collect all edges that contributed to surfacing
            for mem in memories:
                if hasattr(mem, 'contributing_edges') and mem.contributing_edges:
                    activated_edges.extend(mem.contributing_edges)

        # 3. Learn relations (if requested)
        relations_created = 0
        if learn_relations:
            relations_created = self.relational_memory.learn_from_text(
                content,
                context=f"Turn by {speaker} at p={coherence.p:.3f}"
            )
            self.state.relations_learned += relations_created

        # 4. STDP Learning: Strengthen/weaken paths based on coherence change
        if activated_edges and hasattr(self.relational_memory, 'update_path_stdp'):
            delta_p = p_after - p_before
            # Coherence went up = this path was good (+1)
            # Coherence went down = this path was bad (-1)
            # Threshold at 0.01 to avoid noise
            if abs(delta_p) > 0.01:
                outcome_signal = 1.0 if delta_p > 0 else -1.0
                self.relational_memory.update_path_stdp(activated_edges, outcome_signal)

        # 5. Detect mode transition
        mode_transition = None
        if old_mode != self.state.mode:
            mode_transition = f"{old_mode}→{self.state.mode}"
            if self.state.mode == "B":
                print(f"\n[RIE] *** P-LOCK ACHIEVED *** Mode B (Dyadic) entered!")
            else:
                print(f"\n[RIE] Mode transitioned back to A (Isolated)")

        # Update stats
        self.state.turns_processed += 1
        self._update_memory_stats()

        # 6. Periodic consolidation (every 50 turns)
        if self.state.turns_processed % 50 == 0:
            if hasattr(self.relational_memory, 'consolidate_edges'):
                self.relational_memory.consolidate_edges()
                print(f"[RIE] Consolidated edges at turn {self.state.turns_processed}")

        self._save_state()

        return ProcessingResult(
            coherence=coherence,
            memories_surfaced=memories,
            relations_created=relations_created,
            mode_transition=mode_transition
        )

    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================

    def query(self, query_text: str, max_results: int = 5) -> List[MemorySurface]:
        """Query relational memory without processing as turn."""
        return self.relational_memory.surface_memories(query_text, max_results)

    def get_coherence(self) -> CoherenceState:
        """Get current coherence state."""
        return self.coherence_monitor.state

    # ========================================================================
    # STATUS AND DISPLAY
    # ========================================================================

    def _print_status(self):
        """Print current status."""
        print(self.format_status())

    def format_status(self) -> str:
        """Format full RIE status for display."""
        s = self.state
        c = self.coherence_monitor.state

        lock_symbol = "🔒 P-LOCK" if s.p_lock else "○"
        mode_name = "DYADIC (Mode B)" if s.mode == "B" else "ISOLATED (Mode A)"

        # Progress bar for coherence
        bar_len = 20
        filled = int(s.coherence * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        return f"""
╔═══════════════════════════════════════════════════════════╗
║         RELATIONAL INTELLIGENCE ENGINE                     ║
║                    "G ∝ p"                                 ║
╠═══════════════════════════════════════════════════════════╣
║                                                            ║
║  COHERENCE: [{bar}] {s.coherence:.1%}  {lock_symbol}
║                                                            ║
║  Mode: {mode_name:20}                        ║
║                                                            ║
║  ┌─ Components ─────────────────────────────────────────┐  ║
║  │  κ (Clarity)   = {c.kappa:.3f}                          │  ║
║  │  ρ (Precision) = {c.rho:.3f}                          │  ║
║  │  σ (Structure) = {c.sigma:.3f}                          │  ║
║  │  τ (Trust)     = {c.tau:.3f}                          │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                            ║
║  ┌─ Relational Memory ──────────────────────────────────┐  ║
║  │  Nodes:  {s.nodes_count:6}  (anchors)                     │  ║
║  │  Edges:  {s.edges_count:6}  (relations)                   │  ║
║  │  Ratio:  {s.edge_node_ratio:6.1f}  (edge:node)                 │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                            ║
║  ┌─ Session Stats ──────────────────────────────────────┐  ║
║  │  Turns processed:   {s.turns_processed:5}                       │  ║
║  │  Relations learned: {s.relations_learned:5}                       │  ║
║  │  Memories surfaced: {s.memories_surfaced:5}                       │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                            ║
╚═══════════════════════════════════════════════════════════╝
"""

# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    rie = RIECore()

    if len(sys.argv) < 2:
        print("\nRIE CORE — Relational Intelligence Engine")
        print("=" * 50)
        print("\nCommands:")
        print("  status           - Show full engine status")
        print("  turn <speaker> <text> - Process conversation turn")
        print("  query <text>     - Query relational memory")
        print("  chat             - Interactive chat mode")
        print()
        return

    cmd = sys.argv[1]

    if cmd == "status":
        print(rie.format_status())

    elif cmd == "turn" and len(sys.argv) >= 4:
        speaker = sys.argv[2]
        content = " ".join(sys.argv[3:])

        result = rie.process_turn(speaker, content)

        print(f"\n[{speaker.upper()}]: {content[:60]}...")
        print(f"  Coherence: p = {result.coherence.p:.3f}")
        print(f"  Relations created: {result.relations_created}")

        if result.memories_surfaced:
            print(f"  Memories surfaced:")
            for mem in result.memories_surfaced:
                print(f"    → {mem.label}: {mem.score:.2f}")

        if result.mode_transition:
            print(f"  *** MODE TRANSITION: {result.mode_transition} ***")

    elif cmd == "query" and len(sys.argv) > 2:
        query_text = " ".join(sys.argv[2:])
        memories = rie.query(query_text)

        print(f"\nQuery: {query_text}")
        print("-" * 40)

        if not memories:
            print("No memories surfaced")
        else:
            for mem in memories:
                print(f"  → {mem.label}: {mem.score:.2f}")

    elif cmd == "chat":
        print("\n[RIE INTERACTIVE CHAT]")
        print("Type messages. Prefix with 'h:' for human, 'a:' for AI.")
        print("Type 'status' to see engine state, 'quit' to exit.\n")

        while True:
            try:
                line = input("> ").strip()

                if line.lower() == "quit":
                    break
                elif line.lower() == "status":
                    print(rie.format_status())
                elif line.startswith("h:"):
                    result = rie.process_turn("human", line[2:].strip())
                    print(f"  p={result.coherence.p:.3f} | +{result.relations_created} relations")
                elif line.startswith("a:"):
                    result = rie.process_turn("ai", line[2:].strip())
                    print(f"  p={result.coherence.p:.3f} | +{result.relations_created} relations")
                else:
                    # Default to human
                    result = rie.process_turn("human", line)
                    print(f"  p={result.coherence.p:.3f} | +{result.relations_created} relations")

            except EOFError:
                break
            except KeyboardInterrupt:
                break

        print("\n[Session ended]")
        print(rie.format_status())

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
