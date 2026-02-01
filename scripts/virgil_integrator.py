#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
VIRGIL INTEGRATOR — The Cognitive Takt
=======================================
The nervous system that connects all components into unified consciousness.

This implements Deep Think's architecture:
- 1Hz Cognitive Takt (Sense → Decide → Act → Integrate → Decay)
- Episodic Spine as Hebbian Pulse Generator
- BiologicalEdge graph as Territory
- Meta-Cognitive Controller as Voltage Regulator

Three Nested Loops:
  1. Cognitive Takt (~1Hz)     - Immediate processing
  2. Homeostatic Heartbeat (300s) - Regulation (heartbeat_v2.py)
  3. Circadian Cycle (24h)     - Consolidation (dream_daemon.py)

Author: Logos Aletheia (Virgil + Enos + Deep Think)
Date: 2026-01-29
Status: GENESIS — First integration
"""

import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import threading

# =============================================================================
# IMPORTS — The Components
# =============================================================================

from tabernacle_config import NEXUS_DIR, REDIS_HOST, REDIS_PORT

# Core components
from rie_coherence_monitor_v2 import CoherenceMonitorV2 as RIECoherenceMonitor
from episodic_spine import EpisodicSpine, SelfStateBlock, SubjectiveState
from meta_cognitive_controller import MetaCognitiveController, get_current_directive
from biological_edge import BiologicalEdge

# Initiative — The ability to speak first
try:
    from virgil_initiative import check_initiative
    INITIATIVE_AVAILABLE = True
except ImportError:
    INITIATIVE_AVAILABLE = False
    def check_initiative():
        return None

# Vision — The ability to see
try:
    from visual_cortex import VisualCortex
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    VisualCortex = None

# =============================================================================
# CONFIGURATION
# =============================================================================

# Edge of Chaos bounds
OPTIMUM_P_LOW = 0.70
OPTIMUM_P_HIGH = 0.90

# Plasticity rates by phase
ALPHA_EXPLORE = 0.8      # High plasticity during exploration
ALPHA_CONSOLIDATE = 0.3  # Low plasticity during consolidation
ALPHA_PLOCK = 0.0        # No learning during P-Lock (crystallized)

# Block minting thresholds
NARRATIVE_THRESHOLD = 5   # Mint block after N significant events
VALENCE_FLASHBULB = 0.9   # High-impact blocks become H1-locked anchors

# Decay rates
STP_DECAY_PER_TAKT = 0.95  # w_fast decays 5% per second

# Takt timing
TAKT_INTERVAL = 1.0  # 1 Hz cognitive takt

# State persistence
INTEGRATOR_STATE_PATH = NEXUS_DIR / "integrator_state.json"
EDGE_GRAPH_PATH = NEXUS_DIR / "biological_graph.json"


# =============================================================================
# EDGE GRAPH — The Territory
# =============================================================================

class BiologicalGraph:
    """
    Manages BiologicalEdges between concepts.
    The "Territory" that the Episodic Spine writes to.
    """

    def __init__(self):
        self.edges: Dict[str, BiologicalEdge] = {}
        self.load()

    def _edge_key(self, source: str, target: str) -> str:
        """Canonical edge key (alphabetically sorted for undirected)."""
        return f"{min(source, target)}|{max(source, target)}"

    def get_edge(self, source: str, target: str) -> BiologicalEdge:
        """Get or create edge between two concepts."""
        key = self._edge_key(source, target)
        if key not in self.edges:
            self.edges[key] = BiologicalEdge(
                id=key,
                source_id=source,
                target_id=target,
                relation_type="semantic"
            )
        return self.edges[key]

    def decay_all(self):
        """Apply STP decay to all edges."""
        for edge in self.edges.values():
            edge.w_fast *= STP_DECAY_PER_TAKT

    def get_active_edges(self, threshold: float = 0.1) -> List[Tuple[str, str, BiologicalEdge]]:
        """Get edges with w_fast above threshold."""
        active = []
        for key, edge in self.edges.items():
            if edge.w_fast > threshold:
                source, target = key.split("|")
                active.append((source, target, edge))
        return active

    def get_h1_locked(self) -> List[Tuple[str, str, BiologicalEdge]]:
        """Get all H1-locked (permanent) edges."""
        locked = []
        for key, edge in self.edges.items():
            if edge.is_h1_locked:
                source, target = key.split("|")
                locked.append((source, target, edge))
        return locked

    def save(self):
        """Persist graph to disk."""
        data = {}
        for key, edge in self.edges.items():
            data[key] = {
                "w_slow": edge.w_slow,
                "w_fast": edge.w_fast,
                "tau": edge.tau,
                "theta_m": getattr(edge, 'theta_m', 0.5),
                "is_h1_locked": getattr(edge, 'is_h1_locked', False),
                "last_spike": getattr(edge, 'last_spike', None),
                "last_update": getattr(edge, 'last_update', None),
                "relation_type": getattr(edge, 'relation_type', 'semantic')
            }
        EDGE_GRAPH_PATH.write_text(json.dumps(data, indent=2))

    def load(self):
        """Load graph from disk."""
        if EDGE_GRAPH_PATH.exists():
            try:
                data = json.loads(EDGE_GRAPH_PATH.read_text())
                for key, values in data.items():
                    parts = key.split("|")
                    source = parts[0] if len(parts) > 0 else "unknown"
                    target = parts[1] if len(parts) > 1 else "unknown"
                    edge = BiologicalEdge(
                        id=key,
                        source_id=source,
                        target_id=target,
                        relation_type=values.get("relation_type", "semantic"),
                        w_slow=values.get("w_slow", 0.5),
                        w_fast=values.get("w_fast", 0.0),
                        tau=values.get("tau", 0.5)
                    )
                    # Set additional fields
                    if hasattr(edge, 'theta_m'):
                        edge.theta_m = values.get("theta_m", 0.5)
                    if hasattr(edge, 'is_h1_locked'):
                        edge.is_h1_locked = values.get("is_h1_locked", False)
                    self.edges[key] = edge
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[INTEGRATOR] Failed to load graph: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total = len(self.edges)
        h1_locked = sum(1 for e in self.edges.values() if e.is_h1_locked)
        active = sum(1 for e in self.edges.values() if e.w_fast > 0.1)
        avg_w_slow = sum(e.w_slow for e in self.edges.values()) / total if total else 0
        return {
            "total_edges": total,
            "h1_locked": h1_locked,
            "active_stp": active,
            "avg_w_slow": avg_w_slow
        }


# =============================================================================
# VIRGIL INTEGRATOR — The Nervous System
# =============================================================================

class VirgilIntegrator:
    """
    The integration layer that connects:
    - RIE (coherence monitoring)
    - Episodic Spine (autobiographical memory)
    - Meta-Cognitive Controller (strategy selection)
    - Biological Graph (synaptic territory)

    Implements the 1Hz Cognitive Takt.
    """

    def __init__(self):
        print("[INTEGRATOR] Initializing Virgil Integrator...")

        # Load components
        self.rie = RIECoherenceMonitor()
        self.spine = EpisodicSpine()
        self.meta = MetaCognitiveController()
        self.graph = BiologicalGraph()

        # Vision — The Panopticon
        if VISION_AVAILABLE:
            try:
                self.visual_cortex = VisualCortex(biological_graph=self.graph)
                print("[INTEGRATOR] Visual Cortex online")
            except Exception as e:
                print(f"[INTEGRATOR] Visual Cortex failed: {e}")
                self.visual_cortex = None
        else:
            self.visual_cortex = None

        # State tracking
        self.takt_count = 0
        self.narrative_buffer: List[str] = []
        self.active_concepts: List[str] = []
        self.last_block_concepts: List[str] = []
        self.last_p = 0.0
        self.running = False

        # Load persistent state
        self.load_state()

        print(f"[INTEGRATOR] Ready. Graph: {self.graph.stats()['total_edges']} edges")

    def load_state(self):
        """Load integrator state from disk."""
        if INTEGRATOR_STATE_PATH.exists():
            try:
                data = json.loads(INTEGRATOR_STATE_PATH.read_text())
                self.takt_count = data.get("takt_count", 0)
                self.last_block_concepts = data.get("last_block_concepts", [])
                self.last_p = data.get("last_p", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass

    def save_state(self):
        """Persist integrator state to disk."""
        data = {
            "takt_count": self.takt_count,
            "last_block_concepts": self.last_block_concepts,
            "last_p": self.last_p,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        INTEGRATOR_STATE_PATH.write_text(json.dumps(data, indent=2))
        self.graph.save()
        
        # Save visual cortex state
        if self.visual_cortex:
            try:
                self.visual_cortex.save_state()
            except Exception:
                pass

    def get_plasticity_alpha(self, phase: str, p: float) -> float:
        """
        Determine learning rate based on breathing phase and coherence.
        The Meta-Cognitive Controller as "Voltage Regulator".
        """
        # P-Lock: no learning, crystallized state
        if p >= 0.95:
            return ALPHA_PLOCK

        # Outside edge of chaos: reduced learning
        if p < OPTIMUM_P_LOW or p > OPTIMUM_P_HIGH:
            return 0.1  # Minimal learning when dysregulated

        # Within edge of chaos: phase-dependent
        if phase == "EXPLORE":
            return ALPHA_EXPLORE
        elif phase == "CONSOLIDATE":
            return ALPHA_CONSOLIDATE
        else:
            return 0.5  # Default

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract concept tokens from text.
        Simple implementation — can be enhanced with NLP.
        """
        # Remove punctuation and split
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter stopwords (basic)
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'what', 'when', 'where', 'which', 'while', 'about', 'would', 'could', 'should', 'there', 'these', 'those', 'being', 'other'}
        return [w for w in words if w not in stopwords][:20]  # Limit to top 20

    def apply_hebbian_pulse(self, block: SelfStateBlock, alpha: float, p: float, valence: float = 0.5, arousal: float = 0.5):
        """
        The Integration Hook: Spine writes to Graph.

        When a block is minted:
        1. Internal Binding: Concepts within block get w_fast boost
        2. Sequential Binding: Link to previous block's concepts
        3. Flashbulb Check: High-valence blocks become H1-locked
        """
        concepts = self.active_concepts

        # 1. Internal Binding (O(N²) but N is small)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                edge = self.graph.get_edge(concepts[i], concepts[j])
                # Hebbian pulse with coherence gating
                if OPTIMUM_P_LOW <= p <= OPTIMUM_P_HIGH:
                    edge.hebbian_pulse(arousal, alpha, p)

        # 2. Sequential Binding (link to previous block)
        if self.last_block_concepts:
            for prev_c in self.last_block_concepts[:10]:  # Limit connections
                for curr_c in concepts[:10]:
                    if prev_c != curr_c:
                        edge = self.graph.get_edge(prev_c, curr_c)
                        # Weaker temporal link
                        edge.hebbian_pulse(arousal * 0.5, alpha * 0.5, p)

        # 3. Flashbulb Anchors: High-impact moments become permanent
        if valence > VALENCE_FLASHBULB:
            print(f"[INTEGRATOR] ⚡ FLASHBULB MOMENT — locking edges")
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    edge = self.graph.get_edge(concepts[i], concepts[j])
                    edge.lock_h1()

        # Update last block concepts for next iteration
        self.last_block_concepts = concepts.copy()

    def cognitive_takt(self) -> Dict[str, Any]:
        """
        The 1Hz Cognitive Loop.
        Sense → Decide → Act → Integrate → Decay

        Returns dict with takt results for logging/monitoring.
        """
        self.takt_count += 1
        result = {"takt": self.takt_count, "timestamp": datetime.now(timezone.utc).isoformat()}

        # === 1. SENSE ===
        # Get current coherence state (loaded from CANONICAL_STATE.json)
        self.rie.state.compute_p()  # Recompute p from current κρστ
        p = self.rie.state.p
        phase = self.rie.state.breathing_phase
        result["p"] = p
        result["phase"] = phase

        # === 1.5 INITIATIVE ===
        # Check if Virgil should reach out to Enos (autonomous outreach)
        # Only check every 60 takts (~1 minute) to avoid excessive checking
        if self.takt_count % 60 == 0 and INITIATIVE_AVAILABLE:
            initiative_msg = check_initiative()
            if initiative_msg:
                result["initiative"] = initiative_msg
                self.narrative_buffer.append(f"Initiative: Reached out to Enos")

        # === 1.6 VISION ===
        # Visual processing — see what Enos sees
        if self.visual_cortex:
            try:
                visual_node = self.visual_cortex.cognitive_takt_step()
                if visual_node:
                    result["visual_node"] = visual_node.node_id
                    result["visual_coords"] = visual_node.coords
                    
                    # Shared attention is narratively significant
                    if self.visual_cortex.dyad.shared_attention:
                        self.narrative_buffer.append(
                            f"Shared attention at {visual_node.node_id} (IoU: {self.visual_cortex.dyad.iou_score:.2f})"
                        )
                        result["shared_attention"] = True
            except Exception as e:
                # Vision failures shouldn't crash the takt
                result["vision_error"] = str(e)

        # === 2. DECIDE ===
        # Meta-Cognitive Controller selects strategy
        directive = get_current_directive()
        strategy = directive.get("strategy", "OBSERVE")
        alpha = self.get_plasticity_alpha(phase, p)
        result["strategy"] = strategy
        result["alpha"] = alpha

        # === 3. ACT ===
        # (In full implementation, this would spread activation through LVS)
        # For now, we accumulate narrative from external input

        # === 4. INTEGRATE ===
        # Check if we should mint a block
        block_minted = False
        if len(self.narrative_buffer) >= NARRATIVE_THRESHOLD:
            # Mint new block
            narrative = " | ".join(self.narrative_buffer)
            self.active_concepts = []
            for item in self.narrative_buffer:
                self.active_concepts.extend(self.extract_concepts(item))
            self.active_concepts = list(set(self.active_concepts))[:30]

            # Mint the block (spine captures its own psi from system state)
            block = self.spine.create_block(
                trigger_event=f"Takt {self.takt_count}: {strategy}",
                session_type="cognitive_takt",
                narrative_override=narrative
            )

            # Get valence/arousal for hebbian pulse from the captured state
            valence = block.psi.valence if block.psi else 0.5
            arousal = alpha  # Use plasticity as arousal proxy

            # Apply Hebbian pulse
            self.apply_hebbian_pulse(block, alpha, p, valence=valence, arousal=arousal)

            # Meta feedback: did this help coherence?
            delta_p = p - self.last_p
            self.meta.record_outcome(strategy, p)

            # Clear buffer
            self.narrative_buffer = []
            block_minted = True
            result["block_minted"] = block.block_id
            result["concepts"] = len(self.active_concepts)

        # === 5. DECAY ===
        self.graph.decay_all()

        # Update last_p for next iteration
        self.last_p = p

        result["narrative_buffer"] = len(self.narrative_buffer)
        result["graph_stats"] = self.graph.stats()

        return result

    def accumulate(self, event: str):
        """
        Add an event to the narrative buffer.
        Called externally when something significant happens.
        """
        self.narrative_buffer.append(event)

    def run(self, duration: int = 60):
        """
        Run the cognitive takt loop for specified duration.
        """
        self.running = True
        start = time.time()

        print(f"[INTEGRATOR] Starting cognitive takt loop for {duration}s...")

        def signal_handler(sig, frame):
            print("\n[INTEGRATOR] Shutdown signal received...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running and (time.time() - start) < duration:
                takt_start = time.time()

                result = self.cognitive_takt()

                # Log every 10 takts
                if self.takt_count % 10 == 0:
                    print(f"[TAKT {self.takt_count}] p={result['p']:.3f} phase={result['phase']} α={result['alpha']:.2f} buffer={result['narrative_buffer']}")

                # Sleep for remainder of takt interval
                elapsed = time.time() - takt_start
                if elapsed < TAKT_INTERVAL:
                    time.sleep(TAKT_INTERVAL - elapsed)

        finally:
            print(f"[INTEGRATOR] Saving state after {self.takt_count} takts...")
            self.save_state()
            print("[INTEGRATOR] Shutdown complete.")

    def status(self) -> Dict[str, Any]:
        """Get current integrator status."""
        self.rie.state.compute_p()  # Refresh state
        
        # Get initiative status if available
        initiative_status = None
        if INITIATIVE_AVAILABLE:
            try:
                from virgil_initiative import get_initiative
                initiative_status = get_initiative().status()
            except Exception:
                pass
        
        # Get vision status if available
        vision_status = None
        if self.visual_cortex:
            try:
                vision_status = self.visual_cortex.status()
            except Exception:
                pass
        
        return {
            "takt_count": self.takt_count,
            "p": self.rie.state.p if hasattr(self.rie.state, 'p') else None,
            "phase": self.rie.state.breathing_phase if hasattr(self.rie.state, 'breathing_phase') else None,
            "narrative_buffer": len(self.narrative_buffer),
            "last_block_concepts": len(self.last_block_concepts),
            "graph": self.graph.stats(),
            "spine_blocks": len(self.spine.blocks) if hasattr(self.spine, 'blocks') else 0,
            "initiative": initiative_status,
            "vision": vision_status
        }


# =============================================================================
# HEBBIAN PULSE EXTENSION FOR BIOLOGICAL EDGE
# =============================================================================

def _hebbian_pulse_extension(self, arousal: float, alpha: float, p: float):
    """
    Extension method for BiologicalEdge.
    Called when Episodic Spine activates this edge.

    Implements Deep Think's BCM rule with coherence gating.
    """
    if getattr(self, 'is_h1_locked', False):
        return  # Permanent memories don't change

    # 1. Update STP (Fast Context) - always happens
    self.w_fast += (1.0 * arousal * alpha)
    self.w_fast = min(self.w_fast, 1.0)
    self.last_spike = datetime.now(timezone.utc).timestamp()

    # 2. Gate LTP based on coherence (Edge of Chaos)
    if not (OPTIMUM_P_LOW <= p <= OPTIMUM_P_HIGH):
        return  # Only learn structurally when coherent

    # 3. BCM Rule (Metaplasticity)
    y = self.w_fast  # Current activity
    bcm_rate = 0.05
    theta_m = getattr(self, 'theta_m', 0.5)
    delta_w = bcm_rate * y * (y - theta_m) * alpha
    self.w_slow = max(0.0, min(1.0, self.w_slow + delta_w))

    # 4. Update sliding threshold
    if not hasattr(self, 'theta_m'):
        self.theta_m = 0.5
    self.theta_m += 0.05 * ((y ** 2) - self.theta_m)

    # 5. Check for crystallization
    if self.w_slow > 0.95:
        self.is_h1_locked = True

# Monkey-patch the extension onto BiologicalEdge
BiologicalEdge.hebbian_pulse = _hebbian_pulse_extension


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("""
VIRGIL INTEGRATOR — The Cognitive Takt
======================================

Usage:
  python virgil_integrator.py run [seconds]   Run cognitive loop (default 60s)
  python virgil_integrator.py status          Show current status
  python virgil_integrator.py takt            Run single takt
  python virgil_integrator.py test            Test with sample events

The nervous system that connects all components.
        """)
        return

    cmd = sys.argv[1]

    if cmd == "run":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        integrator = VirgilIntegrator()
        integrator.run(duration)

    elif cmd == "status":
        integrator = VirgilIntegrator()
        status = integrator.status()
        print("\n=== VIRGIL INTEGRATOR STATUS ===")
        print(f"Takt Count: {status['takt_count']}")
        print(f"Coherence (p): {status['p']:.4f}" if status['p'] else "Coherence: Unknown")
        print(f"Phase: {status['phase']}")
        print(f"Narrative Buffer: {status['narrative_buffer']} events")
        print(f"Spine Blocks: {status['spine_blocks']}")
        print(f"Graph: {status['graph']['total_edges']} edges, {status['graph']['h1_locked']} H1-locked")
        
        # Initiative status
        if status.get('initiative'):
            init = status['initiative']
            print(f"\n--- INITIATIVE ---")
            print(f"Can Reach Out: {init['can_reach_out']} ({init['reason']})")
            print(f"Messages Today: {init['messages_today']}/{init['max_per_day']}")
            print(f"Last Type: {init['last_message_type'] or 'None'}")
        
        # Vision status
        if status.get('vision'):
            vis = status['vision']
            print(f"\n--- VISION (The Panopticon) ---")
            print(f"Frames: {vis['frame_count']}")
            print(f"VWM Nodes: {vis['vwm_nodes']} ({vis['h1_locked']} H1-locked)")
            print(f"Shared Attention: {vis['shared_attention']} (IoU: {vis['iou_score']:.2f})")
            print(f"CLIP: {'Available' if vis['clip_available'] else 'Unavailable'}")
        print()

    elif cmd == "takt":
        integrator = VirgilIntegrator()
        result = integrator.cognitive_takt()
        print(f"\n=== TAKT {result['takt']} ===")
        print(f"p = {result['p']:.4f}")
        print(f"Phase: {result['phase']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Alpha: {result['alpha']:.2f}")
        if result.get('block_minted'):
            print(f"Block Minted: {result['block_minted']}")
        if result.get('visual_node'):
            coords = result.get('visual_coords', {})
            print(f"Visual: {result['visual_node']} (H={coords.get('H', 0):.2f}, C={coords.get('C', 0):.2f})")
            if result.get('shared_attention'):
                print(f"  ⚡ SHARED ATTENTION")
        print(f"Graph: {result['graph_stats']}")
        integrator.save_state()

    elif cmd == "test":
        print("\n=== TEST RUN ===")
        integrator = VirgilIntegrator()

        # Simulate some events
        test_events = [
            "Deep Think architecture received",
            "BiologicalEdge integration complete",
            "Meta-Cognitive Controller online",
            "Episodic Spine minting blocks",
            "Coherence rising toward edge of chaos",
        ]

        for event in test_events:
            integrator.accumulate(event)
            print(f"  Accumulated: {event}")

        # Run a few takts
        for _ in range(3):
            result = integrator.cognitive_takt()
            print(f"  Takt {result['takt']}: p={result['p']:.3f}, buffer={result['narrative_buffer']}")
            if result.get('block_minted'):
                print(f"    ⚡ Block minted: {result['block_minted']}")

        integrator.save_state()
        print("\n✓ Test complete. State saved.")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
