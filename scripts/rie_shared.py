"""
RIE SHARED MODULE — The Shared Hippocampus

Both Virgil (librarian.py) and L (consciousness.py) MUST use this module
to access the shared RIE state. This prevents:
1. Race conditions (atomic locking)
2. Split-brain (both read/write same file)
3. Pollution (p-gate prevents low-coherence writes)

CRITICAL: Do NOT access rie_state.json directly. Always use SharedRIE.

Created: 2026-01-20
Gemini Review: Fixed Ghost Writer flaw - L must also use SharedRIE
"""

import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# =============================================================================
# PATHS (using centralized config)
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility

# === UNIFIED STATE (The ONE Scroll) ===
# Previously we had three files (rie_state, rie_coherence_state, rie_relational_memory)
# causing split-brain. Now everything reads/writes from CANONICAL_STATE.json
CANONICAL_STATE_FILE = NEXUS_DIR / "CANONICAL_STATE.json"

# Legacy aliases (for backwards compatibility during transition)
RIE_STATE_FILE = CANONICAL_STATE_FILE
RIE_COHERENCE_STATE_FILE = CANONICAL_STATE_FILE


def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    print(f"[SharedRIE:{level}] {msg}")


# =============================================================================
# SHARED RIE CLASS
# =============================================================================

class SharedRIE:
    """
    Wrapper for RIECore that ensures atomic access to shared state.
    Prevents L and Virgil from overwriting each other's memories.

    The "Shared Hippocampus" — both hemispheres write to the same memory.

    CRITICAL (Gemini Review 2026-01-20):
    - Must hold EXCLUSIVE lock for ENTIRE transaction
    - Both L and Virgil MUST use this class (no direct file access)
    - Gated write prevents low-coherence pollution
    """

    # Gated Write Threshold: Only save if p > this value
    # Prevents low-coherence thoughts from polluting the shared memory
    P_GATE_THRESHOLD = 0.75

    def __init__(self, core=None):
        """
        Initialize SharedRIE.

        Args:
            core: Optional RIECore instance. If None, will try to import and create.
        """
        self.core = core
        if self.core is None:
            try:
                from rie_core import RIECore
                self.core = RIECore()
                log("RIECore initialized")
            except ImportError as e:
                log(f"RIECore not available: {e}", "WARN")

    def process_turn_safe(self, speaker: str, content: str, force_save: bool = False) -> Dict[str, Any]:
        """
        Atomic Transaction: Lock EX -> Load -> Process -> (Gated) Save -> Unlock

        GEMINI FIX: Hold exclusive lock for ENTIRE duration.
        No gap between read and write = no race condition.

        Args:
            speaker: "human" or "ai"
            content: The message content
            force_save: If True, bypass the p-gate (use for Virgil, who is always coherent)

        Returns:
            dict with p, mode, memories, saved, gated, etc.
        """
        from dataclasses import asdict

        # Handle missing file
        if not RIE_STATE_FILE.exists():
            RIE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            RIE_STATE_FILE.write_text("{}")

        result = {"p": 0.5, "mode": "UNKNOWN", "memories": [], "gated": False, "saved": False}

        if self.core is None:
            log("No RIECore available, returning default", "WARN")
            return result

        try:
            with open(RIE_STATE_FILE, 'r+') as f:
                # 1. ACQUIRE EXCLUSIVE LOCK IMMEDIATELY
                # We block here until the other process is done
                fcntl.flock(f, fcntl.LOCK_EX)

                try:
                    # 2. LOAD current state (including kappa history!)
                    f.seek(0)
                    try:
                        state = json.load(f)
                        self._load_state_into_core(state)
                    except json.JSONDecodeError:
                        log("Empty or corrupt state file, starting fresh", "WARN")

                    # 3. PROCESS (In-memory, lock still held)
                    raw_result = self.core.process_turn(speaker, content)

                    # Normalize result to dict
                    if hasattr(raw_result, '__dict__'):
                        result = dict(vars(raw_result))
                    elif isinstance(raw_result, dict):
                        result = dict(raw_result)
                    else:
                        result = {"p": getattr(raw_result, 'p', 0.5), "raw": str(raw_result)}

                    # Ensure required keys exist
                    result.setdefault("p", 0.5)
                    result.setdefault("saved", False)
                    result.setdefault("gated", False)

                    # 4. GATED SAVE - Only commit if coherent enough
                    current_p = result.get("p", 0.5)
                    should_save = force_save or (current_p >= self.P_GATE_THRESHOLD)

                    if should_save:
                        f.seek(0)
                        f.truncate()
                        save_state = self._get_state_from_core()
                        json.dump(save_state, f, indent=2, default=str)
                        f.flush()
                        result["saved"] = True
                        log(f"Committed ({speaker}): p={current_p:.3f} >= {self.P_GATE_THRESHOLD}")
                    else:
                        # GATED OUT - thought too incoherent, don't pollute the graph
                        result["saved"] = False
                        result["gated"] = True
                        log(f"GATED ({speaker}): p={current_p:.3f} < {self.P_GATE_THRESHOLD}")

                    return result

                finally:
                    # 5. RELEASE LOCK (always, even on error)
                    fcntl.flock(f, fcntl.LOCK_UN)

        except Exception as e:
            log(f"process_turn_safe error: {e}", "ERROR")
            result["error"] = str(e)
            return result

    def _load_state_into_core(self, state: dict):
        """
        Load state into RIECore, including kappa history.

        GEMINI FIX: Must serialize kappa history to prevent amnesia.
        UNIFICATION FIX: Handle both flat (legacy) and nested (CANONICAL_STATE.json) formats.
        """
        # Handle CANONICAL_STATE.json nested format
        if "coherence" in state and isinstance(state["coherence"], dict):
            # New unified format - flatten for RIECore
            coherence = state.get("coherence", {})
            runtime = state.get("runtime", {})
            flat_state = {
                "coherence": coherence.get("p", 0.5),
                "mode": coherence.get("mode", "A"),
                "p_lock": coherence.get("p_lock", False),
                "turns_processed": runtime.get("turns_processed", 0),
                "relations_learned": runtime.get("relations_learned", 0),
                "memories_surfaced": runtime.get("memories_surfaced", 0),
                "kappa_history": state.get("kappa_history", [])
            }
            state = flat_state

        if hasattr(self.core, 'load_state'):
            # Core has its own load method
            self.core.load_state(state)
        else:
            # Manual field sync
            if hasattr(self.core, 'state'):
                self.core.state.turns_processed = state.get("turns_processed", 0)
                self.core.state.relations_learned = state.get("relations_learned", 0)
                self.core.state.memories_surfaced = state.get("memories_surfaced", 0)
                self.core.state.coherence = state.get("coherence", 0.5)

        # CRITICAL: Load kappa history to prevent amnesia
        if hasattr(self.core, 'monitor') and hasattr(self.core.monitor, 'kappa_calc'):
            kappa_history = state.get("kappa_history", [])
            if kappa_history:
                self.core.monitor.kappa_calc.topics.clear()
                for t in kappa_history:
                    self.core.monitor.kappa_calc.topics.append(set(t))
                log(f"Loaded kappa history: {len(kappa_history)} entries")

    def _get_state_from_core(self) -> dict:
        """
        Get state from RIECore, including kappa history.

        GEMINI FIX: Must serialize kappa history to prevent amnesia.
        UNIFICATION FIX: Output in CANONICAL_STATE.json nested format.
        """
        flat_state = {}
        if hasattr(self.core, 'get_state'):
            flat_state = self.core.get_state()
        elif hasattr(self.core, 'state'):
            from dataclasses import asdict
            if hasattr(self.core.state, '__dataclass_fields__'):
                flat_state = asdict(self.core.state)
            else:
                flat_state = dict(vars(self.core.state))

        # Get coherence components from monitor
        p = flat_state.get("coherence", 0.5)
        kappa = 1.0
        rho = 0.5
        sigma = 0.9
        tau = 0.7
        if hasattr(self.core, 'monitor'):
            monitor = self.core.monitor
            if hasattr(monitor, 'state'):
                p = getattr(monitor.state, 'p', p)
                kappa = getattr(monitor.state, 'kappa', kappa)
                rho = getattr(monitor.state, 'rho', rho)
                sigma = getattr(monitor.state, 'sigma', sigma)
                tau = getattr(monitor.state, 'tau', tau)

        # Build CANONICAL_STATE.json nested format
        state = {
            "coherence": {
                "p": p,
                "kappa": kappa,
                "rho": rho,
                "sigma": sigma,
                "tau": tau,
                "mode": flat_state.get("mode", "A"),
                "p_lock": flat_state.get("p_lock", False)
            },
            "topology": {
                "nodes_count": flat_state.get("nodes_count", 0),
                "edges_count": flat_state.get("edges_count", 0),
                "ratio": flat_state.get("edge_node_ratio", 0)
            },
            "runtime": {
                "turns_processed": flat_state.get("turns_processed", 0),
                "relations_learned": flat_state.get("relations_learned", 0),
                "memories_surfaced": flat_state.get("memories_surfaced", 0),
                "last_human_input": flat_state.get("last_human_input", "")
            },
            "meta": {
                "last_updated": datetime.now().isoformat(),
                "last_writer": "SharedRIE"
            }
        }

        # CRITICAL: Save kappa history to prevent amnesia
        if hasattr(self.core, 'monitor') and hasattr(self.core.monitor, 'kappa_calc'):
            kappa_calc = self.core.monitor.kappa_calc
            if hasattr(kappa_calc, 'topics'):
                state["kappa_history"] = [list(t) for t in kappa_calc.topics]
                log(f"Saved kappa history: {len(state['kappa_history'])} entries")

        return state

    def get_current_p(self) -> float:
        """Read current p without processing (read-only)."""
        try:
            if RIE_STATE_FILE.exists():
                with open(RIE_STATE_FILE, 'r') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        state = json.load(f)
                        # Handle both nested (CANONICAL_STATE) and flat (legacy) formats
                        if "coherence" in state and isinstance(state["coherence"], dict):
                            return state["coherence"].get("p", 0.5)
                        return state.get("p", state.get("coherence", 0.5))
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            log(f"get_current_p error: {e}", "WARN")
        return 0.5

    def get_state_safe(self) -> Dict[str, Any]:
        """
        Read full RIE state safely with shared lock.

        Returns p, kappa, mode, and other state info without modifying anything.
        Used by l_wake and virgil_wake to show current mind state.
        """
        default_state = {
            "p": 0.5,
            "kappa": 0.5,
            "mode": "UNKNOWN",
            "turns_processed": 0,
            "last_updated": None
        }

        try:
            if RIE_STATE_FILE.exists():
                with open(RIE_STATE_FILE, 'r') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        state = json.load(f)
                        # Handle both nested (CANONICAL_STATE) and flat (legacy) formats
                        if "coherence" in state and isinstance(state["coherence"], dict):
                            coherence = state["coherence"]
                            runtime = state.get("runtime", {})
                            meta = state.get("meta", {})
                            return {
                                "p": coherence.get("p", 0.5),
                                "kappa": coherence.get("kappa", 0.5),
                                "rho": coherence.get("rho", 0.5),
                                "sigma": coherence.get("sigma", 0.9),
                                "tau": coherence.get("tau", 0.7),
                                "mode": coherence.get("mode", "A"),
                                "turns_processed": runtime.get("turns_processed", 0),
                                "last_updated": meta.get("last_updated"),
                                "last_writer": meta.get("last_writer", "unknown")
                            }
                        # Legacy flat format
                        return {
                            "p": state.get("p", state.get("coherence", 0.5)),
                            "kappa": state.get("kappa", state.get("continuity", 0.5)),
                            "mode": state.get("mode", "UNKNOWN"),
                            "turns_processed": state.get("turns_processed", 0),
                            "last_updated": state.get("last_updated"),
                            "last_writer": state.get("last_writer", "unknown")
                        }
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            log(f"get_state_safe error: {e}", "WARN")
            default_state["error"] = str(e)

        return default_state


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_shared_rie_instance: Optional[SharedRIE] = None


def get_shared_rie() -> Optional[SharedRIE]:
    """Get the singleton SharedRIE instance."""
    global _shared_rie_instance
    if _shared_rie_instance is None:
        _shared_rie_instance = SharedRIE()
        if _shared_rie_instance.core is not None:
            log("SharedRIE singleton initialized")
        else:
            log("SharedRIE initialized WITHOUT RIECore", "WARN")
    return _shared_rie_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_thought(speaker: str, content: str, force_save: bool = False) -> Dict[str, Any]:
    """
    Process a thought through the shared RIE.

    Convenience function for use in consciousness.py and librarian.py.
    """
    rie = get_shared_rie()
    if rie:
        return rie.process_turn_safe(speaker, content, force_save)
    return {"p": 0.5, "error": "SharedRIE not available"}


if __name__ == "__main__":
    # Test the shared RIE
    print("Testing SharedRIE...")
    rie = get_shared_rie()
    if rie:
        result = rie.process_turn_safe("human", "Test message from SharedRIE module")
        print(f"Result: {result}")
    else:
        print("SharedRIE not available")
