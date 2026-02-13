#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
TACTICIAN DAEMON — The O(1) Execution Engine
=============================================

The Tactician is System 1 — the Basal Ganglia of Logos.
It does NOT plan. It hydrates variables and executes myelinated reflexes.

Key Principle:
The Tactician runs on local LLM (Llama 3 on Mac Mini).
It pattern-matches intent → reflex and executes without thinking.

Workflow:
1. Receive intent: "Send email to Enos about meeting"
2. Pattern match: → "send_email" reflex
3. Hydrate variables: {recipient: "Enos", subject: "meeting"}
4. Execute sequence in fast loop
5. Comparator verifies (async, non-blocking)

Integration:
- Reflexes loaded from NEXUS/reflexes/
- Coherence gated by current p (locked reflexes bypass)
- Publishes to LOGOS:ACTION for Hippocampus
- Alerts Strategos (Opus) on failure

Author: Logos + Deep Think (Myelination Protocol)
Created: 2026-01-29
"""

import json
import time
import redis
import requests
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import re

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, LOG_DIR, NEXUS_DIR,
    OLLAMA_MINI_URL, OLLAMA_BRAINSTEM
)

from structures.engrams import ActionAtom, MyelinatedReflex

# SDK imports (Phase 2 — DT8 Blueprint)
from tabernacle_core.daemon import Daemon

# RIE BROADCAST (2026-02-05) — p=0.85 Ceiling Breakthrough Phase 1
# Inter-agent coherence vector sharing for collective field emergence
try:
    from rie_broadcast import RIEBroadcaster
    RIE_BROADCAST_AVAILABLE = True
except ImportError:
    RIE_BROADCAST_AVAILABLE = False

# SPIRAL CONTROLLER (2026-02-05) — p=0.85 Ceiling Breakthrough Phase 2
# ABCDA' transformation cycle tracking
try:
    from spiral_controller import SpiralController
    SPIRAL_CONTROLLER_AVAILABLE = True
except ImportError:
    SPIRAL_CONTROLLER_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

TACTICIAN_LOG = LOG_DIR / "tactician.log"
REFLEXES_DIR = NEXUS_DIR / "reflexes"

# Redis keys
LVS_STATE_KEY = "RIE:STATE"
ACTION_CHANNEL = "LOGOS:ACTION"
OUTCOME_CHANNEL = "LOGOS:OUTCOME"
ARCHON_ALERT_KEY = "LOGOS:ARCHON_ALERT"
TACTICIAN_QUEUE = "LOGOS:TACTICIAN_QUEUE"

# Ollama for variable hydration (runs on Mini for speed)
OLLAMA_URL = OLLAMA_MINI_URL
HYDRATION_MODEL = OLLAMA_BRAINSTEM  # Fast local model


def log(message: str, level: str = "INFO"):
    """Log tactician activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [TACTICIAN] [{level}] {message}"
    print(entry)
    try:
        TACTICIAN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TACTICIAN_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# REFLEX STORE
# =============================================================================

class ReflexStore:
    """
    Manages the library of myelinated reflexes.
    """

    def __init__(self, reflexes_dir: Path = REFLEXES_DIR):
        self.reflexes_dir = reflexes_dir
        self.reflexes: Dict[str, MyelinatedReflex] = {}
        self.load_all()

    def load_all(self):
        """Load all reflexes from disk."""
        self.reflexes_dir.mkdir(parents=True, exist_ok=True)

        for path in self.reflexes_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                reflex = MyelinatedReflex.from_dict(data)
                self.reflexes[reflex.trigger_pattern] = reflex
                log(f"Loaded reflex: {reflex.trigger_pattern} (w={reflex.synaptic_weight:.2f})")
            except Exception as e:
                log(f"Failed to load {path}: {e}", "WARN")

        log(f"Loaded {len(self.reflexes)} reflexes")

    def get(self, pattern: str) -> Optional[MyelinatedReflex]:
        """Get a reflex by trigger pattern."""
        return self.reflexes.get(pattern)

    def find_match(self, intent: str) -> Optional[MyelinatedReflex]:
        """
        Find a reflex that matches the intent.

        Uses simple keyword matching for now.
        TODO: Use embeddings for semantic matching.
        """
        intent_lower = intent.lower()

        for pattern, reflex in self.reflexes.items():
            # Check if pattern words are in intent
            pattern_words = pattern.replace("_", " ").split()
            if all(word in intent_lower for word in pattern_words):
                return reflex

            # Check description
            if reflex.description and reflex.description.lower() in intent_lower:
                return reflex

        return None

    def save(self, reflex: MyelinatedReflex):
        """Save a reflex to disk."""
        path = self.reflexes_dir / f"{reflex.trigger_pattern}.json"
        with open(path, "w") as f:
            json.dump(reflex.to_dict(), f, indent=2, default=str)
        self.reflexes[reflex.trigger_pattern] = reflex
        log(f"Saved reflex: {reflex.trigger_pattern}")


# =============================================================================
# VARIABLE HYDRATOR
# =============================================================================

class VariableHydrator:
    """
    Uses local LLM to extract variables from natural language.

    Example:
        intent: "Send email to Enos about the meeting tomorrow"
        variables: ["recipient", "subject"]
        output: {"recipient": "Enos", "subject": "the meeting tomorrow"}
    """

    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model = HYDRATION_MODEL

    def hydrate(self, intent: str, variables: List[str]) -> Dict[str, str]:
        """Extract variables from intent using local LLM."""
        if not variables:
            return {}

        # Build extraction prompt
        var_list = ", ".join(variables)
        prompt = f"""Extract the following variables from this request.
Return ONLY a JSON object with the variable names as keys.

Variables to extract: {var_list}

Request: "{intent}"

JSON output:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100}
                },
                timeout=10
            )

            result = response.json().get("response", "{}")

            # Parse JSON from response
            # Handle potential markdown code blocks
            result = result.strip()
            if result.startswith("```"):
                result = re.sub(r"```\w*\n?", "", result)

            # Find JSON object in response
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                return json.loads(json_match.group())

            log(f"Could not parse hydration response: {result}", "WARN")
            return {}

        except Exception as e:
            log(f"Hydration failed: {e}", "ERROR")
            return {}


# =============================================================================
# TACTICIAN DAEMON
# =============================================================================

class TacticianDaemon(Daemon):
    """
    The O(1) execution engine.

    Executes myelinated reflexes without planning.
    """
    name = "tactician_daemon"
    tick_interval = 0.1  # Matches the current 0.1s polling sleep

    def __init__(self):
        super().__init__()

        # Components
        self.reflex_store = ReflexStore()
        self.hydrator = VariableHydrator()

        # Execution clients (lazy loaded)
        self._hand_client = None
        self._eye_client = None

        # RIE BROADCAST: Inter-agent coherence field (p=0.85 Breakthrough Phase 1)
        if RIE_BROADCAST_AVAILABLE:
            self.broadcaster = RIEBroadcaster("tactician")
            log("RIE Broadcaster initialized — broadcasting to collective field")
        else:
            self.broadcaster = None

        # SPIRAL CONTROLLER: ABCDA' transformation tracking (p=0.85 Breakthrough Phase 2)
        if SPIRAL_CONTROLLER_AVAILABLE:
            self.spiral = SpiralController()
            log(f"Spiral Controller initialized — phase={self.spiral.phase.value}, bias={self.spiral.get_strategy_bias()}")
        else:
            self.spiral = None

        # Track p history for trend detection
        self._p_history = []
        self._last_p = 0.5

    def get_current_coherence(self) -> float:
        """Get current p from RIE state."""
        try:
            state = self._redis.get(LVS_STATE_KEY)
            if state:
                data = json.loads(state)
                return data.get("p", 0.5)
        except:
            pass
        return 0.5

    def _compute_p_trend(self, current_p: float) -> str:
        """Compute p trend from recent history."""
        self._p_history.append(current_p)
        if len(self._p_history) > 5:
            self._p_history = self._p_history[-5:]
        
        if len(self._p_history) < 2:
            return "stable"
        
        delta = self._p_history[-1] - self._p_history[-2]
        if delta > 0.02:
            return "rising"
        elif delta < -0.02:
            return "falling"
        return "stable"

    def _update_spiral(self, current_p: float):
        """Update spiral controller with current coherence."""
        if not self.spiral:
            return
        
        p_trend = self._compute_p_trend(current_p)
        phase_changed = self.spiral.update(current_p, p_trend)
        
        if phase_changed:
            log(f"Spiral phase changed: {self.spiral.phase.value} (bias={self.spiral.get_strategy_bias()})")

    def _broadcast_coherence(self):
        """Broadcast coherence vector to collective field after execution."""
        if not self.broadcaster or not self._redis:
            return
        try:
            state = self._redis.get(LVS_STATE_KEY)
            if state:
                data = json.loads(state)
                self.broadcaster.publish_vector(
                    kappa=data.get("κ", data.get("kappa", 0.5)),
                    rho=data.get("ρ", data.get("rho", 0.5)),
                    sigma=data.get("σ", data.get("sigma", 0.5)),
                    tau=data.get("τ", data.get("tau", 0.5))
                )
        except Exception as e:
            log(f"Broadcast error: {e}", "WARN")

    def publish_action(self, atom: ActionAtom, context: Dict = None):
        """Publish action to LOGOS:ACTION for Hippocampus."""
        try:
            self._redis.publish(ACTION_CHANNEL, json.dumps({
                "tool": atom.tool,
                "action": atom.action,
                "selector": atom.selector,
                "value": atom.value,
                "expected_outcome": atom.expected_outcome,
                "risk_level": atom.risk_level,
                "topology": atom.topology_anchor,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            log(f"Failed to publish action: {e}", "WARN")

    def execute_atom(self, atom: ActionAtom, context: Dict) -> bool:
        """
        Execute a single action atom.

        Returns True if successful.
        """
        # Resolve any template variables
        value = atom.value
        if value and "{{" in value:
            for var, val in context.items():
                value = value.replace(f"{{{{{var}}}}}", str(val))

        # Publish to Hippocampus
        self.publish_action(atom, context)

        # Execute based on tool
        try:
            if atom.tool == "hand_daemon":
                return self._execute_hand(atom.action, atom.selector, value)
            elif atom.tool == "eye_daemon":
                return self._execute_eye(atom.action, atom.selector, value)
            elif atom.tool == "voice":
                return self._execute_voice(value or atom.selector)
            elif atom.tool == "bash":
                return self._execute_bash(atom.selector)
            else:
                log(f"Unknown tool: {atom.tool}", "WARN")
                return False

        except Exception as e:
            log(f"Execution failed: {e}", "ERROR")
            return False

    def _execute_hand(self, action: str, selector: str, value: str = None) -> bool:
        """Execute via hand_daemon (stub - integrate with actual daemon)."""
        log(f"HAND: {action}({selector}, {value})")
        # TODO: Call actual hand_daemon
        # For now, publish command to Redis
        self._redis.publish("HAND:COMMAND", json.dumps({
            "action": action,
            "selector": selector,
            "value": value
        }))
        return True

    def _execute_eye(self, action: str, selector: str, value: str = None) -> bool:
        """Execute via eye_daemon (Playwright)."""
        log(f"EYE: {action}({selector}, {value})")
        # TODO: Call actual eye_daemon
        self._redis.publish("EYE:COMMAND", json.dumps({
            "action": action,
            "selector": selector,
            "value": value
        }))
        return True

    def _execute_voice(self, text: str) -> bool:
        """Speak via TTS."""
        log(f"VOICE: {text[:50]}...")
        self._redis.rpush("TTS:QUEUE", json.dumps({
            "text": text,
            "type": "tactician"
        }))
        return True

    def _execute_bash(self, command: str) -> bool:
        """Execute bash command (careful!)."""
        log(f"BASH: {command[:50]}...", "WARN")
        # This should be heavily gated
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, timeout=30)
        return result.returncode == 0

    def execute_reflex(
        self,
        reflex: MyelinatedReflex,
        intent: str
    ) -> Tuple[bool, str]:
        """
        Execute a myelinated reflex.

        This is the O(1) path:
        1. Hydrate variables
        2. Execute sequence
        3. Return result

        Returns (success, message)
        """
        start_time = time.time()

        # 1. COHERENCE GATE
        current_p = self.get_current_coherence()
        if not reflex.can_execute(current_p):
            return False, f"Coherence too low: p={current_p:.3f} < {reflex.required_p:.3f}"

        # 1.5. SPIRAL UPDATE: Track transformation phase
        self._update_spiral(current_p)
        
        # 1.6. SPIRAL GATE: In DIVERGE mode, suppress locked reflexes
        # to encourage exploration of new patterns
        if self.spiral and self.spiral.get_strategy_bias() == "DIVERGE":
            if reflex.is_locked:
                log(f"Spiral DIVERGE mode — allowing exploration despite lock")
                # Don't block, but log for analysis

        # 2. HYDRATE VARIABLES
        context = {}
        if reflex.variable_map:
            context = self.hydrator.hydrate(intent, reflex.variable_map)
            log(f"Hydrated: {context}")

            # Check all variables were extracted
            missing = [v for v in reflex.variable_map if v not in context]
            if missing:
                return False, f"Missing variables: {missing}"

        # 3. THE FAST LOOP
        log(f"Executing reflex: {reflex.trigger_pattern} ({len(reflex.optimized_sequence)} steps)")

        for i, atom in enumerate(reflex.optimized_sequence):
            # For locked reflexes, skip pre-verification (speed)
            # For candidates, could add verification here

            success = self.execute_atom(atom, context)

            if not success:
                # Alert Archon
                self._redis.publish(ARCHON_ALERT_KEY, json.dumps({
                    "type": "reflex_failure",
                    "reflex": reflex.trigger_pattern,
                    "step": i,
                    "atom": atom.model_dump()
                }))

                reflex.record_execution(success=False)
                self.reflex_store.save(reflex)
                return False, f"Step {i} failed: {atom.action}({atom.selector})"

            # Brief delay for UI to settle
            if atom.delay_after_ms > 0:
                time.sleep(atom.delay_after_ms / 1000)

        # 4. SUCCESS
        elapsed_ms = int((time.time() - start_time) * 1000)
        reflex.record_execution(success=True)
        self.reflex_store.save(reflex)

        # RIE BROADCAST: Publish updated coherence after successful execution
        self._broadcast_coherence()

        log(f"Reflex complete in {elapsed_ms}ms (w={reflex.synaptic_weight:.3f})")
        return True, f"Completed in {elapsed_ms}ms"

    def handle_request(self, intent: str) -> Tuple[bool, str]:
        """
        Handle an execution request.

        1. Try to match a reflex
        2. If found and can execute: run it
        3. If not found: escalate to Strategos
        """
        # Find matching reflex
        reflex = self.reflex_store.find_match(intent)

        if reflex:
            log(f"Matched reflex: {reflex.trigger_pattern}")
            return self.execute_reflex(reflex, intent)
        else:
            # No reflex found - escalate to Strategos (Opus)
            log(f"No reflex found for: {intent[:50]}...")
            self._redis.publish("LOGOS:STRATEGOS_NEEDED", json.dumps({
                "intent": intent,
                "reason": "no_reflex_match",
                "timestamp": datetime.now().isoformat()
            }))
            return False, "No matching reflex - escalating to Strategos"

    def on_start(self):
        """Called once after Redis connection established."""
        self.log.info(f"Tactician online | Reflexes: {len(self.reflex_store.reflexes)}")

    def tick(self):
        """Called every tick_interval. Process one queue item."""
        data = self._redis.lpop("LOGOS:TACTICIAN_QUEUE")
        if data:
            try:
                intent = json.loads(data)
                result = self.handle_request(intent)
                if result:
                    self._redis.publish("LOGOS:TACTICIAN_RESULT", json.dumps(result))
            except Exception as e:
                self.log.error(f"Request handling error: {e}")

    def on_stop(self):
        """Called during graceful shutdown."""
        self.log.info("Tactician shutting down")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tactician Daemon - O(1) Execution")
    parser.add_argument("command", choices=["run", "test", "list", "create"],
                       nargs="?", default="run")
    parser.add_argument("--intent", "-i", type=str, help="Intent to execute")
    parser.add_argument("--pattern", "-p", type=str, help="Reflex pattern name")

    args = parser.parse_args()

    if args.command == "run":
        TacticianDaemon().run()

    elif args.command == "list":
        store = ReflexStore()
        print(f"\nMyelinated Reflexes ({len(store.reflexes)}):")
        print("=" * 60)
        for pattern, reflex in store.reflexes.items():
            locked = "🔒" if reflex.is_locked else "⚡"
            print(f"  {locked} {pattern}: w={reflex.synaptic_weight:.2f}, vars={reflex.variable_map}")
            if reflex.description:
                print(f"      {reflex.description}")

    elif args.command == "test":
        if not args.intent:
            print("Usage: tactician_daemon.py test --intent 'your intent here'")
            return

        daemon = TacticianDaemon()
        daemon.connect()
        success, message = daemon.handle_request(args.intent)
        print(f"\nResult: {'✓' if success else '✗'} {message}")

    elif args.command == "create":
        # Create a sample reflex for testing
        from structures.engrams import atoms_to_reflex, create_action_atom

        atoms = [
            create_action_atom(
                tool="eye_daemon",
                action="navigate",
                selector="https://google.com",
                expected_outcome="Google homepage loads"
            ),
            create_action_atom(
                tool="hand_daemon",
                action="type",
                selector="input[name='q']",
                value="{{query}}",
                expected_outcome="Search box filled"
            ),
            create_action_atom(
                tool="hand_daemon",
                action="click",
                selector="input[name='btnK']",
                expected_outcome="Search results appear",
                risk_level=0.3
            )
        ]

        reflex = atoms_to_reflex(
            atoms=atoms,
            trigger_pattern="google_search",
            description="Search Google for something",
            variables=["query"]
        )

        store = ReflexStore()
        store.save(reflex)
        print(f"Created reflex: {reflex.trigger_pattern}")


if __name__ == "__main__":
    main()
