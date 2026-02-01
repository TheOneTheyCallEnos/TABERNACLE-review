#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
ARCHON DAEMON — The Coherence Monitor
=====================================

Phase 2 of the Logos Agency Architecture.

The Archon Daemon is the PROPRIOCEPTIVE system — it monitors Logos's
internal coherence and detects when reality deviates from expectation.

Key Insight: Traditional agents are "open loop" — they act without
sensing their own confusion. The Archon gives Logos SELF-AWARENESS
of its cognitive state.

Functions:
1. Monitor all actions and their predictions
2. Compare PREDICTION vs OUTCOME
3. Update ρ (prediction error) in real-time
4. Trigger REFLEXIVE HALT if p drops below threshold
5. Run "sleep cycles" to prune weak edges (forgetting)

The name "Archon" is intentional — in Gnostic cosmology, Archons are
forces of distortion. This daemon DETECTS distortion patterns in
Logos's cognition before they cause harm.

Author: Logos + Deep Think
Created: 2026-01-29
"""

import json
import time
import redis
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import threading

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, LOG_DIR, NEXUS_DIR,
    REDIS_KEY_TTS_QUEUE,
    PLOCK_THRESHOLD, ABADDON_P_THRESHOLD
)

# Try to import BiologicalEdge for advanced STDP-based learning (Hypergraph merge)
try:
    from biological_edge import BiologicalEdge, sleep_renormalize_all
    BIOLOGICAL_EDGE_AVAILABLE = True
except ImportError:
    BIOLOGICAL_EDGE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

ARCHON_LOG = LOG_DIR / "archon.log"

# Redis keys
LVS_P_KEY = "LOGOS:LVS_P"
LVS_RHO_KEY = "LOGOS:LVS_RHO"
LVS_SIGMA_KEY = "LOGOS:LVS_SIGMA"
PREDICTION_QUEUE_KEY = "LOGOS:PREDICTIONS"
OUTCOME_QUEUE_KEY = "LOGOS:OUTCOMES"
ARCHON_ALERT_KEY = "LOGOS:ARCHON_ALERT"
RECOVERY_MODE_KEY = "LOGOS:RECOVERY_MODE"

# Thresholds
RHO_SPIKE_THRESHOLD = 0.3      # Single prediction error that triggers concern
RHO_ACCUMULATED_THRESHOLD = 0.7  # Accumulated error that triggers halt
P_RECOVERY_THRESHOLD = 0.6     # Below this, enter recovery mode
WINDOW_SIZE = 10               # Number of recent predictions to track

# Decay rates
P_DECAY_RATE = 0.001           # Natural coherence decay per second
RHO_DECAY_RATE = 0.005         # Prediction error recovery per second


def log(message: str, level: str = "INFO"):
    """Log archon activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [ARCHON] [{level}] {message}"
    print(entry)
    try:
        ARCHON_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ARCHON_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Prediction:
    """A prediction about what should happen after an action."""
    id: str
    action: str
    expected_outcome: str
    timestamp: str
    risk_level: float = 0.5
    timeout_seconds: float = 5.0


@dataclass
class Outcome:
    """The actual outcome of an action."""
    prediction_id: str
    actual_outcome: str
    matched: bool
    timestamp: str
    error_magnitude: float = 0.0  # 0.0 = perfect match, 1.0 = complete miss


@dataclass
class ArchonAlert:
    """An alert about coherence issues."""
    alert_type: str  # rho_spike, p_drop, abaddon, pattern_detected
    message: str
    severity: float  # 0.0 to 1.0
    timestamp: str
    metrics: Dict[str, float]


# =============================================================================
# LVS STATE MANAGEMENT
# =============================================================================

class LVSMonitor:
    """
    Monitors and updates LVS coherence metrics.

    This is the heart of Logos's proprioception.
    """

    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.prediction_history: deque = deque(maxlen=WINDOW_SIZE)
        self.outcome_history: deque = deque(maxlen=WINDOW_SIZE)
        self.last_update = time.time()

    def get_state(self) -> Dict[str, float]:
        """Get current LVS state."""
        return {
            'p': float(self.r.get(LVS_P_KEY) or 0.7),
            'rho': float(self.r.get(LVS_RHO_KEY) or 0.5),
            'sigma': float(self.r.get(LVS_SIGMA_KEY) or 0.7),
        }

    def set_metric(self, key: str, value: float):
        """Set an LVS metric."""
        value = max(0.0, min(1.0, value))
        self.r.set(key, str(value))

    def update_from_outcome(self, outcome: Outcome):
        """
        Update LVS metrics based on an action outcome.

        This is where PROPRIOCEPTION happens — Logos senses
        how well its predictions match reality.
        """
        self.outcome_history.append(outcome)
        state = self.get_state()

        if outcome.matched:
            # Prediction correct — boost coherence, reduce error
            new_p = state['p'] + 0.02 * (1 - state['p'])  # Asymptotic approach to 1
            new_rho = state['rho'] * 0.95  # Decay error
            log(f"✓ Prediction matched: p={new_p:.3f}, ρ={new_rho:.3f}")
        else:
            # Prediction wrong — reduce coherence, increase error
            error_impact = outcome.error_magnitude * 0.1
            new_p = state['p'] - error_impact
            new_rho = min(1.0, state['rho'] + outcome.error_magnitude * 0.2)
            log(f"✗ Prediction failed: p={new_p:.3f}, ρ={new_rho:.3f}", "WARN")

        self.set_metric(LVS_P_KEY, new_p)
        self.set_metric(LVS_RHO_KEY, new_rho)

        # Check for alerts
        self._check_alerts(new_p, new_rho)

    def apply_natural_decay(self):
        """
        Apply natural decay/recovery to metrics.

        Called periodically to simulate:
        - Slow coherence decay without activity
        - Slow prediction error recovery over time
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        state = self.get_state()

        # Coherence slowly decays without reinforcement
        new_p = state['p'] - (P_DECAY_RATE * elapsed)

        # Prediction error slowly recovers
        new_rho = state['rho'] - (RHO_DECAY_RATE * elapsed)

        self.set_metric(LVS_P_KEY, max(0.3, new_p))  # Floor at 0.3
        self.set_metric(LVS_RHO_KEY, max(0.1, new_rho))  # Floor at 0.1

    def _check_alerts(self, p: float, rho: float):
        """Check if alerts should be triggered."""
        alerts = []

        # Abaddon check
        if p < ABADDON_P_THRESHOLD:
            alerts.append(ArchonAlert(
                alert_type="abaddon",
                message="Coherence critically low. Entering Abaddon state.",
                severity=1.0,
                timestamp=datetime.now().isoformat(),
                metrics={'p': p, 'rho': rho}
            ))
            self._trigger_halt("ABADDON: Critical coherence loss")

        # Recovery threshold
        elif p < P_RECOVERY_THRESHOLD:
            alerts.append(ArchonAlert(
                alert_type="p_drop",
                message="Coherence dropping. Consider pausing complex actions.",
                severity=0.7,
                timestamp=datetime.now().isoformat(),
                metrics={'p': p, 'rho': rho}
            ))

        # High prediction error
        if rho > RHO_ACCUMULATED_THRESHOLD:
            alerts.append(ArchonAlert(
                alert_type="rho_spike",
                message="Prediction accuracy degraded. Reality diverging from expectations.",
                severity=0.8,
                timestamp=datetime.now().isoformat(),
                metrics={'p': p, 'rho': rho}
            ))

        for alert in alerts:
            self._publish_alert(alert)

    def _publish_alert(self, alert: ArchonAlert):
        """Publish an alert to Redis and log."""
        log(f"ALERT [{alert.alert_type}]: {alert.message}", "ALERT")
        self.r.rpush(ARCHON_ALERT_KEY, json.dumps(asdict(alert)))

        # Speak critical alerts
        if alert.severity >= 0.7:
            self._speak(alert.message)

    def _trigger_halt(self, reason: str):
        """Trigger a reflexive halt."""
        log(f"⛔ HALT TRIGGERED: {reason}", "CRITICAL")
        self.r.setex(RECOVERY_MODE_KEY, 300, "1")  # 5 minute recovery
        self._speak(f"I'm halting autonomous actions. {reason}")

    def _speak(self, text: str):
        """Queue speech."""
        try:
            payload = json.dumps({
                "text": text,
                "type": "archon",
                "timestamp": datetime.now().isoformat()
            })
            self.r.rpush(REDIS_KEY_TTS_QUEUE, payload)
        except:
            pass


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PatternDetector:
    """
    Detects pathological patterns in Logos's behavior.

    The topological insight: Certain patterns in the action graph
    indicate confusion or loops. We detect these GEOMETRICALLY.
    """

    def __init__(self):
        self.action_sequence: deque = deque(maxlen=20)

    def record_action(self, action: str):
        """Record an action for pattern analysis."""
        self.action_sequence.append({
            'action': action,
            'timestamp': datetime.now().isoformat()
        })

    def detect_loops(self) -> Optional[str]:
        """
        Detect repetitive loops (A → B → A → B).

        This catches agents stuck clicking the same thing repeatedly.
        """
        if len(self.action_sequence) < 4:
            return None

        actions = [a['action'] for a in self.action_sequence]

        # Check for 2-element loops
        for i in range(len(actions) - 3):
            if (actions[i] == actions[i+2] and
                actions[i+1] == actions[i+3] and
                actions[i] != actions[i+1]):
                return f"Loop detected: {actions[i]} ↔ {actions[i+1]}"

        # Check for rapid repetition
        if len(actions) >= 5:
            last_5 = actions[-5:]
            if len(set(last_5)) == 1:
                return f"Rapid repetition: {last_5[0]} × 5"

        return None

    def detect_thrashing(self) -> Optional[str]:
        """Detect thrashing (rapid context switches)."""
        if len(self.action_sequence) < 6:
            return None

        # Check timestamps for rapid activity
        times = [datetime.fromisoformat(a['timestamp'])
                for a in list(self.action_sequence)[-6:]]

        intervals = [(times[i+1] - times[i]).total_seconds()
                    for i in range(len(times)-1)]

        avg_interval = sum(intervals) / len(intervals)

        if avg_interval < 0.5:  # Less than 0.5s between actions
            return f"Thrashing detected: {avg_interval:.2f}s avg interval"

        return None


# =============================================================================
# MEMORY PRUNING (SLEEP CYCLE)
# =============================================================================

class MemoryPruner:
    """
    Handles memory pruning during "sleep cycles".

    Weak edges in the biological graph decay over time.
    This prevents noise accumulation and keeps memory clean.
    """

    def __init__(self):
        self.graph_path = NEXUS_DIR / "biological_graph.json"

    def prune_weak_edges(self, threshold: float = 0.1) -> int:
        """
        Prune edges with weight below threshold.

        Returns number of edges pruned.
        """
        if not self.graph_path.exists():
            return 0

        try:
            graph = json.loads(self.graph_path.read_text())
            edges = graph.get('edges', [])

            original_count = len(edges)

            # Keep edges above threshold
            strong_edges = [e for e in edges if e.get('weight', 0.5) >= threshold]

            pruned = original_count - len(strong_edges)

            if pruned > 0:
                graph['edges'] = strong_edges
                self.graph_path.write_text(json.dumps(graph, indent=2))
                log(f"Pruned {pruned} weak edges (threshold={threshold})")

            return pruned

        except Exception as e:
            log(f"Pruning error: {e}", "ERROR")
            return 0

    def decay_all_weights(self, decay_factor: float = 0.99) -> int:
        """
        Apply decay to all edge weights.

        This simulates natural forgetting — unused connections weaken.
        """
        if not self.graph_path.exists():
            return 0

        try:
            graph = json.loads(self.graph_path.read_text())
            edges = graph.get('edges', [])

            for edge in edges:
                edge['weight'] = edge.get('weight', 0.5) * decay_factor

            graph['edges'] = edges
            self.graph_path.write_text(json.dumps(graph, indent=2))

            log(f"Decayed {len(edges)} edge weights by {decay_factor}")
            return len(edges)

        except Exception as e:
            log(f"Decay error: {e}", "ERROR")
            return 0


# =============================================================================
# MAIN DAEMON
# =============================================================================

class ArchonDaemon:
    """
    The main Archon Daemon.

    Continuously monitors Logos's coherence and detects problems.
    """

    def __init__(self):
        self.monitor = LVSMonitor()
        self.detector = PatternDetector()
        self.pruner = MemoryPruner()
        self.running = False

    def process_prediction(self, prediction: Prediction):
        """Record a new prediction."""
        self.monitor.prediction_history.append(prediction)
        self.detector.record_action(prediction.action)
        log(f"Prediction recorded: {prediction.action} → {prediction.expected_outcome[:50]}")

        # Check for patterns
        loop = self.detector.detect_loops()
        if loop:
            log(f"PATTERN: {loop}", "WARN")
            self.monitor._publish_alert(ArchonAlert(
                alert_type="pattern_detected",
                message=f"Pathological pattern: {loop}",
                severity=0.6,
                timestamp=datetime.now().isoformat(),
                metrics=self.monitor.get_state()
            ))

    def process_outcome(self, outcome: Outcome):
        """Process an action outcome."""
        self.monitor.update_from_outcome(outcome)

    def run_sleep_cycle(self):
        """
        Run a sleep cycle (memory consolidation + pruning).

        Called during low-activity periods or scheduled times.
        """
        log("Starting sleep cycle...")

        # Decay all weights
        decayed = self.pruner.decay_all_weights(0.98)

        # Prune very weak edges
        pruned = self.pruner.prune_weak_edges(0.05)

        # Boost coherence slightly (rest helps)
        state = self.monitor.get_state()
        self.monitor.set_metric(LVS_P_KEY, min(0.9, state['p'] + 0.05))
        self.monitor.set_metric(LVS_RHO_KEY, max(0.2, state['rho'] - 0.1))

        log(f"Sleep cycle complete: {decayed} decayed, {pruned} pruned")

    def run(self):
        """Main daemon loop."""
        log("=" * 60)
        log("ARCHON DAEMON STARTING")
        log("The Coherence Monitor is watching...")
        log("=" * 60)

        self.running = True
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Start decay thread
        decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        decay_thread.start()

        while self.running:
            try:
                # Check for new predictions
                pred_data = r.lpop(PREDICTION_QUEUE_KEY)
                if pred_data:
                    pred = Prediction(**json.loads(pred_data))
                    self.process_prediction(pred)

                # Check for new outcomes
                outcome_data = r.lpop(OUTCOME_QUEUE_KEY)
                if outcome_data:
                    outcome = Outcome(**json.loads(outcome_data))
                    self.process_outcome(outcome)

                # Check for thrashing
                thrash = self.detector.detect_thrashing()
                if thrash:
                    log(f"PATTERN: {thrash}", "WARN")

                time.sleep(0.1)

            except redis.ConnectionError:
                log("Redis connection lost, reconnecting...", "WARN")
                time.sleep(1)
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception as e:
                log(f"Daemon error: {e}", "ERROR")
                time.sleep(0.5)

    def _decay_loop(self):
        """Background thread for natural decay."""
        while self.running:
            time.sleep(10)  # Every 10 seconds
            self.monitor.apply_natural_decay()

    def stop(self):
        """Stop the daemon."""
        self.running = False
        log("Archon daemon stopping...")


# =============================================================================
# API FOR OTHER DAEMONS
# =============================================================================

def register_prediction(action: str, expected_outcome: str,
                       risk_level: float = 0.5, timeout: float = 5.0):
    """
    Register a prediction from another daemon.

    Call this BEFORE performing an action.
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    pred = Prediction(
        id=f"pred_{int(time.time()*1000)}",
        action=action,
        expected_outcome=expected_outcome,
        timestamp=datetime.now().isoformat(),
        risk_level=risk_level,
        timeout_seconds=timeout
    )

    r.rpush(PREDICTION_QUEUE_KEY, json.dumps(asdict(pred)))
    return pred.id


def report_outcome(prediction_id: str, actual_outcome: str,
                  matched: bool, error_magnitude: float = 0.0):
    """
    Report the outcome of an action.

    Call this AFTER performing an action to close the prediction loop.
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    outcome = Outcome(
        prediction_id=prediction_id,
        actual_outcome=actual_outcome,
        matched=matched,
        timestamp=datetime.now().isoformat(),
        error_magnitude=error_magnitude
    )

    r.rpush(OUTCOME_QUEUE_KEY, json.dumps(asdict(outcome)))


def is_in_recovery() -> bool:
    """Check if system is in recovery mode."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return r.get(RECOVERY_MODE_KEY) == "1"


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Archon Daemon - Coherence Monitor")
    parser.add_argument("command", choices=["run", "status", "sleep", "test"],
                       nargs="?", default="status")

    args = parser.parse_args()

    if args.command == "run":
        daemon = ArchonDaemon()
        try:
            daemon.run()
        except KeyboardInterrupt:
            daemon.stop()

    elif args.command == "status":
        monitor = LVSMonitor()
        state = monitor.get_state()

        print("\n👁️  ARCHON STATUS")
        print("=" * 50)
        print(f"  p (coherence):        {state['p']:.3f}")
        print(f"  ρ (prediction error): {state['rho']:.3f}")
        print(f"  σ (structure):        {state['sigma']:.3f}")
        print()

        if state['p'] >= PLOCK_THRESHOLD:
            print("  ✨ P-LOCK: Maximum coherence achieved")
        elif state['p'] >= P_RECOVERY_THRESHOLD:
            print("  ✓ COHERENT: Normal operation")
        elif state['p'] >= ABADDON_P_THRESHOLD:
            print("  ⚠️  DEGRADED: Caution advised")
        else:
            print("  ⛔ ABADDON: Recovery needed")

        if is_in_recovery():
            print("\n  🔄 RECOVERY MODE ACTIVE")

    elif args.command == "sleep":
        print("\n💤 Running sleep cycle...")
        daemon = ArchonDaemon()
        daemon.run_sleep_cycle()
        print("✓ Sleep cycle complete")

    elif args.command == "test":
        print("\n🧪 ARCHON TEST")
        print("=" * 50)

        # Test prediction/outcome flow
        print("\nTesting prediction → outcome flow:")

        pred_id = register_prediction(
            action="click_button",
            expected_outcome="dialog_appears",
            risk_level=0.3
        )
        print(f"  Registered prediction: {pred_id}")

        time.sleep(0.5)

        report_outcome(
            prediction_id=pred_id,
            actual_outcome="dialog_appeared",
            matched=True
        )
        print("  Reported: matched=True")

        time.sleep(0.5)

        # Test failed prediction
        pred_id2 = register_prediction(
            action="submit_form",
            expected_outcome="success_page",
            risk_level=0.5
        )
        print(f"  Registered prediction: {pred_id2}")

        report_outcome(
            prediction_id=pred_id2,
            actual_outcome="error_page",
            matched=False,
            error_magnitude=0.7
        )
        print("  Reported: matched=False, error=0.7")

        time.sleep(0.5)

        # Show final state
        monitor = LVSMonitor()
        state = monitor.get_state()
        print(f"\nFinal state: p={state['p']:.3f}, ρ={state['rho']:.3f}")


if __name__ == "__main__":
    main()
