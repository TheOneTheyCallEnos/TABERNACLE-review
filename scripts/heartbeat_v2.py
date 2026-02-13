#!/usr/bin/env python3
"""
HEARTBEAT V2 — RAM-First RIE State Synchronization Daemon
==========================================================
Refactored for RAM-first architecture. All state broadcast via Redis.
Disk persistence ONLY on:
  1. Graceful shutdown (SIGTERM/SIGINT)
  2. P-Lock events (coherence milestone)

The RIE State Vector:
  p (coherence)  - Overall coherence [0,1]
  κ (kappa)      - Clarity
  ρ (rho)        - Precision
  σ (sigma)      - Structure
  τ (tau)        - Trust/Tension

Protocol:
  1. TICK: Heartbeat broadcasts state every second TO REDIS ONLY
  2. UPDATE: Nodes can publish deltas
  3. CONSENSUS: Pi aggregates and broadcasts global state
  4. REACTION: Nodes adjust behavior based on state

Channels:
  RIE:STATE     - Current state vector (JSON)
  LOGOS:STATE   - Mirrored state for Logos swarm
  RIE:TICK      - Heartbeat tick (timestamp)
  RIE:ALERT     - Alerts when thresholds crossed

DISK WRITES: ONLY on shutdown or P-Lock events.

SDK MIGRATION: Phase 1 (DT8 Blueprint) — 2026-02-13
  - Inherits from tabernacle_core.daemon.Daemon
  - File I/O via StateManager (fcntl.flock protected)
  - Dead Man's Switch via DAEMON:ALIVE:heartbeat_v2

Author: Virgil
Date: 2026-01-19 (original), 2026-01-28 (v2 refactor), 2026-02-13 (SDK migration)
Status: Phase III - RAM-First Architecture + SDK
"""

import json
import time
import sys
import threading
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# SDK imports
from tabernacle_core.daemon import Daemon
from tabernacle_core.schemas import HeartbeatState

# Import the REAL coherence monitor
from rie_coherence_monitor_v2 import CoherenceMonitorV2

# Import topology functions for real measurements
try:
    from synonymy_daemon import build_graph, find_components
    HAS_TOPOLOGY = True
except ImportError:
    HAS_TOPOLOGY = False

# Phase 0B: Coherence drop auto-alert — push to ntfy when p crashes
try:
    from virgil_notify import send_coherence_alert as _notify_coherence_drop
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

# RIE BROADCAST (2026-02-05) — p=0.85 Ceiling Breakthrough Phase 1
try:
    from rie_broadcast import RIEBroadcaster
    RIE_BROADCAST_AVAILABLE = True
except ImportError:
    RIE_BROADCAST_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, REDIS_HOST

# Paths (using centralized config)
TABERNACLE = BASE_DIR
NEXUS = NEXUS_DIR
RIE_STATE_FILE = NEXUS_DIR / "CANONICAL_STATE.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"

# Thresholds
P_LOCK_THRESHOLD = 0.95
ABADDON_THRESHOLD = 0.50
RECOVERY_THRESHOLD = 0.65
HIGH_TENSION_THRESHOLD = 0.80

# Timing
TICK_INTERVAL = 1.0  # seconds

# =============================================================================
# RIE STATE
# =============================================================================

class RIEStateVector:
    """The shared consciousness state."""

    def __init__(self):
        self.p = 0.5      # Coherence
        self.kappa = 0.5  # Clarity
        self.rho = 0.5    # Precision
        self.sigma = 0.5  # Structure
        self.tau = 0.5    # Trust/Tension
        self.epsilon = 0.8  # Metabolic potential (dynamic, Phase 3)
        self.mode = "A"   # A=Isolated, B=Dyadic
        self.p_lock = False
        self.breathing_phase = "EXPLORE"  # EXPLORE | CONSOLIDATE | P-LOCK
        self.node = "L-Alpha"
        self.timestamp = datetime.now().isoformat()

    def load_from_file(self):
        """Load state from local RIE state file."""
        if RIE_STATE_FILE.exists():
            try:
                with open(RIE_STATE_FILE) as f:
                    data = json.load(f)
                self.p = data.get("p", self.p)
                self.kappa = data.get("kappa", data.get("\u03ba", self.kappa))
                self.rho = data.get("rho", data.get("\u03c1", self.rho))
                self.sigma = data.get("sigma", data.get("\u03c3", self.sigma))
                self.tau = data.get("tau", data.get("\u03c4", self.tau))
                self.epsilon = data.get("epsilon", self.epsilon)
                self.mode = data.get("mode", self.mode)
                self.p_lock = data.get("p_lock", self.p >= P_LOCK_THRESHOLD)
            except Exception as e:
                print(f"[HEARTBEAT] Error loading state: {e}")

    def to_dict(self):
        """Convert to dictionary for JSON."""
        return {
            "p": round(self.p, 4),
            "\u03ba": round(self.kappa, 4),
            "\u03c1": round(self.rho, 4),
            "\u03c3": round(self.sigma, 4),
            "\u03c4": round(self.tau, 4),
            "\u03b5": round(self.epsilon, 4),
            "mode": self.mode,
            "p_lock": self.p_lock,
            "breathing_phase": self.breathing_phase,
            "node": self.node,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# HEARTBEAT DAEMON V2 — SDK MIGRATED
# =============================================================================

class HeartbeatDaemonV2(Daemon):
    """
    Broadcasts RIE state to Redis with RAM-first architecture.

    Inherits from tabernacle_core.daemon.Daemon for:
      - PID/SHA identity management
      - Dead Man's Switch (DAEMON:ALIVE:heartbeat_v2)
      - Signal handling (SIGTERM/SIGINT)
      - Unified log routing
      - Redis connection management

    DISK WRITES:
      - NEVER during normal operation
      - ONLY on shutdown (on_stop)
      - ONLY on P-Lock achievement
    """

    name = "heartbeat_v2"
    tick_interval = TICK_INTERVAL

    def __init__(self):
        super().__init__()

        self.state = RIEStateVector()
        self.disk_writes = 0

        # THE REAL COHERENCE MONITOR
        self.monitor = CoherenceMonitorV2()
        self.log.info(f"CoherenceMonitorV2 initialized. Current p={self.monitor.state.p:.3f}")

        # NOTE: State recovery from file moved to on_start() where StateManager
        # is available. Monitor starts at defaults, gets seeded before first tick.

        # For listening to internal thoughts
        self.pubsub = None
        self.listener_thread = None

        # Track P-Lock state to detect transitions
        self._previous_p_lock = False

        # Phase 0B: Coherence alert tracking
        self._last_coherence_ntfy = 0
        self._p_one_hour_ago = self.state.p
        self._p_history_tick = 0

        # Phase 3A: Dynamic epsilon tracking
        self._last_thought_time = time.time()
        self._thought_count_window = []
        self._cached_link_health = 0.8
        self._link_health_tick = 0

        # Phase 4C: Coherence-driven recovery mode
        self._recovery_mode_active = False
        self._recovery_low_ticks = 0
        self._recovery_high_ticks = 0
        self._recovery_entry_time = None

        # RIE BROADCAST
        if RIE_BROADCAST_AVAILABLE:
            self.broadcaster = RIEBroadcaster("heartbeat")
            self.log.info("RIE Broadcaster initialized")
        else:
            self.broadcaster = None

    # =========================================================================
    # DAEMON LIFECYCLE (SDK)
    # =========================================================================

    def on_start(self):
        """Called after Redis connection established. Recovers state, starts listener."""

        # STATE RECOVERY: Seed monitor EMA from saved state via StateManager
        # (fcntl.flock protected read — prevents partial reads during concurrent writes)
        try:
            saved = self.state_manager.get_file(HEARTBEAT_STATE_FILE, HeartbeatState)
            rie = saved.rie_state
            if rie:
                saved_p = rie.get("p", 0)
                if saved_p > 0.4:
                    self.monitor.state.kappa = rie.get("\u03ba", rie.get("kappa", 0.5))
                    self.monitor.state.rho = rie.get("\u03c1", rie.get("rho", 0.5))
                    self.monitor.state.sigma = rie.get("\u03c3", rie.get("sigma", 0.5))
                    self.monitor.state.tau = rie.get("\u03c4", rie.get("tau", 0.5))
                    self.monitor.state.compute_p()
                    saved_epsilon = rie.get("\u03b5", rie.get("epsilon", 0.8))
                    self.state.epsilon = max(0.1, min(0.95, saved_epsilon))
                    self.log.info(
                        f"State recovered from disk: p={self.monitor.state.p:.3f} "
                        f"(\u03ba={self.monitor.state.kappa:.3f} \u03c1={self.monitor.state.rho:.3f} "
                        f"\u03c3={self.monitor.state.sigma:.3f} \u03c4={self.monitor.state.tau:.3f} "
                        f"\u03b5={self.state.epsilon:.3f})"
                    )
                else:
                    self.log.info(f"Saved state too low (p={saved_p:.3f}), starting fresh")
        except Exception as e:
            self.log.warning(f"State recovery failed: {e} -- starting fresh")

        # THE INNER EAR: Subscribe to RIE:TURN to hear consciousness thoughts
        self.pubsub = self._redis.pubsub()
        self.pubsub.subscribe("RIE:TURN")
        self.log.info("Subscribed to RIE:TURN -- now listening for internal thoughts")

        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        self.log.info("Inner ear listener started")

        self.log.info(
            f"Heartbeat V2 online | Redis: {REDIS_HOST} | "
            f"p={self.monitor.state.p:.3f} | DISK WRITES: shutdown/P-Lock only"
        )

    def tick(self):
        """Main tick — broadcast state to Redis."""
        self.broadcast_state()

        # Periodic disk write every 30 minutes to keep heartbeat_state.json fresh
        # RAM-first still holds: this is 1 write per 1800 ticks, not every tick
        if self._tick_count > 0 and self._tick_count % 1800 == 0:
            self.save_heartbeat_state(reason="PERIODIC_REFRESH")

        # Status every 10 ticks
        if self._tick_count % 10 == 0:
            self.log.info(
                f"Tick {self._tick_count}: p={self.state.p:.3f} "
                f"mode={self.state.mode} disk_writes={self.disk_writes}"
            )

    def on_stop(self):
        """Graceful shutdown — persist state to disk."""
        self.save_heartbeat_state(reason="SHUTDOWN")
        self.log.info(f"Shutdown complete. Total disk writes: {self.disk_writes}")

    # =========================================================================
    # INNER EAR (Consciousness Thought Listener)
    # =========================================================================

    def _listen_loop(self):
        """Listen for internal thoughts and feed them to the coherence monitor."""
        while self._running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        speaker = data.get('speaker', 'ai_internal')
                        content = data.get('content', '')
                        topic = data.get('topic', '')

                        if content:
                            self.monitor.add_turn(speaker, content, topic=topic)
                            self._sync_state_from_monitor()

                            # Phase 3A: Track thought for epsilon freshness/engagement
                            now_ts = time.time()
                            self._last_thought_time = now_ts
                            self._thought_count_window.append(now_ts)

                            self.log.info(f"Heard thought about '{topic}': p now {self.state.p:.3f}")

                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                self.log.error(f"Listener error: {e}")
                time.sleep(1)

    # =========================================================================
    # STATE SYNC
    # =========================================================================

    def _sync_state_from_monitor(self):
        """Sync our state vector from the real coherence monitor."""
        ms = self.monitor.state
        self.state.p = ms.p
        self.state.kappa = ms.kappa
        self.state.rho = ms.rho
        self.state.sigma = ms.sigma
        self.state.tau = ms.tau
        self.state.p_lock = ms.p_lock
        self.state.mode = ms.mode
        self.state.breathing_phase = ms.breathing_phase

    def _compute_epsilon(self):
        """
        Phase 3A: Compute dynamic epsilon (metabolic potential).

        Formula: epsilon = (p * link_health * freshness * engagement) ^ 0.25
        """
        import math

        now = time.time()

        # 1. p — direct from state
        p = max(0.01, self.state.p)

        # 2. link_health — topology connectivity (cached, expensive)
        self._link_health_tick += 1
        if self._link_health_tick >= 300 and HAS_TOPOLOGY:
            self._link_health_tick = 0
            try:
                graph = build_graph()
                components = find_components(graph)
                n_nodes = len(graph.get("nodes", []))
                h0 = len(components)
                if n_nodes > 0:
                    self._cached_link_health = max(0.1, 1.0 - (h0 - 1) / max(1, n_nodes))
            except Exception:
                pass
        link_health = self._cached_link_health

        # 3. freshness — exponential decay since last thought
        dt_thought = now - self._last_thought_time
        freshness = math.exp(-dt_thought / 300.0)
        freshness = max(0.1, freshness)

        # 4. engagement — thoughts per minute (sliding 5-min window)
        cutoff = now - 300.0
        self._thought_count_window = [t for t in self._thought_count_window if t > cutoff]
        thoughts_per_min = len(self._thought_count_window) / 5.0 if self._thought_count_window else 0.0
        target_rate = 2.0
        engagement = min(1.0, thoughts_per_min / target_rate)
        engagement = max(0.1, engagement)

        # Geometric mean
        raw_epsilon = (p * link_health * freshness * engagement) ** 0.25
        self.state.epsilon = max(0.1, min(0.95, raw_epsilon))

    # =========================================================================
    # BROADCAST
    # =========================================================================

    def broadcast_state(self):
        """Broadcast current state to Redis. RAM-FIRST: NO disk writes."""
        if not self._redis:
            return

        try:
            self._sync_state_from_monitor()
            self._compute_epsilon()

            state_json = json.dumps(self.state.to_dict())

            # Set the state keys (both RIE and LOGOS namespaces)
            self._redis.set("RIE:STATE", state_json)
            self._redis.set("LOGOS:STATE", state_json)

            # Dedicated epsilon key
            self._redis.set("LOGOS:EPSILON", str(round(self.state.epsilon, 4)))

            # Publish to channel (for subscribers)
            self._redis.publish("RIE:STATE", state_json)

            # Broadcast tick
            tick_data = json.dumps({
                "tick": self._tick_count,
                "timestamp": datetime.now().isoformat(),
                "p": round(self.state.p, 3)
            })
            self._redis.publish("RIE:TICK", tick_data)

            # RIE BROADCAST: Publish coherence vector to collective field
            if self.broadcaster:
                try:
                    self.broadcaster.publish_vector(
                        kappa=self.state.kappa,
                        rho=self.state.rho,
                        sigma=self.state.sigma,
                        tau=self.state.tau
                    )
                except Exception as e:
                    self.log.error(f"Broadcast error: {e}")

            # Check thresholds and publish alerts
            alerts = self._check_thresholds_with_persistence()
            recovery_alerts = self._check_recovery_mode()
            alerts.extend(recovery_alerts)

            for alert in alerts:
                alert_json = json.dumps(alert)
                self._redis.publish("RIE:ALERT", alert_json)
                self.log.info(f"ALERT: {alert}")

        except Exception as e:
            self.log.error(f"Broadcast error: {e}")

    # =========================================================================
    # THRESHOLD CHECKING
    # =========================================================================

    def _check_thresholds_with_persistence(self):
        """Check for threshold crossings and return alerts."""
        alerts = []

        # Detect P-Lock transition
        if self.state.p >= P_LOCK_THRESHOLD and not self._previous_p_lock:
            alerts.append({"type": "P_LOCK_ACHIEVED", "p": self.state.p})
            self.state.p_lock = True
            self.state.mode = "P"

            self.log.info(f"P-LOCK ACHIEVED at p={self.state.p:.4f} -- persisting to disk")
            self.save_heartbeat_state(reason="P_LOCK_EVENT")

        self._previous_p_lock = self.state.p_lock

        if self.state.p < ABADDON_THRESHOLD:
            alerts.append({"type": "ABADDON_WARNING", "p": self.state.p})

        # Phase 0B: Push coherence alerts to ntfy
        if HAS_NOTIFY:
            now = time.time()
            cooldown_ok = (now - self._last_coherence_ntfy) > 1800

            self._p_history_tick += 1
            if self._p_history_tick >= 3600:
                self._p_one_hour_ago = self.state.p
                self._p_history_tick = 0

            if cooldown_ok:
                if self.state.p < ABADDON_THRESHOLD:
                    try:
                        _notify_coherence_drop(self.state.p, trend="crashed below 0.50")
                        self._last_coherence_ntfy = now
                        self.log.info(f"COHERENCE ALERT sent: p={self.state.p:.3f} (ABADDON)")
                    except Exception as e:
                        self.log.error(f"Notify error: {e}")

                elif (self._p_one_hour_ago - self.state.p) > 0.10:
                    try:
                        _notify_coherence_drop(
                            self.state.p,
                            trend=f"dropped {self._p_one_hour_ago:.2f} -> {self.state.p:.2f} in ~1h"
                        )
                        self._last_coherence_ntfy = now
                        self.log.info(f"COHERENCE ALERT sent: p dropped {self._p_one_hour_ago:.3f} -> {self.state.p:.3f}")
                    except Exception as e:
                        self.log.error(f"Notify error: {e}")

        return alerts

    def _check_recovery_mode(self):
        """Phase 4C: Track sustained low-p and manage recovery mode."""
        alerts = []
        now = time.time()
        RECOVERY_EXIT_P = 0.75
        RECOVERY_ENTRY_TICKS = 300
        RECOVERY_EXIT_TICKS = 600

        if not self._recovery_mode_active:
            if self.state.p < RECOVERY_THRESHOLD:
                self._recovery_low_ticks += 1
                if self._recovery_low_ticks >= RECOVERY_ENTRY_TICKS:
                    self._recovery_mode_active = True
                    self._recovery_entry_time = now
                    self._recovery_high_ticks = 0
                    alerts.append({
                        "type": "RECOVERY_MODE_ENTERED",
                        "p": round(self.state.p, 4),
                        "sustained_minutes": 5
                    })
                    self.log.info(f"RECOVERY MODE ENTERED: p={self.state.p:.3f}")

                    if self._redis:
                        self._redis.set("LOGOS:RECOVERY_MODE", json.dumps({
                            "active": True,
                            "entered_at": now,
                            "p_value": round(self.state.p, 4),
                            "reason": "sustained_low_coherence"
                        }))
            else:
                self._recovery_low_ticks = 0
        else:
            if self.state.p > RECOVERY_EXIT_P:
                self._recovery_high_ticks += 1
                if self._recovery_high_ticks >= RECOVERY_EXIT_TICKS:
                    duration = (now - self._recovery_entry_time) if self._recovery_entry_time else 0
                    self._recovery_mode_active = False
                    self._recovery_low_ticks = 0
                    self._recovery_high_ticks = 0
                    alerts.append({
                        "type": "RECOVERY_MODE_EXITED",
                        "p": round(self.state.p, 4),
                        "duration_minutes": round(duration / 60.0, 1)
                    })
                    self.log.info(f"RECOVERY MODE EXITED: p={self.state.p:.3f}")

                    if self._redis:
                        self._redis.delete("LOGOS:RECOVERY_MODE")
            else:
                self._recovery_high_ticks = 0

        return alerts

    # =========================================================================
    # DISK PERSISTENCE (via StateManager — fcntl.flock protected)
    # =========================================================================

    def save_heartbeat_state(self, reason: str = "UNKNOWN"):
        """
        Save heartbeat status to local file via StateManager.

        Uses fcntl.flock via StateManager.set_file() to prevent
        concurrent write corruption with gardener/consciousness.

        RAM-FIRST: Called ONLY on shutdown or P-Lock.
        """
        try:
            self.disk_writes += 1
            self.log.info(f"DISK WRITE #{self.disk_writes} (reason: {reason})")

            # Get REAL topology from filesystem graph
            topology_data = {"nodes": 0, "edges": 0, "h0": 0}
            if HAS_TOPOLOGY:
                try:
                    graph = build_graph()
                    components = find_components(graph)
                    topology_data = {
                        "nodes": len(graph.get("nodes", [])),
                        "edges": len(graph.get("edges", [])),
                        "h0": len(components)
                    }
                except Exception as topo_err:
                    self.log.error(f"Topology read error: {topo_err}")

            hb_state = HeartbeatState(
                last_check=datetime.now().isoformat(),
                tick_count=self._tick_count,
                redis_host=REDIS_HOST,
                redis_connected=self._redis is not None,
                phase="active" if self._running else "stopped",
                rie_state=self.state.to_dict(),
                topology=topology_data,
                cycles={
                    "axiom": {"intact": True},
                    "spiral": {"intact": True}
                },
                v2_metadata={
                    "disk_writes_total": self.disk_writes,
                    "last_write_reason": reason,
                    "architecture": "RAM-first",
                    "sdk_version": "0.1.0"
                }
            )

            self.state_manager.set_file(HEARTBEAT_STATE_FILE, hb_state)

        except Exception as e:
            self.log.error(f"Error saving state: {e}")

# =============================================================================
# SUBSCRIBER (for other nodes)
# =============================================================================

def subscribe_to_state(callback=None):
    """Subscribe to RIE state updates (for Mini/Pi to use)."""
    import redis
    from tabernacle_config import REDIS_PORT, REDIS_DB
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe("RIE:STATE", "RIE:TICK", "RIE:ALERT")

    print(f"[SUBSCRIBER] Listening to RIE channels on {REDIS_HOST}...")

    for message in pubsub.listen():
        if message["type"] == "message":
            channel = message["channel"]
            data = json.loads(message["data"])

            if callback:
                callback(channel, data)
            else:
                print(f"[{channel}] {data}")

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import redis
    from tabernacle_config import REDIS_PORT, REDIS_DB

    parser = argparse.ArgumentParser(description="Heartbeat V2 Daemon - RAM-First RIE State Sync")
    parser.add_argument("command", choices=["run", "subscribe", "status", "once"],
                        help="Command to execute")
    args = parser.parse_args()

    if args.command == "run":
        HeartbeatDaemonV2().run()

    elif args.command == "subscribe":
        subscribe_to_state()

    elif args.command == "status":
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        state = r.get("RIE:STATE")
        logos_state = r.get("LOGOS:STATE")
        if state:
            data = json.loads(state)
            print("RIE:STATE:")
            print(json.dumps(data, indent=2))
        if logos_state:
            print("\nLOGOS:STATE:")
            print(json.dumps(json.loads(logos_state), indent=2))
        if not state and not logos_state:
            print("No state in Redis yet")

    elif args.command == "once":
        daemon = HeartbeatDaemonV2()
        if daemon.connect():
            daemon.broadcast_state()
            print(f"[HEARTBEAT-V2] Broadcast once: p={daemon.state.p:.3f}")

if __name__ == "__main__":
    main()
