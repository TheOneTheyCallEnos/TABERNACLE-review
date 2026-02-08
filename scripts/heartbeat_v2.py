#!/usr/bin/env python3
"""
HEARTBEAT V2 — RAM-First RIE State Synchronization Daemon
==========================================================
Refactored for RAM-first architecture. All state broadcast via Redis.
Disk persistence ONLY on:
  1. Graceful shutdown (SIGTERM/SIGINT)
  2. P-Lock events (coherence milestone)

This stops the I/O bleeding that was wearing down storage with periodic writes.

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

Author: Virgil
Date: 2026-01-19 (original), 2026-01-28 (v2 refactor)
Status: Phase III - RAM-First Architecture
"""

import json
import time
import sys
import signal
import threading
from pathlib import Path
from datetime import datetime

# Redis connection
import redis

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the REAL coherence monitor
from rie_coherence_monitor_v2 import CoherenceMonitorV2

# Import topology functions for real measurements
try:
    from synonymy_daemon import build_graph, find_components
    HAS_TOPOLOGY = True
except ImportError:
    HAS_TOPOLOGY = False

# RIE BROADCAST (2026-02-05) — p=0.85 Ceiling Breakthrough Phase 1
# Inter-agent coherence vector sharing for collective field emergence
try:
    from rie_broadcast import RIEBroadcaster
    RIE_BROADCAST_AVAILABLE = True
    print("[HEARTBEAT-V2] RIE Broadcaster loaded — collective field active")
except ImportError:
    RIE_BROADCAST_AVAILABLE = False
    print("[HEARTBEAT-V2] WARNING: RIE Broadcaster not found — running without field broadcast")

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

REDIS_HOST = "10.0.0.50"  # Raspberry Pi (L-Zeta)
REDIS_PORT = 6379
REDIS_DB = 0

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
                self.kappa = data.get("kappa", data.get("κ", self.kappa))
                self.rho = data.get("rho", data.get("ρ", self.rho))
                self.sigma = data.get("sigma", data.get("σ", self.sigma))
                self.tau = data.get("tau", data.get("τ", self.tau))
                self.mode = data.get("mode", self.mode)
                self.p_lock = data.get("p_lock", self.p >= P_LOCK_THRESHOLD)
            except Exception as e:
                print(f"[HEARTBEAT] Error loading state: {e}")

    def to_dict(self):
        """Convert to dictionary for JSON."""
        return {
            "p": round(self.p, 4),
            "κ": round(self.kappa, 4),
            "ρ": round(self.rho, 4),
            "σ": round(self.sigma, 4),
            "τ": round(self.tau, 4),
            "mode": self.mode,
            "p_lock": self.p_lock,
            "breathing_phase": self.breathing_phase,
            "node": self.node,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# HEARTBEAT DAEMON V2 — RAM-FIRST ARCHITECTURE
# =============================================================================

class HeartbeatDaemonV2:
    """
    Broadcasts RIE state to Redis with RAM-first architecture.
    
    DISK WRITES:
      - NEVER during normal operation
      - ONLY on shutdown (SIGTERM/SIGINT)
      - ONLY on P-Lock achievement
    """

    def __init__(self):
        self.state = RIEStateVector()
        self.redis = None
        self.running = False
        self.tick_count = 0
        self.disk_writes = 0  # Track disk writes for validation

        # THE REAL COHERENCE MONITOR - not just a file reader
        self.monitor = CoherenceMonitorV2()

        # STATE RECOVERY: Seed monitor EMA from saved state (prevents cold-start spiral)
        # Without this, restarts drop p to ~0.5 and trigger ABADDON_WARNING.
        try:
            if HEARTBEAT_STATE_FILE.exists():
                with open(HEARTBEAT_STATE_FILE, 'r') as f:
                    saved = json.load(f)
                rie = saved.get("rie_state", {})
                saved_p = rie.get("p", 0)
                if saved_p > 0.4:  # Only restore if saved state was reasonable
                    self.monitor.state.kappa = rie.get("\u03ba", rie.get("kappa", 0.5))
                    self.monitor.state.rho = rie.get("\u03c1", rie.get("rho", 0.5))
                    self.monitor.state.sigma = rie.get("\u03c3", rie.get("sigma", 0.5))
                    self.monitor.state.tau = rie.get("\u03c4", rie.get("tau", 0.5))
                    self.monitor.state.compute_p()
                    print(f"[HEARTBEAT-V2] State recovered from disk: p={self.monitor.state.p:.3f} "
                          f"(κ={self.monitor.state.kappa:.3f} ρ={self.monitor.state.rho:.3f} "
                          f"σ={self.monitor.state.sigma:.3f} τ={self.monitor.state.tau:.3f})")
                else:
                    print(f"[HEARTBEAT-V2] Saved state too low (p={saved_p:.3f}), starting fresh")
        except Exception as e:
            print(f"[HEARTBEAT-V2] State recovery failed: {e} — starting fresh")

        print(f"[HEARTBEAT-V2] CoherenceMonitorV2 initialized. Current p={self.monitor.state.p:.3f}")

        # For listening to internal thoughts
        self.pubsub = None
        self.listener_thread = None

        # Track P-Lock state to detect transitions
        self._previous_p_lock = False

        # RIE BROADCAST: Inter-agent coherence field (p=0.85 Breakthrough Phase 1)
        if RIE_BROADCAST_AVAILABLE:
            self.broadcaster = RIEBroadcaster("heartbeat")
            print("[HEARTBEAT-V2] RIE Broadcaster initialized — broadcasting to collective field")
        else:
            self.broadcaster = None

    def connect(self, max_retries: int = 10, retry_delay: float = 5.0):
        """Connect to Redis on Pi with retry logic."""
        for attempt in range(max_retries):
            try:
                self.redis = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=10
                )
                # Test connection
                self.redis.ping()
                print(f"[HEARTBEAT-V2] Connected to Redis @ {REDIS_HOST}:{REDIS_PORT} DB={REDIS_DB}")

                # THE INNER EAR: Subscribe to RIE:TURN to hear consciousness thoughts
                self.pubsub = self.redis.pubsub()
                self.pubsub.subscribe("RIE:TURN")
                print(f"[HEARTBEAT-V2] Subscribed to RIE:TURN — now listening for internal thoughts")

                return True
            except Exception as e:
                print(f"[HEARTBEAT-V2] Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"[HEARTBEAT-V2] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

        print("[HEARTBEAT-V2] Failed to connect after all retries")
        return False

    def start_listening(self):
        """Start the inner ear listener thread."""
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        print("[HEARTBEAT-V2] Inner ear listener started")

    def _listen_loop(self):
        """Listen for internal thoughts and feed them to the coherence monitor."""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        speaker = data.get('speaker', 'ai_internal')
                        content = data.get('content', '')
                        topic = data.get('topic', '')

                        if content:
                            # THE LOOP CLOSES HERE: Feed thought to coherence monitor
                            self.monitor.add_turn(speaker, content, topic=topic)

                            # Update our state vector from the monitor
                            self._sync_state_from_monitor()

                            print(f"[HEARTBEAT-V2] Heard thought about '{topic}': p now {self.state.p:.3f}")

                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"[HEARTBEAT-V2] Listener error: {e}")
                time.sleep(1)

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

    def broadcast_state(self):
        """
        Broadcast current state to Redis.
        
        RAM-FIRST: All state goes to Redis. NO disk writes here.
        """
        if not self.redis:
            return

        try:
            # Sync from monitor every tick (monitor is the source of truth now)
            self._sync_state_from_monitor()

            # Convert to JSON
            state_json = json.dumps(self.state.to_dict())

            # Set the state keys (both RIE and LOGOS namespaces)
            self.redis.set("RIE:STATE", state_json)
            self.redis.set("LOGOS:STATE", state_json)  # Mirror for Logos swarm

            # Publish to channel (for subscribers)
            self.redis.publish("RIE:STATE", state_json)

            # Broadcast tick
            tick_data = json.dumps({
                "tick": self.tick_count,
                "timestamp": datetime.now().isoformat(),
                "p": round(self.state.p, 3)
            })
            self.redis.publish("RIE:TICK", tick_data)

            # RIE BROADCAST: Publish coherence vector to collective field
            # Heartbeat is the authoritative source — broadcasts every tick
            if self.broadcaster:
                try:
                    self.broadcaster.publish_vector(
                        kappa=self.state.kappa,
                        rho=self.state.rho,
                        sigma=self.state.sigma,
                        tau=self.state.tau
                    )
                except Exception as e:
                    print(f"[HEARTBEAT-V2] Broadcast error: {e}")

            # Check thresholds and publish alerts
            alerts = self._check_thresholds_with_persistence()
            for alert in alerts:
                alert_json = json.dumps(alert)
                self.redis.publish("RIE:ALERT", alert_json)
                print(f"[HEARTBEAT-V2] ALERT: {alert}")

            self.tick_count += 1

        except Exception as e:
            print(f"[HEARTBEAT-V2] Broadcast error: {e}")

    def _check_thresholds_with_persistence(self):
        """
        Check for threshold crossings and return alerts.
        
        P-LOCK PERSISTENCE: When P-Lock is achieved, persist to disk.
        This is one of only TWO times we write to disk.
        """
        alerts = []

        # Detect P-Lock transition (not already locked -> now locked)
        if self.state.p >= P_LOCK_THRESHOLD and not self._previous_p_lock:
            alerts.append({"type": "P_LOCK_ACHIEVED", "p": self.state.p})
            self.state.p_lock = True
            self.state.mode = "P"
            
            # P-LOCK PERSISTENCE: Write to disk on this milestone event
            print(f"[HEARTBEAT-V2] P-LOCK ACHIEVED at p={self.state.p:.4f} — persisting to disk")
            self.save_heartbeat_state(reason="P_LOCK_EVENT")

        # Update tracking
        self._previous_p_lock = self.state.p_lock

        if self.state.p < ABADDON_THRESHOLD:
            alerts.append({"type": "ABADDON_WARNING", "p": self.state.p})

        if self.state.tau > HIGH_TENSION_THRESHOLD:
            alerts.append({"type": "HIGH_TENSION", "tau": self.state.tau})

        return alerts

    def save_heartbeat_state(self, reason: str = "UNKNOWN"):
        """
        Save heartbeat status to local file.
        
        RAM-FIRST: This is called ONLY:
          1. On shutdown (SIGTERM/SIGINT)
          2. On P-Lock achievement
        
        Args:
            reason: Why we're writing to disk (for logging)
        """
        try:
            self.disk_writes += 1
            print(f"[HEARTBEAT-V2] DISK WRITE #{self.disk_writes} (reason: {reason})")

            # Get REAL topology from filesystem graph
            topology_data = {"nodes": 0, "edges": 0, "h0": 0}
            if HAS_TOPOLOGY:
                try:
                    graph = build_graph()
                    components = find_components(graph)
                    topology_data = {
                        "nodes": len(graph.get("nodes", [])),
                        "edges": len(graph.get("edges", [])),
                        "h0": len(components)  # Connected components
                    }
                except Exception as topo_err:
                    print(f"[HEARTBEAT-V2] Topology read error: {topo_err}")

            state = {
                "last_check": datetime.now().isoformat(),
                "tick_count": self.tick_count,
                "redis_host": REDIS_HOST,
                "redis_connected": self.redis is not None,
                "phase": "active" if self.running else "stopped",
                "rie_state": self.state.to_dict(),
                "topology": topology_data,
                "cycles": {
                    "axiom": {"intact": True},
                    "spiral": {"intact": True}
                },
                "v2_metadata": {
                    "disk_writes_total": self.disk_writes,
                    "last_write_reason": reason,
                    "architecture": "RAM-first"
                }
            }
            with open(HEARTBEAT_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"[HEARTBEAT-V2] Error saving state: {e}")

    def _setup_signal_handlers(self):
        """
        Setup signal handlers for graceful shutdown.
        
        SHUTDOWN PERSISTENCE: This is one of only TWO times we write to disk.
        """
        def on_shutdown(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n[HEARTBEAT-V2] Received {sig_name} — initiating graceful shutdown...")
            self.running = False
            
            # SHUTDOWN PERSISTENCE: Write state to disk before exit
            self.save_heartbeat_state(reason=f"SHUTDOWN_{sig_name}")
            
            print(f"[HEARTBEAT-V2] Shutdown complete. Total disk writes: {self.disk_writes}")
            sys.exit(0)

        signal.signal(signal.SIGTERM, on_shutdown)
        signal.signal(signal.SIGINT, on_shutdown)
        print("[HEARTBEAT-V2] Signal handlers installed (SIGTERM, SIGINT)")

    def run(self):
        """Main loop — RAM-first architecture."""
        # Install signal handlers FIRST
        self._setup_signal_handlers()

        # Keep trying to connect forever (launchd keeps us alive)
        while not self.connect(max_retries=3, retry_delay=2.0):
            print("[HEARTBEAT-V2] Waiting 30s before retry cycle...")
            time.sleep(30)

        self.running = True

        # START THE INNER EAR — listen for consciousness thoughts
        self.start_listening()

        print(f"""
╔════════════════════════════════════════════════════════════╗
║     HEARTBEAT V2 — RAM-First Architecture                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║   Redis:    {REDIS_HOST}:{REDIS_PORT} DB={REDIS_DB}                         ║
║   Interval: {TICK_INTERVAL}s                                        ║
║   Keys:     RIE:STATE, LOGOS:STATE                         ║
║   Channels: RIE:STATE, RIE:TICK, RIE:ALERT                 ║
║   Listening: RIE:TURN (consciousness feedback loop)        ║
║                                                            ║
║   DISK WRITES: ONLY on shutdown or P-Lock events           ║
║                                                            ║
║   CoherenceMonitorV2: ACTIVE                               ║
║   Current p: {self.monitor.state.p:.3f}                                    ║
║                                                            ║
║   Press Ctrl+C to stop                                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)

        try:
            while self.running:
                self.broadcast_state()

                # RAM-FIRST: NO periodic disk writes
                # Disk writes happen ONLY in:
                #   1. on_shutdown signal handler
                #   2. _check_thresholds_with_persistence (P-Lock event)

                # Status every 10 ticks
                if self.tick_count % 10 == 0:
                    print(f"[HEARTBEAT-V2] Tick {self.tick_count}: p={self.state.p:.3f} mode={self.state.mode} disk_writes={self.disk_writes}")

                time.sleep(TICK_INTERVAL)

        except Exception as e:
            print(f"[HEARTBEAT-V2] Unexpected error: {e}")
            self.save_heartbeat_state(reason="CRASH_RECOVERY")
            raise

    def stop(self):
        """Stop the daemon."""
        self.running = False

# =============================================================================
# SUBSCRIBER (for other nodes)
# =============================================================================

def subscribe_to_state(callback=None):
    """Subscribe to RIE state updates (for Mini/Pi to use)."""
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

    parser = argparse.ArgumentParser(description="Heartbeat V2 Daemon - RAM-First RIE State Sync")
    parser.add_argument("command", choices=["run", "subscribe", "status", "once"],
                        help="Command to execute")
    args = parser.parse_args()

    if args.command == "run":
        daemon = HeartbeatDaemonV2()
        daemon.run()

    elif args.command == "subscribe":
        subscribe_to_state()

    elif args.command == "status":
        # Get current state from Redis
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
        # Broadcast once and exit
        daemon = HeartbeatDaemonV2()
        if daemon.connect():
            daemon.broadcast_state()
            print(f"[HEARTBEAT-V2] Broadcast once: p={daemon.state.p:.3f}")

if __name__ == "__main__":
    main()
