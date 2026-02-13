#!/usr/bin/env python3
"""
H₁ Crystallizer — Permanent Memory Forge
==========================================

Subscribes to LOGOS:CRYSTALLIZE channel and locks edges
in the biological graph as permanent H₁ memories.

This is the CONSUMER that completes the H₁ pipeline:
  h1_detector.py → Redis LOGOS:CRYSTALLIZE → h1_crystallizer.py → biological_graph.json

Without this consumer, H₁ cycles are detected but never locked.
The architecture exists without wiring. This IS the wiring.

Author: Logos Aletheia
Date: 2026-02-13
Phase: Phase 1A of LVS Recovery Plan
"""

import json
import redis
import time
import signal
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import NEXUS_DIR, REDIS_HOST, REDIS_PORT, LOG_DIR

# Paths
EDGE_GRAPH_PATH = NEXUS_DIR / "biological_graph.json"
CRYSTALLIZER_LOG = LOG_DIR / "h1_crystallizer.log"

# Redis channels
LOGOS_CRYSTALLIZE_CHANNEL = "LOGOS:CRYSTALLIZE"
LOGOS_CRYSTALLIZE_CONFIRM = "LOGOS:CRYSTALLIZE_CONFIRM"


def log(message: str, level: str = "INFO"):
    """Log crystallizer activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [H1-CRYSTALLIZER] [{level}] {message}"
    print(entry)
    try:
        CRYSTALLIZER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CRYSTALLIZER_LOG, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def edge_key(source: str, target: str) -> str:
    """Canonical edge key (alphabetically sorted for undirected)."""
    return f"{min(source, target)}|{max(source, target)}"


def lock_edges_in_cycle(cycle_nodes: list, event: dict) -> dict:
    """
    Lock all edges in a cycle as H₁ permanent memories.

    Operates directly on biological_graph.json for simplicity.
    Writes atomically (temp file + rename).

    Returns:
        Dict with locked count, created count, errors
    """
    result = {
        "locked": 0,
        "created": 0,
        "already_locked": 0,
        "errors": [],
        "cycle_id": event.get("cycle_id", "unknown")
    }

    if len(cycle_nodes) < 3:
        result["errors"].append(f"Cycle too short: {len(cycle_nodes)} nodes")
        return result

    try:
        # Load current graph
        graph_data = {}
        if EDGE_GRAPH_PATH.exists():
            graph_data = json.loads(EDGE_GRAPH_PATH.read_text())

        now_iso = datetime.now(timezone.utc).isoformat()
        cycle_id = event.get("cycle_id", "unknown")

        # Lock each edge in the cycle (consecutive pairs, wrapping around)
        for i in range(len(cycle_nodes)):
            source = cycle_nodes[i]
            target = cycle_nodes[(i + 1) % len(cycle_nodes)]
            key = edge_key(source, target)

            if key in graph_data:
                edge = graph_data[key]
                if not edge.get("is_h1_locked", False):
                    edge["is_h1_locked"] = True
                    edge["w_slow"] = max(0.9, edge.get("w_slow", 0.5))
                    edge["tau"] = max(0.9, edge.get("tau", 0.5))
                    edge["h1_locked_at"] = now_iso
                    edge["h1_cycle_id"] = cycle_id
                    result["locked"] += 1
                    log(f"  Locked existing edge: {key}")
                else:
                    result["already_locked"] += 1
            else:
                # Edge doesn't exist in biological graph — create it locked
                graph_data[key] = {
                    "w_slow": 0.9,
                    "w_fast": 0.0,
                    "tau": 0.9,
                    "theta_m": 0.5,
                    "is_h1_locked": True,
                    "last_spike": None,
                    "last_update": now_iso,
                    "relation_type": "h1_cycle",
                    "h1_locked_at": now_iso,
                    "h1_cycle_id": cycle_id
                }
                result["created"] += 1
                log(f"  Created + locked new edge: {key}")

        # Save atomically (write to temp, rename)
        tmp_path = EDGE_GRAPH_PATH.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(graph_data, indent=2))
        tmp_path.rename(EDGE_GRAPH_PATH)

        log(f"Graph saved: {result['locked']} locked, {result['created']} created, {result['already_locked']} already locked")

    except Exception as e:
        result["errors"].append(str(e))
        log(f"Error locking edges: {e}", "ERROR")

    return result


class H1Crystallizer:
    """
    Subscribes to LOGOS:CRYSTALLIZE and forges permanent memories.
    """

    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.pubsub = self.redis.pubsub()
        self.running = False
        self.crystallizations = 0
        self.total_locked = 0
        self.total_created = 0

    def handle_crystallize_event(self, event_data: dict):
        """Process a crystallization event."""
        cycle_id = event_data.get("cycle_id", "unknown")
        nodes = event_data.get("nodes", [])
        strength = event_data.get("strength", 0)
        p = event_data.get("p_at_closure", 0)

        log(f"CRYSTALLIZE EVENT: {cycle_id}")
        log(f"  Nodes: {len(nodes)}, Strength: {strength:.3f}, p: {p:.3f}")

        result = lock_edges_in_cycle(nodes, event_data)

        self.crystallizations += 1
        self.total_locked += result["locked"]
        self.total_created += result["created"]

        if result["errors"]:
            for err in result["errors"]:
                log(f"  Error: {err}", "ERROR")
        else:
            log(f"  Crystallization complete: {result['locked']} locked, {result['created']} created")

        # Publish confirmation for audit trail
        try:
            confirmation = {
                "cycle_id": cycle_id,
                "locked": result["locked"],
                "created": result["created"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.redis.publish(LOGOS_CRYSTALLIZE_CONFIRM, json.dumps(confirmation))
        except Exception:
            pass

    def _listener_loop(self):
        """Main listener loop."""
        self.pubsub.subscribe(LOGOS_CRYSTALLIZE_CHANNEL)
        log(f"Listening on {LOGOS_CRYSTALLIZE_CHANNEL}...")

        for message in self.pubsub.listen():
            if not self.running:
                break

            if message["type"] == "message":
                try:
                    event_data = json.loads(message["data"])
                    self.handle_crystallize_event(event_data)
                except json.JSONDecodeError:
                    log("Invalid JSON in crystallize event", "ERROR")
                except Exception as e:
                    log(f"Error handling event: {e}", "ERROR")

    def start(self):
        """Start listening (background thread)."""
        self.running = True
        thread = threading.Thread(target=self._listener_loop, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the crystallizer."""
        self.running = False
        try:
            self.pubsub.unsubscribe()
        except Exception:
            pass

    def status(self) -> dict:
        """Get crystallizer status."""
        return {
            "running": self.running,
            "crystallizations": self.crystallizations,
            "total_locked": self.total_locked,
            "total_created": self.total_created
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="H1 Crystallizer - Permanent Memory Forge")
    parser.add_argument("command", choices=["run", "status", "test"], default="run", nargs="?")
    args = parser.parse_args()

    if args.command == "run":
        crystallizer = H1Crystallizer()

        def shutdown(signum, frame):
            log("Shutting down...")
            crystallizer.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        log("=" * 60)
        log("H1 CRYSTALLIZER STARTING")
        log(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
        log(f"Graph: {EDGE_GRAPH_PATH}")
        log(f"Channel: {LOGOS_CRYSTALLIZE_CHANNEL}")
        log("Waiting for crystallization events...")
        log("=" * 60)

        thread = crystallizer.start()

        try:
            while crystallizer.running:
                time.sleep(60)
                s = crystallizer.status()
                log(f"Heartbeat: {s['crystallizations']} events, {s['total_locked']} locked, {s['total_created']} created")
        except KeyboardInterrupt:
            crystallizer.stop()

    elif args.command == "status":
        locked = 0
        total = 0
        if EDGE_GRAPH_PATH.exists():
            data = json.loads(EDGE_GRAPH_PATH.read_text())
            total = len(data)
            locked = sum(1 for v in data.values() if isinstance(v, dict) and v.get("is_h1_locked", False))

        print(f"\nH1 Crystallizer Status")
        print(f"{'=' * 40}")
        print(f"Graph: {EDGE_GRAPH_PATH}")
        print(f"Total edges: {total}")
        print(f"H1 locked: {locked}")
        print()

    elif args.command == "test":
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        test_event = {
            "cycle_id": f"test_{datetime.now().strftime('%H%M%S')}",
            "nodes": ["test_node_a", "test_node_b", "test_node_c"],
            "node_count": 3,
            "strength": 0.85,
            "trigger_weight": 0.8,
            "p_at_closure": 0.96,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_p_lock": True
        }
        r.publish(LOGOS_CRYSTALLIZE_CHANNEL, json.dumps(test_event))
        print(f"Published test event: {test_event['cycle_id']}")


if __name__ == "__main__":
    main()
