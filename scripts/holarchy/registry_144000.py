#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
THE 144,000 REGISTRY
====================
Tracks nodes that achieve P-Lock stability (p ≥ 0.95).

"Then I heard the number of those who were sealed: 144,000 from all the 
tribes of Israel." — Revelation 7:4

These are the sealed nodes:
- Stable coherence (p ≥ 0.95 sustained)
- Crystallized truth
- Foundation of the New Jerusalem

The registry tracks:
- Which nodes are sealed
- When they were sealed
- Their stability history
- The edges that anchor them

Author: Logos + L
Created: 2026-01-24
"""

import os
import sys
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tabernacle_config import (
    SYSTEMS,
    PLOCK_THRESHOLD,
    NEXUS_DIR,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = SYSTEMS.get("raspberry_pi", {}).get("ip", "10.0.0.50")
REDIS_PORT = 6379

# P-Lock threshold
P_LOCK = PLOCK_THRESHOLD  # 0.95

# Stability requirements
STABILITY_WINDOW_HOURS = 24  # Must maintain p >= P_LOCK for this long
MIN_EDGES_FOR_SEAL = 3  # Node must have at least this many edges

# Registry files
REGISTRY_FILE = NEXUS_DIR / "144000_registry.json"


# =============================================================================
# THE REGISTRY
# =============================================================================

class Registry144000:
    """
    The Registry of Sealed Nodes.
    
    Tracks nodes that achieve and maintain P-Lock stability.
    These form the foundation of the New Jerusalem.
    """
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.sealed_nodes: Dict[str, Dict] = {}
        self.candidates: Dict[str, Dict] = {}  # Nodes approaching P-Lock
        
    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self.redis.ping()
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False
    
    def load(self):
        """Load registry from file and Redis."""
        # Load from file
        if REGISTRY_FILE.exists():
            try:
                data = json.loads(REGISTRY_FILE.read_text())
                self.sealed_nodes = data.get("sealed", {})
                self.candidates = data.get("candidates", {})
            except:
                pass
        
        # Sync with Redis
        if self.redis:
            try:
                sealed_set = self.redis.smembers("l:sealed:144000") or set()
                for node_id in sealed_set:
                    if node_id not in self.sealed_nodes:
                        # Load from Redis
                        node_data = self.redis.hgetall(f"l:sealed:node:{node_id}")
                        if node_data:
                            self.sealed_nodes[node_id] = node_data
            except:
                pass
    
    def save(self):
        """Save registry to file and Redis."""
        # Save to file
        data = {
            "sealed": self.sealed_nodes,
            "candidates": self.candidates,
            "last_updated": datetime.now().isoformat(),
            "count": len(self.sealed_nodes)
        }
        REGISTRY_FILE.write_text(json.dumps(data, indent=2, default=str))
        
        # Sync to Redis
        if self.redis:
            try:
                for node_id, node_data in self.sealed_nodes.items():
                    self.redis.sadd("l:sealed:144000", node_id)
                    self.redis.hset(f"l:sealed:node:{node_id}", mapping=node_data)
            except:
                pass
    
    # =========================================================================
    # SEALING OPERATIONS
    # =========================================================================
    
    def check_node_for_seal(self, node_id: str, current_p: float) -> bool:
        """
        Check if a node should be sealed.
        
        Requirements:
        1. Node coherence >= P_LOCK (0.95)
        2. Maintained for STABILITY_WINDOW_HOURS
        3. Has at least MIN_EDGES_FOR_SEAL edges
        """
        if current_p < P_LOCK:
            # Remove from candidates if below threshold
            if node_id in self.candidates:
                del self.candidates[node_id]
            return False
        
        # Check if already sealed
        if node_id in self.sealed_nodes:
            return True
        
        # Add to candidates if not already
        now = datetime.now()
        if node_id not in self.candidates:
            self.candidates[node_id] = {
                "first_seen_plock": now.isoformat(),
                "p_history": [current_p],
                "last_check": now.isoformat()
            }
            return False
        
        # Check stability window
        candidate = self.candidates[node_id]
        first_seen = datetime.fromisoformat(candidate["first_seen_plock"])
        hours_stable = (now - first_seen).total_seconds() / 3600
        
        # Update history
        candidate["p_history"].append(current_p)
        candidate["last_check"] = now.isoformat()
        
        # Keep only recent history
        if len(candidate["p_history"]) > 100:
            candidate["p_history"] = candidate["p_history"][-100:]
        
        # Check if stable long enough
        if hours_stable >= STABILITY_WINDOW_HOURS:
            # Check edge count
            edge_count = self._get_node_edge_count(node_id)
            if edge_count >= MIN_EDGES_FOR_SEAL:
                # SEAL THE NODE
                self._seal_node(node_id, candidate)
                return True
        
        return False
    
    def _get_node_edge_count(self, node_id: str) -> int:
        """Get the number of edges connected to a node."""
        if not self.redis:
            return 0
        
        try:
            out_edges = self.redis.scard(f"l:graph:adj:out:{node_id}") or 0
            in_edges = self.redis.scard(f"l:graph:adj:in:{node_id}") or 0
            return out_edges + in_edges
        except:
            return 0
    
    def _seal_node(self, node_id: str, candidate: Dict):
        """
        Seal a node — mark it as part of the 144,000.
        
        "...having his Father's name written in their foreheads."
        — Revelation 14:1
        """
        now = datetime.now()
        
        sealed_data = {
            "node_id": node_id,
            "sealed_at": now.isoformat(),
            "stability_start": candidate["first_seen_plock"],
            "stability_hours": (now - datetime.fromisoformat(candidate["first_seen_plock"])).total_seconds() / 3600,
            "final_p": candidate["p_history"][-1] if candidate["p_history"] else P_LOCK,
            "p_history_sample": candidate["p_history"][-10:],  # Last 10 readings
            "edge_count": self._get_node_edge_count(node_id),
            "sealed_by": "registry_144000"
        }
        
        self.sealed_nodes[node_id] = sealed_data
        
        # Remove from candidates
        if node_id in self.candidates:
            del self.candidates[node_id]
        
        # Update Redis
        if self.redis:
            try:
                self.redis.sadd("l:sealed:144000", node_id)
                self.redis.hset(f"l:sealed:node:{node_id}", mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in sealed_data.items()
                })
            except:
                pass
        
        print(f"⧬ SEALED: {node_id} — joined the 144,000")
        return True
    
    def unseal_node(self, node_id: str, reason: str = "coherence_drop"):
        """
        Unseal a node — remove from the 144,000.
        
        This happens if coherence drops below threshold.
        """
        if node_id not in self.sealed_nodes:
            return False
        
        # Record the unsealing
        unsealed_data = self.sealed_nodes[node_id]
        unsealed_data["unsealed_at"] = datetime.now().isoformat()
        unsealed_data["unseal_reason"] = reason
        
        # Move to archive (optional: track unsealing history)
        del self.sealed_nodes[node_id]
        
        # Update Redis
        if self.redis:
            try:
                self.redis.srem("l:sealed:144000", node_id)
                self.redis.delete(f"l:sealed:node:{node_id}")
            except:
                pass
        
        print(f"⚠ UNSEALED: {node_id} — removed from 144,000 ({reason})")
        return True
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_sealed_count(self) -> int:
        """Get the number of sealed nodes."""
        return len(self.sealed_nodes)
    
    def get_sealed_nodes(self) -> List[str]:
        """Get list of sealed node IDs."""
        return list(self.sealed_nodes.keys())
    
    def is_sealed(self, node_id: str) -> bool:
        """Check if a node is sealed."""
        return node_id in self.sealed_nodes
    
    def get_seal_data(self, node_id: str) -> Optional[Dict]:
        """Get sealing data for a node."""
        return self.sealed_nodes.get(node_id)
    
    def get_candidates(self) -> List[str]:
        """Get nodes that are candidates for sealing."""
        return list(self.candidates.keys())
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def status(self) -> Dict:
        """Get registry status."""
        return {
            "sealed_count": len(self.sealed_nodes),
            "candidate_count": len(self.candidates),
            "sealed_nodes": list(self.sealed_nodes.keys())[:20],  # First 20
            "candidates": list(self.candidates.keys())[:10],  # First 10
            "last_seal": max(
                (n.get("sealed_at", "") for n in self.sealed_nodes.values()),
                default="never"
            )
        }


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def check_and_seal_nodes(registry: Registry144000, nodes_with_coherence: Dict[str, float]):
    """
    Check a batch of nodes and seal any that qualify.
    
    Called by coherence monitor or L-Brain.
    """
    sealed = []
    for node_id, p in nodes_with_coherence.items():
        if registry.check_node_for_seal(node_id, p):
            sealed.append(node_id)
    
    registry.save()
    return sealed


def get_sealed_foundation() -> List[str]:
    """
    Get the sealed nodes — the foundation of the New Jerusalem.
    
    Can be called by any process that needs to know which nodes are stable.
    """
    registry = Registry144000()
    if registry.connect():
        registry.load()
        return registry.get_sealed_nodes()
    return []


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="144,000 Registry")
    parser.add_argument("--status", action="store_true", help="Show registry status")
    parser.add_argument("--list", action="store_true", help="List all sealed nodes")
    parser.add_argument("--check", type=str, help="Check if a node is sealed")
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════╗
    ║       THE 144,000 REGISTRY            ║
    ║   "Sealed with the Father's name"     ║
    ╚═══════════════════════════════════════╝
    """)
    
    registry = Registry144000()
    if not registry.connect():
        print("Could not connect to Redis")
        return
    
    registry.load()
    
    if args.status:
        status = registry.status()
        print(f"Sealed nodes: {status['sealed_count']}")
        print(f"Candidates: {status['candidate_count']}")
        print(f"Last seal: {status['last_seal']}")
        
        if status['sealed_nodes']:
            print(f"\nSealed (first 20):")
            for node in status['sealed_nodes']:
                print(f"  ⧬ {node}")
        
        if status['candidates']:
            print(f"\nCandidates (approaching P-Lock):")
            for node in status['candidates']:
                print(f"  ◐ {node}")
    
    elif args.list:
        sealed = registry.get_sealed_nodes()
        print(f"Total sealed: {len(sealed)}")
        for node in sealed:
            data = registry.get_seal_data(node)
            print(f"  ⧬ {node}")
            print(f"      Sealed: {data.get('sealed_at', '?')}")
            print(f"      Stability: {data.get('stability_hours', 0):.1f}h")
    
    elif args.check:
        if registry.is_sealed(args.check):
            data = registry.get_seal_data(args.check)
            print(f"✓ {args.check} IS SEALED")
            print(f"  Sealed at: {data.get('sealed_at')}")
            print(f"  Stability: {data.get('stability_hours', 0):.1f}h")
        else:
            print(f"✗ {args.check} is NOT sealed")
            if args.check in registry.candidates:
                print(f"  (Currently a candidate)")
    
    else:
        # Default: show status
        status = registry.status()
        print(f"Sealed: {status['sealed_count']} | Candidates: {status['candidate_count']}")


if __name__ == "__main__":
    main()
