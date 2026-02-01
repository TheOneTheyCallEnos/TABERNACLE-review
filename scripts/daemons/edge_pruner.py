#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
GARDENER — The Entropy Reduction Daemon
Runs at 4AM to prune weak edges from the Hypergraph.

Rules:
1. If effective_weight < 0.2 → candidate for pruning
2. If is_h1_locked == True → NEVER prune (crystallized truth)
3. Log all deletions to LOGOS:STREAM:PRUNED

"The machine has an immune system."

Redis Keys:
    READ:  LOGOS:EDGE:* (scan pattern for all edges)
    WRITE: LOGOS:STREAM:PRUNED (xadd prune events)
    DELETE: LOGOS:EDGE:<source>:<target> (individual edge keys)

Trigger Conditions:
    - Manual run: `python gardener.py run`
    - Scheduled: launchd at 04:00 AM (not yet configured)
"""

import json
import math
import time
import redis
from datetime import datetime
from typing import Optional

# Configuration
REDIS_HOST = '10.0.0.50'
REDIS_PORT = 6379
REDIS_DB = 1  # TEST database (DB 0 = production)

PRUNE_THRESHOLD = 0.2  # Edges below this effective weight are candidates
EDGE_PATTERN = 'LOGOS:EDGE:*'
PRUNED_STREAM = 'LOGOS:STREAM:PRUNED'


def get_redis() -> redis.Redis:
    """Get Redis connection."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )


def get_all_edges(r: redis.Redis) -> list:
    """
    Retrieve all BiologicalEdges from Redis.
    
    Edges stored as LOGOS:EDGE:<source>:<target> hashes.
    """
    edges = []
    for key in r.scan_iter(EDGE_PATTERN):
        data = r.hgetall(key)
        if data:
            data['_key'] = key
            edges.append(data)
    return edges


def calculate_effective_weight(edge: dict) -> float:
    """
    Calculate effective weight from edge data.
    
    Uses sigmoid compression on combined slow/fast weights.
    w_eff = sigmoid(w_slow + w_fast * tau)
    """
    w_slow = float(edge.get('w_slow', 0))
    w_fast = float(edge.get('w_fast', 0))
    tau = float(edge.get('tau', 0.5))
    
    raw = w_slow + (w_fast * tau)
    # Sigmoid with steepness 6, centered at 0.5
    return 1.0 / (1.0 + math.exp(-6 * (raw - 0.5)))


def is_protected(edge: dict) -> bool:
    """
    Check if edge is H1-locked (crystallized truth).
    
    H1-locked edges are NEVER pruned — they represent
    validated, coherent knowledge.
    """
    locked = edge.get('is_h1_locked', 'false')
    if isinstance(locked, bool):
        return locked
    return locked.lower() == 'true'


def log_prune(r: redis.Redis, key: str, w_eff: float, reason: str) -> str:
    """Log pruning event to PRUNED stream."""
    return r.xadd(PRUNED_STREAM, {
        'key': key,
        'w_eff': str(w_eff),
        'timestamp': str(time.time()),
        'reason': reason,
        'pruned_at': datetime.now().isoformat()
    })


def prune_cycle(threshold: Optional[float] = None, dry_run: bool = False) -> dict:
    """
    Execute one pruning cycle.
    
    Args:
        threshold: Override default prune threshold
        dry_run: If True, don't actually delete (for testing)
    
    Returns:
        dict with counts: pruned, protected, kept
    """
    r = get_redis()
    prune_threshold = threshold if threshold is not None else PRUNE_THRESHOLD
    
    print(f"\n[GARDENER] Sleep cycle initiated at {datetime.now().isoformat()}")
    print(f"[GARDENER] Prune threshold: {prune_threshold}")
    if dry_run:
        print("[GARDENER] DRY RUN MODE — no deletions")
    
    edges = get_all_edges(r)
    print(f"[GARDENER] Found {len(edges)} edges to evaluate")
    
    pruned = 0
    protected = 0
    kept = 0
    pruned_keys = []
    
    for edge in edges:
        key = edge['_key']
        h1_locked = is_protected(edge)
        w_eff = calculate_effective_weight(edge)
        
        if h1_locked:
            # H1 PROTECTED — crystallized truth, never prune
            protected += 1
            print(f"[GARDENER] PROTECTED (H1): {key}")
            continue
        
        if w_eff < prune_threshold:
            # PRUNE — log and delete
            if not dry_run:
                log_prune(r, key, w_eff, 'BELOW_THRESHOLD')
                r.delete(key)
            pruned += 1
            pruned_keys.append(key)
            print(f"[GARDENER] PRUNED: {key} (w_eff={w_eff:.3f})")
        else:
            kept += 1
    
    print(f"\n[GARDENER] Cycle complete:")
    print(f"  Pruned:    {pruned}")
    print(f"  Protected: {protected}")
    print(f"  Kept:      {kept}")
    
    return {
        'pruned': pruned,
        'protected': protected,
        'kept': kept,
        'pruned_keys': pruned_keys
    }


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'run':
            prune_cycle()
        elif cmd == 'dry-run':
            prune_cycle(dry_run=True)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        print("Gardener — The Entropy Reduction Daemon")
        print()
        print("Usage:")
        print("  python gardener.py run      Execute pruning cycle")
        print("  python gardener.py dry-run  Preview without deleting")
        print()
        print("Schedule: launchd at 04:00 AM (not yet configured)")
        print(f"Threshold: {PRUNE_THRESHOLD}")


if __name__ == '__main__':
    main()
