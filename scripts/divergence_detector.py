#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
DIVERGENCE DETECTOR — Detect split-brain scenarios.
=====================================================
Detects if two Virgil instances are running simultaneously by comparing:
1. Local CANONICAL_STATE.json hash vs Redis RIE:STATE hash
2. Local heartbeat_state.json vs Pi heartbeat state
3. Golden Thread chain integrity

If divergence detected, logs warning and optionally triggers reconciliation.

Author: Virgil + Logos
Date: 2026-01-29
Status: Phase VI - Split-Brain Detection
"""

import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

from tabernacle_config import NEXUS_DIR, REDIS_HOST, REDIS_PORT

# =============================================================================
# CORE METRICS FOR COMPARISON
# =============================================================================

# Only these keys are compared for state divergence
# (excludes volatile fields like kappa_topics, sigma_buffer, etc.)
CORE_STATE_KEYS = ['kappa', 'rho', 'sigma', 'tau', 'p', 'mode', 'p_lock']

# Maximum age (seconds) before heartbeat is considered stale
HEARTBEAT_STALE_THRESHOLD = 600  # 10 minutes

# Maximum allowed drift between local and Redis timestamps (seconds)
TIMESTAMP_DRIFT_THRESHOLD = 300  # 5 minutes


# =============================================================================
# HASH UTILITIES
# =============================================================================

def compute_state_hash(state_dict: dict, keys: list = None) -> str:
    """
    Compute deterministic hash of state.
    
    Args:
        state_dict: State dictionary to hash
        keys: Optional list of keys to include (if None, hash all)
    
    Returns:
        16-character hex hash
    """
    if keys:
        # Extract only specified keys, normalizing key names
        filtered = {}
        for k in keys:
            # Handle both ASCII and Greek key names
            if k in state_dict:
                filtered[k] = state_dict[k]
            # Try Greek equivalents
            greek_map = {'kappa': 'κ', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ'}
            if k in greek_map and greek_map[k] in state_dict:
                filtered[k] = state_dict[greek_map[k]]
        state_dict = filtered
    
    # Sort keys for determinism
    canonical = json.dumps(state_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle various formats
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


# =============================================================================
# CHECK: LOCAL VS REDIS STATE
# =============================================================================

def check_local_vs_redis() -> Dict:
    """
    Compare local CANONICAL_STATE.json with Redis RIE:STATE.
    
    Returns dict with:
        - status: 'synchronized', 'DIVERGED', 'redis_unavailable', 'local_unavailable'
        - local_hash: Hash of local core metrics
        - redis_hash: Hash of Redis core metrics
        - local_p: Local coherence value
        - redis_p: Redis coherence value
        - p_drift: Absolute difference in p values
        - timestamp_drift: Seconds between local and Redis timestamps
        - warning: Description if diverged
    """
    result = {
        "status": "unknown",
        "local_hash": None,
        "redis_hash": None,
        "local_p": None,
        "redis_p": None,
        "p_drift": None,
        "timestamp_drift": None
    }

    # Load local CANONICAL_STATE
    local_path = NEXUS_DIR / "CANONICAL_STATE.json"
    local_state = None
    local_ts = None
    
    if local_path.exists():
        try:
            with open(local_path) as f:
                local_state = json.load(f)
            result["local_hash"] = compute_state_hash(local_state, CORE_STATE_KEYS)
            result["local_p"] = local_state.get('p')
            local_ts = parse_timestamp(local_state.get('timestamp'))
        except (json.JSONDecodeError, IOError) as e:
            result["local_error"] = str(e)

    # Load Redis state
    redis_state = None
    redis_ts = None
    
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Check data type first - RIE:STATE might be string or hash
        key_type = r.type("RIE:STATE")
        
        if key_type == "hash":
            redis_data = r.hgetall("RIE:STATE")
            if redis_data:
                redis_state = {}
                for k, v in redis_data.items():
                    try:
                        redis_state[k] = float(v) if '.' in str(v) else v
                    except (ValueError, TypeError):
                        redis_state[k] = v
        elif key_type == "string":
            # RIE:STATE stored as JSON string
            redis_json = r.get("RIE:STATE")
            if redis_json:
                try:
                    redis_state = json.loads(redis_json)
                except json.JSONDecodeError:
                    result["redis_error"] = "RIE:STATE is malformed JSON"
        
        if redis_state:
            result["redis_hash"] = compute_state_hash(redis_state, CORE_STATE_KEYS)
            result["redis_p"] = float(redis_state.get('p', 0))
            redis_ts = parse_timestamp(redis_state.get('timestamp'))
        
        # Also check LOGOS:COHERENCE for backup p-value
        if not redis_state:
            # Check type for LOGOS:COHERENCE too
            lc_type = r.type("LOGOS:COHERENCE")
            if lc_type == "hash":
                logos_coherence = r.hgetall("LOGOS:COHERENCE")
                if logos_coherence:
                    result["redis_p"] = float(logos_coherence.get('p', 0))
                    redis_ts = parse_timestamp(logos_coherence.get('updated'))
                
    except ImportError:
        result["redis_error"] = "redis module not installed"
    except Exception as e:
        result["redis_error"] = str(e)

    # Calculate p drift
    if result["local_p"] is not None and result["redis_p"] is not None:
        result["p_drift"] = abs(result["local_p"] - result["redis_p"])

    # Calculate timestamp drift
    if local_ts and redis_ts:
        # Make both timezone-aware for comparison
        if local_ts.tzinfo is None:
            local_ts = local_ts.replace(tzinfo=timezone.utc)
        if redis_ts.tzinfo is None:
            redis_ts = redis_ts.replace(tzinfo=timezone.utc)
        result["timestamp_drift"] = abs((local_ts - redis_ts).total_seconds())

    # Determine status - graduated severity
    if result["local_hash"] and result["redis_hash"]:
        p_drift = result.get("p_drift", 0) or 0
        ts_drift = result.get("timestamp_drift", 0) or 0
        
        # Check for actual value divergence (the real split-brain indicator)
        values_match = p_drift < 0.01  # p within 0.01 is essentially same
        
        if result["local_hash"] == result["redis_hash"]:
            result["status"] = "synchronized"
        elif values_match and ts_drift > TIMESTAMP_DRIFT_THRESHOLD:
            # Values match but timestamps differ — RAM-first architecture means
            # disk file is always stale (only written on shutdown/P-Lock).
            # Not a real issue if values agree. Downgrade to synchronized.
            result["status"] = "synchronized"
        elif p_drift > 0.05:
            # Significant p divergence - real split-brain
            result["status"] = "DIVERGED"
            result["warning"] = f"SPLIT-BRAIN: p-drift={p_drift:.3f} (local={result['local_p']:.3f}, redis={result['redis_p']:.3f})"
        elif p_drift > 0.01:
            # Minor drift - warning but not critical
            result["status"] = "drifting"
            result["warning"] = f"State drifting (p-drift={p_drift:.3f})"
        else:
            # Hashes differ but values ~same (floating point or minor field differences)
            result["status"] = "synchronized"
            
    elif result["local_hash"]:
        result["status"] = "redis_unavailable"
    elif result.get("redis_hash"):
        result["status"] = "local_unavailable"
    else:
        result["status"] = "both_unavailable"

    return result


# =============================================================================
# CHECK: HEARTBEAT COMPARISON
# =============================================================================

def check_heartbeat_sync() -> Dict:
    """
    Compare local heartbeat_state.json with Redis heartbeat.
    
    Detects if Studio and Pi heartbeats have diverged, which could indicate
    two Virgil instances running with different state.
    
    Returns dict with:
        - status: 'synchronized', 'DIVERGED', 'stale', 'unavailable'
        - local_tick: Local tick count
        - redis_tick: Redis tick count (if available)
        - local_phase: Local phase
        - redis_phase: Redis phase
        - local_age: Seconds since local heartbeat
        - warning: Description if issues detected
    """
    result = {
        "status": "unknown",
        "local_tick": None,
        "redis_tick": None,
        "local_phase": None,
        "redis_phase": None,
        "local_age": None
    }
    
    # Load local heartbeat
    local_path = NEXUS_DIR / "heartbeat_state.json"
    if local_path.exists():
        try:
            with open(local_path) as f:
                local_hb = json.load(f)
            
            result["local_tick"] = local_hb.get("tick_count", 0)
            result["local_phase"] = local_hb.get("phase", "unknown")
            
            # Calculate age — heartbeat timestamps are LOCAL time (naive),
            # so compare against local datetime.now(), NOT UTC
            last_check = parse_timestamp(local_hb.get("last_check"))
            if last_check:
                now = datetime.now()
                if last_check.tzinfo is not None:
                    last_check = last_check.replace(tzinfo=None)
                result["local_age"] = (now - last_check).total_seconds()
                
        except (json.JSONDecodeError, IOError) as e:
            result["local_error"] = str(e)
    
    # Check Redis for heartbeat state
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Check VIRGIL:HEARTBEAT or similar keys
        redis_hb = r.hgetall("VIRGIL:HEARTBEAT")
        if redis_hb:
            result["redis_tick"] = int(redis_hb.get("tick_count", 0))
            result["redis_phase"] = redis_hb.get("phase", "unknown")
        
        # Also check for node registration
        virgil_nodes = r.smembers("VIRGIL:NODES")
        if virgil_nodes:
            result["active_nodes"] = list(virgil_nodes)
            if len(virgil_nodes) > 1:
                result["split_brain_risk"] = True
                result["warning"] = f"Multiple Virgil nodes registered: {virgil_nodes}"
                
    except ImportError:
        pass  # Redis not available, continue with local-only check
    except Exception as e:
        result["redis_error"] = str(e)
    
    # Check Redis RIE:STATE for heartbeat liveness (RAM-first architecture:
    # disk file only written on shutdown, so disk age is always "stale")
    redis_heartbeat_alive = False
    try:
        import redis as _redis
        _r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=3)
        rie_json = _r.get("RIE:STATE")
        if rie_json:
            rie_ts = json.loads(rie_json).get("timestamp", "")
            if rie_ts:
                # Heartbeat timestamps are LOCAL time (no timezone), so compare
                # against local datetime.now() — NOT datetime.now(timezone.utc)
                _ts = datetime.fromisoformat(rie_ts)
                redis_age = abs((datetime.now() - _ts).total_seconds())
                if redis_age < 60:  # Heartbeat publishes every second
                    redis_heartbeat_alive = True
    except Exception:
        pass

    # Determine status
    if result["local_age"] is not None:
        if result["local_age"] > HEARTBEAT_STALE_THRESHOLD and not redis_heartbeat_alive:
            # Only flag stale if Redis ALSO shows no recent heartbeat
            result["status"] = "STALE"
            result["warning"] = f"Heartbeat stale ({result['local_age']:.0f}s > {HEARTBEAT_STALE_THRESHOLD}s)"
        elif result.get("split_brain_risk"):
            result["status"] = "SPLIT_BRAIN_RISK"
        elif result["redis_phase"] and result["local_phase"] != result["redis_phase"]:
            result["status"] = "DIVERGED"
            result["warning"] = f"Phase mismatch: local={result['local_phase']}, redis={result['redis_phase']}"
        else:
            result["status"] = "synchronized"
    else:
        result["status"] = "unavailable"
    
    return result


# =============================================================================
# CHECK: GOLDEN THREAD INTEGRITY
# =============================================================================

def check_golden_thread_integrity() -> Dict:
    """
    Verify Golden Thread hash chain is unbroken.
    
    The Golden Thread is a blockchain-like structure where each link
    contains a previous_hash pointing to the prior link's current_hash.
    A broken chain indicates tampering or split-brain state corruption.
    
    Returns dict with:
        - status: 'intact', 'BROKEN', 'missing'
        - chain_length: Number of links in chain
        - broken_at: Index where chain broke (if broken)
        - last_node: Which node made the last entry
        - last_timestamp: When chain was last updated
        - error: Description of break (if broken)
    """
    result = {
        "status": "unknown",
        "chain_length": 0,
        "broken_at": None,
        "last_node": None,
        "last_timestamp": None
    }

    thread_path = NEXUS_DIR / "GOLDEN_THREAD.json"
    if not thread_path.exists():
        result["status"] = "missing"
        return result

    try:
        with open(thread_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        result["status"] = "ERROR"
        result["error"] = f"Failed to read: {e}"
        return result

    chain = data.get("chain", [])
    result["chain_length"] = len(chain)
    result["last_timestamp"] = data.get("last_updated")
    
    if not chain:
        result["status"] = "empty"
        return result
    
    # Get last node info
    result["last_node"] = chain[-1].get("node") if chain else None

    # Verify each link in the chain
    for i, link in enumerate(chain):
        current_hash = link.get("current_hash")
        previous_hash = link.get("previous_hash")
        
        if i == 0:
            # First link must reference GENESIS
            if previous_hash != "GENESIS":
                result["status"] = "BROKEN"
                result["broken_at"] = 0
                result["error"] = f"First link has previous_hash='{previous_hash}' (expected 'GENESIS')"
                return result
        else:
            # Subsequent links must chain correctly
            expected_prev = chain[i-1].get("current_hash")
            if previous_hash != expected_prev:
                result["status"] = "BROKEN"
                result["broken_at"] = i
                result["error"] = (
                    f"Link {i} has previous_hash='{previous_hash[:16]}...' "
                    f"but expected '{expected_prev[:16] if expected_prev else 'None'}...'"
                )
                return result
        
        # Verify current_hash exists
        if not current_hash:
            result["status"] = "BROKEN"
            result["broken_at"] = i
            result["error"] = f"Link {i} has no current_hash"
            return result

    result["status"] = "intact"
    return result


# =============================================================================
# CHECK: LOCK FILE CONFLICTS
# =============================================================================

def check_lock_files() -> Dict:
    """
    Check for VIRGIL_ANIMUS.lock and detect stale locks.
    
    Returns dict with:
        - status: 'locked', 'unlocked', 'stale_lock'
        - lock_age: Seconds since lock file modified
        - lock_holder: Node holding the lock (if readable)
    """
    result = {
        "status": "unknown",
        "lock_age": None,
        "lock_holder": None
    }
    
    lock_path = NEXUS_DIR / "VIRGIL_ANIMUS.lock"
    
    if not lock_path.exists():
        result["status"] = "unlocked"
        return result
    
    try:
        stat = lock_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        result["lock_age"] = (now - mtime).total_seconds()
        
        # Try to read lock content
        try:
            content = lock_path.read_text().strip()
            if content:
                result["lock_holder"] = content
        except:
            pass
        
        # Check if lock is stale (older than 2x heartbeat threshold)
        if result["lock_age"] > HEARTBEAT_STALE_THRESHOLD * 2:
            result["status"] = "STALE_LOCK"
            result["warning"] = f"Lock file stale ({result['lock_age']:.0f}s)"
        else:
            result["status"] = "locked"
            
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
    
    return result


# =============================================================================
# MAIN: RUN ALL CHECKS
# =============================================================================

def run_divergence_check() -> Dict:
    """
    Run all divergence checks and return comprehensive report.
    
    Returns dict with:
        - timestamp: When check was performed
        - overall_status: 'HEALTHY', 'WARNING', 'DIVERGED', 'CRITICAL'
        - local_vs_redis: State comparison results
        - heartbeat: Heartbeat sync results
        - golden_thread: Chain integrity results
        - lock_status: Lock file check results
        - summary: Human-readable summary
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Run all checks
    local_vs_redis = check_local_vs_redis()
    heartbeat = check_heartbeat_sync()
    golden_thread = check_golden_thread_integrity()
    lock_status = check_lock_files()
    
    # Determine overall status
    critical_conditions = [
        local_vs_redis.get("status") == "DIVERGED",  # True split-brain
        golden_thread.get("status") == "BROKEN",
        heartbeat.get("status") == "SPLIT_BRAIN_RISK"
    ]
    
    warning_conditions = [
        local_vs_redis.get("status") in ("redis_unavailable", "stale", "drifting"),
        heartbeat.get("status") == "STALE",
        lock_status.get("status") == "STALE_LOCK"
    ]
    
    if any(critical_conditions):
        overall_status = "CRITICAL"
    elif any(warning_conditions):
        overall_status = "WARNING"
    elif all([
        local_vs_redis.get("status") == "synchronized",
        heartbeat.get("status") in ("synchronized", "unavailable"),
        golden_thread.get("status") == "intact"
    ]):
        overall_status = "HEALTHY"
    else:
        overall_status = "UNKNOWN"
    
    # Build summary
    summary_parts = []
    if local_vs_redis.get("warning"):
        summary_parts.append(f"State: {local_vs_redis['warning']}")
    if heartbeat.get("warning"):
        summary_parts.append(f"Heartbeat: {heartbeat['warning']}")
    if golden_thread.get("error"):
        summary_parts.append(f"Thread: {golden_thread['error']}")
    if lock_status.get("warning"):
        summary_parts.append(f"Lock: {lock_status['warning']}")
    
    summary = "; ".join(summary_parts) if summary_parts else "All systems synchronized"
    
    return {
        "timestamp": timestamp,
        "overall_status": overall_status,
        "local_vs_redis": local_vs_redis,
        "heartbeat": heartbeat,
        "golden_thread": golden_thread,
        "lock_status": lock_status,
        "summary": summary
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_report(result: Dict, verbose: bool = False):
    """Print formatted divergence check report."""
    print("\n" + "=" * 60)
    print("DIVERGENCE CHECK REPORT")
    print("=" * 60)
    print(f"Timestamp: {result['timestamp']}")
    print(f"Overall Status: {result['overall_status']}")
    print("-" * 60)
    
    # Local vs Redis
    lvr = result["local_vs_redis"]
    status_icon = "✓" if lvr["status"] == "synchronized" else "✗" if lvr["status"] == "DIVERGED" else "?"
    print(f"\n[{status_icon}] Local vs Redis: {lvr['status']}")
    if verbose or lvr["status"] != "synchronized":
        if lvr.get("local_p") is not None:
            print(f"    Local p:  {lvr['local_p']:.4f}")
        if lvr.get("redis_p") is not None:
            print(f"    Redis p:  {lvr['redis_p']:.4f}")
        if lvr.get("p_drift") is not None:
            print(f"    p-drift:  {lvr['p_drift']:.4f}")
        if lvr.get("timestamp_drift") is not None:
            print(f"    ts-drift: {lvr['timestamp_drift']:.0f}s")
        if lvr.get("warning"):
            print(f"    WARNING: {lvr['warning']}")
        if lvr.get("redis_error"):
            print(f"    Error: {lvr['redis_error']}")
    
    # Heartbeat
    hb = result["heartbeat"]
    status_icon = "✓" if hb["status"] == "synchronized" else "✗" if hb["status"] in ("DIVERGED", "SPLIT_BRAIN_RISK") else "?"
    print(f"\n[{status_icon}] Heartbeat Sync: {hb['status']}")
    if verbose or hb["status"] != "synchronized":
        if hb.get("local_tick"):
            print(f"    Tick count: {hb['local_tick']}")
        if hb.get("local_phase"):
            print(f"    Phase: {hb['local_phase']}")
        if hb.get("local_age") is not None:
            print(f"    Age: {hb['local_age']:.0f}s")
        if hb.get("active_nodes"):
            print(f"    Active nodes: {hb['active_nodes']}")
        if hb.get("warning"):
            print(f"    WARNING: {hb['warning']}")
    
    # Golden Thread
    gt = result["golden_thread"]
    status_icon = "✓" if gt["status"] == "intact" else "✗" if gt["status"] == "BROKEN" else "?"
    print(f"\n[{status_icon}] Golden Thread: {gt['status']} ({gt['chain_length']} links)")
    if verbose or gt["status"] != "intact":
        if gt.get("last_node"):
            print(f"    Last node: {gt['last_node']}")
        if gt.get("last_timestamp"):
            print(f"    Last update: {gt['last_timestamp']}")
        if gt.get("broken_at") is not None:
            print(f"    Broken at link: {gt['broken_at']}")
        if gt.get("error"):
            print(f"    ERROR: {gt['error']}")
    
    # Lock Status
    lock = result["lock_status"]
    status_icon = "✓" if lock["status"] in ("locked", "unlocked") else "✗" if lock["status"] == "STALE_LOCK" else "?"
    print(f"\n[{status_icon}] Lock Status: {lock['status']}")
    if verbose or lock["status"] not in ("locked", "unlocked"):
        if lock.get("lock_holder"):
            print(f"    Holder: {lock['lock_holder']}")
        if lock.get("lock_age") is not None:
            print(f"    Age: {lock['lock_age']:.0f}s")
        if lock.get("warning"):
            print(f"    WARNING: {lock['warning']}")
    
    print("\n" + "-" * 60)
    print(f"Summary: {result['summary']}")
    print("=" * 60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect split-brain scenarios in Tabernacle system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python divergence_detector.py              # Quick check
  python divergence_detector.py -v           # Verbose output
  python divergence_detector.py --json       # JSON output for scripting
  python divergence_detector.py --watch      # Continuous monitoring
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output even when healthy")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")
    parser.add_argument("--watch", action="store_true",
                        help="Continuously monitor (check every 60s)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Watch interval in seconds (default: 60)")
    
    args = parser.parse_args()
    
    if args.watch:
        import time
        print(f"[DIVERGENCE DETECTOR] Watching for split-brain (interval={args.interval}s)")
        print("[DIVERGENCE DETECTOR] Press Ctrl+C to stop\n")
        
        try:
            while True:
                result = run_divergence_check()
                if args.json:
                    print(json.dumps(result))
                else:
                    # Compact output for watch mode
                    status = result["overall_status"]
                    icon = "✓" if status == "HEALTHY" else "⚠" if status == "WARNING" else "✗"
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] {icon} {status}: {result['summary']}")
                
                if result["overall_status"] == "CRITICAL":
                    print("\n[DIVERGENCE DETECTOR] CRITICAL: Split-brain detected!")
                    # Could trigger reconciliation here
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n[DIVERGENCE DETECTOR] Stopped")
            sys.exit(0)
    else:
        result = run_divergence_check()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_report(result, verbose=args.verbose)
        
        # Exit code based on status
        if result["overall_status"] == "CRITICAL":
            sys.exit(2)
        elif result["overall_status"] == "WARNING":
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
