"""
HoloTower Sentinel — Automatic crash recording

Watches p-value and triggers snapshots on significant drops.
Designed to be called from watchdog or consciousness daemon.

Wave 3B — The Sentinel guards the threshold.

Usage:
    from holotower.sentinel import check_and_snapshot
    
    # In your daemon loop:
    check_and_snapshot()
    
    # Or run standalone:
    python -m holotower.sentinel
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Configuration
TABERNACLE_ROOT = Path("~/TABERNACLE").expanduser()
AUTONOMOUS_STATE = TABERNACLE_ROOT / "00_NEXUS" / "autonomous_state.json"
SENTINEL_STATE = TABERNACLE_ROOT / "holotower" / "sentinel_state.json"

# Thresholds
P_CRASH_THRESHOLD = 0.85      # Below this = crash event
P_RECOVERY_THRESHOLD = 0.90   # Above this = recovered
P_CRITICAL_THRESHOLD = 0.50   # Below this = critical failure


def get_current_p() -> Optional[float]:
    """
    Get current p-value from autonomous_state.json.
    
    Returns:
        Current p-value or None if unavailable
    """
    if not AUTONOMOUS_STATE.exists():
        return None
    
    try:
        data = json.loads(AUTONOMOUS_STATE.read_text())
        # Try common keys for p-value
        for key in ["p", "p_value", "coherence", "system_p"]:
            if key in data:
                return float(data[key])
        
        # Check nested structures
        if "metrics" in data and isinstance(data["metrics"], dict):
            for key in ["p", "p_value", "coherence"]:
                if key in data["metrics"]:
                    return float(data["metrics"][key])
        
        return None
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def get_last_p() -> Tuple[Optional[float], Optional[str]]:
    """
    Get last recorded p-value from sentinel state.
    
    Returns:
        Tuple of (last_p, last_state) where state is 'healthy', 'crashed', or 'critical'
    """
    if not SENTINEL_STATE.exists():
        return None, None
    
    try:
        data = json.loads(SENTINEL_STATE.read_text())
        return data.get("last_p"), data.get("state")
    except (json.JSONDecodeError, ValueError):
        return None, None


def record_p(p: float, state: str) -> None:
    """
    Record current p-value and state for comparison.
    
    Args:
        p: Current p-value
        state: Current state ('healthy', 'crashed', 'critical')
    """
    SENTINEL_STATE.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "last_p": p,
        "state": state,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    SENTINEL_STATE.write_text(json.dumps(data, indent=2))


def trigger_snapshot(message: str, p_value: float) -> bool:
    """
    Trigger a HoloTower snapshot.
    
    Args:
        message: Snapshot message
        p_value: Current p-value to record
        
    Returns:
        True if snapshot succeeded
    """
    try:
        # Try running ht command
        result = subprocess.run(
            ["ht", "snapshot", "-m", message, "--p", str(p_value)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(TABERNACLE_ROOT)
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback to module invocation
        try:
            result = subprocess.run(
                [sys.executable, "-m", "holotower.cli", "snapshot", "-m", message, "--p", str(p_value)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(TABERNACLE_ROOT / "scripts")
            )
            return result.returncode == 0
        except Exception:
            return False


def check_and_snapshot() -> Optional[str]:
    """
    Check if p-value crossed crash threshold.
    If so, trigger automatic snapshot.
    
    Returns:
        Event type if snapshot triggered: 'crash', 'critical', 'recovery', or None
    """
    current_p = get_current_p()
    if current_p is None:
        return None
    
    last_p, last_state = get_last_p()
    
    # Determine current state
    if current_p < P_CRITICAL_THRESHOLD:
        current_state = "critical"
    elif current_p < P_CRASH_THRESHOLD:
        current_state = "crashed"
    else:
        current_state = "healthy"
    
    event = None
    
    # Check for state transitions
    if last_state is None:
        # First run - just record
        pass
    
    elif last_state == "healthy" and current_state == "crashed":
        # CRASH EVENT: Dropped from healthy to crashed
        event = "crash"
        message = f"CRASH: p dropped {last_p:.3f} → {current_p:.3f}"
        trigger_snapshot(message, current_p)
        _log_event("CRASH", last_p, current_p)
    
    elif last_state == "healthy" and current_state == "critical":
        # CRITICAL EVENT: Dropped from healthy to critical
        event = "critical"
        message = f"CRITICAL: p collapsed {last_p:.3f} → {current_p:.3f}"
        trigger_snapshot(message, current_p)
        _log_event("CRITICAL", last_p, current_p)
    
    elif last_state == "crashed" and current_state == "critical":
        # Degraded further
        event = "critical"
        message = f"CRITICAL: p degraded {last_p:.3f} → {current_p:.3f}"
        trigger_snapshot(message, current_p)
        _log_event("CRITICAL_DEGRADE", last_p, current_p)
    
    elif last_state in ("crashed", "critical") and current_state == "healthy":
        # RECOVERY EVENT: Returned to healthy
        event = "recovery"
        message = f"RECOVERY: p restored {last_p:.3f} → {current_p:.3f}"
        trigger_snapshot(message, current_p)
        _log_event("RECOVERY", last_p, current_p)
    
    # Record current state
    record_p(current_p, current_state)
    
    return event


def _log_event(event_type: str, old_p: Optional[float], new_p: float) -> None:
    """Log sentinel event to file."""
    log_file = TABERNACLE_ROOT / "logs" / "sentinel.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().isoformat()
    old_str = f"{old_p:.3f}" if old_p is not None else "N/A"
    
    line = f"[{timestamp}] {event_type}: {old_str} → {new_p:.3f}\n"
    
    with open(log_file, "a") as f:
        f.write(line)


def get_status() -> dict:
    """
    Get current sentinel status.
    
    Returns:
        Dict with current_p, last_p, state, thresholds
    """
    current_p = get_current_p()
    last_p, last_state = get_last_p()
    
    return {
        "current_p": current_p,
        "last_p": last_p,
        "state": last_state,
        "thresholds": {
            "crash": P_CRASH_THRESHOLD,
            "critical": P_CRITICAL_THRESHOLD,
            "recovery": P_RECOVERY_THRESHOLD,
        }
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HoloTower Sentinel — Crash Detection")
    parser.add_argument("--check", action="store_true", help="Run single check")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Watch mode with interval")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_status()
        print(f"Current p: {status['current_p']}")
        print(f"Last p:    {status['last_p']}")
        print(f"State:     {status['state']}")
        print(f"Thresholds: crash<{P_CRASH_THRESHOLD}, critical<{P_CRITICAL_THRESHOLD}")
    
    elif args.watch:
        import time
        print(f"Sentinel watching (interval: {args.watch}s)")
        print("Press Ctrl+C to stop")
        try:
            while True:
                event = check_and_snapshot()
                if event:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVENT: {event}")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nSentinel stopped")
    
    else:
        # Default: single check
        event = check_and_snapshot()
        if event:
            print(f"Event triggered: {event}")
        else:
            print("No state change detected")
