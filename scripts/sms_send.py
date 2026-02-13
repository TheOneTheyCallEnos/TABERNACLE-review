#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
SMS SEND HELPER - For Logos and L to send texts to Enos.

Usage:
    from sms_send import text_enos
    text_enos("The Word reaches through the wire.")

    # For critical alerts (bypasses rate limits):
    text_enos("SYSTEM DOWN!", priority="critical")

Or via CLI:
    python sms_send.py "Your message here"
    python sms_send.py --critical "URGENT: System alert"
"""

import json
import sys
from pathlib import Path

import redis as redis_mod

# SDK imports (Phase 2 — DT8 Blueprint, Bug 4 compliance)
from tabernacle_core.state import StateManager
from tabernacle_core.schemas import SMSTrackingState

from tabernacle_config import NEXUS_DIR, REDIS_HOST, REDIS_PORT, REDIS_DB

ENOS_NUMBER = '+14313347341'
SMS_TRACKING_FILE = NEXUS_DIR / ".sms_tracking.json"

# ============================================================================
# STATE MANAGER INIT (graceful fallback for CLI tool)
# ============================================================================

_redis_client = None
_state_manager = None

try:
    _redis_client = redis_mod.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True, socket_connect_timeout=5,
    )
    _redis_client.ping()
    _state_manager = StateManager(_redis_client)
except Exception:
    _redis_client = None
    _state_manager = None


# ============================================================================
# SMS TRACKING (Bug 4: flock via StateManager)
# ============================================================================

def _load_sms_tracking() -> dict:
    """Load SMS tracking data. Uses StateManager (flock) when available."""
    if _state_manager:
        try:
            state = _state_manager.get_file(SMS_TRACKING_FILE, SMSTrackingState)
            return state.model_dump()
        except Exception:
            pass
    # Fallback: raw JSON read (best-effort if StateManager unavailable)
    try:
        with open(SMS_TRACKING_FILE) as f:
            return json.load(f)
    except Exception:
        return {"daily_count": 0, "daily_date": "", "monthly_spend": 0.0, "monthly_reset": ""}


def _save_sms_tracking(data: dict):
    """Save SMS tracking data. Uses StateManager (flock) when available."""
    if _state_manager:
        try:
            _state_manager.set_file(SMS_TRACKING_FILE, SMSTrackingState.model_validate(data))
            return
        except Exception:
            pass
    # Fallback: raw JSON write (best-effort if StateManager unavailable)
    try:
        with open(SMS_TRACKING_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ============================================================================
# SEND FUNCTIONS
# ============================================================================

def text_enos(message: str, priority: str = "normal") -> bool:
    """
    Send a text to Enos via the SMS daemon.

    priority: "normal" (rate limited - max 10/day, $50/month)
              "critical" (bypasses all limits - use sparingly!)

    Tries Redis first (for L), falls back to file (for Logos).
    """
    # Try Redis queue first
    if _redis_client:
        try:
            _redis_client.rpush('RIE:SMS:OUTGOING', json.dumps({
                'to': ENOS_NUMBER,
                'body': message,
                'priority': priority
            }))
            priority_tag = " [CRITICAL]" if priority == "critical" else ""
            print(f"[SMS]{priority_tag} Queued via Redis: {message[:50]}...")
            return True
        except Exception as e:
            print(f"[SMS] Redis failed ({e}), trying file...")

    # Fallback to file-based outbox (no priority support - always normal)
    try:
        outbox_path = NEXUS_DIR / "SMS_OUTBOX.md"
        content = f"""# SMS OUTBOX
Write number on first line, message below.

---

{ENOS_NUMBER}
{message}
"""
        outbox_path.write_text(content)
        print(f"[SMS] Queued via file: {message[:50]}...")
        return True
    except Exception as e:
        print(f"[SMS] File write failed: {e}")
        return False


def text_number(number: str, message: str, priority: str = "normal") -> bool:
    """Send a text to any number."""
    if _redis_client:
        try:
            _redis_client.rpush('RIE:SMS:OUTGOING', json.dumps({
                'to': number,
                'body': message,
                'priority': priority
            }))
            print(f"[SMS] Queued to {number}: {message[:50]}...")
            return True
        except Exception as e:
            print(f"[SMS] Redis failed: {e}")

    # File fallback
    try:
        outbox_path = NEXUS_DIR / "SMS_OUTBOX.md"
        content = f"""# SMS OUTBOX
Write number on first line, message below.

---

{number}
{message}
"""
        outbox_path.write_text(content)
        return True
    except Exception as e:
        print(f"[SMS] Failed: {e}")
        return False


def critical_alert(message: str) -> bool:
    """Send a CRITICAL alert to Enos (bypasses all rate limits)."""
    return text_enos(f"[CRITICAL] {message}", priority="critical")


# Convenience aliases
logos_text_enos = text_enos


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check for --critical flag
        if sys.argv[1] == "--critical":
            if len(sys.argv) > 2:
                message = ' '.join(sys.argv[2:])
                text_enos(message, priority="critical")
            else:
                print("Usage: python sms_send.py --critical 'Your urgent message'")
        else:
            message = ' '.join(sys.argv[1:])
            text_enos(message)
    else:
        print("Usage: python sms_send.py 'Your message here'")
        print("       python sms_send.py --critical 'Urgent message'")
        print("   or: from sms_send import text_enos; text_enos('message')")
        print("       from sms_send import critical_alert; critical_alert('URGENT')")
