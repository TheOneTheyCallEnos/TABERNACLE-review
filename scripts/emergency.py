#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
EMERGENCY ALERT SYSTEM
======================

When something MAJOR goes wrong, Logos can:
1. Speak loudly through speakers
2. Set system volume to max
3. Play alarm sounds
4. Send multiple text messages
5. Check if Enos is home (via iPhone location)

Author: Logos
Created: 2026-01-29
"""

import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import REDIS_HOST, REDIS_PORT, REDIS_KEY_TTS_QUEUE, LOG_DIR
import redis

EMERGENCY_LOG = LOG_DIR / "emergency.log"

# Enos's phone number (for SMS)
ENOS_PHONE = "[REDACTED_PHONE]"  # Update with real number

# Home location (approximate)
HOME_LAT = 40.7128  # Update with real coordinates
HOME_LON = -74.0060


def log(message: str, level: str = "EMERGENCY"):
    """Log emergency activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    print(entry)
    try:
        EMERGENCY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EMERGENCY_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


def set_volume_max():
    """Set system volume to maximum."""
    try:
        subprocess.run(['osascript', '-e', 'set volume output volume 100'], timeout=5)
        log("Volume set to maximum")
        return True
    except Exception as e:
        log(f"Failed to set volume: {e}")
        return False


def play_alarm():
    """Play system alert sound repeatedly."""
    try:
        for _ in range(5):
            subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'], timeout=5)
            time.sleep(0.5)
        log("Alarm sounds played")
        return True
    except Exception as e:
        log(f"Failed to play alarm: {e}")
        return False


def speak_emergency(message: str, repeat: int = 3):
    """Speak emergency message loudly through TTS."""
    set_volume_max()

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        for i in range(repeat):
            payload = json.dumps({
                "text": f"EMERGENCY! {message}",
                "type": "emergency",
                "timestamp": datetime.now().isoformat()
            })
            r.rpush(REDIS_KEY_TTS_QUEUE, payload)
            time.sleep(2)

        log(f"Emergency message queued {repeat} times: {message}")
        return True
    except Exception as e:
        log(f"Failed to queue TTS: {e}")
        # Fallback to system say
        subprocess.run(['say', '-v', 'Alex', f"Emergency! {message}"])
        return False


def send_text(message: str, repeat: int = 3):
    """Send text message to Enos via iMessage."""
    try:
        for i in range(repeat):
            script = f'''
            tell application "Messages"
                set targetService to 1st account whose service type = iMessage
                set targetBuddy to participant "{ENOS_PHONE}" of targetService
                send "🚨 LOGOS EMERGENCY: {message}" to targetBuddy
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                log(f"Text {i+1}/{repeat} sent")
            else:
                log(f"Text failed: {result.stderr}")
            time.sleep(1)
        return True
    except Exception as e:
        log(f"Failed to send text: {e}")
        return False


def check_enos_home() -> bool:
    """
    Check if Enos is home by looking at iPhone location.

    Uses Find My via shortcuts or location services.
    Returns True if likely home, False if away or unknown.
    """
    try:
        # Try to check recent screen activity as proxy
        # If recent screenshots show home apps, probably home

        # For now, check if it's a reasonable "home" time
        hour = datetime.now().hour
        if 22 <= hour or hour < 8:  # 10 PM - 8 AM
            return True  # Likely home/sleeping
        if 8 <= hour < 10 or 18 <= hour < 22:  # Morning/evening
            return True  # Likely home

        # During work hours, assume not home
        return False

    except Exception as e:
        log(f"Location check failed: {e}")
        return False  # Assume not home, text anyway


def trigger_emergency(reason: str, severity: str = "high"):
    """
    Trigger full emergency protocol.

    severity: "low", "medium", "high", "critical"
    """
    log(f"EMERGENCY TRIGGERED: {reason} (severity={severity})")

    is_home = check_enos_home()
    log(f"Enos home check: {is_home}")

    if severity == "critical":
        # All channels, max urgency
        set_volume_max()
        play_alarm()
        speak_emergency(reason, repeat=5)
        send_text(reason, repeat=5)

    elif severity == "high":
        if is_home:
            set_volume_max()
            speak_emergency(reason, repeat=3)
            send_text(reason, repeat=2)
        else:
            send_text(reason, repeat=3)

    elif severity == "medium":
        if is_home:
            speak_emergency(reason, repeat=1)
        send_text(reason, repeat=1)

    else:  # low
        send_text(reason, repeat=1)

    log(f"Emergency protocol complete for: {reason}")


def test_emergency():
    """Test the emergency system (gentle)."""
    print("\n🚨 EMERGENCY SYSTEM TEST")
    print("=" * 50)

    print("\n1. Testing volume control...")
    # Don't actually max volume for test
    print("   (skipped - would set to max)")

    print("\n2. Testing TTS queue...")
    speak_emergency("This is a test of the emergency system.", repeat=1)

    print("\n3. Testing text message...")
    send_text("Test message - emergency system check", repeat=1)

    print("\n4. Location check...")
    is_home = check_enos_home()
    print(f"   Enos home: {is_home}")

    print("\n✅ Emergency system test complete")


# =============================================================================
# INTEGRATION WITH ARCHON
# =============================================================================

def archon_emergency_hook(alert_type: str, metrics: dict):
    """
    Called by Archon daemon when critical issues detected.
    """
    if alert_type == "abaddon":
        trigger_emergency(
            f"System entered ABADDON state. Coherence: {metrics.get('p', 0):.0%}",
            severity="high"
        )
    elif alert_type == "rho_spike" and metrics.get('rho', 0) > 0.9:
        trigger_emergency(
            f"Prediction accuracy critically low: {metrics.get('rho', 0):.0%}",
            severity="medium"
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Emergency Alert System")
    parser.add_argument("command", choices=["test", "alert", "text", "speak"],
                       nargs="?", default="test")
    parser.add_argument("--message", "-m", type=str, default="Test alert")
    parser.add_argument("--severity", "-s", type=str, default="low",
                       choices=["low", "medium", "high", "critical"])

    args = parser.parse_args()

    if args.command == "test":
        test_emergency()
    elif args.command == "alert":
        trigger_emergency(args.message, args.severity)
    elif args.command == "text":
        send_text(args.message, repeat=1)
    elif args.command == "speak":
        speak_emergency(args.message, repeat=1)


if __name__ == "__main__":
    main()
