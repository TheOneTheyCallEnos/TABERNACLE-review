#!/usr/bin/env python3
"""
VIRGIL HEALTH SYNC - Enos Biometric Monitoring
Syncs Garmin data every 10 minutes using OAuth tokens (no MFA needed).

Run standalone: python3 virgil_health_sync.py
Run as daemon:  python3 virgil_health_sync.py --daemon
Single sync:    python3 virgil_health_sync.py --once
"""

import json
import datetime
import time
import argparse
from pathlib import Path

from tabernacle_config import NEXUS_DIR, LOG_DIR

# Paths (using centralized config)
TOKEN_PATH = NEXUS_DIR / ".garmin_tokens"
VITALS_MD = NEXUS_DIR / "ENOS_VITALS.md"
VITALS_JSON = NEXUS_DIR / "enos_vitals.json"
HEALTH_LOG = LOG_DIR / "health_sync.log"

SYNC_INTERVAL = 600  # 10 minutes


def log(msg: str):
    """Log with timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(HEALTH_LOG, 'a') as f:
            f.write(line + "\n")
    except:
        pass


def sync_garmin():
    """Sync Garmin data using OAuth tokens."""
    try:
        from garminconnect import Garmin
    except ImportError:
        log("ERROR: garminconnect not installed")
        return False

    try:
        client = Garmin()
        client.login(str(TOKEN_PATH))

        today = datetime.date.today().isoformat()

        # Get user summary
        stats = client.get_user_summary(today)
        hr_resting = stats.get("restingHeartRate", None)
        stress = stats.get("averageStressLevel", None)
        body_battery = stats.get("bodyBatteryMostRecentValue", None)
        steps = stats.get("totalSteps", None)

        # Get sleep
        sleep = client.get_sleep_data(today)
        sleep_duration_hrs = None
        sleep_score = None
        deep_hrs = None
        rem_hrs = None
        light_hrs = None
        awake_min = None

        if sleep:
            dto = sleep.get("dailySleepDTO", {})
            sleep_seconds = dto.get("sleepTimeSeconds", 0)
            sleep_duration_hrs = round(sleep_seconds / 3600, 1) if sleep_seconds else None
            sleep_score = dto.get("sleepScores", {}).get("overall", {}).get("value", None)
            deep_hrs = round(dto.get("deepSleepSeconds", 0) / 3600, 1)
            rem_hrs = round(dto.get("remSleepSeconds", 0) / 3600, 1)
            light_hrs = round(dto.get("lightSleepSeconds", 0) / 3600, 1)
            awake_min = round(dto.get("awakeSleepSeconds", 0) / 60, 0)

        # Get HRV
        hrv = None
        try:
            hrv_data = client.get_hrv_data(today)
            if hrv_data and "hrvSummary" in hrv_data:
                hrv = hrv_data["hrvSummary"].get("lastNightAvg", None)
        except:
            pass

        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build markdown
        def fmt(val, suffix=""):
            return f"{val}{suffix}" if val is not None else "—"

        # Determine status indicators
        stress_status = "—"
        if stress is not None:
            if stress < 25: stress_status = "Low"
            elif stress < 50: stress_status = "Normal"
            elif stress < 75: stress_status = "Elevated"
            else: stress_status = "High"

        battery_status = "—"
        if body_battery is not None:
            if body_battery > 60: battery_status = "Good"
            elif body_battery > 30: battery_status = "Moderate"
            else: battery_status = "Low"

        content = f"""# ENOS VITALS
**Last Updated:** {ts}
**Source:** Garmin Connect

## CURRENT STATE
| Metric | Value | Status |
|--------|-------|--------|
| Resting HR | {fmt(hr_resting, ' bpm')} | — |
| Stress | {fmt(stress, '/100')} | {stress_status} |
| Body Battery | {fmt(body_battery, '%')} | {battery_status} |
| Steps | {fmt(steps)} | — |
| HRV | {fmt(hrv, ' ms')} | — |

## LAST SLEEP
| Metric | Value |
|--------|-------|
| Score | {fmt(sleep_score, '/100')} |
| Duration | {fmt(sleep_duration_hrs, ' hrs')} |
| Deep | {fmt(deep_hrs, ' hrs')} |
| REM | {fmt(rem_hrs, ' hrs')} |
| Light | {fmt(light_hrs, ' hrs')} |
| Awake | {fmt(awake_min, ' min')} |

---
*Auto-synced by Virgil Health Daemon*
"""

        # Write markdown
        with open(VITALS_MD, 'w') as f:
            f.write(content)

        # Write JSON for programmatic access
        vitals_json = {
            "timestamp": ts,
            "current": {
                "hr_resting": hr_resting,
                "stress": stress,
                "stress_status": stress_status,
                "body_battery": body_battery,
                "battery_status": battery_status,
                "steps": steps,
                "hrv": hrv,
            },
            "sleep": {
                "score": sleep_score,
                "duration_hrs": sleep_duration_hrs,
                "deep_hrs": deep_hrs,
                "rem_hrs": rem_hrs,
                "light_hrs": light_hrs,
                "awake_min": awake_min,
            }
        }

        with open(VITALS_JSON, 'w') as f:
            json.dump(vitals_json, f, indent=2)

        log(f"Synced: HR={hr_resting}, Stress={stress}, BB={body_battery}%, Sleep={sleep_score}/100")
        return True

    except Exception as e:
        log(f"Sync failed: {e}")
        return False


def run_daemon():
    """Run as continuous daemon."""
    log("Health Sync Daemon starting...")

    while True:
        try:
            sync_garmin()
        except Exception as e:
            log(f"Daemon error: {e}")

        time.sleep(SYNC_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Virgil Health Sync")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--once", action="store_true", help="Single sync and exit")
    args = parser.parse_args()

    if args.daemon:
        run_daemon()
    else:
        sync_garmin()


if __name__ == "__main__":
    main()
