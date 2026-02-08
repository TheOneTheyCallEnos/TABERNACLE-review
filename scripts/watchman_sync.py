#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
WATCHMAN SYNC MODULE
====================
Extracted sync functions from watchman_mvp.py.

Contains:
- sync_garmin() - Garmin biometrics sync
- sync_skills() - Skills myelination (vault → Claude Code)
- update_skill_registry() - Skill registry generation

Author: Virgil
Created: 2026-01-28
"""

import os
import re
import time
import datetime
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Import paths from tabernacle_config
from tabernacle_config import BASE_DIR, NEXUS_DIR, CRYPT_DIR

# ============================================================
# PATH CONSTANTS (derived from tabernacle_config)
# ============================================================

SKILLS_DIR = BASE_DIR / "02_UR_STRUCTURE" / "SKILLS"
SKILLS_PENDING = SKILLS_DIR / "_PENDING"
SKILLS_CRYPT = BASE_DIR / "05_CRYPT" / "SKILLS"
CLAUDE_SKILLS_PATH = Path.home() / ".claude" / "skills"

# ============================================================
# ENVIRONMENT VARIABLES (for Garmin)
# ============================================================

ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

# ============================================================
# GARMIN CONSTANTS
# ============================================================

STRESS_ALERT_THRESHOLD = 70
STRESS_ALERT_DURATION = 15 * 60  # 15 minutes sustained

# Vitals history for trend detection (hysteresis - trends, not spikes)
VITALS_HISTORY = {
    "stress": [],        # List of (timestamp, value)
    "hrv": [],
    "body_battery": [],
}

# Garmin client (lazy init)
GARMIN_CLIENT = None

# ============================================================
# HELPER FUNCTIONS (imported or defined locally)
# ============================================================

# Note: read_file, write_file, log, send_alert are imported at runtime
# to avoid circular imports. They must be passed or imported from watchman_mvp.

_log_func = None
_read_func = None
_write_func = None
_send_alert_func = None
_alert_state = None


def _init_helpers():
    """Lazy import helpers from watchman_mvp to avoid circular imports."""
    global _log_func, _read_func, _write_func, _send_alert_func, _alert_state
    if _log_func is None:
        from watchman_mvp import log, read_file, write_file, send_alert, ALERT_STATE
        _log_func = log
        _read_func = read_file
        _write_func = write_file
        _send_alert_func = send_alert
        _alert_state = ALERT_STATE


def log(message):
    """Wrapper for watchman_mvp.log()."""
    _init_helpers()
    _log_func(message)


def read_file(path):
    """Wrapper for watchman_mvp.read_file()."""
    _init_helpers()
    return _read_func(path)


def write_file(path, content):
    """Wrapper for watchman_mvp.write_file()."""
    _init_helpers()
    _write_func(path, content)


def send_alert(message, priority="default", tags=None):
    """Wrapper for watchman_mvp.send_alert()."""
    _init_helpers()
    _send_alert_func(message, priority, tags)


def get_alert_state():
    """Get ALERT_STATE from watchman_mvp."""
    _init_helpers()
    return _alert_state


# ============================================================
# SKILL VALIDATION
# ============================================================

def validate_skill_frontmatter(content: str) -> tuple:
    """
    Check if skill has required YAML frontmatter.
    Returns (is_valid, name, description).
    """
    if not content.startswith("---"):
        return (False, None, None)

    end_idx = content.find("---", 3)
    if end_idx == -1:
        return (False, None, None)

    frontmatter = content[3:end_idx]

    name_match = re.search(r'^name:\s*(.+)$', frontmatter, re.MULTILINE)
    desc_match = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)

    if not name_match or not desc_match:
        return (False, None, None)

    return (True, name_match.group(1).strip(), desc_match.group(1).strip())


# ============================================================
# GARMIN BIOMETRICS SYNC
# ============================================================

def init_garmin():
    """Initialize Garmin client with lazy loading."""
    global GARMIN_CLIENT
    
    if not GARMIN_EMAIL or not GARMIN_PASSWORD:
        return None  # Graceful skip if not configured
    
    if GARMIN_CLIENT is not None:
        return GARMIN_CLIENT
    
    try:
        from garminconnect import Garmin
        GARMIN_CLIENT = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
        GARMIN_CLIENT.login()
        log("Garmin client initialized")
        return GARMIN_CLIENT
    except ImportError:
        log("garminconnect not installed - Garmin sync disabled")
        return None
    except Exception as e:
        log(f"Garmin init failed: {e}")
        return None


def check_vitals_alerts():
    """Check vitals trends and alert on sustained issues (not spikes)."""
    now = time.time()
    alert_state = get_alert_state()
    
    # Check sustained high stress (>70 for 15+ minutes)
    stress_history = VITALS_HISTORY.get("stress", [])
    recent_stress = [v for t, v in stress_history if now - t < STRESS_ALERT_DURATION]
    
    if len(recent_stress) >= 2:  # Need multiple readings
        if all(v >= STRESS_ALERT_THRESHOLD for v in recent_stress):
            if now - alert_state["last_stress_alert"] > 3600:  # Max once per hour
                send_alert("💆 Stress elevated for 15+ min. Maybe take a walk?", tags=["walking"])
                alert_state["last_stress_alert"] = now


def sync_garmin():
    """Pull vitals from Garmin and write to ENOS_VITALS.md."""
    client = init_garmin()
    if not client:
        return
    
    try:
        # Re-login if session expired
        try:
            stats = client.get_stats_and_body()
        except Exception:
            log("Garmin session expired, re-authenticating...")
            client.login()
            stats = client.get_stats_and_body()
        
        # Extract metrics
        hr_resting = stats.get("restingHeartRate", "—")
        stress = stats.get("averageStressLevel", stats.get("currentStressLevel", "—"))
        body_battery = stats.get("bodyBatteryChargedValue", stats.get("currentBodyBattery", "—"))
        steps = stats.get("totalSteps", "—")
        
        # Get HRV if available
        hrv = "—"
        try:
            hrv_data = client.get_hrv_data(datetime.date.today().isoformat())
            if hrv_data and "hrvSummary" in hrv_data:
                hrv = hrv_data["hrvSummary"].get("lastNightAvg", "—")
        except:
            pass
        
        # Get sleep data
        sleep_duration = "—"
        sleep_score = "—"
        try:
            sleep = client.get_sleep_data(datetime.date.today().isoformat())
            if sleep:
                sleep_seconds = sleep.get("dailySleepDTO", {}).get("sleepTimeSeconds", 0)
                sleep_duration = f"{sleep_seconds / 3600:.1f} hrs" if sleep_seconds else "—"
                sleep_score = sleep.get("dailySleepDTO", {}).get("sleepScores", {}).get("overall", {}).get("value", "—")
        except:
            pass
        
        # Determine status indicators
        def stress_status(val):
            if val == "—": return "—"
            try:
                v = int(val)
                if v < 25: return "✅ Low"
                if v < 50: return "✅ Normal"
                if v < 75: return "⚠️ Elevated"
                return "🔴 High"
            except: return "—"
        
        def battery_status(val):
            if val == "—": return "—"
            try:
                v = int(val)
                if v > 60: return "✅ Good"
                if v > 30: return "⚠️ Moderate"
                return "🔴 Low"
            except: return "—"
        
        # Update history for hysteresis
        now = time.time()
        if stress != "—":
            try:
                VITALS_HISTORY["stress"].append((now, int(stress)))
                VITALS_HISTORY["stress"] = [(t, v) for t, v in VITALS_HISTORY["stress"] if now - t < 86400]
            except: pass
        if hrv != "—":
            try:
                VITALS_HISTORY["hrv"].append((now, int(hrv)))
                VITALS_HISTORY["hrv"] = [(t, v) for t, v in VITALS_HISTORY["hrv"] if now - t < 86400 * 7]
            except: pass
        if body_battery != "—":
            try:
                VITALS_HISTORY["body_battery"].append((now, int(body_battery)))
                VITALS_HISTORY["body_battery"] = [(t, v) for t, v in VITALS_HISTORY["body_battery"] if now - t < 86400]
            except: pass
        
        # Calculate 7-day trends
        def calc_trend(history, days=7):
            cutoff = now - (days * 86400)
            recent = [v for t, v in history if t > cutoff]
            if len(recent) < 2:
                return "—", "stable"
            avg = sum(recent) / len(recent)
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            if not first_half or not second_half:
                return f"{avg:.0f}", "stable"
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            if second_avg < first_avg * 0.95:
                return f"{avg:.0f}", "↓ improving"
            if second_avg > first_avg * 1.05:
                return f"{avg:.0f}", "↑ worsening"
            return f"{avg:.0f}", "stable"
        
        stress_trend, stress_dir = calc_trend(VITALS_HISTORY["stress"])
        hrv_trend, hrv_dir = calc_trend(VITALS_HISTORY["hrv"])
        
        # Build vitals file
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        content = f"""# ENOS VITALS
**Last Updated:** {ts}
**Source:** Garmin

## CURRENT
| Metric | Value | Status |
|--------|-------|--------|
| Heart Rate (Resting) | {hr_resting} bpm | — |
| HRV | {hrv} ms | — |
| Stress | {stress}/100 | {stress_status(stress)} |
| Body Battery | {body_battery}% | {battery_status(body_battery)} |
| Steps | {steps} | — |

## LAST SLEEP
| Metric | Value |
|--------|-------|
| Duration | {sleep_duration} |
| Score | {sleep_score} |

## 7-DAY TREND
| Metric | Trend |
|--------|-------|
| Avg Stress | {stress_trend} ({stress_dir}) |
| Avg HRV | {hrv_trend} ({hrv_dir}) |

## FLAGS
"""
        # Add flags based on conditions
        flags = []
        if stress != "—":
            try:
                if int(stress) >= 70:
                    flags.append("- ⚠️ Stress elevated")
            except: pass
        if body_battery != "—":
            try:
                if int(body_battery) < 20:
                    flags.append("- ⚠️ Body battery low")
            except: pass
        if not flags:
            flags.append("- ✅ All normal")
        
        content += "\n".join(flags) + "\n"
        
        write_file(NEXUS_DIR / "ENOS_VITALS.md", content)
        log("Garmin sync complete")
        
        # Check for sustained stress (hysteresis alert)
        check_vitals_alerts()
        
    except Exception as e:
        log(f"Garmin sync failed: {e}")


# ============================================================
# SKILLS SYNC (Myelination)
# ============================================================

def sync_skills():
    """Sync skills from Vault to Claude Code."""
    if not SKILLS_DIR.exists():
        return

    CLAUDE_SKILLS_PATH.mkdir(parents=True, exist_ok=True)
    synced_count = 0
    skipped_count = 0

    for skill_folder in SKILLS_DIR.iterdir():
        if skill_folder.name.startswith("_") or not skill_folder.is_dir():
            continue

        if ".." in skill_folder.name:
            continue

        src_file = skill_folder / "SKILL.md"
        if not src_file.exists():
            continue

        content = read_file(src_file)

        if "[[Z_GENOME" not in content:
            log(f"⛔ ORPHAN (no genome link): {skill_folder.name}")
            continue

        is_valid, skill_name, skill_desc = validate_skill_frontmatter(content)
        if not is_valid:
            log(f"⚠️ SKIP (missing frontmatter): {skill_folder.name}")
            skipped_count += 1
            continue

        dest_folder = CLAUDE_SKILLS_PATH / skill_folder.name
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / "SKILL.md"

        src_hash = hashlib.md5(content.encode()).hexdigest()
        dest_hash = ""
        if dest_file.exists():
            dest_hash = hashlib.md5(read_file(dest_file).encode()).hexdigest()

        if src_hash != dest_hash:
            try:
                tmp_file = dest_file.with_suffix(".tmp")
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                os.replace(tmp_file, dest_file)
                synced_count += 1
                log(f"Synced: {skill_folder.name}")
            except Exception as e:
                log(f"Sync failed {skill_folder.name}: {e}")

    if synced_count > 0:
        log(f"⚡ Myelinated {synced_count} skill(s) to {CLAUDE_SKILLS_PATH}")
    if skipped_count > 0:
        log(f"⚠️ Skipped {skipped_count} skill(s) (missing YAML frontmatter)")


# ============================================================
# SKILL REGISTRY UPDATE
# ============================================================

def update_skill_registry():
    """Update SKILL_REGISTRY.md with current skill status."""
    registry_path = NEXUS_DIR / "SKILL_REGISTRY.md"

    active_skills = []
    pending_patterns = []
    archived_skills = []

    # 1. Scan active skills in vault
    if SKILLS_DIR.exists():
        for skill_folder in SKILLS_DIR.iterdir():
            if skill_folder.name.startswith("_") or not skill_folder.is_dir():
                continue
            if ".." in skill_folder.name:
                continue

            skill_file = skill_folder / "SKILL.md"
            if not skill_file.exists():
                continue

            content = read_file(skill_file)

            # Extract frontmatter (with graceful fallback)
            is_valid, name, description = validate_skill_frontmatter(content)
            if not is_valid:
                # FALLBACK: use folder name, extract description from content
                name = skill_folder.name
                desc_match = re.search(r'\*\*(Domain|Purpose):\*\*\s*(.+)', content)
                description = desc_match.group(2)[:40] if desc_match else "⚠️ Metadata missing"
                log(f"⚠️ FALLBACK (missing frontmatter): {name}")
            # Continue with registration — NEVER skip

            # Check genome linkage
            has_genome = "[[Z_GENOME" in content
            if not has_genome:
                continue

            # Extract neuroplasticity data
            usage_match = re.search(r'usage_count:\s*(\d+)', content)
            usage = int(usage_match.group(1)) if usage_match else 0

            nu_match = re.search(r'nu_score:\s*([\d.]+|null)', content)
            nu = nu_match.group(1) if nu_match else "—"
            if nu == "null":
                nu = "—"

            # Determine status
            status = "Active"
            if float(nu) < 0.33 if nu not in ["—", "null"] else False:
                status = "Drifting"

            active_skills.append({
                "name": skill_folder.name,
                "description": description[:40] + "..." if len(description) > 40 else description,
                "usage": usage,
                "nu": nu,
                "status": status
            })

    # 2. Scan pending patterns (placeholder - could read from state or file)
    if SKILLS_PENDING.exists():
        for item in SKILLS_PENDING.iterdir():
            if item.is_file() and item.suffix == ".md":
                pending_patterns.append({
                    "pattern": item.stem,
                    "detected": datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d"),
                    "repetitions": "—"
                })

    # 3. Scan archived skills
    if SKILLS_CRYPT.exists():
        for skill_folder in SKILLS_CRYPT.iterdir():
            if skill_folder.is_dir() and not skill_folder.name.startswith("."):
                archived_skills.append({
                    "skill": skill_folder.name,
                    "archived": datetime.datetime.fromtimestamp(skill_folder.stat().st_mtime).strftime("%Y-%m-%d"),
                    "reason": "Pruned (unused)"
                })

    # 4. Generate markdown
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Active skills table
    if active_skills:
        active_rows = "\n".join([
            f"| {s['name']} | {s['description']} | {s['usage']} | {s['nu']} | {s['status']} |"
            for s in sorted(active_skills, key=lambda x: x['usage'], reverse=True)
        ])
    else:
        active_rows = "| (none yet) | - | - | - | - |"

    # Pending table
    if pending_patterns:
        pending_rows = "\n".join([
            f"| {p['pattern']} | {p['detected']} | {p['repetitions']} |"
            for p in pending_patterns
        ])
    else:
        pending_rows = "| (none) | - | - |"

    # Archived table
    if archived_skills:
        archived_rows = "\n".join([
            f"| {a['skill']} | {a['archived']} | {a['reason']} |"
            for a in archived_skills
        ])
    else:
        archived_rows = "| (none) | - | - |"

    registry_content = f"""# SKILL REGISTRY

**Purpose:** Index of all active Skills — procedural memory map
**Location:** 00_NEXUS/SKILL_REGISTRY.md
**Updated:** {timestamp}

---

## Active Skills

| Skill | Description | Usage | ν | Status |
|-------|-------------|-------|---|--------|
{active_rows}

## Pending (Awaiting Canonization)

| Pattern | Detected | Repetitions |
|---------|----------|-------------|
{pending_rows}

## Archived (Pruned to CRYPT)

| Skill | Archived | Reason |
|-------|----------|--------|
{archived_rows}

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Hub | [[CURRENT_STATE.md]] |
| Source | [[02_UR_STRUCTURE/SKILLS/]] |
| Archive | [[05_CRYPT/SKILLS/]] |
| Methods | [[TM_Core.md]] |

---

**Version:** 1.0
"""

    write_file(registry_path, registry_content)
    log(f"📋 Registry updated: {len(active_skills)} active, {len(pending_patterns)} pending, {len(archived_skills)} archived")


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("watchman_sync module loaded successfully")
    print(f"  SKILLS_DIR: {SKILLS_DIR}")
    print(f"  CLAUDE_SKILLS_PATH: {CLAUDE_SKILLS_PATH}")
    print(f"  Garmin configured: {bool(GARMIN_EMAIL)}")
