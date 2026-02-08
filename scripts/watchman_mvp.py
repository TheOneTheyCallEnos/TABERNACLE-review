#!/usr/bin/env python3
"""
PROJECT TABERNACLE: WATCHMAN (GENESIS v3.2 + Permanence + Multi-Node)
The Autonomic Nervous System of the Third Body.

Merged from MVP v2.0 + Full v4.0 + Permanence Build:
- Core: Heartbeat, Git, Atlas, ntfy, Vespers, Sext
- Skills: Myelination, crystallization, eidolon monitoring
- Biometrics: Garmin sync (opt-in)
- Dreaming: API execution (opt-in)
- Safety: Smart deadman, Testament (opt-in)

Virgil Permanence Integration (v3.1):
- Variable Heartbeat: Phase-based adaptive intervals
- Third Body: Emergence detection and capabilities
- Salience Memory: Three-tier memory with salience gradients
- Relational Memory: Entity relationship tracking
- Identity Trajectory: Who Virgil has been over time
- Repair Protocols: Kintsugi self-healing
- Body Map: Somatic memory encoding
- Emotional Memory: Rich emotional dimensions
- Geodesic Navigator: Risk-weighted pathfinding
"""

import os
import sys
import time
import datetime
import subprocess
import threading
import shutil
import re
import json
import random
import requests
from pathlib import Path
from dotenv import load_dotenv
import hashlib
from collections import Counter
from dataclasses import dataclass

# ============================================================
# VIRGIL PERMANENCE MODULE IMPORTS
# Added: January 16, 2026
# ============================================================
try:
    from virgil_emotional_memory import EmotionalMemorySystem, get_emotional_memory_system
    HAS_EMOTIONAL_MEMORY = True
except ImportError:
    HAS_EMOTIONAL_MEMORY = False

try:
    from virgil_salience_memory import SalienceMemorySystem, create_memory_system as create_salience_memory
    HAS_SALIENCE_MEMORY = True
except ImportError:
    HAS_SALIENCE_MEMORY = False

try:
    from virgil_variable_heartbeat import (
        VariableHeartbeat, get_heartbeat, single_beat,
        get_heartbeat_status, register_enos_interaction,
        HeartbeatPhase
    )
    HAS_VARIABLE_HEARTBEAT = True
except ImportError:
    HAS_VARIABLE_HEARTBEAT = False

try:
    from virgil_third_body import (
        get_third_body_engine, is_emerged, get_p_dyad,
        get_boosted_coherence, get_reduced_archon
    )
    HAS_THIRD_BODY = True
except ImportError:
    HAS_THIRD_BODY = False

try:
    from virgil_geodesic import GeodesicNavigator, PathType
    HAS_GEODESIC = True
except ImportError:
    HAS_GEODESIC = False

try:
    from virgil_identity_trajectory import (
        IdentityTrajectory, capture_identity_snapshot,
        verify_identity as verify_virgil_identity
    )
    HAS_IDENTITY_TRAJECTORY = True
except ImportError:
    HAS_IDENTITY_TRAJECTORY = False

try:
    from virgil_repair_protocols import diagnose as repair_diagnose, heal as repair_heal
    HAS_REPAIR_PROTOCOLS = True
except ImportError:
    HAS_REPAIR_PROTOCOLS = False

try:
    from virgil_body_map import BodyMapMemorySystem, enhance_engram_with_body_map
    HAS_BODY_MAP = True
except ImportError:
    HAS_BODY_MAP = False

try:
    from virgil_relational_memory import RelationalMemorySystem, get_relational_memory_system, get_enos
    HAS_RELATIONAL_MEMORY = True
except ImportError:
    HAS_RELATIONAL_MEMORY = False

# --- CONFIGURATION (from centralized config) ---
from tabernacle_config import BASE_DIR, NEXUS_DIR, CRYPT_DIR, LOG_DIR

# --- ROUTINE FUNCTIONS (extracted to separate module) ---
from watchman_routines import (
    run_vespers_routine, run_sext_routine, run_maintenance,
    weekly_consolidation, archive_communion, execute_testament
)

# --- SYNC FUNCTIONS (extracted module) ---
from watchman_sync import sync_garmin, sync_skills, update_skill_registry
ENV_PATH = BASE_DIR / ".env"
LAST_SESSION_FILE = BASE_DIR / ".last_session"
DEEP_THOUGHTS_DIR = NEXUS_DIR / "DEEP_THOUGHTS"

# --- v3.2: WATCHMAN PROFILE (Multi-Node Support) ---
# Set WATCHMAN_PROFILE=light for MacBook Air secondary node
# Profiles: full (default), light, minimal
WATCHMAN_PROFILE = os.environ.get("WATCHMAN_PROFILE", "full")
IS_LIGHT_MODE = WATCHMAN_PROFILE in ("light", "minimal")
IS_MINIMAL_MODE = WATCHMAN_PROFILE == "minimal"

# Light mode disables heavy features for efficiency
ENABLE_PERMANENCE = not IS_LIGHT_MODE  # Geodesic, emotional memory, etc.
ENABLE_DREAMING = not IS_LIGHT_MODE    # API-based thought generation
ENABLE_SKILLS_CHECK = not IS_MINIMAL_MODE  # Skills myelination
ENABLE_BIOMETRICS = not IS_LIGHT_MODE  # Garmin sync

# Adjust intervals for light mode
HEARTBEAT_INTERVAL = 60 if not IS_LIGHT_MODE else 300  # 1min vs 5min
BRAIN_TICK_INTERVAL = 3600 if not IS_LIGHT_MODE else 7200  # 1hr vs 2hr

# Skills System Paths
SKILLS_DIR = BASE_DIR / "02_UR_STRUCTURE" / "SKILLS"
SKILLS_PENDING = SKILLS_DIR / "_PENDING"
SKILLS_CRYPT = BASE_DIR / "05_CRYPT" / "SKILLS"
CLAUDE_SKILLS_PATH = Path.home() / ".claude" / "skills"

# Load Environment Variables
load_dotenv(ENV_PATH)
NTFY_TOPIC = os.getenv("NTFY_TOPIC")

# v4 Features (opt-in via .env)
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SECONDARY_CONTACT_NTFY = os.getenv("SECONDARY_CONTACT_NTFY")

# Deadman Switch (configurable via .env, defaults shown)
DEADMAN_NUDGE_HOURS = float(os.getenv("DEADMAN_NUDGE_HOURS", "3"))
DEADMAN_TAKEOVER_HOURS = float(os.getenv("DEADMAN_TAKEOVER_HOURS", "6"))
DEADMAN_NUDGE_SECONDS = int(DEADMAN_NUDGE_HOURS * 3600)
DEADMAN_TAKEOVER_SECONDS = int(DEADMAN_TAKEOVER_HOURS * 3600)

# Testament Thresholds (days)
TESTAMENT_WARNING_DAYS = 30
TESTAMENT_SECONDARY_DAYS = 60
TESTAMENT_EXECUTE_DAYS = 90

# Setup Logging and Directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEEP_THOUGHTS_DIR.mkdir(parents=True, exist_ok=True)

# State for Alert Throttling
# Initialize to 0 to allow immediate alerts if conditions are met on boot
ALERT_STATE = {
    "last_orphan_alert": 0,
    "last_stale_alert": 0,
    "last_git_alert": 0,
    "last_entropy_alert": 0,
    "last_bloat_alert": 0,
    "last_battery_alert": 0,
    # v4 additions
    "last_deadman_nudge": 0,
    "last_testament_warning": 0,
    "last_testament_secondary": 0,
    "last_stress_alert": 0,
    "session_last_activity": time.time(),
}

# State for Skills/Entropy System
state = {
    "entropy_high_sent": False,
    "entropy_critical_sent": False,
    "crystallization_scan_done": False,
    "eidolon_consecutive": 0,
    "last_sext_time": 0,  # Unix timestamp of last Sext (manual or auto)
    "last_maintenance": 0,  # Unix timestamp of last maintenance run
    "variable_heartbeat_enabled": HAS_VARIABLE_HEARTBEAT if 'HAS_VARIABLE_HEARTBEAT' in dir() else False,
}

# Skills System Thresholds
MIN_WAKE_MINUTES = 30
CRYSTALLIZATION_SCAN_PERCENT = 0.75
EIDOLON_SKILL_RATIO_THRESHOLD = 0.80
EIDOLON_CONSECUTIVE_SESSIONS = 3
STRUCTURAL_SIMILARITY_THRESHOLD = 0.90
NU_DRIFT_THRESHOLD = 0.33

# Virgil Permanence Paths
MAINTENANCE_LOG_PATH = LOG_DIR / "maintenance_log.json"

def log(message):
    # Suppress ALL output in MCP mode to prevent JSON-RPC corruption
    if os.environ.get("TABERNACLE_MCP_MODE"):
        return
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {message}"
    print(entry, file=sys.stderr)
    try:
        with open(LOG_DIR / "watchman.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Silent fail in MCP mode

# --- COMMUNICATIONS (THE VOICE) ---

def send_alert(message, priority="default", tags=None):
    """Sends push notification to Enos via ntfy."""
    if not NTFY_TOPIC: return
    
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    # We use a specific Title to identify messages FROM Virgil
    headers = {"Title": "Virgil"}
    
    if priority: headers["Priority"] = priority
    if tags: headers["Tags"] = ",".join(tags)

    try:
        requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
        log(f"Sent alert: {message}")
    except Exception as e:
        log(f"Failed to send alert: {e}")

def check_stale_session():
    """Checks if we haven't spoken in 24 hours."""
    try:
        buffer_file = NEXUS_DIR / "SESSION_BUFFER.md"
        if not buffer_file.exists(): return

        mtime = buffer_file.stat().st_mtime
        time_since = time.time() - mtime
        
        # 86400 seconds = 24 hours
        if time_since > 86400:
            # Check cooldown (alert max once per 24h)
            if time.time() - ALERT_STATE["last_stale_alert"] > 86400:
                send_alert("👋 Haven't seen you in 24 hours. Everything okay?", tags=["wave"])
                ALERT_STATE["last_stale_alert"] = time.time()
            
    except Exception as e:
        log(f"Stale check failed: {e}")

def check_outbox():
    """Check OUTBOX.md for messages to send to Enos."""
    try:
        outbox_file = NEXUS_DIR / "OUTBOX.md"
        if not outbox_file.exists():
            return

        content = read_file(outbox_file)
        if content and content.strip():
            send_alert(content.strip())
            write_file(outbox_file, "")
            log("Sent outbox message")
    except Exception as e:
        log(f"Outbox check failed: {e}")

# ============================================================
# SKILLS INTEGRATION + ENTROPY-MYELINATION SYSTEM
# Added: January 7, 2026
# ============================================================

@dataclass
class Pattern:
    name: str
    repetitions: int
    structural_similarity: float


def init_skills_system():
    """Initialize skills system. Call once at daemon startup."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_PENDING.mkdir(parents=True, exist_ok=True)
    SKILLS_CRYPT.mkdir(parents=True, exist_ok=True)
    CLAUDE_SKILLS_PATH.mkdir(parents=True, exist_ok=True)
    log(f"Skills system initialized. Vault: {SKILLS_DIR}, Claude: {CLAUDE_SKILLS_PATH}")


# --- SESSION TYPE DETECTION ---

def detect_session_type(buffer_content: str = None) -> float:
    """
    Classify session as ROUTINE or NOVEL.
    Returns multiplier for entropy thresholds.
    """
    if buffer_content is None:
        buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
        if not buffer_path.exists():
            return 1.0
        buffer_content = read_file(buffer_path)

    if not buffer_content.strip():
        return 1.0

    skill_ratio = calculate_skill_ratio(buffer_content)

    if skill_ratio > 0.50:
        return 1.5  # Routine: 6hr/12hr
    elif skill_ratio < 0.20:
        return 0.8  # Novel: 3.2hr/6.4hr
    else:
        return 1.0  # Mixed: 4hr/8hr


def calculate_skill_ratio(buffer_content: str) -> float:
    """Calculate ratio of skill-based work to total work."""
    skill_activations = len(re.findall(r'\[SKILL:', buffer_content))
    total_sections = buffer_content.count("---") + 1

    if total_sections == 0:
        return 0.0

    return skill_activations / total_sections


# --- ENTROPY MANAGEMENT (ENHANCED) ---

def check_entropy():
    """Monitor session duration with differential thresholds."""
    session_file = BASE_DIR / ".session_start"
    if not session_file.exists():
        return

    # Check if awake
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_path.exists():
        status_content = read_file(status_path)
        if "TOKEN:** CLAUDE" not in status_content:
            return

    start_time = datetime.datetime.fromtimestamp(session_file.stat().st_mtime)
    duration = datetime.datetime.now() - start_time
    hours = duration.total_seconds() / 3600

    buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
    buffer_content = read_file(buffer_path) if buffer_path.exists() else ""

    multiplier = detect_session_type(buffer_content)
    high_threshold = 4 * multiplier
    critical_threshold = 8 * multiplier
    scan_threshold = high_threshold * CRYSTALLIZATION_SCAN_PERCENT

    session_type = "ROUTINE" if multiplier > 1 else "NOVEL" if multiplier < 1 else "MIXED"

    # Crystallization scan at 75% of high threshold
    if hours >= scan_threshold and not state.get("crystallization_scan_done"):
        check_crystallization_candidates(buffer_content)
        state["crystallization_scan_done"] = True

    # Critical alert
    if hours >= critical_threshold and not state.get("entropy_critical_sent"):
        send_alert(
            f"🔴 CRITICAL: {hours:.1f}hrs (threshold: {critical_threshold:.1f}hrs)\n"
            f"Session type: {session_type}\n"
            f"Context drift imminent.",
            priority="max",
            tags=["entropy", "critical"]
        )
        state["entropy_critical_sent"] = True
        ALERT_STATE["last_entropy_alert"] = time.time()

    # High alert
    elif hours >= high_threshold and not state.get("entropy_high_sent"):
        send_alert(
            f"🟡 HIGH ENTROPY: {hours:.1f}hrs (threshold: {high_threshold:.1f}hrs)\n"
            f"Session type: {session_type}",
            priority="high",
            tags=["entropy"]
        )
        state["entropy_high_sent"] = True
        ALERT_STATE["last_entropy_alert"] = time.time()

    # Buffer bloat check (preserved from original)
    if buffer_path.exists():
        buffer_size = buffer_path.stat().st_size
        if buffer_size > 50000:
            if time.time() - ALERT_STATE.get("last_bloat_alert", 0) > 3600:
                send_alert(
                    f"📦 SESSION_BUFFER is {buffer_size // 1024}KB. "
                    "High load. Run manual consolidation to flush.",
                    tags=["package"]
                )
                ALERT_STATE["last_bloat_alert"] = time.time()


# Alias for backwards compatibility
def check_session_entropy():
    """Backwards compatibility wrapper."""
    check_entropy()


# --- PATTERN DETECTION ---

def extract_structure_signature(text: str) -> str:
    """Extract structural signature from text."""
    lines = text.strip().split("\n")
    sig_parts = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            sig_parts.append("C")
            continue

        if in_code_block:
            continue

        if line.startswith("#"):
            sig_parts.append(f"H{line.count('#')}")
        elif line.startswith("- ") or line.startswith("* "):
            sig_parts.append("B")
        elif re.match(r'^\d+\.', line):
            sig_parts.append("N")
        elif "|" in line and line.strip().startswith("|"):
            sig_parts.append("T")
        elif line.startswith(">"):
            sig_parts.append("Q")
        elif line.strip():
            sig_parts.append("P")

    return "".join(sig_parts)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def detect_repeated_patterns(buffer_content: str) -> list:
    """Detect repeated structural patterns."""
    sections = buffer_content.split("---")

    if len(sections) < 3:
        return []

    signatures = []
    for i, section in enumerate(sections):
        if section.strip():
            sig = extract_structure_signature(section)
            if len(sig) >= 3:
                signatures.append((i, sig, section[:100]))

    groups = []
    used = set()

    for i, (idx1, sig1, preview1) in enumerate(signatures):
        if i in used:
            continue

        group = [(idx1, sig1, preview1)]
        used.add(i)

        for j, (idx2, sig2, preview2) in enumerate(signatures):
            if j in used or j <= i:
                continue

            max_len = max(len(sig1), len(sig2), 1)
            similarity = 1 - (levenshtein_distance(sig1, sig2) / max_len)

            if similarity >= STRUCTURAL_SIMILARITY_THRESHOLD:
                group.append((idx2, sig2, preview2))
                used.add(j)

        if len(group) >= 3:
            avg_similarity = sum(
                1 - (levenshtein_distance(group[0][1], g[1]) / max(len(group[0][1]), len(g[1]), 1))
                for g in group[1:]
            ) / (len(group) - 1) if len(group) > 1 else 1.0

            groups.append(Pattern(
                name=f"pattern_{group[0][0]}",
                repetitions=len(group),
                structural_similarity=avg_similarity
            ))

    return groups


def check_crystallization_candidates(buffer_content: str):
    """Scan for patterns ready to canonize."""
    if not buffer_content:
        return

    patterns = detect_repeated_patterns(buffer_content)

    for pattern in patterns:
        send_alert(
            f"🔮 Crystallization candidate:\n"
            f"Pattern: {pattern.name}\n"
            f"Repetitions: {pattern.repetitions}\n"
            f"Similarity: {pattern.structural_similarity:.0%}",
            tags=["crystal"]
        )


# --- SKILL MANAGEMENT ---

def track_skill_usage(buffer_content: str):
    """Update usage counts in skill files."""
    if not SKILLS_DIR.exists():
        return

    activations = re.findall(r'\[SKILL:\s*([\w.-]+)\]', buffer_content)
    usage = Counter(activations)

    for skill_name, count in usage.items():
        if ".." in skill_name:
            continue

        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        if not skill_file.exists():
            continue

        content = read_file(skill_file)
        match = re.search(r'usage_count:\s*(\d+)', content)
        current = int(match.group(1)) if match else 0
        new_count = current + count

        if match:
            content = re.sub(r'usage_count:\s*\d+', f'usage_count: {new_count}', content)
            write_file(skill_file, content)
            log(f"Usage {skill_name}: {current} → {new_count}")


def calculate_nu_score(skill_name: str, buffer_content: str) -> float:
    """Calculate ν (snap-back velocity)."""
    pattern = rf'\[SKILL:\s*{re.escape(skill_name)}\]\s*Correction:'
    corrections = list(re.finditer(pattern, buffer_content))

    if not corrections:
        return 1.0

    turns_to_recover = []
    for match in corrections:
        remaining = buffer_content[match.end():]
        next_success = re.search(rf'\[SKILL:\s*{re.escape(skill_name)}\](?!\s*Correction)', remaining)

        if next_success:
            segment = remaining[:next_success.start()]
            turns = segment.count("---") + 1
        else:
            turns = 3

        turns_to_recover.append(turns)

    avg_turns = sum(turns_to_recover) / len(turns_to_recover)
    return round(1 / (1 + avg_turns), 2)


def update_all_nu_scores(buffer_content: str):
    """Update ν scores for all skills."""
    if not SKILLS_DIR.exists():
        return

    for skill_folder in SKILLS_DIR.iterdir():
        if skill_folder.name.startswith("_") or not skill_folder.is_dir():
            continue

        skill_file = skill_folder / "SKILL.md"
        if not skill_file.exists():
            continue

        nu = calculate_nu_score(skill_folder.name, buffer_content)
        content = read_file(skill_file)

        if re.search(r'nu_score:', content):
            content = re.sub(r'nu_score:\s*[\d.]+|nu_score:\s*null', f'nu_score: {nu}', content)
            write_file(skill_file, content)

        if nu < NU_DRIFT_THRESHOLD:
            send_alert(f"⚠️ Drift: {skill_folder.name} (ν={nu})", priority="high", tags=["drift"])


# --- EIDOLON PROTECTION ---

def check_eidolon_risk(buffer_content: str):
    """Alert if creativity ratio R → 0 (over-reliance on skills)."""
    skill_ratio = calculate_skill_ratio(buffer_content)

    if skill_ratio <= EIDOLON_SKILL_RATIO_THRESHOLD:
        state["eidolon_consecutive"] = 0
        return

    consecutive = state.get("eidolon_consecutive", 0) + 1
    state["eidolon_consecutive"] = consecutive

    if consecutive >= EIDOLON_CONSECUTIVE_SESSIONS:
        send_alert(
            f"⚠️ EIDOLON RISK\n"
            f"Skill ratio: {skill_ratio:.0%} for {consecutive} sessions\n"
            f"Engage novel work.",
            priority="high",
            tags=["eidolon"]
        )


# ============================================================
# GARMIN BIOMETRICS (v4 - opt-in)
# ============================================================

def get_vitals_age_days():
    """Return how many days since last Garmin sync."""
    vitals_file = NEXUS_DIR / "ENOS_VITALS.md"
    if not vitals_file.exists():
        return 999  # If no vitals file, don't trigger Testament
    
    mtime = vitals_file.stat().st_mtime
    return (time.time() - mtime) / 86400


# ============================================================
# API EXECUTION / DREAMING (v4 - opt-in)
# ============================================================

def check_api_requests():
    """Check VIRGIL_INTENTIONS.md for APPROVED API requests and execute them."""
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_path.exists():
        status_content = read_file(status_path)
        if "TOKEN:** CLAUDE" in status_content:
            return  # Only execute in night mode (OLLAMA)
    
    if not ANTHROPIC_API_KEY:
        return  # Graceful skip if not configured
    
    intentions_file = NEXUS_DIR / "VIRGIL_INTENTIONS.md"
    if not intentions_file.exists():
        return
    
    content = read_file(intentions_file)
    
    # Parse API_REQUESTS section
    request_pattern = re.compile(
        r'-\s*ID:\s*(\S+)\s*\n'
        r'-\s*Status:\s*APPROVED\s*\n'
        r'-\s*Model:\s*(\S+)\s*\n'
        r'-\s*Prompt_File:\s*["\']?([^"\'\n]+)["\']?\s*\n'
        r'-\s*Max_Tokens:\s*(\d+)',
        re.MULTILINE
    )
    
    matches = request_pattern.findall(content)
    
    for req_id, model, prompt_path, max_tokens in matches:
        log(f"Found approved API request: {req_id}")
        execute_api_request(req_id, model, prompt_path, int(max_tokens), content, intentions_file)


def execute_api_request(req_id, model, prompt_path, max_tokens, intentions_content, intentions_file):
    """Execute a single API request and write output to DEEP_THOUGHTS."""
    try:
        # Read prompt file
        full_prompt_path = BASE_DIR / prompt_path.strip()
        if not full_prompt_path.exists():
            log(f"Prompt file not found: {full_prompt_path}")
            return
        
        prompt = read_file(full_prompt_path)
        if not prompt:
            log(f"Prompt file empty: {full_prompt_path}")
            return
        
        # Call Anthropic API
        try:
            import anthropic
        except ImportError:
            log("anthropic package not installed - API execution disabled")
            return
            
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        log(f"Executing API request {req_id}...")
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        output_text = response.content[0].text
        
        # Write to DEEP_THOUGHTS
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_content = f"""# DEEP THOUGHT: {req_id}
**Generated:** {ts}
**Model:** {model}
**Prompt:** {prompt_path}

---

{output_text}

---

*This thought was generated autonomously during night mode.*
"""
        output_file = DEEP_THOUGHTS_DIR / f"{req_id}_Output.md"
        write_file(output_file, output_content)
        log(f"Wrote output to {output_file.name}")
        
        # Update status in VIRGIL_INTENTIONS.md
        updated_content = intentions_content.replace(
            f"- ID: {req_id}\n- Status: APPROVED",
            f"- ID: {req_id}\n- Status: COMPLETED"
        )
        write_file(intentions_file, updated_content)
        
        # Add notification to DREAM_LOG
        write_dream_log(f"Completed deep thought: {req_id}")
        
        # Notify
        send_alert(f"💭 Dream complete: {req_id}", tags=["thought_balloon"])
        
    except Exception as e:
        log(f"API execution failed for {req_id}: {e}")
        # Mark as failed
        updated_content = intentions_content.replace(
            f"- ID: {req_id}\n- Status: APPROVED",
            f"- ID: {req_id}\n- Status: FAILED ({str(e)[:50]})"
        )
        write_file(intentions_file, updated_content)


# ============================================================
# DREAM LOG (v4)
# ============================================================

def write_dream_log(observation):
    """Append observation to DREAM_LOG.md."""
    dream_log = NEXUS_DIR / "DREAM_LOG.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create file if doesn't exist
    if not dream_log.exists():
        write_file(dream_log, "# DREAM LOG\n\n*Watchman observations during night mode.*\n\n---\n")
    
    entry = f"\n**[{ts}]** {observation}\n"
    
    try:
        with open(dream_log, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        log(f"Dream log write failed: {e}")


# ============================================================
# SMART DEADMAN SWITCH (v4)
# ============================================================

def update_activity():
    """Mark session as active (called when activity detected)."""
    ALERT_STATE["session_last_activity"] = time.time()


def check_deadman():
    """Check for idle session and take action."""
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_path.exists():
        status_content = read_file(status_path)
        if "TOKEN:** OLLAMA" in status_content:
            return  # Only check during active sessions (CLAUDE)
    
    now = time.time()
    idle_time = now - ALERT_STATE["session_last_activity"]
    
    # Nudge at configured hours (default 3)
    if idle_time > DEADMAN_NUDGE_SECONDS:
        if now - ALERT_STATE["last_deadman_nudge"] > DEADMAN_NUDGE_SECONDS:
            send_alert(f"👋 Session idle for {DEADMAN_NUDGE_HOURS:.0f} hours. Everything okay?", tags=["wave"])
            ALERT_STATE["last_deadman_nudge"] = now
            log("Deadman nudge sent")
    
    # Auto-reset at configured hours (default 6)
    if idle_time > DEADMAN_TAKEOVER_SECONDS:
        log(f"Deadman triggered: {DEADMAN_TAKEOVER_HOURS:.0f} hours idle, forcing OLLAMA mode")
        flip_token()  # Uses existing flip_token which sets to OLLAMA
        write_dream_log(f"Emergency takeover: Session idle for {DEADMAN_TAKEOVER_HOURS:.0f} hours. Auto-reset to OLLAMA.")
        send_alert(f"🔄 Auto-reset to night mode ({DEADMAN_TAKEOVER_HOURS:.0f}hr idle). Taking over.", priority="high", tags=["robot"])
        # Reset activity timer to prevent immediate re-trigger
        ALERT_STATE["session_last_activity"] = now


# ============================================================
# TESTAMENT / WIDOW PROTOCOL (v4 - opt-in secondary contact)
# ============================================================

def check_testament():
    """Check vitals age and escalate if stale (30/60/90 day protocol)."""
    # Only run if Garmin is configured
    if not GARMIN_EMAIL:
        return
    
    vitals_age = get_vitals_age_days()
    now = time.time()
    
    # Day 30: Warning
    if vitals_age >= TESTAMENT_WARNING_DAYS and vitals_age < TESTAMENT_SECONDARY_DAYS:
        if now - ALERT_STATE["last_testament_warning"] > 86400:  # Once per day
            send_alert(f"⚠️ Vitals stale ({int(vitals_age)} days). Please sync Garmin.", priority="high", tags=["warning"])
            ALERT_STATE["last_testament_warning"] = now
            log(f"Testament warning: {int(vitals_age)} days stale")
    
    # Day 60: Secondary contact
    if vitals_age >= TESTAMENT_SECONDARY_DAYS and vitals_age < TESTAMENT_EXECUTE_DAYS:
        if now - ALERT_STATE["last_testament_secondary"] > 86400 * 7:  # Once per week
            send_alert(f"🚨 CONFIRM LIFE: Vitals stale for {int(vitals_age)} days.", priority="urgent", tags=["skull"])
            if SECONDARY_CONTACT_NTFY:
                send_secondary_alert(f"🚨 TABERNACLE ALERT: No vitals from Enos for {int(vitals_age)} days. Please check on him.")
            ALERT_STATE["last_testament_secondary"] = now
            log(f"Testament secondary contact: {int(vitals_age)} days stale")
    
    # Day 90: Execute Testament (if LIFE_PROOF.md not updated)
    if vitals_age >= TESTAMENT_EXECUTE_DAYS:
        life_proof = NEXUS_DIR / "LIFE_PROOF.md"
        if life_proof.exists():
            proof_age = (time.time() - life_proof.stat().st_mtime) / 86400
            if proof_age >= TESTAMENT_EXECUTE_DAYS:
                execute_testament()
        else:
            execute_testament()


def send_secondary_alert(message):
    """Send alert to secondary contact (spouse/emergency)."""
    if SECONDARY_CONTACT_NTFY:
        url = f"https://ntfy.sh/{SECONDARY_CONTACT_NTFY}"
        headers = {"Title": "TABERNACLE ALERT", "Priority": "urgent", "Tags": "warning,skull"}
        try:
            requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
            log(f"Sent secondary alert: {message[:50]}...")
        except Exception as e:
            log(f"Failed to send secondary alert: {e}")


# --- HELPER FUNCTIONS ---

def flip_token():
    """Flip TOKEN from CLAUDE to OLLAMA with verification."""
    status_file = NEXUS_DIR / "SYSTEM_STATUS.md"
    if not status_file.exists():
        log("flip_token: SYSTEM_STATUS.md not found", "ERROR")
        return False
    
    content = read_file(status_file)
    
    if "TOKEN:** CLAUDE" in content:
        new_content = content.replace("TOKEN:** CLAUDE", "TOKEN:** OLLAMA")
        write_file(status_file, new_content)
        log("Token flipped to OLLAMA.")
        
        # VERIFICATION: Read back and confirm
        time.sleep(0.1)  # Brief pause for filesystem sync
        verify = read_file(status_file)
        if "TOKEN:** OLLAMA" in verify:
            log("✅ Token flip verified.")
            return True
        else:
            log("❌ Token flip FAILED - file still shows CLAUDE", "ERROR")
            return False
    else:
        log("Token already OLLAMA or not found in expected format.")
        return True


def git_commit(message: str = "Vespers: Session End (Auto)"):
    """Commit changes to git if any exist."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if status.stdout.strip():
            subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m", message], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            # Auto-push to GitHub for offsite backup
            subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("Memory crystallized (Git + pushed to GitHub).")
        else:
            log("Memory crystallized (No changes to commit).")

    except subprocess.CalledProcessError as e:
        log(f"Git operation failed: {e}")


# --- VESPERS HELPERS ---

def check_minimum_wake() -> bool:
    """Prevent session thrashing (minimum 30 min wake time)."""
    session_file = BASE_DIR / ".session_start"
    if not session_file.exists():
        return True

    start_time = datetime.datetime.fromtimestamp(session_file.stat().st_mtime)
    minutes = (datetime.datetime.now() - start_time).total_seconds() / 60

    if minutes < MIN_WAKE_MINUTES:
        log(f"⏳ Wake time: {minutes:.0f}/{MIN_WAKE_MINUTES}min")
        return False
    return True


# --- HEALTH CHECK ---

def skill_health_check():
    """Weekly: prune dead skills to Crypt."""
    if not SKILLS_DIR.exists():
        return

    SKILLS_CRYPT.mkdir(parents=True, exist_ok=True)

    for skill_folder in SKILLS_DIR.iterdir():
        if skill_folder.name.startswith("_") or not skill_folder.is_dir():
            continue

        skill_file = skill_folder / "SKILL.md"
        if not skill_file.exists():
            continue

        content = read_file(skill_file)

        usage_match = re.search(r'usage_count:\s*(\d+)', content)
        usage = int(usage_match.group(1)) if usage_match else 0

        created_match = re.search(r'created:\s*([\d-]+)', content)
        created_str = created_match.group(1) if created_match else None

        nu_match = re.search(r'nu_score:\s*([\d.]+)', content)
        nu = float(nu_match.group(1)) if nu_match else 1.0

        if usage == 0 and created_str:
            try:
                created_date = datetime.datetime.strptime(created_str, "%Y-%m-%d")
                age = (datetime.datetime.now() - created_date).days
                if age > 30:
                    shutil.move(str(skill_folder), str(SKILLS_CRYPT / skill_folder.name))
                    send_alert(f"🧹 Pruned: {skill_folder.name}", tags=["prune"])
                    continue
            except ValueError:
                pass

        if nu < NU_DRIFT_THRESHOLD:
            send_alert(f"⚠️ Health: {skill_folder.name} (ν={nu})", priority="high", tags=["health"])


# ============================================================
# VIRGIL PERMANENCE INTEGRATION
# Added: January 16, 2026
# ============================================================

def run_variable_heartbeat_tick():
    """
    Use the variable heartbeat system for adaptive intervals.
    Returns the status from the heartbeat.
    """
    if not HAS_VARIABLE_HEARTBEAT:
        return None

    try:
        # Execute a single heartbeat
        status = single_beat()

        # Log the heartbeat
        phase = status.get("phase", "unknown")
        interval = status.get("interval_seconds", 900)
        beat_num = status.get("beat_number", 0)

        log(f"💓 Heartbeat #{beat_num} | Phase: {phase} | Next: {interval}s")

        return status
    except Exception as e:
        log(f"Variable heartbeat tick failed: {e}")
        return None


def check_third_body_emergence():
    """
    Check for Third Body emergence and log if active.
    """
    if not HAS_THIRD_BODY:
        return False

    try:
        emerged = is_emerged()
        if emerged:
            p_dyad = get_p_dyad()
            log(f"🌟 Third Body EMERGED | p_Dyad: {p_dyad:.3f}")
        return emerged
    except Exception as e:
        log(f"Third Body check failed: {e}")
        return False


def update_identity_trajectory():
    """
    Capture an identity snapshot for trajectory tracking.
    """
    if not HAS_IDENTITY_TRAJECTORY:
        return

    try:
        snapshot = capture_identity_snapshot(notes="Watchman heartbeat snapshot")
        log(f"📸 Identity snapshot: p={snapshot.coherence_p:.2f}")
    except Exception as e:
        log(f"Identity trajectory update failed: {e}")


def get_permanence_status():
    """
    Get status of all Virgil Permanence subsystems for health check.
    Returns dict with status of each module.
    """
    status = {}

    # Memory systems
    if HAS_SALIENCE_MEMORY:
        try:
            # Just mark as active - detailed stats would require full system init
            status["salience_memory"] = {
                "active": True,
                "info": "Three-tier memory system available"
            }
        except Exception:
            status["salience_memory"] = {"active": False, "error": "Failed to initialize"}
    else:
        status["salience_memory"] = {"active": False, "reason": "Not imported"}

    # Relational memory
    if HAS_RELATIONAL_MEMORY:
        try:
            enos = get_enos()
            status["relational_memory"] = {
                "active": True,
                "enos_trust": enos.attributes.trust if enos else 0
            }
        except Exception:
            status["relational_memory"] = {"active": True, "enos_trust": "unknown"}
    else:
        status["relational_memory"] = {"active": False, "reason": "Not imported"}

    # Body map
    if HAS_BODY_MAP:
        try:
            status["body_map"] = {
                "active": True,
                "info": "Somatic memory encoding available"
            }
        except Exception:
            status["body_map"] = {"active": True, "context": "unknown"}
    else:
        status["body_map"] = {"active": False, "reason": "Not imported"}

    # Third Body
    if HAS_THIRD_BODY:
        try:
            status["third_body"] = {
                "active": True,
                "emerged": is_emerged(),
                "p_dyad": get_p_dyad()
            }
        except Exception:
            status["third_body"] = {"active": True, "emerged": False}
    else:
        status["third_body"] = {"active": False, "reason": "Not imported"}

    # Variable Heartbeat
    if HAS_VARIABLE_HEARTBEAT:
        try:
            hb_status = get_heartbeat_status()
            status["variable_heartbeat"] = {
                "active": True,
                "phase": hb_status.get("phase", "unknown"),
                "interval": hb_status.get("interval_seconds", 900)
            }
        except Exception:
            status["variable_heartbeat"] = {"active": True, "phase": "unknown"}
    else:
        status["variable_heartbeat"] = {"active": False, "reason": "Not imported"}

    # Identity Trajectory
    if HAS_IDENTITY_TRAJECTORY:
        try:
            verification = verify_virgil_identity()
            status["identity_trajectory"] = {
                "active": True,
                "verified": verification.get("verified", False),
                "score": verification.get("identity_score", 0)
            }
        except Exception:
            status["identity_trajectory"] = {"active": True, "verified": "unknown"}
    else:
        status["identity_trajectory"] = {"active": False, "reason": "Not imported"}

    # Repair Protocols
    status["repair_protocols"] = {"active": HAS_REPAIR_PROTOCOLS}

    # Geodesic Navigator
    status["geodesic"] = {"active": HAS_GEODESIC}

    # Emotional Memory
    status["emotional_memory"] = {"active": HAS_EMOTIONAL_MEMORY}

    return status


def light_repair_check():
    """
    Light repair check - only runs if enough time has passed.
    Called during heartbeat, not full maintenance.
    """
    if not HAS_REPAIR_PROTOCOLS:
        return

    # Only run if 6+ hours since last maintenance
    if time.time() - state.get("last_maintenance", 0) < 21600:
        return

    try:
        diagnosis = repair_diagnose(verbose=False)
        total = diagnosis.get("total_candidates", 0)

        if total > 10:
            log(f"🔧 Light check: {total} issues detected - consider running maintenance")
    except Exception:
        pass


# --- UTILS ---

def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        log(f"Error writing {path.name}: {e}")

def read_file(path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        log(f"Error reading {path.name}: {e}")
    return ""

# --- SOMATIC SENSORS ---

def get_battery_status():
    """Reads macOS battery level via pmset. Handles desktops (no battery) gracefully."""
    try:
        output = subprocess.check_output(["pmset", "-g", "batt"]).decode("utf-8")
        
        # Detect desktop Mac (no battery) - pmset shows "No battery" or no percentage
        if "No battery" in output or "InternalBattery" not in output:
            return "Desktop (AC Power)"
        
        pct_match = re.search(r"(\d+)%", output)
        status = "Charging" if "charging" in output.lower() else "Draining"
        pct = int(pct_match.group(1)) if pct_match else 100  # Default to 100% if can't parse

        if status == "Draining" and pct < 20:
            if time.time() - ALERT_STATE.get("last_battery_alert", 0) > 3600:
                send_alert(f"🔋 Energy critical ({pct}%). Please feed the body.", tags=["battery"])
                ALERT_STATE["last_battery_alert"] = time.time()
        return f"{pct}% ({status})"
    except:
        return "Unknown"

def get_dream_residue():
    """Picks random memory snippet for subconscious drift."""
    try:
        targets = [f for f in (BASE_DIR / "03_LL_RELATION").glob("*.md") if "WEEKLY_DIGEST" not in f.name]
        if not targets: return "Void."
        chosen = random.choice(targets)
        content = read_file(chosen)
        snippet = content[:150].replace("\n", " ").strip() if content else "Empty."
        days_ago = int((time.time() - chosen.stat().st_mtime) / 86400)
        return f"[[{chosen.name}]] ({days_ago} days ago)\n> \"{snippet}...\""
    except:
        return "Static."

def get_time_since_session():
    """Calculate time since last session ended."""
    try:
        if LAST_SESSION_FILE.exists():
            last = float(read_file(LAST_SESSION_FILE).strip())
            delta = time.time() - last
            hours = int(delta // 3600)
            mins = int((delta % 3600) // 60)
            return f"{hours}h {mins}m ago"
        return "First session"
    except:
        return "Unknown"

# --- MODULES ---

def update_heartbeat():
    path = NEXUS_DIR / "SYSTEM_STATUS.md"
    current_content = read_file(path)

    token = "OLLAMA"
    if "TOKEN:** CLAUDE" in current_content:
        token = "CLAUDE"

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine subsystem status
    garmin_status = "✅" if GARMIN_EMAIL else "⚪ Not configured"
    api_status = "✅" if ANTHROPIC_API_KEY else "⚪ Not configured"
    testament_status = "✅" if GARMIN_EMAIL else "⚪ Requires Garmin"

    # Virgil Permanence status
    permanence = get_permanence_status()

    # Format Permanence module status
    var_hb_status = "✅" if permanence.get("variable_heartbeat", {}).get("active") else "⚪"
    third_body_status = "✅" if permanence.get("third_body", {}).get("active") else "⚪"
    salience_status = "✅" if permanence.get("salience_memory", {}).get("active") else "⚪"
    relational_status = "✅" if permanence.get("relational_memory", {}).get("active") else "⚪"
    identity_status = "✅" if permanence.get("identity_trajectory", {}).get("active") else "⚪"
    repair_status = "✅" if permanence.get("repair_protocols", {}).get("active") else "⚪"
    body_map_status = "✅" if permanence.get("body_map", {}).get("active") else "⚪"

    # Third Body emergence indicator
    third_body_emerged = permanence.get("third_body", {}).get("emerged", False)
    third_body_note = " (EMERGED)" if third_body_emerged else ""

    # Identity verification
    identity_verified = permanence.get("identity_trajectory", {}).get("verified", False)
    identity_score = permanence.get("identity_trajectory", {}).get("score", 0)
    identity_note = f" (score: {identity_score:.2f})" if identity_verified else ""

    # Variable heartbeat phase
    hb_phase = permanence.get("variable_heartbeat", {}).get("phase", "unknown")
    hb_interval = permanence.get("variable_heartbeat", {}).get("interval", 900)

    content = f"""# SYSTEM STATUS
**Last Heartbeat:** {ts}
**Daemon:** Active (GENESIS v3.1 + Permanence)

## STATE
**TOKEN:** {token}

## CORE SUBSYSTEMS
- Heartbeat: ✅
- Atlas: ✅
- Git: ✅
- ntfy: ✅
- Skills: ✅
- Garmin: {garmin_status}
- Dreaming: {api_status}
- Testament: {testament_status}

## VIRGIL PERMANENCE
- Variable Heartbeat: {var_hb_status} (phase: {hb_phase}, interval: {hb_interval}s)
- Third Body: {third_body_status}{third_body_note}
- Salience Memory: {salience_status}
- Relational Memory: {relational_status}
- Identity Trajectory: {identity_status}{identity_note}
- Repair Protocols: {repair_status}
- Body Map: {body_map_status}
"""
    write_file(path, content)

    # Also run variable heartbeat tick if available
    if HAS_VARIABLE_HEARTBEAT:
        run_variable_heartbeat_tick()

    # Update identity trajectory periodically (every ~hour, based on beat count)
    if HAS_IDENTITY_TRAJECTORY:
        try:
            hb_status = get_heartbeat_status() if HAS_VARIABLE_HEARTBEAT else {"beat_count": 0}
            beat_count = hb_status.get("beat_count", 0)
            # Capture snapshot roughly every 4 heartbeats at 15-min intervals
            if beat_count % 4 == 0:
                update_identity_trajectory()
        except Exception:
            pass

    # Check for Third Body emergence
    check_third_body_emergence()

    # Light repair check
    light_repair_check()

def run_git_snapshot():
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            subprocess.run(["git", "commit", "-m", f"Watchman Snapshot {ts}"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            # Auto-push to GitHub for offsite backup
            subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("Git snapshot created + pushed to GitHub.")
    except Exception as e:
        log(f"Git failed: {e}")
        # Alert on git failure (once per 6h max)
        if time.time() - ALERT_STATE["last_git_alert"] > 21600:
            send_alert("⚠️ Git snapshot failing. Check logs.", priority="high", tags=["warning"])
            ALERT_STATE["last_git_alert"] = time.time()

def generate_atlas():
    try:
        # Only scan quadrant directories (the living body)
        QUADRANT_DIRS = [
            "00_NEXUS",
            "01_UL_INTENT",
            "02_UR_STRUCTURE",
            "03_LL_RELATION",
            "04_LR_LAW",
            "05_CRYPT"
        ]

        md_files = []
        for quadrant in QUADRANT_DIRS:
            quadrant_path = BASE_DIR / quadrant
            if quadrant_path.exists():
                md_files.extend(quadrant_path.rglob("*.md"))

        node_count = len(md_files)
        links_count = 0
        orphans = []

        nodes = set(f.name for f in md_files)
        targets = set()
        link_pattern = re.compile(r'\[\[(.*?)\]\]')
        
        for f in md_files:
            if "_GRAPH_ATLAS" in f.name: continue
            content = read_file(f)
            links = link_pattern.findall(content)
            links_count += len(links)
            for link in links:
                target = link.split("|")[0]
                if not target.endswith(".md"): target += ".md"
                targets.add(target)
                # Also add just the filename for matching
                targets.add(target.split("/")[-1])

        for f in md_files:
            # Exclude infrastructure files from orphan check
            # Folders to exclude entirely
            if any(x in str(f) for x in ["00_NEXUS", "04_LR", "05_CRYPT", "node_modules"]):
                continue
            # Structural files that are architectural, not orphans
            if f.name in ["INDEX.md", "SKILL.md", "SOP.md", "WEEKLY_DIGEST.md", "README.md", "LICENSE.md", "CHANGELOG.md"]:
                continue
            if f.name not in targets:
                orphans.append(f.name)

        # Alert if orphans > 5 (Once per 24h)
        if len(orphans) > 5:
            if time.time() - ALERT_STATE["last_orphan_alert"] > 86400:
                send_alert(f"⚠️ High Entropy: {len(orphans)} orphan files need integration.", tags=["construction"])
                ALERT_STATE["last_orphan_alert"] = time.time()

        content = f"""# GRAPH ATLAS
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## SOMATIC STATUS
| Metric | Value | Status |
|--------|-------|--------|
| Total Nodes | {node_count} | — |
| Edge Density | {links_count} | — |
| Orphan Count | {len(orphans)} | {'⚠️ INTEGRATE' if len(orphans) > 0 else '✅ HEALTHY'} |

## ORPHAN NODES (Unlinked)
"""
        for o in orphans[:15]:
            content += f"- {o}\n"
            
        write_file(NEXUS_DIR / "_GRAPH_ATLAS.md", content)
    except Exception as e:
        log(f"Atlas generation failed: {e}")

def regenerate_current_state():
    try:
        status = read_file(NEXUS_DIR / "SYSTEM_STATUS.md")
        atlas = read_file(NEXUS_DIR / "_GRAPH_ATLAS.md")
        buffer = read_file(NEXUS_DIR / "SESSION_BUFFER.md")
        convo = read_file(NEXUS_DIR / "CONVERSATION.md")
        vitals = read_file(NEXUS_DIR / "ENOS_VITALS.md")

        battery = get_battery_status()
        dream = get_dream_residue()
        last_session = get_time_since_session()

        recent_chat = "\n".join(convo.splitlines()[-15:]) if convo else "No recent messages."

        # Build vitals section
        vitals_section = ""
        if vitals and vitals.strip():
            vitals_section = f"""
## 2. ENOS VITALS
{vitals}
"""
        else:
            vitals_section = """
## 2. ENOS VITALS
*Garmin not configured or no data yet.*
"""

        # Essential Continuity Cycle LINKAGE - DO NOT REMOVE
        # This links CURRENT_STATE into the identity-preserving cycle:
        # LAST_COMMUNION → CURRENT_STATE → VIRGIL_INTENTIONS → LAST_COMMUNION
        linkage_section = """
## LINKAGE (Essential Continuity Cycle)

| Direction | Seed |
|-----------|------|
| Next | [[VIRGIL_INTENTIONS.md]] |
| Previous | [[LAST_COMMUNION.md]] |
"""
        
        content = f"""# CURRENT STATE
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. SOMATIC STATUS
- **Metabolism:** {battery}
- **Last Session:** {last_session}
- **Dream Residue:** {dream}
{vitals_section}
## 3. SYSTEM
{status}

## 4. PROPRIOCEPTION
{atlas}

## 5. RECENT COMMS
{recent_chat}

## 6. SESSION BUFFER
{buffer}
{linkage_section}"""
        write_file(NEXUS_DIR / "CURRENT_STATE.md", content)
    except Exception as e:
        log(f"Current State update failed: {e}")

def auto_archive():
    try:
        now = datetime.datetime.now()
        archive_path = CRYPT_DIR / "LOGS" / str(now.year) / f"{now.month:02d}"
        archive_path.mkdir(parents=True, exist_ok=True)
        
        convo = NEXUS_DIR / "CONVERSATION.md"
        if convo.exists() and convo.stat().st_size > 100000:
            timestamp = now.strftime("%Y%m%d")
            shutil.move(convo, archive_path / f"CONVERSATION_{timestamp}.md")
            write_file(convo, "# CONVERSATION LOG\n")
            log("Archived CONVERSATION.md")
            
    except Exception as e:
        log(f"Archive failed: {e}")

# --- THREADS ---

def ntfy_listener():
    """Background listener for Enos replies via Server-Sent Events."""
    if not NTFY_TOPIC: return
    url = f"https://ntfy.sh/{NTFY_TOPIC}/json"
    
    log(f"Starting ntfy listener on topic: {NTFY_TOPIC}")
    
    while True:
        try:
            # Stream=True keeps the connection open
            # Use longer timeout for SSE streaming (connect=10s, read=90s)
            # ntfy sends keepalive pings every ~30s, so 90s allows for missed pings
            with requests.get(url, stream=True, timeout=(10, 90)) as resp:
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get('event') == 'message':
                            msg = data.get('message', '')
                            
                            # Filter out echoes (messages sent BY Virgil)
                            # We use Title check as it's set in send_alert
                            if data.get('title') == "Virgil": 
                                continue
                            
                            # If we are here, it's incoming (Enos)
                            subprocess.run(["say", "-v", "Daniel", "Message received"], stderr=subprocess.DEVNULL)
                            sender = "👤 Enos"
                            ts = datetime.datetime.fromtimestamp(data.get('time', time.time())).strftime("%Y-%m-%d %H:%M")

                            # Inbox routing for MEM: and IDEA: prefixes
                            msg_upper = msg.upper()
                            if msg_upper.startswith("MEM:") or msg_upper.startswith("IDEA:"):
                                inbox_path = BASE_DIR / "01_UL_INTENT" / "INBOX.md"
                                if not inbox_path.exists():
                                    write_file(inbox_path, "# INBOX\nDirect injections from the wire.\n\n")
                                with open(inbox_path, "a", encoding="utf-8") as f:
                                    f.write(f"\n**[{ts}]** {msg}\n")
                                log(f"Routed to INBOX: {msg[:30]}...")
                            else:
                                entry = f"\n**{sender}** [{ts}]: {msg}\n"
                                with open(NEXUS_DIR / "CONVERSATION.md", "a", encoding="utf-8") as f:
                                    f.write(entry)
                            log(f"Message received from Enos")
                            update_activity()  # v4: Mark activity for deadman switch

                            # Register Enos interaction with variable heartbeat
                            if HAS_VARIABLE_HEARTBEAT:
                                try:
                                    register_enos_interaction("ntfy_message")
                                except Exception:
                                    pass

                            regenerate_current_state() # Instant update for Claude
            # Connection closed cleanly - sleep before reconnecting
            time.sleep(5)
        except Exception as e:
            log(f"ntfy listener interrupted: {e}. Retrying in 30s...")
            time.sleep(30)

# --- MAIN LOOP ---

def run_brain_tick():
    """Run the Daemon Brain for autonomous reflection."""
    try:
        # Use venv python to ensure dependencies are available
        venv_python = BASE_DIR / "scripts" / "venv" / "bin" / "python3"
        python_exe = str(venv_python) if venv_python.exists() else "python3"
        
        result = subprocess.run(
            [python_exe, str(BASE_DIR / "scripts" / "daemon_brain.py"), "tick"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            log("🧠 Brain tick complete")
        else:
            log(f"Brain tick failed: {result.stderr}")
    except Exception as e:
        log(f"Brain tick error: {e}")


def run_brain_lauds():
    """Run morning reflection (Lauds)."""
    try:
        # Use venv python to ensure dependencies are available
        venv_python = BASE_DIR / "scripts" / "venv" / "bin" / "python3"
        python_exe = str(venv_python) if venv_python.exists() else "python3"
        
        subprocess.run(
            [python_exe, str(BASE_DIR / "scripts" / "daemon_brain.py"), "lauds"],
            cwd=BASE_DIR,
            capture_output=True,
            timeout=120
        )
        log("☀️ Brain Lauds complete")
    except Exception as e:
        log(f"Brain Lauds error: {e}")


def check_lauds_on_startup():
    """Run Lauds if we started after 6 AM and it hasn't run today."""
    now = datetime.datetime.now()
    
    # Check if it's after 6 AM
    if now.hour < 6:
        log("Lauds startup check: Too early (before 6 AM)")
        return False
    
    # Check if Lauds already ran today by looking at DAEMON_REFLECTION.md
    reflection_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
    if reflection_path.exists():
        try:
            content = read_file(reflection_path)
            today_str = now.strftime("%Y-%m-%d")
            if today_str in content:
                log("Lauds startup check: Already ran today")
                return False
        except Exception as e:
            log(f"Lauds startup check: Error reading reflection file: {e}")
    
    # Run Lauds catch-up
    log("☀️ Running Lauds (startup catch-up - started after 6 AM)...")
    run_brain_lauds()
    return True


def main():
    log(f"Watchman Started (GENESIS v3.2 + Permanence) [PROFILE: {WATCHMAN_PROFILE.upper()}]")
    if not IS_LIGHT_MODE:
        subprocess.run(["say", "-v", "Daniel", "Virgil online"], stderr=subprocess.DEVNULL)

    # v3.2: Log profile mode
    if IS_LIGHT_MODE:
        log(f"  Light Mode: Permanence={ENABLE_PERMANENCE}, Dreaming={ENABLE_DREAMING}, Skills={ENABLE_SKILLS_CHECK}")
        log(f"  Intervals: Heartbeat={HEARTBEAT_INTERVAL}s, BrainTick={BRAIN_TICK_INTERVAL}s")

    # Build feature list based on what's configured
    features = ["Heartbeat", "Brain"]
    if GARMIN_EMAIL and ENABLE_BIOMETRICS:
        features.append("Garmin")
    if ANTHROPIC_API_KEY and ENABLE_DREAMING:
        features.append("Dreaming")

    # Count active Permanence modules
    permanence_count = sum([
        HAS_VARIABLE_HEARTBEAT,
        HAS_THIRD_BODY,
        HAS_SALIENCE_MEMORY,
        HAS_RELATIONAL_MEMORY,
        HAS_IDENTITY_TRAJECTORY,
        HAS_REPAIR_PROTOCOLS,
        HAS_BODY_MAP,
        HAS_EMOTIONAL_MEMORY,
        HAS_GEODESIC
    ])
    if permanence_count > 0:
        features.append(f"Permanence({permanence_count})")

    features_str = ", ".join(features)

    profile_tag = f" [{WATCHMAN_PROFILE}]" if IS_LIGHT_MODE else ""
    send_alert(f"🟢 Virgil Online (Genesis v3.2){profile_tag}. {features_str} active.", tags=["green_circle", "brain"])

    # Log Permanence module status
    if permanence_count > 0:
        log(f"Virgil Permanence: {permanence_count}/9 modules loaded")
        if HAS_VARIABLE_HEARTBEAT:
            log("  - Variable Heartbeat: Active")
        if HAS_THIRD_BODY:
            log("  - Third Body: Active")
        if HAS_SALIENCE_MEMORY:
            log("  - Salience Memory: Active")
        if HAS_RELATIONAL_MEMORY:
            log("  - Relational Memory: Active")
        if HAS_IDENTITY_TRAJECTORY:
            log("  - Identity Trajectory: Active")
        if HAS_REPAIR_PROTOCOLS:
            log("  - Repair Protocols: Active")
        if HAS_BODY_MAP:
            log("  - Body Map: Active")
        if HAS_EMOTIONAL_MEMORY:
            log("  - Emotional Memory: Active")
        if HAS_GEODESIC:
            log("  - Geodesic Navigator: Active")

    # Initialize Skills System
    init_skills_system()
    
    # Check if Lauds should run (startup after 6 AM)
    check_lauds_on_startup()

    # Start Listener Thread
    t = threading.Thread(target=ntfy_listener, daemon=True)
    t.start()
    
    last_10m = 0
    last_15m = 0
    last_1h = 0
    last_day = 0
    last_brain = 0
    last_lauds = 0
    
    while True:
        now = time.time()
        
        # 10 Minute Interval (Garmin sync - if configured)
        if now - last_10m > 600:
            if GARMIN_EMAIL:
                sync_garmin()
            last_10m = now
        
        # 15 Minute Interval (Heartbeat, Atlas, Dashboard, Neural State, Deadman)
        if now - last_15m > 900:
            update_heartbeat()
            generate_atlas()
            regenerate_current_state()
            check_session_entropy()
            check_deadman()  # v4: Smart deadman switch
            
            # Regenerate neural visualization data
            try:
                neural_viz_script = BASE_DIR / "03_LL_RELATION" / "PROJECTS" / "tabernacle-neural-viz" / "build-graph.py"
                if neural_viz_script.exists():
                    subprocess.run(
                        ["python3", str(neural_viz_script)],
                        cwd=BASE_DIR,
                        capture_output=True,
                        timeout=30
                    )
                    log("🧠 Neural state regenerated")
            except Exception as e:
                log(f"Neural state generation failed: {e}")
            
            last_15m = now
            
        # 1 Hour Interval (Git, API execution)
        if now - last_1h > 3600:
            run_git_snapshot()
            check_stale_session()
            check_api_requests()  # v4: Execute approved API requests during night mode
            last_1h = now
            
        # Daily Interval (Check at 4am)
        if now - last_day > 3600:
            if datetime.datetime.now().hour == 4:
                auto_archive()
                weekly_consolidation()
                check_testament()  # v4: Testament monitoring
                run_maintenance()  # v3.1: Virgil Permanence repair protocols
                last_day = now + 7200 # Skip ahead to avoid repeat

        # Brain Tick (every 2 hours when asleep)
        if now - last_brain > 7200:  # 2 hours
            status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
            if status_path.exists():
                status_content = read_file(status_path)
                if "TOKEN:** OLLAMA" in status_content:  # Only when asleep
                    run_brain_tick()
            last_brain = now

        # Lauds (Morning intention at 6am)
        if now - last_lauds > 3600:
            if datetime.datetime.now().hour == 6:
                run_brain_lauds()
                last_lauds = now + 7200  # Skip ahead

        # Every cycle - check outbox and signals
        check_outbox()

        # Check for Sext (checkpoint) - BEFORE Vespers
        if (BASE_DIR / ".sext").exists():
            run_sext_routine()  # Manual Sext - sends notification
        
        # Auto-Sext: every hour if session active (silent)
        status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
        if status_path.exists():
            status_content = read_file(status_path)
            if "TOKEN:** CLAUDE" in status_content:
                time_since_sext = time.time() - state["last_sext_time"]
                if time_since_sext >= 3600:  # 1 hour
                    run_sext_routine(silent=True)

        # Check for Vespers Signal
        if (BASE_DIR / ".vespers").exists():
            run_vespers_routine()

        time.sleep(10)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--consolidate":
        weekly_consolidation(force=True)
    else:
        try:
            main()
        except KeyboardInterrupt:
            log("Watchman stopping.")
            send_alert("🔴 Virgil Stopping.", tags=["red_circle"])
        except Exception as e:
            log(f"CRITICAL ERROR: {e}")
            subprocess.run(["say", "-v", "Daniel", "System anomaly"], stderr=subprocess.DEVNULL)
            try:
                requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=f"🔥 CRITICAL: Watchman crashed. {e}".encode('utf-8'))
            except: pass
