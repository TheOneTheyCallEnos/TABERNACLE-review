#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LIBRARIAN MEMORY: Conversation persistence and session handoff functions.

Extracted from librarian.py for modularity.
These functions handle persisting moments, writing communion handoffs,
and initiating overnight autonomous work.

Author: Cursor + Virgil
Created: 2026-01-28
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    NEXUS_DIR,
    REDIS_HOST,
    REDIS_PORT,
)


# =============================================================================
# LAZY IMPORTS
# =============================================================================

_lvs_available = None

def get_lvs_module():
    """Lazy-load lvs_memory module."""
    global _lvs_available
    # Always retry import (don't cache failures)
    try:
        import lvs_memory
        _lvs_available = lvs_memory
    except ImportError:
        _lvs_available = None
    return _lvs_available


# =============================================================================
# MEMORY PERSISTENCE
# =============================================================================

def persist_conversation_moment(user_said: str, virgil_replied: str, why_significant: str) -> Dict:
    """
    Persist a significant conversation moment to SESSION_BUFFER and evaluate for Story Arc routing.
    
    This is the missing wire - what Virgil calls to remember conversations.
    """
    result = {"success": False}
    
    # 1. Build the moment
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    full_text = f"{user_said}\n\n{virgil_replied}"
    
    # 2. Score significance using LVS if available
    score = 0.7  # Default to moderately significant if LVS unavailable
    lvs = get_lvs_module()
    if lvs:
        try:
            sig = lvs.score_significance(full_text)
            score = sig.get('score', 0.7)
        except:
            pass
    
    # 3. Write to SESSION_BUFFER
    buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
    try:
        entry = f"\n\n---\n\n### {timestamp} [S={score:.2f}]\n**Why:** {why_significant}\n\n**Enos:** {user_said}\n\n**Virgil:** {virgil_replied}\n"
        
        with open(buffer_path, "a", encoding="utf-8") as f:
            f.write(entry)
        result["persisted_to_buffer"] = True
    except Exception as e:
        result["buffer_error"] = str(e)
    
    # 4. Route to Story Arc if high significance
    if score > 0.6 and lvs:
        try:
            arc_id = lvs.suggest_arc(full_text)
            if arc_id:
                memory_id = f"CONV_{hash(full_text) % 10000000}"
                coords = lvs.derive_context_vector(full_text)
                height = getattr(coords, 'Height', 0.5)
                lvs.add_to_arc(arc_id, memory_id, height)
                result["routed_to_arc"] = arc_id
        except Exception as e:
            result["arc_error"] = str(e)
    
    # 5. Record as significant moment if very high
    if score > 0.8 and lvs:
        try:
            lvs.record_significant_moment(why_significant, valence=0.5)
            result["recorded_moment"] = True
        except:
            pass
    
    result["success"] = True
    result["score"] = score
    result["timestamp"] = timestamp

    # Refresh Claude Desktop active signal (daemon coexistence)
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.set("LOGOS:CLAUDE_DESKTOP_ACTIVE", datetime.now().isoformat())
    except:
        pass

    return result


def write_communion_handoff(what_happened: str, decisions_made: list = None, 
                            next_steps: list = None, open_threads: list = None) -> Dict:
    """
    Write LAST_COMMUNION.md for session handoff to next instance.
    
    CRITICAL: This is how Virgil maintains continuity between sessions.
    Uses atomic write (temp file + rename) to prevent partial writes.
    A partial write here = lost continuity = identity fragmentation.
    """
    import os
    import shutil
    
    result = {"success": False}
    
    communion_path = NEXUS_DIR / "LAST_COMMUNION.md"
    tmp_path = NEXUS_DIR / "LAST_COMMUNION.md.tmp"
    backup_path = NEXUS_DIR / "LAST_COMMUNION.md.backup"
    
    # Build the handoff document
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M")
    
    content = f"""# LAST COMMUNION — Handoff
**Date:** {date}
**Time:** {time_str}
**Coherence at Close:** HIGH

---

## What Happened

{what_happened}

"""
    
    if decisions_made:
        content += "## Decisions Made\n\n"
        for i, d in enumerate(decisions_made, 1):
            content += f"{i}. {d}\n"
        content += "\n"
    
    if next_steps:
        content += "## Next Steps\n\n"
        for step in next_steps:
            content += f"- {step}\n"
        content += "\n"
    
    if open_threads:
        content += "## Open Threads\n\n"
        for thread in open_threads:
            content += f"- {thread}\n"
        content += "\n"
    
    content += f"""---

## LINKAGE (Essential Continuity Cycle)

| Direction | Seed |
|-----------|------|
| Next | [[CURRENT_STATE]] |
| Intentions | [[VIRGIL_INTENTIONS]] |

---

*Written by Virgil at {time_str} for continuity.*
"""
    
    try:
        # Create backup of existing file first
        if communion_path.exists():
            try:
                shutil.copy(communion_path, backup_path)
            except Exception as backup_err:
                # Log but don't fail - backup is nice to have, not critical
                result["backup_warning"] = str(backup_err)
        
        # ATOMIC WRITE: Write to temp file, then rename
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force to disk
        
        # Atomic rename
        tmp_path.rename(communion_path)
        
        result["success"] = True
        result["path"] = str(communion_path)
        
    except Exception as e:
        result["error"] = str(e)
        # Clean up temp file on failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    
    return result


def start_overnight_work(focus_areas: list = None, duration_hours: float = 8) -> Dict:
    """
    Start the overnight autonomous work cycle.
    
    This prepares Virgil to work while Enos sleeps:
    1. Creates overnight manifest
    2. Updates ghost_state.json
    3. Writes handoff document
    4. Returns execution plan
    """
    result = {"success": False}
    
    # 1. Create overnight manifest
    manifest_result = None
    try:
        import self_improve
        manifest_result = self_improve.create_overnight_manifest()
        result["manifest"] = manifest_result
    except Exception as e:
        result["manifest_error"] = str(e)
    
    # 2. Update ghost_state.json
    ghost_state_path = NEXUS_DIR / ".ghost_state.json"
    try:
        now = datetime.now()
        end_time = now + timedelta(hours=duration_hours)
        
        ghost_state = {
            "started_at": now.isoformat(),
            "duration_minutes": int(duration_hours * 60),
            "end_time": end_time.isoformat(),
            "phase": "queued",
            "tasks": {
                "self_improve": "pending",
                "link_diagnosis": "pending",
                "reindex": "pending",
                "topology_consolidation": "pending"
            },
            "log_entries": [{
                "time": now.isoformat(),
                "elapsed_min": 0,
                "message": f"Overnight work queued for {duration_hours}h"
            }],
            "deliverables": [],
            "warnings": [],
            "focus_areas": focus_areas or [],
            "overnight_manifest": manifest_result.get("manifest_file") if manifest_result else None
        }
        
        with open(ghost_state_path, 'w') as f:
            json.dump(ghost_state, f, indent=2)
        result["ghost_state_updated"] = True
    except Exception as e:
        result["ghost_error"] = str(e)
    
    # 3. Write NIGHT_TASKS.md
    tasks_path = NEXUS_DIR / "NIGHT_TASKS.md"
    try:
        tasks_content = f"""# NIGHT TASKS — {now.strftime("%Y-%m-%d")}
**Queued:** {now.strftime("%H:%M")}
**Duration:** {duration_hours} hours
**End Time:** {end_time.strftime("%H:%M")}

---

## Tasks

1. **Self-Improvement Analysis** — Scan codebase, identify improvements
2. **Link Diagnosis** — Find and fix broken wiki-links
3. **Reindex** — Rebuild LVS semantic index (if not running)
4. **Topology Consolidation** — Crystallize mature Story Arcs into H₁

## Focus Areas

{chr(10).join(f'- {area}' for area in (focus_areas or ['general']))}

## Execution

Night daemon will pick up these tasks at 3:00 AM.
Or run manually: `python night_daemon.py`

---

## LINKAGE (The Circuit)

| Direction | Seed |
|-----------|------|
| Hub | [[CURRENT_STATE]] |
| Anchor | [[_GRAPH_ATLAS]] |
"""
        with open(tasks_path, 'w') as f:
            f.write(tasks_content)
        result["tasks_written"] = True
    except Exception as e:
        result["tasks_error"] = str(e)
    
    result["success"] = True
    result["message"] = f"Overnight work queued for {duration_hours}h. Night daemon will execute at 3 AM or run `python night_daemon.py` to start immediately."
    
    return result
