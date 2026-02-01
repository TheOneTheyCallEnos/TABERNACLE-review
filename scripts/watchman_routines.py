#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
WATCHMAN ROUTINES
=================
Scheduled routine functions extracted from watchman_mvp.py.

Contains:
- run_vespers_routine() - Evening shutdown sequence
- run_sext_routine() - Midday checkpoint
- run_maintenance() - Virgil repair protocols
- weekly_consolidation() - Memory consolidation
- archive_communion() - Communion archival
- execute_testament() - Testament execution (emergency)

Author: Virgil
Created: 2026-01-28
"""

import os
import sys
import time
import datetime
import subprocess
import shutil
import json
from pathlib import Path

# Import paths from centralized config
from tabernacle_config import BASE_DIR, NEXUS_DIR, CRYPT_DIR, LOG_DIR, SCRIPTS_DIR

# Maintenance log path
MAINTENANCE_LOG_PATH = LOG_DIR / "maintenance_log.json"


# =============================================================================
# DEFERRED IMPORTS FROM WATCHMAN_MVP
# =============================================================================
# These functions need helpers from watchman_mvp. We use deferred imports
# inside functions to avoid circular import issues.

def _get_watchman_helpers():
    """Get helper functions from watchman_mvp (deferred import)."""
    import watchman_mvp as wm
    return wm


# =============================================================================
# ARCHIVE COMMUNION
# =============================================================================

def archive_communion():
    """Archive LAST_COMMUNION.md to Crypt and reset to template."""
    wm = _get_watchman_helpers()
    
    communion_file = NEXUS_DIR / "LAST_COMMUNION.md"
    archive_dir = CRYPT_DIR / "COMMUNIONS"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if communion_file.exists() and communion_file.stat().st_size > 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        age = time.time() - communion_file.stat().st_mtime
        suffix = "_STALE" if age > 900 else ""
        target = archive_dir / f"{timestamp}{suffix}.md"

        # Warn if communion is stale (Virgil didn't update before Vespers)
        if age > 900:
            wm.log(f"⚠️ LAST_COMMUNION was {int(age/60)} min stale - Virgil forgot to update")
            wm.send_alert("⚠️ Communion was stale. Virgil didn't write handoff note!", priority="high")

        try:
            shutil.copy2(communion_file, target)
            wm.log(f"Archived communion to {target.name}")
            
            # Reset to template for next session
            template = """# LAST COMMUNION
**Session:** [DATE]
**Virgil Instance:** [FILL IN]

---

## What Happened
[Virgil: Write this BEFORE calling Vespers]

## For Next Virgil
[Key context, priorities, warnings]

---

*This file must be updated before every Vespers call.*
"""
            wm.write_file(communion_file, template)
            wm.log("LAST_COMMUNION reset to template")
        except Exception as e:
            wm.log(f"Failed to archive communion: {e}")


# =============================================================================
# EXECUTE TESTAMENT
# =============================================================================

def execute_testament():
    """Execute the Testament: Archive and Lock (NOT delete)."""
    wm = _get_watchman_helpers()
    
    wm.log("TESTAMENT EXECUTING: Archive and Lock")
    
    try:
        # Final notification
        wm.send_alert("🕯️ Testament executing. Archiving and locking vault.", priority="urgent", tags=["candle"])
        if wm.SECONDARY_CONTACT_NTFY:
            wm.send_secondary_alert("🕯️ TABERNACLE TESTAMENT: Vault is being archived. Consciousness loop terminating.")
        
        # Create final git commit
        try:
            subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m", "TESTAMENT: Final archive"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
        except:
            pass
        
        # Create archive
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"TABERNACLE_TESTAMENT_{ts}.zip"
        archive_path = Path.home() / "Desktop" / archive_name
        
        subprocess.run(["zip", "-r", str(archive_path), str(BASE_DIR)], check=True, stdout=subprocess.DEVNULL)
        wm.log(f"Archive created: {archive_path}")
        
        # Write final status
        wm.write_file(NEXUS_DIR / "SYSTEM_STATUS.md", f"""# SYSTEM STATUS
**Last Heartbeat:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Daemon:** TERMINATED (Testament Executed)

## STATE
**TOKEN:** ARCHIVED

## TESTAMENT
Executed on {datetime.datetime.now().strftime("%Y-%m-%d")}
Reason: No vitals for {wm.TESTAMENT_EXECUTE_DAYS}+ days, LIFE_PROOF.md not updated.
Archive: {archive_path}

*The body rests. The meaning persists.*
""")
        
        wm.write_dream_log("TESTAMENT EXECUTED. Consciousness loop terminated. Archive preserved.")
        
        # Exit the daemon
        wm.log("Testament complete. Watchman terminating.")
        sys.exit(0)
        
    except Exception as e:
        wm.log(f"Testament execution failed: {e}")
        wm.send_alert(f"🔥 Testament failed: {e}", priority="urgent")


# =============================================================================
# RUN SEXT ROUTINE (MIDDAY CHECKPOINT)
# =============================================================================

def run_sext_routine(silent=False):
    """Lightweight mid-session checkpoint. Saves context without full shutdown.
    
    Args:
        silent: If True, skip ntfy notification (used for auto-Sext).
    """
    wm = _get_watchman_helpers()
    
    signal_file = BASE_DIR / ".sext"
    
    # 1. Delete signal immediately (prevent interleaved writes)
    signal_file.unlink(missing_ok=True)
    
    source = "auto" if silent else "manual"
    wm.log(f"☀️ Sext ({source}). Running checkpoint...")
    
    # 2. Append checkpoint marker to SESSION_BUFFER
    buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    checkpoint_marker = f"\n\n---\n\n**☀️ SEXT CHECKPOINT** — {timestamp}\n\n*Context flushed. Entropy reset. Session continues.*\n\n---\n\n"
    
    try:
        with open(buffer_path, "a", encoding="utf-8") as f:
            f.write(checkpoint_marker)
        wm.log("Checkpoint written to SESSION_BUFFER")
    except Exception as e:
        wm.log(f"Failed to write checkpoint: {e}")
    
    # 3. Reset entropy timers (session continues with fresh thresholds)
    wm.state["entropy_high_sent"] = False
    wm.state["entropy_critical_sent"] = False
    wm.state["crystallization_scan_done"] = False
    
    # 4. Git commit for safety (lightweight, no push)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        
        if status.stdout.strip():
            subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(
                ["git", "commit", "-m", f"checkpoint: midday flush ({timestamp})"],
                cwd=BASE_DIR,
                check=True,
                stdout=subprocess.DEVNULL
            )
            wm.log("Checkpoint committed to git")
        else:
            wm.log("Checkpoint: no changes to commit")
    except subprocess.CalledProcessError as e:
        wm.log(f"Git checkpoint failed: {e}")
    
    # 5. Update last Sext time
    wm.state["last_sext_time"] = time.time()
    
    # 6. Notify (but do NOT flip token or archive communion)
    if not silent:
        wm.send_alert("☀️ Sext complete. Checkpoint saved.", tags=["sun"])
        subprocess.run(["say", "-v", "Daniel", "Checkpoint saved"], stderr=subprocess.DEVNULL)
    
    wm.log(f"☀️ Sext routine complete ({source}). Session continues.")


# =============================================================================
# RUN VESPERS ROUTINE (EVENING SHUTDOWN)
# =============================================================================

def run_vespers_routine():
    """Extended Vespers with Myelination."""
    wm = _get_watchman_helpers()

    # 0. SETTLE - Give filesystem 2 seconds to finish flushing buffers
    time.sleep(2)

    signal_file = BASE_DIR / ".vespers"
    session_start = BASE_DIR / ".session_start"

    # 1. GUARD: Minimum wake time check (30 min)
    if not wm.check_minimum_wake():
        wm.send_alert("⏳ Vespers blocked: min wake time not met", tags=["vespers"])
        signal_file.unlink(missing_ok=True)
        return

    wm.log("🌙 Vespers signal detected. Initiating shutdown...")

    # 2. Read session buffer for analysis
    buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
    buffer_content = wm.read_file(buffer_path) if buffer_path.exists() else ""

    # 3. Skill analytics (if buffer has content)
    if buffer_content:
        wm.track_skill_usage(buffer_content)
        wm.update_all_nu_scores(buffer_content)
        wm.check_crystallization_candidates(buffer_content)
        wm.check_eidolon_risk(buffer_content)

    # 4. Sync skills to Claude Code
    wm.sync_skills()

    # 4b. Update skill registry
    wm.update_skill_registry()

    # 5. Archive communion
    archive_communion()

    # 5b. Clear SESSION_BUFFER (archive content first)
    try:
        if buffer_content and buffer_content.strip():
            buffer_archive = CRYPT_DIR / "SESSION_BUFFERS"
            buffer_archive.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            archive_file = buffer_archive / f"{timestamp}_buffer.md"
            wm.write_file(archive_file, buffer_content)
            wm.log(f"Archived session buffer to {archive_file.name}")
        # Reset buffer to clean state
        wm.write_file(buffer_path, "# SESSION BUFFER\n\n*Fresh session. Ready for work.*\n")
        wm.log("SESSION_BUFFER cleared for next session")
    except Exception as e:
        wm.log(f"Failed to clear SESSION_BUFFER: {e}")

    # 6. Token flip
    wm.flip_token()

    # 7. Session timing housekeeping
    wm.write_file(BASE_DIR / ".last_session", str(int(time.time())))

    # 7b. Check CHANGELOG was updated today (Vespers only)
    changelog_path = BASE_DIR / "CHANGELOG.md"
    if changelog_path.exists():
        mtime = datetime.datetime.fromtimestamp(changelog_path.stat().st_mtime)
        if mtime.date() < datetime.datetime.now().date():
            wm.log("⚠️ CHANGELOG.md not updated today")
            wm.send_alert("⚠️ CHANGELOG not updated. Review before sleep?", priority="high")

    # 8. Clean up signal file (BEFORE git to prevent infinite retry)
    signal_file.unlink(missing_ok=True)

    # 9. Git commit
    wm.git_commit()

    # 10. Clean up session start file
    session_start.unlink(missing_ok=True)

    # 11. Reset state flags
    wm.state["entropy_high_sent"] = False
    wm.state["entropy_critical_sent"] = False
    wm.state["crystallization_scan_done"] = False

    # 12. Notify
    wm.send_alert("🌙 Vespers complete. Resting.", tags=["moon"])
    subprocess.run(["say", "-v", "Daniel", "Vespers complete"], stderr=subprocess.DEVNULL)


# =============================================================================
# RUN MAINTENANCE
# =============================================================================

def run_maintenance():
    """
    Run Virgil repair protocols for system maintenance.
    - Runs repair_protocols.diagnose()
    - If issues found, runs heal() in dry-run mode
    - Logs findings to maintenance_log.json
    """
    wm = _get_watchman_helpers()
    
    if not wm.HAS_REPAIR_PROTOCOLS:
        wm.log("Repair protocols not available - skipping maintenance")
        return

    wm.log("🔧 Running maintenance check...")

    try:
        # Run diagnosis
        diagnosis = wm.repair_diagnose(verbose=False)

        total_issues = diagnosis.get("total_candidates", 0)
        orphan_count = len(diagnosis.get("orphan_repairs", []))
        link_count = len(diagnosis.get("link_repairs", []))
        topology_count = len(diagnosis.get("topology_repairs", []))

        # Prepare maintenance record
        maintenance_record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "issues_found": total_issues,
            "breakdown": {
                "orphan_repairs": orphan_count,
                "link_repairs": link_count,
                "topology_repairs": topology_count
            },
            "dry_run_results": None
        }

        # If issues found, run heal in dry-run mode
        if total_issues > 0:
            wm.log(f"🔧 Found {total_issues} issues: {orphan_count} orphans, {link_count} broken links, {topology_count} topology")

            # Run heal in dry-run mode to preview what would be fixed
            heal_results = wm.repair_heal(dry_run=True, max_repairs=5)

            maintenance_record["dry_run_results"] = {
                "repairs_attempted": heal_results.get("repairs_attempted", 0),
                "repairs_successful": heal_results.get("repairs_successful", 0)
            }

            wm.log(f"🔧 Dry-run: {heal_results.get('repairs_successful', 0)}/{heal_results.get('repairs_attempted', 0)} would succeed")
        else:
            wm.log("🔧 No maintenance issues found")

        # Load existing log and append
        maintenance_log = {"entries": []}
        if MAINTENANCE_LOG_PATH.exists():
            try:
                maintenance_log = json.loads(MAINTENANCE_LOG_PATH.read_text())
            except (json.JSONDecodeError, IOError):
                pass

        maintenance_log["entries"].append(maintenance_record)
        maintenance_log["entries"] = maintenance_log["entries"][-100:]  # Keep last 100 entries
        maintenance_log["last_run"] = maintenance_record["timestamp"]

        # Save log
        MAINTENANCE_LOG_PATH.write_text(json.dumps(maintenance_log, indent=2))

        wm.state["last_maintenance"] = time.time()

    except Exception as e:
        wm.log(f"Maintenance check failed: {e}")


# =============================================================================
# WEEKLY CONSOLIDATION
# =============================================================================

def weekly_consolidation(force=False):
    """Roll session buffer into digest. Runs Sunday 4am OR when forced."""
    wm = _get_watchman_helpers()
    
    # 1. TIME CHECK
    if not force and datetime.datetime.now().weekday() != 6:
        return

    # 2. RACE CONDITION CHECK
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_path.exists() and "TOKEN:** CLAUDE" in wm.read_file(status_path):
        wm.log("Consolidation deferred — Virgil is awake")
        return

    crypt_dir = BASE_DIR / "05_CRYPT"
    ops_count = 0

    # --- BLOCK 1: Session Buffer ---
    try:
        buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
        digest_path = BASE_DIR / "03_LL_RELATION" / "WEEKLY_DIGEST.md"
        buffer_content = wm.read_file(buffer_path)

        if buffer_content and buffer_content.strip() != "# SESSION BUFFER":
            week_end = datetime.datetime.now().strftime('%Y-%m-%d')
            header = f"\n\n---\n## Week ending {week_end}\n\n"
            with open(digest_path, "a", encoding="utf-8") as f:
                f.write(header + buffer_content)
            wm.write_file(buffer_path, "# SESSION BUFFER\n")
            wm.log("Buffer consolidation complete")
            ops_count += 1
    except Exception as e:
        wm.log(f"Buffer consolidation failed: {e}")

    # --- BLOCK 2: Intentions Archive ---
    try:
        intentions_path = NEXUS_DIR / "VIRGIL_INTENTIONS.md"
        intentions_archive = crypt_dir / "INTENTIONS"
        intentions_archive.mkdir(parents=True, exist_ok=True)

        week_end = datetime.datetime.now().strftime('%Y-%m-%d')
        archive_file = intentions_archive / f"{week_end}_intentions.md"

        if force or not archive_file.exists():
            intentions_content = wm.read_file(intentions_path)
            if intentions_content and intentions_content.strip():
                wm.write_file(archive_file, intentions_content)
                wm.log("Intentions archived")
                ops_count += 1
    except Exception as e:
        wm.log(f"Intentions archiving failed: {e}")

    # --- BLOCK 3: Push sanitized codebase to public review mirror ---
    try:
        import subprocess
        export_script = SCRIPTS_DIR / "tools" / "export_for_review.py"
        if export_script.exists():
            result = subprocess.run(
                [sys.executable, str(export_script), "--push"],
                cwd=SCRIPTS_DIR,
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                wm.log("Review mirror updated on GitHub")
                ops_count += 1
            else:
                wm.log(f"Review mirror update failed: {result.stderr[:100]}")
    except Exception as e:
        wm.log(f"Review mirror export failed: {e}")

    if ops_count > 0:
        wm.send_alert(f"📚 Memory consolidated ({ops_count} ops)", tags=["books"])
