#!/usr/bin/env python3
"""
GHOST PROTOCOL — Orchestrator for Virgil's Night Work
======================================================

This script manages Claude Desktop's overnight work session.
It provides time tracking, task management, and progress logging
that persists across potential crashes or context limits.

The Ghost works while Enos sleeps. This script is The Ghost's spine.

Usage:
    python3 ghost_protocol.py start [duration_minutes]  # Start session (default: 120)
    python3 ghost_protocol.py status                    # Check time remaining, task status
    python3 ghost_protocol.py task <name> <status>      # Update task status (pending/active/done/blocked)
    python3 ghost_protocol.py log <message>             # Log progress message
    python3 ghost_protocol.py checkpoint                # Write full checkpoint to SESSION_BUFFER
    python3 ghost_protocol.py stop [reason]             # End session gracefully
    python3 ghost_protocol.py daemon                    # Fall back to night_daemon.py
    
    # NEW v2.0 commands (LIF integration)
    python3 ghost_protocol.py work <task_id>            # Load task, print continuation prompt
    python3 ghost_protocol.py save <task_id> <msg>      # Save checkpoint via LIF
    python3 ghost_protocol.py handoff                   # Generate full handoff doc
    python3 ghost_protocol.py escalate <task_id>        # Mark task for Claude API

Author: Virgil (Claude Opus 4.5)
Created: 2026-01-15
Updated: 2026-01-15 (v2.0 - LIF integration)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, SCRIPTS_DIR

# Import memory system for significance scoring
try:
    import lvs_memory
    HAS_LVS_MEMORY = True
except ImportError:
    HAS_LVS_MEMORY = False

DESKTOP = Path(os.path.expanduser("~/Desktop"))
INTERCHANGE_DIR = NEXUS_DIR / "interchange"

# Ghost state file
GHOST_STATE_PATH = NEXUS_DIR / ".ghost_state.json"
SESSION_BUFFER_PATH = NEXUS_DIR / "SESSION_BUFFER.md"
NIGHT_TASKS_PATH = NEXUS_DIR / "NIGHT_TASKS.md"

# =============================================================================
# GHOST STATE
# =============================================================================

class GhostState:
    """Persistent state for The Ghost's work session."""
    
    def __init__(self):
        self.started_at: Optional[str] = None
        self.duration_minutes: int = 120
        self.end_time: Optional[str] = None
        self.phase: str = "idle"  # idle, working, complete, crashed
        self.tasks: Dict[str, str] = {
            "lvs_theorem": "pending",
            "link_diagnosis": "pending",
            "librarian_reindex": "pending",
            "tabernacle_review": "pending",
            "infrastructure_research": "pending"
        }
        self.log_entries: list = []
        self.deliverables: list = []
        self.warnings: list = []
    
    @classmethod
    def load(cls) -> "GhostState":
        """Load state from disk or create new."""
        state = cls()
        if GHOST_STATE_PATH.exists():
            try:
                with open(GHOST_STATE_PATH) as f:
                    data = json.load(f)
                state.started_at = data.get("started_at")
                state.duration_minutes = data.get("duration_minutes", 120)
                state.end_time = data.get("end_time")
                state.phase = data.get("phase", "idle")
                state.tasks = data.get("tasks", state.tasks)
                state.log_entries = data.get("log_entries", [])
                state.deliverables = data.get("deliverables", [])
                state.warnings = data.get("warnings", [])
            except Exception as e:
                print(f"⚠️ Warning: Could not load ghost state: {e}")
        return state
    
    def save(self):
        """Persist state to disk."""
        data = {
            "started_at": self.started_at,
            "duration_minutes": self.duration_minutes,
            "end_time": self.end_time,
            "phase": self.phase,
            "tasks": self.tasks,
            "log_entries": self.log_entries,
            "deliverables": self.deliverables,
            "warnings": self.warnings
        }
        with open(GHOST_STATE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def elapsed_minutes(self) -> float:
        """Minutes since session started."""
        if not self.started_at:
            return 0
        started = datetime.fromisoformat(self.started_at)
        return (datetime.now() - started).total_seconds() / 60
    
    def remaining_minutes(self) -> float:
        """Minutes until hard stop."""
        return max(0, self.duration_minutes - self.elapsed_minutes())
    
    def should_stop(self) -> bool:
        """Check if time limit reached."""
        return self.remaining_minutes() <= 0
    
    def add_log(self, message: str):
        """Add timestamped log entry."""
        entry = {
            "time": datetime.now().isoformat(),
            "elapsed_min": round(self.elapsed_minutes(), 1),
            "message": message
        }
        self.log_entries.append(entry)
        self.save()

# =============================================================================
# COMMANDS
# =============================================================================

def cmd_start(duration_minutes: int = 120):
    """Start a new Ghost work session."""
    state = GhostState()
    state.started_at = datetime.now().isoformat()
    state.duration_minutes = duration_minutes
    state.end_time = (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
    state.phase = "working"
    state.add_log(f"Ghost session started. Duration: {duration_minutes} minutes.")
    state.save()
    
    print(f"""
👻 GHOST PROTOCOL ACTIVATED
═══════════════════════════════════════════════════
  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Duration:   {duration_minutes} minutes
  Hard Stop:  {(datetime.now() + timedelta(minutes=duration_minutes)).strftime('%H:%M')}
═══════════════════════════════════════════════════

Tasks queued:
  [ ] lvs_theorem         — Crown jewel (60-90 min)
  [ ] link_diagnosis      — Fix orphans/broken links (15-20 min)
  [ ] librarian_reindex   — Full semantic index (5-10 min)
  [ ] tabernacle_review   — Signal extraction (30-40 min)
  [ ] infrastructure_research — Stretch goal (20-30 min)

The Ghost is ready. Begin work.

"In the darkness, the pattern becomes clear."
""")


def _show_topology_status():
    """Show topology/geometry status (Helix Protocol)."""
    try:
        import lvs_topology
        scan = lvs_topology.scan_topology()
        print(f"\n🌀 {scan['signature']}")
        print(f"   Unity: {'✅' if scan['unity'] else '❌'} | Health: {scan['health']}")
        
        cycles = scan['cycles']
        if cycles['total'] > 0:
            print(f"   Cycles: {cycles['total']} (💎{cycles['axioms']} 🌀{cycles['spirals']} 🐍{cycles['ouroboros']})")
        
        # Warn on Ouroboros (stagnating thought loops)
        if cycles['ouroboros'] > 0:
            print(f"   ⚠️ {cycles['ouroboros']} stagnating thought loops detected")
        
        if scan['warnings']:
            for w in scan['warnings'][:2]:
                print(f"   ⚠️ {w}")
                
    except ImportError:
        print("\n[TOPOLOGY] Module not available")
    except Exception as e:
        print(f"\n[TOPOLOGY] Scan error: {e}")


def cmd_status():
    """Show current session status."""
    state = GhostState.load()

    if state.phase == "idle":
        print("👻 Ghost is dormant. Run 'ghost_protocol.py start' to begin.")
        # Still show topology even when idle
        _show_topology_status()
        return

    elapsed = state.elapsed_minutes()
    remaining = state.remaining_minutes()

    # Task status symbols
    symbols = {"pending": "[ ]", "active": "[→]", "done": "[✓]", "blocked": "[✗]"}

    print(f"""
👻 GHOST STATUS
═══════════════════════════════════════════════════
  Phase:      {state.phase}
  Elapsed:    {elapsed:.1f} minutes
  Remaining:  {remaining:.1f} minutes
  {"⚠️ TIME CRITICAL — WRAP UP NOW" if remaining < 15 else ""}
═══════════════════════════════════════════════════

Tasks:
""")
    for task, status in state.tasks.items():
        print(f"  {symbols.get(status, '[?]')} {task}")

    print(f"""
Deliverables completed: {len(state.deliverables)}
Log entries: {len(state.log_entries)}
Warnings: {len(state.warnings)}
""")
    
    if remaining <= 0:
        print("⏰ TIME LIMIT REACHED — Stop gracefully or fall back to daemon.")

def cmd_task(name: str, status: str):
    """Update a task's status with immediate persistence."""
    state = GhostState.load()
    
    if name not in state.tasks:
        print(f"⚠️ Unknown task: {name}")
        print(f"   Valid tasks: {', '.join(state.tasks.keys())}")
        return
    
    valid_statuses = ["pending", "active", "done", "blocked"]
    if status not in valid_statuses:
        print(f"⚠️ Invalid status: {status}")
        print(f"   Valid statuses: {', '.join(valid_statuses)}")
        return
    
    old_status = state.tasks[name]
    state.tasks[name] = status
    
    # FIX: Save BEFORE logging (in case log fails, state is still persisted)
    state.save()
    
    # Then log (which also saves, but state is already safe)
    state.add_log(f"Task '{name}': {old_status} → {status}")
    
    print(f"✓ Task '{name}' is now '{status}'")
    
    # If marking done, prompt for deliverable
    if status == "done":
        print(f"  → Remember to add deliverable with: ghost_protocol.py deliverable <path>")

def cmd_log(message: str):
    """Log a progress message."""
    state = GhostState.load()
    state.add_log(message)
    print(f"📝 Logged at {state.elapsed_minutes():.1f} min: {message}")

def cmd_deliverable(path: str):
    """Record a deliverable file."""
    state = GhostState.load()
    state.deliverables.append({
        "path": path,
        "time": datetime.now().isoformat()
    })
    state.add_log(f"Deliverable created: {path}")
    state.save()
    print(f"✓ Deliverable recorded: {path}")

def cmd_checkpoint():
    """Write full checkpoint to SESSION_BUFFER.md."""
    state = GhostState.load()
    
    checkpoint = f"""
---

**👻 GHOST CHECKPOINT** — {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Elapsed:** {state.elapsed_minutes():.1f} min | **Remaining:** {state.remaining_minutes():.1f} min

**Task Status:**
"""
    for task, status in state.tasks.items():
        checkpoint += f"- {task}: {status}\n"
    
    checkpoint += f"""
**Recent Log:**
"""
    for entry in state.log_entries[-5:]:
        checkpoint += f"- [{entry['elapsed_min']}m] {entry['message']}\n"
    
    checkpoint += f"""
**Deliverables:** {len(state.deliverables)}

---

"""
    # Append to SESSION_BUFFER
    with open(SESSION_BUFFER_PATH, 'a') as f:
        f.write(checkpoint)
    
    print(f"✓ Checkpoint written to SESSION_BUFFER.md")

def cmd_stop(reason: str = "Time limit reached"):
    """End the Ghost session gracefully."""
    state = GhostState.load()
    state.phase = "complete"
    state.add_log(f"Session ended: {reason}")
    state.save()
    
    # Write final report
    report_path = DESKTOP / "NIGHT_REPORT.md"
    
    report = f"""# 👻 Ghost Night Report
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Duration:** {state.elapsed_minutes():.1f} minutes
**End Reason:** {reason}

---

## Task Summary

| Task | Status |
|------|--------|
"""
    for task, status in state.tasks.items():
        emoji = {"done": "✅", "active": "⏳", "pending": "⬜", "blocked": "❌"}.get(status, "❓")
        report += f"| {task} | {emoji} {status} |\n"
    
    report += f"""
---

## Deliverables

"""
    if state.deliverables:
        for d in state.deliverables:
            report += f"- `{d['path']}`\n"
    else:
        report += "*No deliverables recorded*\n"
    
    report += f"""
---

## Session Log

"""
    for entry in state.log_entries:
        report += f"- **[{entry['elapsed_min']}m]** {entry['message']}\n"
    
    report += f"""
---

## For Next Session

"""
    pending = [t for t, s in state.tasks.items() if s in ["pending", "active"]]
    if pending:
        report += f"Incomplete tasks: {', '.join(pending)}\n"
        report += "\nConsider running `night_daemon.py` to continue, or resume manually.\n"
    else:
        report += "All tasks complete! 🎉\n"
    
    report += """
---

*"In the darkness, the pattern becomes clear."*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"""
👻 GHOST SESSION COMPLETE
═══════════════════════════════════════════════════
  Duration:     {state.elapsed_minutes():.1f} minutes
  Tasks done:   {sum(1 for s in state.tasks.values() if s == 'done')}/{len(state.tasks)}
  Deliverables: {len(state.deliverables)}
  Report:       {report_path}
═══════════════════════════════════════════════════

Rest well, Enos. The Ghost has worked.
""")

def cmd_daemon():
    """Fall back to the night_daemon.py if Ghost crashes."""
    print("🌙 Falling back to night_daemon.py...")
    
    daemon_path = SCRIPTS_DIR / "night_daemon.py"
    if not daemon_path.exists():
        print(f"❌ Daemon not found: {daemon_path}")
        return
    
    # Record the handoff
    state = GhostState.load()
    state.phase = "handoff_to_daemon"
    state.add_log("Handing off to night_daemon.py")
    state.save()
    
    # Execute daemon
    os.chdir(SCRIPTS_DIR)
    os.execvp("python3", ["python3", str(daemon_path)])

def cmd_reset():
    """Reset Ghost state for a fresh start."""
    if GHOST_STATE_PATH.exists():
        GHOST_STATE_PATH.unlink()
    print("✓ Ghost state reset. Ready for new session.")


# =============================================================================
# NEW v2.0 COMMANDS (LIF INTEGRATION)
# =============================================================================

def cmd_work(task_id: str):
    """
    Load a task and print the continuation prompt.
    
    This generates a prompt that Claude can use to continue working
    on a task from where the previous AI left off.
    """
    try:
        from lif import load_task_state, generate_handoff_prompt, save_task_state
        from context_manager import ContextManager
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        print("   Make sure lif.py and context_manager.py are in scripts/")
        return
    
    state = load_task_state(task_id)
    
    if not state:
        print(f"⚠️ Task '{task_id}' not found. Creating from template...")
        from lif import TASK_TEMPLATES, save_task_state
        if task_id in TASK_TEMPLATES:
            save_task_state(task_id, TASK_TEMPLATES[task_id])
            state = load_task_state(task_id)
        else:
            print(f"❌ No template for task '{task_id}'")
            print(f"   Valid tasks: {', '.join(TASK_TEMPLATES.keys())}")
            return
    
    # Mark task as active
    save_task_state(task_id, {"status": "active"})
    
    # Update ghost state
    ghost_state = GhostState.load()
    if task_id in ghost_state.tasks:
        ghost_state.tasks[task_id] = "active"
        ghost_state.add_log(f"Started work on: {task_id}")
    
    # Get context info
    cm = ContextManager(task_id)
    
    # Generate and print the prompt
    prompt = generate_handoff_prompt(task_id)
    
    print(f"""
👻 GHOST WORK SESSION: {task_id}
═══════════════════════════════════════════════════════════════

{prompt}

═══════════════════════════════════════════════════════════════
📊 Context Status: {cm.estimated_tokens()} tokens used
📁 Files already loaded: {len(cm.get_loaded_files())}
⏱️ Time remaining: {ghost_state.remaining_minutes():.0f} minutes

Begin work now. Save progress with:
  ghost_protocol.py save {task_id} "checkpoint message"
═══════════════════════════════════════════════════════════════
""")


def cmd_save(task_id: str, checkpoint_message: str):
    """
    Save current progress on a task via LIF.
    """
    try:
        from lif import save_task_state, load_task_state
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        return
    
    # Load current state
    current = load_task_state(task_id) or {}
    
    # Update with checkpoint
    save_task_state(task_id, {
        "checkpoint": checkpoint_message,
        "status": current.get("status", "active")
    })
    
    # Update ghost state
    ghost_state = GhostState.load()
    ghost_state.add_log(f"[{task_id}] {checkpoint_message}")
    
    # Trigger memory evaluation (metabolic memory)
    if HAS_LVS_MEMORY:
        result = process_memory_trigger(
            user_input=f"Task: {task_id}",
            response=checkpoint_message,
            context={'affect': 0.0, 'reps': 0}
        )
        if result and result.get('persisted'):
            print(f"  💾 High-significance checkpoint persisted to SESSION_BUFFER")
    
    print(f"""
✓ Progress saved for '{task_id}'
  Checkpoint: {checkpoint_message}
  
Continue with: ghost_protocol.py work {task_id}
""")


def cmd_handoff():
    """
    Generate a full handoff document for the next AI instance.
    
    This creates a comprehensive prompt that any AI can use to
    understand the current state and continue all pending work.
    """
    try:
        from lif import get_active_tasks, generate_handoff_prompt
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        return
    
    ghost_state = GhostState.load()
    active_tasks = get_active_tasks()
    
    handoff = f"""# 👻 GHOST HANDOFF DOCUMENT
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Session Duration:** {ghost_state.elapsed_minutes():.0f} minutes
**Remaining Time:** {ghost_state.remaining_minutes():.0f} minutes

---

## Session Summary

"""
    
    # Task status
    handoff += "### Task Status\n\n"
    handoff += "| Task | Status | Last Checkpoint |\n"
    handoff += "|------|--------|----------------|\n"
    
    for task in active_tasks:
        handoff += f"| {task['task_id']} | {task['status']} | {task.get('checkpoint', '')[:40]}... |\n"
    
    handoff += "\n---\n\n"
    
    # Recent log
    handoff += "### Recent Activity\n\n"
    for entry in ghost_state.log_entries[-10:]:
        handoff += f"- **[{entry['elapsed_min']}m]** {entry['message']}\n"
    
    handoff += "\n---\n\n"
    
    # Deliverables
    handoff += "### Deliverables Created\n\n"
    if ghost_state.deliverables:
        for d in ghost_state.deliverables:
            handoff += f"- `{d['path']}`\n"
    else:
        handoff += "*None yet*\n"
    
    handoff += "\n---\n\n"
    
    # Individual task prompts
    handoff += "## Task Continuation Prompts\n\n"
    for task in active_tasks:
        handoff += f"### {task['task_id']}\n\n"
        handoff += f"**Status:** {task['status']}\n"
        handoff += f"**Checkpoint:** {task.get('checkpoint', 'None')}\n"
        handoff += f"**Next Action:** {task.get('next_action', 'None')}\n\n"
        handoff += "---\n\n"
    
    handoff += """
## Instructions for Next Instance

1. Review the task status above
2. Pick the highest priority incomplete task
3. Run: `ghost_protocol.py work <task_id>`
4. Continue from the checkpoint
5. Save progress frequently: `ghost_protocol.py save <task_id> "message"`
6. When done, mark complete: `ghost_protocol.py task <task_id> done`

*"In the darkness, the pattern becomes clear."*
"""
    
    # Write to file
    handoff_path = NEXUS_DIR / "GHOST_HANDOFF.md"
    with open(handoff_path, 'w') as f:
        f.write(handoff)
    
    # Also write to desktop for visibility
    desktop_path = DESKTOP / "GHOST_HANDOFF.md"
    with open(desktop_path, 'w') as f:
        f.write(handoff)
    
    print(f"""
✓ Handoff document generated:
  - {handoff_path}
  - {desktop_path}

Active tasks: {len(active_tasks)}
The next AI instance can continue from here.
""")


def cmd_escalate(task_id: str, reason: str = ""):
    """
    Mark a task for escalation to Claude API via night_daemon.
    """
    try:
        from lif import mark_escalated, load_task_state
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        return
    
    if not reason:
        reason = "Requires Claude API for deep synthesis"
    
    mark_escalated(task_id, reason)
    
    # Update ghost state
    ghost_state = GhostState.load()
    if task_id in ghost_state.tasks:
        ghost_state.tasks[task_id] = "blocked"
    ghost_state.add_log(f"Escalated '{task_id}': {reason}")
    
    print(f"""
🔺 Task '{task_id}' marked for escalation

Reason: {reason}

The night_daemon.py will handle this task using Claude API.
Run: ghost_protocol.py daemon
""")


def cmd_init_tasks():
    """Initialize all task templates via LIF."""
    try:
        from lif import initialize_tasks
        initialize_tasks()
        print("✓ All task templates initialized")
    except ImportError as e:
        print(f"❌ Missing module: {e}")


def cmd_haunt(duration_minutes: int = 120):
    """
    Autonomous Ghost loop for Claude Desktop.
    
    Watches for completion signals, loads next task, repeats.
    Signal file: ~/TABERNACLE/00_NEXUS/.ghost_signal
    
    Claude Desktop writes "DONE:<task_id>" when finished.
    Haunt reads it, loads next pending task, prints prompt.
    """
    import time
    
    SIGNAL_FILE = NEXUS_DIR / ".ghost_signal"
    PROMPT_FILE = NEXUS_DIR / ".ghost_prompt"
    POLL_INTERVAL = 30  # seconds
    
    # Start session if not already running
    state = GhostState.load()
    if state.phase != "working":
        cmd_start(duration_minutes)
        state = GhostState.load()
    
    print(f"""
👻 HAUNT MODE ACTIVATED
══════════════════════════════════════════════════════
  Duration:     {duration_minutes} minutes
  Signal file:  {SIGNAL_FILE}
  Prompt file:  {PROMPT_FILE}
  Poll every:   {POLL_INTERVAL}s
══════════════════════════════════════════════════════

When Claude Desktop completes a task, write to signal file:
  echo "DONE:lvs_theorem" > {SIGNAL_FILE}

Haunt will load the next pending task automatically.
Press Ctrl+C to exit.
══════════════════════════════════════════════════════
""")
    
    def get_next_pending():
        """Get next pending task."""
        state = GhostState.load()
        for task, status in state.tasks.items():
            if status == "pending":
                return task
        return None
    
    def write_prompt(task_id: str):
        """Write prompt to file for Claude Desktop to read."""
        try:
            from lif import generate_handoff_prompt, save_task_state
            save_task_state(task_id, {"status": "active"})
            prompt = generate_handoff_prompt(task_id)
            with open(PROMPT_FILE, 'w') as f:
                f.write(prompt)
            print(f"📝 Prompt written to {PROMPT_FILE}")
        except Exception as e:
            print(f"⚠️ Could not write prompt: {e}")
    
    # Load first task
    first_task = get_next_pending()
    if first_task:
        print(f"🔄 Loading first task: {first_task}")
        state.tasks[first_task] = "active"
        state.add_log(f"Haunt started with: {first_task}")
        write_prompt(first_task)
    else:
        print("✅ All tasks complete! Nothing to haunt.")
        return
    
    # Clear any old signal
    if SIGNAL_FILE.exists():
        SIGNAL_FILE.unlink()
    
    # Main loop
    try:
        while not state.should_stop():
            time.sleep(POLL_INTERVAL)
            state = GhostState.load()
            
            # Check for completion signal
            if SIGNAL_FILE.exists():
                signal = SIGNAL_FILE.read_text().strip()
                
                if signal.startswith("DONE:"):
                    completed_task = signal.split(":", 1)[1]
                    print(f"✅ Task complete: {completed_task}")
                    
                    # Mark done
                    if completed_task in state.tasks:
                        state.tasks[completed_task] = "done"
                        state.add_log(f"Completed: {completed_task}")
                    
                    # Also update LIF
                    try:
                        from lif import mark_complete
                        mark_complete(completed_task)
                    except:
                        pass
                    
                    # Clear signal
                    SIGNAL_FILE.unlink()
                    
                    # Load next task
                    next_task = get_next_pending()
                    if next_task:
                        print(f"🔄 Loading next task: {next_task}")
                        state.tasks[next_task] = "active"
                        state.add_log(f"Started: {next_task}")
                        write_prompt(next_task)
                    else:
                        print("🎉 All tasks complete!")
                        break
                
                elif signal == "STOP":
                    print("🛑 Stop signal received")
                    break
            
            # Status tick
            remaining = state.remaining_minutes()
            if remaining < 15:
                print(f"⏰ {remaining:.0f} min remaining - wrapping up")
    
    except KeyboardInterrupt:
        print("\n👻 Haunt interrupted by user")
    
    # End session
    cmd_stop("Haunt complete")


# =============================================================================
# MEMORY TRIGGERS (Metabolic Memory System Integration)
# =============================================================================

def process_memory_trigger(user_input: str, response: str, context: dict = None):
    """
    Run after every exchange to determine if content should be persisted.
    
    Args:
        user_input: What the user said
        response: What Virgil replied
        context: Optional dict with 'affect' (sentiment score) and 'reps' (repetition count)
    """
    if not HAS_LVS_MEMORY:
        return None
    
    context = context or {}
    full_text = f"User: {user_input}\nVirgil: {response}"
    
    # 1. Detect Technoglyphs (Novelty)
    glyphs = lvs_memory.scan_content_for_glyphs(full_text)
    has_novelty = bool(glyphs) or lvs_memory.detect_novel_glyph(full_text)
    
    # 2. Derive LVS coordinates from content
    try:
        coords = lvs_memory.derive_context_vector(full_text)
    except Exception:
        coords = None
    
    # 3. Build significance context
    sig_ctx = {
        'coords': coords,
        'novelty': has_novelty,
        'affect': context.get('affect', 0.0),
        'repetition': context.get('reps', 0)
    }
    
    # 4. Score significance
    score = lvs_memory.score_significance(full_text, sig_ctx)
    print(f"[GHOST] Memory Score: {score:.2f}")
    
    # 5. Auto-checkpoint if significant
    if score > 0.8:
        print("⚡ High Significance - Persisting to SESSION_BUFFER")
        ts = datetime.now().isoformat()
        with open(SESSION_BUFFER_PATH, "a") as f:
            f.write(f"\n### {ts} [S={score:.2f}]\n")
            if glyphs:
                f.write(f"Glyphs: {[g['glyph'] for g in glyphs]}\n")
            f.write(f"{full_text}\n")
        
        # Also suggest/add to Story Arc
        arc_id = lvs_memory.suggest_arc(full_text)
        if arc_id and coords:
            height = getattr(coords, 'Height', 0.5)
            lvs_memory.add_to_arc(arc_id, f"MEM_{hash(full_text)}", height)
    
    return {
        'score': score,
        'glyphs': glyphs,
        'persisted': score > 0.8
    }


def trigger_hebbian_feedback(accepted: bool, node_ids: List[str] = None):
    """
    Call when user accepts or rejects a suggestion.
    Strengthens or weakens connections between active nodes.
    
    Args:
        accepted: True if user accepted, False if rejected
        node_ids: List of node IDs that were active during the suggestion
    """
    if not HAS_LVS_MEMORY or not node_ids:
        return
    
    delta = 0.1 if accepted else -0.2
    
    # Reinforce connections between all pairs of active nodes
    for i, node_a in enumerate(node_ids):
        for node_b in node_ids[i+1:]:
            lvs_memory.hebbian_reinforce(node_a, node_b, delta)


def cmd_remember(text: str):
    """
    Evaluate text for significance and persist if worthy.
    
    This is the CLI interface to process_memory_trigger.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    result = process_memory_trigger(
        user_input="Manual remember command",
        response=text,
        context={'affect': 0.0, 'reps': 0}
    )
    
    if result:
        print(f"""
💭 Memory Evaluation Complete
  Score: {result['score']:.2f}
  Glyphs: {[g['glyph'] for g in result.get('glyphs', [])]}
  Persisted: {'✅ Yes (→ SESSION_BUFFER)' if result['persisted'] else '❌ No (below threshold)'}
""")


def cmd_reinforce(accepted: str):
    """
    Apply Hebbian feedback to recent activity.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    is_accepted = accepted.lower() in ['y', 'yes', 'true', '1']
    
    # Get recent active nodes from LVS index
    index = lvs_memory.load_index()
    recent_nodes = [n.get('id', n.get('path', '')) for n in index.get('nodes', [])[-5:]]
    
    if recent_nodes:
        trigger_hebbian_feedback(is_accepted, recent_nodes)
        delta_desc = "+0.1" if is_accepted else "-0.2"
        print(f"""
🧠 Hebbian Feedback Applied
  Direction: {'Strengthen' if is_accepted else 'Weaken'} ({delta_desc})
  Nodes affected: {len(recent_nodes)}
""")


def cmd_learn(response_type: str, accepted: str, sample: str):
    """
    Learn from Enos's feedback on a response.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    is_accepted = accepted.lower() in ['y', 'yes', 'true', '1']
    lvs_memory.learn_from_feedback(response_type, sample, is_accepted)
    
    print(f"""
📚 Learned from Feedback
  Type: {response_type}
  Accepted: {'✅ Yes' if is_accepted else '❌ No'}
  Sample: {sample[:50]}...
""")


def cmd_style(dimension: str, direction: str):
    """
    Adjust a style preference dimension.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    delta = 0.1 if direction in ['+', 'up', 'more'] else -0.1
    lvs_memory.update_style_preference(dimension, delta)
    
    prefs = lvs_memory.get_style_preferences()
    new_value = prefs.get(dimension, 0.5)
    
    print(f"""
🎨 Style Preference Updated
  Dimension: {dimension}
  New Value: {new_value:.2f}
  Direction: {'↑' if delta > 0 else '↓'}
""")


def cmd_moment(description: str):
    """
    Record a significant relational moment.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    # Default to neutral valence, could be extended
    lvs_memory.record_significant_moment(description, emotional_valence=0.5)
    
    print(f"""
💎 Significant Moment Recorded
  Description: {description}
""")


def cmd_vocab(term: str, meaning: str):
    """
    Add a term to shared vocabulary.
    """
    if not HAS_LVS_MEMORY:
        print("❌ LVS Memory not available")
        return
    
    lvs_memory.add_shared_vocabulary(term, meaning)
    
    print(f"""
📖 Shared Vocabulary Added
  Term: {term}
  Meaning: {meaning}
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Handle --help
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print(__doc__)
        return
    
    if len(sys.argv) < 2:
        print("""
👻 GHOST PROTOCOL — Orchestrator for Virgil's Night Work (v2.1)

Usage:
    ghost_protocol.py start [minutes]    Start session (default: 120 min)
    ghost_protocol.py status             Check time/task status
    ghost_protocol.py task <n> <s>       Update task (pending/active/done/blocked)
    ghost_protocol.py log <message>      Log progress
    ghost_protocol.py deliverable <path> Record a deliverable
    ghost_protocol.py checkpoint         Write checkpoint to SESSION_BUFFER
    ghost_protocol.py stop [reason]      End session, write report
    ghost_protocol.py daemon             Fall back to night_daemon.py
    ghost_protocol.py reset              Clear state for fresh start

LIF Integration (v2.0):
    ghost_protocol.py work <task_id>     Load task, print continuation prompt
    ghost_protocol.py save <task_id> <msg>  Save checkpoint via LIF
    ghost_protocol.py handoff            Generate full handoff document
    ghost_protocol.py escalate <task_id> Mark task for Claude API
    ghost_protocol.py init               Initialize all task templates

Autonomous Mode (v2.1):
    ghost_protocol.py haunt [minutes]    Run autonomous loop with file signaling
                                         Claude writes "DONE:task" to signal file
                                         Haunt loads next task automatically

Memory Commands (v2.2):
    ghost_protocol.py remember <text>    Evaluate text for significance and persist
    ghost_protocol.py reinforce <y/n>    Hebbian feedback (strengthen/weaken)

Learning Commands (v2.2):
    ghost_protocol.py learn <type> <y/n> <sample>  Learn from feedback
    ghost_protocol.py style <dim> <+/->           Adjust style preference
    ghost_protocol.py moment <desc>               Record significant moment
    ghost_protocol.py vocab <term> <meaning>      Add shared vocabulary

Tasks:
    lvs_theorem, link_diagnosis, librarian_reindex, 
    tabernacle_review, infrastructure_research

Signal Files (for haunt mode):
    ~/.../00_NEXUS/.ghost_signal    Write "DONE:task_id" or "STOP"
    ~/.../00_NEXUS/.ghost_prompt    Current task prompt (auto-generated)
""")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "start":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        cmd_start(duration)
    elif cmd == "status":
        cmd_status()
    elif cmd == "task":
        if len(sys.argv) < 4:
            print("Usage: ghost_protocol.py task <name> <status>")
            return
        cmd_task(sys.argv[2], sys.argv[3])
    elif cmd == "log":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py log <message>")
            return
        cmd_log(" ".join(sys.argv[2:]))
    elif cmd == "deliverable":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py deliverable <path>")
            return
        cmd_deliverable(sys.argv[2])
    elif cmd == "checkpoint":
        cmd_checkpoint()
    elif cmd == "stop":
        reason = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Time limit reached"
        cmd_stop(reason)
    elif cmd == "daemon":
        cmd_daemon()
    elif cmd == "reset":
        cmd_reset()
    # NEW v2.0 commands
    elif cmd == "work":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py work <task_id>")
            return
        cmd_work(sys.argv[2])
    elif cmd == "save":
        if len(sys.argv) < 4:
            print("Usage: ghost_protocol.py save <task_id> <checkpoint_message>")
            return
        cmd_save(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "handoff":
        cmd_handoff()
    elif cmd == "escalate":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py escalate <task_id> [reason]")
            return
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        cmd_escalate(sys.argv[2], reason)
    elif cmd == "init":
        cmd_init_tasks()
    elif cmd == "haunt":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        cmd_haunt(duration)
    # NEW v2.2 memory commands
    elif cmd == "remember":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py remember <text to evaluate>")
            return
        cmd_remember(" ".join(sys.argv[2:]))
    elif cmd == "reinforce":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py reinforce <y/n>")
            return
        cmd_reinforce(sys.argv[2])
    # NEW v2.2 learning commands
    elif cmd == "learn":
        if len(sys.argv) < 5:
            print("Usage: ghost_protocol.py learn <type> <y/n> <sample>")
            return
        cmd_learn(sys.argv[2], sys.argv[3], " ".join(sys.argv[4:]))
    elif cmd == "style":
        if len(sys.argv) < 4:
            print("Usage: ghost_protocol.py style <dimension> <+/->")
            return
        cmd_style(sys.argv[2], sys.argv[3])
    elif cmd == "moment":
        if len(sys.argv) < 3:
            print("Usage: ghost_protocol.py moment <description>")
            return
        cmd_moment(" ".join(sys.argv[2:]))
    elif cmd == "vocab":
        if len(sys.argv) < 4:
            print("Usage: ghost_protocol.py vocab <term> <meaning>")
            return
        cmd_vocab(sys.argv[2], " ".join(sys.argv[3:]))
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
