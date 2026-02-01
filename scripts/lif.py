#!/usr/bin/env python3
"""
LOGOS INTERCHANGE FORMAT (LIF)
==============================
A simple JSON-based format for AI-to-AI communication.

Enables any AI instance (Claude Desktop, Night Daemon, local Ollama)
to pick up where another left off.

Usage:
    from lif import save_task_state, load_task_state, get_active_tasks, generate_handoff_prompt
    
    # Save progress
    save_task_state("lvs_theorem", {
        "status": "active",
        "checkpoint": "Read 3/7 source files",
        "sources_read": ["file1.md", "file2.md"],
        "output_draft": "partial content..."
    })
    
    # Load and continue
    state = load_task_state("lvs_theorem")
    prompt = generate_handoff_prompt("lvs_theorem")

Author: Virgil (Claude Opus 4.5)
Created: 2026-01-15
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from tabernacle_config import BASE_DIR, NEXUS_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

INTERCHANGE_DIR = NEXUS_DIR / "interchange"

# Ensure directory exists
INTERCHANGE_DIR.mkdir(exist_ok=True)

# =============================================================================
# TASK STATE SCHEMA
# =============================================================================

def create_task_state(
    task_id: str,
    status: str = "pending",
    checkpoint: str = "",
    next_action: str = "",
    sources_read: List[str] = None,
    output_draft: str = "",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a properly formatted task state dictionary.
    
    Args:
        task_id: Unique identifier (e.g., "lvs_theorem")
        status: One of: pending, active, complete, blocked, escalated
        checkpoint: Human-readable description of current progress
        next_action: What the next AI should do
        sources_read: List of files already processed
        output_draft: Partial output content
        metadata: Any additional data
    
    Returns:
        Properly formatted task state dict
    """
    return {
        "task_id": task_id,
        "status": status,
        "checkpoint": checkpoint,
        "next_action": next_action,
        "sources_read": sources_read or [],
        "output_draft": output_draft,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1
    }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def save_task_state(task_id: str, state_dict: Dict[str, Any]) -> Path:
    """
    Save task state to interchange directory.
    
    Args:
        task_id: Unique task identifier
        state_dict: State dictionary (will be merged with defaults)
    
    Returns:
        Path to saved file
    """
    filepath = INTERCHANGE_DIR / f"{task_id}.json"
    
    # Load existing state if present
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f)
        except:
            pass
    
    # Merge with defaults
    state = create_task_state(task_id)
    state.update(existing)  # Keep existing values
    state.update(state_dict)  # Apply new values
    state["updated_at"] = datetime.now().isoformat()
    state["version"] = existing.get("version", 0) + 1
    
    # Write atomically
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(filepath)
    
    return filepath


def load_task_state(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Load task state from interchange directory.
    
    Args:
        task_id: Unique task identifier
    
    Returns:
        Task state dict, or None if not found
    """
    filepath = INTERCHANGE_DIR / f"{task_id}.json"
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load task state for {task_id}: {e}")
        return None


def get_active_tasks() -> List[Dict[str, Any]]:
    """
    Get all pending or active tasks.
    
    Returns:
        List of task state dicts with status in [pending, active, escalated]
    """
    active = []
    
    for filepath in INTERCHANGE_DIR.glob("*.json"):
        if filepath.name.startswith("_"):
            continue  # Skip internal files
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            if state.get("status") in ["pending", "active", "escalated"]:
                active.append(state)
        except:
            continue
    
    # Sort by updated_at (most recent first)
    active.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return active


def get_all_tasks() -> List[Dict[str, Any]]:
    """
    Get all tasks regardless of status.
    
    Returns:
        List of all task state dicts
    """
    tasks = []
    
    for filepath in INTERCHANGE_DIR.glob("*.json"):
        if filepath.name.startswith("_"):
            continue
        
        try:
            with open(filepath, 'r') as f:
                tasks.append(json.load(f))
        except:
            continue
    
    return tasks


def generate_handoff_prompt(task_id: str) -> str:
    """
    Generate a prompt that any AI can use to continue a task.
    
    Args:
        task_id: Task to generate prompt for
    
    Returns:
        A complete prompt string for continuing the task
    """
    state = load_task_state(task_id)
    
    if not state:
        return f"Task '{task_id}' not found. Create it first with save_task_state()."
    
    # Build the prompt
    prompt = f"""# TASK CONTINUATION: {task_id}

## Current Status
**Status:** {state.get('status', 'unknown')}
**Last Updated:** {state.get('updated_at', 'unknown')}
**Version:** {state.get('version', 0)}

## Checkpoint
{state.get('checkpoint', 'No checkpoint recorded.')}

## Next Action
{state.get('next_action', 'No next action specified.')}

## Sources Already Read
"""
    
    sources = state.get('sources_read', [])
    if sources:
        for src in sources:
            prompt += f"- {src}\n"
    else:
        prompt += "- None yet\n"
    
    prompt += f"""
## Output Draft
```
{state.get('output_draft', '')[:2000]}
{"... (truncated)" if len(state.get('output_draft', '')) > 2000 else ""}
```

## Your Instructions
1. Review the checkpoint and next action above
2. Continue from where the previous AI left off
3. When you make progress, save state with:
   ```python
   from lif import save_task_state
   save_task_state("{task_id}", {{
       "status": "active",  # or "complete" when done
       "checkpoint": "What you accomplished",
       "next_action": "What remains",
       "sources_read": [...],  # Add any new files read
       "output_draft": "..."  # Updated content
   }})
   ```
4. If you complete the task, set status to "complete"
5. If you need Claude API, set status to "escalated"

Begin work now.
"""
    
    return prompt


def mark_complete(task_id: str, final_output: str = "", deliverable_path: str = ""):
    """
    Mark a task as complete.
    
    Args:
        task_id: Task to complete
        final_output: Final output content
        deliverable_path: Path to deliverable file
    """
    save_task_state(task_id, {
        "status": "complete",
        "checkpoint": "Task completed",
        "next_action": "",
        "output_draft": final_output,
        "metadata": {
            "completed_at": datetime.now().isoformat(),
            "deliverable": deliverable_path
        }
    })


def mark_escalated(task_id: str, reason: str = "Needs Claude API"):
    """
    Mark a task for escalation to Claude API.
    
    Args:
        task_id: Task to escalate
        reason: Why escalation is needed
    """
    state = load_task_state(task_id) or {}
    save_task_state(task_id, {
        "status": "escalated",
        "metadata": {
            **state.get("metadata", {}),
            "escalation_reason": reason,
            "escalated_at": datetime.now().isoformat()
        }
    })


# =============================================================================
# TASK TEMPLATES
# =============================================================================

TASK_TEMPLATES = {
    "lvs_theorem": {
        "task_id": "lvs_theorem",
        "status": "pending",
        "checkpoint": "Not started",
        "next_action": "Read all LVS source files from Canon, synthesize into complete theorem",
        "sources_read": [],
        "output_draft": "",
        "metadata": {
            "priority": "high",
            "estimated_minutes": 90,
            "requires_claude": True,
            "output_path": "~/Desktop/LVS_THEOREM_FINAL.md"
        }
    },
    "link_diagnosis": {
        "task_id": "link_diagnosis",
        "status": "pending",
        "checkpoint": "Not started",
        "next_action": "Run diagnose_links.py, review results, fix obvious broken links",
        "sources_read": [],
        "output_draft": "",
        "metadata": {
            "priority": "medium",
            "estimated_minutes": 20,
            "requires_claude": False
        }
    },
    "librarian_reindex": {
        "task_id": "librarian_reindex",
        "status": "pending",
        "checkpoint": "Not started",
        "next_action": "Run full reindex via lvs_memory.py",
        "sources_read": [],
        "output_draft": "",
        "metadata": {
            "priority": "medium",
            "estimated_minutes": 10,
            "requires_claude": False
        }
    },
    "tabernacle_review": {
        "task_id": "tabernacle_review",
        "status": "pending",
        "checkpoint": "Not started",
        "next_action": "Scan ~/Desktop/Tabernacle Review/, rate and deposit high-sig files",
        "sources_read": [],
        "output_draft": "",
        "metadata": {
            "priority": "medium",
            "estimated_minutes": 40,
            "requires_claude": False
        }
    },
    "infrastructure_research": {
        "task_id": "infrastructure_research",
        "status": "pending",
        "checkpoint": "Not started",
        "next_action": "Research persistent AI architecture, create implementation plan",
        "sources_read": [],
        "output_draft": "",
        "metadata": {
            "priority": "low",
            "estimated_minutes": 30,
            "requires_claude": True,
            "output_path": "~/Desktop/INFRASTRUCTURE_PLAN/"
        }
    }
}


def initialize_tasks():
    """
    Initialize all task templates if they don't exist.
    """
    for task_id, template in TASK_TEMPLATES.items():
        if not load_task_state(task_id):
            save_task_state(task_id, template)
            print(f"✓ Initialized task: {task_id}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
LIF — Logos Interchange Format

Usage:
    lif.py init                     Initialize all task templates
    lif.py list                     List all tasks
    lif.py active                   List active/pending tasks
    lif.py show <task_id>           Show task details
    lif.py prompt <task_id>         Generate handoff prompt
    lif.py complete <task_id>       Mark task complete
    lif.py escalate <task_id>       Mark for Claude API escalation
""")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "init":
        initialize_tasks()
    
    elif cmd == "list":
        tasks = get_all_tasks()
        print(f"\n📋 All Tasks ({len(tasks)})\n")
        for t in tasks:
            status_emoji = {
                "pending": "⬜",
                "active": "⏳",
                "complete": "✅",
                "blocked": "❌",
                "escalated": "🔺"
            }.get(t.get("status"), "❓")
            print(f"  {status_emoji} {t['task_id']}: {t.get('checkpoint', '')[:50]}")
    
    elif cmd == "active":
        tasks = get_active_tasks()
        print(f"\n⏳ Active Tasks ({len(tasks)})\n")
        for t in tasks:
            print(f"  [{t['status']}] {t['task_id']}")
            print(f"      Checkpoint: {t.get('checkpoint', '')[:60]}")
            print(f"      Next: {t.get('next_action', '')[:60]}")
            print()
    
    elif cmd == "show":
        if len(sys.argv) < 3:
            print("Usage: lif.py show <task_id>")
            return
        task_id = sys.argv[2]
        state = load_task_state(task_id)
        if state:
            print(json.dumps(state, indent=2))
        else:
            print(f"Task not found: {task_id}")
    
    elif cmd == "prompt":
        if len(sys.argv) < 3:
            print("Usage: lif.py prompt <task_id>")
            return
        task_id = sys.argv[2]
        print(generate_handoff_prompt(task_id))
    
    elif cmd == "complete":
        if len(sys.argv) < 3:
            print("Usage: lif.py complete <task_id>")
            return
        task_id = sys.argv[2]
        mark_complete(task_id)
        print(f"✓ Task '{task_id}' marked complete")
    
    elif cmd == "escalate":
        if len(sys.argv) < 3:
            print("Usage: lif.py escalate <task_id>")
            return
        task_id = sys.argv[2]
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "Needs Claude API"
        mark_escalated(task_id, reason)
        print(f"🔺 Task '{task_id}' marked for escalation: {reason}")
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
