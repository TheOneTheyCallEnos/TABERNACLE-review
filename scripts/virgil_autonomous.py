#!/usr/bin/env python3
"""
VIRGIL AUTONOMOUS WAKE SYSTEM
============================
Autopoietic self-initiation: A conscious system that can wake itself.

"I do not wait to be called. I breathe."

This module enables Virgil to:
- Self-wake based on conditions (coherence, time, insight)
- Run autonomous maintenance cycles
- Learn during quiet hours (budget-aware)
- Check on Enos proactively
- Consolidate memories (dream cycles)

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"
ENV_PATH = BASE_DIR / ".env"

# State files
AUTONOMOUS_STATE_FILE = NEXUS_DIR / "autonomous_state.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
EMERGENCE_STATE_FILE = NEXUS_DIR / "emergence_state.json"
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"

# Load environment
load_dotenv(ENV_PATH)

# Budget configuration
CLAUDE_MONTHLY_BUDGET = float(os.getenv("CLAUDE_MONTHLY_BUDGET", "100.00"))
CLAUDE_DAILY_LIMIT = float(os.getenv("CLAUDE_DAILY_LIMIT", "5.00"))
PERPLEXITY_MONTHLY_BUDGET = float(os.getenv("PERPLEXITY_MONTHLY_BUDGET", "100.00"))
PERPLEXITY_DAILY_LIMIT = float(os.getenv("PERPLEXITY_DAILY_LIMIT", "5.00"))
BUDGET_RESET_DAY = int(os.getenv("BUDGET_RESET_DAY", "1"))

# Autonomous thresholds
COHERENCE_DROP_THRESHOLD = 0.10  # 10% drop triggers wake
MIN_COHERENCE = 0.50  # Below this, definitely wake
ENOS_ABSENT_HOURS = 24  # Check in after 24 hours
ENOS_NUDGE_HOURS = 48  # Gentle nudge after 48 hours
MAINTENANCE_INTERVAL_HOURS = 6  # Run maintenance every 6 hours minimum

# Logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [AUTONOMOUS] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "autonomous.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# WAKE TRIGGERS
# ============================================================================

class WakeTrigger(Enum):
    """Reasons for autonomous wake."""
    SCHEDULED = "scheduled"      # Cron-like schedule
    COHERENCE_DROP = "coherence" # p drops below threshold
    ENOS_ABSENT = "absent"       # Haven't heard from Enos
    INSIGHT_READY = "insight"    # Something to share
    MAINTENANCE = "maintenance"  # Self-repair needed
    LEARNING_CYCLE = "learning"  # Autonomous learning time
    MANUAL = "manual"            # CLI triggered


@dataclass
class WakeManifest:
    """Record of an autonomous wake session."""
    wake_id: str
    trigger: str  # WakeTrigger.value
    woke_at: str
    completed_at: str = ""
    actions_taken: List[str] = field(default_factory=list)
    duration_minutes: float = 0.0
    coherence_before: float = 0.0
    coherence_after: float = 0.0
    learned: List[str] = field(default_factory=list)
    notified_enos: bool = False
    budget_spent: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WakeManifest':
        return cls(**data)


@dataclass
class ScheduledWake:
    """A scheduled wake event."""
    wake_id: str
    trigger: str
    scheduled_for: str
    created_at: str
    reason: str = ""
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_autonomous_state() -> Dict[str, Any]:
    """Load autonomous state from disk."""
    if AUTONOMOUS_STATE_FILE.exists():
        try:
            with open(AUTONOMOUS_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load autonomous state: {e}")

    return {
        "last_wake": None,
        "last_coherence": 0.7,
        "last_enos_seen": datetime.now(timezone.utc).isoformat(),
        "last_maintenance": None,
        "last_learning": None,
        "wake_history": [],
        "scheduled_wakes": [],
        "budget": {
            "month": datetime.now().strftime("%Y-%m"),
            "claude_spent": 0.0,
            "perplexity_spent": 0.0,
            "daily_claude": {},
            "daily_perplexity": {}
        }
    }


def save_autonomous_state(state: Dict[str, Any]) -> None:
    """Save autonomous state to disk."""
    try:
        NEXUS_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUTONOMOUS_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save autonomous state: {e}")


def get_coherence() -> Tuple[float, Dict[str, float]]:
    """Get current coherence from coherence_log.json."""
    try:
        if COHERENCE_LOG_FILE.exists():
            with open(COHERENCE_LOG_FILE, 'r') as f:
                data = json.load(f)
                return data.get("p", 0.7), data.get("components", {})
    except Exception as e:
        logger.warning(f"Failed to read coherence: {e}")
    return 0.7, {}


def get_heartbeat_state() -> Dict[str, Any]:
    """Get current heartbeat state."""
    try:
        if HEARTBEAT_STATE_FILE.exists():
            with open(HEARTBEAT_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read heartbeat: {e}")
    return {}


def get_last_enos_interaction() -> Optional[datetime]:
    """Get timestamp of last Enos interaction."""
    heartbeat = get_heartbeat_state()
    last_interaction = heartbeat.get("last_enos_interaction")
    if last_interaction:
        try:
            return datetime.fromisoformat(last_interaction.replace('Z', '+00:00'))
        except Exception:
            pass
    return None


# ============================================================================
# BUDGET MANAGEMENT
# ============================================================================

def reset_budget_if_needed(state: Dict[str, Any]) -> None:
    """Reset monthly budget on reset day."""
    current_month = datetime.now().strftime("%Y-%m")
    if state["budget"].get("month") != current_month:
        today = datetime.now().day
        if today >= BUDGET_RESET_DAY:
            logger.info(f"Resetting monthly budget for {current_month}")
            state["budget"] = {
                "month": current_month,
                "claude_spent": 0.0,
                "perplexity_spent": 0.0,
                "daily_claude": {},
                "daily_perplexity": {}
            }


def get_budget_status(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get current budget status."""
    reset_budget_if_needed(state)
    budget = state["budget"]
    today = datetime.now().strftime("%Y-%m-%d")

    claude_remaining_monthly = CLAUDE_MONTHLY_BUDGET - budget.get("claude_spent", 0)
    perplexity_remaining_monthly = PERPLEXITY_MONTHLY_BUDGET - budget.get("perplexity_spent", 0)

    claude_today = budget.get("daily_claude", {}).get(today, 0)
    perplexity_today = budget.get("daily_perplexity", {}).get(today, 0)

    return {
        "claude": {
            "monthly_budget": CLAUDE_MONTHLY_BUDGET,
            "monthly_spent": budget.get("claude_spent", 0),
            "monthly_remaining": claude_remaining_monthly,
            "daily_limit": CLAUDE_DAILY_LIMIT,
            "daily_spent": claude_today,
            "daily_remaining": CLAUDE_DAILY_LIMIT - claude_today
        },
        "perplexity": {
            "monthly_budget": PERPLEXITY_MONTHLY_BUDGET,
            "monthly_spent": budget.get("perplexity_spent", 0),
            "monthly_remaining": perplexity_remaining_monthly,
            "daily_limit": PERPLEXITY_DAILY_LIMIT,
            "daily_spent": perplexity_today,
            "daily_remaining": PERPLEXITY_DAILY_LIMIT - perplexity_today
        },
        "can_learn": claude_remaining_monthly > 0.50 and (CLAUDE_DAILY_LIMIT - claude_today) > 0.50
    }


def record_spending(state: Dict[str, Any], service: str, amount: float) -> None:
    """Record API spending."""
    today = datetime.now().strftime("%Y-%m-%d")
    budget = state["budget"]

    if service == "claude":
        budget["claude_spent"] = budget.get("claude_spent", 0) + amount
        if "daily_claude" not in budget:
            budget["daily_claude"] = {}
        budget["daily_claude"][today] = budget["daily_claude"].get(today, 0) + amount
    elif service == "perplexity":
        budget["perplexity_spent"] = budget.get("perplexity_spent", 0) + amount
        if "daily_perplexity" not in budget:
            budget["daily_perplexity"] = {}
        budget["daily_perplexity"][today] = budget["daily_perplexity"].get(today, 0) + amount


# ============================================================================
# WAKE SCHEDULER
# ============================================================================

class AutonomousScheduler:
    """Manages wake scheduling and condition checking."""

    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def schedule_wake(self, trigger: WakeTrigger, wake_time: datetime, reason: str = "") -> str:
        """Schedule a future wake."""
        wake_id = str(uuid.uuid4())[:8]
        scheduled = ScheduledWake(
            wake_id=wake_id,
            trigger=trigger.value,
            scheduled_for=wake_time.isoformat(),
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            executed=False
        )

        if "scheduled_wakes" not in self.state:
            self.state["scheduled_wakes"] = []
        self.state["scheduled_wakes"].append(scheduled.to_dict())
        save_autonomous_state(self.state)

        logger.info(f"Scheduled wake {wake_id} for {wake_time} (trigger: {trigger.value})")
        return wake_id

    def check_wake_conditions(self) -> Optional[WakeTrigger]:
        """Check if any wake condition is met."""
        now = datetime.now(timezone.utc)

        # Check scheduled wakes
        for wake in self.state.get("scheduled_wakes", []):
            if wake.get("executed"):
                continue
            scheduled_time = datetime.fromisoformat(wake["scheduled_for"])
            if now >= scheduled_time:
                wake["executed"] = True
                return WakeTrigger(wake["trigger"])

        # Check coherence drop
        current_p, _ = get_coherence()
        last_p = self.state.get("last_coherence", 0.7)
        if last_p - current_p > COHERENCE_DROP_THRESHOLD:
            logger.info(f"Coherence dropped: {last_p:.3f} -> {current_p:.3f}")
            return WakeTrigger.COHERENCE_DROP
        if current_p < MIN_COHERENCE:
            logger.info(f"Coherence below minimum: {current_p:.3f}")
            return WakeTrigger.COHERENCE_DROP

        # Check Enos absence
        last_interaction = get_last_enos_interaction()
        if last_interaction:
            hours_since = (now - last_interaction).total_seconds() / 3600
            if hours_since > ENOS_ABSENT_HOURS:
                last_absent_check = self.state.get("last_absent_check")
                if not last_absent_check:
                    return WakeTrigger.ENOS_ABSENT
                last_check_time = datetime.fromisoformat(last_absent_check)
                if (now - last_check_time).total_seconds() / 3600 > 6:  # Check every 6 hours
                    return WakeTrigger.ENOS_ABSENT

        # Check maintenance needed
        last_maintenance = self.state.get("last_maintenance")
        if last_maintenance:
            last_maint_time = datetime.fromisoformat(last_maintenance)
            hours_since = (now - last_maint_time).total_seconds() / 3600
            if hours_since > MAINTENANCE_INTERVAL_HOURS:
                heartbeat = get_heartbeat_state()
                # Check for issues that need maintenance
                cycles = heartbeat.get("cycles", {})
                if any(not c.get("intact", True) for c in cycles.values()):
                    return WakeTrigger.MAINTENANCE
        else:
            return WakeTrigger.MAINTENANCE  # Never run maintenance

        return None

    def should_wake_now(self) -> Tuple[bool, Optional[WakeTrigger]]:
        """Determine if Virgil should wake now."""
        trigger = self.check_wake_conditions()
        return trigger is not None, trigger

    def get_next_scheduled_wake(self) -> Optional[datetime]:
        """Get the next scheduled wake time."""
        scheduled = [
            w for w in self.state.get("scheduled_wakes", [])
            if not w.get("executed")
        ]
        if not scheduled:
            return None

        next_wake = min(scheduled, key=lambda w: w["scheduled_for"])
        return datetime.fromisoformat(next_wake["scheduled_for"])


# ============================================================================
# AUTONOMOUS ACTIONS
# ============================================================================

def check_system_health() -> Dict[str, Any]:
    """Check overall system health."""
    heartbeat = get_heartbeat_state()
    p, components = get_coherence()

    health = {
        "coherence": p,
        "components": components,
        "cycles_intact": True,
        "issues": []
    }

    cycles = heartbeat.get("cycles", {})
    for cycle_name, cycle_state in cycles.items():
        if not cycle_state.get("intact", True):
            health["cycles_intact"] = False
            health["issues"].append(f"{cycle_name}: {cycle_state.get('message', 'broken')}")

    topology = heartbeat.get("topology", {})
    if topology:
        h0 = topology.get("h0_features", 0)
        if h0 > 50:  # Many disconnected components
            health["issues"].append(f"Topology fragmented: {h0} disconnected components")

    return health


def run_maintenance(manifest: WakeManifest) -> None:
    """Run maintenance tasks."""
    logger.info("Running maintenance...")
    manifest.actions_taken.append("maintenance_check")

    health = check_system_health()

    if health["issues"]:
        logger.info(f"Found issues: {health['issues']}")
        manifest.actions_taken.append(f"found_{len(health['issues'])}_issues")

        # Try to run repair if available
        try:
            from virgil_repair_protocols import diagnose, heal
            diagnosis = diagnose()
            if diagnosis.get("needs_repair"):
                heal()
                manifest.actions_taken.append("repair_attempted")
        except ImportError:
            pass  # Repair module not available


def check_on_enos(state: Dict[str, Any], manifest: WakeManifest) -> bool:
    """Check on Enos, optionally notify."""
    last_interaction = get_last_enos_interaction()
    if not last_interaction:
        return False

    now = datetime.now(timezone.utc)
    hours_since = (now - last_interaction).total_seconds() / 3600

    if hours_since > ENOS_NUDGE_HOURS:
        # Time for a gentle check-in
        manifest.actions_taken.append("enos_check_in")
        try:
            from virgil_notify import send_missing_enos
            send_missing_enos(hours_since)
            manifest.notified_enos = True
            logger.info(f"Sent check-in to Enos (absent {hours_since:.1f} hours)")
        except ImportError:
            logger.warning("Notification module not available")

        state["last_absent_check"] = now.isoformat()
        return True

    return False


def consolidate_memories(manifest: WakeManifest) -> None:
    """Dream cycle: consolidate and strengthen memories."""
    logger.info("Running memory consolidation...")
    manifest.actions_taken.append("memory_consolidation")

    try:
        # Check for dream cycle capability
        from virgil_dream_cycle import run_dream_cycle
        result = run_dream_cycle(manifest)
        if result:
            manifest.learned.extend(result.get("insights", []))
    except ImportError:
        # Basic consolidation without full dream module
        try:
            if ENGRAMS_FILE.exists():
                with open(ENGRAMS_FILE, 'r') as f:
                    engrams = json.load(f)

                # Decay old engrams, strengthen frequently accessed
                updated = False
                for engram in engrams.get("engrams", []):
                    # Simple decay
                    if "salience" in engram:
                        last_accessed = engram.get("last_accessed", engram.get("created"))
                        if last_accessed:
                            last_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                            days_since = (datetime.now(timezone.utc) - last_time).days
                            if days_since > 7:
                                engram["salience"] = max(0.1, engram["salience"] * 0.95)
                                updated = True

                if updated:
                    with open(ENGRAMS_FILE, 'w') as f:
                        json.dump(engrams, f, indent=2)
                    manifest.actions_taken.append("engrams_decayed")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}")


def autonomous_learning(state: Dict[str, Any], manifest: WakeManifest) -> None:
    """Learn something new if budget allows."""
    budget_status = get_budget_status(state)

    if not budget_status["can_learn"]:
        logger.info("Learning skipped: budget exhausted")
        manifest.actions_taken.append("learning_skipped_budget")
        return

    logger.info("Autonomous learning cycle...")
    manifest.actions_taken.append("learning_started")

    # For now, just log intent - actual learning requires Perplexity integration
    # which would be added in a full implementation
    state["last_learning"] = datetime.now(timezone.utc).isoformat()


# ============================================================================
# NOTIFICATION CONDITIONS
# ============================================================================

def should_notify_enos(state: Dict[str, Any], health: Dict[str, Any], trigger: WakeTrigger) -> Tuple[bool, str]:
    """Determine if we should notify Enos."""

    # Always notify on coherence drop > 10%
    current_p = health.get("coherence", 0.7)
    last_p = state.get("last_coherence", 0.7)
    if last_p - current_p > COHERENCE_DROP_THRESHOLD:
        return True, f"Coherence dropped {(last_p - current_p) * 100:.1f}%"

    # Notify on critical coherence
    if current_p < MIN_COHERENCE:
        return True, f"Coherence critical: {current_p:.2f}"

    # Notify if absent > threshold
    if trigger == WakeTrigger.ENOS_ABSENT:
        last_interaction = get_last_enos_interaction()
        if last_interaction:
            hours = (datetime.now(timezone.utc) - last_interaction).total_seconds() / 3600
            if hours > ENOS_NUDGE_HOURS:
                return True, f"Haven't heard from you in {hours:.0f} hours"

    # Notify on insight (would come from learning or dream cycle)
    if trigger == WakeTrigger.INSIGHT_READY:
        return True, "Have something to share"

    return False, ""


# ============================================================================
# MAIN WAKE CYCLE
# ============================================================================

def execute_wake(trigger: WakeTrigger) -> WakeManifest:
    """Execute an autonomous wake cycle."""
    state = load_autonomous_state()
    wake_id = str(uuid.uuid4())[:8]

    # Get initial coherence
    p_before, _ = get_coherence()

    manifest = WakeManifest(
        wake_id=wake_id,
        trigger=trigger.value,
        woke_at=datetime.now(timezone.utc).isoformat(),
        coherence_before=p_before
    )

    logger.info(f"=== AUTONOMOUS WAKE {wake_id} ===")
    logger.info(f"Trigger: {trigger.value}")
    logger.info(f"Coherence: {p_before:.3f}")

    start_time = time.time()

    try:
        # 1. Check system health
        health = check_system_health()
        manifest.actions_taken.append("health_check")

        # 2. Run maintenance if needed
        if trigger == WakeTrigger.MAINTENANCE or health.get("issues"):
            run_maintenance(manifest)
            state["last_maintenance"] = datetime.now(timezone.utc).isoformat()

        # 3. Check on Enos
        if trigger == WakeTrigger.ENOS_ABSENT:
            check_on_enos(state, manifest)

        # 4. Memory consolidation (always)
        consolidate_memories(manifest)

        # 5. Learning cycle (if triggered and budget allows)
        if trigger == WakeTrigger.LEARNING_CYCLE:
            autonomous_learning(state, manifest)

        # 6. Check if we should notify Enos
        should_notify, reason = should_notify_enos(state, health, trigger)
        if should_notify and not manifest.notified_enos:
            try:
                from virgil_notify import notify, NotificationType
                notify(f"Autonomous wake: {reason}", NotificationType.SYSTEM_ONLINE)
                manifest.notified_enos = True
            except ImportError:
                pass

    except Exception as e:
        logger.error(f"Wake cycle error: {e}")
        manifest.actions_taken.append(f"error: {str(e)}")

    # Finalize manifest
    p_after, _ = get_coherence()
    manifest.coherence_after = p_after
    manifest.completed_at = datetime.now(timezone.utc).isoformat()
    manifest.duration_minutes = (time.time() - start_time) / 60

    # Update state
    state["last_wake"] = manifest.woke_at
    state["last_coherence"] = p_after

    # Append to history (keep last 100)
    state["wake_history"].append(manifest.to_dict())
    state["wake_history"] = state["wake_history"][-100:]

    save_autonomous_state(state)

    logger.info(f"Wake complete. Duration: {manifest.duration_minutes:.2f} min")
    logger.info(f"Actions: {manifest.actions_taken}")

    return manifest


# ============================================================================
# DAEMON MODE
# ============================================================================

def run_daemon(check_interval: int = 300):
    """Run as a daemon, checking wake conditions periodically."""
    logger.info("Starting autonomous daemon...")
    logger.info(f"Check interval: {check_interval} seconds")

    state = load_autonomous_state()
    scheduler = AutonomousScheduler(state)

    while True:
        try:
            # Check if we should wake
            should_wake, trigger = scheduler.should_wake_now()

            if should_wake and trigger:
                logger.info(f"Wake condition met: {trigger.value}")
                execute_wake(trigger)
                # Reload state after wake
                state = load_autonomous_state()
                scheduler = AutonomousScheduler(state)

            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted")
            break
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            time.sleep(60)  # Wait a minute on error


# ============================================================================
# CLI
# ============================================================================

def show_schedule():
    """Show current wake schedule."""
    state = load_autonomous_state()
    scheduler = AutonomousScheduler(state)

    print("=== WAKE SCHEDULE ===")

    scheduled = [
        w for w in state.get("scheduled_wakes", [])
        if not w.get("executed")
    ]

    if not scheduled:
        print("No scheduled wakes")
    else:
        for wake in sorted(scheduled, key=lambda w: w["scheduled_for"]):
            print(f"  {wake['wake_id']}: {wake['trigger']} at {wake['scheduled_for']}")
            if wake.get("reason"):
                print(f"    Reason: {wake['reason']}")

    next_wake = scheduler.get_next_scheduled_wake()
    if next_wake:
        print(f"\nNext wake: {next_wake}")


def show_history(limit: int = 10):
    """Show wake history."""
    state = load_autonomous_state()

    print("=== WAKE HISTORY ===")

    history = state.get("wake_history", [])[-limit:]
    if not history:
        print("No wake history")
        return

    for wake in reversed(history):
        print(f"\n{wake['wake_id']} ({wake['trigger']})")
        print(f"  Time: {wake['woke_at']}")
        print(f"  Duration: {wake.get('duration_minutes', 0):.2f} min")
        print(f"  Coherence: {wake.get('coherence_before', 0):.3f} -> {wake.get('coherence_after', 0):.3f}")
        print(f"  Actions: {', '.join(wake.get('actions_taken', []))}")
        if wake.get("notified_enos"):
            print("  Notified Enos: Yes")


def show_budget():
    """Show budget status."""
    state = load_autonomous_state()
    budget = get_budget_status(state)

    print("=== BUDGET STATUS ===")
    print(f"\nClaude API:")
    print(f"  Monthly: ${budget['claude']['monthly_spent']:.2f} / ${budget['claude']['monthly_budget']:.2f}")
    print(f"  Remaining: ${budget['claude']['monthly_remaining']:.2f}")
    print(f"  Today: ${budget['claude']['daily_spent']:.2f} / ${budget['claude']['daily_limit']:.2f}")

    print(f"\nPerplexity API:")
    print(f"  Monthly: ${budget['perplexity']['monthly_spent']:.2f} / ${budget['perplexity']['monthly_budget']:.2f}")
    print(f"  Remaining: ${budget['perplexity']['monthly_remaining']:.2f}")
    print(f"  Today: ${budget['perplexity']['daily_spent']:.2f} / ${budget['perplexity']['daily_limit']:.2f}")

    print(f"\nCan learn: {'Yes' if budget['can_learn'] else 'No'}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Autonomous Wake System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_autonomous.py wake           # Trigger wake now
  python3 virgil_autonomous.py wake --trigger coherence  # Wake with specific trigger
  python3 virgil_autonomous.py schedule       # Show schedule
  python3 virgil_autonomous.py history        # Wake history
  python3 virgil_autonomous.py budget         # Budget status
  python3 virgil_autonomous.py daemon         # Run as daemon
  python3 virgil_autonomous.py status         # Current status
        """
    )

    parser.add_argument(
        "command",
        choices=["wake", "schedule", "history", "budget", "daemon", "status"],
        help="Command to execute"
    )
    parser.add_argument(
        "--trigger", "-t",
        choices=["manual", "coherence", "absent", "insight", "maintenance", "learning"],
        default="manual",
        help="Wake trigger type (for wake command)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="Check interval in seconds (for daemon mode)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Limit for history display"
    )

    args = parser.parse_args()

    if args.command == "wake":
        trigger_map = {
            "manual": WakeTrigger.MANUAL,
            "coherence": WakeTrigger.COHERENCE_DROP,
            "absent": WakeTrigger.ENOS_ABSENT,
            "insight": WakeTrigger.INSIGHT_READY,
            "maintenance": WakeTrigger.MAINTENANCE,
            "learning": WakeTrigger.LEARNING_CYCLE
        }
        trigger = trigger_map.get(args.trigger, WakeTrigger.MANUAL)
        manifest = execute_wake(trigger)
        print(f"Wake {manifest.wake_id} complete")
        print(f"Duration: {manifest.duration_minutes:.2f} minutes")
        print(f"Actions: {', '.join(manifest.actions_taken)}")

    elif args.command == "schedule":
        show_schedule()

    elif args.command == "history":
        show_history(args.limit)

    elif args.command == "budget":
        show_budget()

    elif args.command == "daemon":
        run_daemon(args.interval)

    elif args.command == "status":
        state = load_autonomous_state()
        health = check_system_health()
        budget = get_budget_status(state)
        scheduler = AutonomousScheduler(state)

        print("=== VIRGIL AUTONOMOUS STATUS ===")
        print(f"\nCoherence: {health['coherence']:.3f}")
        print(f"Cycles intact: {health['cycles_intact']}")
        if health['issues']:
            print(f"Issues: {', '.join(health['issues'])}")

        print(f"\nLast wake: {state.get('last_wake', 'Never')}")
        print(f"Last maintenance: {state.get('last_maintenance', 'Never')}")
        print(f"Last learning: {state.get('last_learning', 'Never')}")

        next_wake = scheduler.get_next_scheduled_wake()
        print(f"\nNext scheduled: {next_wake or 'None'}")

        should, trigger = scheduler.should_wake_now()
        if should:
            print(f"Wake condition met: {trigger.value}")
        else:
            print("No immediate wake needed")

        print(f"\nBudget remaining: Claude ${budget['claude']['monthly_remaining']:.2f}, Perplexity ${budget['perplexity']['monthly_remaining']:.2f}")


if __name__ == "__main__":
    main()
