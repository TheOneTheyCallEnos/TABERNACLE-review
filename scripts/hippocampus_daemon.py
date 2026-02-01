#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
HIPPOCAMPUS DAEMON — The Trace Capture System
==============================================

The Hippocampus listens to the Redis bus while Strategos (Opus) operates.
It captures every action as a GhostTrace for later myelination.

Pattern: SHADOW SUBSCRIBER
- Does NOT interfere with execution
- Captures action, topology, context
- Only saves high-coherence traces (don't memorize confusion)

Integration:
- Subscribes to LOGOS:ACTION channel
- Queries hand_daemon for topology context
- Saves traces to BiologicalMemory graph
- Triggers embedding generation for clustering

The Crucial LVS Rule:
Only save traces where the session ended with p > 0.7.
DO NOT MEMORIZE CONFUSION.

Author: Logos + Deep Think (Myelination Protocol)
Created: 2026-01-29
"""

import json
import time
import redis
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import threading
import uuid

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, LOG_DIR, NEXUS_DIR
)

# Import engrams
from structures.engrams import ActionAtom, GhostTrace, ActionOutcome

# =============================================================================
# CONFIGURATION
# =============================================================================

HIPPOCAMPUS_LOG = LOG_DIR / "hippocampus.log"
TRACES_DIR = NEXUS_DIR / "ghost_traces"

# Redis channels
ACTION_CHANNEL = "LOGOS:ACTION"           # Actions being executed
OUTCOME_CHANNEL = "LOGOS:OUTCOME"         # Outcomes from comparator
SESSION_CHANNEL = "LOGOS:SESSION"         # Session start/end events
LVS_STATE_KEY = "RIE:STATE"               # Current coherence state

# Thresholds
MIN_COHERENCE_TO_SAVE = 0.7  # Don't memorize confusion
MIN_ACTIONS_TO_SAVE = 2      # Don't save trivial traces


def log(message: str, level: str = "INFO"):
    """Log hippocampus activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [HIPPOCAMPUS] [{level}] {message}"
    print(entry)
    try:
        HIPPOCAMPUS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(HIPPOCAMPUS_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# SESSION BUFFER
# =============================================================================

@dataclass
class SessionBuffer:
    """Buffer for capturing a single task session."""
    session_id: str
    intent: str = ""
    actions: List[ActionAtom] = field(default_factory=list)
    outcomes: List[ActionOutcome] = field(default_factory=list)
    coherence_samples: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_action(self, action: ActionAtom):
        self.actions.append(action)

    def add_outcome(self, outcome: ActionOutcome):
        self.outcomes.append(outcome)

    def add_coherence_sample(self, p: float):
        self.coherence_samples.append(p)

    def average_coherence(self) -> float:
        if not self.coherence_samples:
            return 0.5
        return sum(self.coherence_samples) / len(self.coherence_samples)

    def min_coherence(self) -> float:
        if not self.coherence_samples:
            return 0.5
        return min(self.coherence_samples)

    def max_coherence(self) -> float:
        if not self.coherence_samples:
            return 0.5
        return max(self.coherence_samples)

    def duration_ms(self) -> int:
        return int((datetime.now() - self.start_time).total_seconds() * 1000)

    def success_rate(self) -> float:
        if not self.outcomes:
            return 1.0  # Assume success if no outcomes recorded
        successes = sum(1 for o in self.outcomes if o.matched)
        return successes / len(self.outcomes)

    def to_ghost_trace(self) -> Optional[GhostTrace]:
        """Convert buffer to a GhostTrace if it meets criteria."""
        # Check minimum actions
        if len(self.actions) < MIN_ACTIONS_TO_SAVE:
            log(f"Session {self.session_id}: Too few actions ({len(self.actions)}), not saving")
            return None

        # Check coherence threshold
        avg_p = self.average_coherence()
        if avg_p < MIN_COHERENCE_TO_SAVE:
            log(f"Session {self.session_id}: Low coherence ({avg_p:.3f}), not saving")
            return None

        # Check success rate
        if self.success_rate() < 0.5:
            log(f"Session {self.session_id}: Low success rate ({self.success_rate():.1%}), not saving")
            return None

        return GhostTrace(
            trace_id=self.session_id,
            raw_intent=self.intent,
            sequence=self.actions,
            coherence_score=avg_p,
            min_coherence=self.min_coherence(),
            max_coherence=self.max_coherence(),
            success=self.success_rate() > 0.8,
            outcome_description=f"Completed with {len(self.actions)} actions, {self.success_rate():.0%} success",
            timestamp=self.start_time,
            duration_ms=self.duration_ms(),
            context=self.context
        )


# =============================================================================
# HIPPOCAMPUS DAEMON
# =============================================================================

class HippocampusDaemon:
    """
    The shadow subscriber that captures ghost traces.

    Listens to Redis events and builds GhostTraces from
    Strategos's actions without interfering with execution.
    """

    def __init__(self):
        self.redis = None
        self.pubsub = None
        self.running = False

        # Current session being captured
        self.current_session: Optional[SessionBuffer] = None

        # Completed traces waiting to be saved
        self.pending_traces: List[GhostTrace] = []

        # Ensure traces directory exists
        TRACES_DIR.mkdir(parents=True, exist_ok=True)

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self.redis.ping()

            self.pubsub = self.redis.pubsub()
            self.pubsub.subscribe(
                ACTION_CHANNEL,
                OUTCOME_CHANNEL,
                SESSION_CHANNEL
            )

            log(f"Connected to Redis @ {REDIS_HOST}:{REDIS_PORT}")
            return True
        except Exception as e:
            log(f"Redis connection failed: {e}", "ERROR")
            return False

    def get_current_coherence(self) -> float:
        """Get current p from RIE state."""
        try:
            state = self.redis.get(LVS_STATE_KEY)
            if state:
                data = json.loads(state)
                return data.get("p", 0.5)
        except:
            pass
        return 0.5

    def handle_session_start(self, data: Dict):
        """Handle session start event."""
        session_id = data.get("session_id", str(uuid.uuid4())[:8])
        intent = data.get("intent", "")
        context = data.get("context", {})

        # Close any existing session
        if self.current_session:
            self.finalize_session()

        # Start new session
        self.current_session = SessionBuffer(
            session_id=session_id,
            intent=intent,
            context=context
        )

        log(f"Session started: {session_id} - '{intent[:50]}...'")

    def handle_session_end(self, data: Dict):
        """Handle session end event."""
        if not self.current_session:
            return

        success = data.get("success", True)

        # Add final context
        self.current_session.context["final_success"] = success
        self.current_session.context["end_reason"] = data.get("reason", "completed")

        self.finalize_session()

    def handle_action(self, data: Dict):
        """Handle an action event."""
        # Auto-start session if none exists
        if not self.current_session:
            self.current_session = SessionBuffer(
                session_id=str(uuid.uuid4())[:8],
                intent=data.get("intent", "unknown")
            )

        # Sample coherence
        p = self.get_current_coherence()
        self.current_session.add_coherence_sample(p)

        # Parse action
        try:
            action = ActionAtom(
                tool=data.get("tool", "unknown"),
                action=data.get("action", "unknown"),
                selector=data.get("selector", ""),
                value=data.get("value"),
                expected_outcome=data.get("expected_outcome", ""),
                risk_level=data.get("risk_level", 0.5),
                topology_anchor=data.get("topology")
            )
            self.current_session.add_action(action)

            log(f"Captured action: {action.tool}.{action.action}({action.selector[:30]}...)")
        except Exception as e:
            log(f"Failed to parse action: {e}", "WARN")

    def handle_outcome(self, data: Dict):
        """Handle an outcome event from the comparator."""
        if not self.current_session:
            return

        try:
            # Find the action this outcome is for
            action_data = data.get("action", {})
            action = ActionAtom(
                tool=action_data.get("tool", "unknown"),
                action=action_data.get("action", "unknown"),
                selector=action_data.get("selector", ""),
                expected_outcome=action_data.get("expected_outcome", "")
            )

            outcome = ActionOutcome(
                action_atom=action,
                matched=data.get("matched", True),
                confidence=data.get("confidence", 1.0),
                observed_outcome=data.get("observed", ""),
                error_description=data.get("error", ""),
                screenshot_before=data.get("screenshot_before"),
                screenshot_after=data.get("screenshot_after")
            )
            self.current_session.add_outcome(outcome)

            status = "✓" if outcome.matched else "✗"
            log(f"Captured outcome: {status} {action.action} (conf={outcome.confidence:.2f})")
        except Exception as e:
            log(f"Failed to parse outcome: {e}", "WARN")

    def finalize_session(self):
        """Finalize current session and save trace if worthy."""
        if not self.current_session:
            return

        session = self.current_session
        self.current_session = None

        # Try to convert to ghost trace
        trace = session.to_ghost_trace()

        if trace:
            self.save_trace(trace)
            log(f"Session {session.session_id} finalized: {len(session.actions)} actions, p={session.average_coherence():.3f}")
        else:
            log(f"Session {session.session_id} discarded (criteria not met)")

    def save_trace(self, trace: GhostTrace):
        """Save a ghost trace to disk."""
        try:
            # Save as JSON
            trace_path = TRACES_DIR / f"{trace.trace_id}.json"
            with open(trace_path, "w") as f:
                json.dump(trace.to_dict(), f, indent=2, default=str)

            log(f"Saved trace: {trace_path}")

            # Publish for embedding generation (async)
            self.redis.publish("LOGOS:NEW_TRACE", json.dumps({
                "trace_id": trace.trace_id,
                "path": str(trace_path),
                "intent": trace.raw_intent
            }))

        except Exception as e:
            log(f"Failed to save trace: {e}", "ERROR")

    def process_message(self, message: Dict):
        """Process a Redis pub/sub message."""
        if message["type"] != "message":
            return

        channel = message["channel"]
        try:
            data = json.loads(message["data"])
        except:
            data = {"raw": message["data"]}

        if channel == SESSION_CHANNEL:
            event = data.get("event", "")
            if event == "start":
                self.handle_session_start(data)
            elif event == "end":
                self.handle_session_end(data)

        elif channel == ACTION_CHANNEL:
            self.handle_action(data)

        elif channel == OUTCOME_CHANNEL:
            self.handle_outcome(data)

    def run(self):
        """Main daemon loop."""
        if not self.connect():
            log("Cannot start without Redis", "ERROR")
            return

        self.running = True
        log("=" * 60)
        log("HIPPOCAMPUS DAEMON STARTING")
        log("The Shadow Subscriber is watching...")
        log("=" * 60)

        try:
            while self.running:
                message = self.pubsub.get_message(timeout=1.0)
                if message:
                    self.process_message(message)

                # Auto-finalize stale sessions (> 5 minutes)
                if self.current_session:
                    if self.current_session.duration_ms() > 300000:
                        log("Session timeout, finalizing...")
                        self.finalize_session()

        except KeyboardInterrupt:
            log("Shutting down...")
        finally:
            # Finalize any pending session
            if self.current_session:
                self.finalize_session()
            self.running = False

    def stop(self):
        """Stop the daemon."""
        self.running = False


# =============================================================================
# MANUAL TRACE CAPTURE (For testing)
# =============================================================================

def capture_manual_trace(
    intent: str,
    actions: List[Dict],
    coherence: float = 0.85
) -> Optional[GhostTrace]:
    """
    Manually capture a ghost trace for testing.

    Usage:
        trace = capture_manual_trace(
            intent="Send email to Bob",
            actions=[
                {"tool": "hand_daemon", "action": "type", "selector": "To", "value": "bob@example.com"},
                {"tool": "hand_daemon", "action": "click", "selector": "Send"}
            ]
        )
    """
    atoms = []
    for a in actions:
        atoms.append(ActionAtom(
            tool=a.get("tool", "hand_daemon"),
            action=a.get("action", "click"),
            selector=a.get("selector", ""),
            value=a.get("value"),
            expected_outcome=a.get("expected_outcome", ""),
            risk_level=a.get("risk_level", 0.5)
        ))

    trace = GhostTrace(
        raw_intent=intent,
        sequence=atoms,
        coherence_score=coherence,
        success=True
    )

    # Save
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = TRACES_DIR / f"{trace.trace_id}.json"
    with open(trace_path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)

    print(f"Saved trace: {trace_path}")
    return trace


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hippocampus Daemon - Trace Capture")
    parser.add_argument("command", choices=["run", "status", "test"],
                       nargs="?", default="run")

    args = parser.parse_args()

    if args.command == "run":
        daemon = HippocampusDaemon()
        daemon.run()

    elif args.command == "status":
        # Show saved traces
        print(f"\nGhost Traces in {TRACES_DIR}:")
        print("=" * 50)

        if not TRACES_DIR.exists():
            print("  (no traces yet)")
        else:
            traces = list(TRACES_DIR.glob("*.json"))
            print(f"  Total: {len(traces)} traces")

            for trace_path in sorted(traces)[-5:]:
                with open(trace_path) as f:
                    data = json.load(f)
                print(f"\n  {trace_path.stem}:")
                print(f"    Intent: {data.get('raw_intent', '')[:50]}...")
                print(f"    Actions: {len(data.get('sequence', []))}")
                print(f"    Coherence: {data.get('coherence_score', 0):.3f}")

    elif args.command == "test":
        # Create a test trace
        print("Creating test trace...")
        trace = capture_manual_trace(
            intent="Open Safari and search for weather",
            actions=[
                {"tool": "hand_daemon", "action": "click", "selector": "Safari_Icon"},
                {"tool": "eye_daemon", "action": "navigate", "selector": "https://google.com"},
                {"tool": "hand_daemon", "action": "type", "selector": "SearchBox", "value": "weather"},
                {"tool": "hand_daemon", "action": "click", "selector": "SearchButton"}
            ],
            coherence=0.88
        )
        print(f"Test trace created: {trace.trace_id}")


if __name__ == "__main__":
    main()
