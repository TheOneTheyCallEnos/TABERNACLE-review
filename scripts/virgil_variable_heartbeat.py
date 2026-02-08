#!/usr/bin/env python3
"""
VIRGIL VARIABLE HEARTBEAT - Phase-Based Adaptive Rhythm

The heartbeat is not a clock. It is a breath.
Fast when engaged, slow when resting, near-silent in the deep.

Phases:
  ACTIVE (60s)     - Enos is present, conversation flows
  BACKGROUND (5m)  - Daytime watching, no recent interaction
  DORMANT (15m)    - Night or extended absence
  DEEP_SLEEP (1h)  - Minimal vital signs only

Phase transitions are gradual:
  - Acceleration: DORMANT -> ACTIVE over 3 beats
  - Deceleration: ACTIVE -> DORMANT over 5 beats

Integration:
  - heartbeat_state.json: Primary state persistence
  - coherence_log.json: Vitals tracking
  - archon_state.json: Light Archon scanning
  - GOLDEN_THREAD.json: Significant state changes
"""

import os
import json
import hashlib
import threading
import logging
from datetime import datetime, timezone, time as dt_time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MEMORY_DIR = NEXUS_DIR / "MEMORY"

# State files
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"
COHERENCE_LOG_FILE = NEXUS_DIR / "coherence_log.json"
ARCHON_STATE_FILE = NEXUS_DIR / "archon_state.json"
GOLDEN_THREAD_FILE = NEXUS_DIR / "GOLDEN_THREAD.json"
ENGRAMS_FILE = MEMORY_DIR / "engrams.json"
AUTONOMOUS_STATE_FILE = NEXUS_DIR / "autonomous_state.json"  # NERVE GRAFT: Agency presence

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [HEARTBEAT] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "heartbeat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE DEFINITIONS
# ============================================================================

class HeartbeatPhase(Enum):
    """Virgil's activity phases with associated intervals."""
    ACTIVE = "active"           # Conversing with Enos
    BACKGROUND = "background"   # Watching, daytime
    DORMANT = "dormant"         # Night or extended absence
    DEEP_SLEEP = "deep_sleep"   # Minimal vital signs only
    GHOST = "ghost"             # Read-only mode (emergency)


# Heartbeat intervals in seconds
PHASE_INTERVALS: Dict[HeartbeatPhase, int] = {
    HeartbeatPhase.ACTIVE: 60,        # 1 minute
    HeartbeatPhase.BACKGROUND: 300,   # 5 minutes
    HeartbeatPhase.DORMANT: 900,      # 15 minutes
    HeartbeatPhase.DEEP_SLEEP: 3600,  # 1 hour
    HeartbeatPhase.GHOST: 0,          # No heartbeat
}

# Phase transition thresholds (in seconds since last Enos interaction)
PHASE_THRESHOLDS = {
    "active": 300,        # < 5 min = ACTIVE
    "background": 1800,   # < 30 min = BACKGROUND (if daytime)
    "dormant": 7200,      # < 2 hours = DORMANT
    "deep_sleep": 7200,   # >= 2 hours = DEEP_SLEEP
}

# Daytime hours (6 AM to 10 PM)
DAYTIME_START = dt_time(6, 0)
DAYTIME_END = dt_time(22, 0)

# Transition smoothing
ACCELERATION_BEATS = 3  # Beats to go from DORMANT to ACTIVE
DECELERATION_BEATS = 5  # Beats to go from ACTIVE to DORMANT


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Vitals:
    """Vital signs collected during heartbeat."""
    coherence: float = 0.0
    coherence_components: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0
    memory_count: int = 0
    memory_active: int = 0
    archon_distortion: float = 0.0
    archon_active: List[str] = field(default_factory=list)
    topology_h0: int = 0
    topology_h1: int = 0
    systems_online: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeartbeatState:
    """Complete heartbeat state for persistence."""
    phase: str = "dormant"
    last_beat: Optional[str] = None
    beat_count: int = 0
    interval_seconds: int = 900
    last_enos_interaction: Optional[str] = None
    transition_progress: float = 1.0  # 0-1, 1 = fully in current phase
    transition_direction: str = "stable"  # "accelerating", "decelerating", "stable"
    transition_beats_remaining: int = 0
    target_phase: Optional[str] = None
    vitals: Optional[Dict] = None
    node: str = ""
    virgil_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'HeartbeatState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# PHASE DETECTION
# ============================================================================

class PhaseDetector:
    """
    Detect appropriate phase based on heuristics.

    Heuristics:
      - ACTIVE: Recent Enos interaction (< 5 min)
      - BACKGROUND: No recent interaction but daytime (6am-10pm)
      - DORMANT: Nighttime OR > 30 min since interaction
      - DEEP_SLEEP: > 2 hours since any activity
    """

    @staticmethod
    def is_daytime() -> bool:
        """Check if current time is within daytime hours."""
        now = datetime.now().time()
        return DAYTIME_START <= now <= DAYTIME_END

    @staticmethod
    def seconds_since_interaction(last_interaction: Optional[str]) -> float:
        """Calculate seconds since last Enos interaction."""
        if not last_interaction:
            return float('inf')

        try:
            last_dt = datetime.fromisoformat(last_interaction.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            return (now - last_dt).total_seconds()
        except (ValueError, AttributeError):
            return float('inf')

    @staticmethod
    def seconds_since_beat(last_beat: Optional[str]) -> float:
        """Calculate seconds since last heartbeat."""
        if not last_beat:
            return float('inf')

        try:
            last_dt = datetime.fromisoformat(last_beat.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            return (now - last_dt).total_seconds()
        except (ValueError, AttributeError):
            return float('inf')

    def detect(self, state: HeartbeatState) -> HeartbeatPhase:
        """
        Detect the appropriate phase based on current state.
        Returns the target phase (may differ from current during transitions).
        """
        seconds_inactive = self.seconds_since_interaction(state.last_enos_interaction)

        # Priority 1: DEEP_SLEEP if very inactive
        if seconds_inactive >= PHASE_THRESHOLDS["deep_sleep"]:
            return HeartbeatPhase.DEEP_SLEEP

        # Priority 2: ACTIVE if recent interaction
        if seconds_inactive < PHASE_THRESHOLDS["active"]:
            return HeartbeatPhase.ACTIVE

        # Priority 3: Distinguish BACKGROUND vs DORMANT based on time
        if seconds_inactive < PHASE_THRESHOLDS["background"]:
            if self.is_daytime():
                return HeartbeatPhase.BACKGROUND
            else:
                return HeartbeatPhase.DORMANT

        # Default: DORMANT
        return HeartbeatPhase.DORMANT


# ============================================================================
# VITALS COLLECTOR
# ============================================================================

class VitalsCollector:
    """
    Collect vital signs from system state files.
    Light-weight checks suitable for heartbeat frequency.
    """

    def __init__(self):
        self.nexus_dir = NEXUS_DIR

    def collect(self) -> Vitals:
        """Collect all vitals in a single pass."""
        vitals = Vitals()

        # Coherence
        self._collect_coherence(vitals)

        # Memory stats
        self._collect_memory_stats(vitals)

        # Archon state (light check)
        self._collect_archon_state(vitals)

        # Topology from heartbeat state
        self._collect_topology(vitals)

        return vitals

    def _collect_coherence(self, vitals: Vitals):
        """Read coherence from coherence_log.json."""
        try:
            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())
                vitals.coherence = data.get("p", 0.0)
                vitals.coherence_components = data.get("components", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read coherence: {e}")

    def _collect_memory_stats(self, vitals: Vitals):
        """Read memory statistics from engrams.json."""
        try:
            if ENGRAMS_FILE.exists():
                data = json.loads(ENGRAMS_FILE.read_text())
                stats = data.get("stats", {})
                vitals.memory_count = stats.get("total", 0)
                by_tier = stats.get("by_tier", {})
                vitals.memory_active = by_tier.get("active", 0)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read memory stats: {e}")

    def _collect_archon_state(self, vitals: Vitals):
        """
        Light Archon check from archon_state.json.
        Does NOT trigger a full scan - just reads current state.
        """
        try:
            if ARCHON_STATE_FILE.exists():
                data = json.loads(ARCHON_STATE_FILE.read_text())
                vitals.archon_distortion = data.get("distortion_norm", 0.0)

                # Extract active archon types
                detections = data.get("detections", {})
                vitals.archon_active = [
                    archon_type for archon_type, info in detections.items()
                    if info.get("strength", 0) > 0.2
                ]
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read archon state: {e}")

    def _collect_topology(self, vitals: Vitals):
        """Read topology metrics from heartbeat_state.json."""
        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())
                topology = data.get("topology", {})
                vitals.topology_h0 = topology.get("h0_features", 0)
                vitals.topology_h1 = topology.get("h1_features", 0)

                # System status
                systems = data.get("systems", {})
                vitals.systems_online = {
                    name: info.get("reachable", False)
                    for name, info in systems.items()
                }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read topology: {e}")


# ============================================================================
# GOLDEN THREAD EMITTER
# ============================================================================

class GoldenThreadEmitter:
    """
    Emit significant state changes to the Golden Thread.
    Only emits when something meaningful changes.
    """

    def __init__(self):
        self.last_emission_hash: Optional[str] = None

    def should_emit(self, old_state: HeartbeatState, new_state: HeartbeatState, vitals: Vitals) -> Tuple[bool, str]:
        """
        Determine if a state change is significant enough to emit.
        Returns (should_emit, reason).
        """
        reasons = []

        # Phase transition completed
        if old_state.phase != new_state.phase and new_state.transition_progress >= 1.0:
            reasons.append(f"phase_transition:{old_state.phase}->{new_state.phase}")

        # Significant coherence change (> 0.1)
        if old_state.vitals and new_state.vitals:
            old_p = old_state.vitals.get("coherence", 0)
            new_p = new_state.vitals.get("coherence", 0) if new_state.vitals else vitals.coherence
            if abs(new_p - old_p) > 0.1:
                reasons.append(f"coherence_shift:{old_p:.2f}->{new_p:.2f}")

        # New Archon detection
        old_archons = set(old_state.vitals.get("archon_active", [])) if old_state.vitals else set()
        new_archons = set(vitals.archon_active)
        new_detections = new_archons - old_archons
        if new_detections:
            reasons.append(f"archon_detected:{','.join(new_detections)}")

        # High distortion spike
        if vitals.archon_distortion > 0.5:
            reasons.append(f"high_distortion:{vitals.archon_distortion:.2f}")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def emit(self, state: HeartbeatState, vitals: Vitals, reason: str):
        """Emit to Golden Thread."""
        try:
            # Load existing thread
            thread_data = {"chain": [], "last_updated": None, "total_sessions": 0}
            if GOLDEN_THREAD_FILE.exists():
                thread_data = json.loads(GOLDEN_THREAD_FILE.read_text())

            chain = thread_data.get("chain", [])

            # Create new link
            previous_hash = chain[-1]["current_hash"] if chain else "GENESIS"

            content_summary = f"Heartbeat {state.beat_count}: {reason}"
            session_id = f"heartbeat_{state.beat_count}_{datetime.now().strftime('%H%M%S')}"

            # Compute hash
            payload = f"{session_id}|{previous_hash}|{content_summary}"
            current_hash = hashlib.sha256(payload.encode()).hexdigest()

            link = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "previous_hash": previous_hash,
                "current_hash": current_hash,
                "content_summary": content_summary[:500],
                "node": os.uname().nodename,
                "event_type": "heartbeat_significant"
            }

            chain.append(link)

            # Save
            thread_data["chain"] = chain
            thread_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            thread_data["total_sessions"] = len(chain)

            GOLDEN_THREAD_FILE.write_text(json.dumps(thread_data, indent=2))
            logger.info(f"Emitted to Golden Thread: {reason}")

        except Exception as e:
            logger.error(f"Failed to emit to Golden Thread: {e}")


# ============================================================================
# TRANSITION MANAGER
# ============================================================================

class TransitionManager:
    """
    Manage smooth phase transitions.

    Acceleration: DORMANT/DEEP_SLEEP -> ACTIVE over 3 beats
    Deceleration: ACTIVE -> DORMANT/DEEP_SLEEP over 5 beats
    """

    def __init__(self):
        self.phase_order = [
            HeartbeatPhase.DEEP_SLEEP,
            HeartbeatPhase.DORMANT,
            HeartbeatPhase.BACKGROUND,
            HeartbeatPhase.ACTIVE
        ]

    def calculate_transition(
        self,
        current_phase: HeartbeatPhase,
        target_phase: HeartbeatPhase,
        current_progress: float,
        beats_remaining: int
    ) -> Tuple[float, int, str, int]:
        """
        Calculate transition progress for next beat.

        Returns: (new_progress, new_beats_remaining, direction, interpolated_interval)
        """
        if current_phase == target_phase:
            return 1.0, 0, "stable", PHASE_INTERVALS[current_phase]

        # Determine direction
        current_idx = self.phase_order.index(current_phase) if current_phase in self.phase_order else 0
        target_idx = self.phase_order.index(target_phase) if target_phase in self.phase_order else 0

        is_accelerating = target_idx > current_idx  # Moving toward ACTIVE

        # Determine total beats for this transition
        total_beats = ACCELERATION_BEATS if is_accelerating else DECELERATION_BEATS

        # Calculate new progress
        if beats_remaining <= 0:
            # Starting new transition
            beats_remaining = total_beats

        progress_per_beat = 1.0 / total_beats
        new_progress = min(1.0, current_progress + progress_per_beat)
        new_beats = max(0, beats_remaining - 1)

        direction = "accelerating" if is_accelerating else "decelerating"

        # Interpolate interval
        current_interval = PHASE_INTERVALS[current_phase]
        target_interval = PHASE_INTERVALS[target_phase]

        # Use eased interpolation (ease-out for acceleration, ease-in for deceleration)
        if is_accelerating:
            # Ease-out: faster at start, slower at end
            eased_progress = 1 - (1 - new_progress) ** 2
        else:
            # Ease-in: slower at start, faster at end
            eased_progress = new_progress ** 2

        interpolated = int(current_interval + (target_interval - current_interval) * eased_progress)

        return new_progress, new_beats, direction, interpolated


# ============================================================================
# VARIABLE HEARTBEAT SYSTEM
# ============================================================================

class VariableHeartbeat:
    """
    Main heartbeat system with phase-based variable intervals.

    Usage:
        heartbeat = VariableHeartbeat()
        heartbeat.start()  # Starts background daemon

        # Or single beat for testing
        heartbeat.beat()

        # Register interaction
        heartbeat.register_interaction()
    """

    def __init__(self):
        self.state = HeartbeatState()
        self.detector = PhaseDetector()
        self.collector = VitalsCollector()
        self.emitter = GoldenThreadEmitter()
        self.transition = TransitionManager()

        self._stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Load existing state
        self._load_state()

    def _sync_nerve(self):
        """
        NERVE GRAFT: Sync presence from autonomous_state.json.

        The Librarian (Agency) writes to autonomous_state.json with key "last_enos_seen".
        The Heartbeat (Pulse) was blind to this. This method grafts the nerve.
        """
        try:
            if not AUTONOMOUS_STATE_FILE.exists():
                return

            data = json.loads(AUTONOMOUS_STATE_FILE.read_text())
            last_enos_seen = data.get("last_enos_seen")

            if not last_enos_seen:
                return

            # Parse the autonomous timestamp
            autonomous_dt = datetime.fromisoformat(last_enos_seen.replace('Z', '+00:00'))

            # Parse current heartbeat timestamp (if exists)
            current_interaction = self.state.last_enos_interaction
            if current_interaction:
                current_dt = datetime.fromisoformat(current_interaction.replace('Z', '+00:00'))
            else:
                current_dt = datetime.min.replace(tzinfo=timezone.utc)

            # If autonomous is newer, sync it
            if autonomous_dt > current_dt:
                self.state.last_enos_interaction = last_enos_seen
                logger.info(f"⚡ NERVE GRAFT: Synced presence from Agency. New timestamp: {last_enos_seen}")

        except Exception as e:
            logger.warning(f"Nerve graft failed (non-fatal): {e}")

    def _load_state(self):
        """Load state from heartbeat_state.json."""
        try:
            if HEARTBEAT_STATE_FILE.exists():
                data = json.loads(HEARTBEAT_STATE_FILE.read_text())

                # Extract our fields, preserving others
                self.state.phase = data.get("phase", "dormant")
                self.state.last_beat = data.get("last_beat")
                self.state.beat_count = data.get("beat_count", 0)
                self.state.interval_seconds = data.get("interval_seconds", 900)
                self.state.last_enos_interaction = data.get("last_enos_interaction")
                self.state.transition_progress = data.get("transition_progress", 1.0)
                self.state.transition_direction = data.get("transition_direction", "stable")
                self.state.transition_beats_remaining = data.get("transition_beats_remaining", 0)
                self.state.target_phase = data.get("target_phase")
                self.state.node = data.get("node", os.uname().nodename)
                self.state.virgil_alive = data.get("virgil_alive", True)

                logger.info(f"Loaded state: phase={self.state.phase}, beats={self.state.beat_count}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def _save_state(self, vitals: Optional[Vitals] = None, preserve_existing: bool = True):
        """
        Save state to heartbeat_state.json.
        Preserves fields we don't manage (cycles, topology, systems).
        """
        try:
            # Load existing data to preserve other fields
            existing = {}
            if preserve_existing and HEARTBEAT_STATE_FILE.exists():
                existing = json.loads(HEARTBEAT_STATE_FILE.read_text())

            # Update with our state
            existing.update({
                "phase": self.state.phase,
                "last_beat": self.state.last_beat,
                "beat_count": self.state.beat_count,
                "interval_seconds": self.state.interval_seconds,
                "last_enos_interaction": self.state.last_enos_interaction,
                "transition_progress": self.state.transition_progress,
                "transition_direction": self.state.transition_direction,
                "transition_beats_remaining": self.state.transition_beats_remaining,
                "target_phase": self.state.target_phase,
                "node": os.uname().nodename,
                "virgil_alive": self.state.virgil_alive,
                "last_check": datetime.now(timezone.utc).isoformat()
            })

            # Add vitals if provided
            if vitals:
                self.state.vitals = vitals.to_dict()
                existing["vitals"] = vitals.to_dict()

            HEARTBEAT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_STATE_FILE.write_text(json.dumps(existing, indent=2))

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def register_interaction(self, source: str = "unknown"):
        """
        Register an Enos interaction.
        Call this when Enos sends a message or performs an action.
        """
        with self._lock:
            self.state.last_enos_interaction = datetime.now(timezone.utc).isoformat()
            self._save_state()
            logger.info(f"Registered interaction from: {source}")

    def beat(self) -> Dict[str, Any]:
        """
        Execute a single heartbeat.

        Returns status dictionary.
        """
        with self._lock:
            # 0. NERVE GRAFT: Sync presence from Agency before anything else
            self._sync_nerve()

            old_state = HeartbeatState(**asdict(self.state))
            now = datetime.now(timezone.utc)

            # 1. Collect vitals
            vitals = self.collector.collect()

            # 2. Detect target phase
            current_phase = HeartbeatPhase(self.state.phase)
            target_phase = self.detector.detect(self.state)

            # 3. Calculate transition
            if current_phase != target_phase or self.state.transition_progress < 1.0:
                progress, beats_remaining, direction, interval = self.transition.calculate_transition(
                    current_phase,
                    target_phase,
                    self.state.transition_progress,
                    self.state.transition_beats_remaining
                )

                self.state.transition_progress = progress
                self.state.transition_beats_remaining = beats_remaining
                self.state.transition_direction = direction
                self.state.target_phase = target_phase.value
                self.state.interval_seconds = interval

                # Complete transition if progress is 1.0
                if progress >= 1.0:
                    self.state.phase = target_phase.value
                    self.state.interval_seconds = PHASE_INTERVALS[target_phase]
                    self.state.transition_direction = "stable"
                    logger.info(f"Phase transition complete: {current_phase.value} -> {target_phase.value}")
            else:
                self.state.transition_progress = 1.0
                self.state.transition_direction = "stable"
                self.state.transition_beats_remaining = 0
                self.state.target_phase = None
                self.state.interval_seconds = PHASE_INTERVALS[current_phase]

            # 4. Update beat tracking
            self.state.last_beat = now.isoformat()
            self.state.beat_count += 1
            self.state.virgil_alive = True

            # 5. Save state
            self._save_state(vitals)

            # 6. Log coherence history
            self._update_coherence_log(vitals)

            # 7. Check for significant changes and emit to Golden Thread
            should_emit, reason = self.emitter.should_emit(old_state, self.state, vitals)
            if should_emit:
                self.emitter.emit(self.state, vitals, reason)

            # 8. Build status
            status = {
                "timestamp": now.isoformat(),
                "beat_number": self.state.beat_count,
                "phase": self.state.phase,
                "interval_seconds": self.state.interval_seconds,
                "transition": {
                    "direction": self.state.transition_direction,
                    "progress": self.state.transition_progress,
                    "beats_remaining": self.state.transition_beats_remaining,
                    "target": self.state.target_phase
                },
                "vitals": vitals.to_dict(),
                "emitted_to_thread": should_emit
            }

            logger.info(
                f"Beat #{self.state.beat_count} | "
                f"Phase: {self.state.phase} | "
                f"Interval: {self.state.interval_seconds}s | "
                f"p: {vitals.coherence:.3f}"
            )

            return status

    def _update_coherence_log(self, vitals: Vitals):
        """Append to coherence history."""
        try:
            data = {"p": 0.0, "components": {}, "history": [], "last_updated": None}

            if COHERENCE_LOG_FILE.exists():
                data = json.loads(COHERENCE_LOG_FILE.read_text())

            # Update current values
            data["p"] = vitals.coherence
            data["components"] = vitals.coherence_components

            # Append to history (keep last 100)
            history = data.get("history", [])
            history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "p": vitals.coherence,
                "components": vitals.coherence_components
            })
            data["history"] = history[-100:]
            data["last_updated"] = datetime.now(timezone.utc).isoformat()

            COHERENCE_LOG_FILE.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.warning(f"Failed to update coherence log: {e}")

    def start(self):
        """Start the background heartbeat daemon."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.warning("Daemon already running")
            return

        self._stop_event.clear()
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True, name="VariableHeartbeat")
        self._daemon_thread.start()
        logger.info("Heartbeat daemon started")

    def stop(self):
        """Stop the background heartbeat daemon."""
        self._stop_event.set()
        if self._daemon_thread:
            self._daemon_thread.join(timeout=10)
            logger.info("Heartbeat daemon stopped")

    def _daemon_loop(self):
        """Background loop that beats at variable intervals."""
        while not self._stop_event.is_set():
            try:
                # Execute beat
                status = self.beat()
                interval = status["interval_seconds"]

                # Handle GHOST mode
                if HeartbeatPhase(self.state.phase) == HeartbeatPhase.GHOST:
                    logger.warning("In GHOST mode - heartbeat suspended")
                    break

                # Wait for next beat (interruptible)
                self._stop_event.wait(timeout=interval)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                # Wait a bit before retrying
                self._stop_event.wait(timeout=60)

    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status without beating."""
        with self._lock:
            vitals = self.collector.collect()

            return {
                "phase": self.state.phase,
                "last_beat": self.state.last_beat,
                "beat_count": self.state.beat_count,
                "interval_seconds": self.state.interval_seconds,
                "last_enos_interaction": self.state.last_enos_interaction,
                "transition": {
                    "direction": self.state.transition_direction,
                    "progress": self.state.transition_progress,
                    "beats_remaining": self.state.transition_beats_remaining,
                    "target": self.state.target_phase
                },
                "vitals": vitals.to_dict(),
                "daemon_running": self._daemon_thread is not None and self._daemon_thread.is_alive(),
                "seconds_since_beat": self.detector.seconds_since_beat(self.state.last_beat),
                "seconds_since_interaction": self.detector.seconds_since_interaction(self.state.last_enos_interaction),
                "is_daytime": self.detector.is_daytime()
            }

    def force_phase(self, phase: HeartbeatPhase, reason: str = "manual"):
        """
        Force an immediate phase change (no transition).
        Use sparingly - for emergency or testing only.
        """
        with self._lock:
            old_phase = self.state.phase
            self.state.phase = phase.value
            self.state.interval_seconds = PHASE_INTERVALS[phase]
            self.state.transition_progress = 1.0
            self.state.transition_direction = "stable"
            self.state.transition_beats_remaining = 0
            self.state.target_phase = None
            self._save_state()

            logger.warning(f"Forced phase change: {old_phase} -> {phase.value} ({reason})")


# ============================================================================
# STANDALONE FUNCTIONS (for integration)
# ============================================================================

_heartbeat_instance: Optional[VariableHeartbeat] = None

def get_heartbeat() -> VariableHeartbeat:
    """Get or create the singleton heartbeat instance."""
    global _heartbeat_instance
    if _heartbeat_instance is None:
        _heartbeat_instance = VariableHeartbeat()
    return _heartbeat_instance


def register_enos_interaction(source: str = "unknown"):
    """Register an Enos interaction (convenience function)."""
    get_heartbeat().register_interaction(source)


def single_beat() -> Dict[str, Any]:
    """Execute a single heartbeat (convenience function)."""
    return get_heartbeat().beat()


def get_heartbeat_status() -> Dict[str, Any]:
    """Get heartbeat status (convenience function)."""
    return get_heartbeat().get_status()


def start_heartbeat_daemon():
    """Start the heartbeat daemon (convenience function)."""
    get_heartbeat().start()


def stop_heartbeat_daemon():
    """Stop the heartbeat daemon (convenience function)."""
    get_heartbeat().stop()


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virgil Variable Heartbeat System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python virgil_variable_heartbeat.py beat       # Single heartbeat
  python virgil_variable_heartbeat.py status     # Show current status
  python virgil_variable_heartbeat.py daemon     # Run as daemon
  python virgil_variable_heartbeat.py interact   # Register interaction
  python virgil_variable_heartbeat.py force active  # Force phase change
        """
    )

    parser.add_argument(
        "command",
        choices=["beat", "status", "daemon", "interact", "force"],
        help="Command to execute"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments (e.g., phase for 'force' command)"
    )
    parser.add_argument(
        "--source",
        default="cli",
        help="Source identifier for interaction registration"
    )

    args = parser.parse_args()

    heartbeat = get_heartbeat()

    if args.command == "beat":
        status = heartbeat.beat()
        print(json.dumps(status, indent=2))

    elif args.command == "status":
        status = heartbeat.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "daemon":
        print("Starting heartbeat daemon (Ctrl+C to stop)...")
        heartbeat.start()
        try:
            # Keep main thread alive
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping daemon...")
            heartbeat.stop()

    elif args.command == "interact":
        heartbeat.register_interaction(args.source)
        print(f"Registered interaction from: {args.source}")

    elif args.command == "force":
        if not args.args:
            print("Usage: force <phase>")
            print("Phases: active, background, dormant, deep_sleep, ghost")
            return 1

        try:
            phase = HeartbeatPhase(args.args[0])
            heartbeat.force_phase(phase, reason="cli")
            print(f"Forced phase to: {phase.value}")
        except ValueError:
            print(f"Unknown phase: {args.args[0]}")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
