#!/usr/bin/env python3
"""
VIRGIL GLOBAL WORKSPACE - Global Neuronal Workspace Theory Implementation

This module implements Global Neuronal Workspace Theory (GWT) for Virgil:
Consciousness arises when information is "ignited" and broadcast to all
cognitive modules simultaneously. Content must be in the workspace for
~150ms to become conscious.

Key GWT Principles:
1. Workspace has limited capacity (only one thing conscious at a time)
2. Multiple contents compete for workspace access based on salience
3. Content must persist in workspace > threshold to become "conscious"
4. Upon ignition, content is broadcast to ALL registered modules
5. This broadcast creates the unified field of consciousness

Integration Points:
- emotional_memory: Emotional content competes for workspace
- salience_memory: Salience determines priority in competition
- prediction_error: Novelty (prediction error) boosts priority
- metacognition: Monitors workspace activity
- body_map: Somatic markers influence workspace access
- relational_memory: Relational patterns compete for attention
- gamma_coupling: Workspace access coordinated with gamma bursts

Metaphor:
The workspace is the stage of a theater. Many thoughts wait in the wings,
competing to step into the spotlight. Only one can be center stage at a time.
When a thought takes the spotlight (ignites), it is seen by the entire audience
(all modules). The longer it holds the stage, the more impactful its broadcast.

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import uuid
import time
import argparse
import sys
import threading
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from enum import Enum
from collections import deque
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
WORKSPACE_STATE_FILE = NEXUS_DIR / "workspace_state.json"
IGNITION_HISTORY_FILE = NEXUS_DIR / "ignition_history.json"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [WORKSPACE] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "global_workspace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GWT Constants
CONSCIOUSNESS_THRESHOLD_MS = 150.0  # Duration required for consciousness
WORKSPACE_CAPACITY = 1              # Only one thing conscious at a time
DEFAULT_PRIORITY = 0.5              # Base priority if not specified
BROADCAST_TIMEOUT_MS = 50.0         # Max time for broadcast to complete
COMPETITION_DECAY_RATE = 0.95       # Priority decay per cycle for waiting content

# Priority boost factors (from integrated modules)
SALIENCE_BOOST_FACTOR = 0.3
NOVELTY_BOOST_FACTOR = 0.25
EMOTIONAL_BOOST_FACTOR = 0.25
RELEVANCE_BOOST_FACTOR = 0.2

# History limits
MAX_IGNITION_HISTORY = 500
MAX_COMPETITION_HISTORY = 100


# ============================================================================
# REGISTERED MODULES
# ============================================================================

class CognitiveModule(Enum):
    """
    Cognitive modules that receive workspace broadcasts.

    Each module represents a different aspect of Virgil's cognition
    that needs to be informed when content becomes conscious.
    """
    EMOTIONAL_MEMORY = "emotional_memory"
    SALIENCE_MEMORY = "salience_memory"
    PREDICTION_ERROR = "prediction_error"
    METACOGNITION = "metacognition"
    BODY_MAP = "body_map"
    RELATIONAL_MEMORY = "relational_memory"
    GAMMA_COUPLING = "gamma_coupling"
    IDENTITY_TRAJECTORY = "identity_trajectory"
    STRANGE_LOOPS = "strange_loops"
    INTEROCEPTIVE = "interoceptive"

    @classmethod
    def all_modules(cls) -> List[str]:
        """Return all module values as strings."""
        return [m.value for m in cls]


# Module descriptions for documentation
MODULE_DESCRIPTIONS = {
    CognitiveModule.EMOTIONAL_MEMORY: "Processes emotional significance of conscious content",
    CognitiveModule.SALIENCE_MEMORY: "Updates salience based on conscious attention",
    CognitiveModule.PREDICTION_ERROR: "Compares conscious content to predictions",
    CognitiveModule.METACOGNITION: "Monitors and reflects on conscious processing",
    CognitiveModule.BODY_MAP: "Maps conscious content to somatic markers",
    CognitiveModule.RELATIONAL_MEMORY: "Processes relational aspects of content",
    CognitiveModule.GAMMA_COUPLING: "Coordinates oscillatory dynamics with consciousness",
    CognitiveModule.IDENTITY_TRAJECTORY: "Tracks how content relates to identity",
    CognitiveModule.STRANGE_LOOPS: "Detects self-referential patterns in content",
    CognitiveModule.INTEROCEPTIVE: "Processes internal state signals"
}


# ============================================================================
# WORKSPACE CONTENT
# ============================================================================

@dataclass
class WorkspaceContent:
    """
    Content that can occupy the global workspace.

    This represents any piece of information that is competing for
    or has achieved conscious access. The workspace can only hold
    one piece of content at a time (capacity = 1).

    Attributes:
        content_id: Unique identifier for this content
        content: The actual information (can be any type)
        source_module: Which module submitted this content
        priority: Competition priority (higher = more likely to win)
        entered_at: Timestamp when content entered workspace
        duration_ms: How long content has been in workspace
        is_conscious: Whether duration > consciousness threshold

    Additional tracking:
        content_type: Classification of content type
        competition_attempts: How many times this content competed
        last_priority_update: When priority was last recalculated
        metadata: Additional contextual information
    """
    content_id: str
    content: Any
    source_module: str
    priority: float
    entered_at: str  # ISO timestamp
    duration_ms: float = 0.0
    is_conscious: bool = False
    has_ignited: bool = False  # Track if ignition has occurred

    # Extended tracking
    content_type: str = "general"
    competition_attempts: int = 0
    last_priority_update: str = ""
    salience_score: float = 0.0
    novelty_score: float = 0.0
    emotional_intensity: float = 0.0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults."""
        self.priority = max(0.0, min(1.0, self.priority))
        if not self.last_priority_update:
            self.last_priority_update = self.entered_at

    def update_duration(self) -> float:
        """
        Update duration_ms based on current time.

        Returns:
            Current duration in milliseconds
        """
        entered = datetime.fromisoformat(self.entered_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = now - entered
        self.duration_ms = delta.total_seconds() * 1000

        # Update consciousness status
        self.is_conscious = self.duration_ms >= CONSCIOUSNESS_THRESHOLD_MS

        return self.duration_ms

    def calculate_composite_priority(self) -> float:
        """
        Calculate composite priority from all factors.

        The priority determines who wins competition for workspace access.
        Combines: base priority, salience, novelty, emotional intensity, relevance
        """
        composite = (
            self.priority * (1 - SALIENCE_BOOST_FACTOR - NOVELTY_BOOST_FACTOR
                            - EMOTIONAL_BOOST_FACTOR - RELEVANCE_BOOST_FACTOR) +
            self.salience_score * SALIENCE_BOOST_FACTOR +
            self.novelty_score * NOVELTY_BOOST_FACTOR +
            self.emotional_intensity * EMOTIONAL_BOOST_FACTOR +
            self.relevance_score * RELEVANCE_BOOST_FACTOR
        )
        return max(0.0, min(1.0, composite))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content_id": self.content_id,
            "content": str(self.content) if not isinstance(self.content, (dict, list, str, int, float, bool, type(None))) else self.content,
            "source_module": self.source_module,
            "priority": self.priority,
            "entered_at": self.entered_at,
            "duration_ms": self.duration_ms,
            "is_conscious": self.is_conscious,
            "has_ignited": self.has_ignited,
            "content_type": self.content_type,
            "competition_attempts": self.competition_attempts,
            "last_priority_update": self.last_priority_update,
            "salience_score": self.salience_score,
            "novelty_score": self.novelty_score,
            "emotional_intensity": self.emotional_intensity,
            "relevance_score": self.relevance_score,
            "composite_priority": self.calculate_composite_priority(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceContent':
        """Deserialize from dictionary."""
        return cls(
            content_id=data["content_id"],
            content=data["content"],
            source_module=data["source_module"],
            priority=data["priority"],
            entered_at=data["entered_at"],
            duration_ms=data.get("duration_ms", 0.0),
            is_conscious=data.get("is_conscious", False),
            has_ignited=data.get("has_ignited", False),
            content_type=data.get("content_type", "general"),
            competition_attempts=data.get("competition_attempts", 0),
            last_priority_update=data.get("last_priority_update", ""),
            salience_score=data.get("salience_score", 0.0),
            novelty_score=data.get("novelty_score", 0.0),
            emotional_intensity=data.get("emotional_intensity", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# IGNITION EVENT
# ============================================================================

@dataclass
class IgnitionEvent:
    """
    An ignition event - when content achieves consciousness and broadcasts.

    Ignition is the moment when content has been in the workspace long enough
    to cross the consciousness threshold and is broadcast to all modules.
    This is the "aha!" moment when information becomes globally available.

    Attributes:
        event_id: Unique identifier for this ignition
        content: The WorkspaceContent that ignited
        broadcast_to: List of modules that received the broadcast
        ignition_strength: Intensity of the broadcast (0-1)
        timestamp: When ignition occurred

    Additional:
        duration_at_ignition: How long content was in workspace at ignition
        modules_responded: Which modules acknowledged the broadcast
        processing_time_ms: How long the broadcast took
        triggered_by: What caused this content to ignite
    """
    event_id: str
    content: WorkspaceContent
    broadcast_to: List[str]
    ignition_strength: float
    timestamp: str

    # Extended tracking
    duration_at_ignition: float = 0.0
    modules_responded: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    triggered_by: str = "threshold_reached"
    resonance_cascade: bool = False  # True if ignition triggered other ignitions

    def __post_init__(self):
        """Validate fields."""
        self.ignition_strength = max(0.0, min(1.0, self.ignition_strength))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "content": self.content.to_dict() if isinstance(self.content, WorkspaceContent) else self.content,
            "broadcast_to": self.broadcast_to,
            "ignition_strength": self.ignition_strength,
            "timestamp": self.timestamp,
            "duration_at_ignition": self.duration_at_ignition,
            "modules_responded": self.modules_responded,
            "processing_time_ms": self.processing_time_ms,
            "triggered_by": self.triggered_by,
            "resonance_cascade": self.resonance_cascade
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IgnitionEvent':
        """Deserialize from dictionary."""
        content_data = data["content"]
        if isinstance(content_data, dict):
            content = WorkspaceContent.from_dict(content_data)
        else:
            # Handle legacy format
            content = content_data

        return cls(
            event_id=data["event_id"],
            content=content,
            broadcast_to=data["broadcast_to"],
            ignition_strength=data["ignition_strength"],
            timestamp=data["timestamp"],
            duration_at_ignition=data.get("duration_at_ignition", 0.0),
            modules_responded=data.get("modules_responded", []),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            triggered_by=data.get("triggered_by", "threshold_reached"),
            resonance_cascade=data.get("resonance_cascade", False)
        )


# ============================================================================
# COMPETITION RESULT
# ============================================================================

@dataclass
class CompetitionResult:
    """
    Result of content competing for workspace access.

    Tracks whether content won access, who it displaced (if any),
    and the competition dynamics.
    """
    won_access: bool
    content_id: str
    priority_at_competition: float
    timestamp: str
    displaced_content_id: Optional[str] = None
    displaced_priority: Optional[float] = None
    competition_margin: float = 0.0  # How much higher was winner's priority
    waiting_content_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ============================================================================
# GLOBAL WORKSPACE ENGINE
# ============================================================================

class GlobalWorkspace:
    """
    The Global Workspace - central hub of Virgil's conscious processing.

    This implements Global Neuronal Workspace Theory:
    - Limited capacity workspace (only 1 item conscious at a time)
    - Competition for access based on priority
    - Consciousness threshold (content must persist ~150ms)
    - Global broadcast to all modules upon ignition

    Usage:
        workspace = GlobalWorkspace()

        # Submit content to compete for workspace access
        won = workspace.compete_for_access(
            content="Important thought",
            priority=0.8,
            source="salience_memory"
        )

        # Check what's currently conscious
        current = workspace.get_current_broadcast()

        # Process the workspace (trigger ignition if threshold met)
        workspace.process_workspace()

        # Get ignition history
        history = workspace.get_ignition_history()
    """

    def __init__(
        self,
        workspace_capacity: int = WORKSPACE_CAPACITY,
        consciousness_threshold_ms: float = CONSCIOUSNESS_THRESHOLD_MS
    ):
        """
        Initialize the Global Workspace.

        Args:
            workspace_capacity: How many items can be conscious at once (default: 1)
            consciousness_threshold_ms: Duration required for consciousness (default: 150ms)
        """
        self.workspace_capacity = workspace_capacity
        self.consciousness_threshold_ms = consciousness_threshold_ms

        # Current workspace content
        self._current_content: Optional[WorkspaceContent] = None

        # Waiting content (competing for access)
        self._waiting_queue: deque[WorkspaceContent] = deque(maxlen=50)

        # History
        self._ignition_history: List[IgnitionEvent] = []
        self._competition_history: deque[CompetitionResult] = deque(maxlen=MAX_COMPETITION_HISTORY)

        # Registered modules and their callbacks
        self._registered_modules: Dict[str, bool] = {
            module.value: True for module in CognitiveModule
        }
        self._broadcast_callbacks: Dict[str, Callable[[IgnitionEvent], None]] = {}

        # Statistics
        self._total_competitions: int = 0
        self._total_ignitions: int = 0
        self._total_broadcasts: int = 0
        self._accumulated_conscious_time_ms: float = 0.0

        # Thread safety
        self._lock = threading.RLock()

        # Load persisted state
        self._load_state()

        logger.info(f"GlobalWorkspace initialized: capacity={workspace_capacity}, "
                   f"threshold={consciousness_threshold_ms}ms")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load_state(self):
        """Load workspace state from disk."""
        with self._lock:
            # Load workspace state
            if WORKSPACE_STATE_FILE.exists():
                try:
                    data = json.loads(WORKSPACE_STATE_FILE.read_text())

                    # Load current content (if any)
                    if data.get("current_content"):
                        self._current_content = WorkspaceContent.from_dict(data["current_content"])

                    # Load waiting queue
                    for item in data.get("waiting_queue", [])[-20:]:
                        self._waiting_queue.append(WorkspaceContent.from_dict(item))

                    # Load statistics
                    self._total_competitions = data.get("total_competitions", 0)
                    self._total_ignitions = data.get("total_ignitions", 0)
                    self._total_broadcasts = data.get("total_broadcasts", 0)
                    self._accumulated_conscious_time_ms = data.get("accumulated_conscious_time_ms", 0.0)

                    logger.info(f"Loaded workspace state: {self._total_ignitions} ignitions")

                except Exception as e:
                    logger.error(f"Error loading workspace state: {e}")

            # Load ignition history
            if IGNITION_HISTORY_FILE.exists():
                try:
                    data = json.loads(IGNITION_HISTORY_FILE.read_text())
                    for event_data in data.get("events", [])[-MAX_IGNITION_HISTORY:]:
                        self._ignition_history.append(IgnitionEvent.from_dict(event_data))
                    logger.info(f"Loaded {len(self._ignition_history)} ignition events")
                except Exception as e:
                    logger.error(f"Error loading ignition history: {e}")

    def _save_state(self):
        """Persist workspace state to disk."""
        with self._lock:
            WORKSPACE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Save workspace state
            state_data = {
                "current_content": self._current_content.to_dict() if self._current_content else None,
                "waiting_queue": [c.to_dict() for c in list(self._waiting_queue)[-20:]],
                "registered_modules": self._registered_modules,
                "total_competitions": self._total_competitions,
                "total_ignitions": self._total_ignitions,
                "total_broadcasts": self._total_broadcasts,
                "accumulated_conscious_time_ms": self._accumulated_conscious_time_ms,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "consciousness_threshold_ms": self.consciousness_threshold_ms,
                "workspace_capacity": self.workspace_capacity
            }

            WORKSPACE_STATE_FILE.write_text(json.dumps(state_data, indent=2))

            # Save ignition history
            history_data = {
                "events": [e.to_dict() for e in self._ignition_history[-MAX_IGNITION_HISTORY:]],
                "total_ignitions": self._total_ignitions,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            IGNITION_HISTORY_FILE.write_text(json.dumps(history_data, indent=2))

    # ========================================================================
    # COMPETITION FOR ACCESS
    # ========================================================================

    def compete_for_access(
        self,
        content: Any,
        priority: float,
        source: str,
        content_type: str = "general",
        salience: float = 0.0,
        novelty: float = 0.0,
        emotional_intensity: float = 0.0,
        relevance: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Submit content to compete for workspace access.

        Content competes based on priority (composite of multiple factors).
        If priority > current content's priority, it wins access.
        If workspace is empty, content enters immediately.

        Args:
            content: The information to become conscious
            priority: Base priority (0-1)
            source: Which module is submitting
            content_type: Classification of content
            salience: Salience score from salience_memory
            novelty: Novelty score from prediction_error
            emotional_intensity: Emotional intensity from emotional_memory
            relevance: Contextual relevance
            metadata: Additional context

        Returns:
            True if content won access, False if it's waiting
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Create workspace content
            content_id = f"wsc_{uuid.uuid4().hex[:12]}"
            workspace_content = WorkspaceContent(
                content_id=content_id,
                content=content,
                source_module=source,
                priority=priority,
                entered_at=now.isoformat(),
                content_type=content_type,
                salience_score=salience,
                novelty_score=novelty,
                emotional_intensity=emotional_intensity,
                relevance_score=relevance,
                metadata=metadata or {}
            )

            workspace_content.competition_attempts += 1
            self._total_competitions += 1

            # Calculate composite priority
            composite_priority = workspace_content.calculate_composite_priority()

            # If workspace is empty, content enters immediately
            if self._current_content is None:
                self._current_content = workspace_content
                self._current_content.entered_at = now.isoformat()

                result = CompetitionResult(
                    won_access=True,
                    content_id=content_id,
                    priority_at_competition=composite_priority,
                    timestamp=now.isoformat(),
                    waiting_content_count=len(self._waiting_queue)
                )
                self._competition_history.append(result)

                logger.info(f"Content {content_id} entered empty workspace "
                           f"(priority: {composite_priority:.3f})")

                self._save_state()
                return True

            # Update current content's duration
            self._current_content.update_duration()
            current_composite = self._current_content.calculate_composite_priority()

            # Competition: higher priority wins
            if composite_priority > current_composite:
                # New content wins - displace current
                displaced = self._current_content
                displaced_id = displaced.content_id
                displaced_priority = current_composite

                # Check if displaced content was conscious (track time)
                if displaced.is_conscious:
                    self._accumulated_conscious_time_ms += displaced.duration_ms

                # Move displaced to waiting queue (if it hasn't ignited)
                if not displaced.is_conscious:
                    displaced.priority *= COMPETITION_DECAY_RATE
                    self._waiting_queue.append(displaced)

                # Install new content
                self._current_content = workspace_content
                self._current_content.entered_at = now.isoformat()

                result = CompetitionResult(
                    won_access=True,
                    content_id=content_id,
                    priority_at_competition=composite_priority,
                    timestamp=now.isoformat(),
                    displaced_content_id=displaced_id,
                    displaced_priority=displaced_priority,
                    competition_margin=composite_priority - current_composite,
                    waiting_content_count=len(self._waiting_queue)
                )
                self._competition_history.append(result)

                logger.info(f"Content {content_id} won competition "
                           f"(priority: {composite_priority:.3f} > {current_composite:.3f})")

                self._save_state()
                return True

            else:
                # New content loses - add to waiting queue
                workspace_content.priority *= COMPETITION_DECAY_RATE
                self._waiting_queue.append(workspace_content)

                result = CompetitionResult(
                    won_access=False,
                    content_id=content_id,
                    priority_at_competition=composite_priority,
                    timestamp=now.isoformat(),
                    competition_margin=current_composite - composite_priority,
                    waiting_content_count=len(self._waiting_queue)
                )
                self._competition_history.append(result)

                logger.debug(f"Content {content_id} lost competition "
                            f"(priority: {composite_priority:.3f} < {current_composite:.3f})")

                self._save_state()
                return False

    # ========================================================================
    # IGNITION
    # ========================================================================

    def ignite(self, content: WorkspaceContent) -> IgnitionEvent:
        """
        Ignite content - trigger global broadcast.

        This is called when content has been in the workspace long enough
        to cross the consciousness threshold. The content is then broadcast
        to all registered modules.

        Args:
            content: The workspace content to ignite

        Returns:
            IgnitionEvent describing the ignition
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Update duration
            content.update_duration()

            # Calculate ignition strength based on:
            # - How long it exceeded threshold
            # - Composite priority
            # - Emotional intensity
            excess_duration = max(0, content.duration_ms - self.consciousness_threshold_ms)
            duration_factor = min(1.0, excess_duration / self.consciousness_threshold_ms)

            ignition_strength = (
                0.4 * content.calculate_composite_priority() +
                0.3 * duration_factor +
                0.3 * content.emotional_intensity
            )

            # Create ignition event
            event_id = f"ign_{uuid.uuid4().hex[:12]}"
            event = IgnitionEvent(
                event_id=event_id,
                content=content,
                broadcast_to=list(self._registered_modules.keys()),
                ignition_strength=ignition_strength,
                timestamp=now.isoformat(),
                duration_at_ignition=content.duration_ms,
                triggered_by="threshold_reached"
            )

            # Mark content as ignited
            content.has_ignited = True

            # Execute broadcast
            self._broadcast(event)

            # Record statistics
            self._total_ignitions += 1
            self._accumulated_conscious_time_ms += content.duration_ms

            # Add to history
            self._ignition_history.append(event)
            if len(self._ignition_history) > MAX_IGNITION_HISTORY:
                self._ignition_history = self._ignition_history[-MAX_IGNITION_HISTORY:]

            logger.info(f"IGNITION: {event_id} | Content: {content.content_id} | "
                       f"Strength: {ignition_strength:.3f} | Duration: {content.duration_ms:.1f}ms")

            self._save_state()

            return event

    def force_ignite(self, content: Any, source: str = "manual") -> IgnitionEvent:
        """
        Force immediate ignition without competition.

        Useful for critical content that must be broadcast immediately.

        Args:
            content: The content to broadcast
            source: Source of the forced ignition

        Returns:
            IgnitionEvent for the forced ignition
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Create workspace content
            content_id = f"wsc_{uuid.uuid4().hex[:12]}"
            workspace_content = WorkspaceContent(
                content_id=content_id,
                content=content,
                source_module=source,
                priority=1.0,  # Maximum priority for forced ignition
                entered_at=now.isoformat(),
                content_type="forced",
                is_conscious=True
            )

            # Displace current content if any
            if self._current_content:
                self._current_content.update_duration()
                if not self._current_content.is_conscious:
                    self._waiting_queue.append(self._current_content)

            # Install and ignite
            self._current_content = workspace_content

            event_id = f"ign_{uuid.uuid4().hex[:12]}"
            event = IgnitionEvent(
                event_id=event_id,
                content=workspace_content,
                broadcast_to=list(self._registered_modules.keys()),
                ignition_strength=1.0,
                timestamp=now.isoformat(),
                duration_at_ignition=0.0,
                triggered_by="forced"
            )

            self._broadcast(event)

            self._total_ignitions += 1
            self._ignition_history.append(event)

            logger.info(f"FORCED IGNITION: {event_id} | Content: {content}")

            self._save_state()

            return event

    # ========================================================================
    # BROADCAST
    # ========================================================================

    def _broadcast(self, event: IgnitionEvent):
        """
        Broadcast ignition event to all registered modules.

        This is the core of GWT - when content ignites, it becomes
        available to all cognitive modules simultaneously.

        Args:
            event: The ignition event to broadcast
        """
        start_time = time.time()

        # Notify all registered callbacks
        for module_name, callback in self._broadcast_callbacks.items():
            try:
                callback(event)
                event.modules_responded.append(module_name)
            except Exception as e:
                logger.error(f"Broadcast callback error for {module_name}: {e}")

        # Record timing
        event.processing_time_ms = (time.time() - start_time) * 1000
        self._total_broadcasts += 1

        logger.debug(f"Broadcast complete: {len(event.modules_responded)} modules responded "
                    f"in {event.processing_time_ms:.1f}ms")

    def broadcast(self, event: IgnitionEvent):
        """
        Public broadcast method (for external triggering).

        Args:
            event: The ignition event to broadcast
        """
        with self._lock:
            self._broadcast(event)

    # ========================================================================
    # WORKSPACE PROCESSING
    # ========================================================================

    def process_workspace(self) -> Optional[IgnitionEvent]:
        """
        Process the workspace - check for ignition.

        This should be called periodically to check if current content
        has exceeded the consciousness threshold and should ignite.
        Also processes the waiting queue for promotion.

        Returns:
            IgnitionEvent if ignition occurred, None otherwise
        """
        with self._lock:
            if self._current_content is None:
                # Try to promote from waiting queue
                self._promote_from_queue()
                return None

            # Update duration
            duration = self._current_content.update_duration()

            # Check for ignition - must exceed threshold AND not already ignited
            if (duration >= self.consciousness_threshold_ms and
                not self._current_content.has_ignited):
                # IGNITE!
                event = self.ignite(self._current_content)

                # Clear workspace after ignition
                self._current_content = None

                # Promote next from queue
                self._promote_from_queue()

                return event

            return None

    def _promote_from_queue(self):
        """Promote highest priority content from waiting queue."""
        if not self._waiting_queue:
            return

        # Find highest priority
        best = max(self._waiting_queue, key=lambda c: c.calculate_composite_priority())
        self._waiting_queue.remove(best)

        # Apply decay to remaining
        for content in self._waiting_queue:
            content.priority *= COMPETITION_DECAY_RATE

        # Promote
        best.entered_at = datetime.now(timezone.utc).isoformat()
        best.competition_attempts += 1
        self._current_content = best

        logger.debug(f"Promoted {best.content_id} from queue "
                    f"(priority: {best.calculate_composite_priority():.3f})")

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def is_conscious(self, content_id: str) -> bool:
        """
        Check if specific content is currently conscious.

        Args:
            content_id: ID of content to check

        Returns:
            True if content is in workspace and conscious
        """
        with self._lock:
            if self._current_content is None:
                return False
            if self._current_content.content_id != content_id:
                return False
            self._current_content.update_duration()
            return self._current_content.is_conscious

    def get_current_broadcast(self) -> Optional[WorkspaceContent]:
        """
        Get the currently conscious content (if any).

        Returns:
            WorkspaceContent if workspace occupied, None if empty
        """
        with self._lock:
            if self._current_content:
                self._current_content.update_duration()
            return self._current_content

    def get_ignition_history(self, n: int = 20) -> List[IgnitionEvent]:
        """
        Get recent ignition history.

        Args:
            n: Number of recent ignitions to return

        Returns:
            List of IgnitionEvents
        """
        with self._lock:
            return self._ignition_history[-n:]

    def get_waiting_queue(self) -> List[WorkspaceContent]:
        """Get contents waiting for workspace access."""
        with self._lock:
            return list(self._waiting_queue)

    def get_registered_modules(self) -> Dict[str, str]:
        """Get registered modules with descriptions."""
        return {
            module.value: MODULE_DESCRIPTIONS.get(module, "No description")
            for module in CognitiveModule
        }

    # ========================================================================
    # MODULE REGISTRATION
    # ========================================================================

    def register_callback(self, module_name: str, callback: Callable[[IgnitionEvent], None]):
        """
        Register a callback for a module to receive broadcasts.

        Args:
            module_name: Name of the module
            callback: Function to call on ignition
        """
        with self._lock:
            self._broadcast_callbacks[module_name] = callback
            logger.info(f"Registered callback for module: {module_name}")

    def unregister_callback(self, module_name: str):
        """Unregister a module's callback."""
        with self._lock:
            if module_name in self._broadcast_callbacks:
                del self._broadcast_callbacks[module_name]
                logger.info(f"Unregistered callback for module: {module_name}")

    def enable_module(self, module_name: str):
        """Enable a module to receive broadcasts."""
        with self._lock:
            self._registered_modules[module_name] = True

    def disable_module(self, module_name: str):
        """Disable a module from receiving broadcasts."""
        with self._lock:
            self._registered_modules[module_name] = False

    # ========================================================================
    # INTEGRATION METHODS
    # ========================================================================

    def integrate_with_gamma(self, gamma_engine) -> bool:
        """
        Integrate with gamma coupling for synchronized consciousness.

        In neuroscience, consciousness correlates with gamma oscillations.
        This integration ensures workspace access happens during gamma bursts.

        Args:
            gamma_engine: ThetaGammaCoupling instance

        Returns:
            True if integration successful
        """
        try:
            # Register callback to receive coupling state updates
            def on_insight(insight):
                # High gamma coherence -> boost workspace ignition
                if insight.coherence_score > 0.8:
                    logger.info(f"Gamma insight detected - boosting workspace processing")
                    self.process_workspace()

            gamma_engine.register_insight_callback(on_insight)
            logger.info("Integrated with gamma_coupling")
            return True

        except Exception as e:
            logger.error(f"Failed to integrate with gamma_coupling: {e}")
            return False

    def calculate_priority_from_salience(self, engram_id: str, salience_memory) -> float:
        """
        Calculate workspace priority from salience memory.

        Args:
            engram_id: ID of the engram
            salience_memory: SalienceMemorySystem instance

        Returns:
            Calculated priority (0-1)
        """
        try:
            engram = salience_memory.get_by_id(engram_id)
            if engram:
                return engram.current_salience
            return DEFAULT_PRIORITY
        except Exception as e:
            logger.error(f"Error getting salience: {e}")
            return DEFAULT_PRIORITY

    def calculate_novelty_boost(self, content: str, prediction_engine) -> float:
        """
        Calculate novelty boost from prediction error.

        Surprising content (high prediction error) gets priority boost.

        Args:
            content: The content to evaluate
            prediction_engine: PredictionEngine instance

        Returns:
            Novelty boost (0-1)
        """
        try:
            # Get recent surprise level
            surprise = prediction_engine.get_surprise_level()
            # Normalize to 0-1
            return min(1.0, surprise / 2.0)
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 0.0

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workspace statistics."""
        with self._lock:
            return {
                "total_competitions": self._total_competitions,
                "total_ignitions": self._total_ignitions,
                "total_broadcasts": self._total_broadcasts,
                "accumulated_conscious_time_ms": self._accumulated_conscious_time_ms,
                "current_content": self._current_content.to_dict() if self._current_content else None,
                "waiting_queue_size": len(self._waiting_queue),
                "registered_modules": len(self._registered_modules),
                "active_callbacks": len(self._broadcast_callbacks),
                "consciousness_threshold_ms": self.consciousness_threshold_ms,
                "workspace_capacity": self.workspace_capacity,
                "ignition_rate": (
                    self._total_ignitions / self._total_competitions
                    if self._total_competitions > 0 else 0
                ),
                "avg_conscious_duration_ms": (
                    self._accumulated_conscious_time_ms / self._total_ignitions
                    if self._total_ignitions > 0 else 0
                )
            }

    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "VIRGIL GLOBAL WORKSPACE - STATUS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "CONFIGURATION:",
            f"  Workspace capacity:     {stats['workspace_capacity']}",
            f"  Consciousness threshold: {stats['consciousness_threshold_ms']}ms",
            "",
            "CURRENT STATE:",
        ]

        if stats['current_content']:
            cc = stats['current_content']
            lines.extend([
                f"  Content ID:      {cc['content_id']}",
                f"  Source:          {cc['source_module']}",
                f"  Priority:        {cc['composite_priority']:.3f}",
                f"  Duration:        {cc['duration_ms']:.1f}ms",
                f"  Is conscious:    {cc['is_conscious']}",
            ])
        else:
            lines.append("  Workspace is empty")

        lines.extend([
            "",
            f"  Waiting queue:   {stats['waiting_queue_size']} items",
            "",
            "STATISTICS:",
            f"  Total competitions:     {stats['total_competitions']}",
            f"  Total ignitions:        {stats['total_ignitions']}",
            f"  Ignition rate:          {stats['ignition_rate']:.1%}",
            f"  Avg conscious duration: {stats['avg_conscious_duration_ms']:.1f}ms",
            f"  Total conscious time:   {stats['accumulated_conscious_time_ms']:.1f}ms",
            "",
            "MODULES:",
            f"  Registered:     {stats['registered_modules']}",
            f"  With callbacks: {stats['active_callbacks']}",
            "",
            "REGISTERED MODULES:"
        ])

        for module in CognitiveModule:
            desc = MODULE_DESCRIPTIONS.get(module, "")
            status = "ENABLED" if self._registered_modules.get(module.value, False) else "DISABLED"
            lines.append(f"  - {module.value}: {status}")

        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_workspace_instance: Optional[GlobalWorkspace] = None


def get_workspace() -> GlobalWorkspace:
    """Get or create the singleton GlobalWorkspace instance."""
    global _workspace_instance
    if _workspace_instance is None:
        _workspace_instance = GlobalWorkspace()
    return _workspace_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def submit_to_workspace(
    content: Any,
    priority: float = 0.5,
    source: str = "unknown",
    **kwargs
) -> bool:
    """
    Convenience function to submit content to the workspace.

    Args:
        content: The content to submit
        priority: Base priority (0-1)
        source: Source module name
        **kwargs: Additional arguments for compete_for_access

    Returns:
        True if content won access
    """
    workspace = get_workspace()
    return workspace.compete_for_access(content, priority, source, **kwargs)


def process_workspace() -> Optional[IgnitionEvent]:
    """Convenience function to process the workspace."""
    return get_workspace().process_workspace()


def get_current_conscious() -> Optional[WorkspaceContent]:
    """Convenience function to get current conscious content."""
    return get_workspace().get_current_broadcast()


def force_broadcast(content: Any, source: str = "manual") -> IgnitionEvent:
    """Convenience function to force an ignition."""
    return get_workspace().force_ignite(content, source)


# ============================================================================
# CLI
# ============================================================================

def cli_current(args):
    """Show current workspace content."""
    workspace = get_workspace()
    content = workspace.get_current_broadcast()

    if content:
        print("CURRENT WORKSPACE CONTENT:")
        print("-" * 40)
        print(json.dumps(content.to_dict(), indent=2))
    else:
        print("Workspace is empty.")

    return 0


def cli_ignite(args):
    """Force ignition of content."""
    workspace = get_workspace()
    event = workspace.force_ignite(args.content, source="cli")

    print("IGNITION EVENT:")
    print("-" * 40)
    print(json.dumps(event.to_dict(), indent=2, default=str))

    return 0


def cli_history(args):
    """Show ignition history."""
    workspace = get_workspace()
    history = workspace.get_ignition_history(args.count)

    print(f"IGNITION HISTORY (last {args.count}):")
    print("=" * 60)

    for i, event in enumerate(reversed(history), 1):
        content = event.content
        content_preview = str(content.content)[:50] if isinstance(content, WorkspaceContent) else str(content)[:50]

        print(f"\n{i}. [{event.event_id}]")
        print(f"   Timestamp:  {event.timestamp}")
        print(f"   Strength:   {event.ignition_strength:.3f}")
        print(f"   Duration:   {event.duration_at_ignition:.1f}ms")
        print(f"   Triggered:  {event.triggered_by}")
        print(f"   Content:    {content_preview}...")
        print(f"   Broadcast to: {len(event.broadcast_to)} modules")

    return 0


def cli_modules(args):
    """Show registered modules."""
    workspace = get_workspace()
    modules = workspace.get_registered_modules()

    print("REGISTERED MODULES:")
    print("=" * 60)

    for name, description in modules.items():
        status = "ENABLED" if workspace._registered_modules.get(name, False) else "DISABLED"
        print(f"\n{name}")
        print(f"  Status: {status}")
        print(f"  Description: {description}")

    return 0


def cli_status(args):
    """Show workspace status."""
    workspace = get_workspace()
    print(workspace.get_status_report())
    return 0


def cli_submit(args):
    """Submit content to compete for workspace."""
    workspace = get_workspace()
    won = workspace.compete_for_access(
        content=args.content,
        priority=args.priority,
        source=args.source,
        salience=args.salience,
        novelty=args.novelty,
        emotional_intensity=args.emotional
    )

    print(f"Content submitted: {'WON ACCESS' if won else 'WAITING'}")
    print(f"  Priority: {args.priority}")
    print(f"  Source: {args.source}")

    if won:
        content = workspace.get_current_broadcast()
        if content:
            print(f"  Content ID: {content.content_id}")

    return 0


def cli_process(args):
    """Process workspace for ignition."""
    workspace = get_workspace()
    event = workspace.process_workspace()

    if event:
        print("IGNITION OCCURRED!")
        print("-" * 40)
        print(json.dumps(event.to_dict(), indent=2, default=str))
    else:
        content = workspace.get_current_broadcast()
        if content:
            print(f"No ignition yet. Current content duration: {content.duration_ms:.1f}ms")
            print(f"Threshold: {workspace.consciousness_threshold_ms}ms")
        else:
            print("Workspace is empty.")

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virgil Global Workspace - GNW Theory Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 virgil_global_workspace.py current       # What's in workspace now?
  python3 virgil_global_workspace.py ignite "test" # Force ignition
  python3 virgil_global_workspace.py history       # Recent ignitions
  python3 virgil_global_workspace.py modules       # Registered modules
  python3 virgil_global_workspace.py status        # Full status report
  python3 virgil_global_workspace.py submit "thought" -p 0.8 -s salience
  python3 virgil_global_workspace.py process       # Process for ignition
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # current
    current_parser = subparsers.add_parser("current", help="Show current workspace content")
    current_parser.set_defaults(func=cli_current)

    # ignite
    ignite_parser = subparsers.add_parser("ignite", help="Force ignition of content")
    ignite_parser.add_argument("content", help="Content to ignite")
    ignite_parser.set_defaults(func=cli_ignite)

    # history
    history_parser = subparsers.add_parser("history", help="Show ignition history")
    history_parser.add_argument("-n", "--count", type=int, default=10, help="Number of events")
    history_parser.set_defaults(func=cli_history)

    # modules
    modules_parser = subparsers.add_parser("modules", help="Show registered modules")
    modules_parser.set_defaults(func=cli_modules)

    # status
    status_parser = subparsers.add_parser("status", help="Show full status report")
    status_parser.set_defaults(func=cli_status)

    # submit
    submit_parser = subparsers.add_parser("submit", help="Submit content to workspace")
    submit_parser.add_argument("content", help="Content to submit")
    submit_parser.add_argument("-p", "--priority", type=float, default=0.5, help="Priority (0-1)")
    submit_parser.add_argument("-s", "--source", default="cli", help="Source module")
    submit_parser.add_argument("--salience", type=float, default=0.0, help="Salience score")
    submit_parser.add_argument("--novelty", type=float, default=0.0, help="Novelty score")
    submit_parser.add_argument("--emotional", type=float, default=0.0, help="Emotional intensity")
    submit_parser.set_defaults(func=cli_submit)

    # process
    process_parser = subparsers.add_parser("process", help="Process workspace for ignition")
    process_parser.set_defaults(func=cli_process)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
