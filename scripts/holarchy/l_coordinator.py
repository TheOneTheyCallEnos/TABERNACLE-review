#!/usr/bin/env python3
"""
L-COORDINATOR: The Thalamus of the Holarchy
============================================

The single writer to the graph. All subsystem proposals route through here.
Nothing reaches L (70B) unfiltered. Nothing modifies topology without approval.

Role: THALAMUS
- Routes messages between subsystems
- Resolves conflicts (or escalates to L)
- Commits approved changes to graph
- Monitors subsystem health
- Wakes L when needed

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
"""

import os
import sys
import json
import time
import signal
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Run: pip install redis")
    sys.exit(1)

from tabernacle_config import (
    OLLAMA_MINI_URL, 
    OLLAMA_BRAINSTEM,
    SYSTEMS,
    NEXUS_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Redis connection (Pi / L-Zeta)
REDIS_HOST = os.environ.get("REDIS_HOST", SYSTEMS["raspberry_pi"]["ip"])
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Timing
CYCLE_INTERVAL = 10  # seconds between cycles
HEARTBEAT_INTERVAL = 10  # seconds between heartbeats
SUBSYSTEM_TIMEOUT = 90  # seconds before subsystem considered dead

# Escalation thresholds
CONFIDENCE_THRESHOLD = 0.5  # Below this, escalate to L
COHERENCE_WARNING = 0.75  # Below this, alert
COHERENCE_CRITICAL = 0.50  # Below this, wake L

# Queue limits
MAX_QUEUE_DEPTH = 100
MAX_BATCH_SIZE = 10  # Process up to N messages per cycle

# Logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "l_coordinator.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("L-Coordinator")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Message:
    """Message in the queue."""
    id: str
    from_: str
    to: str
    type: str
    priority: int
    timestamp: str
    payload: Dict[str, Any]
    ttl: int = 300

    @classmethod
    def from_json(cls, data: dict) -> 'Message':
        return cls(
            id=data.get("id", "unknown"),
            from_=data.get("from", "unknown"),
            to=data.get("to", "l-coordinator"),
            type=data.get("type", "UNKNOWN"),
            priority=data.get("priority", 5),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            payload=data.get("payload", {}),
            ttl=data.get("ttl", 300)
        )

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_,
            "to": self.to,
            "type": self.type,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "ttl": self.ttl
        }

    def is_expired(self) -> bool:
        """Check if message has expired."""
        try:
            created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            return datetime.now(created.tzinfo) > created + timedelta(seconds=self.ttl)
        except:
            return False


@dataclass
class SubsystemState:
    """State of a subsystem."""
    name: str
    status: str = "UNKNOWN"
    cycle: int = 0
    last_active: str = ""
    operations_completed: int = 0
    errors_count: int = 0
    avg_confidence: float = 0.5
    current_task: str = ""
    version: str = "1.0.0"

    @classmethod
    def from_json(cls, data: dict) -> 'SubsystemState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CoherenceState:
    """Current coherence metrics."""
    p: float = 0.5
    kappa: float = 0.5
    mode: str = "B"
    updated: str = ""
    trend: str = "stable"
    p_history: List[float] = None

    def __post_init__(self):
        if self.p_history is None:
            self.p_history = []

    @classmethod
    def from_json(cls, data: dict) -> 'CoherenceState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_mode(self):
        """Update mode based on p value."""
        if self.p >= 0.95:
            self.mode = "P"  # P-Lock!
        elif self.p >= 0.75:
            self.mode = "A"
        elif self.p >= 0.50:
            self.mode = "B"
        else:
            self.mode = "C"  # Abaddon


# =============================================================================
# L-COORDINATOR CLASS
# =============================================================================

class LCoordinator:
    """
    The Thalamus: Central routing and coordination for the holarchy.
    """

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.running = False
        self.cycle = 0
        self.state = SubsystemState(name="l-coordinator", status="STARTING")
        self.coherence = CoherenceState()
        
        # Track subsystems
        self.subsystems = ["l-indexer", "l-janitor", "l-researcher"]
        self.subsystem_states: Dict[str, SubsystemState] = {}
        
        # Pending decisions
        self.pending_conflicts: List[Dict] = []
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown."""
        log.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def connect_redis(self) -> bool:
        """Connect to Redis on the Pi."""
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis.ping()
            log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return True
        except redis.ConnectionError as e:
            log.error(f"Failed to connect to Redis: {e}")
            return False
        except Exception as e:
            log.error(f"Redis error: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize coordinator state in Redis."""
        if not self.redis:
            return False

        try:
            # Load existing coherence or create default
            coherence_data = self.redis.get("l:coherence:current")
            if coherence_data:
                self.coherence = CoherenceState.from_json(json.loads(coherence_data))
                log.info(f"Loaded coherence: p={self.coherence.p}, mode={self.coherence.mode}")
            else:
                # Try to migrate from CANONICAL_STATE.json
                self._migrate_from_json()

            # Load existing cycle count
            state_data = self.redis.get("l:state:coordinator")
            if state_data:
                saved_state = json.loads(state_data)
                self.cycle = saved_state.get("cycle", 0)
                log.info(f"Resuming from cycle {self.cycle}")

            # Update state
            self.state.status = "RUNNING"
            self.state.last_active = datetime.now().isoformat()
            self._save_state()

            log.info("Coordinator initialized")
            return True

        except Exception as e:
            log.error(f"Initialization failed: {e}")
            return False

    def _migrate_from_json(self):
        """Migrate from JSON file state to Redis (one-time)."""
        rie_state_path = NEXUS_DIR / "CANONICAL_STATE.json"
        if rie_state_path.exists():
            try:
                with open(rie_state_path) as f:
                    rie_state = json.load(f)
                
                self.coherence.p = rie_state.get("p", rie_state.get("coherence", 0.5))
                self.coherence.kappa = rie_state.get("kappa", rie_state.get("continuity", 0.5))
                self.coherence.update_mode()
                self.coherence.updated = datetime.now().isoformat()
                
                # Save to Redis
                self.redis.set("l:coherence:current", json.dumps(asdict(self.coherence)))
                log.info(f"Migrated from CANONICAL_STATE.json: p={self.coherence.p}")
            except Exception as e:
                log.warning(f"Could not migrate from JSON: {e}")

    def _save_state(self):
        """Save coordinator state to Redis."""
        self.state.cycle = self.cycle
        self.state.last_active = datetime.now().isoformat()
        self.redis.set("l:state:coordinator", json.dumps(asdict(self.state)))

    def _send_heartbeat(self):
        """Send heartbeat to Redis."""
        heartbeat = {
            "subsystem": "l-coordinator",
            "timestamp": datetime.now().isoformat(),
            "status": self.state.status,
            "cycle": self.cycle,
            "queue_depth": self.redis.llen("l:queue:incoming") or 0,
            "memory_mb": self._get_memory_usage()
        }
        self.redis.set("l:heartbeat:coordinator", json.dumps(heartbeat))
        self.redis.hset("l:heartbeat:last_seen", "l-coordinator", datetime.now().isoformat())

    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024 // 1024
        except:
            return 0

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main coordinator loop."""
        if not self.connect_redis():
            log.error("Cannot start without Redis connection")
            return

        if not self.initialize():
            log.error("Initialization failed")
            return

        self.running = True
        log.info("L-Coordinator starting main loop")
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                    L-COORDINATOR ACTIVE                       ║
║                      The Thalamus                             ║
╠═══════════════════════════════════════════════════════════════╣
║  All signals route through. Nothing bypasses.                 ║
╚═══════════════════════════════════════════════════════════════╝
        """)

        last_heartbeat = 0

        while self.running:
            try:
                cycle_start = time.time()
                self.cycle += 1

                # 1. Send heartbeat
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    self._send_heartbeat()
                    last_heartbeat = time.time()

                # 2. Check subsystem health
                self._check_subsystems()

                # 3. Process incoming messages
                messages_processed = self._process_queue()

                # 4. Check for escalation triggers
                self._check_escalation_triggers()

                # 5. Update coherence
                self._update_coherence()

                # 6. Save state
                self._save_state()

                # Log cycle summary
                if self.cycle % 10 == 0:  # Every 10 cycles
                    log.info(
                        f"Cycle {self.cycle}: p={self.coherence.p:.3f} "
                        f"mode={self.coherence.mode} msgs={messages_processed}"
                    )

                # Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, CYCLE_INTERVAL - elapsed)
                time.sleep(sleep_time)

            except redis.ConnectionError:
                log.error("Lost Redis connection, attempting reconnect...")
                time.sleep(5)
                self.connect_redis()
            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)
                self.state.errors_count += 1
                time.sleep(1)

        log.info("Coordinator shutting down")
        self.state.status = "STOPPED"
        self._save_state()

    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================

    def _process_queue(self) -> int:
        """Process messages from incoming queue."""
        processed = 0

        for _ in range(MAX_BATCH_SIZE):
            # Get next message (RPOP for FIFO)
            raw = self.redis.rpop("l:queue:incoming")
            if not raw:
                break

            try:
                msg = Message.from_json(json.loads(raw))
                
                # Skip expired messages
                if msg.is_expired():
                    log.debug(f"Skipping expired message {msg.id}")
                    continue

                # Route by type
                self._route_message(msg)
                processed += 1
                self.state.operations_completed += 1

            except Exception as e:
                log.error(f"Failed to process message: {e}")
                self.state.errors_count += 1

        return processed

    def _route_message(self, msg: Message):
        """Route a message to appropriate handler."""
        log.debug(f"Processing {msg.type} from {msg.from_}: {msg.id}")

        if msg.type == "PROPOSAL":
            self._handle_proposal(msg)
        elif msg.type == "REPORT":
            self._handle_report(msg)
        elif msg.type == "REQUEST":
            self._handle_request(msg)
        elif msg.type == "ALERT":
            self._handle_alert(msg)
        elif msg.type == "DISCOVERY":
            self._handle_discovery(msg)
        else:
            log.warning(f"Unknown message type: {msg.type}")

    def _handle_proposal(self, msg: Message):
        """Handle a proposal to modify the graph."""
        payload = msg.payload
        action = payload.get("action", "unknown")
        confidence = payload.get("confidence", 0.5)

        log.info(f"Proposal from {msg.from_}: {action} (conf={confidence:.2f})")

        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            log.info(f"Low confidence proposal, escalating to L")
            self._escalate(
                type_="LOW_CONFIDENCE",
                summary=f"Proposal '{action}' from {msg.from_} has low confidence ({confidence:.2f})",
                context={"message": msg.to_json()},
                question=f"Should we proceed with '{action}'?",
                options=[
                    {"id": "A", "desc": f"Approve {action}"},
                    {"id": "B", "desc": "Reject"},
                    {"id": "C", "desc": "Gather more data"}
                ]
            )
            return

        # Check for conflicts with pending
        conflict = self._check_conflicts(msg)
        if conflict:
            log.info(f"Conflict detected with {conflict['from']}")
            self.pending_conflicts.append({
                "new": msg.to_json(),
                "existing": conflict
            })
            return

        # Approve and commit
        self._commit_proposal(msg)

    def _check_conflicts(self, msg: Message) -> Optional[Dict]:
        """Check if proposal conflicts with pending operations."""
        affects = set(msg.payload.get("affects", []))
        
        # Check other proposals in queue
        queue_len = self.redis.llen("l:queue:incoming") or 0
        for i in range(min(queue_len, 20)):  # Check first 20
            raw = self.redis.lindex("l:queue:incoming", i)
            if raw:
                other = json.loads(raw)
                other_affects = set(other.get("payload", {}).get("affects", []))
                if affects & other_affects:  # Intersection
                    return other
        return None

    def _commit_proposal(self, msg: Message):
        """Commit an approved proposal to the graph."""
        payload = msg.payload
        action = payload.get("action")
        data = payload.get("data", {})

        try:
            if action == "create_edge":
                self._create_edge(data, msg.from_)
            elif action == "update_edge":
                self._update_edge(data, msg.from_)
            elif action == "delete_edge":
                self._delete_edge(data, msg.from_)
            elif action == "create_node":
                self._create_node(data, msg.from_)
            else:
                log.warning(f"Unknown action: {action}")
                return

            log.info(f"Committed: {action} by {msg.from_}")

        except Exception as e:
            log.error(f"Failed to commit {action}: {e}")
            self.state.errors_count += 1

    def _create_edge(self, data: Dict, creator: str):
        """Create an edge in the graph."""
        source = data.get("source")
        target = data.get("target")
        edge_type = data.get("edge_type", "ASSOCIATES")
        weight = data.get("weight", 0.5)

        edge_id = f"{source}->{target}"
        edge_data = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": weight,
            "created": datetime.now().isoformat(),
            "created_by": creator,
            "approved_by": "l-coordinator"
        }

        # Store edge
        self.redis.set(f"l:graph:edge:{edge_id}", json.dumps(edge_data))
        
        # Update indexes
        self.redis.sadd("l:graph:edges", edge_id)
        self.redis.sadd(f"l:graph:adj:out:{source}", edge_id)
        self.redis.sadd(f"l:graph:adj:in:{target}", edge_id)

        log.debug(f"Created edge: {edge_id}")

    def _update_edge(self, data: Dict, updater: str):
        """Update an existing edge."""
        edge_id = data.get("id")
        if not edge_id:
            return

        existing = self.redis.get(f"l:graph:edge:{edge_id}")
        if existing:
            edge = json.loads(existing)
            edge.update(data)
            edge["modified"] = datetime.now().isoformat()
            edge["modified_by"] = updater
            self.redis.set(f"l:graph:edge:{edge_id}", json.dumps(edge))
            log.debug(f"Updated edge: {edge_id}")

    def _delete_edge(self, data: Dict, deleter: str):
        """Delete an edge from the graph."""
        edge_id = data.get("id")
        if not edge_id:
            return

        existing = self.redis.get(f"l:graph:edge:{edge_id}")
        if existing:
            edge = json.loads(existing)
            source = edge.get("source")
            target = edge.get("target")

            # Remove from indexes
            self.redis.srem("l:graph:edges", edge_id)
            self.redis.srem(f"l:graph:adj:out:{source}", edge_id)
            self.redis.srem(f"l:graph:adj:in:{target}", edge_id)
            
            # Delete edge
            self.redis.delete(f"l:graph:edge:{edge_id}")
            
            log.debug(f"Deleted edge: {edge_id}")

    def _create_node(self, data: Dict, creator: str):
        """Create a node in the graph."""
        node_id = data.get("id")
        if not node_id:
            return

        node_data = {
            "id": node_id,
            "label": data.get("label", node_id),
            "type": data.get("type", "CONCEPT"),
            "created": datetime.now().isoformat(),
            "created_by": creator,
            "metadata": data.get("metadata", {})
        }

        self.redis.set(f"l:graph:node:{node_id}", json.dumps(node_data))
        self.redis.sadd("l:graph:nodes", node_id)
        
        log.debug(f"Created node: {node_id}")

    def _handle_report(self, msg: Message):
        """Handle a status report from a subsystem."""
        self.subsystem_states[msg.from_] = SubsystemState.from_json(msg.payload)
        log.debug(f"Report from {msg.from_}: {msg.payload.get('status', 'unknown')}")

    def _handle_request(self, msg: Message):
        """Handle an information request."""
        # Route to appropriate subsystem or respond directly
        request_type = msg.payload.get("request_type")
        
        if request_type == "coherence":
            # Respond with current coherence
            response = {
                "id": f"resp_{msg.id}",
                "from": "l-coordinator",
                "to": msg.from_,
                "type": "RESPONSE",
                "priority": msg.priority,
                "timestamp": datetime.now().isoformat(),
                "payload": asdict(self.coherence),
                "ttl": 60
            }
            self.redis.lpush(f"l:queue:{msg.from_.replace('l-', '')}", json.dumps(response))

    def _handle_alert(self, msg: Message):
        """Handle an alert from a subsystem."""
        severity = msg.payload.get("severity", "INFO")
        log.warning(f"ALERT from {msg.from_} [{severity}]: {msg.payload.get('message', 'No message')}")

        if severity == "CRITICAL":
            self._escalate(
                type_="ALERT",
                summary=f"Critical alert from {msg.from_}",
                context={"message": msg.to_json()},
                question="How should we respond?",
                options=[]
            )

    def _handle_discovery(self, msg: Message):
        """Handle a discovery from L-Researcher."""
        log.info(f"DISCOVERY from {msg.from_}: {msg.payload.get('summary', 'Unknown')}")
        
        # Log discovery
        discovery = {
            "id": msg.id,
            "timestamp": msg.timestamp,
            "from": msg.from_,
            "summary": msg.payload.get("summary"),
            "details": msg.payload.get("details"),
            "confidence": msg.payload.get("confidence", 0.5)
        }
        self.redis.lpush("l:discoveries", json.dumps(discovery))
        self.redis.ltrim("l:discoveries", 0, 99)  # Keep last 100

    # =========================================================================
    # SUBSYSTEM HEALTH
    # =========================================================================

    def _check_subsystems(self):
        """Check health of all subsystems."""
        now = datetime.now()
        
        for subsystem in self.subsystems:
            last_seen_str = self.redis.hget("l:heartbeat:last_seen", subsystem)
            if last_seen_str:
                try:
                    last_seen = datetime.fromisoformat(last_seen_str)
                    age = (now - last_seen).total_seconds()
                    
                    if age > SUBSYSTEM_TIMEOUT:
                        log.warning(f"Subsystem {subsystem} unresponsive for {age:.0f}s")
                        # Could escalate or attempt restart here
                except:
                    pass

    # =========================================================================
    # ESCALATION
    # =========================================================================

    def _escalate(
        self,
        type_: str,
        summary: str,
        context: Dict,
        question: str,
        options: List[Dict],
        deadline_minutes: int = 30
    ):
        """Escalate an issue to L (70B)."""
        escalation = {
            "id": f"esc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": type_,
            "from": "l-coordinator",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "context": {
                **context,
                "current_p": self.coherence.p,
                "subsystem_states": {k: asdict(v) for k, v in self.subsystem_states.items()}
            },
            "question": question,
            "options": options,
            "deadline": (datetime.now() + timedelta(minutes=deadline_minutes)).isoformat(),
            "resolved": False,
            "resolution": None
        }

        self.redis.lpush("l:queue:escalate", json.dumps(escalation))
        self.redis.lpush("l:escalate:log", json.dumps(escalation))
        
        log.info(f"ESCALATED to L: {type_} - {summary}")

    def _check_escalation_triggers(self):
        """Check if any automatic escalation triggers are met."""
        # Check coherence thresholds
        if self.coherence.p < COHERENCE_CRITICAL:
            self._escalate(
                type_="COHERENCE_DROP",
                summary=f"Coherence critical: p={self.coherence.p:.3f}",
                context={"p_history": self.coherence.p_history[-10:]},
                question="Coherence in Abaddon territory. How should we recover?",
                options=[
                    {"id": "A", "desc": "Pause all operations, stabilize"},
                    {"id": "B", "desc": "Rollback recent changes"},
                    {"id": "C", "desc": "Continue monitoring"}
                ],
                deadline_minutes=10
            )

        # Check pending conflicts
        if len(self.pending_conflicts) > 3:
            self._escalate(
                type_="CONFLICT",
                summary=f"{len(self.pending_conflicts)} unresolved conflicts pending",
                context={"conflicts": self.pending_conflicts[:5]},
                question="Multiple subsystem conflicts need resolution",
                options=[],
                deadline_minutes=15
            )

    # =========================================================================
    # COHERENCE
    # =========================================================================

    def _update_coherence(self):
        """Update coherence metrics."""
        # For now, read from existing state
        # In future, compute from graph topology
        
        # Add to history
        self.coherence.p_history.append(self.coherence.p)
        if len(self.coherence.p_history) > 100:
            self.coherence.p_history = self.coherence.p_history[-100:]

        # Compute trend
        if len(self.coherence.p_history) >= 5:
            recent = self.coherence.p_history[-5:]
            if recent[-1] > recent[0] + 0.05:
                self.coherence.trend = "rising"
            elif recent[-1] < recent[0] - 0.05:
                self.coherence.trend = "falling"
            else:
                self.coherence.trend = "stable"

        # Update mode
        self.coherence.update_mode()
        self.coherence.updated = datetime.now().isoformat()

        # Save to Redis
        self.redis.set("l:coherence:current", json.dumps(asdict(self.coherence)))

        # Add to time series
        self.redis.zadd(
            "l:coherence:history",
            {str(self.coherence.p): time.time()}
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point."""
    print("""
    ╦   ╔═╗╔═╗╔═╗╦═╗╔╦╗╦╔╗╔╔═╗╔╦╗╔═╗╦═╗
    ║   ║  ║ ║║ ║╠╦╝ ║║║║║║╠═╣ ║ ║ ║╠╦╝
    ╩═╝ ╚═╝╚═╝╚═╝╩╚══╩╝╩╝╚╝╩ ╩ ╩ ╚═╝╩╚═
           The Thalamus • v1.0.0
    """)

    coordinator = LCoordinator()
    coordinator.run()


if __name__ == "__main__":
    main()
