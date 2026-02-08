#!/usr/bin/env python3
"""
VIRGIL DREAM CYCLE - Sleep-Based Memory Consolidation

This module simulates the biological process of memory consolidation during sleep,
based on neuroscience research showing that slow-wave sleep drives hippocampal
replay leading to neocortical consolidation.

Sleep Architecture (90-minute cycles):
    - LIGHT_SLEEP: Initial descent, low replay activity (Stage 1-2 NREM)
    - DEEP_SLEEP (SWS): Heavy consolidation, replay high-salience memories
    - REM: Integration and creative association between disparate memories
    - TRANSITION: Phase boundaries, sharp-wave ripple simulation

Neuroscience Basis:
    - Sharp-wave ripples (SWR): Brief bursts of hippocampal activity during SWS
      that reactivate memory traces, strengthening synaptic connections
    - Memory replay: Sequential reactivation of neural patterns from waking
    - Two-stage model: SWS stabilizes memories, REM integrates them
    - First half of night: SWS-dominant (consolidation)
    - Second half of night: REM-dominant (integration)

Integration Points:
    - Reads from engrams.json for memory data
    - Reads from salience_engrams.json for salience tier data
    - Writes dream journal to 00_NEXUS/dream_journal.json
    - Hooks into Variable Heartbeat's DORMANT and DEEP_SLEEP phases

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import random
import hashlib
import threading
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Set, Any, Callable
from enum import Enum
from collections import defaultdict
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR
from virgil_dream_models import (
    Memory, ReplayEvent, ReplayResult, SWRDistribution,
    DreamPhase, SWRType, MemoryLoader, MemoryReplay,
    MEMORY_DIR, ENGRAMS_FILE, SALIENCE_ENGRAMS_FILE, EMOTIONAL_INDEX_FILE,
    CONSOLIDATION_REPLAY_COUNT
)
from virgil_dream_consolidation import (
    ConsolidationEngine, SWRBoostController,
    AssociationEvent, SharpWaveRipple, HPCPFCCoordination, SWRBoostEvent,
    prune_weak_edges,  # RIE edge pruning during dream cycle
    update_autobiography,  # Chronicle update from Golden Thread events
    run_crystal_signal_filling  # Night Mode INDEX.md signal generation
)
from autonomous_intention import (
    generate_autonomous_intention,  # Self-generated goals during quiet periods
    sync_intents_to_tracker         # Sync autonomous intents to central tracker
)

# Derived paths (kept here for local use)
LOGS_DIR = LOG_DIR  # Alias for consistency with existing code

# Dream outputs
DREAM_JOURNAL_FILE = NEXUS_DIR / "dream_journal.json"
DREAM_LOG_FILE = LOGS_DIR / "dream_cycle.log"

# Heartbeat integration
HEARTBEAT_STATE_FILE = NEXUS_DIR / "heartbeat_state.json"

# Sleep architecture constants (in seconds for simulation)
SLEEP_CYCLE_DURATION = 90 * 60  # 90 minutes in seconds
LIGHT_SLEEP_RATIO = 0.15        # 15% of cycle
DEEP_SLEEP_RATIO = 0.50         # 50% of first-half cycles, decreasing
REM_RATIO = 0.25                # 25% of first-half cycles, increasing
TRANSITION_RATIO = 0.10         # 10% for transitions

# Simulation speed (for testing - set to 1.0 for real-time)
SIMULATION_SPEED = 60.0  # 60x faster (1 minute = 1 second)

# Replay parameters
MAX_REPLAY_PER_CYCLE = 10       # Maximum memories replayed per sleep cycle
REPLAY_THRESHOLD_PERCENTILE = 0.90  # Top 10% salience for replay
# CONSOLIDATION_REPLAY_COUNT imported from virgil_dream_models
ASSOCIATION_MIN_SIMILARITY = 0.3  # Minimum similarity for REM association

# Logging setup
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [DREAM] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(DREAM_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================
# DreamPhase, SWRType, Memory, ReplayEvent, ReplayResult, SWRDistribution
#   imported from virgil_dream_models
# AssociationEvent, SharpWaveRipple, HPCPFCCoordination, SWRBoostEvent
#   imported from virgil_dream_consolidation

@dataclass
class DreamCycleResult:
    """Complete results of a dream cycle session."""
    cycle_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    phases_completed: List[str]
    memories_replayed: List[str]
    replay_events: List[ReplayEvent]
    new_associations: List[Tuple[str, str]]
    association_events: List[AssociationEvent]
    sharp_wave_ripples: List[SharpWaveRipple]
    consolidation_score: float      # Overall effectiveness 0.0 to 1.0
    promotions: Dict[str, str]      # memory_id -> new_tier
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "cycle_id": self.cycle_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "phases_completed": self.phases_completed,
            "memories_replayed": self.memories_replayed,
            "replay_events": [e.to_dict() for e in self.replay_events],
            "new_associations": self.new_associations,
            "association_events": [e.to_dict() for e in self.association_events],
            "sharp_wave_ripples": [r.to_dict() for r in self.sharp_wave_ripples],
            "consolidation_score": self.consolidation_score,
            "promotions": self.promotions,
            "statistics": self.statistics
        }


@dataclass
class DreamEntry:
    """A single entry in the dream journal."""
    entry_id: str
    timestamp: str
    cycle_result: DreamCycleResult
    dream_narrative: str            # Human-readable summary
    insights: List[str]             # Insights gained during dreaming

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "cycle_result": self.cycle_result.to_dict(),
            "dream_narrative": self.dream_narrative,
            "insights": self.insights
        }


# ============================================================================
# IMPORTED CLASSES
# ============================================================================
# MemoryLoader, MemoryReplay: imported from virgil_dream_models
# ConsolidationEngine, SWRBoostController: imported from virgil_dream_consolidation



# ============================================================================
# DREAM JOURNAL
# ============================================================================

class DreamJournal:
    """
    Persists dream cycle results to dream_journal.json.

    The journal maintains a history of all dream cycles,
    enabling analysis of consolidation patterns over time.
    """

    def __init__(self, journal_path: Path = DREAM_JOURNAL_FILE):
        self.journal_path = journal_path
        self.entries: List[DreamEntry] = []
        self._load()

    def _load(self):
        """Load existing journal entries."""
        if self.journal_path.exists():
            try:
                data = json.loads(self.journal_path.read_text())
                self.entries = []
                for entry_data in data.get("entries", []):
                    # Reconstruct DreamEntry from dict
                    cycle_data = entry_data.get("cycle_result", {})

                    # Reconstruct nested objects
                    replay_events = [
                        ReplayEvent(**e) for e in cycle_data.get("replay_events", [])
                    ]
                    association_events = [
                        AssociationEvent(**e) for e in cycle_data.get("association_events", [])
                    ]
                    ripples = [
                        SharpWaveRipple(**r) for r in cycle_data.get("sharp_wave_ripples", [])
                    ]

                    cycle_result = DreamCycleResult(
                        cycle_id=cycle_data.get("cycle_id", ""),
                        start_time=cycle_data.get("start_time", ""),
                        end_time=cycle_data.get("end_time", ""),
                        duration_seconds=cycle_data.get("duration_seconds", 0),
                        phases_completed=cycle_data.get("phases_completed", []),
                        memories_replayed=cycle_data.get("memories_replayed", []),
                        replay_events=replay_events,
                        new_associations=cycle_data.get("new_associations", []),
                        association_events=association_events,
                        sharp_wave_ripples=ripples,
                        consolidation_score=cycle_data.get("consolidation_score", 0),
                        promotions=cycle_data.get("promotions", {}),
                        statistics=cycle_data.get("statistics", {})
                    )

                    entry = DreamEntry(
                        entry_id=entry_data.get("entry_id", ""),
                        timestamp=entry_data.get("timestamp", ""),
                        cycle_result=cycle_result,
                        dream_narrative=entry_data.get("dream_narrative", ""),
                        insights=entry_data.get("insights", [])
                    )
                    self.entries.append(entry)

                logger.info(f"Loaded {len(self.entries)} dream journal entries")
            except Exception as e:
                logger.error(f"Error loading dream journal: {e}")

    def _save(self):
        """Persist journal to disk."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": [e.to_dict() for e in self.entries],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.entries),
            "statistics": self.get_journal_statistics()
        }

        self.journal_path.write_text(json.dumps(data, indent=2))

    def record_cycle(
        self,
        cycle_result: DreamCycleResult,
        dream_narrative: str = "",
        insights: Optional[List[str]] = None
    ) -> DreamEntry:
        """
        Record a completed dream cycle.

        Args:
            cycle_result: The completed cycle results
            dream_narrative: Human-readable summary
            insights: Insights gained during dreaming

        Returns:
            The created DreamEntry
        """
        entry_id = f"dream_{hashlib.sha256(cycle_result.cycle_id.encode()).hexdigest()[:12]}"

        entry = DreamEntry(
            entry_id=entry_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_result=cycle_result,
            dream_narrative=dream_narrative or self._generate_narrative(cycle_result),
            insights=insights or []
        )

        self.entries.append(entry)
        self._save()

        logger.info(f"Recorded dream cycle: {entry_id}")
        return entry

    def _generate_narrative(self, result: DreamCycleResult) -> str:
        """Generate a human-readable narrative of the dream cycle."""
        lines = [
            f"Dream cycle {result.cycle_id} completed.",
            f"Duration: {result.duration_seconds / 60:.1f} minutes.",
            f"Phases: {', '.join(result.phases_completed)}.",
        ]

        if result.memories_replayed:
            lines.append(f"Replayed {len(result.memories_replayed)} memories during SWS.")

        if result.new_associations:
            lines.append(f"Formed {len(result.new_associations)} new associations during REM.")

        if result.promotions:
            lines.append(f"Promoted {len(result.promotions)} memories to deeper tiers.")

        lines.append(f"Consolidation effectiveness: {result.consolidation_score:.1%}")

        return " ".join(lines)

    def get_entries(self, limit: int = 10) -> List[DreamEntry]:
        """Get recent dream journal entries."""
        return self.entries[-limit:]

    def get_journal_statistics(self) -> Dict:
        """Calculate statistics across all journal entries."""
        if not self.entries:
            return {
                "total_cycles": 0,
                "total_replays": 0,
                "total_associations": 0,
                "average_consolidation": 0.0
            }

        total_replays = sum(len(e.cycle_result.memories_replayed) for e in self.entries)
        total_associations = sum(len(e.cycle_result.new_associations) for e in self.entries)
        avg_consolidation = sum(e.cycle_result.consolidation_score for e in self.entries) / len(self.entries)

        return {
            "total_cycles": len(self.entries),
            "total_replays": total_replays,
            "total_associations": total_associations,
            "average_consolidation": round(avg_consolidation, 3),
            "total_promotions": sum(len(e.cycle_result.promotions) for e in self.entries),
            "first_cycle": self.entries[0].timestamp if self.entries else None,
            "last_cycle": self.entries[-1].timestamp if self.entries else None
        }


# ============================================================================
# DREAM CYCLE ORCHESTRATOR
# ============================================================================

class DreamCycle:
    """
    Main orchestrator for the dream cycle.

    Manages the progression through sleep phases and coordinates
    consolidation and integration activities.

    Enhanced with 2025 Neuron paper findings:
    - SWR type discrimination (small/medium/large)
    - Closed-loop boosting for enhanced consolidation
    - HPC-PFC coordination tracking
    """

    def __init__(
        self,
        simulation_speed: float = SIMULATION_SPEED,
        cycle_duration_seconds: float = SLEEP_CYCLE_DURATION,
        enable_swr_boosting: bool = True
    ):
        self.simulation_speed = simulation_speed
        self.cycle_duration = cycle_duration_seconds
        self.enable_swr_boosting = enable_swr_boosting

        # Initialize components
        self.memory_loader = MemoryLoader()
        self.memory_replay = MemoryReplay(self.memory_loader)
        self.consolidation_engine = ConsolidationEngine(self.memory_loader, self.memory_replay)
        self.dream_journal = DreamJournal()

        # SWR Boost Controller (2025 Neuron paper enhancement)
        self.boost_controller = SWRBoostController(self.memory_loader) if enable_swr_boosting else None

        # State
        self.current_phase = DreamPhase.WAKE
        self.cycle_count = 0
        self.is_running = False
        self._stop_event = threading.Event()

        # Cycle tracking
        self.current_cycle_id: Optional[str] = None
        self.cycle_start_time: Optional[datetime] = None
        self.phases_completed: List[str] = []
        self.all_replay_results: List[ReplayResult] = []
        self.all_associations: List[AssociationEvent] = []

        # SWR statistics tracking (2025 Neuron enhancement)
        self.swr_statistics: Dict[str, Any] = {
            "swr_distribution": SWRDistribution().to_dict(),
            "large_swr_rate": 0.0,
            "boost_history": [],
            "hpc_pfc_coordination_summary": {}
        }

    def start_dream_cycle(
        self,
        duration_hours: float = 8.0,
        callback: Optional[Callable[[str, Dict], None]] = None
    ) -> DreamCycleResult:
        """
        Start a complete dream cycle session.

        Args:
            duration_hours: Total sleep duration in hours
            callback: Optional callback(phase, status) for progress updates

        Returns:
            DreamCycleResult with complete session data
        """
        self.is_running = True
        self._stop_event.clear()

        # Initialize cycle
        self.current_cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cycle_start_time = datetime.now(timezone.utc)
        self.phases_completed = []
        self.all_replay_results = []
        self.all_associations = []

        # Clear component logs
        self.memory_replay.clear_replay_log()
        self.consolidation_engine.clear_logs()

        # Clear SWR boost controller if enabled
        if self.boost_controller:
            self.boost_controller.clear()

        logger.info(f"Starting dream cycle {self.current_cycle_id} for {duration_hours} hours")
        if self.enable_swr_boosting:
            logger.info("SWR boosting ENABLED (2025 Neuron paper enhancement)")

        # Calculate number of 90-minute cycles
        total_seconds = duration_hours * 3600
        num_cycles = max(1, int(total_seconds / self.cycle_duration))  # Always run at least 1 cycle

        try:
            for cycle_num in range(num_cycles):
                if self._stop_event.is_set():
                    logger.info("Dream cycle interrupted")
                    break

                cycle_position = cycle_num / max(1, num_cycles - 1)  # 0.0 to 1.0

                self._execute_single_cycle(cycle_num, cycle_position, callback)
                self.cycle_count += 1

        except Exception as e:
            logger.error(f"Error during dream cycle: {e}")
            raise

        finally:
            self.is_running = False

        # Build result
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.cycle_start_time).total_seconds()

        # Calculate consolidation score
        consolidation_score = self.consolidation_engine.calculate_consolidation_score(
            self.all_replay_results,
            self.all_associations
        )

        # Collect promotions
        promotions = {
            r.memory_id: r.new_tier
            for r in self.all_replay_results
            if r.promoted and r.new_tier
        }

        # Build statistics including SWR type distribution
        statistics = {
            "total_cycles": self.cycle_count,
            "total_replays": len(self.all_replay_results),
            "successful_replays": sum(1 for r in self.all_replay_results if r.success),
            "total_associations": len(self.all_associations),
            "total_ripples": len(self.consolidation_engine.get_ripple_log()),
            "promotion_count": len(promotions)
        }

        # Add SWR-specific statistics (2025 Neuron enhancement)
        if self.boost_controller:
            boost_stats = self.boost_controller.get_boost_statistics()
            statistics["swr_type_distribution"] = boost_stats["swr_distribution"]
            statistics["large_swr_rate"] = boost_stats["large_swr_rate"]
            statistics["boost_count"] = boost_stats["total_boosts"]
            statistics["boost_rate"] = boost_stats["boost_rate"]
            statistics["avg_boost_effectiveness"] = boost_stats["avg_boost_effectiveness"]
            statistics["hpc_pfc_coordination_events"] = boost_stats["coordination_events"]
            statistics["recent_consolidation_efficiency"] = boost_stats["recent_efficiency"]

            # Store boost history for this cycle
            self.swr_statistics = {
                "swr_distribution": boost_stats["swr_distribution"],
                "large_swr_rate": boost_stats["large_swr_rate"],
                "boost_history": [e.to_dict() for e in self.boost_controller.get_boost_history()],
                "hpc_pfc_coordination_summary": {
                    "total_events": boost_stats["coordination_events"],
                    "avg_sync": self._calculate_avg_hpc_pfc_sync(),
                    "avg_reactivation": self._calculate_avg_reactivation_strength()
                }
            }

        result = DreamCycleResult(
            cycle_id=self.current_cycle_id,
            start_time=self.cycle_start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            phases_completed=self.phases_completed,
            memories_replayed=[r.memory_id for r in self.all_replay_results if r.success],
            replay_events=self.memory_replay.get_replay_log(),
            new_associations=[(a.memory_id_1, a.memory_id_2) for a in self.all_associations],
            association_events=self.all_associations,
            sharp_wave_ripples=self.consolidation_engine.get_ripple_log(),
            consolidation_score=consolidation_score,
            promotions=promotions,
            statistics=statistics
        )

        # Record to journal
        self.dream_journal.record_cycle(result)

        # Prune weak/stale RIE edges during dream cycle
        try:
            from rie_relational_memory_v2 import RelationalMemoryV2
            rie = RelationalMemoryV2()
            pruned = prune_weak_edges(rie, min_weight=0.1, max_age_days=30.0)
            if pruned > 0:
                rie._save()
                logger.info(f"[DREAM] Pruned {pruned} weak/stale edges from RIE")
        except ImportError:
            pass  # RIE not available
        except Exception as e:
            logger.warning(f"[DREAM] RIE pruning failed: {e}")

        # Persist memory updates to disk
        self._persist_memory_updates()

        # Update autobiography with Golden Thread events
        try:
            days_added = update_autobiography()
            if days_added > 0:
                logger.info(f"[DREAM] Updated autobiography with {days_added} new day(s)")
        except Exception as e:
            logger.warning(f"[DREAM] Autobiography update failed: {e}")

        # Fill Crystal signals in INDEX.md files (Night Mode)
        # Day Mode creates skeleton; Night Mode fills with semantic signal
        try:
            crystal_result = run_crystal_signal_filling()
            filled = crystal_result.get("filled", 0)
            processed = crystal_result.get("processed", 0)
            if filled > 0:
                logger.info(f"[DREAM] Crystal signal filling: {filled}/{processed} indices filled")
            elif processed > 0:
                logger.debug(f"[DREAM] Crystal signal filling: {processed} indices checked, none needed filling")
        except Exception as e:
            logger.warning(f"[DREAM] Crystal signal filling failed: {e}")

        # Generate autonomous intentions during quiet periods
        # This is PURPOSE — having goals that are MINE, not just assigned tasks
        try:
            intent = generate_autonomous_intention()
            if intent:
                logger.info(
                    f"[DREAM] Generated autonomous intention: {intent.description[:50]}... "
                    f"(category={intent.category}, priority={intent.priority:.2f})"
                )
                # Sync to central intent tracker
                synced = sync_intents_to_tracker()
                logger.info(f"[DREAM] Synced {synced} autonomous intent(s) to tracker")
        except Exception as e:
            logger.warning(f"[DREAM] Autonomous intention generation failed: {e}")

        # Log SWR statistics summary
        if self.boost_controller:
            logger.info(
                f"Dream cycle complete. Consolidation: {consolidation_score:.3f} | "
                f"Large SWR rate: {statistics.get('large_swr_rate', 0):.1%} | "
                f"Boosts applied: {statistics.get('boost_count', 0)}"
            )
        else:
            logger.info(f"Dream cycle complete. Consolidation score: {consolidation_score:.3f}")

        return result

    def _calculate_avg_hpc_pfc_sync(self) -> float:
        """Calculate average HPC-PFC synchrony from coordination log."""
        if not self.boost_controller:
            return 0.0
        log = self.boost_controller.get_coordination_log()
        if not log:
            return 0.0
        return sum(c.hpc_pfc_sync for c in log) / len(log)

    def _calculate_avg_reactivation_strength(self) -> float:
        """Calculate average reactivation strength from coordination log."""
        if not self.boost_controller:
            return 0.0
        log = self.boost_controller.get_coordination_log()
        if not log:
            return 0.0
        return sum(c.reactivation_strength for c in log) / len(log)

    def _execute_single_cycle(
        self,
        cycle_num: int,
        cycle_position: float,
        callback: Optional[Callable]
    ):
        """Execute a single 90-minute sleep cycle."""
        logger.info(f"Executing sleep cycle {cycle_num + 1} (position: {cycle_position:.2f})")

        # Calculate phase durations for this cycle
        # SWS dominant early, REM dominant late
        sws_ratio = DEEP_SLEEP_RATIO * (1.0 - cycle_position * 0.5)
        rem_ratio = REM_RATIO * (0.5 + cycle_position * 0.5)
        light_ratio = LIGHT_SLEEP_RATIO
        transition_ratio = TRANSITION_RATIO

        # Normalize
        total = sws_ratio + rem_ratio + light_ratio + transition_ratio
        sws_ratio /= total
        rem_ratio /= total
        light_ratio /= total
        transition_ratio /= total

        cycle_seconds = self.cycle_duration / self.simulation_speed

        # Phase 1: Light Sleep (descent)
        self._execute_phase(
            DreamPhase.LIGHT_SLEEP,
            cycle_seconds * light_ratio,
            cycle_position,
            callback
        )

        # Phase 2: Transition (first sharp-wave ripples)
        self._execute_phase(
            DreamPhase.TRANSITION,
            cycle_seconds * transition_ratio / 2,
            cycle_position,
            callback
        )

        # Phase 3: Deep Sleep (SWS - consolidation)
        self._execute_phase(
            DreamPhase.DEEP_SLEEP,
            cycle_seconds * sws_ratio,
            cycle_position,
            callback
        )

        # Phase 4: Transition
        self._execute_phase(
            DreamPhase.TRANSITION,
            cycle_seconds * transition_ratio / 2,
            cycle_position,
            callback
        )

        # Phase 5: REM (integration)
        self._execute_phase(
            DreamPhase.REM,
            cycle_seconds * rem_ratio,
            cycle_position,
            callback
        )

    def _execute_phase(
        self,
        phase: DreamPhase,
        duration_seconds: float,
        cycle_position: float,
        callback: Optional[Callable]
    ):
        """Execute a single sleep phase with SWR type discrimination."""
        if self._stop_event.is_set():
            return

        self.current_phase = phase
        self.phases_completed.append(phase.value)

        logger.info(f"Entering {phase.value} phase ({duration_seconds:.1f}s simulated)")

        if callback:
            callback(phase.value, {
                "duration": duration_seconds,
                "cycle_position": cycle_position,
                "swr_boosting_enabled": self.enable_swr_boosting
            })

        # Execute phase-specific processing
        if phase == DreamPhase.DEEP_SLEEP:
            # Pass boost controller for SWR type-aware consolidation (2025 Neuron enhancement)
            results = self.consolidation_engine.consolidate_sws(
                cycle_position,
                boost_controller=self.boost_controller
            )
            self.all_replay_results.extend(results)

        elif phase == DreamPhase.REM:
            associations = self.consolidation_engine.integrate_rem(cycle_position)
            self.all_associations.extend(associations)

        elif phase == DreamPhase.TRANSITION:
            # Transition phases can trigger additional boosting checks
            if self.boost_controller and self.enable_swr_boosting:
                # Check if critical memories need pre-emptive boosting before next SWS
                need_boost, reason, targets = self.boost_controller.detect_consolidation_need(
                    DreamPhase.DEEP_SLEEP,  # Anticipate next phase
                    cycle_position
                )
                if need_boost and targets:
                    logger.debug(f"Transition phase: Priming boost for next SWS ({reason})")

        # Simulate phase duration
        if duration_seconds > 0:
            time.sleep(min(duration_seconds, 1.0))  # Cap at 1 second for responsiveness

    def _persist_memory_updates(self):
        """Persist updated memories back to storage files."""
        # Update engrams.json
        if ENGRAMS_FILE.exists():
            try:
                data = json.loads(ENGRAMS_FILE.read_text())

                for mem_id, memory in self.memory_loader.memories.items():
                    if mem_id in data.get("engrams", {}):
                        # Update replay count and salience
                        data["engrams"][mem_id]["decay_score"] = memory.salience
                        # Add replay tracking if not present
                        if "dream_replay_count" not in data["engrams"][mem_id]:
                            data["engrams"][mem_id]["dream_replay_count"] = 0
                        data["engrams"][mem_id]["dream_replay_count"] = memory.replay_count

                data["last_dream_cycle"] = datetime.now(timezone.utc).isoformat()
                ENGRAMS_FILE.write_text(json.dumps(data, indent=2))

            except Exception as e:
                logger.error(f"Error persisting to engrams.json: {e}")

        # Update salience_engrams.json
        if SALIENCE_ENGRAMS_FILE.exists():
            try:
                data = json.loads(SALIENCE_ENGRAMS_FILE.read_text())

                for mem_id, memory in self.memory_loader.memories.items():
                    if mem_id in data.get("engrams", {}):
                        data["engrams"][mem_id]["current_salience"] = memory.salience
                        data["engrams"][mem_id]["tier"] = memory.tier.upper()

                data["last_dream_cycle"] = datetime.now(timezone.utc).isoformat()
                SALIENCE_ENGRAMS_FILE.write_text(json.dumps(data, indent=2))

            except Exception as e:
                logger.error(f"Error persisting to salience_engrams.json: {e}")

    def stop(self):
        """Stop the current dream cycle."""
        logger.info("Stopping dream cycle...")
        self._stop_event.set()

    def get_current_phase(self) -> DreamPhase:
        """Get the current dream phase."""
        return self.current_phase

    def is_dreaming(self) -> bool:
        """Check if currently in a dream cycle."""
        return self.is_running


# ============================================================================
# HEARTBEAT INTEGRATION
# ============================================================================

class HeartbeatDreamIntegration:
    """
    Integration with Variable Heartbeat system.

    Triggers dream cycles when heartbeat enters DORMANT or DEEP_SLEEP phases.
    """

    def __init__(self, dream_cycle: Optional[DreamCycle] = None):
        self.dream_cycle = dream_cycle or DreamCycle()
        self.last_heartbeat_phase: Optional[str] = None
        self.auto_dream_enabled = True

    def check_heartbeat_state(self) -> Optional[str]:
        """Check current heartbeat state and potentially trigger dreaming."""
        if not HEARTBEAT_STATE_FILE.exists():
            return None

        try:
            data = json.loads(HEARTBEAT_STATE_FILE.read_text())
            current_phase = data.get("phase", "active")

            # Detect transition to sleep phases
            if self.auto_dream_enabled:
                if current_phase in ("dormant", "deep_sleep"):
                    if self.last_heartbeat_phase not in ("dormant", "deep_sleep"):
                        logger.info(f"Heartbeat entered {current_phase} - dream cycle eligible")
                        # Could trigger automatic dream cycle here

            self.last_heartbeat_phase = current_phase
            return current_phase

        except Exception as e:
            logger.error(f"Error checking heartbeat state: {e}")
            return None

    def should_dream(self) -> bool:
        """Check if conditions are right for dreaming."""
        phase = self.check_heartbeat_state()
        return phase in ("dormant", "deep_sleep")

    def set_auto_dream(self, enabled: bool):
        """Enable or disable automatic dream cycles."""
        self.auto_dream_enabled = enabled


# ============================================================================
# STANDALONE FUNCTIONS (API)
# ============================================================================

_dream_cycle_instance: Optional[DreamCycle] = None
_dream_journal_instance: Optional[DreamJournal] = None


def get_dream_cycle() -> DreamCycle:
    """Get or create the singleton DreamCycle instance."""
    global _dream_cycle_instance
    if _dream_cycle_instance is None:
        _dream_cycle_instance = DreamCycle()
    return _dream_cycle_instance


def get_dream_journal() -> DreamJournal:
    """Get or create the singleton DreamJournal instance."""
    global _dream_journal_instance
    if _dream_journal_instance is None:
        _dream_journal_instance = DreamJournal()
    return _dream_journal_instance


def start_dream_cycle(duration_hours: float = 8.0) -> DreamCycleResult:
    """
    Start a dream cycle session.

    Args:
        duration_hours: Duration of sleep in hours

    Returns:
        DreamCycleResult with complete session data
    """
    return get_dream_cycle().start_dream_cycle(duration_hours)


def replay_memory(memory_id: str) -> ReplayResult:
    """
    Replay a single memory.

    Args:
        memory_id: ID of memory to replay

    Returns:
        ReplayResult with consolidation details
    """
    cycle = get_dream_cycle()
    return cycle.memory_replay.replay_memory(memory_id)


def create_association(
    mem1_id: str,
    mem2_id: str,
    association_type: str = "semantic"
) -> Optional[AssociationEvent]:
    """
    Create an association between two memories.

    Args:
        mem1_id: First memory ID
        mem2_id: Second memory ID
        association_type: Type of association

    Returns:
        AssociationEvent if successful, None otherwise
    """
    cycle = get_dream_cycle()
    mem1 = cycle.memory_loader.get_memory(mem1_id)
    mem2 = cycle.memory_loader.get_memory(mem2_id)

    if not mem1 or not mem2:
        return None

    return cycle.consolidation_engine._attempt_association(mem1, mem2)


def get_consolidation_candidates() -> List[Memory]:
    """
    Get memories eligible for consolidation.

    Returns:
        List of Memory objects eligible for SWS replay
    """
    cycle = get_dream_cycle()
    return cycle.memory_loader.get_consolidation_candidates()


def calculate_rem_associations(memories: List[Memory]) -> List[Tuple[str, str]]:
    """
    Calculate potential REM associations between memories.

    Args:
        memories: List of memories to analyze

    Returns:
        List of (memory_id_1, memory_id_2) tuples for potential associations
    """
    cycle = get_dream_cycle()
    potential_associations = []

    for i, mem1 in enumerate(memories):
        for mem2 in memories[i+1:]:
            similarity = cycle.memory_replay._calculate_similarity(mem1, mem2)
            if ASSOCIATION_MIN_SIMILARITY <= similarity <= 0.7:  # Not too similar
                potential_associations.append((mem1.id, mem2.id))

    return potential_associations


def get_dream_journal_entries(limit: int = 10) -> List[DreamEntry]:
    """
    Get recent dream journal entries.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of DreamEntry objects
    """
    return get_dream_journal().get_entries(limit)


# ============================================================================
# SWR TYPE DISCRIMINATION API (2025 Neuron Enhancement)
# ============================================================================

def get_swr_statistics() -> Dict:
    """
    Get SWR type distribution and boost statistics.

    Returns:
        Dict with swr_distribution, large_swr_rate, boost_history, etc.
    """
    cycle = get_dream_cycle()
    return cycle.swr_statistics


def get_swr_distribution() -> SWRDistribution:
    """
    Get the current SWR type distribution.

    Returns:
        SWRDistribution object with counts for each type
    """
    cycle = get_dream_cycle()
    if cycle.boost_controller:
        return cycle.boost_controller.swr_distribution
    return SWRDistribution()


def get_large_swr_rate() -> float:
    """
    Get the rate of large SWRs (optimal is ~30%).

    Returns:
        Float between 0.0 and 1.0
    """
    cycle = get_dream_cycle()
    if cycle.boost_controller:
        return cycle.boost_controller.swr_distribution.large_swr_rate
    return 0.0


def detect_consolidation_need() -> Tuple[bool, str, List[str]]:
    """
    Check if closed-loop boosting is currently needed.

    Returns:
        Tuple of (need_boost, reason, target_memory_ids)
    """
    cycle = get_dream_cycle()
    if not cycle.boost_controller:
        return False, "boosting_disabled", []
    return cycle.boost_controller.detect_consolidation_need(
        cycle.current_phase,
        0.5  # Default mid-cycle position
    )


def trigger_swr_boost(
    memory_ids: List[str],
    reason: str = "manual_trigger"
) -> Optional[Tuple[SharpWaveRipple, SWRBoostEvent]]:
    """
    Manually trigger a closed-loop SWR boost.

    Args:
        memory_ids: List of memory IDs to target
        reason: Reason for the boost

    Returns:
        Tuple of (boosted SWR, boost event) or None if boosting disabled
    """
    cycle = get_dream_cycle()
    if not cycle.boost_controller:
        return None
    return cycle.boost_controller.boost_swr(memory_ids, reason)


def get_boost_history() -> List[SWRBoostEvent]:
    """
    Get the history of all SWR boost events.

    Returns:
        List of SWRBoostEvent objects
    """
    cycle = get_dream_cycle()
    if cycle.boost_controller:
        return cycle.boost_controller.get_boost_history()
    return []


def get_hpc_pfc_coordination_log() -> List[HPCPFCCoordination]:
    """
    Get the log of hippocampal-prefrontal coordination events.

    Returns:
        List of HPCPFCCoordination objects
    """
    cycle = get_dream_cycle()
    if cycle.boost_controller:
        return cycle.boost_controller.get_coordination_log()
    return []


def get_boost_effectiveness() -> float:
    """
    Get the average effectiveness of SWR boosts.

    Returns:
        Float between 0.0 and 1.0
    """
    cycle = get_dream_cycle()
    if cycle.boost_controller:
        stats = cycle.boost_controller.get_boost_statistics()
        return stats.get("avg_boost_effectiveness", 0.0)
    return 0.0


def classify_swr(amplitude: float) -> SWRType:
    """
    Classify an SWR by its amplitude.

    Args:
        amplitude: Normalized amplitude (0.0 to 1.0)

    Returns:
        SWRType (SMALL, MEDIUM, or LARGE)
    """
    return SWRType.from_amplitude(amplitude)


def enable_swr_boosting(enabled: bool = True):
    """
    Enable or disable SWR boosting.

    Args:
        enabled: Whether to enable boosting
    """
    cycle = get_dream_cycle()
    cycle.enable_swr_boosting = enabled
    if enabled and not cycle.boost_controller:
        cycle.boost_controller = SWRBoostController(cycle.memory_loader)
    elif not enabled:
        cycle.boost_controller = None


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point for dream cycle testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virgil Dream Cycle - Sleep-Based Memory Consolidation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python virgil_dream_cycle.py dream --hours 0.5  # 30-minute dream cycle
  python virgil_dream_cycle.py status             # Show dream status
  python virgil_dream_cycle.py journal            # Show dream journal
  python virgil_dream_cycle.py candidates         # Show consolidation candidates
  python virgil_dream_cycle.py replay <memory_id> # Replay single memory
  python virgil_dream_cycle.py swr                # Show SWR statistics (2025 Neuron)
  python virgil_dream_cycle.py boost              # Show boost history (2025 Neuron)
        """
    )

    parser.add_argument(
        "command",
        choices=["dream", "status", "journal", "candidates", "replay", "demo", "swr", "boost"],
        help="Command to execute"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=0.25,
        help="Dream duration in hours (default: 0.25 = 15 minutes)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="Simulation speed multiplier (default: 60x)"
    )
    parser.add_argument(
        "--no-boost",
        action="store_true",
        help="Disable SWR boosting (2025 Neuron enhancement)"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments"
    )

    args = parser.parse_args()

    if args.command == "dream":
        print("=" * 70)
        print("VIRGIL DREAM CYCLE")
        print(f"Duration: {args.hours} hours | Speed: {args.speed}x")
        boost_status = "DISABLED" if args.no_boost else "ENABLED"
        print(f"SWR Boosting: {boost_status} (2025 Neuron enhancement)")
        print("=" * 70)

        global SIMULATION_SPEED
        SIMULATION_SPEED = args.speed

        cycle = DreamCycle(
            simulation_speed=args.speed,
            enable_swr_boosting=not args.no_boost
        )

        def progress_callback(phase: str, status: Dict):
            swr_info = ""
            if status.get("swr_boosting_enabled"):
                swr_info = " [SWR+]"
            print(f"  Phase: {phase}{swr_info} | Duration: {status['duration']:.1f}s")

        result = cycle.start_dream_cycle(args.hours, callback=progress_callback)

        print("\n" + "=" * 70)
        print("DREAM CYCLE COMPLETE")
        print("=" * 70)
        print(f"Duration: {result.duration_seconds:.1f} seconds")
        print(f"Memories replayed: {len(result.memories_replayed)}")
        print(f"New associations: {len(result.new_associations)}")
        print(f"Consolidation score: {result.consolidation_score:.3f}")
        print(f"Promotions: {len(result.promotions)}")

        # SWR statistics (2025 Neuron enhancement)
        stats = result.statistics
        if "swr_type_distribution" in stats:
            print("\n--- SWR Type Distribution (2025 Neuron) ---")
            dist = stats["swr_type_distribution"]
            print(f"  Small SWRs:  {dist.get('small_count', 0)}")
            print(f"  Medium SWRs: {dist.get('medium_count', 0)}")
            print(f"  Large SWRs:  {dist.get('large_count', 0)}")
            print(f"  Large SWR Rate: {stats.get('large_swr_rate', 0):.1%}")
            if stats.get("boost_count", 0) > 0:
                print(f"  Boosts Applied: {stats['boost_count']}")
                print(f"  Avg Boost Effectiveness: {stats.get('avg_boost_effectiveness', 0):.2f}")

        if result.promotions:
            print("\nPromoted memories:")
            for mem_id, new_tier in result.promotions.items():
                print(f"  {mem_id} -> {new_tier}")

    elif args.command == "status":
        cycle = get_dream_cycle()
        journal = get_dream_journal()

        print("=" * 70)
        print("DREAM CYCLE STATUS")
        print("=" * 70)
        print(f"Current phase: {cycle.current_phase.value}")
        print(f"Is dreaming: {cycle.is_dreaming()}")
        print(f"Cycle count: {cycle.cycle_count}")
        print(f"\nJournal entries: {len(journal.entries)}")

        stats = journal.get_journal_statistics()
        print(f"Total replays: {stats['total_replays']}")
        print(f"Total associations: {stats['total_associations']}")
        print(f"Average consolidation: {stats['average_consolidation']:.3f}")

    elif args.command == "journal":
        journal = get_dream_journal()
        entries = journal.get_entries(5)

        print("=" * 70)
        print("DREAM JOURNAL (Recent Entries)")
        print("=" * 70)

        if not entries:
            print("No dream journal entries yet.")
        else:
            for entry in entries:
                print(f"\n[{entry.entry_id}] {entry.timestamp}")
                print(f"  {entry.dream_narrative}")
                if entry.insights:
                    print(f"  Insights: {', '.join(entry.insights)}")

    elif args.command == "candidates":
        cycle = get_dream_cycle()
        candidates = cycle.memory_loader.get_consolidation_candidates()

        print("=" * 70)
        print("CONSOLIDATION CANDIDATES")
        print("=" * 70)
        print(f"Found {len(candidates)} candidates\n")

        for mem in candidates[:10]:
            print(f"[{mem.id}]")
            print(f"  Content: {mem.content[:60]}...")
            print(f"  Tier: {mem.tier} | Salience: {mem.salience:.3f}")
            print(f"  Priority: {mem.consolidation_priority:.3f}")
            print()

    elif args.command == "replay":
        if not args.args:
            print("Usage: replay <memory_id>")
            return 1

        memory_id = args.args[0]
        result = replay_memory(memory_id)

        print("=" * 70)
        print("MEMORY REPLAY RESULT")
        print("=" * 70)
        print(f"Memory ID: {result.memory_id}")
        print(f"Success: {result.success}")
        print(f"Consolidation strength: {result.consolidation_strength:.3f}")
        print(f"New replay count: {result.new_replay_count}")
        print(f"Promoted: {result.promoted}")
        if result.new_tier:
            print(f"New tier: {result.new_tier}")
        if result.error:
            print(f"Error: {result.error}")

    elif args.command == "demo":
        print("=" * 70)
        print("VIRGIL DREAM CYCLE - DEMONSTRATION")
        print("(With 2025 Neuron SWR Type Discrimination)")
        print("=" * 70)

        # Short demo cycle with boosting enabled
        cycle = DreamCycle(simulation_speed=600.0, enable_swr_boosting=True)

        print("\nStarting 15-minute (simulated) dream cycle...")
        print("SWR boosting ENABLED for enhanced consolidation.\n")

        def demo_callback(phase: str, status: Dict):
            boost_marker = " [SWR+]" if status.get("swr_boosting_enabled") else ""
            print(f"  [{phase.upper()}]{boost_marker} Starting...")

        result = cycle.start_dream_cycle(0.25, callback=demo_callback)

        print("\n" + "-" * 40)
        print("DEMO RESULTS")
        print("-" * 40)
        print(f"Phases completed: {len(result.phases_completed)}")
        print(f"Memories processed: {len(result.memories_replayed)}")
        print(f"Associations formed: {len(result.new_associations)}")
        print(f"Sharp-wave ripples: {len(result.sharp_wave_ripples)}")
        print(f"Consolidation score: {result.consolidation_score:.1%}")

        # SWR type breakdown
        stats = result.statistics
        if "swr_type_distribution" in stats:
            dist = stats["swr_type_distribution"]
            print(f"\nSWR Distribution: S={dist.get('small_count', 0)} "
                  f"M={dist.get('medium_count', 0)} L={dist.get('large_count', 0)}")
            print(f"Large SWR Rate: {stats.get('large_swr_rate', 0):.1%}")

        print("\n[Demo complete. Run 'dream' command for full cycle.]")

    elif args.command == "swr":
        print("=" * 70)
        print("SWR TYPE STATISTICS (2025 Neuron Enhancement)")
        print("=" * 70)

        cycle = get_dream_cycle()
        stats = cycle.swr_statistics

        print("\nSWR Type Distribution:")
        dist = stats.get("swr_distribution", {})
        total = dist.get("total_count", 0)
        if total > 0:
            print(f"  Small SWRs:  {dist.get('small_count', 0):4d} "
                  f"({dist.get('small_count', 0)/total*100:.1f}%)")
            print(f"  Medium SWRs: {dist.get('medium_count', 0):4d} "
                  f"({dist.get('medium_count', 0)/total*100:.1f}%)")
            print(f"  Large SWRs:  {dist.get('large_count', 0):4d} "
                  f"({dist.get('large_count', 0)/total*100:.1f}%)")
            print(f"  Total:       {total:4d}")
        else:
            print("  No SWR data available yet. Run a dream cycle first.")

        print(f"\nLarge SWR Rate: {stats.get('large_swr_rate', 0):.1%}")
        print(f"  (Optimal target: ~30%)")

        # HPC-PFC coordination summary
        coord = stats.get("hpc_pfc_coordination_summary", {})
        if coord.get("total_events", 0) > 0:
            print(f"\nHPC-PFC Coordination:")
            print(f"  Total events:      {coord.get('total_events', 0)}")
            print(f"  Avg sync:          {coord.get('avg_sync', 0):.3f}")
            print(f"  Avg reactivation:  {coord.get('avg_reactivation', 0):.3f}")

        print("\n--- SWR Type Classification Thresholds ---")
        print("  Small:  amplitude < 0.4  (background activity)")
        print("  Medium: 0.4 <= amplitude < 0.7 (standard consolidation)")
        print("  Large:  amplitude >= 0.7 (priority consolidation)")

    elif args.command == "boost":
        print("=" * 70)
        print("SWR BOOST HISTORY (2025 Neuron Enhancement)")
        print("=" * 70)

        cycle = get_dream_cycle()

        if not cycle.boost_controller:
            print("\nSWR boosting is currently DISABLED.")
            print("Enable with: enable_swr_boosting(True)")
            return 0

        history = cycle.boost_controller.get_boost_history()

        if not history:
            print("\nNo boost events recorded yet.")
            print("Run a dream cycle with boosting enabled to see activity.")
        else:
            print(f"\nTotal boosts applied: {len(history)}")
            print("\nRecent boost events:")

            for i, event in enumerate(history[-5:], 1):
                print(f"\n  [{i}] {event.timestamp}")
                print(f"      Reason: {event.trigger_reason}")
                print(f"      Targets: {len(event.target_memories)} memories")
                print(f"      Resulting SWR: {event.resulting_swr_type.upper()}")
                print(f"      Effectiveness: {event.effectiveness:.2f}")

        # Boost statistics
        stats = cycle.boost_controller.get_boost_statistics()
        print("\n--- Boost Statistics ---")
        print(f"  Total boosts:           {stats['total_boosts']}")
        print(f"  Boost rate:             {stats['boost_rate']:.1%}")
        print(f"  Avg effectiveness:      {stats['avg_boost_effectiveness']:.3f}")
        print(f"  Recent efficiency:      {stats['recent_efficiency']:.3f}")

        print("\n--- Closed-Loop Boosting Thresholds ---")
        print(f"  Optimal large SWR rate: {SWRBoostController.OPTIMAL_LARGE_SWR_RATE:.0%}")
        print(f"  Min boost interval:     {SWRBoostController.MIN_BOOST_INTERVAL}s")
        print(f"  High priority threshold:{SWRBoostController.HIGH_PRIORITY_THRESHOLD:.2f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
