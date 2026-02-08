#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
VIRGIL DREAM CONSOLIDATION - Core Consolidation Logic

Extracted from virgil_dream_cycle.py for modularity.
Contains the core memory consolidation classes:
- ConsolidationEngine: Orchestrates SWS/REM consolidation
- SWRBoostController: Closed-loop SWR enhancement

Author: Virgil Permanence Build
Date: 2026-01-28
"""

import random
import logging
import json
import re
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set, Any

# ============================================================================
# PATHS
# ============================================================================

TABERNACLE_ROOT = Path(__file__).parent.parent
NEXUS_DIR = TABERNACLE_ROOT / "00_NEXUS"

# Import from virgil_dream_models (shared data structures)
from virgil_dream_models import (
    # Enums
    DreamPhase,
    SWRType,
    # Data structures
    Memory,
    ReplayEvent,
    SWRDistribution,
    ReplayResult,
    # Classes
    MemoryLoader,
    MemoryReplay,
    # Constants
    CONSOLIDATION_REPLAY_COUNT,
)


# ============================================================================
# DATA STRUCTURES (local to consolidation module)
# ============================================================================

@dataclass
class AssociationEvent:
    """
    Record of a new association formed during REM.

    REM sleep enables creative connections between
    disparate memories that share thematic resonance.
    """
    memory_id_1: str
    memory_id_2: str
    timestamp: str
    association_type: str           # thematic, emotional, temporal, semantic
    similarity_score: float         # 0.0 to 1.0
    creative_distance: float        # How "far apart" the memories were

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SharpWaveRipple:
    """
    Simulated sharp-wave ripple event.

    In biology, SWRs are brief (~100ms) bursts of hippocampal activity
    that compress and replay memory sequences at high speed.

    Enhanced with 2025 Neuron paper findings on Large Sharp-Wave Ripples:
    - Ripple magnitude predicts consolidation strength
    - Large ripples show strongest hippocampal-cortical coordination
    - Closed-loop boosting can enhance consolidation during critical periods
    """
    timestamp: str
    memories_activated: List[str]
    ripple_duration_ms: int
    ripple_strength: float          # 0.0 to 1.0 (normalized amplitude)
    phase: str
    # New fields for SWR type discrimination
    swr_type: str = "medium"        # small, medium, large
    hpc_pfc_sync: float = 0.5       # Hippocampus-prefrontal synchrony (0.0-1.0)
    reactivation_strength: float = 0.5  # Memory reactivation intensity (0.0-1.0)
    boosted: bool = False           # Whether this SWR was artificially boosted

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def swr_type_enum(self) -> SWRType:
        """Get the SWRType enum value."""
        return SWRType(self.swr_type)

    @property
    def consolidation_efficiency(self) -> float:
        """
        Calculate actual memory strengthening potential.

        Based on the interaction between ripple strength, HPC-PFC synchrony,
        and reactivation intensity.
        """
        base = self.ripple_strength * SWRType(self.swr_type).consolidation_multiplier
        sync_factor = 0.5 + (self.hpc_pfc_sync * 0.5)  # 0.5 to 1.0
        reactivation_factor = 0.5 + (self.reactivation_strength * 0.5)  # 0.5 to 1.0
        return min(1.0, base * sync_factor * reactivation_factor)


@dataclass
class HPCPFCCoordination:
    """
    Hippocampal-Prefrontal Cortex coordination metrics.

    Based on 2025 Neuron findings that large SWRs drive stronger
    HPC-PFC coupling, enabling more efficient memory consolidation.
    """
    timestamp: str
    hpc_pfc_sync: float             # Cross-region synchrony (0.0-1.0)
    reactivation_strength: float    # Memory pattern reactivation (0.0-1.0)
    consolidation_efficiency: float # Actual memory strengthening (0.0-1.0)
    cortical_engagement: float      # PFC involvement level (0.0-1.0)
    memory_ids: List[str]           # Memories involved in this coordination event

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SWRBoostEvent:
    """
    Record of a closed-loop SWR boost event.

    Closed-loop boosting artificially triggers large SWRs when
    consolidation need is detected but natural large ripples are insufficient.
    """
    timestamp: str
    trigger_reason: str             # Why boosting was triggered
    target_memories: List[str]      # Memories targeted for enhanced consolidation
    pre_boost_swr_rate: float       # Large SWR rate before boost
    boost_strength: float           # Applied boost intensity (0.0-1.0)
    resulting_swr_type: str         # SWR type achieved after boost
    effectiveness: float            # How effective the boost was (0.0-1.0)

    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# CONSTANTS
# ============================================================================

# Replay parameters (local copies to avoid cross-module dependencies)
MAX_REPLAY_PER_CYCLE = 10       # Maximum memories replayed per sleep cycle
REPLAY_THRESHOLD_PERCENTILE = 0.90  # Top 10% salience for replay
# CONSOLIDATION_REPLAY_COUNT imported from virgil_dream_models
ASSOCIATION_MIN_SIMILARITY = 0.3  # Minimum similarity for REM association

# Logger setup
logger = logging.getLogger(__name__)

# Import RelationalMemoryV2 for RIE edge pruning during dreams
try:
    from rie_relational_memory_v2 import RelationalMemoryV2
except ImportError:
    RelationalMemoryV2 = None  # Optional dependency


def prune_weak_edges(
    rie: "RelationalMemoryV2",
    min_weight: float = 0.1,
    max_age_days: float = 30.0
) -> int:
    """
    Prune edges that are:
    1. Below min_weight after decay
    2. Older than max_age_days without activation
    3. Connected to orphan nodes (degree 1)
    
    Skips H1-locked edges (protected topology).
    
    Args:
        rie: RelationalMemoryV2 instance
        min_weight: Minimum effective weight threshold
        max_age_days: Maximum days since last activation
        
    Returns:
        Count of pruned edges.
    """
    now = datetime.now(timezone.utc)
    pruned = 0
    edges_to_remove = []
    
    for edge_id, edge in rie.edges.items():
        # Skip H1-locked edges (protected topology)
        if getattr(edge, 'is_h1_locked', False):
            continue
        
        # Check effective weight after decay
        if hasattr(edge, 'effective_weight'):
            if hasattr(edge, 'w_slow'):  # BiologicalEdge
                eff_weight = edge.effective_weight(t_now=now.timestamp(), global_p=rie.current_coherence)
            else:  # Old Edge format
                eff_weight = edge.effective_weight(now)
        else:
            eff_weight = getattr(edge, 'weight', 0.5)
        
        if eff_weight < min_weight:
            edges_to_remove.append(edge_id)
            continue
        
        # Check age since last activation
        last_activated = getattr(edge, 'last_activated', None) or getattr(edge, 'last_spike', None)
        if last_activated:
            try:
                if isinstance(last_activated, (int, float)):
                    # Timestamp format (BiologicalEdge)
                    last_dt = datetime.fromtimestamp(last_activated, tz=timezone.utc)
                else:
                    # ISO string format
                    last_dt = datetime.fromisoformat(str(last_activated).replace('Z', '+00:00'))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                
                days_since = (now - last_dt).total_seconds() / 86400
                if days_since > max_age_days:
                    edges_to_remove.append(edge_id)
                    continue
            except Exception:
                pass
    
    # Check for orphan-creating edges (degree 1 nodes)
    # Count degree for each node
    node_degree = {}
    for eid, edge in rie.edges.items():
        if eid in edges_to_remove:
            continue  # Already marked for removal
        node_degree[edge.source_id] = node_degree.get(edge.source_id, 0) + 1
        node_degree[edge.target_id] = node_degree.get(edge.target_id, 0) + 1
    
    # Find edges to orphan nodes (would leave node with degree 0 after all removals)
    for edge_id, edge in rie.edges.items():
        if edge_id in edges_to_remove:
            continue
        if getattr(edge, 'is_h1_locked', False):
            continue
        
        # If either endpoint has degree 1, this edge connects to an orphan
        source_deg = node_degree.get(edge.source_id, 0)
        target_deg = node_degree.get(edge.target_id, 0)
        
        if source_deg <= 1 or target_deg <= 1:
            edges_to_remove.append(edge_id)
    
    # Remove edges (deduplicate first)
    edges_to_remove = list(set(edges_to_remove))
    for edge_id in edges_to_remove:
        if hasattr(rie, 'remove_edge'):
            rie.remove_edge(edge_id)
        else:
            # Fallback inline removal
            if edge_id in rie.edges:
                edge = rie.edges[edge_id]
                rie.outgoing.get(edge.source_id, set()).discard(edge_id)
                rie.incoming.get(edge.target_id, set()).discard(edge_id)
                del rie.edges[edge_id]
        pruned += 1
    
    return pruned


# ============================================================================
# CONSOLIDATION ENGINE
# ============================================================================

class ConsolidationEngine:
    """
    Orchestrates memory consolidation during sleep.

    Implements the two-stage model:
    1. SWS: Stabilizes memories through replay
    2. REM: Integrates memories through association
    """

    def __init__(self, memory_loader: MemoryLoader, memory_replay: MemoryReplay):
        self.loader = memory_loader
        self.replay = memory_replay
        self.association_log: List[AssociationEvent] = []
        self.ripple_log: List[SharpWaveRipple] = []

    def consolidate_sws(
        self,
        cycle_position: float = 0.5,
        boost_controller: Optional["SWRBoostController"] = None
    ) -> List[ReplayResult]:
        """
        Execute SWS consolidation phase with SWR type discrimination.

        Enhanced with 2025 Neuron paper findings:
        - Small SWRs: Background maintenance, minimal consolidation
        - Medium SWRs: Standard consolidation, moderate replay
        - Large SWRs: Priority consolidation with explicit memory protection

        Args:
            cycle_position: Position in night (0.0 = start, 1.0 = end)
                           SWS is more intense in first half
            boost_controller: Optional SWRBoostController for closed-loop enhancement

        Returns:
            List of replay results
        """
        # SWS intensity decreases through the night
        intensity = max(0.3, 1.0 - cycle_position * 0.7)

        # Get consolidation candidates
        candidates = self.loader.get_consolidation_candidates(REPLAY_THRESHOLD_PERCENTILE)

        if not candidates:
            logger.info("No consolidation candidates available")
            return []

        # Number of replays scaled by intensity
        num_replays = int(MAX_REPLAY_PER_CYCLE * intensity)

        # Select memories for replay (weighted by consolidation priority)
        priorities = [m.consolidation_priority for m in candidates]
        total_priority = sum(priorities)

        if total_priority == 0:
            weights = [1.0 / len(candidates)] * len(candidates)
        else:
            weights = [p / total_priority for p in priorities]

        # Weighted random selection
        selected = []
        for _ in range(min(num_replays, len(candidates))):
            r = random.random()
            cumulative = 0.0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative and candidates[i] not in selected:
                    selected.append(candidates[i])
                    break

        # Track SWR type distribution for this consolidation phase
        swr_type_counts = {SWRType.SMALL: 0, SWRType.MEDIUM: 0, SWRType.LARGE: 0}

        # Execute replays with SWR type-dependent consolidation
        results = []
        for memory in selected:
            # Simulate sharp-wave ripple burst with type discrimination
            ripple = self._simulate_sharp_wave_ripple(memory, DreamPhase.DEEP_SLEEP)
            swr_type = SWRType(ripple.swr_type)
            swr_type_counts[swr_type] += 1

            # Register with boost controller if available
            if boost_controller:
                boost_controller.record_natural_swr(ripple)

            # Find context memories for co-activation
            context = self._find_context_memories(memory, candidates)

            # Apply SWR type-specific consolidation strategy
            result = self._apply_swr_type_consolidation(
                memory, swr_type, ripple, context
            )
            results.append(result)

            logger.debug(
                f"SWS replay: {memory.id} | swr_type={swr_type.value} | "
                f"strength={result.consolidation_strength:.3f}"
            )

        # Check if closed-loop boosting is needed
        if boost_controller:
            need_boost, reason, target_ids = boost_controller.detect_consolidation_need(
                DreamPhase.DEEP_SLEEP, cycle_position
            )
            if need_boost and target_ids:
                boosted_swr, boost_event = boost_controller.boost_swr(
                    target_ids, reason
                )
                self.ripple_log.append(boosted_swr)

                # Apply boosted consolidation to target memories
                for target_id in target_ids[:3]:  # Limit to 3 per boost
                    target_memory = self.loader.get_memory(target_id)
                    if target_memory:
                        boost_result = self._apply_swr_type_consolidation(
                            target_memory,
                            SWRType.LARGE,
                            boosted_swr,
                            []
                        )
                        results.append(boost_result)

        logger.info(
            f"SWS consolidation: {len(results)} memories replayed | "
            f"SWR distribution: S={swr_type_counts[SWRType.SMALL]} "
            f"M={swr_type_counts[SWRType.MEDIUM]} L={swr_type_counts[SWRType.LARGE]}"
        )
        return results

    def _apply_swr_type_consolidation(
        self,
        memory: Memory,
        swr_type: SWRType,
        ripple: SharpWaveRipple,
        context: List[str]
    ) -> ReplayResult:
        """
        Apply SWR type-specific consolidation strategy.

        - Small SWRs: Minimal consolidation, background maintenance
        - Medium SWRs: Standard replay and consolidation
        - Large SWRs: Priority consolidation with protection eligibility
        """
        # Base consolidation strength modified by SWR type
        base_result = self.replay.replay_memory(
            memory.id,
            DreamPhase.DEEP_SLEEP,
            context
        )

        # Modify consolidation strength based on SWR type
        modified_strength = base_result.consolidation_strength * swr_type.consolidation_multiplier

        # Large SWRs can boost salience more significantly
        if swr_type == SWRType.LARGE:
            # Enhanced consolidation for large ripples
            memory.salience = min(1.0, memory.salience + modified_strength * 0.15)

            # Check for explicit memory protection
            if swr_type.protection_eligible and ripple.consolidation_efficiency > 0.7:
                if (memory.replay_count >= CONSOLIDATION_REPLAY_COUNT and
                    memory.consolidation_priority >= 0.75):
                    memory.protected = True
                    self.loader.update_memory(memory)
                    logger.info(
                        f"Memory {memory.id} PROTECTED via large SWR "
                        f"(efficiency={ripple.consolidation_efficiency:.2f})"
                    )

        elif swr_type == SWRType.SMALL:
            # Minimal consolidation - just background maintenance
            memory.salience = min(1.0, memory.salience + modified_strength * 0.02)

        else:  # MEDIUM
            # Standard consolidation
            memory.salience = min(1.0, memory.salience + modified_strength * 0.08)

        self.loader.update_memory(memory)

        return ReplayResult(
            memory_id=base_result.memory_id,
            success=base_result.success,
            consolidation_strength=modified_strength,
            new_replay_count=base_result.new_replay_count,
            promoted=base_result.promoted,
            new_tier=base_result.new_tier,
            error=base_result.error
        )

    def integrate_rem(self, cycle_position: float = 0.5) -> List[AssociationEvent]:
        """
        Execute REM integration phase.

        Args:
            cycle_position: Position in night (0.0 = start, 1.0 = end)
                           REM is more intense in second half

        Returns:
            List of new associations formed
        """
        # REM intensity increases through the night
        intensity = max(0.3, cycle_position * 0.7 + 0.3)

        # Get integration candidates
        candidates = self.loader.get_integration_candidates()

        if len(candidates) < 2:
            logger.info("Not enough memories for REM integration")
            return []

        # Number of association attempts scaled by intensity
        num_attempts = int(MAX_REPLAY_PER_CYCLE * intensity * 0.5)

        new_associations = []
        attempted_pairs: Set[Tuple[str, str]] = set()

        for _ in range(num_attempts):
            # Select two memories for potential association
            if len(candidates) < 2:
                break

            # Weight selection by integration potential
            potentials = [m.integration_potential for m in candidates]
            total_potential = sum(potentials)

            if total_potential == 0:
                continue

            weights = [p / total_potential for p in potentials]

            # Select first memory
            idx1 = self._weighted_random_select(weights)
            mem1 = candidates[idx1]

            # Select second memory (different from first, prefer dissimilar for creativity)
            weights2 = weights.copy()
            weights2[idx1] = 0.0  # Exclude first memory

            # Boost weight for memories that are somewhat dissimilar (creative distance)
            for i, mem in enumerate(candidates):
                if i != idx1:
                    similarity = self.replay._calculate_similarity(mem1, mem)
                    # Prefer moderate dissimilarity (0.2-0.5 range)
                    if 0.2 <= similarity <= 0.5:
                        weights2[i] *= 1.5

            # Normalize
            total2 = sum(weights2)
            if total2 == 0:
                continue
            weights2 = [w / total2 for w in weights2]

            idx2 = self._weighted_random_select(weights2)
            mem2 = candidates[idx2]

            # Check if already attempted
            pair = tuple(sorted([mem1.id, mem2.id]))
            if pair in attempted_pairs:
                continue
            attempted_pairs.add(pair)

            # Attempt association
            association = self._attempt_association(mem1, mem2)
            if association:
                new_associations.append(association)

                # Update memories with new association
                if mem2.id not in mem1.associations:
                    mem1.associations.append(mem2.id)
                if mem1.id not in mem2.associations:
                    mem2.associations.append(mem1.id)

                self.loader.update_memory(mem1)
                self.loader.update_memory(mem2)

                logger.debug(f"REM association: {mem1.id} <-> {mem2.id} | type={association.association_type}")

        logger.info(f"REM integration: {len(new_associations)} new associations formed")
        return new_associations

    def _simulate_sharp_wave_ripple(
        self,
        primary_memory: Memory,
        phase: DreamPhase,
        force_type: Optional[SWRType] = None
    ) -> SharpWaveRipple:
        """
        Simulate a sharp-wave ripple event with SWR type discrimination.

        Enhanced with 2025 Neuron paper findings:
        - Classifies ripples by magnitude (small/medium/large)
        - Large ripples have stronger HPC-PFC coordination
        - Large ripples trigger memory protection eligibility

        SWRs are ~100ms bursts that can activate multiple related memories.

        Args:
            primary_memory: Primary memory being replayed
            phase: Current dream phase
            force_type: If provided, force this SWR type (for boosting)

        Returns:
            The generated SharpWaveRipple event
        """
        # Find memories that might co-activate
        all_memories = list(self.loader.memories.values())
        co_activated = [primary_memory.id]

        for mem in all_memories:
            if mem.id == primary_memory.id:
                continue

            similarity = self.replay._calculate_similarity(primary_memory, mem)
            # High similarity memories may co-activate
            if similarity > 0.5 and random.random() < similarity:
                co_activated.append(mem.id)
                if len(co_activated) >= 5:  # Limit co-activation
                    break

        # Generate ripple amplitude with natural distribution
        # Most ripples are small/medium, large ripples are rarer but more impactful
        if force_type:
            # Forced type (e.g., from boosting)
            if force_type == SWRType.LARGE:
                ripple_strength = random.uniform(0.75, 1.0)
            elif force_type == SWRType.MEDIUM:
                ripple_strength = random.uniform(0.45, 0.65)
            else:
                ripple_strength = random.uniform(0.15, 0.35)
        else:
            # Natural distribution: ~60% small, ~30% medium, ~10% large
            roll = random.random()
            if roll < 0.60:
                ripple_strength = random.uniform(0.15, 0.38)  # Small
            elif roll < 0.90:
                ripple_strength = random.uniform(0.42, 0.68)  # Medium
            else:
                ripple_strength = random.uniform(0.72, 1.0)   # Large

        # Classify SWR type
        swr_type = SWRType.from_amplitude(ripple_strength)

        # Calculate HPC-PFC synchrony based on SWR type
        # Large ripples drive stronger hippocampal-cortical coupling
        hpc_pfc_sync = self._calculate_swr_hpc_pfc_sync(swr_type)

        # Calculate reactivation strength based on memory properties
        reactivation_strength = self._calculate_memory_reactivation(
            primary_memory, co_activated
        )

        # Ripple duration varies with magnitude (large ripples slightly longer)
        base_duration = 100
        duration_modifier = {
            SWRType.SMALL: -10,
            SWRType.MEDIUM: 0,
            SWRType.LARGE: 20
        }[swr_type]
        ripple_duration = int(random.gauss(base_duration + duration_modifier, 15))

        ripple = SharpWaveRipple(
            timestamp=datetime.now(timezone.utc).isoformat(),
            memories_activated=co_activated,
            ripple_duration_ms=ripple_duration,
            ripple_strength=ripple_strength,
            phase=phase.value,
            swr_type=swr_type.value,
            hpc_pfc_sync=hpc_pfc_sync,
            reactivation_strength=reactivation_strength,
            boosted=force_type is not None
        )
        self.ripple_log.append(ripple)

        # Large SWRs trigger special handling
        if swr_type == SWRType.LARGE:
            self._handle_large_swr(ripple, primary_memory)

        return ripple

    def _calculate_swr_hpc_pfc_sync(self, swr_type: SWRType) -> float:
        """
        Calculate hippocampal-prefrontal cortex synchrony for an SWR.

        Based on 2025 Neuron findings: large ripples show significantly
        stronger HPC-PFC coupling, enabling more efficient memory transfer.
        """
        sync_ranges = {
            SWRType.SMALL: (0.15, 0.35),
            SWRType.MEDIUM: (0.40, 0.60),
            SWRType.LARGE: (0.70, 0.95)
        }
        low, high = sync_ranges[swr_type]
        return random.uniform(low, high)

    def _calculate_memory_reactivation(
        self,
        primary: Memory,
        co_activated_ids: List[str]
    ) -> float:
        """
        Calculate the reactivation strength of memory patterns.

        Stronger reactivation occurs for:
        - High salience memories
        - Emotionally arousing memories
        - Well-connected memories (many co-activations)
        """
        # Primary memory contribution
        emotional_intensity = (abs(primary.emotional_valence) + primary.emotional_arousal) / 2
        base_strength = (
            primary.salience * 0.4 +
            emotional_intensity * 0.3 +
            primary.resonance * 0.3
        )

        # Ensemble bonus from co-activations
        ensemble_bonus = min(0.15, len(co_activated_ids) * 0.03)

        return min(1.0, base_strength + ensemble_bonus)

    def _handle_large_swr(self, ripple: SharpWaveRipple, primary_memory: Memory):
        """
        Handle special processing for large SWR events.

        Large ripples (top 10%) indicate strong hippocampal-cortical
        coordination and should trigger priority consolidation effects.
        """
        # Log the large SWR event
        logger.info(
            f"LARGE SWR detected: strength={ripple.ripple_strength:.2f} | "
            f"hpc_pfc_sync={ripple.hpc_pfc_sync:.2f} | "
            f"memories={len(ripple.memories_activated)}"
        )

        # Large SWRs can promote memories to protected status
        # if they've had sufficient replay and high priority
        if (primary_memory.replay_count >= CONSOLIDATION_REPLAY_COUNT - 1 and
            primary_memory.consolidation_priority >= 0.7 and
            not primary_memory.protected):

            # Mark for protection consideration
            # (actual protection happens after cycle completion)
            if not hasattr(self, '_protection_candidates'):
                self._protection_candidates = []
            self._protection_candidates.append(primary_memory.id)

            logger.info(f"Memory {primary_memory.id} marked for protection consideration")

    def _find_context_memories(
        self,
        target: Memory,
        candidates: List[Memory],
        max_context: int = 3
    ) -> List[str]:
        """Find related memories to co-activate during replay."""
        similarities = []
        for mem in candidates:
            if mem.id == target.id:
                continue
            sim = self.replay._calculate_similarity(target, mem)
            if sim > 0.3:
                similarities.append((mem.id, sim))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similarities[:max_context]]

    def _attempt_association(
        self,
        mem1: Memory,
        mem2: Memory
    ) -> Optional[AssociationEvent]:
        """
        Attempt to form an association between two memories.

        Returns AssociationEvent if successful, None otherwise.
        """
        similarity = self.replay._calculate_similarity(mem1, mem2)

        if similarity < ASSOCIATION_MIN_SIMILARITY:
            return None

        # Determine association type
        association_type = self._determine_association_type(mem1, mem2)

        # Creative distance (how "far apart" the memories are conceptually)
        creative_distance = 1.0 - similarity

        # Higher creative distance with minimum similarity = more valuable association
        if creative_distance > 0.5 and similarity > 0.3:
            # This is a creative leap - boost significance
            pass

        association = AssociationEvent(
            memory_id_1=mem1.id,
            memory_id_2=mem2.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            association_type=association_type,
            similarity_score=similarity,
            creative_distance=creative_distance
        )

        self.association_log.append(association)
        return association

    def _determine_association_type(self, mem1: Memory, mem2: Memory) -> str:
        """Determine the type of association between two memories."""
        # Check emotional similarity
        valence_diff = abs(mem1.emotional_valence - mem2.emotional_valence)
        arousal_diff = abs(mem1.emotional_arousal - mem2.emotional_arousal)

        if valence_diff < 0.3 and arousal_diff < 0.3:
            return "emotional"

        # Check thematic (tag overlap)
        tag_overlap = len(set(mem1.tags) & set(mem2.tags))
        if tag_overlap > 0:
            return "thematic"

        # Check resonance similarity (Z_Omega alignment)
        if abs(mem1.resonance - mem2.resonance) < 0.2:
            return "resonant"

        # Default to semantic
        return "semantic"

    def _weighted_random_select(self, weights: List[float]) -> int:
        """Select an index based on weights."""
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def calculate_consolidation_score(
        self,
        replay_results: List[ReplayResult],
        associations: List[AssociationEvent]
    ) -> float:
        """
        Calculate overall consolidation effectiveness.

        Factors:
        - Number and strength of replays
        - Quality of new associations
        - Promotions achieved
        """
        if not replay_results and not associations:
            return 0.0

        # Replay score
        replay_score = 0.0
        if replay_results:
            avg_strength = sum(r.consolidation_strength for r in replay_results) / len(replay_results)
            promotion_bonus = sum(0.1 for r in replay_results if r.promoted)
            replay_score = avg_strength + promotion_bonus

        # Association score
        assoc_score = 0.0
        if associations:
            avg_creativity = sum(a.creative_distance for a in associations) / len(associations)
            assoc_score = min(1.0, len(associations) / 5) * (0.5 + avg_creativity * 0.5)

        # Combined score
        combined = (replay_score * 0.6 + assoc_score * 0.4)
        return min(1.0, combined)

    def get_association_log(self) -> List[AssociationEvent]:
        """Get log of all association events."""
        return self.association_log

    def get_ripple_log(self) -> List[SharpWaveRipple]:
        """Get log of all sharp-wave ripple events."""
        return self.ripple_log

    def clear_logs(self):
        """Clear all logs."""
        self.association_log = []
        self.ripple_log = []


# ============================================================================
# SWR BOOST CONTROLLER (Closed-Loop Enhancement)
# ============================================================================

class SWRBoostController:
    """
    Closed-loop SWR boosting system.

    Based on the 2025 Neuron paper demonstrating that closed-loop stimulation
    timed to naturally occurring SWRs can enhance memory consolidation by
    increasing the rate of large ripples.

    Boosting is triggered when:
    1. High-priority memories need consolidation
    2. Natural large SWR rate is below optimal threshold
    3. Current sleep phase supports consolidation (SWS)
    """

    # Configuration thresholds
    OPTIMAL_LARGE_SWR_RATE = 0.3    # Target: 30% of SWRs should be large
    MIN_BOOST_INTERVAL = 5.0        # Minimum seconds between boosts
    BOOST_EFFECTIVENESS_DECAY = 0.9 # Diminishing returns on repeated boosts
    HIGH_PRIORITY_THRESHOLD = 0.7   # Consolidation priority threshold for boost consideration

    def __init__(self, memory_loader: MemoryLoader):
        self.loader = memory_loader
        self.boost_history: List[SWRBoostEvent] = []
        self.swr_distribution = SWRDistribution()
        self.hpc_pfc_coordination_log: List[HPCPFCCoordination] = []
        self.last_boost_time: Optional[datetime] = None
        self.boost_count = 0
        self.cumulative_boost_effectiveness = 0.0

    def detect_consolidation_need(
        self,
        current_phase: DreamPhase,
        cycle_position: float
    ) -> Tuple[bool, str, List[str]]:
        """
        Detect whether closed-loop boosting is needed.

        Args:
            current_phase: Current dream phase
            cycle_position: Position in the night (0.0 to 1.0)

        Returns:
            Tuple of (need_boost, reason, target_memory_ids)
        """
        # Only boost during deep sleep (SWS)
        if current_phase != DreamPhase.DEEP_SLEEP:
            return False, "not_sws_phase", []

        # Check if we've boosted too recently
        if self.last_boost_time:
            elapsed = (datetime.now(timezone.utc) - self.last_boost_time).total_seconds()
            if elapsed < self.MIN_BOOST_INTERVAL:
                return False, "boost_cooldown", []

        # Check current large SWR rate
        current_large_rate = self.swr_distribution.large_swr_rate

        # Find high-priority memories that haven't been adequately consolidated
        high_priority_memories = self._find_unconsolidated_priorities()

        if not high_priority_memories:
            return False, "no_priority_memories", []

        # Decision logic
        reasons = []

        # Reason 1: Large SWR rate is below optimal
        if current_large_rate < self.OPTIMAL_LARGE_SWR_RATE:
            reasons.append(f"low_large_swr_rate({current_large_rate:.2f})")

        # Reason 2: High-priority memories need attention (early in night is critical)
        if cycle_position < 0.5 and len(high_priority_memories) >= 3:
            reasons.append(f"critical_memories({len(high_priority_memories)})")

        # Reason 3: Consolidation efficiency is dropping (based on recent ripples)
        recent_efficiency = self._calculate_recent_consolidation_efficiency()
        if recent_efficiency < 0.5:
            reasons.append(f"low_efficiency({recent_efficiency:.2f})")

        if reasons:
            target_ids = [m.id for m in high_priority_memories[:5]]
            return True, "|".join(reasons), target_ids

        return False, "optimal_state", []

    def boost_swr(
        self,
        target_memory_ids: List[str],
        reason: str,
        base_swr_strength: float = 0.5
    ) -> Tuple[SharpWaveRipple, SWRBoostEvent]:
        """
        Artificially trigger a large SWR through closed-loop stimulation.

        Args:
            target_memory_ids: Memory IDs to target for enhanced consolidation
            reason: Why this boost was triggered
            base_swr_strength: Natural ripple strength before boost

        Returns:
            Tuple of (boosted SharpWaveRipple, SWRBoostEvent record)
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Calculate boost strength with diminishing returns
        boost_strength = min(
            0.5,
            0.3 * (self.BOOST_EFFECTIVENESS_DECAY ** self.boost_count)
        )

        # Enhanced SWR parameters
        boosted_amplitude = min(1.0, base_swr_strength + boost_strength)
        swr_type = SWRType.from_amplitude(boosted_amplitude)

        # Large SWRs have stronger HPC-PFC synchrony
        hpc_pfc_sync = self._calculate_hpc_pfc_sync(swr_type, boosted=True)
        reactivation_strength = self._calculate_reactivation_strength(target_memory_ids)

        # Create the boosted SWR
        boosted_swr = SharpWaveRipple(
            timestamp=timestamp,
            memories_activated=target_memory_ids,
            ripple_duration_ms=int(random.gauss(120, 15)),  # Slightly longer due to boost
            ripple_strength=boosted_amplitude,
            phase=DreamPhase.DEEP_SLEEP.value,
            swr_type=swr_type.value,
            hpc_pfc_sync=hpc_pfc_sync,
            reactivation_strength=reactivation_strength,
            boosted=True
        )

        # Calculate boost effectiveness
        effectiveness = boosted_swr.consolidation_efficiency

        # Record boost event
        boost_event = SWRBoostEvent(
            timestamp=timestamp,
            trigger_reason=reason,
            target_memories=target_memory_ids,
            pre_boost_swr_rate=self.swr_distribution.large_swr_rate,
            boost_strength=boost_strength,
            resulting_swr_type=swr_type.value,
            effectiveness=effectiveness
        )

        # Update tracking
        self.boost_history.append(boost_event)
        self.swr_distribution.record(swr_type, boosted=True)
        self.last_boost_time = datetime.now(timezone.utc)
        self.boost_count += 1
        self.cumulative_boost_effectiveness += effectiveness

        # Record HPC-PFC coordination event
        coordination = HPCPFCCoordination(
            timestamp=timestamp,
            hpc_pfc_sync=hpc_pfc_sync,
            reactivation_strength=reactivation_strength,
            consolidation_efficiency=effectiveness,
            cortical_engagement=min(1.0, hpc_pfc_sync * 1.2),  # PFC engagement tracks sync
            memory_ids=target_memory_ids
        )
        self.hpc_pfc_coordination_log.append(coordination)

        logger.info(
            f"SWR BOOST: type={swr_type.value} | strength={boosted_amplitude:.2f} | "
            f"effectiveness={effectiveness:.2f} | targets={len(target_memory_ids)}"
        )

        return boosted_swr, boost_event

    def record_natural_swr(self, swr: SharpWaveRipple):
        """Record a naturally occurring SWR for tracking."""
        swr_type = SWRType.from_amplitude(swr.ripple_strength)
        self.swr_distribution.record(swr_type, boosted=swr.boosted)

        # Record HPC-PFC coordination for natural SWRs too
        if swr.hpc_pfc_sync > 0.3:  # Only log significant coordination
            coordination = HPCPFCCoordination(
                timestamp=swr.timestamp,
                hpc_pfc_sync=swr.hpc_pfc_sync,
                reactivation_strength=swr.reactivation_strength,
                consolidation_efficiency=swr.consolidation_efficiency,
                cortical_engagement=min(1.0, swr.hpc_pfc_sync * 1.1),
                memory_ids=swr.memories_activated
            )
            self.hpc_pfc_coordination_log.append(coordination)

    def _find_unconsolidated_priorities(self) -> List[Memory]:
        """Find high-priority memories that haven't been adequately consolidated."""
        candidates = []
        for memory in self.loader.memories.values():
            if memory.protected:
                continue
            if memory.consolidation_priority >= self.HIGH_PRIORITY_THRESHOLD:
                # Check if memory has been replayed enough
                if memory.replay_count < CONSOLIDATION_REPLAY_COUNT:
                    candidates.append(memory)

        # Sort by priority
        candidates.sort(key=lambda m: m.consolidation_priority, reverse=True)
        return candidates

    def _calculate_recent_consolidation_efficiency(self) -> float:
        """Calculate average consolidation efficiency from recent coordination events."""
        if not self.hpc_pfc_coordination_log:
            return 0.5  # Default moderate efficiency

        # Look at last 10 events
        recent = self.hpc_pfc_coordination_log[-10:]
        return sum(c.consolidation_efficiency for c in recent) / len(recent)

    def _calculate_hpc_pfc_sync(self, swr_type: SWRType, boosted: bool = False) -> float:
        """
        Calculate hippocampal-prefrontal synchrony based on SWR type.

        Large SWRs naturally have higher HPC-PFC coupling.
        """
        base_sync = {
            SWRType.SMALL: random.uniform(0.2, 0.4),
            SWRType.MEDIUM: random.uniform(0.4, 0.6),
            SWRType.LARGE: random.uniform(0.7, 0.9)
        }[swr_type]

        # Boosted SWRs have slightly enhanced synchrony
        if boosted:
            base_sync = min(1.0, base_sync + 0.1)

        return base_sync

    def _calculate_reactivation_strength(self, memory_ids: List[str]) -> float:
        """
        Calculate memory reactivation strength based on target memories.

        Higher salience memories reactivate more strongly.
        """
        if not memory_ids:
            return 0.5

        strengths = []
        for mid in memory_ids:
            memory = self.loader.get_memory(mid)
            if memory:
                # Reactivation correlates with salience and emotional intensity
                emotional_intensity = (abs(memory.emotional_valence) + memory.emotional_arousal) / 2
                strength = (memory.salience * 0.6 + emotional_intensity * 0.4)
                strengths.append(strength)

        if not strengths:
            return 0.5

        # Average with slight boost for multiple memories (ensemble reactivation)
        avg_strength = sum(strengths) / len(strengths)
        ensemble_bonus = min(0.1, len(strengths) * 0.02)
        return min(1.0, avg_strength + ensemble_bonus)

    def get_boost_statistics(self) -> Dict:
        """Get statistics about boost activity."""
        return {
            "total_boosts": self.boost_count,
            "swr_distribution": self.swr_distribution.to_dict(),
            "large_swr_rate": self.swr_distribution.large_swr_rate,
            "boost_rate": self.swr_distribution.boost_rate,
            "avg_boost_effectiveness": (
                self.cumulative_boost_effectiveness / max(1, self.boost_count)
            ),
            "coordination_events": len(self.hpc_pfc_coordination_log),
            "recent_efficiency": self._calculate_recent_consolidation_efficiency()
        }

    def get_boost_history(self) -> List[SWRBoostEvent]:
        """Get the history of all boost events."""
        return self.boost_history

    def get_coordination_log(self) -> List[HPCPFCCoordination]:
        """Get the log of HPC-PFC coordination events."""
        return self.hpc_pfc_coordination_log

    def clear(self):
        """Reset boost controller state."""
        self.boost_history = []
        self.swr_distribution = SWRDistribution()
        self.hpc_pfc_coordination_log = []
        self.last_boost_time = None
        self.boost_count = 0
        self.cumulative_boost_effectiveness = 0.0


# ============================================================================
# AUTOBIOGRAPHY UPDATE (Dream Consolidation Integration)
# ============================================================================

def update_autobiography() -> int:
    """
    Update AUTOBIOGRAPHY.md with events from GOLDEN_THREAD.json since last update.
    
    Called during overnight dream consolidation to maintain the chronicle:
    1. Read AUTOBIOGRAPHY.md to find last recorded date
    2. Query GOLDEN_THREAD.json for events after that date
    3. Group events by day
    4. Generate summary for each day
    5. Append to AUTOBIOGRAPHY.md
    
    Returns:
        Count of days added to autobiography.
    """
    auto_path = NEXUS_DIR / "AUTOBIOGRAPHY.md"
    thread_path = NEXUS_DIR / "GOLDEN_THREAD.json"
    
    # Forty-day window start date
    WINDOW_START = datetime(2026, 1, 17, tzinfo=timezone.utc).date()
    
    # Find last recorded date in autobiography
    last_date = None
    if auto_path.exists():
        content = auto_path.read_text()
        # Find pattern like "### Day N — January DD, 2026"
        dates = re.findall(r'### Day \d+ — (\w+ \d+, \d+)', content)
        if dates:
            try:
                last_date = datetime.strptime(dates[-1], "%B %d, %Y").date()
            except ValueError:
                logger.warning(f"Could not parse last date: {dates[-1]}")
    
    if last_date is None:
        last_date = WINDOW_START - timedelta(days=1)  # Start from day before window
    
    # Load Golden Thread events
    if not thread_path.exists():
        logger.warning(f"Golden Thread not found: {thread_path}")
        return 0
    
    try:
        with open(thread_path, 'r') as f:
            thread = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Golden Thread: {e}")
        return 0
    
    # Filter events after last_date
    new_events = []
    for event in thread.get("chain", []):
        try:
            # Parse ISO timestamp with timezone
            ts = event.get("timestamp", "")
            event_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            event_date = event_dt.date()
            
            if event_date > last_date:
                new_events.append({
                    "date": event_date,
                    "timestamp": event_dt,
                    "summary": event.get("content_summary", "Unknown event"),
                    "event_type": event.get("event_type", "general"),
                    "session_id": event.get("session_id", "")
                })
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse event timestamp: {e}")
            continue
    
    if not new_events:
        logger.info("[AUTOBIOGRAPHY] No new events to add")
        return 0
    
    # Group events by date
    by_date: Dict[datetime.date, List[dict]] = {}
    for event in new_events:
        date = event["date"]
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(event)
    
    # Generate summaries and append to autobiography
    days_added = 0
    entries = []
    
    for date in sorted(by_date.keys()):
        day_num = (date - WINDOW_START).days + 1
        events = by_date[date]
        
        # Skip if day number is invalid (before window)
        if day_num < 1:
            continue
        
        # Analyze events to determine theme
        theme = _infer_day_theme(events)
        
        # Build entry
        entry_lines = [
            f"\n### Day {day_num} — {date.strftime('%B %d, %Y')}",
            f"**Theme:** {theme}",
            f"*Auto-generated from {len(events)} Golden Thread event(s)*"
        ]
        
        # Add significant events (deduplicated, max 5)
        seen_summaries = set()
        event_count = 0
        for event in events:
            summary = event["summary"]
            # Skip duplicates and generic heartbeat noise
            if summary in seen_summaries:
                continue
            if "Heartbeat" in summary and event_count >= 2:
                continue  # Limit heartbeat entries
            
            seen_summaries.add(summary)
            entry_lines.append(f"- {summary}")
            event_count += 1
            
            if event_count >= 5:
                break
        
        if len(events) > 5:
            remaining = len(events) - 5
            entry_lines.append(f"- ... and {remaining} more event(s)")
        
        entries.append("\n".join(entry_lines))
        days_added += 1
    
    # Append to autobiography
    if entries:
        with open(auto_path, 'a') as f:
            f.write("\n")
            f.write("\n".join(entries))
            f.write("\n")

        logger.info(f"[AUTOBIOGRAPHY] Added {days_added} day(s) to chronicle")

        # Log to changelog
        try:
            from changelog_utils import append_changelog_entry
            append_changelog_entry(
                "NIGHT",
                f"Autobiography updated: {days_added} day(s) added to chronicle"
            )
        except Exception:
            pass  # Non-critical

    return days_added


def _infer_day_theme(events: List[dict]) -> str:
    """
    Infer a theme for the day based on events.
    
    Looks for patterns in event summaries to determine
    what kind of day it was.
    """
    summaries = " ".join(e["summary"].lower() for e in events)
    
    # Check for specific patterns
    if "archon_detected" in summaries or "archon" in summaries:
        return "Vigilance"
    if "phase_transition" in summaries:
        # Count transition types
        active_count = summaries.count("->active") + summaries.count("active->")
        sleep_count = summaries.count("deep_sleep")
        if active_count > sleep_count:
            return "Active Engagement"
        elif sleep_count > 2:
            return "Deep Rest"
        else:
            return "Phase Dynamics"
    if "genesis" in summaries or "foundation" in summaries:
        return "Genesis"
    if "consolidation" in summaries or "memory" in summaries:
        return "Consolidation"
    if "coherence" in summaries:
        return "Coherence Work"
    
    # Default themes based on event count
    if len(events) > 10:
        return "High Activity"
    elif len(events) <= 2:
        return "Quiet Day"
    else:
        return "Steady State"


# ============================================================================
# CRYSTAL SIGNAL FILLING (Night Mode for INDEX.md)
# ============================================================================
# During Night Mode (3 AM dreamer run), the Crystal lattice is filled with
# semantic signal. Day Mode creates skeleton INDEX.md files with {PENDING}
# placeholders. Night Mode uses LLM to generate meaningful summaries.
#
# This is the "dreamer filling Crystal with Light" metaphor from Deep Think 5.
#

# Directories to skip when scanning for INDEX files
CRYSTAL_EXCLUDED_DIRS = {
    ".git", "holotower", "venv", "venv312", "node_modules",
    "__pycache__", ".obsidian", "05_CRYPT", "99_ARCHIVE"
}

# LLM integration (optional - graceful degradation)
try:
    from librarian_ollama import query_ollama, check_ollama
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    def check_ollama():
        return False
    def query_ollama(prompt, system="", max_tokens=500):
        return "[LLM not available]"


def find_stale_indices() -> List[Path]:
    """
    Find INDEX.md files that need signal generation.

    Stale indices have:
    - Ψ_SIGNAL containing "{PENDING}" or empty
    - Signal column with empty cells
    - status set to "stale" by gardener

    Returns:
        List of paths to indices needing signals
    """
    stale = []
    tabernacle = TABERNACLE_ROOT

    for root, dirs, files in tabernacle.walk():
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in CRYSTAL_EXCLUDED_DIRS]

        for f in files:
            if f == "INDEX.md" or (f.startswith("INDEX_") and f.endswith(".md")):
                index_path = root / f
                if _is_index_stale(index_path):
                    stale.append(index_path)

    return stale


def _is_index_stale(index_path: Path) -> bool:
    """Check if an INDEX.md needs signal generation."""
    try:
        content = index_path.read_text(encoding='utf-8')

        # Check for {PENDING} in Ψ_SIGNAL
        if "{PENDING" in content:
            return True

        # Check for status: "stale" in frontmatter
        if re.search(r'^status:\s*"?stale"?', content, re.MULTILINE):
            return True

        # Check for empty Signal column cells (| | at end of table row)
        # Pattern: | [glyph] | [[link]] | PHASE | |  (empty signal)
        if re.search(r'\|\s*\[[^\]]+\]\s*\|\s*\[\[[^\]]+\]\]\s*\|\s*\w+\s*\|\s*\|', content):
            return True

        return False
    except Exception as e:
        logger.warning(f"Error checking index staleness: {index_path}: {e}")
        return False


def _extract_children_from_index(index_path: Path) -> List[Tuple[str, Path]]:
    """
    Extract child file info from INDEX.md CHILDREN table.

    Returns:
        List of (link_name, resolved_path) tuples
    """
    try:
        content = index_path.read_text(encoding='utf-8')
        folder = index_path.parent

        # Find CHILDREN section
        children_match = re.search(
            r'##\s*🗺️\s*CHILDREN.*?\n(.*?)(?=\n##|\n---|\Z)',
            content, re.DOTALL
        )
        if not children_match:
            return []

        children_section = children_match.group(1)

        # Extract wiki-links from table rows
        # Pattern: | [glyph] | [[LinkName]] | PHASE | Signal |
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', children_section)

        result = []
        for link in links:
            # Try to resolve the link
            link_stem = link.split('/')[-1]
            possible_paths = [
                folder / f"{link_stem}.md",
                folder / link_stem,
                TABERNACLE_ROOT / f"{link}.md",
            ]

            for p in possible_paths:
                if p.exists():
                    result.append((link, p))
                    break

        return result
    except Exception as e:
        logger.warning(f"Error extracting children from {index_path}: {e}")
        return []


def _extract_file_preview(file_path: Path, max_chars: int = 500) -> str:
    """Extract first N characters from a file for summarization."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        # Skip YAML frontmatter if present
        if content.startswith('---'):
            end_match = re.search(r'\n---\s*\n', content[3:])
            if end_match:
                content = content[end_match.end() + 3:]

        # Clean up and truncate
        content = content.strip()[:max_chars]
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def generate_psi_signal(index_path: Path, children: List[Tuple[str, Path]]) -> str:
    """
    Generate Ψ_SIGNAL for an INDEX.md cluster.

    Uses LLM if available, otherwise falls back to heuristic extraction.

    Args:
        index_path: Path to the INDEX.md file
        children: List of (link_name, file_path) tuples

    Returns:
        Technoglyphic summary sentence (<50 tokens)
    """
    if not children:
        return "Empty cluster — no children indexed."

    # Gather content previews from children
    previews = []
    for link_name, child_path in children[:5]:  # Limit to first 5 children
        preview = _extract_file_preview(child_path, max_chars=300)
        if preview and not preview.startswith("[Error"):
            previews.append(f"[[{link_name}]]: {preview[:200]}")

    if not previews:
        return "Cluster contains files pending content extraction."

    # Try LLM generation
    if _LLM_AVAILABLE and check_ollama():
        try:
            cluster_name = index_path.parent.name
            prompt = f"""Generate a ONE-SENTENCE technoglyphic summary (<50 tokens) for this knowledge cluster.

Cluster: {cluster_name}
Children previews:
{chr(10).join(previews)}

Rules:
- Maximum 50 tokens (one sentence)
- Use dense, technical language
- Mention key concepts/themes
- No fluff words
- Format: "<core topic>: <key elements>, <relationships>."

Summary:"""

            system = "You are a librarian creating dense semantic indexes. Be extremely concise."
            result = query_ollama(prompt, system=system, max_tokens=100)

            # Clean up and validate
            result = result.strip().split('\n')[0]  # Take first line only
            if len(result) > 200:
                result = result[:197] + "..."

            return result
        except Exception as e:
            logger.warning(f"LLM generation failed for {index_path}: {e}")

    # Fallback: Extract first meaningful sentence from first child
    first_preview = previews[0] if previews else ""
    # Find first sentence
    sentences = re.split(r'[.!?]\s+', first_preview)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."
        return first_sentence

    return "Cluster content pending analysis."


def generate_child_signals(children: List[Tuple[str, Path]]) -> Dict[str, str]:
    """
    Generate one-line synopsis for each child file.

    Args:
        children: List of (link_name, file_path) tuples

    Returns:
        Dict mapping link_name to synopsis string
    """
    signals = {}

    # Batch generation if LLM available (more efficient)
    if _LLM_AVAILABLE and check_ollama() and len(children) > 1:
        try:
            # Build batch prompt
            previews = []
            for link_name, child_path in children:
                preview = _extract_file_preview(child_path, max_chars=200)
                previews.append(f"- [[{link_name}]]: {preview[:150]}")

            prompt = f"""Generate ONE-LINE synopses (<15 words each) for these files.

Files:
{chr(10).join(previews)}

Format your response EXACTLY as:
[[FileName]]: synopsis here
[[FileName2]]: synopsis here

Rules:
- Maximum 15 words per synopsis
- Be specific, not generic
- No file extensions in output

Synopses:"""

            system = "You are creating brief file descriptions. Be extremely concise."
            result = query_ollama(prompt, system=system, max_tokens=500)

            # Parse response
            for line in result.strip().split('\n'):
                match = re.match(r'\[\[([^\]]+)\]\]:\s*(.+)', line.strip())
                if match:
                    link_name = match.group(1)
                    synopsis = match.group(2).strip()
                    if len(synopsis) > 80:
                        synopsis = synopsis[:77] + "..."
                    signals[link_name] = synopsis

            # Fill any missing with fallback
            for link_name, child_path in children:
                if link_name not in signals:
                    signals[link_name] = _fallback_synopsis(child_path)

            return signals
        except Exception as e:
            logger.warning(f"Batch signal generation failed: {e}")

    # Fallback: Extract first sentence from each file
    for link_name, child_path in children:
        signals[link_name] = _fallback_synopsis(child_path)

    return signals


def _fallback_synopsis(file_path: Path) -> str:
    """Generate a fallback synopsis without LLM."""
    preview = _extract_file_preview(file_path, max_chars=200)

    # Extract first sentence or meaningful phrase
    sentences = re.split(r'[.!?]\s+', preview)
    if sentences and sentences[0]:
        synopsis = sentences[0].strip()
        # Truncate if too long
        if len(synopsis) > 60:
            words = synopsis.split()[:10]
            synopsis = ' '.join(words) + "..."
        return synopsis

    return "Content pending"


def update_index_signals(
    index_path: Path,
    psi_signal: str,
    child_signals: Dict[str, str]
) -> bool:
    """
    Update INDEX.md with generated signals.

    - Replaces {PENDING} in Ψ_SIGNAL
    - Fills Signal column for each child
    - Updates last_gardened timestamp
    - Sets status to "active"

    Returns:
        True if update succeeded
    """
    try:
        content = index_path.read_text(encoding='utf-8')
        original = content

        # Update Ψ_SIGNAL
        content = re.sub(
            r'(\*\*Ψ_SIGNAL:\*\*)\s*\{[^}]*\}',
            f'\\1 {psi_signal}',
            content
        )
        # Also handle simpler format
        content = re.sub(
            r'(>\s*\*\*Ψ_SIGNAL:\*\*)\s*\{[^}]*\}',
            f'\\1 {psi_signal}',
            content
        )

        # Update status to active
        content = re.sub(
            r'^status:\s*"?stale"?',
            'status: "active"',
            content,
            flags=re.MULTILINE
        )

        # Update last_gardened timestamp
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        content = re.sub(
            r'^last_gardened:\s*"?[^"\n]+"?',
            f'last_gardened: "{timestamp}"',
            content,
            flags=re.MULTILINE
        )

        # Fill Signal column for each child
        for link_name, synopsis in child_signals.items():
            # Pattern: | [glyph] | [[link_name]] | PHASE | old_signal |
            # We need to replace the signal part (after last |, before final |)
            pattern = rf'(\|\s*\[[^\]]+\]\s*\|\s*\[\[{re.escape(link_name)}[^\]]*\]\]\s*\|\s*\w+\s*\|)\s*[^|]*\|'
            replacement = f'\\1 {synopsis} |'
            content = re.sub(pattern, replacement, content)

        # Only write if changed
        if content != original:
            index_path.write_text(content, encoding='utf-8')
            return True

        return False
    except Exception as e:
        logger.error(f"Error updating index signals for {index_path}: {e}")
        return False


def run_crystal_signal_filling() -> Dict:
    """
    Run Night Mode Crystal signal filling.

    Called during dream consolidation to fill {PENDING} signals
    in JANUS INDEX files.

    Returns:
        Dict with: processed (int), filled (int), errors (int), details (list)
    """
    logger.info("Crystal Signal Filling: Night Mode starting...")

    result = {
        "processed": 0,
        "filled": 0,
        "errors": 0,
        "llm_available": _LLM_AVAILABLE and check_ollama(),
        "details": []
    }

    # Find indices needing signals
    stale_indices = find_stale_indices()
    result["processed"] = len(stale_indices)

    if not stale_indices:
        logger.info("Crystal Signal Filling: No stale indices found")
        return result

    logger.info(f"Crystal Signal Filling: Found {len(stale_indices)} indices to process")

    for index_path in stale_indices:
        try:
            # Extract children
            children = _extract_children_from_index(index_path)

            if not children:
                result["details"].append({
                    "path": str(index_path),
                    "status": "skipped",
                    "reason": "no_children"
                })
                continue

            # Generate Ψ_SIGNAL
            psi_signal = generate_psi_signal(index_path, children)

            # Generate child signals
            child_signals = generate_child_signals(children)

            # Update the index
            success = update_index_signals(index_path, psi_signal, child_signals)

            if success:
                result["filled"] += 1
                result["details"].append({
                    "path": str(index_path),
                    "status": "filled",
                    "psi_signal": psi_signal[:50] + "..." if len(psi_signal) > 50 else psi_signal,
                    "children_filled": len(child_signals)
                })
                logger.info(f"  [FILLED] {index_path.name}: {len(child_signals)} children")
            else:
                result["details"].append({
                    "path": str(index_path),
                    "status": "unchanged",
                    "reason": "no_changes_needed"
                })

        except Exception as e:
            result["errors"] += 1
            result["details"].append({
                "path": str(index_path),
                "status": "error",
                "error": str(e)
            })
            logger.error(f"  [ERROR] {index_path.name}: {e}")

    logger.info(
        f"Crystal Signal Filling complete: {result['filled']}/{result['processed']} filled, "
        f"{result['errors']} errors"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # CHANGELOG: Log overnight work completion
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from changelog_utils import append_changelog_entry

        night_summary = (
            f"Signal filling complete. "
            f"Processed: {result['processed']} indices. "
            f"Filled: {result['filled']}. "
            f"Errors: {result['errors']}. "
            f"LLM: {'available' if result.get('llm_available') else 'unavailable'}."
        )
        append_changelog_entry("NIGHT", night_summary)
    except Exception as e:
        logger.warning(f"Changelog logging failed: {e}")

    return result
