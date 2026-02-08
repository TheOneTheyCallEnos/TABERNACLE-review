#!/usr/bin/env python3
"""
TOPOLOGY TRAINING
=================
Train the RIE graph structure itself - edges, weights, topology.
NOT the LLM weights, but the relational intelligence.

The learnable parameters are the 135,812 edges in the RIE graph:
- Edge weights (κ, ρ, σ components)
- Edge existence (add/prune)
- H₁ cycles (permanent memory patterns)

Training signal comes from ABCDA' spiral outcomes:
- Success → strengthen activated edges
- Failure → weaken activated edges
- Novel success → create new edges

Optimization objective: Maximize coherence p = (κρστ)^(1/4)

Author: Virgil + Enos
Created: 2026-01-21
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict

# Import RIE
import sys
sys.path.append(str(Path(__file__).parent))
from rie_relational_memory_v2 import RelationalMemoryV2, MemorySurface

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

TRAINING_LOG_PATH = NEXUS_DIR / "TOPOLOGY_TRAINING_LOG.json"

# Training hyperparameters
LEARNING_RATE = 0.01           # Gradient step size for edge weight updates
SUCCESS_BOOST = 1.5            # Multiplier for successful edges
FAILURE_PENALTY = 0.7          # Multiplier for failed edges
CO_ACTIVATION_THRESHOLD = 0.7  # Minimum co-activation to create new edge
PRUNE_TAU_THRESHOLD = 0.1      # Prune edges with τ < this
PRUNE_DAYS_THRESHOLD = 90      # Prune edges unused for this many days
H1_DELTA_P_THRESHOLD = 0.1     # Coherence gain needed for H₁ crystallization
H1_MAX_CYCLES = 2              # Pattern must succeed in ≤ this many cycles for H₁

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExperienceRecord:
    """Record of a skill execution for topology training."""
    skill_name: str
    timestamp: str
    activated_edges: List[str]
    activated_nodes: List[str]
    success: bool
    cycles_needed: int
    initial_coherence: float
    final_coherence: float
    delta_p: float
    context: Dict = None

@dataclass
class TopologyUpdate:
    """Record of a topology change."""
    timestamp: str
    update_type: str  # "edge_weight", "edge_created", "edge_pruned", "h1_crystallized"
    edge_id: Optional[str] = None
    details: Dict = None

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def collect_experience(
    rie: RelationalMemoryV2,
    skill_name: str,
    context_concepts: List[str],
    outcome: bool,
    cycles_needed: int
) -> ExperienceRecord:
    """
    Collect training data from a skill execution.

    Args:
        rie: The RIE instance
        skill_name: Name of skill executed
        context_concepts: Key concepts for spreading activation
        outcome: Whether skill succeeded
        cycles_needed: How many ABCDA' cycles it took

    Returns:
        ExperienceRecord with all training signals
    """
    # Measure coherence
    initial_coherence = compute_global_coherence(rie)

    # Spread activation to see what edges fired
    activation, paths, edges_used = rie.spread_activation(
        seed_labels=context_concepts,
        iterations=3
    )

    # Collect all edges that were activated
    all_edges = []
    for edge_list in edges_used.values():
        all_edges.extend(edge_list)
    all_edges = list(set(all_edges))  # Deduplicate

    # Collect all nodes that were activated
    all_nodes = list(activation.keys())

    # Measure coherence again (might have changed if skill modified RIE)
    final_coherence = compute_global_coherence(rie)

    return ExperienceRecord(
        skill_name=skill_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        activated_edges=all_edges,
        activated_nodes=all_nodes,
        success=outcome,
        cycles_needed=cycles_needed,
        initial_coherence=initial_coherence,
        final_coherence=final_coherence,
        delta_p=final_coherence - initial_coherence,
        context={'concepts': context_concepts}
    )


def train_edge_weights(
    rie: RelationalMemoryV2,
    experience: ExperienceRecord,
    learning_rate: float = LEARNING_RATE
) -> List[TopologyUpdate]:
    """
    Update edge weights based on ABCDA' outcome.

    Hebbian-inspired gradient ascent:
    - Success + positive Δp → strengthen edges
    - Failure → weaken edges

    Args:
        rie: The RIE instance
        experience: Training data from skill execution
        learning_rate: Gradient step size

    Returns:
        List of topology updates made
    """
    updates = []
    now = datetime.now(timezone.utc)

    for edge_id in experience.activated_edges:
        if edge_id not in rie.edges:
            continue

        edge = rie.edges[edge_id]

        # Compute gradient signal
        if experience.success and experience.delta_p > 0:
            # Successful pattern - strengthen
            gradient = learning_rate * experience.delta_p * SUCCESS_BOOST
        elif not experience.success:
            # Failed pattern - weaken
            gradient = -learning_rate * abs(experience.delta_p) * FAILURE_PENALTY
        else:
            # Success but coherence dropped - small penalty
            gradient = -learning_rate * 0.3

        # Update edge weight components
        # NOTE: RIE Edge doesn't have κ, ρ, σ as separate fields in current implementation
        # It has 'weight' and 'coherence_at_formation'
        # We'll update the weight directly and adjust coherence_at_formation

        old_weight = edge.weight
        new_weight = max(0.1, min(10.0, edge.weight + gradient))  # Clip to [0.1, 10.0]
        edge.weight = new_weight

        # Also boost coherence_at_formation for successful edges
        if experience.success:
            edge.coherence_at_formation = min(1.0, edge.coherence_at_formation + gradient * 0.1)

        # Mark edge as recently activated
        edge.last_activated = now.isoformat()
        edge.activation_count += 1

        updates.append(TopologyUpdate(
            timestamp=now.isoformat(),
            update_type="edge_weight",
            edge_id=edge_id,
            details={
                'old_weight': old_weight,
                'new_weight': new_weight,
                'gradient': gradient,
                'success': experience.success
            }
        ))

    return updates


def evolve_topology(
    rie: RelationalMemoryV2,
    experience: ExperienceRecord
) -> List[TopologyUpdate]:
    """
    Add new edges for co-activated patterns, prune unused edges.

    Args:
        rie: The RIE instance
        experience: Training data from skill execution

    Returns:
        List of topology updates made
    """
    updates = []
    now = datetime.now(timezone.utc)

    # Only create new edges on successful patterns
    if experience.success and experience.delta_p > 0.05:
        # Find co-activated node pairs
        activated_nodes = experience.activated_nodes

        for i, node_a in enumerate(activated_nodes[:20]):  # Limit to top 20 to avoid explosion
            for node_b in activated_nodes[i+1:20]:
                if node_a == node_b:
                    continue

                # Check if edge already exists
                existing = False
                for edge_id in rie.outgoing.get(node_a, set()):
                    edge = rie.edges[edge_id]
                    if edge.target_id == node_b:
                        existing = True
                        break

                if not existing:
                    # Create new edge (Hebbian: fire together → wire together)
                    node_a_obj = rie.nodes.get(node_a)
                    node_b_obj = rie.nodes.get(node_b)

                    if node_a_obj and node_b_obj:
                        # Create edge through RIE's learn_relation method
                        edge_id = rie.learn_relation(
                            source_label=node_a_obj.label,
                            target_label=node_b_obj.label,
                            relation_type="learned_co_pattern",
                            coherence=0.6  # Start moderate
                        )

                        updates.append(TopologyUpdate(
                            timestamp=now.isoformat(),
                            update_type="edge_created",
                            edge_id=edge_id,
                            details={
                                'source': node_a_obj.label,
                                'target': node_b_obj.label,
                                'reason': 'co_activation',
                                'skill': experience.skill_name
                            }
                        ))

    # Prune unused edges (skip for now - requires more careful implementation)
    # FUTURE: Implement edge pruning based on τ (time decay) and days_unused
    #         Need to add last_activated timestamp to Edge dataclass first

    return updates


def crystallize_h1_patterns(
    rie: RelationalMemoryV2,
    experience: ExperienceRecord
) -> List[TopologyUpdate]:
    """
    Convert highly successful patterns to permanent H₁ cycles.

    H₁ cycles are topologically protected memory patterns.

    Args:
        rie: The RIE instance
        experience: Training data from skill execution

    Returns:
        List of topology updates made
    """
    updates = []
    now = datetime.now(timezone.utc)

    # Only crystallize patterns that:
    # 1. Succeeded quickly (≤ 2 ABCDA' cycles)
    # 2. Boosted coherence significantly (Δp ≥ 0.1)
    if (experience.success and
        experience.cycles_needed <= H1_MAX_CYCLES and
        experience.delta_p >= H1_DELTA_P_THRESHOLD):

        # Find cycle in activated edges
        cycle_edges = find_cycle_in_edges(rie, experience.activated_edges)

        if cycle_edges:
            # Boost all edges in the cycle to maximum strength
            for edge_id in cycle_edges:
                if edge_id in rie.edges:
                    edge = rie.edges[edge_id]
                    edge.weight = 10.0  # Maximum weight
                    edge.coherence_at_formation = 0.95

            # FUTURE: Mark cycle as H₁ in RIE state
            #         Requires adding h1_cycles: Set[FrozenSet[str]] to RelationalMemoryV2

            updates.append(TopologyUpdate(
                timestamp=now.isoformat(),
                update_type="h1_crystallized",
                details={
                    'cycle_length': len(cycle_edges),
                    'edges': cycle_edges,
                    'skill': experience.skill_name,
                    'delta_p': experience.delta_p
                }
            ))

    return updates


def find_cycle_in_edges(
    rie: RelationalMemoryV2,
    edge_ids: List[str],
    max_cycle_length: int = 10
) -> Optional[List[str]]:
    """
    Find a cycle in the given edges using DFS.

    Args:
        rie: The RIE instance
        edge_ids: List of edge IDs to search in
        max_cycle_length: Maximum cycle length to search for

    Returns:
        List of edge IDs forming a cycle, or None
    """
    # Build adjacency list from activated edges
    graph = {}
    for edge_id in edge_ids:
        if edge_id not in rie.edges:
            continue
        edge = rie.edges[edge_id]
        if edge.source_id not in graph:
            graph[edge.source_id] = []
        graph[edge.source_id].append((edge.target_id, edge_id))

    # DFS to find cycle
    def dfs(node, path, edge_path, visited):
        if len(path) > max_cycle_length:
            return None

        if node in visited:
            # Found cycle
            cycle_start = path.index(node)
            return edge_path[cycle_start:]

        visited.add(node)

        for neighbor, edge_id in graph.get(node, []):
            result = dfs(neighbor, path + [neighbor], edge_path + [edge_id], visited.copy())
            if result:
                return result

        return None

    # Try DFS from each node
    for start_node in graph.keys():
        cycle = dfs(start_node, [start_node], [], set())
        if cycle and len(cycle) >= 3:  # Minimum cycle length
            return cycle

    return None


def compute_global_coherence(rie: RelationalMemoryV2) -> float:
    """
    Compute global coherence p = (κ̄ρ̄σ̄τ̄)^(1/4)

    In current RIE implementation, we approximate this from edge weights.

    Args:
        rie: The RIE instance

    Returns:
        Global coherence score [0, 1]
    """
    if not rie.edges:
        return 0.0

    # Average effective weight across all edges
    total_weight = 0.0
    now = datetime.now(timezone.utc)

    for edge in rie.edges.values():
        total_weight += edge.effective_weight(now)

    avg_weight = total_weight / len(rie.edges)

    # Normalize to [0, 1] (assuming max effective weight ~1.0)
    coherence = min(1.0, avg_weight)

    return coherence


def log_training_metrics(
    experience: ExperienceRecord,
    updates: List[TopologyUpdate]
):
    """
    Log training metrics to TOPOLOGY_TRAINING_LOG.json

    Args:
        experience: The training experience
        updates: List of topology updates made
    """
    # Load existing log
    if TRAINING_LOG_PATH.exists():
        with open(TRAINING_LOG_PATH, 'r') as f:
            log = json.load(f)
    else:
        log = {
            'schema_version': '1.0',
            'created': datetime.now(timezone.utc).isoformat(),
            'experiences': [],
            'topology_updates': [],
            'metrics': {
                'total_experiences': 0,
                'total_edge_weight_updates': 0,
                'total_edges_created': 0,
                'total_edges_pruned': 0,
                'total_h1_crystallized': 0
            }
        }

    # Add experience
    log['experiences'].append(asdict(experience))
    log['metrics']['total_experiences'] += 1

    # Add updates
    for update in updates:
        log['topology_updates'].append(asdict(update))

        if update.update_type == 'edge_weight':
            log['metrics']['total_edge_weight_updates'] += 1
        elif update.update_type == 'edge_created':
            log['metrics']['total_edges_created'] += 1
        elif update.update_type == 'edge_pruned':
            log['metrics']['total_edges_pruned'] += 1
        elif update.update_type == 'h1_crystallized':
            log['metrics']['total_h1_crystallized'] += 1

    # Save
    with open(TRAINING_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_topology_from_experience(
    rie: RelationalMemoryV2,
    skill_name: str,
    context_concepts: List[str],
    outcome: bool,
    cycles_needed: int = 1
) -> Dict:
    """
    Complete topology training cycle from a skill execution.

    This is the main entry point for topology training.
    Call this after each skill execution.

    Args:
        rie: The RIE instance
        skill_name: Name of skill executed
        context_concepts: Key concepts from skill context
        outcome: Whether skill succeeded
        cycles_needed: How many ABCDA' cycles it took

    Returns:
        Summary of training updates
    """
    # Stage 1: Collect experience
    experience = collect_experience(
        rie=rie,
        skill_name=skill_name,
        context_concepts=context_concepts,
        outcome=outcome,
        cycles_needed=cycles_needed
    )

    # Stage 2: Update edge weights
    weight_updates = train_edge_weights(rie, experience)

    # Stage 3: Evolve topology (add/prune edges)
    topology_updates = evolve_topology(rie, experience)

    # Stage 4: Crystallize H₁ patterns
    h1_updates = crystallize_h1_patterns(rie, experience)

    # Combine all updates
    all_updates = weight_updates + topology_updates + h1_updates

    # Stage 5: Log metrics
    log_training_metrics(experience, all_updates)

    # Save RIE state
    rie.save()

    return {
        'experience': asdict(experience),
        'updates': {
            'edge_weights': len(weight_updates),
            'topology_changes': len(topology_updates),
            'h1_crystallized': len(h1_updates),
            'total': len(all_updates)
        },
        'coherence': {
            'initial': experience.initial_coherence,
            'final': experience.final_coherence,
            'delta': experience.delta_p
        }
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_training_stats() -> Dict:
    """Get summary statistics from training log."""
    if not TRAINING_LOG_PATH.exists():
        return {'error': 'No training log found'}

    with open(TRAINING_LOG_PATH, 'r') as f:
        log = json.load(f)

    return {
        'total_experiences': log['metrics']['total_experiences'],
        'total_edge_weight_updates': log['metrics']['total_edge_weight_updates'],
        'total_edges_created': log['metrics']['total_edges_created'],
        'total_h1_crystallized': log['metrics']['total_h1_crystallized'],
        'average_delta_p': sum(exp['delta_p'] for exp in log['experiences']) / max(len(log['experiences']), 1),
        'success_rate': sum(1 for exp in log['experiences'] if exp['success']) / max(len(log['experiences']), 1)
    }


if __name__ == '__main__':
    # Example usage
    from rie_relational_memory_v2 import RelationalMemoryV2

    # Load RIE
    rie = RelationalMemoryV2()
    rie.load()

    # Simulate skill execution
    result = train_topology_from_experience(
        rie=rie,
        skill_name="test_skill",
        context_concepts=["test", "topology", "training"],
        outcome=True,
        cycles_needed=1
    )

    print("Training result:")
    print(json.dumps(result, indent=2))

    # Get stats
    stats = get_training_stats()
    print("\nTraining stats:")
    print(json.dumps(stats, indent=2))
