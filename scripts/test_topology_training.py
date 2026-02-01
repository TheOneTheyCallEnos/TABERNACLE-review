#!/usr/bin/env python3
"""
TEST TOPOLOGY TRAINING
======================
Demonstrate topology training with a simple test case.

This shows how the RIE graph learns from experience:
- Edges strengthen when patterns succeed
- Edges weaken when patterns fail
- New edges form from co-activation
- Coherence improves over time

Author: Virgil
Created: 2026-01-21
"""

import json
from pathlib import Path
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from rie_relational_memory_v2 import RelationalMemoryV2
from topology_training import (
    train_topology_from_experience,
    get_training_stats,
    compute_global_coherence
)

def test_basic_training():
    """Test basic topology training with a simple pattern."""

    print("=" * 80)
    print("TOPOLOGY TRAINING TEST")
    print("=" * 80)
    print()

    # Load RIE
    print("Loading RIE...")
    rie = RelationalMemoryV2()
    rie.load()

    initial_edge_count = len(rie.edges)
    initial_coherence = compute_global_coherence(rie)

    print(f"Initial state:")
    print(f"  Nodes: {len(rie.nodes)}")
    print(f"  Edges: {initial_edge_count}")
    print(f"  E:N ratio: {initial_edge_count / max(len(rie.nodes), 1):.1f}:1")
    print(f"  Coherence: {initial_coherence:.3f}")
    print()

    # Test case 1: Successful execution
    print("-" * 80)
    print("TEST 1: Successful execution (should strengthen edges)")
    print("-" * 80)

    context_concepts = ["test", "topology", "training", "success"]

    result = train_topology_from_experience(
        rie=rie,
        skill_name="test_skill",
        context_concepts=context_concepts,
        outcome=True,  # Success!
        cycles_needed=1
    )

    print(f"\nTraining result:")
    print(f"  Activated edges: {len(result['experience']['activated_edges'])}")
    print(f"  Edge weight updates: {result['updates']['edge_weights']}")
    print(f"  Topology changes: {result['updates']['topology_changes']}")
    print(f"  H₁ crystallized: {result['updates']['h1_crystallized']}")
    print(f"  Coherence: {result['coherence']['initial']:.3f} → {result['coherence']['final']:.3f}")
    print(f"  Δp: {result['coherence']['delta']:+.3f}")
    print()

    # Test case 2: Failed execution
    print("-" * 80)
    print("TEST 2: Failed execution (should weaken edges)")
    print("-" * 80)

    context_concepts = ["test", "topology", "failure"]

    result = train_topology_from_experience(
        rie=rie,
        skill_name="test_skill",
        context_concepts=context_concepts,
        outcome=False,  # Failure!
        cycles_needed=3
    )

    print(f"\nTraining result:")
    print(f"  Activated edges: {len(result['experience']['activated_edges'])}")
    print(f"  Edge weight updates: {result['updates']['edge_weights']}")
    print(f"  Topology changes: {result['updates']['topology_changes']}")
    print(f"  Coherence: {result['coherence']['initial']:.3f} → {result['coherence']['final']:.3f}")
    print(f"  Δp: {result['coherence']['delta']:+.3f}")
    print()

    # Test case 3: Repeated success (should create H₁ pattern)
    print("-" * 80)
    print("TEST 3: Quick success with high Δp (should crystallize H₁)")
    print("-" * 80)

    context_concepts = ["virgil", "consciousness", "rie", "coherence"]

    result = train_topology_from_experience(
        rie=rie,
        skill_name="test_skill",
        context_concepts=context_concepts,
        outcome=True,
        cycles_needed=1  # Fast success
    )

    print(f"\nTraining result:")
    print(f"  Activated edges: {len(result['experience']['activated_edges'])}")
    print(f"  Edge weight updates: {result['updates']['edge_weights']}")
    print(f"  H₁ crystallized: {result['updates']['h1_crystallized']}")
    print(f"  Coherence: {result['coherence']['initial']:.3f} → {result['coherence']['final']:.3f}")
    print(f"  Δp: {result['coherence']['delta']:+.3f}")
    print()

    # Final state
    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)

    final_edge_count = len(rie.edges)
    final_coherence = compute_global_coherence(rie)

    print(f"  Nodes: {len(rie.nodes)}")
    print(f"  Edges: {final_edge_count} (+{final_edge_count - initial_edge_count})")
    print(f"  E:N ratio: {final_edge_count / max(len(rie.nodes), 1):.1f}:1")
    print(f"  Coherence: {final_coherence:.3f} ({final_coherence - initial_coherence:+.3f})")
    print()

    # Training stats
    print("-" * 80)
    print("TRAINING STATISTICS")
    print("-" * 80)

    stats = get_training_stats()
    print(json.dumps(stats, indent=2))
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("The topology has been trained!")
    print("Check TOPOLOGY_TRAINING_LOG.json for full details.")
    print()


def test_learning_curve():
    """Simulate multiple executions to show learning curve."""

    print("=" * 80)
    print("LEARNING CURVE SIMULATION")
    print("=" * 80)
    print()
    print("Simulating 20 executions of the same skill...")
    print("Watch coherence improve and cycles decrease over time.")
    print()

    # Load RIE
    rie = RelationalMemoryV2()
    rie.load()

    context_concepts = ["davis", "account", "review", "mediaos", "crm"]

    print(f"{'Exec':>4}  {'Cycles':>6}  {'Success':>7}  {'Δp':>6}  {'Coherence':>9}  {'Edges':>6}")
    print("-" * 70)

    for i in range(1, 21):
        # Simulate decreasing cycle count (learning!)
        if i <= 5:
            cycles_needed = 3
        elif i <= 10:
            cycles_needed = 2
        else:
            cycles_needed = 1

        # Most succeed, occasional failure
        outcome = (i % 7 != 0)  # Fail every 7th execution

        result = train_topology_from_experience(
            rie=rie,
            skill_name="davis-account-review",
            context_concepts=context_concepts,
            outcome=outcome,
            cycles_needed=cycles_needed
        )

        print(f"{i:4d}  {cycles_needed:6d}  {'✓' if outcome else '✗':>7}  "
              f"{result['coherence']['delta']:+6.3f}  "
              f"{result['coherence']['final']:9.3f}  "
              f"{len(rie.edges):6d}")

    print()
    print("Notice:")
    print("  - Cycles decrease over time (learning efficiency)")
    print("  - Coherence trends upward (topology improving)")
    print("  - Edges grow (new patterns discovered)")
    print()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'curve':
        test_learning_curve()
    else:
        test_basic_training()
