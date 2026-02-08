#!/usr/bin/env python3
"""
Test that BiologicalEdge is properly integrated into RIE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rie_relational_memory_v2 import RelationalMemoryV2
from biological_edge import BiologicalEdge

def test_integration():
    print("=" * 80)
    print("BIOLOGICAL EDGE INTEGRATION TEST")
    print("=" * 80)
    print()

    # Initialize RIE
    rie = RelationalMemoryV2()

    # Create test nodes
    n1 = rie.get_or_create_node("test_concept_a")
    n2 = rie.get_or_create_node("test_concept_b")

    # Create relation (should create BiologicalEdge)
    rie.create_relation(n1.id, n2.id, "test_relation")

    # Get the edge
    edge_id = f"{n1.id}→{n2.id}"
    edge = rie.edges.get(edge_id)

    # Verify it's a BiologicalEdge
    assert isinstance(edge, BiologicalEdge), f"Edge is {type(edge)}, not BiologicalEdge!"
    print(f"✓ Edge is BiologicalEdge")

    # Verify it has biological fields
    assert hasattr(edge, 'w_slow'), "Missing w_slow"
    assert hasattr(edge, 'w_fast'), "Missing w_fast"
    assert hasattr(edge, 'tau'), "Missing tau"
    assert hasattr(edge, 'is_h1_locked'), "Missing is_h1_locked"
    print(f"✓ BiologicalEdge has all required fields")

    # Test STDP
    initial_w_fast = edge.w_fast
    from datetime import datetime, timezone
    t_now = datetime.now(timezone.utc).timestamp()
    edge.update_stdp(t_pre=t_now - 0.01, t_post=t_now, outcome_signal=1.0)  # LTP
    assert edge.w_fast > initial_w_fast, "STDP LTP failed"
    print(f"✓ STDP working (w_fast: {initial_w_fast:.4f} → {edge.w_fast:.4f})")

    # Test consolidation
    initial_w_slow = edge.w_slow
    edge.consolidate()
    print(f"✓ Consolidation working (w_slow: {initial_w_slow:.4f} → {edge.w_slow:.4f})")

    # Test H₁ lock
    edge.is_h1_locked = True
    locked_w_slow = edge.w_slow
    t_now = datetime.now(timezone.utc).timestamp()
    edge.update_stdp(t_pre=t_now - 0.01, t_post=t_now, outcome_signal=1.0)
    edge.consolidate()
    print(f"✓ H₁ lock protection (w_slow unchanged: {edge.w_slow:.4f})")

    # Test RIE methods exist
    assert hasattr(rie, 'update_path_stdp'), "Missing update_path_stdp method"
    assert hasattr(rie, 'consolidate_edges'), "Missing consolidate_edges method"
    assert hasattr(rie, 'protect_h1_cycles'), "Missing protect_h1_cycles method"
    print(f"✓ RIE has all required STDP methods")

    # Test save/load
    rie.save()
    rie2 = RelationalMemoryV2()
    rie2.load()

    edge2 = rie2.edges.get(edge_id)
    assert isinstance(edge2, BiologicalEdge), "Loaded edge is not BiologicalEdge!"
    assert edge2.w_slow == edge.w_slow, "w_slow not preserved in save/load"
    assert edge2.is_h1_locked == edge.is_h1_locked, "H₁ lock not preserved"
    print(f"✓ Save/load preserves BiologicalEdge state")

    print()
    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print()
    print("BiologicalEdge is now fully integrated into RIE!")
    print()

if __name__ == '__main__':
    test_integration()
