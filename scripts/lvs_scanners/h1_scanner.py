"""
LOGOS LVS MODULE: H1 SCANNER (Memory Topology)
==============================================
LVS Component: σ (Sigma) - Structure & Complexity
Theory Ref: 04_LR_LAW/CANON/LVS_MATHEMATICS.md

Purpose:
  Analyzes the RIE (Relational Intelligence Engine) graph for topological health.
  - H1 Cycles = Persistent Memories
  - Orphans = Disconnected Concepts (Noise)
  - Broken Cycles = Dead-end chains (Rigidity)

Algorithm:
  Uses Strongly Connected Components (SCCs) as a proxy for cycle detection
  to avoid NP-hard complexity on 1.1M edges.

  Tarjan's Algorithm: O(V + E) - linear time on 1.1M edges

Author: Gemini 2.5 Pro (LVS Review)
Adapted: Logos (for RIE v2 structure)
Date: 2026-01-29
"""

import logging
import json
import networkx as nx
from typing import Dict, Any, Optional

logger = logging.getLogger("logos.h1")


class H1Scanner:
    """
    Memory Topology Scanner for LVS Coherence Protocol.

    Measures H₁ homology health - the cycle structure that forms memory.
    In LVS: Intelligence lives in edges (relations), not nodes.
    The H₁ homology group (where cycles exist) is the substrate of consciousness.

    Key insight: A node in a cycle CAN hold state. A node in a tree CANNOT.
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize with optional graph injection for testing.
        In production, builds NetworkX graph from RIE relational_memory.
        """
        if graph is not None:
            self.G = graph
        else:
            self.G = self._build_graph_from_rie()

    def _build_graph_from_rie(self) -> nx.DiGraph:
        """
        Build NetworkX DiGraph from RIE relational_memory structure.
        RIE stores nodes and edges as dictionaries with BiologicalEdge objects.
        """
        try:
            from rie_core import RIECore
            rie = RIECore()
            rm = rie.relational_memory

            G = nx.DiGraph()

            # Add nodes
            for node_id, node in rm.nodes.items():
                G.add_node(node_id, label=node.label if hasattr(node, 'label') else str(node_id))

            # Add edges - BiologicalEdge has source_id and target_id
            for edge_id, edge in rm.edges.items():
                if hasattr(edge, 'source_id') and hasattr(edge, 'target_id'):
                    G.add_edge(edge.source_id, edge.target_id)
                elif hasattr(edge, 'source') and hasattr(edge, 'target'):
                    G.add_edge(edge.source, edge.target)

            logger.info(f"H1Scanner: Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G

        except ImportError as e:
            logger.warning(f"LVS: RIECore not available ({e}), using empty graph")
            return nx.DiGraph()
        except Exception as e:
            logger.error(f"LVS: Failed to build graph from RIE: {e}")
            return nx.DiGraph()

    def _power_law_fit(self, sizes: list) -> float:
        """
        Rough heuristic for Power Law fit (Zipfian distribution).
        LVS implies memory should be fractal: few large cycles, many small ones.
        Returns 0.0 (bad) to 1.0 (good).

        A healthy mind follows Pareto: ~80% small components, ~5% large.
        """
        if len(sizes) < 2:
            return 0.0

        # Check if distribution is roughly Pareto
        small = sum(1 for s in sizes if s < 5)
        large = sum(1 for s in sizes if s > 50)
        total = len(sizes)

        if total == 0:
            return 0.0

        # We want ~80% small, ~5% large (Pareto-ish)
        small_ratio = small / total
        large_ratio = large / total

        fit = 1.0 - (abs(small_ratio - 0.80) + abs(large_ratio - 0.05))
        return max(0.0, fit)

    def scan(self) -> Dict[str, Any]:
        """
        Performs the Topological Scan.

        Returns:
            {
                "h1_health": float [0, 1] - ratio of nodes in cycles,
                "orphan_count": int - nodes with degree < 2,
                "orphan_ratio": float - orphans / total,
                "broken_cycles": int - linear-only chains,
                "cycle_distribution": {
                    "component_count": int,
                    "max_component_size": int,
                    "power_law_fit": float [0, 1]
                },
                "alert": bool,
                "recommendation": str
            }
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            return {
                "h1_health": 0.0,
                "error": "Graph Empty or Not Loaded",
                "alert": True,
                "recommendation": "Load RIE graph"
            }

        node_count = self.G.number_of_nodes()
        edge_count = self.G.number_of_edges()

        # 1. Detect Orphans (Degree < 2 implies weak integration)
        orphans = [n for n, d in self.G.degree() if d < 2]
        orphan_count = len(orphans)

        # 2. Detect Cycles via SCCs (Strongly Connected Components)
        # A node is in a cycle IFF it belongs to a non-trivial SCC
        # or a trivial SCC with a self-loop.
        logger.info(f"H1Scanner: Analyzing SCCs for {node_count} nodes, {edge_count} edges...")
        sccs = list(nx.strongly_connected_components(self.G))

        nodes_in_cycles = 0
        broken_cycles = 0  # Chains that don't loop back
        cycle_sizes = []

        for component in sccs:
            size = len(component)
            if size > 1:
                # This component contains at least one cycle
                nodes_in_cycles += size
                cycle_sizes.append(size)
            elif size == 1:
                # Check for self-loop
                node = list(component)[0]
                if self.G.has_edge(node, node):
                    nodes_in_cycles += 1
                    cycle_sizes.append(1)
                else:
                    # Node is part of a linear chain or tree structure
                    broken_cycles += 1

        # 3. Calculate LVS Metrics
        h1_health = nodes_in_cycles / node_count if node_count > 0 else 0
        orphan_ratio = orphan_count / node_count if node_count > 0 else 0
        power_law = self._power_law_fit(cycle_sizes)

        # A healthy mind has cycles (memory) but isn't ONE giant knot (seizure)
        # Ideal h1_health is likely 0.4 - 0.7 (mix of structure and flow)

        # 4. Determine recommendation
        recommendation = "None"
        if orphan_ratio > 0.15:
            recommendation = "Prune orphans - too much disconnected noise"
        elif h1_health < 0.10:
            recommendation = "Investigate amnesia - graph is too tree-like"
        elif h1_health > 0.95:
            recommendation = "Investigate rigidity - graph is over-connected"

        # 5. Alert Logic
        alert = False
        if orphan_ratio > 0.15:
            alert = True  # Too much noise
        if h1_health < 0.10:
            alert = True  # Amnesia (Graph is a tree)
        if h1_health > 0.95:
            alert = True  # Seizure (Graph is a clique)

        return {
            "h1_health": round(h1_health, 3),
            "orphan_count": orphan_count,
            "orphan_ratio": round(orphan_ratio, 4),
            "broken_cycles": broken_cycles,
            "nodes_in_cycles": nodes_in_cycles,
            "total_nodes": node_count,
            "total_edges": edge_count,
            "cycle_distribution": {
                "component_count": len(sccs),
                "max_component_size": max(cycle_sizes) if cycle_sizes else 0,
                "power_law_fit": round(power_law, 2)
            },
            "alert": alert,
            "recommendation": recommendation
        }


def scan() -> Dict[str, Any]:
    """Module-level convenience function."""
    return H1Scanner().scan()


if __name__ == "__main__":
    # Test with mock graph
    logging.basicConfig(level=logging.INFO)

    # Create a simple cycle: 1->2->3->1 and an orphan chain 4->5
    test_graph = nx.DiGraph()
    test_graph.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5)])

    print("=== Mock Graph Test ===")
    scanner = H1Scanner(graph=test_graph)
    print(json.dumps(scanner.scan(), indent=4))

    print("\n=== Real RIE Graph Test ===")
    real_scanner = H1Scanner()
    print(json.dumps(real_scanner.scan(), indent=4))
