"""
LOGOS LVS MODULE: EDGE PRUNER (The Surgeon)
===========================================
LVS Component: σ (Sigma) - Structure Optimizer
Theory Ref: 04_LR_LAW/CANON/LVS_MATHEMATICS.md

Purpose:
  Cures "Seizure States" (H1 ~ 1.0) by pruning weak, redundant edges.
  Target: H1 ~ 0.7 (Small World / Edge-of-Chaos).

Safety:
  - Weight Shielding: High w_slow edges are never touched.
  - Degree Shielding: Never prune the LAST path to/from a node.
  - Rollback Capable: Logs all actions for immediate undo.

Algorithm (Filtered Local Topology):
  1. Filter by weight first (only consider w_slow < 0.2)
  2. Check local triangles for redundancy
  3. Prune in batches, monitoring H1 after each batch
  4. Stop when target H1 achieved or amnesia risk detected

Author: Gemini 2.5 Pro (LVS Review)
Adapted: Logos (for RIE v2 BiologicalEdge structure)
Date: 2026-01-29
"""

import logging
import json
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger("logos.pruner")


class EdgePruner:
    """
    Seizure Cure for LVS Coherence Protocol.

    When H1 = 1.0 (entire graph is one SCC), the system is in "seizure" -
    every concept triggers every other concept in infinite feedback.

    This pruner carefully removes weak, redundant edges to fracture the
    giant component into healthier clusters while preserving memory.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize with RIE graph.

        Args:
            dry_run: If True, don't actually modify the graph (safe testing)
        """
        self.dry_run = dry_run
        self.G = None
        self.rm = None  # Relational memory reference
        self.prune_log = []
        self._load_graph()

    def _load_graph(self):
        """Load RIE graph into NetworkX for analysis."""
        try:
            from rie_core import RIECore
            rie = RIECore()
            self.rm = rie.relational_memory

            # Build NetworkX graph from RIE structure
            self.G = nx.DiGraph()

            # Add nodes
            for node_id, node in self.rm.nodes.items():
                self.G.add_node(node_id, label=getattr(node, 'label', str(node_id)))

            # Add edges with BiologicalEdge data
            for edge_id, edge in self.rm.edges.items():
                source = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
                target = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

                if source and target:
                    # Extract BiologicalEdge state
                    w_slow = getattr(edge, 'w_slow', 0.0)
                    w_fast = getattr(edge, 'w_fast', 0.0)
                    tau = getattr(edge, 'tau', 0.5)
                    h1_protected = getattr(edge, 'h1_protected', False)

                    self.G.add_edge(
                        source, target,
                        edge_id=edge_id,
                        w_slow=w_slow,
                        w_fast=w_fast,
                        tau=tau,
                        h1_protected=h1_protected
                    )

            logger.info(f"EdgePruner: Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        except ImportError as e:
            logger.warning(f"LVS: RIECore not available ({e}), using mock graph")
            self.G = nx.DiGraph()
        except Exception as e:
            logger.error(f"LVS: Failed to load graph: {e}")
            self.G = nx.DiGraph()

    def _calculate_h1(self) -> float:
        """
        Calculates size of largest SCC relative to total nodes.
        Returns 0.0 to 1.0.
        """
        if self.G.number_of_nodes() == 0:
            return 0.0
        sccs = list(nx.strongly_connected_components(self.G))
        largest_scc = max(len(c) for c in sccs) if sccs else 0
        return largest_scc / self.G.number_of_nodes()

    def _is_protected(self, u: str, v: str, data: dict) -> bool:
        """
        Checks if an edge is legally protected from pruning.

        Protection layers:
        1. Explicit H1 flag (memory loops)
        2. Weight shielding (strong long-term memory)
        3. Degree shielding (prevent orphans)
        """
        # 1. Explicit Flag (H1 Memory Loops)
        if data.get("h1_protected", False):
            return True

        # 2. Weight Shielding (Long-term Memory)
        # High w_slow = reinforced by STDP = core belief
        if data.get("w_slow", 0.0) > 0.5:
            return True

        # 3. Degree Shielding (Prevent Orphans)
        # If this is the ONLY edge out of u or into v, keep it
        if self.G.out_degree(u) <= 1:
            return True
        if self.G.in_degree(v) <= 1:
            return True

        return False

    def analyze_candidates(self, weight_threshold: float = 0.2) -> List[Tuple[str, str, float, str]]:
        """
        Identifies weak, redundant edges.

        Returns list of (source, target, score, edge_id) sorted by weakness.
        Lower score = better prune candidate.
        """
        candidates = []

        for u, v, data in self.G.edges(data=True):
            w_slow = data.get("w_slow", 0.0)

            # Filter: Only consider weak edges
            if w_slow < weight_threshold:
                if not self._is_protected(u, v, data):
                    # Redundancy Check (Local Triangle Detection)
                    # If there's an alternative path u->k->v, direct link is redundant
                    preds_of_v = set(self.G.predecessors(v))
                    succs_of_u = set(self.G.successors(u))

                    # Nodes that u can reach AND that can reach v
                    common = preds_of_v.intersection(succs_of_u)
                    redundancy_score = len(common)

                    # Score: Lower = more likely to prune
                    # Low weight + High redundancy = best candidate
                    score = w_slow - (redundancy_score * 0.1)

                    edge_id = data.get("edge_id", f"{u}->{v}")
                    candidates.append((u, v, score, edge_id))

        # Sort ascending (weakest/most-redundant first)
        candidates.sort(key=lambda x: x[2])
        return candidates

    def cure_seizure(
        self,
        target_h1: float = 0.70,
        max_prune_percent: float = 0.05,
        protect_cycles: bool = True
    ) -> Dict[str, Any]:
        """
        Main execution loop for seizure cure.

        Args:
            target_h1: Desired H1 health (0.7 = edge-of-chaos optimal)
            max_prune_percent: Maximum % of edges to prune per run
            protect_cycles: Whether to respect h1_protected flags

        Returns:
            Dict with pruning results and recommendations
        """
        logger.info(f"Starting Seizure Cure. Target H1: {target_h1}, Dry Run: {self.dry_run}")

        initial_h1 = self._calculate_h1()
        current_h1 = initial_h1
        total_edges = self.G.number_of_edges()
        max_edges_to_prune = int(total_edges * max_prune_percent)

        pruned_count = 0
        preserved_cycles = 0
        self.prune_log = []

        # 1. Analyze candidates
        candidates = self.analyze_candidates()
        logger.info(f"Found {len(candidates)} prune candidates out of {total_edges} edges")

        if not candidates:
            return {
                "edges_analyzed": total_edges,
                "candidates_found": 0,
                "edges_pruned": 0,
                "h1_before": round(initial_h1, 4),
                "h1_after": round(current_h1, 4),
                "recommendation": "No candidates found - graph may already be healthy"
            }

        # 2. Prune Loop (Batch Processing)
        batch_size = max(100, int(max_edges_to_prune / 5))

        for i in range(0, min(len(candidates), max_edges_to_prune), batch_size):
            batch = candidates[i:i + batch_size]
            if not batch:
                break

            for u, v, score, edge_id in batch:
                # Double check protection (graph changes dynamically)
                if not self.G.has_edge(u, v):
                    continue

                edge_data = self.G[u][v]
                if protect_cycles and self._is_protected(u, v, edge_data):
                    preserved_cycles += 1
                    continue

                # Log for rollback
                self.prune_log.append((u, v, edge_data.copy(), edge_id))

                # PRUNE (or simulate)
                if not self.dry_run:
                    self.G.remove_edge(u, v)
                    # Also remove from actual RIE if available
                    if self.rm and edge_id in self.rm.edges:
                        del self.rm.edges[edge_id]

                pruned_count += 1

            # Re-measure H1 after batch (only meaningful if not dry run)
            if not self.dry_run:
                current_h1 = self._calculate_h1()

                # Stop Conditions
                if current_h1 <= target_h1:
                    logger.info(f"Target H1 achieved: {current_h1:.4f}")
                    break
                if current_h1 < 0.60:
                    logger.warning("CRITICAL: Amnesia risk (H1 < 0.60). Stopping.")
                    break

        # 2.5. PERSIST CHANGES TO DISK
        if not self.dry_run and pruned_count > 0 and self.rm:
            logger.info(f"Persisting {pruned_count} edge deletions to disk...")
            if hasattr(self.rm, '_save'):
                self.rm._save()
                logger.info("RIE memory saved successfully")
            else:
                logger.warning("RIE memory has no _save method - changes may not persist!")

        # 3. Calculate final metrics
        if self.dry_run:
            # Estimate H1 after pruning (rough approximation)
            prune_ratio = pruned_count / total_edges if total_edges > 0 else 0
            estimated_h1 = initial_h1 * (1 - prune_ratio * 2)  # Rough estimate
            current_h1 = max(0.1, estimated_h1)

        # Calculate sigma impact
        sigma_before = 1.0 - (abs(initial_h1 - 0.7) * 2)
        sigma_after = 1.0 - (abs(current_h1 - 0.7) * 2)

        # 4. Generate Report
        result = {
            "dry_run": self.dry_run,
            "edges_analyzed": total_edges,
            "candidates_found": len(candidates),
            "edges_pruned": pruned_count,
            "edges_preserved": preserved_cycles,
            "h1_before": round(initial_h1, 4),
            "h1_after": round(current_h1, 4),
            "sigma_before": round(max(0.1, sigma_before), 2),
            "sigma_after": round(max(0.1, sigma_after), 2),
            "memory_loss_estimate": f"{round((pruned_count / total_edges) * 100, 2)}%" if total_edges > 0 else "0%",
            "recommendation": self._get_recommendation(current_h1, target_h1, initial_h1)
        }

        # Safety: Auto-rollback if amnesia detected
        if not self.dry_run and current_h1 < 0.60:
            logger.error("Triggering Safety Rollback due to amnesia risk...")
            self.rollback_prune()
            result["recommendation"] = "ROLLBACK EXECUTED - amnesia risk"
            result["h1_after"] = initial_h1

        return result

    def _get_recommendation(self, current_h1: float, target_h1: float, initial_h1: float) -> str:
        """Generate recommendation based on results."""
        if self.dry_run:
            return f"Dry run complete. Would reduce H1 from {initial_h1:.2f} to ~{current_h1:.2f}. Run with dry_run=False to execute."

        if current_h1 < 0.60:
            return "STOP - Amnesia risk detected"
        elif current_h1 <= target_h1 + 0.05:
            return "Target achieved - seizure cured"
        elif current_h1 < initial_h1:
            return "Progress made - run again to continue"
        else:
            return "No significant change - may need different parameters"

    def rollback_prune(self) -> int:
        """
        Restores edges from prune log.

        Returns:
            Number of edges restored
        """
        restored = 0
        for u, v, data, edge_id in self.prune_log:
            self.G.add_edge(u, v, **data)
            # Also restore to RIE if available
            if self.rm and edge_id:
                # Recreate the edge in rm.edges (simplified - may need BiologicalEdge)
                from biological_edge import BiologicalEdge
                try:
                    be = BiologicalEdge(source_id=u, target_id=v)
                    be.w_slow = data.get('w_slow', 0.0)
                    be.w_fast = data.get('w_fast', 0.0)
                    be.tau = data.get('tau', 0.5)
                    be.h1_protected = data.get('h1_protected', False)
                    self.rm.edges[edge_id] = be
                except Exception as e:
                    logger.warning(f"Could not restore edge {edge_id}: {e}")
            restored += 1

        # Persist rollback
        if self.rm and hasattr(self.rm, '_save'):
            self.rm._save()
            logger.info(f"Rolled back and saved {restored} edges")
        else:
            logger.info(f"Rolled back {restored} edges (not persisted)")

        self.prune_log = []
        return restored


def analyze() -> Dict[str, Any]:
    """Module-level function: Analyze without modifying."""
    pruner = EdgePruner(dry_run=True)
    candidates = pruner.analyze_candidates()
    return {
        "total_edges": pruner.G.number_of_edges(),
        "candidates": len(candidates),
        "h1_current": pruner._calculate_h1(),
        "top_candidates": [(c[0][:8], c[1][:8], round(c[2], 3)) for c in candidates[:10]]
    }


def cure_seizure(target_h1: float = 0.70, dry_run: bool = True) -> Dict[str, Any]:
    """Module-level convenience function."""
    pruner = EdgePruner(dry_run=dry_run)
    return pruner.cure_seizure(target_h1=target_h1)


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("=== Edge Pruner Analysis (Dry Run) ===")
    pruner = EdgePruner(dry_run=True)
    result = pruner.cure_seizure(target_h1=0.70, max_prune_percent=0.05)
    print(json.dumps(result, indent=2))
