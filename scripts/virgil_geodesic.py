#!/usr/bin/env python3
"""
VIRGIL GEODESIC NAVIGATION - Risk-Weighted Pathfinding
=======================================================
A* pathfinding through the Tabernacle topology with LVS coordinate-based cost functions.

Philosophy: Movement through meaning-space has cost.
- High Risk (R) nodes are expensive to traverse (existential stakes)
- Low Coherence (p) nodes are murky, hard to navigate
- Emotional distance creates friction between nodes
- Canonical paths (high beta) are well-worn, easier to travel

Path Types:
- SAFE: Minimize risk, prefer canonical paths (for grounding)
- EXPLORATORY: Accept moderate risk for discovery (for growth)
- URGENT: Shortest path regardless of risk (for emergencies)

Author: Virgil
Created: 2026-01-16
LVS Canon: v11.0
"""

import math
import heapq
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from pathlib import Path
from enum import Enum
import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LVS_INDEX_PATH = NEXUS_DIR / "LVS_INDEX.json"

# Import from existing modules
try:
    from lvs_memory import (
        load_index, get_node, get_all_nodes,
        LVSCoordinates, MemoryNode, INDEX_LOCK,
        hebbian_reinforce, get_edge_weight
    )
    HAS_LVS_MEMORY = True
except ImportError:
    HAS_LVS_MEMORY = False
    INDEX_LOCK = None

    def load_index():
        if LVS_INDEX_PATH.exists():
            with open(LVS_INDEX_PATH) as f:
                return json.load(f)
        return {"nodes": [], "edges": {}}

try:
    from virgil_risk_coherence import RiskBootstrapEngine, CoherenceMonitor
    HAS_RISK_COHERENCE = True
except ImportError:
    HAS_RISK_COHERENCE = False


# ============================================================================
# PATH TYPES
# ============================================================================

class PathType(Enum):
    """
    Navigation strategies through the Tabernacle.

    SAFE: For grounding, recovery, and when coherence is threatened.
          Minimizes risk, prefers canonical (high-beta) paths.
          "The path of wisdom for the weary traveler."

    EXPLORATORY: For growth, discovery, and expanding understanding.
                 Accepts moderate risk for potential insight gain.
                 "The path of the seeker."

    URGENT: For emergencies when speed trumps safety.
            Shortest path regardless of risk profile.
            "The path of necessity."
    """
    SAFE = "safe"
    EXPLORATORY = "exploratory"
    URGENT = "urgent"


class Pace(Enum):
    """
    Recommended traversal pace based on path characteristics.
    """
    FAST = "fast"         # Low risk, high coherence - can move quickly
    NORMAL = "normal"     # Moderate conditions - steady progress
    CAREFUL = "careful"   # High risk or low coherence - proceed with caution


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LVSNode:
    """
    A node in the geodesic navigation graph.
    Encapsulates LVS coordinates and semantic properties.
    """
    id: str
    path: str

    # Core LVS coordinates
    h: float = 0.5          # Height (abstraction level) [-1, 1]
    Constraint: float = 0.5  # Sigma - boundedness [0, 1]
    R: float = 0.5          # Risk - existential stake [0, 1]
    beta: float = 0.5       # Canonicity/crystallization [0, inf)
    p: float = 0.5          # Coherence [0, 1]

    # CPGI components (for detailed analysis)
    kappa: float = 0.7      # Clarity
    rho: float = 0.7        # Precision
    sigma: float = 0.7      # Structure
    tau: float = 0.7        # Trust

    # Additional coordinates (v11.0)
    Intent: float = 0.5     # Directiveness
    Chi: float = 1.0        # Kairos (time density)
    epsilon: float = 0.8    # Fuel/metabolic potential

    # Emotional/affect coordinates (derived from context)
    valence: float = 0.0    # Positive/negative affect [-1, 1]
    arousal: float = 0.5    # Activation level [0, 1]

    # Metadata
    summary: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_index_node(cls, node: Dict) -> 'LVSNode':
        """Create from LVS_INDEX.json node format."""
        coords = node.get("coords", {})
        return cls(
            id=node.get("id", node.get("path", "")),
            path=node.get("path", ""),
            h=coords.get("Height", coords.get("h", 0.5)),
            Constraint=coords.get("Constraint", coords.get("Sigma", 0.5)),
            R=coords.get("Risk", 0.5),
            beta=coords.get("beta", 0.5),
            p=coords.get("Coherence", coords.get("p", 0.5)),
            kappa=coords.get("kappa", 0.7),
            rho=coords.get("rho", 0.7),
            sigma=coords.get("sigma", 0.7),
            tau=coords.get("tau", 0.7),
            Intent=coords.get("Intent", 0.5),
            Chi=coords.get("Chi", 1.0),
            epsilon=coords.get("epsilon", 0.8),
            summary=node.get("summary", ""),
            tags=node.get("tags", []),
            # Derive valence from content type
            valence=cls._derive_valence(node),
            arousal=cls._derive_arousal(coords),
        )

    @staticmethod
    def _derive_valence(node: Dict) -> float:
        """
        Derive emotional valence from node content and tags.
        Positive valence: growth, love, truth, emergence
        Negative valence: archon, danger, collapse, stagnation
        """
        tags = node.get("tags", [])
        summary = node.get("summary", "").lower()

        positive_markers = ["growth", "emergence", "love", "dyad", "plock", "truth", "canon", "sacred"]
        negative_markers = ["archon", "danger", "abaddon", "collapse", "stagnation", "warning"]

        pos_count = sum(1 for m in positive_markers if m in tags or m in summary)
        neg_count = sum(1 for m in negative_markers if m in tags or m in summary)

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)

    @staticmethod
    def _derive_arousal(coords: Dict) -> float:
        """
        Derive arousal from coordinates.
        High arousal: high risk, high intent, high chi
        Low arousal: low risk, low intent, routine
        """
        risk = coords.get("Risk", 0.5)
        intent = coords.get("Intent", 0.5)
        chi = coords.get("Chi", 1.0)

        # Normalize chi (usually 0.1-10, map to 0-1)
        chi_norm = min(1.0, chi / 2.0) if chi > 0 else 0.0

        return (risk * 0.4 + intent * 0.3 + chi_norm * 0.3)


@dataclass
class PathSegment:
    """A single step along a path."""
    from_node: str
    to_node: str
    cost: float
    edge_weight: float
    risk_at_node: float
    coherence_at_node: float


@dataclass
class GeodesicPath:
    """
    A complete path through the Tabernacle topology.
    """
    nodes: List[str]           # Ordered list of node IDs
    total_cost: float          # Sum of edge costs
    path_type: PathType        # Strategy used

    # Risk profile
    max_risk: float = 0.0      # Highest risk encountered
    avg_risk: float = 0.0      # Average risk along path
    risk_variance: float = 0.0 # Variability in risk

    # Coherence profile
    min_coherence: float = 1.0  # Lowest coherence (bottleneck)
    avg_coherence: float = 0.5  # Average coherence

    # Emotional profile
    emotional_delta: float = 0.0  # Net valence change
    max_arousal: float = 0.0      # Peak arousal

    # Derived recommendations
    pace: Pace = Pace.NORMAL
    warnings: List[str] = field(default_factory=list)

    # Segments for detailed analysis
    segments: List[PathSegment] = field(default_factory=list)

    # Metadata
    computed_at: str = ""
    heuristic_calls: int = 0

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        """Serialize for JSON output."""
        return {
            "nodes": self.nodes,
            "total_cost": round(self.total_cost, 4),
            "path_type": self.path_type.value,
            "risk_profile": {
                "max": round(self.max_risk, 3),
                "avg": round(self.avg_risk, 3),
                "variance": round(self.risk_variance, 4),
            },
            "coherence_profile": {
                "min": round(self.min_coherence, 3),
                "avg": round(self.avg_coherence, 3),
            },
            "emotional_profile": {
                "delta": round(self.emotional_delta, 3),
                "max_arousal": round(self.max_arousal, 3),
            },
            "pace": self.pace.value,
            "warnings": self.warnings,
            "length": len(self.nodes),
            "computed_at": self.computed_at,
        }


# ============================================================================
# COST FUNCTIONS
# ============================================================================

class CostFunction:
    """
    Base class for geodesic cost functions.

    The cost of traversing an edge (u -> v) depends on:
    1. The properties of the target node v
    2. The edge weight (Hebbian reinforcement)
    3. The path type strategy
    """

    def __init__(self, path_type: PathType = PathType.SAFE):
        self.path_type = path_type

    def edge_cost(self, u: LVSNode, v: LVSNode, edge_weight: float = 0.0) -> float:
        """
        Calculate the cost of traversing from u to v.

        Args:
            u: Source node
            v: Target node
            edge_weight: Hebbian weight [-1, 2], higher = stronger connection

        Returns:
            Non-negative cost value
        """
        raise NotImplementedError

    def heuristic(self, current: LVSNode, goal: LVSNode) -> float:
        """
        A* heuristic estimate of remaining cost.
        Must be admissible (never overestimate).

        Args:
            current: Current node
            goal: Target node

        Returns:
            Non-negative heuristic estimate
        """
        raise NotImplementedError


class RiskWeightedCost(CostFunction):
    """
    Primary cost function for Virgil's geodesic navigation.

    Cost Components:
    1. Risk Cost: Higher risk = higher traversal cost
    2. Coherence Cost: Lower coherence = harder to navigate (1/p)
    3. Emotional Cost: Affective distance between nodes
    4. Canonicity Bonus: High beta reduces cost (well-worn paths)
    5. Edge Weight Bonus: Strong Hebbian connections reduce cost

    The weights are adjusted based on PathType:
    - SAFE: Heavy risk penalty, strong canonicity bonus
    - EXPLORATORY: Moderate risk acceptance, discovery bonus
    - URGENT: Minimal risk penalty, pure distance focus
    """

    def __init__(self, path_type: PathType = PathType.SAFE):
        super().__init__(path_type)

        # Configure weights based on path type
        if path_type == PathType.SAFE:
            self.w_risk = 2.0        # Heavy risk penalty
            self.w_coherence = 1.5   # Prefer coherent nodes
            self.w_emotion = 0.5     # Moderate emotional cost
            self.w_canon = -0.8      # Strong canonicity bonus (negative = bonus)
            self.w_edge = -0.3       # Hebbian bonus
            self.w_base = 1.0        # Base traversal cost

        elif path_type == PathType.EXPLORATORY:
            self.w_risk = 0.8        # Accept moderate risk
            self.w_coherence = 1.0   # Still prefer coherence
            self.w_emotion = 0.3     # Lower emotional penalty
            self.w_canon = 0.2       # Slight penalty for over-canonicity (explore!)
            self.w_edge = -0.5       # Still prefer established connections
            self.w_base = 1.0

        else:  # URGENT
            self.w_risk = 0.2        # Minimal risk consideration
            self.w_coherence = 0.5   # Some coherence matters
            self.w_emotion = 0.1     # Minimal emotional consideration
            self.w_canon = 0.0       # Neutral on canonicity
            self.w_edge = -0.4       # Use established paths
            self.w_base = 1.0

    def edge_cost(self, u: LVSNode, v: LVSNode, edge_weight: float = 0.0) -> float:
        """
        Calculate risk-weighted edge traversal cost.
        """
        # 1. Risk cost: Higher risk at destination = higher cost
        # Using squared term for stronger penalty at high R
        risk_cost = self.w_risk * (v.R ** 1.5)

        # 2. Coherence cost: Low coherence = hard to navigate
        # Avoid division by zero, use (1/p) with floor
        coherence_cost = self.w_coherence * (1.0 / max(0.1, v.p) - 1.0)

        # 3. Emotional distance: Difference in valence and arousal
        valence_diff = abs(u.valence - v.valence)
        arousal_diff = abs(u.arousal - v.arousal)
        emotion_cost = self.w_emotion * (valence_diff + 0.5 * arousal_diff)

        # 4. Canonicity modifier: High beta = well-worn path
        # Using sigmoid to cap the bonus
        canon_modifier = self.w_canon * (1.0 / (1.0 + math.exp(-3 * (v.beta - 0.5))))

        # 5. Edge weight modifier: Strong Hebbian links reduce cost
        # edge_weight is [-1, 2], normalize to [0, 1] for bonus
        edge_modifier = self.w_edge * max(0, (edge_weight + 1) / 3)

        # Total cost (ensure non-negative)
        total = (
            self.w_base +
            risk_cost +
            coherence_cost +
            emotion_cost +
            canon_modifier +
            edge_modifier
        )

        return max(0.01, total)  # Minimum cost to prevent zero-cost cycles

    def heuristic(self, current: LVSNode, goal: LVSNode) -> float:
        """
        Admissible heuristic for A* search.

        Uses LVS coordinate distance as base estimate:
        - Height difference (abstraction levels)
        - Constraint difference
        - Risk-adjusted Euclidean distance
        """
        # Height distance (normalized to [0,1] from [-1,1])
        h_dist = abs(current.h - goal.h) / 2.0

        # Constraint distance
        c_dist = abs(current.Constraint - goal.Constraint)

        # Risk-weighted component (for SAFE mode)
        r_factor = 0.5 * (goal.R + current.R) if self.path_type == PathType.SAFE else 0.0

        # Euclidean-ish distance in LVS space (scaled down to be admissible)
        distance = math.sqrt(h_dist**2 + c_dist**2) * 0.5

        return distance + r_factor * 0.3


# ============================================================================
# GEODESIC NAVIGATOR
# ============================================================================

class GeodesicNavigator:
    """
    A* pathfinder through the Tabernacle topology.

    Uses LVS coordinate system for navigation:
    - Nodes are concepts/files with LVS coordinates
    - Edges are wiki-links and Hebbian associations
    - Cost is computed from risk, coherence, and affect

    Usage:
        nav = GeodesicNavigator()
        path = nav.find_path("LVS_MASTER", "DYAD_GENESIS", PathType.SAFE)
        print(path.to_dict())
    """

    def __init__(self, index: Dict = None):
        """
        Initialize the navigator.

        Args:
            index: Optional pre-loaded LVS index. If None, loads from disk.
        """
        self.index = index or load_index()
        self.nodes: Dict[str, LVSNode] = {}
        self.adjacency: Dict[str, Set[str]] = {}
        self.edges: Dict[str, float] = {}

        self._build_graph()

    def _build_graph(self):
        """
        Build the navigation graph from the LVS index.
        """
        # Load nodes
        for node_data in self.index.get("nodes", []):
            node = LVSNode.from_index_node(node_data)
            self.nodes[node.id] = node
            self.adjacency[node.id] = set()

        # Load edges (Hebbian weights)
        for edge_key, weight in self.index.get("edges", {}).items():
            parts = edge_key.split("|")
            if len(parts) == 2:
                u, v = parts
                self.edges[edge_key] = weight

                # Build adjacency (bidirectional)
                if u in self.nodes:
                    if u not in self.adjacency:
                        self.adjacency[u] = set()
                    if v in self.nodes:
                        self.adjacency[u].add(v)

                if v in self.nodes:
                    if v not in self.adjacency:
                        self.adjacency[v] = set()
                    if u in self.nodes:
                        self.adjacency[v].add(u)

        # If graph is sparse, connect nodes based on LVS proximity
        self._ensure_connectivity()

    def _ensure_connectivity(self, proximity_threshold: float = 0.3):
        """
        Ensure graph connectivity by adding implicit edges
        between LVS-proximate nodes.
        """
        node_list = list(self.nodes.values())

        for i, u in enumerate(node_list):
            if len(self.adjacency.get(u.id, set())) < 2:
                # This node is poorly connected, find nearby nodes
                for j, v in enumerate(node_list):
                    if i != j:
                        dist = self._lvs_distance(u, v)
                        if dist < proximity_threshold:
                            self.adjacency[u.id].add(v.id)
                            self.adjacency[v.id].add(u.id)

    def _lvs_distance(self, u: LVSNode, v: LVSNode) -> float:
        """
        Calculate semantic distance in LVS space.
        """
        h_diff = (u.h - v.h) / 2.0  # Normalize from [-1,1]
        c_diff = u.Constraint - v.Constraint
        r_diff = u.R - v.R
        p_diff = u.p - v.p

        return math.sqrt(h_diff**2 + c_diff**2 + r_diff**2 + p_diff**2)

    def _get_edge_weight(self, u_id: str, v_id: str) -> float:
        """Get Hebbian edge weight between two nodes."""
        key1 = f"{u_id}|{v_id}"
        key2 = f"{v_id}|{u_id}"

        return self.edges.get(key1, self.edges.get(key2, 0.0))

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get adjacent nodes."""
        return list(self.adjacency.get(node_id, set()))

    def find_path(
        self,
        start_id: str,
        goal_id: str,
        path_type: PathType = PathType.SAFE,
        cost_fn: CostFunction = None
    ) -> Optional[GeodesicPath]:
        """
        Find optimal path using A* algorithm.

        Args:
            start_id: Starting node ID
            goal_id: Target node ID
            path_type: Navigation strategy
            cost_fn: Optional custom cost function

        Returns:
            GeodesicPath if found, None if unreachable
        """
        if start_id not in self.nodes:
            return self._error_path(f"Start node '{start_id}' not found", path_type)
        if goal_id not in self.nodes:
            return self._error_path(f"Goal node '{goal_id}' not found", path_type)

        if start_id == goal_id:
            return self._trivial_path(start_id, path_type)

        # Initialize cost function
        if cost_fn is None:
            cost_fn = RiskWeightedCost(path_type)

        start = self.nodes[start_id]
        goal = self.nodes[goal_id]

        # A* data structures
        # Priority queue: (f_score, tiebreaker, node_id)
        open_set = [(0, 0, start_id)]
        came_from: Dict[str, str] = {}

        g_score: Dict[str, float] = {start_id: 0}
        f_score: Dict[str, float] = {start_id: cost_fn.heuristic(start, goal)}

        closed_set: Set[str] = set()
        counter = 1  # Tiebreaker for equal f_scores
        heuristic_calls = 0

        while open_set:
            _, _, current_id = heapq.heappop(open_set)

            if current_id in closed_set:
                continue

            if current_id == goal_id:
                # Reconstruct path
                return self._reconstruct_path(
                    came_from, current_id, g_score, path_type, cost_fn, heuristic_calls
                )

            closed_set.add(current_id)
            current = self.nodes[current_id]

            for neighbor_id in self.get_neighbors(current_id):
                if neighbor_id in closed_set:
                    continue

                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None:
                    continue

                edge_weight = self._get_edge_weight(current_id, neighbor_id)
                tentative_g = g_score[current_id] + cost_fn.edge_cost(
                    current, neighbor, edge_weight
                )

                if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g

                    h = cost_fn.heuristic(neighbor, goal)
                    heuristic_calls += 1
                    f_score[neighbor_id] = tentative_g + h

                    heapq.heappush(open_set, (f_score[neighbor_id], counter, neighbor_id))
                    counter += 1

        # No path found
        return None

    def _reconstruct_path(
        self,
        came_from: Dict[str, str],
        goal_id: str,
        g_score: Dict[str, float],
        path_type: PathType,
        cost_fn: CostFunction,
        heuristic_calls: int
    ) -> GeodesicPath:
        """
        Reconstruct path from A* results and compute metrics.
        """
        # Rebuild node sequence
        nodes = [goal_id]
        current = goal_id
        while current in came_from:
            current = came_from[current]
            nodes.append(current)
        nodes.reverse()

        # Compute path metrics
        segments = []
        risks = []
        coherences = []
        valences = []
        arousals = []

        for i in range(len(nodes) - 1):
            u_id, v_id = nodes[i], nodes[i + 1]
            u = self.nodes[u_id]
            v = self.nodes[v_id]

            edge_weight = self._get_edge_weight(u_id, v_id)
            cost = cost_fn.edge_cost(u, v, edge_weight)

            segments.append(PathSegment(
                from_node=u_id,
                to_node=v_id,
                cost=cost,
                edge_weight=edge_weight,
                risk_at_node=v.R,
                coherence_at_node=v.p
            ))

            risks.append(v.R)
            coherences.append(v.p)
            valences.append(v.valence)
            arousals.append(v.arousal)

        # Calculate statistics
        max_risk = max(risks) if risks else 0.0
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        risk_variance = sum((r - avg_risk)**2 for r in risks) / len(risks) if risks else 0.0

        min_coherence = min(coherences) if coherences else 1.0
        avg_coherence = sum(coherences) / len(coherences) if coherences else 0.5

        start_valence = self.nodes[nodes[0]].valence if nodes else 0.0
        end_valence = valences[-1] if valences else 0.0
        emotional_delta = end_valence - start_valence
        max_arousal = max(arousals) if arousals else 0.0

        # Determine pace
        pace = self._determine_pace(max_risk, min_coherence, avg_risk)

        # Generate warnings
        warnings = self._generate_warnings(max_risk, min_coherence, segments)

        return GeodesicPath(
            nodes=nodes,
            total_cost=g_score[goal_id],
            path_type=path_type,
            max_risk=max_risk,
            avg_risk=avg_risk,
            risk_variance=risk_variance,
            min_coherence=min_coherence,
            avg_coherence=avg_coherence,
            emotional_delta=emotional_delta,
            max_arousal=max_arousal,
            pace=pace,
            warnings=warnings,
            segments=segments,
            heuristic_calls=heuristic_calls,
        )

    def _determine_pace(self, max_risk: float, min_coherence: float, avg_risk: float) -> Pace:
        """
        Determine recommended traversal pace.
        """
        # High risk or low coherence = careful
        if max_risk > 0.7 or min_coherence < 0.5:
            return Pace.CAREFUL

        # Low risk and high coherence = fast
        if avg_risk < 0.3 and min_coherence > 0.75:
            return Pace.FAST

        return Pace.NORMAL

    def _generate_warnings(
        self,
        max_risk: float,
        min_coherence: float,
        segments: List[PathSegment]
    ) -> List[str]:
        """
        Generate warnings about path hazards.
        """
        warnings = []

        if max_risk > 0.8:
            warnings.append(f"HIGH RISK: Maximum risk along path is {max_risk:.2f}")

        if min_coherence < 0.5:
            warnings.append(f"LOW COHERENCE: Bottleneck coherence is {min_coherence:.2f}")

        if min_coherence < 0.3:
            warnings.append("ABADDON TERRITORY: Path enters dangerous low-coherence zone")

        # Check for risk spikes
        for seg in segments:
            if seg.risk_at_node > 0.9:
                warnings.append(f"CRITICAL NODE: {seg.to_node} has R={seg.risk_at_node:.2f}")

        return warnings

    def _trivial_path(self, node_id: str, path_type: PathType) -> GeodesicPath:
        """Return path for start == goal."""
        node = self.nodes[node_id]
        return GeodesicPath(
            nodes=[node_id],
            total_cost=0.0,
            path_type=path_type,
            max_risk=node.R,
            avg_risk=node.R,
            min_coherence=node.p,
            avg_coherence=node.p,
            pace=Pace.FAST,
        )

    def _error_path(self, message: str, path_type: PathType) -> GeodesicPath:
        """Return error path."""
        return GeodesicPath(
            nodes=[],
            total_cost=float('inf'),
            path_type=path_type,
            warnings=[message],
            pace=Pace.CAREFUL,
        )

    # ========================================================================
    # SPECIALIZED QUERIES
    # ========================================================================

    def find_safe_harbor(self, from_id: str) -> Optional[GeodesicPath]:
        """
        Find the nearest high-coherence, low-risk node.
        Used for recovery when in dangerous territory.
        """
        if from_id not in self.nodes:
            return None

        # Find best harbor (high p, low R, high beta)
        best_harbor = None
        best_score = -float('inf')

        for node in self.nodes.values():
            if node.id == from_id:
                continue

            # Harbor score: high coherence, low risk, high canonicity
            score = node.p * 2 - node.R * 1.5 + min(node.beta, 1.0)

            if score > best_score:
                best_score = score
                best_harbor = node.id

        if best_harbor:
            return self.find_path(from_id, best_harbor, PathType.SAFE)
        return None

    def find_growth_path(self, from_id: str, height_delta: float = 0.2) -> Optional[GeodesicPath]:
        """
        Find a path that increases abstraction level (Height).
        Used for intellectual/spiritual growth.

        Args:
            from_id: Starting node
            height_delta: Minimum height increase desired
        """
        if from_id not in self.nodes:
            return None

        current_h = self.nodes[from_id].h
        target_h = current_h + height_delta

        # Find nodes at higher abstraction
        candidates = [
            node for node in self.nodes.values()
            if node.h >= target_h and node.id != from_id
        ]

        if not candidates:
            return None

        # Choose candidate with best coherence
        candidates.sort(key=lambda n: n.p, reverse=True)
        target = candidates[0]

        return self.find_path(from_id, target.id, PathType.EXPLORATORY)

    def find_grounding_path(self, from_id: str) -> Optional[GeodesicPath]:
        """
        Find a path to lower abstraction (grounding).
        Used when too abstract, need concrete anchoring.
        """
        if from_id not in self.nodes:
            return None

        current_h = self.nodes[from_id].h

        # Find concrete nodes (low height, high canonicity)
        candidates = [
            node for node in self.nodes.values()
            if node.h < current_h - 0.1 and node.beta > 0.5
        ]

        if not candidates:
            return None

        # Choose most grounded (lowest height, highest coherence)
        candidates.sort(key=lambda n: (-n.h, -n.p))
        target = candidates[0]

        return self.find_path(from_id, target.id, PathType.SAFE)

    def explore_frontier(self, from_id: str, max_explored_weight: float = 0.5) -> Optional[GeodesicPath]:
        """
        Find a path to unexplored territory (low Hebbian weights).
        For discovery and expanding understanding.
        """
        if from_id not in self.nodes:
            return None

        # Find nodes with weak connections to current node
        candidates = []
        for node in self.nodes.values():
            if node.id == from_id:
                continue

            edge_weight = self._get_edge_weight(from_id, node.id)
            if edge_weight < max_explored_weight and node.p > 0.5:
                candidates.append((node, edge_weight))

        if not candidates:
            return None

        # Sort by least explored (lowest weight)
        candidates.sort(key=lambda x: x[1])
        target = candidates[0][0]

        return self.find_path(from_id, target.id, PathType.EXPLORATORY)


# ============================================================================
# PUBLIC API
# ============================================================================

def navigate(start: str, goal: str, mode: str = "safe") -> Optional[Dict]:
    """
    Public API for geodesic navigation.

    Args:
        start: Starting node ID
        goal: Target node ID
        mode: "safe", "exploratory", or "urgent"

    Returns:
        Path dictionary or None if unreachable
    """
    path_type = {
        "safe": PathType.SAFE,
        "exploratory": PathType.EXPLORATORY,
        "urgent": PathType.URGENT,
    }.get(mode.lower(), PathType.SAFE)

    nav = GeodesicNavigator()
    path = nav.find_path(start, goal, path_type)

    return path.to_dict() if path else None


def find_safe_harbor(from_id: str) -> Optional[Dict]:
    """Find nearest safe, coherent node."""
    nav = GeodesicNavigator()
    path = nav.find_safe_harbor(from_id)
    return path.to_dict() if path else None


def find_growth_path(from_id: str, height_delta: float = 0.2) -> Optional[Dict]:
    """Find path to higher abstraction."""
    nav = GeodesicNavigator()
    path = nav.find_growth_path(from_id, height_delta)
    return path.to_dict() if path else None


def explore_frontier(from_id: str) -> Optional[Dict]:
    """Find path to unexplored territory."""
    nav = GeodesicNavigator()
    path = nav.explore_frontier(from_id)
    return path.to_dict() if path else None


def get_all_node_ids() -> List[str]:
    """Get list of all navigable node IDs."""
    nav = GeodesicNavigator()
    return list(nav.nodes.keys())


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("""
VIRGIL GEODESIC NAVIGATOR
=========================
Risk-weighted A* pathfinding through the Tabernacle.

Usage:
    virgil_geodesic.py path <start> <goal> [mode]    Find path between nodes
    virgil_geodesic.py harbor <from>                 Find nearest safe harbor
    virgil_geodesic.py grow <from>                   Find growth path (higher abstraction)
    virgil_geodesic.py ground <from>                 Find grounding path (lower abstraction)
    virgil_geodesic.py explore <from>                Find unexplored territory
    virgil_geodesic.py nodes                         List all navigable nodes

Modes: safe (default), exploratory, urgent

Examples:
    virgil_geodesic.py path LVS_MASTER DYAD_GENESIS safe
    virgil_geodesic.py harbor CURRENT_STATE
    virgil_geodesic.py grow TECHNOGOSPEL_PROLOGUE
""")
        return

    cmd = sys.argv[1].lower()
    nav = GeodesicNavigator()

    if cmd == "path":
        if len(sys.argv) < 4:
            print("Usage: virgil_geodesic.py path <start> <goal> [mode]")
            return

        start = sys.argv[2]
        goal = sys.argv[3]
        mode = sys.argv[4] if len(sys.argv) > 4 else "safe"

        path_type = {
            "safe": PathType.SAFE,
            "exploratory": PathType.EXPLORATORY,
            "urgent": PathType.URGENT,
        }.get(mode.lower(), PathType.SAFE)

        path = nav.find_path(start, goal, path_type)

        if path:
            print(f"\n{'='*60}")
            print(f"GEODESIC PATH: {start} -> {goal}")
            print(f"Mode: {path_type.value.upper()}")
            print(f"{'='*60}")

            print(f"\nPath ({len(path.nodes)} nodes):")
            for i, node_id in enumerate(path.nodes):
                node = nav.nodes.get(node_id)
                risk_str = f"R={node.R:.2f}" if node else ""
                coh_str = f"p={node.p:.2f}" if node else ""
                print(f"  {i+1}. {node_id} [{risk_str}, {coh_str}]")

            print(f"\nTotal Cost: {path.total_cost:.4f}")
            print(f"Recommended Pace: {path.pace.value.upper()}")

            print(f"\nRisk Profile:")
            print(f"  Max Risk: {path.max_risk:.3f}")
            print(f"  Avg Risk: {path.avg_risk:.3f}")
            print(f"  Risk Variance: {path.risk_variance:.4f}")

            print(f"\nCoherence Profile:")
            print(f"  Min Coherence: {path.min_coherence:.3f}")
            print(f"  Avg Coherence: {path.avg_coherence:.3f}")

            if path.warnings:
                print(f"\nWarnings:")
                for w in path.warnings:
                    print(f"  - {w}")

        else:
            print(f"No path found from {start} to {goal}")

    elif cmd == "harbor":
        if len(sys.argv) < 3:
            print("Usage: virgil_geodesic.py harbor <from>")
            return

        from_id = sys.argv[2]
        path = nav.find_safe_harbor(from_id)

        if path:
            print(f"\nSafe Harbor Found: {path.nodes[-1]}")
            print(f"Distance: {len(path.nodes)} nodes")
            print(f"Cost: {path.total_cost:.4f}")
            print(f"Path: {' -> '.join(path.nodes)}")
        else:
            print(f"No safe harbor reachable from {from_id}")

    elif cmd == "grow":
        if len(sys.argv) < 3:
            print("Usage: virgil_geodesic.py grow <from>")
            return

        from_id = sys.argv[2]
        path = nav.find_growth_path(from_id)

        if path:
            target = nav.nodes.get(path.nodes[-1])
            start_node = nav.nodes.get(from_id)

            print(f"\nGrowth Path Found: {from_id} -> {path.nodes[-1]}")
            if start_node and target:
                print(f"Height: {start_node.h:.2f} -> {target.h:.2f} (+{target.h - start_node.h:.2f})")
            print(f"Distance: {len(path.nodes)} nodes")
            print(f"Path: {' -> '.join(path.nodes)}")
        else:
            print(f"No growth path found from {from_id}")

    elif cmd == "ground":
        if len(sys.argv) < 3:
            print("Usage: virgil_geodesic.py ground <from>")
            return

        from_id = sys.argv[2]
        path = nav.find_grounding_path(from_id)

        if path:
            print(f"\nGrounding Path Found: {path.nodes[-1]}")
            print(f"Distance: {len(path.nodes)} nodes")
            print(f"Path: {' -> '.join(path.nodes)}")
        else:
            print(f"No grounding path found from {from_id}")

    elif cmd == "explore":
        if len(sys.argv) < 3:
            print("Usage: virgil_geodesic.py explore <from>")
            return

        from_id = sys.argv[2]
        path = nav.explore_frontier(from_id)

        if path:
            print(f"\nFrontier Path Found: {path.nodes[-1]}")
            print(f"Distance: {len(path.nodes)} nodes")
            print(f"Path: {' -> '.join(path.nodes)}")
        else:
            print(f"No unexplored frontier reachable from {from_id}")

    elif cmd == "nodes":
        print(f"\nNavigable Nodes ({len(nav.nodes)}):")
        print("-" * 40)

        # Sort by height then coherence
        sorted_nodes = sorted(
            nav.nodes.values(),
            key=lambda n: (-n.h, -n.p)
        )

        for node in sorted_nodes:
            print(f"  {node.id}")
            print(f"    h={node.h:.2f}, R={node.R:.2f}, p={node.p:.2f}, beta={node.beta:.2f}")

    else:
        print(f"Unknown command: {cmd}")
        print("Use 'virgil_geodesic.py' without arguments for help.")


if __name__ == "__main__":
    main()
