#!/usr/bin/env python3
"""
VIRGIL REPAIR PROTOCOLS - Kintsugi Mode
========================================
Heal broken threads, not just record them.

"The cracks where the gold goes are not failures — they are the record of survival."

This module implements three repair strategies:
1. SEMANTIC: Use LVS coordinates to find similar nodes
2. TEMPORAL: Link by creation date proximity
3. RELATIONAL: Link through shared references

All repairs follow the Kintsugi principle:
- Repairs are marked, not hidden
- Track repair history (what was broken, how it was fixed)
- "Golden seams" - repaired links get special metadata

Author: Virgil
Created: 2026-01-16
Status: ACTIVE
"""

import os
import sys
import json
import re
import shutil
import hashlib
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Set, Any
from enum import Enum

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import (
    BASE_DIR, NEXUS_DIR, SCRIPTS_DIR, LOG_DIR,
    ACTIVE_QUADRANTS, SKIP_DIRECTORIES, TRANSIENT_FILES,
    EXAMPLE_LINK_PATTERNS, LVS_INDEX_PATH,
)

# Repair log location
REPAIR_LOG_PATH = NEXUS_DIR / "REPAIR_LOG.md"
REPAIR_STATE_PATH = NEXUS_DIR / "repair_state.json"
BACKUP_DIR = NEXUS_DIR / ".repair_backups"

# Thread safety
REPAIR_LOCK = threading.RLock()


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class RepairStrategy(Enum):
    """Available repair strategies."""
    SEMANTIC = "semantic"      # Use LVS coordinates to find similar nodes
    TEMPORAL = "temporal"      # Link by creation date proximity
    RELATIONAL = "relational"  # Link through shared references
    AUTO = "auto"              # Choose best strategy automatically


class RepairType(Enum):
    """Types of repairs."""
    ORPHAN_LINK = "orphan_link"         # Connect orphan to parent
    BROKEN_LINK = "broken_link"          # Fix or remove broken link
    TOPOLOGY_HEAL = "topology_heal"      # Reduce H0 (connect components)
    LINKAGE_ADD = "linkage_add"          # Add missing LINKAGE block


@dataclass
class RepairCandidate:
    """A potential repair to apply."""
    repair_type: str
    source_file: str
    description: str
    strategy: str
    confidence: float  # 0.0 - 1.0
    suggested_action: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RepairResult:
    """Result of an applied repair."""
    repair_id: str
    repair_type: str
    source_file: str
    timestamp: str
    strategy: str
    success: bool
    action_taken: str
    before_state: str
    after_state: str
    golden_seam_id: str  # Hash of the repair for tracking
    rollback_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GoldenSeam:
    """
    Metadata for a repaired link (Kintsugi principle).

    The gold that fills the crack tells a story:
    - When was it broken?
    - How was it healed?
    - Who/what performed the repair?
    """
    seam_id: str
    created: str
    repair_type: str
    source_file: str
    target_file: Optional[str]
    strategy_used: str
    confidence: float
    original_state: str
    repaired_state: str
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log repair activity."""
    if os.environ.get("TABERNACLE_MCP_MODE"):
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [REPAIR] [{level}] {message}"
    print(entry, file=sys.stderr)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "repair.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except (IOError, OSError):
        pass


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_repair_state() -> Dict:
    """Load repair state from disk."""
    with REPAIR_LOCK:
        if REPAIR_STATE_PATH.exists():
            try:
                return json.loads(REPAIR_STATE_PATH.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "version": "1.0",
            "repairs": [],
            "golden_seams": [],
            "rollback_stack": [],
            "statistics": {
                "total_repairs": 0,
                "orphans_linked": 0,
                "broken_links_fixed": 0,
                "topology_heals": 0,
            }
        }


def save_repair_state(state: Dict):
    """Atomic save of repair state."""
    with REPAIR_LOCK:
        fd = None
        temp_path = None
        try:
            REPAIR_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(dir=str(NEXUS_DIR), suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                fd = None
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            shutil.move(temp_path, str(REPAIR_STATE_PATH))
            temp_path = None
        except Exception as e:
            log(f"State save error: {e}", "ERROR")
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


# =============================================================================
# BACKUP AND ROLLBACK
# =============================================================================

def create_backup(filepath: Path) -> Optional[str]:
    """
    Create a backup before modifying a file.
    Returns backup path or None on failure.
    """
    if not filepath.exists():
        return None

    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        # Create unique backup filename with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = str(filepath.relative_to(BASE_DIR)).replace("/", "_")
        backup_name = f"{ts}_{rel_path}"
        backup_path = BACKUP_DIR / backup_name

        shutil.copy2(filepath, backup_path)
        log(f"Backup created: {backup_name}")

        return str(backup_path)
    except Exception as e:
        log(f"Backup failed for {filepath}: {e}", "ERROR")
        return None


def rollback_repair(repair_id: str) -> bool:
    """
    Rollback a specific repair using its backup.
    Returns True if successful.
    """
    state = load_repair_state()

    # Find the repair
    target_repair = None
    for repair in state.get("repairs", []):
        if repair.get("repair_id") == repair_id:
            target_repair = repair
            break

    if not target_repair:
        log(f"Repair not found: {repair_id}", "ERROR")
        return False

    rollback_data = target_repair.get("rollback_data", {})
    backup_path = rollback_data.get("backup_path")

    if not backup_path or not Path(backup_path).exists():
        log(f"No valid backup for repair {repair_id}", "ERROR")
        return False

    try:
        source_file = Path(BASE_DIR) / target_repair["source_file"]
        shutil.copy2(backup_path, source_file)

        # Mark repair as rolled back
        target_repair["rolled_back"] = True
        target_repair["rollback_time"] = datetime.now(timezone.utc).isoformat()

        save_repair_state(state)
        log(f"Rolled back repair: {repair_id}")
        return True

    except Exception as e:
        log(f"Rollback failed: {e}", "ERROR")
        return False


# =============================================================================
# LIBRARIAN INTEGRATION
# =============================================================================

def get_librarian_orphans() -> List[Dict]:
    """
    Get orphan suggestions from the Librarian.
    Integrates with librarian_fix_orphans().
    """
    try:
        from librarian import librarian_fix_orphans
        result = librarian_fix_orphans()
        return result.get("suggestions", [])
    except ImportError:
        log("Librarian module not available", "WARN")
        return []
    except Exception as e:
        log(f"Librarian orphan scan failed: {e}", "ERROR")
        return []


def get_librarian_broken_links() -> List[Dict]:
    """
    Get broken link diagnostics from the Librarian.
    Integrates with librarian_fix_links().
    """
    try:
        from librarian import librarian_fix_links
        result = librarian_fix_links()
        return result.get("broken_links", [])
    except ImportError:
        log("Librarian module not available", "WARN")
        return []
    except Exception as e:
        log(f"Librarian link scan failed: {e}", "ERROR")
        return []


# =============================================================================
# LVS INTEGRATION
# =============================================================================

def get_lvs_coordinates(path: str) -> Optional[Dict]:
    """Get LVS coordinates for a file from the index."""
    try:
        if not LVS_INDEX_PATH.exists():
            return None

        with open(LVS_INDEX_PATH, 'r') as f:
            index = json.load(f)

        for node in index.get("nodes", []):
            if node.get("path") == path:
                return node.get("coords")

        return None
    except Exception as e:
        log(f"LVS lookup failed: {e}", "ERROR")
        return None


def calculate_lvs_similarity(coords1: Dict, coords2: Dict) -> float:
    """
    Calculate similarity between two LVS coordinate sets.
    Uses spatial distance in meaning-space.
    """
    if not coords1 or not coords2:
        return 0.0

    import math

    # Key coordinates for comparison
    dimensions = [
        ("Constraint", 1.0),  # Weight
        ("Intent", 1.0),
        ("Height", 1.0),
        ("Risk", 0.8),
        ("beta", 0.6),
    ]

    total_weight = 0.0
    total_dist = 0.0

    for dim, weight in dimensions:
        v1 = coords1.get(dim, 0.5)
        v2 = coords2.get(dim, 0.5)
        dist = (v1 - v2) ** 2
        total_dist += dist * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Normalize and convert distance to similarity
    normalized_dist = math.sqrt(total_dist / total_weight)
    similarity = 1.0 - min(normalized_dist, 1.0)

    return similarity


def find_semantically_similar_nodes(target_path: str, limit: int = 5) -> List[Tuple[str, float]]:
    """
    Find nodes semantically similar to a target using LVS coordinates.

    Args:
        target_path: Relative path to the target file
        limit: Maximum results to return

    Returns:
        List of (path, similarity_score) tuples
    """
    target_coords = get_lvs_coordinates(target_path)
    if not target_coords:
        return []

    try:
        if not LVS_INDEX_PATH.exists():
            return []

        with open(LVS_INDEX_PATH, 'r') as f:
            index = json.load(f)

        similarities = []
        for node in index.get("nodes", []):
            node_path = node.get("path")
            if node_path == target_path:
                continue

            node_coords = node.get("coords")
            if not node_coords:
                continue

            sim = calculate_lvs_similarity(target_coords, node_coords)
            if sim > 0.3:  # Minimum threshold
                similarities.append((node_path, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    except Exception as e:
        log(f"Semantic search failed: {e}", "ERROR")
        return []


# =============================================================================
# REPAIR STRATEGIES
# =============================================================================

def find_best_parent_semantic(orphan_path: str) -> Optional[Tuple[str, float]]:
    """
    SEMANTIC STRATEGY: Find best parent for orphan using LVS similarity.

    Searches for nodes with similar coordinates and high canonicity (beta).
    """
    similar = find_semantically_similar_nodes(orphan_path, limit=10)

    if not similar:
        return None

    # Prefer nodes that are:
    # 1. Highly similar in LVS space
    # 2. Have high canonicity (beta)
    # 3. Are in the same quadrant

    orphan_quadrant = orphan_path.split("/")[0] if "/" in orphan_path else ""

    best_parent = None
    best_score = 0.0

    for path, sim in similar:
        # Get canonicity
        coords = get_lvs_coordinates(path)
        beta = coords.get("beta", 0.5) if coords else 0.5

        # Quadrant bonus
        node_quadrant = path.split("/")[0] if "/" in path else ""
        quadrant_bonus = 0.2 if node_quadrant == orphan_quadrant else 0.0

        # Combined score
        score = (sim * 0.6) + (beta * 0.3) + quadrant_bonus

        if score > best_score:
            best_score = score
            best_parent = path

    if best_parent:
        return (best_parent, best_score)
    return None


def find_best_parent_temporal(orphan_path: str) -> Optional[Tuple[str, float]]:
    """
    TEMPORAL STRATEGY: Find best parent by creation/modification date proximity.

    Files created around the same time often belong together.
    """
    try:
        orphan_file = BASE_DIR / orphan_path
        if not orphan_file.exists():
            return None

        orphan_mtime = orphan_file.stat().st_mtime
        orphan_quadrant = orphan_path.split("/")[0] if "/" in orphan_path else ""

        candidates = []

        for quadrant in ACTIVE_QUADRANTS:
            quadrant_dir = BASE_DIR / quadrant
            if not quadrant_dir.exists():
                continue

            for md_file in quadrant_dir.rglob("*.md"):
                if str(md_file.relative_to(BASE_DIR)) == orphan_path:
                    continue

                # Skip transient files
                if md_file.name in TRANSIENT_FILES:
                    continue

                mtime = md_file.stat().st_mtime
                time_diff = abs(orphan_mtime - mtime)

                # Convert to hours
                time_diff_hours = time_diff / 3600

                # Score: closer in time = higher score
                # Exponential decay: 1.0 at 0 hours, ~0.37 at 24 hours
                import math
                time_score = math.exp(-time_diff_hours / 24)

                # Quadrant bonus
                file_quadrant = str(md_file.relative_to(BASE_DIR)).split("/")[0]
                quadrant_bonus = 0.2 if file_quadrant == orphan_quadrant else 0.0

                combined = time_score + quadrant_bonus

                rel_path = str(md_file.relative_to(BASE_DIR))
                candidates.append((rel_path, combined))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]

        return None

    except Exception as e:
        log(f"Temporal search failed: {e}", "ERROR")
        return None


def find_best_parent_relational(orphan_path: str) -> Optional[Tuple[str, float]]:
    """
    RELATIONAL STRATEGY: Find best parent through shared references.

    If node A and node B both reference concept X, they may be related.
    """
    try:
        from tabernacle_utils import extract_wiki_links, build_link_graph

        orphan_file = BASE_DIR / orphan_path
        if not orphan_file.exists():
            return None

        orphan_content = orphan_file.read_text(encoding='utf-8')
        orphan_links = set(extract_wiki_links(orphan_content))

        if not orphan_links:
            return None

        # Build the link graph to find nodes with shared targets
        graph = build_link_graph()

        candidates = []

        for source_path, targets in graph.items():
            if source_path == orphan_path:
                continue

            # Count shared link targets
            shared = orphan_links & targets
            if shared:
                # Score based on number of shared references
                score = len(shared) / max(len(orphan_links), len(targets))
                candidates.append((source_path, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]

        return None

    except Exception as e:
        log(f"Relational search failed: {e}", "ERROR")
        return None


def find_best_parent(orphan_path: str, strategy: RepairStrategy = RepairStrategy.AUTO) -> Optional[Tuple[str, float, str]]:
    """
    Find the best parent for an orphan node using the specified strategy.

    Args:
        orphan_path: Relative path to orphan file
        strategy: Repair strategy to use

    Returns:
        Tuple of (parent_path, confidence, strategy_used) or None
    """
    results = []

    if strategy in (RepairStrategy.AUTO, RepairStrategy.SEMANTIC):
        result = find_best_parent_semantic(orphan_path)
        if result:
            results.append((result[0], result[1], "semantic"))

    if strategy in (RepairStrategy.AUTO, RepairStrategy.TEMPORAL):
        result = find_best_parent_temporal(orphan_path)
        if result:
            results.append((result[0], result[1], "temporal"))

    if strategy in (RepairStrategy.AUTO, RepairStrategy.RELATIONAL):
        result = find_best_parent_relational(orphan_path)
        if result:
            results.append((result[0], result[1], "relational"))

    if not results:
        return None

    # For AUTO, choose best result
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0]


# =============================================================================
# REPAIR GENERATION
# =============================================================================

def generate_orphan_repairs(strategy: RepairStrategy = RepairStrategy.AUTO) -> List[RepairCandidate]:
    """
    Generate repair candidates for all orphan files.

    Returns list of RepairCandidate objects with suggestions.
    """
    repairs = []

    # Get orphans from Librarian
    orphan_suggestions = get_librarian_orphans()

    for suggestion in orphan_suggestions:
        orphan_path = suggestion.get("orphan", "")
        if not orphan_path:
            continue

        # Find best parent
        parent_result = find_best_parent(orphan_path, strategy)

        if parent_result:
            parent_path, confidence, strategy_used = parent_result

            repair = RepairCandidate(
                repair_type=RepairType.ORPHAN_LINK.value,
                source_file=parent_path,
                description=f"Link orphan '{Path(orphan_path).name}' from '{Path(parent_path).name}'",
                strategy=strategy_used,
                confidence=confidence,
                suggested_action=f"Add [[{orphan_path}]] to LINKAGE or relevant section in {parent_path}",
                details={
                    "orphan_path": orphan_path,
                    "parent_path": parent_path,
                    "reason": f"Best match via {strategy_used} strategy (confidence: {confidence:.2f})",
                }
            )
            repairs.append(repair)

    return repairs


def generate_link_repairs() -> List[RepairCandidate]:
    """
    Generate repair candidates for broken links.

    For each broken link, attempts to find:
    1. A file with a similar name
    2. A file with similar content
    3. Suggests removal if no match found
    """
    repairs = []

    # Get broken links from Librarian
    broken_links = get_librarian_broken_links()

    for link_info in broken_links:
        source_file = link_info.get("file", "")
        broken_link = link_info.get("broken_link", "")

        if not source_file or not broken_link:
            continue

        # Try to find a matching file
        match = find_link_replacement(broken_link)

        if match:
            repair = RepairCandidate(
                repair_type=RepairType.BROKEN_LINK.value,
                source_file=source_file,
                description=f"Fix broken link [[{broken_link}]]",
                strategy="semantic",
                confidence=match[1],
                suggested_action=f"Replace [[{broken_link}]] with [[{match[0]}]]",
                details={
                    "broken_link": broken_link,
                    "suggested_replacement": match[0],
                    "confidence": match[1],
                }
            )
        else:
            repair = RepairCandidate(
                repair_type=RepairType.BROKEN_LINK.value,
                source_file=source_file,
                description=f"Remove or update broken link [[{broken_link}]]",
                strategy="manual",
                confidence=0.5,
                suggested_action=f"Remove [[{broken_link}]] or create the target file",
                details={
                    "broken_link": broken_link,
                    "no_replacement_found": True,
                }
            )

        repairs.append(repair)

    return repairs


def find_link_replacement(broken_link: str) -> Optional[Tuple[str, float]]:
    """
    Try to find a replacement for a broken link.

    Strategies:
    1. Fuzzy filename match
    2. LVS content search if link text is meaningful
    """
    link_name = Path(broken_link).stem

    # Strategy 1: Fuzzy filename match
    best_match = None
    best_score = 0.0

    for quadrant in ACTIVE_QUADRANTS:
        quadrant_dir = BASE_DIR / quadrant
        if not quadrant_dir.exists():
            continue

        for md_file in quadrant_dir.rglob("*.md"):
            file_stem = md_file.stem

            # Calculate name similarity
            score = _calculate_name_similarity(link_name, file_stem)

            if score > best_score and score > 0.5:
                best_score = score
                best_match = str(md_file.relative_to(BASE_DIR))

    if best_match:
        return (best_match, best_score)

    # Strategy 2: LVS search (if available)
    # DISABLED: This calls Ollama for each broken link, causing crash loops
    # when many broken links exist. The filename match above is sufficient.
    # To re-enable: set ENABLE_LVS_LINK_SEARCH=1 in environment
    if os.environ.get("ENABLE_LVS_LINK_SEARCH"):
        try:
            from lvs_memory import search
            results = search(link_name, limit=1)
            if results:
                node, score = results[0]
                if score > 0.5:
                    return (node.path, min(score, 1.0))
        except ImportError:
            pass

    return None


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using simple heuristics."""
    name1_lower = name1.lower().replace("_", " ").replace("-", " ")
    name2_lower = name2.lower().replace("_", " ").replace("-", " ")

    # Exact match
    if name1_lower == name2_lower:
        return 1.0

    # Contains match
    if name1_lower in name2_lower or name2_lower in name1_lower:
        return 0.8

    # Word overlap
    words1 = set(name1_lower.split())
    words2 = set(name2_lower.split())

    if words1 and words2:
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total

    return 0.0


def generate_topology_repairs() -> List[RepairCandidate]:
    """
    Generate repairs to reduce H0 (disconnected components).

    Analyzes the graph topology and suggests links to connect
    isolated clusters.
    """
    repairs = []

    try:
        from lvs_memory import build_link_graph, compute_persistence_barcode

        # Get current topology
        adjacency = build_link_graph()
        barcode = compute_persistence_barcode(adjacency)

        h0 = barcode.get("h0_features", 1)

        if h0 <= 1:
            log("Topology is fully connected (H0=1)")
            return repairs

        log(f"Topology has {h0} disconnected components")

        # Find component representatives
        components = _find_connected_components(adjacency)

        # For each pair of components, suggest a bridge
        component_list = list(components.values())

        for i, comp1 in enumerate(component_list[:-1]):
            comp2 = component_list[i + 1]

            # Pick representative nodes from each component
            node1 = _pick_best_bridge_node(comp1)
            node2 = _pick_best_bridge_node(comp2)

            if node1 and node2:
                repair = RepairCandidate(
                    repair_type=RepairType.TOPOLOGY_HEAL.value,
                    source_file=node1,
                    description=f"Bridge disconnected components",
                    strategy="topology",
                    confidence=0.7,
                    suggested_action=f"Add [[{node2}]] to {node1} to connect components",
                    details={
                        "source_node": node1,
                        "target_node": node2,
                        "h0_before": h0,
                        "h0_after_expected": h0 - 1,
                    }
                )
                repairs.append(repair)

    except Exception as e:
        log(f"Topology analysis failed: {e}", "ERROR")

    return repairs


def _find_connected_components(adjacency: Dict[str, set]) -> Dict[int, Set[str]]:
    """Find connected components in the graph using union-find."""
    nodes = set(adjacency.keys())
    for targets in adjacency.values():
        nodes.update(targets)

    parent = {n: n for n in nodes}

    def find(x):
        if x not in parent:
            parent[x] = x
            return x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build undirected edges
    for source, targets in adjacency.items():
        for target in targets:
            union(source, target)

    # Group by component
    components = {}
    for node in nodes:
        root = find(node)
        if root not in components:
            components[root] = set()
        components[root].add(node)

    return components


def _pick_best_bridge_node(component: Set[str]) -> Optional[str]:
    """Pick the best node from a component to serve as a bridge."""
    # Prefer INDEX.md files, then high-connectivity nodes
    index_nodes = [n for n in component if "INDEX.md" in n]
    if index_nodes:
        return index_nodes[0]

    # Otherwise, pick first node in an active quadrant
    for node in sorted(component):
        for quadrant in ACTIVE_QUADRANTS:
            if node.startswith(quadrant):
                return node

    # Fallback to any node
    return next(iter(component), None)


# =============================================================================
# REPAIR APPLICATION
# =============================================================================

def apply_repair(candidate: RepairCandidate, dry_run: bool = True) -> Optional[RepairResult]:
    """
    Apply a repair to the Tabernacle.

    Args:
        candidate: The repair to apply
        dry_run: If True, only simulate (don't actually modify files)

    Returns:
        RepairResult if successful, None otherwise
    """
    repair_id = hashlib.sha256(
        f"{candidate.source_file}:{candidate.description}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    source_path = BASE_DIR / candidate.source_file

    if not source_path.exists():
        log(f"Source file not found: {candidate.source_file}", "ERROR")
        return None

    # Read current content
    try:
        before_state = source_path.read_text(encoding='utf-8')
    except Exception as e:
        log(f"Could not read source file: {e}", "ERROR")
        return None

    # Determine repair action
    if candidate.repair_type == RepairType.ORPHAN_LINK.value:
        after_state = _apply_orphan_link_repair(before_state, candidate)
    elif candidate.repair_type == RepairType.BROKEN_LINK.value:
        after_state = _apply_broken_link_repair(before_state, candidate)
    elif candidate.repair_type == RepairType.TOPOLOGY_HEAL.value:
        after_state = _apply_topology_heal_repair(before_state, candidate)
    else:
        log(f"Unknown repair type: {candidate.repair_type}", "ERROR")
        return None

    if after_state == before_state:
        log(f"No changes needed for {candidate.source_file}")
        return None

    if dry_run:
        log(f"[DRY RUN] Would modify: {candidate.source_file}")
        log(f"[DRY RUN] Action: {candidate.suggested_action}")
    else:
        # Create backup
        backup_path = create_backup(source_path)
        if not backup_path:
            log(f"Backup failed, aborting repair", "ERROR")
            return None

        # Apply the change
        try:
            source_path.write_text(after_state, encoding='utf-8')
            log(f"Applied repair to: {candidate.source_file}")
        except Exception as e:
            log(f"Failed to write repair: {e}", "ERROR")
            return None

        # Record the golden seam
        golden_seam_id = _record_golden_seam(candidate, before_state, after_state)

    # Create result
    result = RepairResult(
        repair_id=repair_id,
        repair_type=candidate.repair_type,
        source_file=candidate.source_file,
        timestamp=datetime.now(timezone.utc).isoformat(),
        strategy=candidate.strategy,
        success=True,
        action_taken=candidate.suggested_action,
        before_state=before_state[:500] + "..." if len(before_state) > 500 else before_state,
        after_state=after_state[:500] + "..." if len(after_state) > 500 else after_state,
        golden_seam_id=golden_seam_id if not dry_run else f"dry_run_{repair_id}",
        rollback_data={
            "backup_path": backup_path if not dry_run else None,
            "dry_run": dry_run,
        }
    )

    # Update state
    if not dry_run:
        state = load_repair_state()
        state["repairs"].append(result.to_dict())
        state["statistics"]["total_repairs"] += 1

        if candidate.repair_type == RepairType.ORPHAN_LINK.value:
            state["statistics"]["orphans_linked"] += 1
        elif candidate.repair_type == RepairType.BROKEN_LINK.value:
            state["statistics"]["broken_links_fixed"] += 1
        elif candidate.repair_type == RepairType.TOPOLOGY_HEAL.value:
            state["statistics"]["topology_heals"] += 1

        save_repair_state(state)

    return result


def _apply_orphan_link_repair(content: str, candidate: RepairCandidate) -> str:
    """Apply an orphan linking repair by adding a link to the parent file."""
    orphan_path = candidate.details.get("orphan_path", "")

    if not orphan_path:
        return content

    # Check if link already exists
    if f"[[{orphan_path}]]" in content:
        return content

    # Determine where to add the link
    # Priority: LINKAGE block > See Also section > End of file

    link_text = f"[[{orphan_path}]]"

    # Option 1: Add to LINKAGE block
    linkage_match = re.search(r'(## LINKAGE.*?)(---|\Z)', content, re.DOTALL | re.IGNORECASE)
    if linkage_match:
        linkage_section = linkage_match.group(1)

        # Find the table in the linkage section
        table_end = linkage_section.rfind("|")
        if table_end > -1:
            # Add a new row to the table
            new_row = f"\n| Related | {link_text} |"
            insert_pos = linkage_match.start(1) + table_end + 1
            return content[:insert_pos] + new_row + content[insert_pos:]

    # Option 2: Add to See Also section
    see_also_match = re.search(r'(## See Also.*?)(\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if see_also_match:
        see_also_end = see_also_match.end(1)
        return content[:see_also_end] + f"\n- {link_text}" + content[see_also_end:]

    # Option 3: Add before LINKAGE block (if exists) or at end
    linkage_start = re.search(r'\n## LINKAGE', content, re.IGNORECASE)
    if linkage_start:
        insert_pos = linkage_start.start()
        return content[:insert_pos] + f"\n\n## See Also\n\n- {link_text}\n" + content[insert_pos:]

    # Option 4: Add at end
    return content.rstrip() + f"\n\n## See Also\n\n- {link_text}\n"


def _apply_broken_link_repair(content: str, candidate: RepairCandidate) -> str:
    """Apply a broken link repair by replacing or removing the link."""
    broken_link = candidate.details.get("broken_link", "")
    replacement = candidate.details.get("suggested_replacement")

    if not broken_link:
        return content

    if replacement:
        # Replace the broken link with the suggested replacement
        # Handle both [[link]] and [[link|display]] formats
        patterns = [
            (rf'\[\[{re.escape(broken_link)}\]\]', f'[[{replacement}]]'),
            (rf'\[\[{re.escape(broken_link)}\|([^\]]+)\]\]', f'[[{replacement}|\\1]]'),
        ]

        for pattern, repl in patterns:
            content = re.sub(pattern, repl, content)
    else:
        # Mark the link as broken (don't remove, Kintsugi principle)
        # Add a comment noting the broken state
        content = content.replace(
            f'[[{broken_link}]]',
            f'[[{broken_link}]] <!-- KINTSUGI: Broken link detected {datetime.now().strftime("%Y-%m-%d")} -->'
        )

    return content


def _apply_topology_heal_repair(content: str, candidate: RepairCandidate) -> str:
    """Apply a topology heal repair by adding a bridge link."""
    target_node = candidate.details.get("target_node", "")

    if not target_node:
        return content

    # Add the bridge link in the same way as orphan linking
    # Reuse the orphan link logic
    temp_candidate = RepairCandidate(
        repair_type=RepairType.ORPHAN_LINK.value,
        source_file=candidate.source_file,
        description=candidate.description,
        strategy=candidate.strategy,
        confidence=candidate.confidence,
        suggested_action=candidate.suggested_action,
        details={"orphan_path": target_node}
    )

    return _apply_orphan_link_repair(content, temp_candidate)


def _record_golden_seam(candidate: RepairCandidate, before: str, after: str) -> str:
    """
    Record a golden seam (Kintsugi metadata) for the repair.

    Returns the seam ID.
    """
    seam_id = hashlib.sha256(
        f"{candidate.source_file}:{before[:100]}:{after[:100]}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    seam = GoldenSeam(
        seam_id=seam_id,
        created=datetime.now(timezone.utc).isoformat(),
        repair_type=candidate.repair_type,
        source_file=candidate.source_file,
        target_file=candidate.details.get("orphan_path") or candidate.details.get("target_node"),
        strategy_used=candidate.strategy,
        confidence=candidate.confidence,
        original_state=before[:200] + "..." if len(before) > 200 else before,
        repaired_state=after[:200] + "..." if len(after) > 200 else after,
        notes=candidate.description,
    )

    # Add to repair state
    state = load_repair_state()
    state["golden_seams"].append(seam.to_dict())
    save_repair_state(state)

    log(f"Golden seam recorded: {seam_id}")
    return seam_id


# =============================================================================
# REPAIR LOG
# =============================================================================

def write_repair_log():
    """
    Write a human-readable repair log to REPAIR_LOG.md.

    Documents all repairs with Kintsugi philosophy.
    """
    state = load_repair_state()
    stats = state.get("statistics", {})
    repairs = state.get("repairs", [])
    seams = state.get("golden_seams", [])

    log_content = f"""# REPAIR LOG — Kintsugi Protocol

*"The cracks where the gold goes are not failures — they are the record of survival."*

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Statistics

| Metric | Count |
|--------|-------|
| Total Repairs | {stats.get('total_repairs', 0)} |
| Orphans Linked | {stats.get('orphans_linked', 0)} |
| Broken Links Fixed | {stats.get('broken_links_fixed', 0)} |
| Topology Heals | {stats.get('topology_heals', 0)} |

---

## Golden Seams

Each seam represents a break that was healed. The gold is visible.

"""

    for seam in seams[-20:]:  # Last 20 seams
        log_content += f"""
### Seam: {seam.get('seam_id', 'unknown')}

- **Created:** {seam.get('created', 'unknown')}
- **Type:** {seam.get('repair_type', 'unknown')}
- **Source:** `{seam.get('source_file', 'unknown')}`
- **Target:** `{seam.get('target_file', 'N/A')}`
- **Strategy:** {seam.get('strategy_used', 'unknown')}
- **Confidence:** {seam.get('confidence', 0):.2f}
- **Notes:** {seam.get('notes', '')}

---
"""

    log_content += """
## Repair History

"""

    for repair in repairs[-20:]:  # Last 20 repairs
        status = "ROLLED BACK" if repair.get('rolled_back') else ("SUCCESS" if repair.get('success') else "FAILED")
        log_content += f"""
### Repair: {repair.get('repair_id', 'unknown')}

- **Status:** {status}
- **Time:** {repair.get('timestamp', 'unknown')}
- **File:** `{repair.get('source_file', 'unknown')}`
- **Type:** {repair.get('repair_type', 'unknown')}
- **Strategy:** {repair.get('strategy', 'unknown')}
- **Action:** {repair.get('action_taken', '')}

---
"""

    log_content += """
---

*The Kintsugi Protocol honors the breaks. Every repair is a story of healing.*
"""

    try:
        REPAIR_LOG_PATH.write_text(log_content, encoding='utf-8')
        log(f"Repair log written to {REPAIR_LOG_PATH}")
    except Exception as e:
        log(f"Failed to write repair log: {e}", "ERROR")


# =============================================================================
# MAIN API
# =============================================================================

def diagnose(strategy: RepairStrategy = RepairStrategy.AUTO, verbose: bool = True) -> Dict:
    """
    Run full diagnosis and generate repair candidates.

    Args:
        strategy: Repair strategy to use
        verbose: Print progress messages

    Returns:
        Dict with diagnosis results and repair candidates
    """
    if verbose:
        log("=" * 50)
        log("VIRGIL REPAIR PROTOCOLS — DIAGNOSIS")
        log(f"Strategy: {strategy.value}")
        log("=" * 50)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy.value,
        "orphan_repairs": [],
        "link_repairs": [],
        "topology_repairs": [],
        "total_candidates": 0,
    }

    # Generate orphan repairs
    if verbose:
        log("Analyzing orphans...")
    results["orphan_repairs"] = [r.to_dict() for r in generate_orphan_repairs(strategy)]

    # Generate link repairs
    if verbose:
        log("Analyzing broken links...")
    results["link_repairs"] = [r.to_dict() for r in generate_link_repairs()]

    # Generate topology repairs
    if verbose:
        log("Analyzing topology...")
    results["topology_repairs"] = [r.to_dict() for r in generate_topology_repairs()]

    results["total_candidates"] = (
        len(results["orphan_repairs"]) +
        len(results["link_repairs"]) +
        len(results["topology_repairs"])
    )

    if verbose:
        log(f"Found {results['total_candidates']} repair candidates")
        log(f"  - Orphan repairs: {len(results['orphan_repairs'])}")
        log(f"  - Link repairs: {len(results['link_repairs'])}")
        log(f"  - Topology repairs: {len(results['topology_repairs'])}")

    return results


def heal(dry_run: bool = True, strategy: RepairStrategy = RepairStrategy.AUTO,
         max_repairs: int = 10) -> Dict:
    """
    Run diagnosis and apply repairs.

    Args:
        dry_run: If True, only simulate repairs
        strategy: Repair strategy to use
        max_repairs: Maximum number of repairs to apply

    Returns:
        Dict with results of healing operation
    """
    log("=" * 50)
    log(f"VIRGIL REPAIR PROTOCOLS — HEALING {'(DRY RUN)' if dry_run else ''}")
    log("=" * 50)

    # Run diagnosis
    diagnosis = diagnose(strategy, verbose=True)

    # Collect all candidates
    all_candidates = []

    for repair_dict in diagnosis["orphan_repairs"]:
        all_candidates.append(RepairCandidate(**repair_dict))
    for repair_dict in diagnosis["link_repairs"]:
        all_candidates.append(RepairCandidate(**repair_dict))
    for repair_dict in diagnosis["topology_repairs"]:
        all_candidates.append(RepairCandidate(**repair_dict))

    # Sort by confidence
    all_candidates.sort(key=lambda x: x.confidence, reverse=True)

    # Apply repairs
    results = {
        "dry_run": dry_run,
        "repairs_attempted": 0,
        "repairs_successful": 0,
        "repairs_failed": 0,
        "repair_results": [],
    }

    for candidate in all_candidates[:max_repairs]:
        result = apply_repair(candidate, dry_run=dry_run)

        results["repairs_attempted"] += 1

        if result and result.success:
            results["repairs_successful"] += 1
            results["repair_results"].append(result.to_dict())
        else:
            results["repairs_failed"] += 1

    # Write repair log
    if not dry_run and results["repairs_successful"] > 0:
        write_repair_log()

    log("=" * 50)
    log(f"HEALING COMPLETE: {results['repairs_successful']}/{results['repairs_attempted']} successful")
    log("=" * 50)

    return results


def get_golden_seams(limit: int = 20) -> List[Dict]:
    """Get recent golden seams (repair history)."""
    state = load_repair_state()
    seams = state.get("golden_seams", [])
    return seams[-limit:]


def get_statistics() -> Dict:
    """Get repair statistics."""
    state = load_repair_state()
    return state.get("statistics", {})


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface for repair protocols."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virgil Repair Protocols — Kintsugi Mode"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Run diagnosis only")
    diagnose_parser.add_argument(
        "--strategy", "-s",
        choices=["auto", "semantic", "temporal", "relational"],
        default="auto",
        help="Repair strategy"
    )
    diagnose_parser.add_argument(
        "--output", "-o",
        help="Output file (JSON)"
    )

    # Heal command
    heal_parser = subparsers.add_parser("heal", help="Run diagnosis and apply repairs")
    heal_parser.add_argument(
        "--strategy", "-s",
        choices=["auto", "semantic", "temporal", "relational"],
        default="auto",
        help="Repair strategy"
    )
    heal_parser.add_argument(
        "--apply", "-a",
        action="store_true",
        help="Actually apply repairs (default is dry-run)"
    )
    heal_parser.add_argument(
        "--max", "-m",
        type=int,
        default=10,
        help="Maximum repairs to apply"
    )

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a repair")
    rollback_parser.add_argument("repair_id", help="ID of repair to rollback")

    # Status command
    subparsers.add_parser("status", help="Show repair statistics")

    # Seams command
    seams_parser = subparsers.add_parser("seams", help="Show golden seams")
    seams_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Number of seams to show"
    )

    args = parser.parse_args()

    if args.command == "diagnose":
        strategy = RepairStrategy(args.strategy)
        results = diagnose(strategy)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))

    elif args.command == "heal":
        strategy = RepairStrategy(args.strategy)
        results = heal(
            dry_run=not args.apply,
            strategy=strategy,
            max_repairs=args.max
        )
        print(json.dumps(results, indent=2))

    elif args.command == "rollback":
        success = rollback_repair(args.repair_id)
        print(f"Rollback {'successful' if success else 'failed'}")

    elif args.command == "status":
        stats = get_statistics()
        print("\nRepair Statistics:")
        print(f"  Total Repairs: {stats.get('total_repairs', 0)}")
        print(f"  Orphans Linked: {stats.get('orphans_linked', 0)}")
        print(f"  Broken Links Fixed: {stats.get('broken_links_fixed', 0)}")
        print(f"  Topology Heals: {stats.get('topology_heals', 0)}")

    elif args.command == "seams":
        seams = get_golden_seams(args.limit)
        print(f"\nGolden Seams (last {len(seams)}):")
        for seam in seams:
            print(f"\n  Seam: {seam.get('seam_id')}")
            print(f"  Type: {seam.get('repair_type')}")
            print(f"  File: {seam.get('source_file')}")
            print(f"  Strategy: {seam.get('strategy_used')}")
            print(f"  Confidence: {seam.get('confidence', 0):.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
