"""
HoloTower Prism — Semantic Diff Engine

Computes meaningful differences between HoloSnapshots:
- Manifest tree changes (files added/removed/modified)
- Runtime vital transitions (alive/dead/drift)
- Topology mutations (nodes, edges, density)
- Coherence shifts (p-value changes)

Wave 3 — The Prism refracts state into change.
"""

from typing import Dict, List, Any, Optional

from deepdiff import DeepDiff

from .models import HoloSnapshot, RuntimeVital, TopologyMeta


# =============================================================================
# MANIFEST DIFFING
# =============================================================================

def diff_manifests(old: Dict, new: Dict) -> Dict[str, Any]:
    """
    Compare two manifest trees using DeepDiff.
    
    Args:
        old: ManifestTree dict (from old snapshot)
        new: ManifestTree dict (from new snapshot)
        
    Returns:
        Dict with:
            - added: List of new file paths
            - removed: List of removed file paths
            - modified: List of files with changed content_hash
            - root_hash_changed: bool
    """
    if not old and not new:
        return {"added": [], "removed": [], "modified": [], "root_hash_changed": False}
    
    if not old:
        # Everything is new
        new_files = new.get("files", [])
        return {
            "added": [f.get("path", "") for f in new_files],
            "removed": [],
            "modified": [],
            "root_hash_changed": True
        }
    
    if not new:
        # Everything was removed
        old_files = old.get("files", [])
        return {
            "added": [],
            "removed": [f.get("path", "") for f in old_files],
            "modified": [],
            "root_hash_changed": True
        }
    
    # Build lookup by path
    old_by_path = {f.get("path"): f for f in old.get("files", [])}
    new_by_path = {f.get("path"): f for f in new.get("files", [])}
    
    old_paths = set(old_by_path.keys())
    new_paths = set(new_by_path.keys())
    
    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)
    
    # Find modified (same path, different hash)
    modified = []
    for path in old_paths & new_paths:
        old_hash = old_by_path[path].get("content_hash", "")
        new_hash = new_by_path[path].get("content_hash", "")
        if old_hash != new_hash:
            modified.append(path)
    modified.sort()
    
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "root_hash_changed": old.get("root_hash") != new.get("root_hash")
    }


# =============================================================================
# VITAL DIFFING
# =============================================================================

def diff_vitals(old: List[RuntimeVital], new: List[RuntimeVital]) -> Dict[str, Any]:
    """
    Set operations on runtime vitals.
    
    Args:
        old: List of RuntimeVital from old snapshot
        new: List of RuntimeVital from new snapshot
        
    Returns:
        Dict with:
            - newly_alive: List of (holon_id, old_status) tuples
            - newly_dead: List of (holon_id, old_status) tuples
            - drift_changes: List of (holon_id, old_drift, new_drift) tuples
            - new_holons: List of holon_ids not in old
            - missing_holons: List of holon_ids not in new
    """
    old_map = {v.holon_id: v for v in old}
    new_map = {v.holon_id: v for v in new}
    
    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())
    
    newly_alive = []
    newly_dead = []
    drift_changes = []
    
    # Check common holons for status changes
    for holon_id in old_ids & new_ids:
        old_vital = old_map[holon_id]
        new_vital = new_map[holon_id]
        
        # Status transitions
        if old_vital.status != new_vital.status:
            if new_vital.status == "ALIVE":
                newly_alive.append((holon_id, old_vital.status))
            elif new_vital.status == "DEAD":
                newly_dead.append((holon_id, old_vital.status))
        
        # Drift changes
        if old_vital.drift != new_vital.drift:
            drift_changes.append((holon_id, old_vital.drift, new_vital.drift))
    
    return {
        "newly_alive": newly_alive,
        "newly_dead": newly_dead,
        "drift_changes": drift_changes,
        "new_holons": sorted(new_ids - old_ids),
        "missing_holons": sorted(old_ids - new_ids)
    }


# =============================================================================
# TOPOLOGY DIFFING
# =============================================================================

def diff_topology(old: Optional[TopologyMeta], new: Optional[TopologyMeta]) -> Dict[str, Any]:
    """
    Compare graph topology metadata.
    
    Args:
        old: TopologyMeta from old snapshot (can be None)
        new: TopologyMeta from new snapshot (can be None)
        
    Returns:
        Dict with:
            - node_delta: new.node_count - old.node_count
            - edge_delta: new.edge_count - old.edge_count
            - density_change: new.density - old.density
            - orphan_delta: new.orphans - old.orphans
            - hash_changed: True if content_hash differs
    """
    if old is None and new is None:
        return {
            "node_delta": 0,
            "edge_delta": 0,
            "density_change": 0.0,
            "orphan_delta": 0,
            "hash_changed": False,
            "old_exists": False,
            "new_exists": False
        }
    
    if old is None:
        return {
            "node_delta": new.node_count,
            "edge_delta": new.edge_count,
            "density_change": new.density,
            "orphan_delta": new.orphans,
            "hash_changed": True,
            "old_exists": False,
            "new_exists": True
        }
    
    if new is None:
        return {
            "node_delta": -old.node_count,
            "edge_delta": -old.edge_count,
            "density_change": -old.density,
            "orphan_delta": -old.orphans,
            "hash_changed": True,
            "old_exists": True,
            "new_exists": False
        }
    
    return {
        "node_delta": new.node_count - old.node_count,
        "edge_delta": new.edge_count - old.edge_count,
        "density_change": new.density - old.density,
        "orphan_delta": new.orphans - old.orphans,
        "hash_changed": old.content_hash != new.content_hash,
        "old_exists": True,
        "new_exists": True
    }


# =============================================================================
# FULL SNAPSHOT DIFFING
# =============================================================================

def diff_snapshots(old: HoloSnapshot, new: HoloSnapshot) -> Dict[str, Any]:
    """
    Full semantic diff between two HoloSnapshots.
    
    Args:
        old: Previous snapshot
        new: Current snapshot
        
    Returns:
        Comprehensive diff report with sections:
            - meta: Basic info about both snapshots
            - manifests: File-level changes
            - vitals: Runtime state changes
            - topology: Graph structure changes
            - coherence: p-value and void changes
    """
    # Convert manifest_root to comparable dicts
    # Note: We don't have the full manifest tree in the snapshot,
    # just the root hash. For detailed manifest diff, we'd need to
    # load from blob storage. For now, compare root hashes.
    manifest_diff = {
        "added": [],
        "removed": [],
        "modified": [],
        "root_hash_changed": old.manifest_root != new.manifest_root
    }
    
    return {
        "meta": {
            "old_id": old.id,
            "new_id": new.id,
            "old_timestamp": old.timestamp.isoformat(),
            "new_timestamp": new.timestamp.isoformat(),
            "old_message": old.message,
            "new_message": new.message,
            "old_git_ref": old.git_ref,
            "new_git_ref": new.git_ref
        },
        "manifests": manifest_diff,
        "vitals": diff_vitals(old.runtime_state, new.runtime_state),
        "topology": diff_topology(old.topology, new.topology),
        "coherence": {
            "old_p": old.system_p_value,
            "new_p": new.system_p_value,
            "p_delta": new.system_p_value - old.system_p_value,
            "old_voids": old.void_count,
            "new_voids": new.void_count,
            "void_delta": new.void_count - old.void_count
        }
    }


# =============================================================================
# RICH FORMATTING
# =============================================================================

def format_diff_report(diff: Dict[str, Any]) -> str:
    """
    Format diff as Rich-compatible string for CLI output.
    
    Colors:
        - green: additions, improvements
        - red: removals, degradations
        - yellow: modifications, changes
        - cyan: neutral info
        
    Args:
        diff: Output from diff_snapshots()
        
    Returns:
        Rich-formatted string ready for console.print()
    """
    lines = []
    meta = diff.get("meta", {})
    
    # Header
    old_short = meta.get("old_id", "")[:8]
    new_short = meta.get("new_id", "")[:8]
    lines.append(f"[bold cyan]⚖️  SHIFT REPORT[/bold cyan] ({old_short} → {new_short})")
    lines.append("─" * 40)
    
    # === MANIFESTS ===
    manifests = diff.get("manifests", {})
    has_manifest_changes = (
        manifests.get("added") or 
        manifests.get("removed") or 
        manifests.get("modified") or
        manifests.get("root_hash_changed")
    )
    
    if has_manifest_changes:
        lines.append("\n[bold][MANIFESTS][/bold]")
        
        for path in manifests.get("added", []):
            lines.append(f"  [green]+[/green] {path} [dim](added)[/dim]")
        
        for path in manifests.get("removed", []):
            lines.append(f"  [red]-[/red] {path} [dim](removed)[/dim]")
        
        for path in manifests.get("modified", []):
            lines.append(f"  [yellow]~[/yellow] {path} [dim](modified)[/dim]")
        
        if manifests.get("root_hash_changed") and not (
            manifests.get("added") or manifests.get("removed") or manifests.get("modified")
        ):
            lines.append("  [yellow]~[/yellow] [dim]Root hash changed[/dim]")
    
    # === VITALS ===
    vitals = diff.get("vitals", {})
    has_vital_changes = (
        vitals.get("newly_alive") or
        vitals.get("newly_dead") or
        vitals.get("drift_changes") or
        vitals.get("new_holons") or
        vitals.get("missing_holons")
    )
    
    if has_vital_changes:
        lines.append("\n[bold][VITALS][/bold]")
        
        for holon_id, old_status in vitals.get("newly_alive", []):
            lines.append(f"  [green]📈[/green] {holon_id}: {old_status} → ALIVE")
        
        for holon_id, old_status in vitals.get("newly_dead", []):
            lines.append(f"  [red]📉[/red] {holon_id}: {old_status} → DEAD")
        
        for holon_id, old_drift, new_drift in vitals.get("drift_changes", []):
            drift_str = "now drifted" if new_drift else "no longer drifted"
            color = "yellow" if new_drift else "green"
            lines.append(f"  [{color}]⚡[/{color}] {holon_id}: {drift_str}")
        
        if vitals.get("new_holons"):
            for holon_id in vitals["new_holons"][:5]:
                lines.append(f"  [blue]+[/blue] {holon_id} [dim](new)[/dim]")
            if len(vitals["new_holons"]) > 5:
                lines.append(f"  [dim]... and {len(vitals['new_holons']) - 5} more[/dim]")
        
        if vitals.get("missing_holons"):
            for holon_id in vitals["missing_holons"][:5]:
                lines.append(f"  [red]-[/red] {holon_id} [dim](missing)[/dim]")
            if len(vitals["missing_holons"]) > 5:
                lines.append(f"  [dim]... and {len(vitals['missing_holons']) - 5} more[/dim]")
    
    # === TOPOLOGY ===
    topology = diff.get("topology", {})
    has_topo_changes = (
        topology.get("node_delta", 0) != 0 or
        topology.get("edge_delta", 0) != 0 or
        topology.get("hash_changed")
    )
    
    if has_topo_changes:
        lines.append("\n[bold][TOPOLOGY][/bold]")
        
        node_d = topology.get("node_delta", 0)
        edge_d = topology.get("edge_delta", 0)
        
        parts = []
        if node_d != 0:
            sign = "+" if node_d > 0 else ""
            color = "green" if node_d > 0 else "red"
            parts.append(f"[{color}]{sign}{node_d} nodes[/{color}]")
        
        if edge_d != 0:
            sign = "+" if edge_d > 0 else ""
            color = "green" if edge_d > 0 else "red"
            parts.append(f"[{color}]{sign}{edge_d} edges[/{color}]")
        
        if parts:
            lines.append(f"  🕸️  Graph: {', '.join(parts)}")
        
        orphan_d = topology.get("orphan_delta", 0)
        if orphan_d != 0:
            sign = "+" if orphan_d > 0 else ""
            color = "red" if orphan_d > 0 else "green"
            lines.append(f"  [{color}]{sign}{orphan_d} orphans[/{color}]")
    
    # === COHERENCE ===
    coherence = diff.get("coherence", {})
    p_delta = coherence.get("p_delta", 0)
    void_delta = coherence.get("void_delta", 0)
    
    if p_delta != 0 or void_delta != 0:
        lines.append("\n[bold][COHERENCE][/bold]")
        
        if p_delta != 0:
            old_p = coherence.get("old_p", 0)
            new_p = coherence.get("new_p", 0)
            sign = "▲ +" if p_delta > 0 else "▼ "
            color = "green" if p_delta > 0 else "red"
            lines.append(f"  📊 p-value: {old_p:.2f} → {new_p:.2f} ([{color}]{sign}{p_delta:.2f}[/{color}])")
        
        if void_delta != 0:
            old_v = coherence.get("old_voids", 0)
            new_v = coherence.get("new_voids", 0)
            sign = "+" if void_delta > 0 else ""
            color = "red" if void_delta > 0 else "green"
            lines.append(f"  🕳️  Voids: {old_v} → {new_v} ([{color}]{sign}{void_delta}[/{color}])")
    
    # === SUMMARY ===
    total_changes = (
        len(manifests.get("added", [])) +
        len(manifests.get("removed", [])) +
        len(manifests.get("modified", [])) +
        len(vitals.get("newly_alive", [])) +
        len(vitals.get("newly_dead", [])) +
        len(vitals.get("drift_changes", [])) +
        (1 if topology.get("hash_changed") else 0) +
        (1 if p_delta != 0 else 0)
    )
    
    if total_changes == 0:
        lines.append("\n[dim]No significant changes detected.[/dim]")
    
    return "\n".join(lines)
