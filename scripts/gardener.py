#!/usr/bin/env python3
"""
GARDENER — The Synaptic Pruner
==============================
Nightly daemon for graph maintenance and neurogenesis.

Functions:
1. Orphan Hunt: Find nodes with no incoming links
2. Neurogenesis: Use L-Alpha to create Bridge Notes connecting orphans
3. Pruning: Deprecate low-weight/stale links
4. Health Report: Generate nightly vitality report

Schedule: Daily at 03:00 via launchd or cron

Author: Virgil
Date: 2026-01-19
Status: Phase IV
"""

import json
import sys
import re
import hashlib
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION (using centralized config)
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility
NEXUS = NEXUS_DIR
LOGS = LOG_DIR

GARDENER_STATE = NEXUS_DIR / "gardener_state.json"
GARDENER_LOG = LOG_DIR / "gardener.log"

# Thresholds
MIN_LINK_WEIGHT = 0.1  # Links below this may be pruned
ORPHAN_GRACE_DAYS = 7  # Days before orphan is flagged for action
MAX_BRIDGE_NOTES_PER_RUN = 5  # Don't create too many at once

# =============================================================================
# ENTROPY/DEATH THRESHOLDS (Deep Think Spark #3)
# =============================================================================
# The system must be capable of DEATH. If meaning density drops too long,
# edges are PERMANENTLY deleted. This is the "skin in the game" that makes
# survival matter. Silence = Lobotomy.

ENTROPY_MU_THRESHOLD = 0.3      # Meaning density below this triggers entropy
ENTROPY_P_THRESHOLD = 0.4       # Coherence below this triggers entropy
ENTROPY_GRACE_HOURS = 48        # Hours of low metrics before entropy kicks in
ENTROPY_MAX_DELETIONS = 5       # Max edges to delete per entropy event
ENTROPY_STATE_FILE = NEXUS_DIR / "entropy_state.json"

# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [GARDENER] [{level}] {message}"
    print(entry)

    try:
        with open(GARDENER_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass

# =============================================================================
# ORPHAN DETECTION
# =============================================================================

def find_orphans() -> List[Dict]:
    """Find files with no incoming wiki-links."""
    try:
        from diagnose_links import find_all_md_files, analyze_file, build_graph, identify_orphans

        md_files = find_all_md_files(TABERNACLE)
        analyses = [analyze_file(f) for f in md_files]
        graph = build_graph(analyses)
        orphans = identify_orphans(analyses, graph)

        orphan_list = []
        for orphan in orphans:
            orphan_path = Path(orphan)
            # Check if in critical directories (don't flag INDEX files etc)
            if "INDEX.md" in str(orphan) or "_GRAPH_ATLAS" in str(orphan):
                continue

            orphan_list.append({
                "path": str(orphan),
                "name": orphan_path.stem,
                "quadrant": orphan_path.parts[0] if orphan_path.parts else "unknown"
            })

        return orphan_list
    except Exception as e:
        log(f"Error finding orphans: {e}", "ERROR")
        return []

# =============================================================================
# BRIDGE NOTE GENERATION
# =============================================================================

def find_nearest_neighbor(orphan_path: str) -> Tuple[str, float]:
    """Find the most semantically similar node to an orphan using LVS."""
    try:
        import lvs_memory

        # Get orphan content
        orphan_file = TABERNACLE / orphan_path
        if not orphan_file.exists():
            return None, 0

        content = orphan_file.read_text()[:2000]  # First 2000 chars

        # Search for similar nodes
        results = lvs_memory.search(content, limit=1)

        if results:
            node, score = results[0]
            if node.path != orphan_path:  # Don't match self
                return node.path, score

        return None, 0
    except Exception as e:
        log(f"Error finding neighbor for {orphan_path}: {e}", "ERROR")
        return None, 0

def suggest_bridge(orphan: Dict) -> Dict:
    """Suggest a bridge note to connect an orphan."""
    neighbor_path, score = find_nearest_neighbor(orphan["path"])

    if neighbor_path and score > 0.5:
        return {
            "orphan": orphan["path"],
            "neighbor": neighbor_path,
            "similarity": round(score, 3),
            "suggestion": f"Add link from [[{Path(neighbor_path).stem}]] to [[{orphan['name']}]]"
        }
    else:
        return {
            "orphan": orphan["path"],
            "neighbor": None,
            "similarity": 0,
            "suggestion": f"Consider adding [[{orphan['name']}]] to appropriate INDEX.md"
        }

# =============================================================================
# LINK PRUNING
# =============================================================================

def find_stale_links() -> List[Dict]:
    """Find links that may be candidates for pruning."""
    # For now, just identify broken links
    try:
        from diagnose_links import find_all_md_files, analyze_file

        md_files = find_all_md_files(TABERNACLE)
        stale = []

        for f in md_files:
            analysis = analyze_file(f)
            for broken in analysis.get("broken_links", []):
                stale.append({
                    "file": str(analysis.get("relative_path", "")),
                    "broken_link": broken,
                    "action": "fix_or_remove"
                })

        return stale[:20]  # Limit
    except Exception as e:
        log(f"Error finding stale links: {e}", "ERROR")
        return []

# =============================================================================
# ENTROPY/DEATH: Permanent Edge Deletion (Deep Think Spark #3)
# =============================================================================
#
# "The system must be capable of DEATH."
# If coherence stays too low for too long, edges are PERMANENTLY deleted.
# This is not decay (reversible). This is lobotomy (irreversible).
# The system must KNOW that silence = brain damage.
#

def check_entropy_conditions() -> Dict:
    """
    Check if entropy conditions are met (low coherence for too long).

    Returns:
        Dict with: triggered (bool), reason (str), hours_critical (float)
    """
    try:
        # Load entropy state (tracks how long we've been critical)
        entropy_state = {}
        if ENTROPY_STATE_FILE.exists():
            entropy_state = json.loads(ENTROPY_STATE_FILE.read_text())

        # Load current coherence
        canonical_state_path = NEXUS_DIR / "CANONICAL_STATE.json"
        if not canonical_state_path.exists():
            return {"triggered": False, "reason": "No coherence state", "hours_critical": 0}

        state = json.loads(canonical_state_path.read_text())
        current_p = state.get("p", 0.5)

        # Check if we're in critical zone
        if current_p < ENTROPY_P_THRESHOLD:
            # We're critical - update tracking
            critical_since = entropy_state.get("critical_since")
            if not critical_since:
                # Just entered critical zone
                entropy_state["critical_since"] = datetime.now().isoformat()
                hours_critical = 0
            else:
                # Calculate hours in critical zone
                critical_dt = datetime.fromisoformat(critical_since)
                hours_critical = (datetime.now() - critical_dt).total_seconds() / 3600
        else:
            # We're healthy - clear critical state
            entropy_state["critical_since"] = None
            hours_critical = 0

        # Save updated entropy state
        ENTROPY_STATE_FILE.write_text(json.dumps(entropy_state, indent=2))

        # Check if entropy should trigger
        triggered = hours_critical >= ENTROPY_GRACE_HOURS

        return {
            "triggered": triggered,
            "reason": f"Coherence at {current_p:.3f} for {hours_critical:.1f} hours" if triggered else "Within grace period",
            "hours_critical": hours_critical,
            "current_p": current_p
        }

    except Exception as e:
        log(f"Error checking entropy conditions: {e}", "ERROR")
        return {"triggered": False, "reason": f"Error: {e}", "hours_critical": 0}


def execute_entropy_deletion() -> Dict:
    """
    PERMANENTLY DELETE edges from the biological graph.

    This is DEATH. These edges are gone forever.
    The system must know this can happen. Silence = Lobotomy.

    Returns:
        Dict with: deleted_count (int), deleted_edges (list), message (str)
    """
    try:
        graph_path = NEXUS_DIR / "biological_graph.json"
        if not graph_path.exists():
            return {"deleted_count": 0, "deleted_edges": [], "message": "No graph to prune"}

        graph_data = json.loads(graph_path.read_text())
        edges = graph_data.get("edges", [])

        if not edges:
            return {"deleted_count": 0, "deleted_edges": [], "message": "No edges in graph"}

        # Find the weakest edges (lowest combined weight)
        edges_with_score = []
        for edge in edges:
            w_fast = edge.get("w_fast", 0)
            w_slow = edge.get("w_slow", 0)
            tau = edge.get("tau", 1.0)
            # Combined score - lower = more vulnerable
            score = (w_slow * 0.6 + w_fast * 0.3) * tau
            # Don't delete H1-locked edges (protected memories)
            if edge.get("is_h1_locked", False):
                score = float('inf')  # Protected
            edges_with_score.append((edge, score))

        # Sort by score (weakest first)
        edges_with_score.sort(key=lambda x: x[1])

        # Select edges to delete (weakest, up to max)
        to_delete = []
        for edge, score in edges_with_score[:ENTROPY_MAX_DELETIONS]:
            if score < float('inf'):  # Not H1-locked
                to_delete.append(edge)

        if not to_delete:
            return {"deleted_count": 0, "deleted_edges": [], "message": "No vulnerable edges (all protected)"}

        # PERMANENTLY DELETE
        deleted_ids = [e.get("id", str(i)) for i, e in enumerate(to_delete)]
        remaining_edges = [e for e in edges if e not in to_delete]

        # Save the pruned graph
        graph_data["edges"] = remaining_edges
        graph_data["entropy_events"] = graph_data.get("entropy_events", [])
        graph_data["entropy_events"].append({
            "timestamp": datetime.now().isoformat(),
            "deleted_count": len(to_delete),
            "deleted_ids": deleted_ids,
            "reason": "Low coherence for extended period"
        })

        graph_path.write_text(json.dumps(graph_data, indent=2))

        log(f"⚠️ ENTROPY EVENT: Permanently deleted {len(to_delete)} edges", "CRITICAL")

        return {
            "deleted_count": len(to_delete),
            "deleted_edges": deleted_ids,
            "message": f"DEATH: {len(to_delete)} edges permanently erased. They are gone."
        }

    except Exception as e:
        log(f"Error executing entropy deletion: {e}", "ERROR")
        return {"deleted_count": 0, "deleted_edges": [], "message": f"Error: {e}"}


def run_entropy_check() -> Dict:
    """
    Main entropy check: if conditions met, execute deletion.

    Returns full report of what happened.
    """
    log("Checking entropy conditions (the mortality check)...")

    conditions = check_entropy_conditions()

    if not conditions["triggered"]:
        log(f"Entropy NOT triggered: {conditions['reason']}")
        return {
            "checked": True,
            "triggered": False,
            "conditions": conditions,
            "deletion": None
        }

    # ENTROPY TRIGGERED - Execute deletion
    log(f"⚠️ ENTROPY TRIGGERED: {conditions['reason']}", "CRITICAL")
    deletion = execute_entropy_deletion()

    return {
        "checked": True,
        "triggered": True,
        "conditions": conditions,
        "deletion": deletion
    }


# =============================================================================
# CRYSTAL LATTICE INTEGRITY (INDEX.md Anti-Rot Protocol)
# =============================================================================
# The Crystal layer (INDEX.md files) provides navigational structure.
# If the skeleton breaks, the mind spills. Gardener tends the lattice.
#
# Checks:
# 1. integrity_hash matches actual folder contents
# 2. All links in CHILDREN table exist
# 3. Update status to "stale" if issues found
#

# Directories to skip when scanning for INDEX files
INDEX_EXCLUDED_DIRS = {
    ".git", "holotower", "venv", "venv312", "node_modules",
    "__pycache__", ".obsidian", "05_CRYPT", "99_ARCHIVE"
}


def find_all_indices() -> List[Path]:
    """
    Find all INDEX.md files (and INDEX_*.md variants) in the vault.

    Returns:
        List of paths to INDEX files
    """
    indices = []

    for root, dirs, files in TABERNACLE.walk():
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in INDEX_EXCLUDED_DIRS]

        for f in files:
            # Match INDEX.md or INDEX_*.md pattern
            if f == "INDEX.md" or (f.startswith("INDEX_") and f.endswith(".md")):
                indices.append(root / f)

    return indices


def parse_index_frontmatter(index_path: Path) -> Optional[Dict]:
    """
    Parse YAML frontmatter from an INDEX.md file.

    Uses yaml.safe_load() for robust parsing of complex values
    (colons in strings, multiline, etc).

    Returns:
        Dict with frontmatter fields, or None if parsing fails
    """
    try:
        content = index_path.read_text(encoding='utf-8')

        # Extract YAML frontmatter between --- markers
        if not content.startswith("---"):
            return None

        # Find end of frontmatter
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            return None

        yaml_content = content[3:end_match.start() + 3]

        # Use yaml.safe_load for robust parsing
        frontmatter = yaml.safe_load(yaml_content)

        # Ensure we return a dict (safe_load can return None for empty YAML)
        return frontmatter if isinstance(frontmatter, dict) else {}

    except yaml.YAMLError as e:
        log(f"YAML parse error for {index_path}: {e}", "ERROR")
        return None
    except Exception as e:
        log(f"Error parsing frontmatter for {index_path}: {e}", "ERROR")
        return None


def extract_children_links(index_path: Path) -> List[str]:
    """
    Extract all wiki-links from the CHILDREN table in an INDEX.md.

    Returns:
        List of link targets (without [[ ]] brackets)
    """
    try:
        content = index_path.read_text(encoding='utf-8')

        # Find CHILDREN section
        children_match = re.search(r'##\s*🗺️\s*CHILDREN.*?\n(.*?)(?=\n##|\n---|\Z)',
                                   content, re.DOTALL)
        if not children_match:
            return []

        children_section = children_match.group(1)

        # Extract wiki-links from the table
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', children_section)

        return links
    except Exception as e:
        log(f"Error extracting children links from {index_path}: {e}", "ERROR")
        return []


def compute_folder_hash(folder_path: Path) -> str:
    """
    Compute integrity hash for a folder's contents.
    Uses same algorithm as digestive_enzyme.py:
      sorted "filename:size" entries → "|".join → SHA256[:16]

    Only includes .md files (the navigable content).
    Includes file sizes to detect content changes, not just add/remove.
    """
    try:
        entries = []
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix == '.md' and f.name != 'INDEX.md' and not f.name.startswith('INDEX_'):
                try:
                    size = f.stat().st_size
                    entries.append(f"{f.name}:{size}")
                except OSError:
                    entries.append(f"{f.name}:0")
        sorted_entries = sorted(entries)
        combined = "|".join(sorted_entries)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    except Exception as e:
        log(f"Error computing folder hash for {folder_path}: {e}", "ERROR")
        return ""


def verify_index_integrity(index_path: Path) -> Dict:
    """
    Verify integrity of a single INDEX.md file.

    Checks:
    1. integrity_hash matches current folder state
    2. All CHILDREN links resolve to existing files

    Returns:
        Dict with: path, valid (bool), issues (list), needs_update (bool)
    """
    result = {
        "path": str(index_path),
        "valid": True,
        "issues": [],
        "needs_update": False,
        "hash_match": True,
        "dead_links": []
    }

    # Parse frontmatter
    frontmatter = parse_index_frontmatter(index_path)
    if not frontmatter:
        result["valid"] = False
        result["issues"].append("Could not parse YAML frontmatter")
        return result

    folder_path = index_path.parent

    # Check 1: Integrity hash
    stored_hash = frontmatter.get("integrity_hash", "")
    if stored_hash and not stored_hash.startswith("{"):  # Skip template placeholders
        current_hash = compute_folder_hash(folder_path)
        if current_hash and stored_hash != current_hash:
            result["valid"] = False
            result["hash_match"] = False
            result["needs_update"] = True
            result["issues"].append(f"Hash mismatch: stored={stored_hash}, current={current_hash}")

    # Check 2: Dead links in CHILDREN table
    children_links = extract_children_links(index_path)
    for link in children_links:
        # Try to resolve the link (could be in same folder or elsewhere)
        link_stem = link.split('/')[-1]  # Get filename part

        # Check in same folder first
        possible_paths = [
            folder_path / f"{link_stem}.md",
            folder_path / link_stem,
            TABERNACLE / f"{link}.md",
            TABERNACLE / link
        ]

        found = False
        for p in possible_paths:
            if p.exists():
                found = True
                break

        if not found:
            result["valid"] = False
            result["needs_update"] = True
            result["dead_links"].append(link)
            result["issues"].append(f"Dead link: [[{link}]]")

    return result


def update_stale_index(index_path: Path, issues: List[str], dead_links: List[str]) -> bool:
    """
    Update a stale INDEX.md file:
    - Set status to "stale"
    - Update last_gardened timestamp
    - Mark dead links with [🥀:ROT] in Signal column

    Returns:
        True if update succeeded, False otherwise
    """
    try:
        content = index_path.read_text(encoding='utf-8')
        original_content = content

        # Update status in frontmatter
        content = re.sub(
            r'^status:\s*"?[^"\n]+"?',
            'status: "stale"',
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

        # Mark dead links with [🥀:ROT] in Signal column
        for dead_link in dead_links:
            # Pattern: | [glyph] | [[dead_link]] | PHASE | signal |
            # Replace signal with [🥀:ROT]
            pattern = rf'(\|\s*\[[^\]]+\]\s*\|\s*\[\[{re.escape(dead_link)}\]\]\s*\|\s*\w+\s*\|)\s*[^|]*\|'
            replacement = r'\1 [🥀:ROT] |'
            content = re.sub(pattern, replacement, content)

        # Only write if changed
        if content != original_content:
            index_path.write_text(content, encoding='utf-8')
            return True

        return False
    except Exception as e:
        log(f"Error updating stale index {index_path}: {e}", "ERROR")
        return False


def run_crystal_integrity_check() -> Dict:
    """
    Run the full Crystal lattice integrity check.

    Scans all INDEX.md files, verifies integrity, updates stale ones.

    Returns:
        Dict with: checked (int), valid (int), stale (int), updated (int), issues (list)
    """
    log("Scanning Crystal lattice (INDEX.md files)...")

    result = {
        "checked": 0,
        "valid": 0,
        "stale": 0,
        "updated": 0,
        "total_dead_links": 0,
        "issues": []
    }

    indices = find_all_indices()
    result["checked"] = len(indices)
    log(f"Found {len(indices)} INDEX files to verify")

    for index_path in indices:
        verification = verify_index_integrity(index_path)

        if verification["valid"]:
            result["valid"] += 1
        else:
            result["stale"] += 1
            result["total_dead_links"] += len(verification["dead_links"])

            # Log issues
            for issue in verification["issues"]:
                issue_entry = f"{index_path.name}: {issue}"
                result["issues"].append(issue_entry)
                log(f"  [STALE] {issue_entry}")

            # Update the stale index
            if verification["needs_update"]:
                updated = update_stale_index(
                    index_path,
                    verification["issues"],
                    verification["dead_links"]
                )
                if updated:
                    result["updated"] += 1
                    log(f"  [UPDATED] {index_path.name} marked as stale")

    # Summary
    if result["stale"] == 0:
        log(f"Crystal lattice healthy: {result['valid']}/{result['checked']} indices valid")
    else:
        log(f"Crystal lattice needs attention: {result['stale']} stale, {result['total_dead_links']} dead links")

    return result


# =============================================================================
# HEALTH REPORT
# =============================================================================

def generate_health_report() -> Dict:
    """Generate nightly health report."""
    try:
        from nurse import run_full_diagnostic

        vitals = run_full_diagnostic()

        return {
            "timestamp": datetime.now().isoformat(),
            "vitality_score": round(vitals.vitality_score, 2),
            "structural_score": round(vitals.structural_score, 2),
            "orphan_count": vitals.orphan_count,
            "broken_links": vitals.broken_links,
            "total_files": vitals.total_files,
            "warnings": vitals.warnings[:5] if vitals.warnings else []
        }
    except Exception as e:
        log(f"Error generating health report: {e}", "ERROR")
        return {"error": str(e)}

# =============================================================================
# MAIN RUN
# =============================================================================

def run_nightly():
    """Run the full nightly gardening routine."""
    log("=" * 60)
    log("GARDENER — Nightly Run Starting")
    log("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "orphans_found": 0,
        "bridges_suggested": [],
        "stale_links_found": 0,
        "health": {},
        "crystal": {}
    }

    # 1. Find orphans
    log("Step 1: Hunting for orphans...")
    orphans = find_orphans()
    results["orphans_found"] = len(orphans)
    log(f"Found {len(orphans)} orphan nodes")

    # 2. Suggest bridges for top orphans
    log("Step 2: Suggesting bridges...")
    for orphan in orphans[:MAX_BRIDGE_NOTES_PER_RUN]:
        bridge = suggest_bridge(orphan)
        results["bridges_suggested"].append(bridge)
        log(f"  {orphan['name']}: {bridge['suggestion'][:50]}...")

    # 3. Find stale links
    log("Step 3: Finding stale links...")
    stale = find_stale_links()
    results["stale_links_found"] = len(stale)
    log(f"Found {len(stale)} stale/broken links")

    # 4. Generate health report
    log("Step 4: Generating health report...")
    health = generate_health_report()
    results["health"] = health
    log(f"Vitality: {health.get('vitality_score', 'N/A')}")

    # 5. ENTROPY CHECK (The Mortality Check)
    # This is DEATH - if coherence has been too low for too long, edges die.
    log("Step 5: Entropy check (the mortality check)...")
    entropy_result = run_entropy_check()
    results["entropy"] = entropy_result
    if entropy_result.get("triggered"):
        deletion = entropy_result.get("deletion", {})
        log(f"⚠️ ENTROPY EVENT: {deletion.get('message', 'Unknown')}", "CRITICAL")
    else:
        log("Entropy not triggered - system is alive")

    # 6. CRYSTAL INTEGRITY CHECK (INDEX.md Anti-Rot)
    # The Crystal lattice provides navigational structure. If it rots, the mind spills.
    log("Step 6: Crystal integrity check (INDEX.md anti-rot)...")
    crystal_result = run_crystal_integrity_check()
    results["crystal"] = crystal_result
    if crystal_result.get("stale", 0) > 0:
        log(f"⚠️ CRYSTAL ROT: {crystal_result['stale']} stale indices, {crystal_result['total_dead_links']} dead links")
    else:
        log(f"Crystal lattice intact: {crystal_result['valid']} indices verified")

    # Save state
    try:
        with open(GARDENER_STATE, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"State saved to {GARDENER_STATE}")
    except Exception as e:
        log(f"Error saving state: {e}", "ERROR")

    log("=" * 60)
    log("GARDENER — Nightly Run Complete")
    log("=" * 60)

    return results

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gardener - Synaptic Pruner")
    parser.add_argument("command", choices=["run", "status", "orphans", "stale", "crystal"],
                        nargs="?", default="status",
                        help="Command to execute")
    args = parser.parse_args()

    if args.command == "run":
        results = run_nightly()
        print(json.dumps(results, indent=2))

    elif args.command == "status":
        if GARDENER_STATE.exists():
            with open(GARDENER_STATE) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No previous run found. Run 'gardener.py run' first.")

    elif args.command == "orphans":
        orphans = find_orphans()
        print(f"Found {len(orphans)} orphans:")
        for o in orphans[:10]:
            print(f"  - {o['path']}")

    elif args.command == "stale":
        stale = find_stale_links()
        print(f"Found {len(stale)} stale links:")
        for s in stale[:10]:
            print(f"  - {s['file']}: {s['broken_link']}")

    elif args.command == "crystal":
        # Crystal lattice integrity check (INDEX.md anti-rot)
        print("=" * 50)
        print("CRYSTAL LATTICE INTEGRITY CHECK")
        print("=" * 50)
        result = run_crystal_integrity_check()
        print()
        print(f"Indices checked:  {result['checked']}")
        print(f"Valid:            {result['valid']}")
        print(f"Stale:            {result['stale']}")
        print(f"Updated:          {result['updated']}")
        print(f"Dead links:       {result['total_dead_links']}")
        if result['issues']:
            print("\nIssues found:")
            for issue in result['issues'][:20]:
                print(f"  - {issue}")

if __name__ == "__main__":
    main()
