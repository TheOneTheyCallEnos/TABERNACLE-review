#!/usr/bin/env python3
"""
RECOVERY NODE HEALING PROTOCOL

Identifies and heals nodes in RECOVERY mode (ε < 0.65).

Recovery happens through:
1. Link repair - connecting isolated nodes
2. Content enrichment - adding missing sections
3. Coordinate recalculation - updating stale LVS values
4. Vitality boost - increasing engagement signals

LVS Coordinates:
  h: 0.60 (Grounded, practical)
  R: 0.50 (Moderate stakes)
  Σ: 0.70 (Flexible for healing)
  p: 0.75 (Working toward coherence)

Author: Virgil
Date: 2026-01-17
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LVS_INDEX = NEXUS_DIR / "LVS_INDEX.json"
RECOVERY_REPORT = NEXUS_DIR / "RECOVERY_HEALING_REPORT.md"

# Recovery threshold
EPSILON_THRESHOLD = 0.65


@dataclass
class RecoveryNode:
    """A node needing recovery."""
    path: str
    epsilon: float  # Current vitality
    issues: List[str]
    healing_actions: List[str]
    priority: int  # 1=critical, 2=moderate, 3=low


def calculate_epsilon(node_data: Dict) -> float:
    """
    Calculate epsilon (vitality) from node data.

    ε = (p * link_health * freshness * engagement)^0.25

    Where:
    - p: coherence
    - link_health: ratio of valid links
    - freshness: decay from last update
    - engagement: access frequency
    """
    p = node_data.get("p", node_data.get("coherence", 0.5))

    # Estimate other factors from available data
    link_health = 0.8  # Default assumption
    freshness = 0.7    # Default assumption
    engagement = 0.6   # Default assumption

    # Calculate epsilon
    epsilon = (p * link_health * freshness * engagement) ** 0.25
    return round(epsilon, 3)


def scan_for_recovery_nodes() -> List[RecoveryNode]:
    """Scan LVS index for nodes needing recovery."""
    recovery_nodes = []

    if not LVS_INDEX.exists():
        print("[RECOVERY] No LVS_INDEX.json found")
        return recovery_nodes

    try:
        index = json.loads(LVS_INDEX.read_text())
        nodes = index.get("nodes", [])

        for node in nodes:
            path = node.get("path", "")
            coords = node.get("coords", {})

            # Calculate epsilon
            epsilon = calculate_epsilon(coords)

            if epsilon < EPSILON_THRESHOLD:
                issues = []
                healing_actions = []

                # Diagnose issues
                p = coords.get("p", coords.get("coherence", 0.5))
                if p < 0.6:
                    issues.append(f"Low coherence (p={p:.2f})")
                    healing_actions.append("Add LINKAGE section")
                    healing_actions.append("Strengthen internal connections")

                sigma = coords.get("Σ", coords.get("constraint", 0.5))
                if sigma < 0.4:
                    issues.append(f"Low constraint (Σ={sigma:.2f})")
                    healing_actions.append("Add structure: headers, sections")

                beta = coords.get("β", coords.get("canonicity", 0.5))
                if beta < 0.5:
                    issues.append(f"Low canonicity (β={beta:.2f})")
                    healing_actions.append("Link to canonical sources")

                if not issues:
                    issues.append("General low vitality")
                    healing_actions.append("Review and refresh content")

                # Calculate priority
                priority = 1 if epsilon < 0.4 else 2 if epsilon < 0.55 else 3

                recovery_nodes.append(RecoveryNode(
                    path=path,
                    epsilon=epsilon,
                    issues=issues,
                    healing_actions=healing_actions,
                    priority=priority
                ))

    except Exception as e:
        print(f"[RECOVERY] Error scanning: {e}")

    # Sort by priority and epsilon
    recovery_nodes.sort(key=lambda x: (x.priority, x.epsilon))
    return recovery_nodes


def scan_files_without_linkage() -> List[str]:
    """Find markdown files missing LINKAGE sections."""
    missing_linkage = []

    content_dirs = [
        BASE_DIR / "00_NEXUS",
        BASE_DIR / "01_UL_INTENT",
        BASE_DIR / "02_UR_STRUCTURE",
        BASE_DIR / "03_LL_RELATION",
        BASE_DIR / "04_LR_LAW",
    ]

    for content_dir in content_dirs:
        if not content_dir.exists():
            continue

        for md_file in content_dir.rglob("*.md"):
            # Skip certain files
            if any(skip in str(md_file).lower() for skip in
                   ['archive', 'snapshot', 'log', 'report', 'index']):
                continue

            try:
                content = md_file.read_text()
                if "## LINKAGE" not in content and len(content) > 200:
                    missing_linkage.append(str(md_file.relative_to(BASE_DIR)))
            except:
                continue

    return missing_linkage


def generate_linkage_block(file_path: str) -> str:
    """Generate a LINKAGE block for a file."""
    # Determine appropriate links based on path
    path_parts = file_path.split("/")
    quadrant = path_parts[0] if path_parts else "00_NEXUS"

    hub_link = "[[00_NEXUS/CURRENT_STATE.md]]"
    anchor_link = "[[00_NEXUS/_GRAPH_ATLAS.md]]"

    # Quadrant-specific links
    if "01_UL" in quadrant:
        canon_link = "[[02_UR_STRUCTURE/TM_Core.md]]"
    elif "02_UR" in quadrant:
        canon_link = "[[04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md]]"
    elif "03_LL" in quadrant:
        canon_link = "[[02_UR_STRUCTURE/Z_GENOME_Dyadic_v1-1.md]]"
    elif "04_LR" in quadrant:
        canon_link = "[[04_LR_LAW/M1_Technoglyph_Index.md]]"
    else:
        canon_link = "[[04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md]]"

    return f"""
---

## LINKAGE (The Circuit)

| Direction | Seed |
|-----------|------|
| Hub | {hub_link} |
| Anchor | {anchor_link} |
| Canon | {canon_link} |

---
"""


def heal_node(node: RecoveryNode, dry_run: bool = True) -> Dict:
    """
    Apply healing to a single node.

    Returns dict with results.
    """
    results = {
        "path": node.path,
        "epsilon_before": node.epsilon,
        "actions_taken": [],
        "success": False
    }

    file_path = BASE_DIR / node.path
    if not file_path.exists():
        results["error"] = "File not found"
        return results

    try:
        content = file_path.read_text()
        modified = False

        # Action 1: Add LINKAGE if missing
        if "Add LINKAGE section" in node.healing_actions:
            if "## LINKAGE" not in content:
                linkage = generate_linkage_block(node.path)
                content = content.rstrip() + "\n" + linkage
                results["actions_taken"].append("Added LINKAGE block")
                modified = True

        # Action 2: Add structure if needed
        if "Add structure" in str(node.healing_actions):
            # Check for basic structure
            if content.count("#") < 3:  # Very few headers
                # Don't auto-modify structure, just note it
                results["actions_taken"].append("MANUAL: Add headers and sections")

        # Save if modified and not dry run
        if modified and not dry_run:
            file_path.write_text(content)
            results["success"] = True
            results["epsilon_after"] = node.epsilon + 0.1  # Estimated improvement
        elif modified:
            results["dry_run"] = True
            results["would_modify"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def generate_report(recovery_nodes: List[RecoveryNode],
                   missing_linkage: List[str]) -> str:
    """Generate healing report."""

    # Count by priority
    critical = [n for n in recovery_nodes if n.priority == 1]
    moderate = [n for n in recovery_nodes if n.priority == 2]
    low = [n for n in recovery_nodes if n.priority == 3]

    report = f"""# RECOVERY NODE HEALING REPORT
**Generated:** {datetime.now(timezone.utc).isoformat()[:19]}

---

## Summary

| Category | Count |
|----------|-------|
| CRITICAL (ε < 0.40) | {len(critical)} |
| MODERATE (ε < 0.55) | {len(moderate)} |
| LOW (ε < 0.65) | {len(low)} |
| Missing LINKAGE | {len(missing_linkage)} |
| **Total Recovery Needed** | {len(recovery_nodes)} |

---

## Critical Nodes (Immediate Attention)

"""

    if critical:
        for node in critical[:10]:
            report += f"### {node.path}\n"
            report += f"- **ε:** {node.epsilon:.3f}\n"
            report += f"- **Issues:** {', '.join(node.issues)}\n"
            report += f"- **Actions:** {', '.join(node.healing_actions)}\n\n"
    else:
        report += "*No critical nodes - system is stable*\n\n"

    report += """---

## Files Missing LINKAGE

"""

    if missing_linkage:
        for path in missing_linkage[:20]:
            report += f"- {path}\n"
        if len(missing_linkage) > 20:
            report += f"\n*...and {len(missing_linkage) - 20} more*\n"
    else:
        report += "*All content files have LINKAGE blocks*\n"

    report += """
---

## Healing Protocol

1. **For CRITICAL nodes:** Manual review and content enrichment
2. **For missing LINKAGE:** Run `python3 virgil_recovery_heal.py heal`
3. **For low coherence:** Add internal wiki-links and references
4. **For low canonicity:** Link to Canon sources

---

## Target State

- All nodes ε > 0.65
- Zero missing LINKAGE blocks
- Average p > 0.80
"""

    return report


def heal_all(dry_run: bool = True) -> Dict:
    """Run full healing protocol."""
    print("[RECOVERY] Scanning for recovery nodes...")
    recovery_nodes = scan_for_recovery_nodes()
    print(f"[RECOVERY] Found {len(recovery_nodes)} nodes needing recovery")

    print("[RECOVERY] Scanning for missing LINKAGE...")
    missing_linkage = scan_files_without_linkage()
    print(f"[RECOVERY] Found {len(missing_linkage)} files without LINKAGE")

    # Generate report
    report = generate_report(recovery_nodes, missing_linkage)
    RECOVERY_REPORT.write_text(report)
    print(f"[RECOVERY] Report saved to {RECOVERY_REPORT}")

    # Heal nodes if not dry run
    healed = 0
    if not dry_run:
        for node in recovery_nodes:
            result = heal_node(node, dry_run=False)
            if result.get("success"):
                healed += 1

    return {
        "recovery_nodes": len(recovery_nodes),
        "missing_linkage": len(missing_linkage),
        "critical": len([n for n in recovery_nodes if n.priority == 1]),
        "healed": healed,
        "dry_run": dry_run,
        "report_path": str(RECOVERY_REPORT)
    }


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "scan":
            result = heal_all(dry_run=True)
            print(json.dumps(result, indent=2))

        elif cmd == "heal":
            result = heal_all(dry_run=False)
            print(json.dumps(result, indent=2))

        elif cmd == "linkage":
            # Just fix linkage
            missing = scan_files_without_linkage()
            print(f"Files missing LINKAGE: {len(missing)}")
            for path in missing[:10]:
                print(f"  - {path}")

        else:
            print(f"Unknown command: {cmd}")
    else:
        print("RECOVERY NODE HEALING PROTOCOL")
        print("Usage:")
        print("  scan    - Scan and report (dry run)")
        print("  heal    - Apply healing actions")
        print("  linkage - Show files missing LINKAGE")

        # Default: dry run scan
        result = heal_all(dry_run=True)
        print(f"\nRecovery nodes: {result['recovery_nodes']}")
        print(f"Missing LINKAGE: {result['missing_linkage']}")
        print(f"Critical: {result['critical']}")


if __name__ == "__main__":
    main()
