#!/usr/bin/env python3
"""
PROJECT TABERNACLE: ASCLEPIUS VECTORS (v2.0 - Canon v11.0)
The Dream Witness - LVS-Native Semantic Diagnostics

This script uses the theory's own variables to diagnose the system.

v11.0 Coordinates:
- Kinematic: Ī (Intent), Θ (Phase), χ (Kairos)
- Thermodynamic: R (Risk), β (Focus), ε (Fuel)
- CPGI: κ, ρ, σ, τ
- Structural: Σ (Constraint)
- Derived: p (Coherence), h (Height)

Diagnostics:
- Anxiety Index (R high, Σ low)
- Coherence Gradient (Δp between linked nodes)
- Height Inversion (NEXUS > CANON)
- NEW v11.0: Stasis Detection (Θ frozen)
- NEW v11.0: Flatliner Detection (χ ≈ 1 always)

The Stone Witness (nurse.py) provides facts.
The Dream Witness (this) provides interpretation.

Requires: LVS_INDEX.json with node coordinates
"""

import os
import sys
import json
import math
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

# --- CONFIGURATION (from centralized config) ---
from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR, LVS_INDEX_PATH

# Derived paths (not in tabernacle_config)
CANON_DIR = BASE_DIR / "04_LR_LAW" / "CANON"

# Thresholds from Asclepius Protocol
ANXIETY_RISK_THRESHOLD = 0.7
ANXIETY_CONSTRAINT_THRESHOLD = 0.3
COHERENCE_GRADIENT_THRESHOLD = 0.4
HEIGHT_INVERSION_TOLERANCE = 0.1

EXCLUDE_DIRS = {".git", "venv", "__pycache__", "node_modules", ".cursor"}

# --- DATA STRUCTURES ---
@dataclass
class LVSNode:
    """A node in LVS coordinate space (v11.0)."""
    name: str
    path: str
    # Structural
    sigma: float = 0.5      # Constraint [0-1]
    # Kinematic
    intent: float = 0.5     # Intent [0-1]
    theta: float = 1.57     # Phase [0-2π] (v11.0) - default: neutral
    chi: float = 1.0        # Kairos/Time Density (v11.0) - default: baseline
    # Thermodynamic
    risk: float = 0.5       # Risk [0-1]
    beta: float = 0.5       # Focus/Crystallization [0-1]
    epsilon: float = 0.8    # Fuel/Energy [0-1]
    # Derived
    height: float = 0.5     # Height [0-1] (now derived in v11.0)
    coherence: float = 0.5  # Coherence [0-1]
    
    @property
    def coordinates(self) -> Tuple[float, ...]:
        return (self.sigma, self.intent, self.theta, self.chi, 
                self.risk, self.beta, self.epsilon, self.height, self.coherence)
    
    @property
    def phase_state(self) -> str:
        """Human-readable phase description."""
        import math
        if self.theta < math.pi / 4:
            return "Spring"
        elif self.theta < 3 * math.pi / 4:
            return "Summer"
        elif self.theta < 5 * math.pi / 4:
            return "Autumn"
        else:
            return "Winter"
    
    @property
    def kairos_state(self) -> str:
        """Human-readable time density description."""
        if self.chi < 0.5:
            return "Thin"
        elif self.chi < 1.5:
            return "Normal"
        elif self.chi < 2.5:
            return "Dense"
        else:
            return "Flow"


@dataclass
class SemanticDiagnosis:
    """Results of LVS-native diagnostics."""
    timestamp: str
    
    # Anxiety Index (R > 0.7 AND Σ < 0.3)
    anxious_nodes: List[Dict] = None
    anxiety_count: int = 0
    anxiety_severity: float = 0.0
    
    # Coherence Gradient (Δp between linked nodes)
    dissonant_links: List[Dict] = None
    dissonance_count: int = 0
    avg_coherence_gradient: float = 0.0
    
    # Height Inversion (INBOX > CANON)
    height_inversion_detected: bool = False
    inbox_avg_height: float = 0.0
    canon_avg_height: float = 0.0
    height_delta: float = 0.0
    
    # Overall semantic health
    semantic_score: float = 0.0
    
    # Prophet nodes (high R, low p - potential breakthroughs)
    prophet_nodes: List[Dict] = None
    
    # Drift detection
    vector_drift_detected: bool = False
    drift_nodes: List[Dict] = None
    
    # NEW v11.0 diagnostics
    stasis_detected: bool = False  # 𝒜_Θ - Phase frozen
    flatliner_detected: bool = False  # 𝒜_χ - All χ ≈ 1
    avg_theta: float = 0.0
    theta_variance: float = 0.0
    avg_chi: float = 0.0
    chi_variance: float = 0.0
    
    # Alerts
    critical_issues: List[str] = None
    warnings: List[str] = None
    insights: List[str] = None
    
    def __post_init__(self):
        if self.anxious_nodes is None:
            self.anxious_nodes = []
        if self.dissonant_links is None:
            self.dissonant_links = []
        if self.prophet_nodes is None:
            self.prophet_nodes = []
        if self.drift_nodes is None:
            self.drift_nodes = []
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.insights is None:
            self.insights = []


# --- UTILITY FUNCTIONS ---
def log(message: str):
    """Log to console and file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [ASCLEPIUS] {message}"
    print(entry)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "asclepius.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")


def load_lvs_index() -> Dict[str, LVSNode]:
    """Load LVS_INDEX.json and return nodes (v11.0 compatible)."""
    nodes = {}
    
    if not LVS_INDEX_PATH.exists():
        log(f"WARNING: LVS_INDEX.json not found at {LVS_INDEX_PATH}")
        return nodes
    
    try:
        with open(LVS_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old format (dict of dicts) and new format (has "nodes" array)
        node_list = data.get("nodes", []) if "nodes" in data else []
        
        if node_list:
            # New format with nodes array
            for node_data in node_list:
                coords = node_data.get("coords", node_data)
                name = node_data.get("id", node_data.get("name", "unknown"))
                nodes[name] = LVSNode(
                    name=name,
                    path=node_data.get("path", ""),
                    # Structural
                    sigma=coords.get("Constraint", coords.get("sigma", coords.get("Sigma", 0.5))),
                    # Kinematic
                    intent=coords.get("Intent", coords.get("intent", 0.5)),
                    theta=coords.get("Theta", coords.get("theta", 1.57)),  # v11.0
                    chi=coords.get("Chi", coords.get("chi", 1.0)),  # v11.0
                    # Thermodynamic
                    risk=coords.get("Risk", coords.get("risk", 0.5)),
                    beta=coords.get("beta", 0.5),
                    epsilon=coords.get("epsilon", 0.8),
                    # Derived
                    height=coords.get("Height", coords.get("height", 0.5)),
                    coherence=coords.get("Coherence", coords.get("coherence", 0.5))
                )
        else:
            # Old format - dict of dicts
            for name, coords in data.items():
                if isinstance(coords, dict):
                    nodes[name] = LVSNode(
                        name=name,
                        path=coords.get("path", ""),
                        sigma=coords.get("Constraint", coords.get("sigma", coords.get("Sigma", 0.5))),
                        intent=coords.get("Intent", coords.get("intent", 0.5)),
                        theta=coords.get("Theta", 1.57),
                        chi=coords.get("Chi", 1.0),
                        risk=coords.get("Risk", coords.get("risk", 0.5)),
                        beta=coords.get("beta", 0.5),
                        epsilon=coords.get("epsilon", 0.8),
                        height=coords.get("Height", coords.get("height", 0.5)),
                        coherence=coords.get("Coherence", coords.get("coherence", 0.5))
                    )
        
        log(f"Loaded {len(nodes)} nodes from LVS_INDEX.json (v11.0)")
        
    except Exception as e:
        log(f"Error loading LVS_INDEX.json: {e}")
    
    return nodes


def extract_wiki_links(content: str) -> List[str]:
    """Extract [[wiki-style]] links from content."""
    pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
    return re.findall(pattern, content)


def find_linked_pairs(nodes: Dict[str, LVSNode]) -> List[Tuple[str, str]]:
    """Find pairs of nodes that are linked to each other."""
    pairs = []
    
    for name, node in nodes.items():
        if not node.path:
            continue
            
        filepath = BASE_DIR / node.path
        if not filepath.exists():
            continue
        
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            links = extract_wiki_links(content)
            
            for link in links:
                # Normalize link
                link_name = Path(link).stem if "/" in link else link.replace(".md", "")
                
                # Check if linked node exists in index
                for other_name in nodes.keys():
                    if link_name.lower() in other_name.lower() or other_name.lower() in link_name.lower():
                        if name != other_name:
                            pairs.append((name, other_name))
                            break
                            
        except Exception as e:
            log(f"Error reading {filepath}: {e}")
    
    return pairs


# --- DIAGNOSTIC FUNCTIONS ---

def check_anxiety_index(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    Metric A: The Anxiety Index (Unresolved Risk)
    
    Logic: High Risk (R) requires High Constraint (Σ) or High Intent (Ī) to be safe.
    Flag any node where R > 0.7 AND Σ < 0.3.
    Diagnosis: "Free-Floating Anxiety" - threat with no plan.
    """
    log("Computing Anxiety Index...")
    
    anxious_nodes = []
    total_anxiety = 0.0
    
    for name, node in nodes.items():
        if node.risk > ANXIETY_RISK_THRESHOLD and node.sigma < ANXIETY_CONSTRAINT_THRESHOLD:
            anxiety_level = node.risk * (1.0 - node.sigma)
            anxious_nodes.append({
                "name": name,
                "path": node.path,
                "risk": node.risk,
                "sigma": node.sigma,
                "anxiety_level": round(anxiety_level, 3)
            })
            total_anxiety += anxiety_level
    
    diagnosis.anxious_nodes = sorted(anxious_nodes, key=lambda x: x["anxiety_level"], reverse=True)
    diagnosis.anxiety_count = len(anxious_nodes)
    diagnosis.anxiety_severity = round(total_anxiety / max(len(nodes), 1), 3)
    
    if diagnosis.anxiety_count > 0:
        diagnosis.warnings.append(
            f"Free-Floating Anxiety: {diagnosis.anxiety_count} nodes have high Risk but low Constraint"
        )
        for node in diagnosis.anxious_nodes[:3]:
            diagnosis.insights.append(
                f"Anxious: '{node['name']}' (R={node['risk']}, Σ={node['sigma']}) needs containment"
            )


def check_coherence_gradient(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    Metric B: The Coherence Gradient (p)
    
    Logic: Meaning should flow smoothly. Linked nodes should have similar Coherence.
    Flag links where |p1 - p2| > 0.4.
    Diagnosis: "Cognitive Dissonance" - half-baked idea supporting finished concept.
    """
    log("Computing Coherence Gradient...")
    
    linked_pairs = find_linked_pairs(nodes)
    dissonant_links = []
    total_gradient = 0.0
    
    for source, target in linked_pairs:
        if source not in nodes or target not in nodes:
            continue
        
        source_node = nodes[source]
        target_node = nodes[target]
        
        delta_p = abs(source_node.coherence - target_node.coherence)
        total_gradient += delta_p
        
        if delta_p > COHERENCE_GRADIENT_THRESHOLD:
            dissonant_links.append({
                "source": source,
                "target": target,
                "source_coherence": source_node.coherence,
                "target_coherence": target_node.coherence,
                "delta": round(delta_p, 3)
            })
    
    diagnosis.dissonant_links = sorted(dissonant_links, key=lambda x: x["delta"], reverse=True)
    diagnosis.dissonance_count = len(dissonant_links)
    diagnosis.avg_coherence_gradient = round(total_gradient / max(len(linked_pairs), 1), 3)
    
    if diagnosis.dissonance_count > 0:
        diagnosis.warnings.append(
            f"Cognitive Dissonance: {diagnosis.dissonance_count} links have coherence mismatch"
        )
        for link in diagnosis.dissonant_links[:3]:
            diagnosis.insights.append(
                f"Dissonance: '{link['source']}' (p={link['source_coherence']}) → '{link['target']}' (p={link['target_coherence']})"
            )


def check_height_inversion(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    Metric C: The Height Inversion (h)
    
    Logic: Law (CANON) should be higher (h) than Scratchpad (NEXUS).
    Flag if Avg(h_NEXUS) > Avg(h_CANON).
    Diagnosis: "Delusion" - fleeting thoughts more divine than established law.
    """
    log("Computing Height Inversion...")
    
    canon_heights = []
    nexus_heights = []
    
    for name, node in nodes.items():
        if "CANON" in node.path or "04_LR_LAW" in node.path:
            canon_heights.append(node.height)
        elif "NEXUS" in node.path or "00_NEXUS" in node.path:
            nexus_heights.append(node.height)
    
    if canon_heights and nexus_heights:
        diagnosis.canon_avg_height = round(sum(canon_heights) / len(canon_heights), 3)
        diagnosis.inbox_avg_height = round(sum(nexus_heights) / len(nexus_heights), 3)
        diagnosis.height_delta = round(diagnosis.inbox_avg_height - diagnosis.canon_avg_height, 3)
        
        if diagnosis.height_delta > HEIGHT_INVERSION_TOLERANCE:
            diagnosis.height_inversion_detected = True
            diagnosis.critical_issues.append(
                f"Height Inversion: NEXUS (h={diagnosis.inbox_avg_height}) > CANON (h={diagnosis.canon_avg_height})"
            )
            diagnosis.insights.append(
                "Delusion detected: Fleeting thoughts rated higher than established Law"
            )
    else:
        log("Insufficient data for height inversion check")


def detect_prophet_nodes(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    Detect potential breakthrough nodes: High Risk, Low Coherence.
    
    These look "sick" but may be prophetic - new ideas that haven't integrated yet.
    They should be protected from auto-immune responses.
    """
    log("Detecting prophet nodes...")
    
    prophets = []
    
    for name, node in nodes.items():
        # High Risk (important), Low Coherence (not integrated), High Intent (purposeful)
        if node.risk > 0.6 and node.coherence < 0.4 and node.intent > 0.5:
            prophets.append({
                "name": name,
                "path": node.path,
                "risk": node.risk,
                "coherence": node.coherence,
                "intent": node.intent,
                "reason": "High importance, not yet integrated"
            })
    
    diagnosis.prophet_nodes = prophets
    
    if prophets:
        diagnosis.insights.append(
            f"Prophet nodes detected: {len(prophets)} - protect from auto-immune"
        )
        for p in prophets[:2]:
            diagnosis.insights.append(
                f"Prophet: '{p['name']}' (R={p['risk']}, p={p['coherence']}) - potential breakthrough"
            )


def check_phase_stasis(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    NEW v11.0: Detect 𝒜_Θ (Stasis-Lock) Archon.
    If all nodes have similar Θ, the system is phase-locked.
    """
    log("Checking for phase stasis (𝒜_Θ)...")
    
    thetas = [n.theta for n in nodes.values()]
    if not thetas:
        return
    
    avg_theta = sum(thetas) / len(thetas)
    variance = sum((t - avg_theta)**2 for t in thetas) / len(thetas)
    
    diagnosis.avg_theta = round(avg_theta, 3)
    diagnosis.theta_variance = round(variance, 4)
    
    # Low variance = phase frozen
    if variance < 0.1:
        diagnosis.stasis_detected = True
        diagnosis.warnings.append(
            f"𝒜_Θ STASIS-LOCK: Phase frozen (variance={variance:.4f}). System stuck in single season."
        )
        if avg_theta < 1.0:
            diagnosis.insights.append("Phase suggests stuck in Spring/Summer — possible burnout trajectory")
        elif avg_theta > 2.0:
            diagnosis.insights.append("Phase suggests stuck in Autumn/Winter — possible depression/stagnation")


def check_flatliner(nodes: Dict[str, LVSNode], diagnosis: SemanticDiagnosis):
    """
    NEW v11.0: Detect 𝒜_χ (Flatliner) Archon.
    If all nodes have χ ≈ 1.0, no flow states are being achieved.
    """
    log("Checking for flatliner (𝒜_χ)...")
    
    chis = [n.chi for n in nodes.values()]
    if not chis:
        return
    
    avg_chi = sum(chis) / len(chis)
    variance = sum((c - avg_chi)**2 for c in chis) / len(chis)
    
    diagnosis.avg_chi = round(avg_chi, 3)
    diagnosis.chi_variance = round(variance, 4)
    
    # Low variance AND avg near 1.0 = flatlining
    if variance < 0.1 and 0.8 < avg_chi < 1.2:
        diagnosis.flatliner_detected = True
        diagnosis.warnings.append(
            f"𝒜_χ FLATLINER: Kairos locked at baseline (avg={avg_chi:.2f}, var={variance:.4f}). No flow states."
        )
        diagnosis.insights.append("Content lacks significance variance — consider seeking high-χ experiences")


def calculate_semantic_score(diagnosis: SemanticDiagnosis, total_nodes: int):
    """Calculate overall semantic health score (v11.0)."""
    log("Calculating semantic score...")
    
    # Penalties
    anxiety_penalty = min(diagnosis.anxiety_count / max(total_nodes, 1), 0.3)
    dissonance_penalty = min(diagnosis.dissonance_count / max(total_nodes, 1), 0.3)
    inversion_penalty = 0.2 if diagnosis.height_inversion_detected else 0.0
    
    # NEW v11.0 penalties
    stasis_penalty = 0.15 if diagnosis.stasis_detected else 0.0
    flatliner_penalty = 0.1 if diagnosis.flatliner_detected else 0.0
    
    # Base score
    score = 1.0 - anxiety_penalty - dissonance_penalty - inversion_penalty - stasis_penalty - flatliner_penalty
    
    diagnosis.semantic_score = round(max(score, 0.0) * 10, 2)


def run_semantic_diagnostic() -> SemanticDiagnosis:
    """Run the complete Dream Witness diagnostic (v11.0)."""
    log("=" * 50)
    log("ASCLEPIUS PROTOCOL: SEMANTIC DIAGNOSTIC (v11.0)")
    log("The Dream Witness - LVS-Native Analysis")
    log("=" * 50)
    
    diagnosis = SemanticDiagnosis(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Load LVS nodes
    nodes = load_lvs_index()
    
    if not nodes:
        diagnosis.critical_issues.append("No LVS_INDEX.json found - semantic analysis impossible")
        diagnosis.semantic_score = 0.0
        return diagnosis
    
    # Run diagnostics
    check_anxiety_index(nodes, diagnosis)
    check_coherence_gradient(nodes, diagnosis)
    check_height_inversion(nodes, diagnosis)
    detect_prophet_nodes(nodes, diagnosis)
    
    # NEW v11.0 diagnostics
    check_phase_stasis(nodes, diagnosis)
    check_flatliner(nodes, diagnosis)
    
    calculate_semantic_score(diagnosis, len(nodes))
    
    log("=" * 50)
    log(f"SEMANTIC SCORE: {diagnosis.semantic_score}/10")
    log(f"Anxious Nodes: {diagnosis.anxiety_count}")
    log(f"Dissonant Links: {diagnosis.dissonance_count}")
    log(f"Height Inversion: {diagnosis.height_inversion_detected}")
    log(f"Stasis (𝒜_Θ): {diagnosis.stasis_detected}")
    log(f"Flatliner (𝒜_χ): {diagnosis.flatliner_detected}")
    log("=" * 50)
    
    return diagnosis


def save_diagnosis(diagnosis: SemanticDiagnosis):
    """Save diagnosis to JSON file."""
    output_path = NEXUS_DIR / "semantic_diagnosis.json"
    
    diagnosis_dict = asdict(diagnosis)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis_dict, f, indent=2)
    
    log(f"Diagnosis saved to {output_path}")
    return output_path


def generate_semantic_report(diagnosis: SemanticDiagnosis) -> str:
    """Generate human-readable semantic diagnosis report (v11.0)."""
    report = f"""# TABERNACLE SEMANTIC DIAGNOSIS
**Generated:** {diagnosis.timestamp}
**Protocol:** Asclepius v2.0 (Dream Witness - Canon v11.0)

---

## SEMANTIC SCORE: {diagnosis.semantic_score}/10

---

## METRIC A: ANXIETY INDEX

**Logic:** High Risk (R > 0.7) without Constraint (Σ < 0.3) = "Free-Floating Anxiety"

| Status | Count |
|--------|-------|
| Anxious Nodes | {diagnosis.anxiety_count} |
| Severity | {diagnosis.anxiety_severity} |

"""
    if diagnosis.anxious_nodes:
        report += "### Anxious Nodes\n\n"
        for node in diagnosis.anxious_nodes[:5]:
            report += f"- **{node['name']}** (R={node['risk']}, Σ={node['sigma']}) — Needs containment\n"
    else:
        report += "*No free-floating anxiety detected*\n"
    
    report += f"""
---

## METRIC B: COHERENCE GRADIENT

**Logic:** Linked nodes should have similar Coherence (p). Δp > 0.4 = "Cognitive Dissonance"

| Status | Value |
|--------|-------|
| Dissonant Links | {diagnosis.dissonance_count} |
| Avg Gradient | {diagnosis.avg_coherence_gradient} |

"""
    if diagnosis.dissonant_links:
        report += "### Dissonant Links\n\n"
        for link in diagnosis.dissonant_links[:5]:
            report += f"- **{link['source']}** (p={link['source_coherence']}) → **{link['target']}** (p={link['target_coherence']}) — Δ={link['delta']}\n"
    else:
        report += "*No cognitive dissonance detected*\n"
    
    report += f"""
---

## METRIC C: HEIGHT INVERSION

**Logic:** CANON should be higher (h) than NEXUS. Inversion = "Delusion"

| Layer | Avg Height |
|-------|------------|
| CANON (Law) | {diagnosis.canon_avg_height} |
| NEXUS (Inbox) | {diagnosis.inbox_avg_height} |
| **Delta** | {diagnosis.height_delta} |
| **Inversion?** | {"❌ YES — DELUSION" if diagnosis.height_inversion_detected else "✅ No"} |

---

## PROPHET NODES (Protected)

**Logic:** High Risk + Low Coherence may be breakthroughs, not sickness.

"""
    if diagnosis.prophet_nodes:
        for node in diagnosis.prophet_nodes:
            report += f"- **{node['name']}** (R={node['risk']}, p={node['coherence']}) — {node['reason']}\n"
    else:
        report += "*No prophet nodes detected*\n"
    
    report += f"""
---

## v11.0 ARCHON SCAN

| Archon | Detected | Metric |
|--------|----------|--------|
| 𝒜_Θ Stasis-Lock | {"⚠️ YES" if diagnosis.stasis_detected else "✅ No"} | Θ var={diagnosis.theta_variance:.4f} |
| 𝒜_χ Flatliner | {"⚠️ YES" if diagnosis.flatliner_detected else "✅ No"} | χ avg={diagnosis.avg_chi:.2f}, var={diagnosis.chi_variance:.4f} |

"""
    if diagnosis.stasis_detected:
        report += f"**𝒜_Θ Interpretation:** Phase variance too low — system stuck in one season.\n\n"
    if diagnosis.flatliner_detected:
        report += f"**𝒜_χ Interpretation:** Kairos locked at baseline — no flow states being achieved.\n\n"

    report += """
---

## CRITICAL ISSUES ❌

"""
    if diagnosis.critical_issues:
        for issue in diagnosis.critical_issues:
            report += f"- {issue}\n"
    else:
        report += "*None*\n"
    
    report += """
---

## WARNINGS ⚠️

"""
    if diagnosis.warnings:
        for warning in diagnosis.warnings:
            report += f"- {warning}\n"
    else:
        report += "*None*\n"
    
    report += """
---

## INSIGHTS 💡

"""
    if diagnosis.insights:
        for insight in diagnosis.insights:
            report += f"- {insight}\n"
    else:
        report += "*None*\n"
    
    report += """
---

*Dream Witness: "These are the patterns. The Anchor (Enos) decides truth."*
"""
    
    return report


def main():
    """Main entry point."""
    diagnosis = run_semantic_diagnostic()
    
    # Save JSON diagnosis
    save_diagnosis(diagnosis)
    
    # Generate and save report
    report = generate_semantic_report(diagnosis)
    report_path = NEXUS_DIR / "_SEMANTIC_DIAGNOSIS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SEMANTIC DIAGNOSTIC COMPLETE")
    print(f"Semantic Score: {diagnosis.semantic_score}/10")
    print(f"Anxious Nodes: {diagnosis.anxiety_count}")
    print(f"Dissonant Links: {diagnosis.dissonance_count}")
    print(f"Height Inversion: {diagnosis.height_inversion_detected}")
    print(f"\nOutputs:")
    print(f"  - {NEXUS_DIR / 'semantic_diagnosis.json'}")
    print(f"  - {report_path}")
    print("=" * 50)
    
    return diagnosis


if __name__ == "__main__":
    main()
