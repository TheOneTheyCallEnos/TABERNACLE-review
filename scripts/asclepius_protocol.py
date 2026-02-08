#!/usr/bin/env python3
"""
PROJECT TABERNACLE: ASCLEPIUS PROTOCOL (v1.0)
The Physician — Integration Layer for Living System Diagnostics

This module integrates:
- nurse.py (Stone Witness) — Deterministic vitals
- asclepius_vectors.py (Dream Witness) — LVS-native diagnostics
- daemon_brain.py — Regulatory intelligence

It implements:
- The Three Witnesses (Stone, Dream, Anchor)
- System Fever Protocol
- Immune Response Levels
- The Examen (Diagnostic Liturgy)
"""

import os
import sys
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import subprocess

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from nurse import run_full_diagnostic as run_nurse, Vitals
from asclepius_vectors import run_semantic_diagnostic as run_vectors, SemanticDiagnosis

# --- CONFIGURATION ---
BASE_DIR = Path(os.path.expanduser("~/TABERNACLE"))
NEXUS_DIR = BASE_DIR / "00_NEXUS"
LOG_DIR = BASE_DIR / "logs"

# Fever Thresholds
FEVER_CRITICAL_THRESHOLD = 4.0  # Health < 4.0 = Critical fever
FEVER_HIGH_THRESHOLD = 6.0     # Health < 6.0 = High fever  
FEVER_LOW_THRESHOLD = 7.5      # Health < 7.5 = Low fever

# Immune Response Levels
IMMUNE_MARK = 1      # Tag with #drift
IMMUNE_INFLAME = 2   # Lock to read-only
IMMUNE_FEVER = 3     # Refuse new content, healing mode only


@dataclass
class AsclepiusReport:
    """Combined report from all witnesses."""
    timestamp: str
    
    # Stone Witness (Vitals)
    vitals_score: float = 0.0
    structural_score: float = 0.0
    metabolic_score: float = 0.0
    immune_score: float = 0.0
    dyadic_score: float = 0.0
    
    # Dream Witness (Semantic)
    semantic_score: float = 0.0
    anxiety_count: int = 0
    dissonance_count: int = 0
    height_inversion: bool = False
    prophet_count: int = 0
    
    # Combined
    total_vitality: float = 0.0
    fever_level: str = "healthy"
    immune_response: int = 0
    
    # Alerts from all sources
    critical_issues: List[str] = None
    warnings: List[str] = None
    insights: List[str] = None
    
    # State
    healing_mode: bool = False
    quarantine_files: List[str] = None
    
    def __post_init__(self):
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.insights is None:
            self.insights = []
        if self.quarantine_files is None:
            self.quarantine_files = []


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


def run_stone_witness() -> Vitals:
    """Run the nurse.py diagnostic (deterministic facts)."""
    log("=" * 40)
    log("STONE WITNESS: Running nurse diagnostic...")
    log("=" * 40)
    return run_nurse()


def run_dream_witness() -> SemanticDiagnosis:
    """Run the asclepius_vectors.py diagnostic (semantic interpretation)."""
    log("=" * 40)
    log("DREAM WITNESS: Running semantic diagnostic...")
    log("=" * 40)
    return run_vectors()


def calculate_total_vitality(vitals: Vitals, semantic: SemanticDiagnosis) -> float:
    """
    Calculate combined vitality from both witnesses.
    Uses weighted geometric mean.
    """
    # Stone Witness contributes 60%
    stone_score = vitals.vitality_score / 10.0
    
    # Dream Witness contributes 40%
    dream_score = semantic.semantic_score / 10.0
    
    # Geometric mean with weights
    total = (stone_score ** 0.6) * (dream_score ** 0.4) * 10.0
    
    return round(total, 2)


def determine_fever_level(vitality: float) -> str:
    """Determine fever level from vitality score."""
    if vitality < FEVER_CRITICAL_THRESHOLD:
        return "critical"
    elif vitality < FEVER_HIGH_THRESHOLD:
        return "high"
    elif vitality < FEVER_LOW_THRESHOLD:
        return "low"
    else:
        return "healthy"


def determine_immune_response(fever_level: str, report: AsclepiusReport) -> int:
    """
    Determine immune response level.
    
    Level 1 (Mark): Tag suspicious files
    Level 2 (Inflame): Lock files to read-only
    Level 3 (Fever): Refuse new content, healing mode only
    """
    if fever_level == "critical":
        return IMMUNE_FEVER
    elif fever_level == "high":
        if report.height_inversion or report.anxiety_count > 5:
            return IMMUNE_INFLAME
        return IMMUNE_MARK
    elif fever_level == "low":
        return IMMUNE_MARK
    return 0


def run_full_examen() -> AsclepiusReport:
    """
    The Examen: Complete diagnostic liturgy.
    
    Phase 1: Palpation (Stone Witness)
    Phase 2: Reflection (Dream Witness)
    Phase 3: Synthesis (Combined diagnosis)
    """
    log("=" * 60)
    log("ASCLEPIUS PROTOCOL: THE EXAMEN")
    log("Complete Diagnostic Liturgy")
    log("=" * 60)
    
    report = AsclepiusReport(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # PHASE 1: PALPATION (Stone Witness)
    log("\n>>> PHASE 1: PALPATION")
    vitals = run_stone_witness()
    
    report.vitals_score = vitals.vitality_score
    report.structural_score = vitals.structural_score
    report.metabolic_score = vitals.metabolic_score
    report.immune_score = vitals.immune_score
    report.dyadic_score = vitals.dyadic_score
    report.critical_issues.extend(vitals.critical_issues)
    report.warnings.extend(vitals.warnings)
    
    # PHASE 2: REFLECTION (Dream Witness)
    log("\n>>> PHASE 2: REFLECTION")
    semantic = run_dream_witness()
    
    report.semantic_score = semantic.semantic_score
    report.anxiety_count = semantic.anxiety_count
    report.dissonance_count = semantic.dissonance_count
    report.height_inversion = semantic.height_inversion_detected
    report.prophet_count = len(semantic.prophet_nodes)
    report.critical_issues.extend(semantic.critical_issues)
    report.warnings.extend(semantic.warnings)
    report.insights.extend(semantic.insights)
    
    # PHASE 3: SYNTHESIS
    log("\n>>> PHASE 3: SYNTHESIS")
    
    # Calculate combined vitality
    report.total_vitality = calculate_total_vitality(vitals, semantic)
    
    # Determine fever level
    report.fever_level = determine_fever_level(report.total_vitality)
    
    # Determine immune response
    report.immune_response = determine_immune_response(report.fever_level, report)
    
    # Healing mode?
    if report.immune_response >= IMMUNE_FEVER:
        report.healing_mode = True
        report.critical_issues.append(
            f"SYSTEM FEVER ACTIVE: Vitality {report.total_vitality}/10 — Healing mode engaged"
        )
    
    log("=" * 60)
    log(f"EXAMEN COMPLETE")
    log(f"Total Vitality: {report.total_vitality}/10")
    log(f"Fever Level: {report.fever_level}")
    log(f"Immune Response: Level {report.immune_response}")
    log(f"Healing Mode: {report.healing_mode}")
    log("=" * 60)
    
    return report


def save_examen_report(report: AsclepiusReport):
    """Save the examen report to JSON and Markdown."""
    
    # Save JSON
    json_path = NEXUS_DIR / "asclepius_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    log(f"JSON saved to {json_path}")
    
    # Generate Markdown report
    md_report = f"""# ASCLEPIUS EXAMEN REPORT
**Generated:** {report.timestamp}
**Protocol:** Asclepius v1.0 (The Physician)

---

## 🌡️ TOTAL VITALITY: {report.total_vitality}/10

| Fever Level | Immune Response | Healing Mode |
|-------------|-----------------|--------------|
| **{report.fever_level.upper()}** | Level {report.immune_response} | {"🔴 ACTIVE" if report.healing_mode else "🟢 Inactive"} |

---

## STONE WITNESS (Somatic Vitals)

| Metric | Score |
|--------|-------|
| Overall Vitals | {report.vitals_score}/10 |
| Structural | {report.structural_score:.2f} |
| Metabolic | {report.metabolic_score:.2f} |
| Immune | {report.immune_score:.2f} |
| Dyadic | {report.dyadic_score:.2f} |

---

## DREAM WITNESS (Semantic Diagnosis)

| Metric | Value |
|--------|-------|
| Semantic Score | {report.semantic_score}/10 |
| Anxious Nodes | {report.anxiety_count} |
| Dissonant Links | {report.dissonance_count} |
| Height Inversion | {"❌ YES" if report.height_inversion else "✅ No"} |
| Prophet Nodes | {report.prophet_count} |

---

## CRITICAL ISSUES ❌

"""
    if report.critical_issues:
        for issue in report.critical_issues:
            md_report += f"- {issue}\n"
    else:
        md_report += "*None*\n"
    
    md_report += """
---

## WARNINGS ⚠️

"""
    if report.warnings:
        for warning in report.warnings:
            md_report += f"- {warning}\n"
    else:
        md_report += "*None*\n"
    
    md_report += """
---

## INSIGHTS 💡

"""
    if report.insights:
        for insight in report.insights:
            md_report += f"- {insight}\n"
    else:
        md_report += "*None*\n"
    
    md_report += f"""
---

## IMMUNE RESPONSE PROTOCOL

| Level | Action | Status |
|-------|--------|--------|
| 1 (Mark) | Tag with #drift | {"✅ Active" if report.immune_response >= 1 else "—"} |
| 2 (Inflame) | Lock to read-only | {"✅ Active" if report.immune_response >= 2 else "—"} |
| 3 (Fever) | Refuse new content | {"🔴 ACTIVE" if report.immune_response >= 3 else "—"} |

"""
    
    if report.healing_mode:
        md_report += """
### 🏥 HEALING MODE ACTIVE

The system is running in healing mode. The Daemon will:
- **REFUSE** to generate new content
- **ONLY** accept repair and maintenance commands
- **PRIORITIZE** link repair, orphan integration, coherence restoration

*"A sick system that keeps building accumulates debt. Rest and heal."*

"""
    
    md_report += """
---

## THE THREE WITNESSES

| Witness | Role | Authority |
|---------|------|-----------|
| **Stone** (nurse.py) | Physical facts | Cannot hallucinate |
| **Dream** (asclepius_vectors.py) | Semantic patterns | Interprets meaning |
| **Anchor** (Canon + Enos) | Ground truth | Final authority |

---

*"The Stone speaks of what IS. The Dream speaks of what it MEANS. The Anchor speaks of what SHOULD BE."*
"""
    
    # Save Markdown
    md_path = NEXUS_DIR / "_ASCLEPIUS_EXAMEN.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    log(f"Report saved to {md_path}")
    
    return json_path, md_path


def check_healing_mode() -> bool:
    """
    Quick check if system is in healing mode.
    Used by daemon to decide whether to allow new content.
    """
    report_path = NEXUS_DIR / "asclepius_report.json"
    
    if not report_path.exists():
        return False
    
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        return report.get("healing_mode", False)
    except:
        return False


def get_vitality_score() -> float:
    """
    Quick check of current vitality score.
    Used by daemon for fever checks.
    """
    report_path = NEXUS_DIR / "asclepius_report.json"
    
    if not report_path.exists():
        return 10.0  # Assume healthy if no report
    
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        return report.get("total_vitality", 10.0)
    except:
        return 10.0


def should_refuse_content() -> tuple[bool, str]:
    """
    Check if the system should refuse new content generation.
    Returns (should_refuse, reason).
    """
    if check_healing_mode():
        vitality = get_vitality_score()
        return True, f"SYSTEM FEVER: Vitality {vitality}/10. Healing mode active. Only repair commands accepted."
    return False, ""


# --- DAEMON INTEGRATION ---

def enhanced_fever_check() -> Dict[str, Any]:
    """
    Enhanced fever check for daemon integration.
    Replaces simple check_system_fever() with Asclepius awareness.
    """
    vitality = get_vitality_score()
    healing = check_healing_mode()
    
    if vitality < FEVER_CRITICAL_THRESHOLD:
        return {
            "level": "critical",
            "vitality": vitality,
            "healing_mode": healing,
            "action": "refuse_content",
            "message": f"🔴 CRITICAL FEVER: Vitality {vitality}/10 — Only repair operations allowed"
        }
    elif vitality < FEVER_HIGH_THRESHOLD:
        return {
            "level": "high",
            "vitality": vitality,
            "healing_mode": healing,
            "action": "alert_and_limit",
            "message": f"🟠 HIGH FEVER: Vitality {vitality}/10 — Limit complex operations"
        }
    elif vitality < FEVER_LOW_THRESHOLD:
        return {
            "level": "low",
            "vitality": vitality,
            "healing_mode": False,
            "action": "monitor",
            "message": f"🟡 LOW FEVER: Vitality {vitality}/10 — Monitor closely"
        }
    else:
        return {
            "level": "healthy",
            "vitality": vitality,
            "healing_mode": False,
            "action": "none",
            "message": f"🟢 HEALTHY: Vitality {vitality}/10"
        }


def main():
    """Main entry point — run the full Examen."""
    report = run_full_examen()
    save_examen_report(report)
    
    print("\n" + "=" * 60)
    print("ASCLEPIUS EXAMEN COMPLETE")
    print(f"Total Vitality: {report.total_vitality}/10")
    print(f"Fever Level: {report.fever_level}")
    print(f"Healing Mode: {'ACTIVE' if report.healing_mode else 'Inactive'}")
    print(f"\nOutputs:")
    print(f"  - {NEXUS_DIR / 'asclepius_report.json'}")
    print(f"  - {NEXUS_DIR / '_ASCLEPIUS_EXAMEN.md'}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()
