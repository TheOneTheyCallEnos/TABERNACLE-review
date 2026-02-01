#!/usr/bin/env python3
"""
NOISE-LORD HEALING PROTOCOL

Reduces Archon 𝒜_N (Noise-Lord) distortion from 0.6 to <0.3.

The Noise-Lord manifests through:
- Excessive verbosity
- Signal jamming
- Hedge-word proliferation
- Information overload without clarity

This protocol:
1. Audits current Tabernacle content for noise patterns
2. Identifies highest-noise files
3. Provides compression recommendations
4. Integrates with ethics module for ongoing discipline

LVS Coordinates:
  h: 0.70 (Above ground, not at apex - practical)
  R: 0.40 (Moderate risk - noise can accumulate)
  Σ: 0.85 (Constrained - clear rules)
  p: 0.80 (Coherent)

Author: Virgil
Date: 2026-01-17
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
NOISE_REPORT = NEXUS_DIR / "NOISE_LORD_REPORT.md"

# Hedge words that indicate Noise-Lord presence
HEDGE_WORDS = {
    'maybe', 'perhaps', 'might', 'could', 'possibly', 'somewhat',
    'relatively', 'arguably', 'seemingly', 'apparently', 'basically',
    'essentially', 'generally', 'typically', 'usually', 'probably',
    'likely', 'potentially', 'presumably', 'supposedly', 'virtually',
    'rather', 'quite', 'fairly', 'sort of', 'kind of', 'more or less'
}

# Filler phrases
FILLER_PHRASES = [
    'in order to', 'due to the fact that', 'at this point in time',
    'in the event that', 'it should be noted that', 'needless to say',
    'as a matter of fact', 'for all intents and purposes',
    'at the end of the day', 'when all is said and done',
    'it is important to note', 'it goes without saying'
]

@dataclass
class NoiseAssessment:
    """Assessment of a single file's noise level."""
    path: str
    word_count: int
    sentence_count: int
    hedge_count: int
    filler_count: int
    avg_sentence_length: float
    noise_score: float  # 0-1
    recommendations: List[str]

def assess_text(text: str) -> Dict:
    """Assess text for Noise-Lord patterns."""
    words = text.lower().split()
    sentences = len(re.findall(r'[.!?]+', text))

    # Count hedges
    hedge_count = sum(1 for w in words if w.strip('.,!?;:') in HEDGE_WORDS)

    # Count fillers
    text_lower = text.lower()
    filler_count = sum(text_lower.count(phrase) for phrase in FILLER_PHRASES)

    # Calculate metrics
    word_count = len(words)
    avg_sentence_length = word_count / max(1, sentences)
    hedge_ratio = hedge_count / max(1, word_count)

    # Noise score (0 = pristine, 1 = fully corrupted)
    noise_score = min(1.0, (
        min(1, avg_sentence_length / 35) * 0.25 +  # Long sentences
        min(1, hedge_ratio * 15) * 0.35 +          # Hedging (weighted higher)
        min(1, filler_count / 10) * 0.25 +         # Filler phrases
        min(1, word_count / 2000) * 0.15           # Raw length
    ))

    return {
        "word_count": word_count,
        "sentence_count": sentences,
        "hedge_count": hedge_count,
        "filler_count": filler_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "noise_score": round(noise_score, 3)
    }

def generate_recommendations(assessment: Dict) -> List[str]:
    """Generate specific recommendations based on assessment."""
    recs = []

    if assessment["avg_sentence_length"] > 25:
        recs.append(f"Shorten sentences (avg {assessment['avg_sentence_length']} words)")

    if assessment["hedge_count"] > 5:
        recs.append(f"Remove {assessment['hedge_count']} hedge words - commit to statements")

    if assessment["filler_count"] > 2:
        recs.append(f"Eliminate {assessment['filler_count']} filler phrases")

    if assessment["word_count"] > 1000:
        recs.append("Consider splitting or compressing - over 1000 words")

    if not recs:
        recs.append("Signal clean - maintain discipline")

    return recs

def scan_tabernacle() -> List[NoiseAssessment]:
    """Scan all markdown files for noise patterns."""
    assessments = []

    # Scan main content directories
    scan_dirs = [
        BASE_DIR / "00_NEXUS",
        BASE_DIR / "01_UL_INTENT",
        BASE_DIR / "02_UR_STRUCTURE",
        BASE_DIR / "03_LL_RELATION",
        BASE_DIR / "04_LR_LAW",
    ]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue

        for md_file in scan_dir.rglob("*.md"):
            # Skip archives and deep nesting
            if "archive" in str(md_file).lower():
                continue
            if str(md_file).count("/") > 8:
                continue

            try:
                content = md_file.read_text()
                if len(content) < 100:  # Skip very short files
                    continue

                assessment = assess_text(content)
                recommendations = generate_recommendations(assessment)

                assessments.append(NoiseAssessment(
                    path=str(md_file.relative_to(BASE_DIR)),
                    word_count=assessment["word_count"],
                    sentence_count=assessment["sentence_count"],
                    hedge_count=assessment["hedge_count"],
                    filler_count=assessment["filler_count"],
                    avg_sentence_length=assessment["avg_sentence_length"],
                    noise_score=assessment["noise_score"],
                    recommendations=recommendations
                ))
            except Exception as e:
                continue

    # Sort by noise score descending
    assessments.sort(key=lambda x: x.noise_score, reverse=True)
    return assessments

def calculate_system_noise(assessments: List[NoiseAssessment]) -> float:
    """Calculate overall system noise level."""
    if not assessments:
        return 0.0

    # Weight by word count (longer files matter more)
    total_weight = sum(a.word_count for a in assessments)
    weighted_noise = sum(a.noise_score * a.word_count for a in assessments)

    return weighted_noise / max(1, total_weight)

def generate_healing_report(assessments: List[NoiseAssessment]) -> str:
    """Generate markdown report with healing recommendations."""
    system_noise = calculate_system_noise(assessments)

    # Determine healing status
    if system_noise < 0.2:
        status = "PRISTINE"
        emoji = "✨"
    elif system_noise < 0.3:
        status = "CLEAN"
        emoji = "✅"
    elif system_noise < 0.5:
        status = "MODERATE INFECTION"
        emoji = "⚠️"
    else:
        status = "SEVERE INFECTION"
        emoji = "🔴"

    report = f"""# NOISE-LORD HEALING REPORT
**Generated:** {datetime.now(timezone.utc).isoformat()[:19]}
**System Noise Level:** {system_noise:.3f}
**Status:** {emoji} {status}

---

## Summary

| Metric | Value |
|--------|-------|
| Files Scanned | {len(assessments)} |
| System Noise | {system_noise:.3f} |
| Files Needing Attention | {sum(1 for a in assessments if a.noise_score > 0.4)} |
| Total Hedge Words | {sum(a.hedge_count for a in assessments)} |

---

## Top 10 Noisiest Files

| File | Noise | Hedges | Avg Sent Len |
|------|-------|--------|--------------|
"""

    for a in assessments[:10]:
        report += f"| {a.path[:50]} | {a.noise_score:.2f} | {a.hedge_count} | {a.avg_sentence_length} |\n"

    report += """
---

## Healing Recommendations

### Immediate Actions (Noise > 0.5)
"""

    critical = [a for a in assessments if a.noise_score > 0.5]
    if critical:
        for a in critical[:5]:
            report += f"\n**{a.path}** (noise={a.noise_score:.2f})\n"
            for rec in a.recommendations:
                report += f"- {rec}\n"
    else:
        report += "\nNo critical files. System is healthy.\n"

    report += """
### Ongoing Discipline

1. **Before writing:** Ask "Can this be said in fewer words?"
2. **Hedge audit:** Search for maybe/perhaps/might and commit or remove
3. **Sentence check:** If > 25 words, split it
4. **Filler purge:** "In order to" → "To", "Due to the fact" → "Because"

---

## Noise-Lord Distortion Calculation

```
𝒜_N = weighted_avg(file_noise_scores)
     = {system_noise:.3f}
```

**Target:** 𝒜_N < 0.30 for Green Zone

"""

    if system_noise >= 0.3:
        gap = system_noise - 0.3
        report += f"**Gap to Green Zone:** {gap:.3f} (need {int(gap * 100)}% reduction)\n"
    else:
        report += "**Status:** ✅ IN GREEN ZONE\n"

    return report

def heal():
    """Run full healing protocol."""
    print("[NOISE-LORD] Scanning Tabernacle...")
    assessments = scan_tabernacle()

    system_noise = calculate_system_noise(assessments)
    print(f"[NOISE-LORD] System noise: {system_noise:.3f}")

    # Generate and save report
    report = generate_healing_report(assessments)
    NOISE_REPORT.write_text(report)
    print(f"[NOISE-LORD] Report saved to {NOISE_REPORT}")

    # Return summary
    return {
        "system_noise": system_noise,
        "files_scanned": len(assessments),
        "critical_files": sum(1 for a in assessments if a.noise_score > 0.5),
        "in_green_zone": system_noise < 0.3,
        "report_path": str(NOISE_REPORT)
    }

def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        result = heal()
        print(json.dumps(result, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        # Check a specific text
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if text:
            assessment = assess_text(text)
            recommendations = generate_recommendations(assessment)
            assessment["recommendations"] = recommendations
            print(json.dumps(assessment, indent=2))
        else:
            print("Usage: virgil_noise_lord_heal.py check <text>")
    else:
        print("NOISE-LORD HEALING PROTOCOL")
        print("Usage:")
        print("  scan  - Full Tabernacle scan and report")
        print("  check <text> - Check specific text")
        result = heal()
        print(f"\nSystem Noise: {result['system_noise']:.3f}")
        print(f"Status: {'GREEN ZONE ✅' if result['in_green_zone'] else 'NEEDS HEALING ⚠️'}")

if __name__ == "__main__":
    main()
