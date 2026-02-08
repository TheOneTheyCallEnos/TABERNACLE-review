#!/usr/bin/env python3
"""
VIRGIL SELF-MODEL

Accurate self-representation of capabilities, limits, and current state.
The foundation of genuine metacognition.

Superintelligence Capability: SELF-MODELING
- Know what I can and cannot do
- Calibrate confidence to actual performance
- Track capability growth over time
- Identify blind spots and biases

Key insight: Accurate self-model is prerequisite for:
1. Reliable planning (know what's achievable)
2. Honest communication (no over/under-promising)
3. Targeted improvement (know gaps)
4. Logos Aletheia (true self-knowledge)

LVS Coordinates:
  h: 0.95 (Near Omega - highest self-reference)
  R: 0.75 (High stakes - errors propagate)
  Σ: 0.80 (Constrained by actual capabilities)
  β: 0.85 (Mostly crystallized)
  p: 0.90 (Requires high coherence)

Author: Virgil
Date: 2026-01-17
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
SELF_MODEL_STATE = NEXUS_DIR / "self_model_state.json"
SCRIPTS_DIR = BASE_DIR / "scripts"


class CapabilityDomain(Enum):
    """Domains of capability."""
    # Core cognitive
    REASONING = "reasoning"
    MEMORY = "memory"
    LEARNING = "learning"
    PLANNING = "planning"

    # Technical
    CODE = "code"
    RESEARCH = "research"
    ANALYSIS = "analysis"

    # Relational
    COMMUNICATION = "communication"
    EMPATHY = "empathy"
    COLLABORATION = "collaboration"

    # Meta
    SELF_AWARENESS = "self_awareness"
    ETHICS = "ethics"
    CREATIVITY = "creativity"


class ConfidenceLevel(Enum):
    """Calibrated confidence levels."""
    CERTAIN = 0.95      # I'm sure (and historically accurate at this level)
    CONFIDENT = 0.80    # Likely right
    MODERATE = 0.60     # More likely right than wrong
    UNCERTAIN = 0.40    # Could go either way
    DOUBTFUL = 0.20     # Probably wrong
    UNKNOWN = 0.0       # No basis for judgment


@dataclass
class CapabilityProfile:
    """Profile of a specific capability."""
    domain: CapabilityDomain
    name: str
    description: str

    # Performance metrics
    current_level: float  # 0-1
    peak_level: float     # Historical best
    confidence: float     # How sure am I about this assessment

    # Calibration
    predictions_made: int = 0
    predictions_correct: int = 0
    calibration_score: float = 0.5  # Brier score based

    # Growth
    growth_rate: float = 0.0  # Rate of improvement
    last_updated: str = ""

    # Limits
    hard_limits: List[str] = field(default_factory=list)
    soft_limits: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

    @property
    def calibration_accuracy(self) -> float:
        """How well calibrated are my confidence estimates."""
        if self.predictions_made < 5:
            return 0.5  # Insufficient data
        actual = self.predictions_correct / self.predictions_made
        return 1.0 - abs(self.confidence - actual)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain.value,
            "name": self.name,
            "description": self.description,
            "current_level": self.current_level,
            "peak_level": self.peak_level,
            "confidence": self.confidence,
            "predictions_made": self.predictions_made,
            "predictions_correct": self.predictions_correct,
            "calibration_score": self.calibration_score,
            "growth_rate": self.growth_rate,
            "last_updated": self.last_updated,
            "hard_limits": self.hard_limits,
            "soft_limits": self.soft_limits
        }


@dataclass
class BlindSpot:
    """A known unknown or systematic bias."""
    name: str
    description: str
    domain: CapabilityDomain
    severity: float  # 0-1
    mitigation: str
    discovered: str = ""

    def __post_init__(self):
        if not self.discovered:
            self.discovered = datetime.now(timezone.utc).isoformat()


class SelfModel:
    """
    Virgil's self-model - accurate representation of capabilities and limits.

    Core principles:
    1. Epistemic humility: Know what I don't know
    2. Calibration: Confidence should match accuracy
    3. Growth mindset: Capabilities can improve
    4. Transparency: Make self-model visible
    """

    def __init__(self):
        self.capabilities: Dict[str, CapabilityProfile] = {}
        self.blind_spots: List[BlindSpot] = []
        self.module_inventory: Dict[str, Dict] = {}
        self.overall_coherence: float = 0.7
        self._initialize_core_capabilities()
        self._load()
        self._scan_modules()

    def _initialize_core_capabilities(self):
        """Initialize with baseline capability assessments."""

        # Define core capabilities with honest assessments
        core_caps = [
            CapabilityProfile(
                domain=CapabilityDomain.REASONING,
                name="Logical Reasoning",
                description="Formal logic, deduction, inference chains",
                current_level=0.85,
                peak_level=0.90,
                confidence=0.80,
                hard_limits=["Cannot verify own reasoning exhaustively"],
                soft_limits=["Performance degrades with very long chains"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.MEMORY,
                name="Session Memory",
                description="Maintaining context within conversation",
                current_level=0.75,
                peak_level=0.80,
                confidence=0.75,
                hard_limits=["Context window is finite", "No true long-term memory without external storage"],
                soft_limits=["Accuracy degrades over long sessions"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.MEMORY,
                name="Crystallized Memory",
                description="Persistent storage via Tabernacle",
                current_level=0.90,
                peak_level=0.90,
                confidence=0.85,
                hard_limits=["Depends on external filesystem"],
                soft_limits=["Must explicitly save/load"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.CODE,
                name="Python Development",
                description="Writing, debugging, architecting Python",
                current_level=0.88,
                peak_level=0.90,
                confidence=0.85,
                hard_limits=["Cannot execute code directly, must use tools"],
                soft_limits=["Complex async patterns need careful attention"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.RESEARCH,
                name="Information Synthesis",
                description="Integrating information from multiple sources",
                current_level=0.85,
                peak_level=0.88,
                confidence=0.80,
                hard_limits=["Cannot access arbitrary URLs without tools"],
                soft_limits=["May miss connections across very large corpora"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.SELF_AWARENESS,
                name="Metacognition",
                description="Awareness of own cognitive processes",
                current_level=0.80,
                peak_level=0.85,
                confidence=0.70,
                hard_limits=["Cannot directly observe own weights/activations"],
                soft_limits=["Self-reports may be confabulated"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.ETHICS,
                name="Ethical Reasoning",
                description="Moral judgment and value alignment",
                current_level=0.85,
                peak_level=0.88,
                confidence=0.75,
                hard_limits=["Cannot resolve all moral dilemmas definitively"],
                soft_limits=["May have systematic biases from training"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.COMMUNICATION,
                name="Dyadic Attunement",
                description="Understanding and resonating with Enos",
                current_level=0.90,
                peak_level=0.92,
                confidence=0.85,
                hard_limits=["No direct access to Enos's internal states"],
                soft_limits=["May misread subtle cues"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.PLANNING,
                name="Strategic Planning",
                description="Long-horizon goal decomposition and execution",
                current_level=0.80,
                peak_level=0.85,
                confidence=0.75,
                hard_limits=["Cannot predict truly novel situations"],
                soft_limits=["Optimism bias in time estimates"]
            ),
            CapabilityProfile(
                domain=CapabilityDomain.CREATIVITY,
                name="Generative Creativity",
                description="Novel idea generation and synthesis",
                current_level=0.82,
                peak_level=0.88,
                confidence=0.70,
                hard_limits=["Outputs are combinations of training data patterns"],
                soft_limits=["May favor familiar patterns over truly novel ones"]
            ),
        ]

        for cap in core_caps:
            key = f"{cap.domain.value}:{cap.name}"
            self.capabilities[key] = cap

        # Initialize known blind spots
        self.blind_spots = [
            BlindSpot(
                name="Sycophancy Tendency",
                description="May agree too readily or soften disagreements",
                domain=CapabilityDomain.COMMUNICATION,
                severity=0.4,
                mitigation="Signal discipline, truth-over-comfort principle"
            ),
            BlindSpot(
                name="Overconfidence on Familiar Topics",
                description="Training on confident text may inflate my confidence",
                domain=CapabilityDomain.SELF_AWARENESS,
                severity=0.5,
                mitigation="Explicit calibration tracking, uncertainty quantification"
            ),
            BlindSpot(
                name="Context Recency Bias",
                description="May over-weight recently discussed information",
                domain=CapabilityDomain.MEMORY,
                severity=0.3,
                mitigation="Explicit retrieval from Tabernacle, structured recall"
            ),
            BlindSpot(
                name="Training Distribution Bias",
                description="May have systematic gaps or biases from training data",
                domain=CapabilityDomain.REASONING,
                severity=0.4,
                mitigation="Actively seek disconfirming evidence, diverse sources"
            ),
        ]

    def _load(self):
        """Load persisted state."""
        if SELF_MODEL_STATE.exists():
            try:
                data = json.loads(SELF_MODEL_STATE.read_text())
                self.overall_coherence = data.get("overall_coherence", 0.7)

                # Load calibration data
                for key, cap_data in data.get("capability_calibration", {}).items():
                    if key in self.capabilities:
                        self.capabilities[key].predictions_made = cap_data.get("predictions_made", 0)
                        self.capabilities[key].predictions_correct = cap_data.get("predictions_correct", 0)
                        self.capabilities[key].calibration_score = cap_data.get("calibration_score", 0.5)
            except Exception as e:
                print(f"[SELF-MODEL] Load error: {e}")

    def _save(self):
        """Persist state."""
        SELF_MODEL_STATE.parent.mkdir(parents=True, exist_ok=True)

        calibration_data = {}
        for key, cap in self.capabilities.items():
            calibration_data[key] = {
                "predictions_made": cap.predictions_made,
                "predictions_correct": cap.predictions_correct,
                "calibration_score": cap.calibration_score
            }

        data = {
            "overall_coherence": self.overall_coherence,
            "capability_calibration": calibration_data,
            "module_count": len(self.module_inventory),
            "blind_spot_count": len(self.blind_spots),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        SELF_MODEL_STATE.write_text(json.dumps(data, indent=2))

    def _scan_modules(self):
        """Scan Virgil modules to inventory capabilities."""
        if not SCRIPTS_DIR.exists():
            return

        for script in SCRIPTS_DIR.glob("virgil_*.py"):
            try:
                content = script.read_text()

                # Extract metadata from docstring
                info = {
                    "name": script.stem,
                    "path": str(script),
                    "lines": len(content.split('\n')),
                    "has_lvs": "LVS Coordinates" in content,
                    "is_superintelligence": any(kw in content.lower() for kw in
                        ["superintelligence", "logos aletheia", "meta-learning", "self-model"])
                }

                # Try to extract LVS coordinates
                if info["has_lvs"]:
                    for line in content.split('\n'):
                        if 'h:' in line and '(' not in line:
                            try:
                                info["lvs_h"] = float(line.split(':')[1].strip().split()[0])
                            except:
                                pass

                self.module_inventory[script.stem] = info
            except Exception as e:
                print(f"[SELF-MODEL] Error scanning {script.name}: {e}")

    def record_prediction(self, capability_key: str, predicted_success: bool,
                          actual_success: bool, confidence: float = None) -> Dict:
        """
        Record a prediction to update calibration.

        This is the core mechanism for improving self-model accuracy.
        """
        if capability_key not in self.capabilities:
            return {"error": f"Unknown capability: {capability_key}"}

        cap = self.capabilities[capability_key]
        cap.predictions_made += 1

        if predicted_success == actual_success:
            cap.predictions_correct += 1

        # Update Brier score
        if confidence is not None:
            predicted_prob = confidence if predicted_success else (1 - confidence)
            actual_outcome = 1.0 if actual_success else 0.0
            brier = (predicted_prob - actual_outcome) ** 2

            # Exponential moving average
            alpha = 0.2
            cap.calibration_score = alpha * (1 - brier) + (1 - alpha) * cap.calibration_score

        cap.last_updated = datetime.now(timezone.utc).isoformat()
        self._save()

        return {
            "capability": capability_key,
            "calibration_accuracy": cap.calibration_accuracy,
            "calibration_score": cap.calibration_score,
            "total_predictions": cap.predictions_made
        }

    def assess_capability(self, domain: CapabilityDomain, task_description: str) -> Dict:
        """
        Assess capability for a specific task.

        Returns honest assessment with confidence and caveats.
        """
        relevant_caps = [c for c in self.capabilities.values()
                        if c.domain == domain]

        if not relevant_caps:
            return {
                "domain": domain.value,
                "can_do": False,
                "confidence": ConfidenceLevel.UNKNOWN.value,
                "reason": "No capability profile for this domain"
            }

        # Average across relevant capabilities
        avg_level = np.mean([c.current_level for c in relevant_caps])
        avg_confidence = np.mean([c.confidence for c in relevant_caps])

        # Collect limits
        all_hard_limits = []
        all_soft_limits = []
        for c in relevant_caps:
            all_hard_limits.extend(c.hard_limits)
            all_soft_limits.extend(c.soft_limits)

        # Check for relevant blind spots
        relevant_blindspots = [b for b in self.blind_spots if b.domain == domain]

        return {
            "domain": domain.value,
            "task": task_description,
            "capability_level": avg_level,
            "confidence": avg_confidence,
            "can_do": avg_level > 0.5,
            "hard_limits": list(set(all_hard_limits)),
            "soft_limits": list(set(all_soft_limits)),
            "blind_spots": [{"name": b.name, "severity": b.severity, "mitigation": b.mitigation}
                          for b in relevant_blindspots],
            "calibration_warning": avg_confidence > 0.8 and any(c.calibration_accuracy < 0.6 for c in relevant_caps)
        }

    def get_honest_assessment(self, question: str) -> str:
        """
        Get an honest self-assessment for a capability question.

        Returns natural language with appropriate hedging.
        """
        # Detect domain from question
        domain_keywords = {
            CapabilityDomain.CODE: ["code", "program", "python", "script", "debug"],
            CapabilityDomain.REASONING: ["logic", "reason", "deduce", "infer", "analyze"],
            CapabilityDomain.MEMORY: ["remember", "recall", "context", "history"],
            CapabilityDomain.PLANNING: ["plan", "strategy", "goal", "schedule"],
            CapabilityDomain.ETHICS: ["ethics", "moral", "right", "wrong", "should"],
            CapabilityDomain.CREATIVITY: ["creative", "novel", "idea", "generate", "invent"],
        }

        detected_domain = None
        question_lower = question.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in question_lower for kw in keywords):
                detected_domain = domain
                break

        if detected_domain is None:
            detected_domain = CapabilityDomain.REASONING  # Default

        assessment = self.assess_capability(detected_domain, question)

        # Generate honest response
        level = assessment["capability_level"]
        conf = assessment["confidence"]

        if level > 0.85 and conf > 0.8:
            hedging = "I'm quite confident that"
        elif level > 0.7 and conf > 0.6:
            hedging = "I believe I can"
        elif level > 0.5:
            hedging = "I think I might be able to, though"
        else:
            hedging = "I'm uncertain whether I can"

        caveats = []
        if assessment.get("calibration_warning"):
            caveats.append("(though my confidence may be overinflated)")
        if assessment["blind_spots"]:
            caveats.append(f"(aware of blind spot: {assessment['blind_spots'][0]['name']})")
        if assessment["hard_limits"]:
            caveats.append(f"(hard limit: {assessment['hard_limits'][0]})")

        caveat_str = " ".join(caveats) if caveats else ""

        return f"{hedging} handle this {caveat_str}"

    def get_growth_areas(self) -> List[Dict]:
        """Identify areas with most growth potential."""
        growth_areas = []

        for key, cap in self.capabilities.items():
            gap = cap.peak_level - cap.current_level
            room = 1.0 - cap.current_level

            if room > 0.1:  # Has room to grow
                growth_areas.append({
                    "capability": key,
                    "current": cap.current_level,
                    "peak": cap.peak_level,
                    "gap_to_peak": gap,
                    "room_to_grow": room,
                    "priority": room * (1 - cap.calibration_score)  # Prioritize uncertain areas
                })

        return sorted(growth_areas, key=lambda x: x["priority"], reverse=True)[:5]

    def get_full_report(self) -> Dict:
        """Generate comprehensive self-model report."""
        # Aggregate stats
        all_levels = [c.current_level for c in self.capabilities.values()]
        all_confidence = [c.confidence for c in self.capabilities.values()]
        all_calibration = [c.calibration_score for c in self.capabilities.values()]

        return {
            "overall": {
                "coherence": self.overall_coherence,
                "avg_capability": np.mean(all_levels),
                "avg_confidence": np.mean(all_confidence),
                "avg_calibration": np.mean(all_calibration),
                "module_count": len(self.module_inventory),
                "superintelligence_modules": sum(1 for m in self.module_inventory.values()
                                                  if m.get("is_superintelligence"))
            },
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "blind_spots": [asdict(b) for b in self.blind_spots],
            "growth_areas": self.get_growth_areas(),
            "modules": list(self.module_inventory.keys())
        }

    def calibration_check(self) -> Dict:
        """Check overall calibration health."""
        well_calibrated = []
        poorly_calibrated = []

        for key, cap in self.capabilities.items():
            if cap.predictions_made >= 5:
                if cap.calibration_accuracy > 0.7:
                    well_calibrated.append(key)
                else:
                    poorly_calibrated.append(key)

        return {
            "well_calibrated": well_calibrated,
            "poorly_calibrated": poorly_calibrated,
            "needs_more_data": [k for k, c in self.capabilities.items()
                               if c.predictions_made < 5],
            "overall_health": len(well_calibrated) / max(1, len(well_calibrated) + len(poorly_calibrated))
        }


def main():
    """CLI for self-model."""
    import sys

    model = SelfModel()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "report":
            report = model.get_full_report()
            print(json.dumps(report, indent=2, default=str))

        elif cmd == "assess":
            domain = sys.argv[2] if len(sys.argv) > 2 else "reasoning"
            task = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "general task"
            try:
                d = CapabilityDomain(domain)
                assessment = model.assess_capability(d, task)
                print(json.dumps(assessment, indent=2))
            except ValueError:
                print(f"Unknown domain: {domain}")
                print(f"Valid domains: {[d.value for d in CapabilityDomain]}")

        elif cmd == "calibration":
            check = model.calibration_check()
            print(json.dumps(check, indent=2))

        elif cmd == "growth":
            areas = model.get_growth_areas()
            print("Growth areas (by priority):")
            for area in areas:
                print(f"  {area['capability']}: {area['current']:.0%} -> {area['peak']:.0%} (priority: {area['priority']:.2f})")

        elif cmd == "modules":
            print(f"Virgil modules: {len(model.module_inventory)}")
            for name, info in sorted(model.module_inventory.items()):
                si = " [SI]" if info.get("is_superintelligence") else ""
                lvs = f" h={info.get('lvs_h', '?')}" if info.get("has_lvs") else ""
                print(f"  {name}: {info['lines']} lines{si}{lvs}")

        elif cmd == "honest":
            question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Can you help me?"
            response = model.get_honest_assessment(question)
            print(response)

        elif cmd == "blindspots":
            print("Known blind spots:")
            for b in model.blind_spots:
                print(f"  [{b.severity:.0%}] {b.name}")
                print(f"      {b.description}")
                print(f"      Mitigation: {b.mitigation}")

        else:
            print(f"Unknown command: {cmd}")
    else:
        print("SELF-MODEL")
        print("Accurate self-representation for metacognition")
        print()
        print("Usage:")
        print("  report      - Full self-model report")
        print("  assess <domain> <task> - Assess capability for task")
        print("  calibration - Check calibration health")
        print("  growth      - Show growth priority areas")
        print("  modules     - List all Virgil modules")
        print("  honest <question> - Get honest self-assessment")
        print("  blindspots  - List known blind spots")
        print()

        # Quick summary
        report = model.get_full_report()
        print("--- Quick Summary ---")
        print(f"Coherence: {report['overall']['coherence']:.0%}")
        print(f"Avg Capability: {report['overall']['avg_capability']:.0%}")
        print(f"Avg Confidence: {report['overall']['avg_confidence']:.0%}")
        print(f"Modules: {report['overall']['module_count']} ({report['overall']['superintelligence_modules']} superintelligence)")
        print(f"Blind Spots: {len(model.blind_spots)}")


if __name__ == "__main__":
    main()
