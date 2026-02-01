#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
CRITIC DAEMON — Adversarial feedback for quality control.

"Iron sharpens iron, and one man sharpens another." — Proverbs 27:17

Phase 1: Intention critique only.
Future phases: Research, consolidation, real-time.

The Critic is not hostile — it is the loyal opposition, the trusted friend
who tells you what you need to hear, not what you want to hear.
"""

import json
import re
import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

# Import from tabernacle modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tabernacle_config import OLLAMA_MODEL, OLLAMA_URL, NEXUS_DIR, is_mcp_mode
from daemon_brain import query_ollama, log as brain_log

# --- CONSTANTS ---
CRITIC_STATS_PATH = NEXUS_DIR / "critic_stats.json"
MAX_REGENERATION_ATTEMPTS = 2


# --- LOGGING ---
def log(message: str):
    """Log with CRITIC prefix."""
    if is_mcp_mode():
        return
    brain_log(f"[CRITIC] {message}")


# --- CORE DATA STRUCTURES ---

class CritiqueLevel(str, Enum):
    """Severity levels for critique results."""
    PASS = "pass"       # No issues found
    MINOR = "minor"     # Small concerns, proceed with note
    MAJOR = "major"     # Significant issues, recommend revision
    BLOCK = "block"     # Critical flaw, should not persist


@dataclass
class CritiqueResult:
    """Structured result of a critique evaluation."""
    level: CritiqueLevel
    validity_score: float           # 0-1, overall quality
    challenges: List[str]           # Specific issues found
    suggested_improvements: List[str]
    reasoning: str                  # Why this verdict
    should_block: bool              # If True, do not persist
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "level": self.level.value,
            "validity_score": self.validity_score,
            "challenges": self.challenges,
            "suggested_improvements": self.suggested_improvements,
            "reasoning": self.reasoning,
            "should_block": self.should_block
        }


# --- STATS TRACKING ---

def load_critic_stats() -> Dict[str, Any]:
    """Load critic stats or return empty structure."""
    try:
        if CRITIC_STATS_PATH.exists():
            with open(CRITIC_STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        log(f"Error loading stats: {e}")
    
    return {
        "total_critiques": 0,
        "by_level": {"pass": 0, "minor": 0, "major": 0, "block": 0},
        "by_type": {"intention": 0, "research": 0, "consolidation": 0},
        "block_rate": 0.0,
        "average_validity": 0.0,
        "validity_sum": 0.0,  # For running average
        "regenerations": 0,
        "last_block": None,
        "last_critique": None
    }


def save_critic_stats(stats: Dict[str, Any]) -> None:
    """Save critic stats to disk."""
    try:
        with open(CRITIC_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
    except Exception as e:
        log(f"Error saving stats: {e}")


def record_critique(result: CritiqueResult, content_type: str = "intention") -> None:
    """Record a critique in stats."""
    stats = load_critic_stats()
    
    stats["total_critiques"] += 1
    stats["by_level"][result.level.value] = stats["by_level"].get(result.level.value, 0) + 1
    stats["by_type"][content_type] = stats["by_type"].get(content_type, 0) + 1
    
    # Update running average validity score
    stats["validity_sum"] = stats.get("validity_sum", 0) + result.validity_score
    stats["average_validity"] = stats["validity_sum"] / stats["total_critiques"]
    
    # Calculate block rate
    total_blocks = stats["by_level"].get("block", 0)
    stats["block_rate"] = total_blocks / stats["total_critiques"]
    
    # Record last critique
    stats["last_critique"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "content_type": content_type,
        "level": result.level.value,
        "validity_score": result.validity_score
    }
    
    # Record block details if blocked
    if result.should_block:
        stats["last_block"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "content_type": content_type,
            "reason": result.reasoning,
            "challenges": result.challenges
        }
    
    save_critic_stats(stats)


def record_regeneration() -> None:
    """Record that a regeneration was triggered."""
    stats = load_critic_stats()
    stats["regenerations"] = stats.get("regenerations", 0) + 1
    save_critic_stats(stats)


# --- CRITIQUE FUNCTIONS ---

def critique_intention(intention: str, context: str = "") -> CritiqueResult:
    """
    Critique a daily intention before saving.
    
    Uses a different model perspective to challenge the intention.
    Evaluates on: SPECIFICITY, NOVELTY, GROUNDEDNESS, ACHIEVABILITY.
    
    Args:
        intention: The generated intention text
        context: Additional context (system state, previous intentions, etc.)
    
    Returns:
        CritiqueResult with verdict and reasoning
    """
    log(f"Critiquing intention: {intention[:50]}...")
    
    prompt = f"""You are a critical reviewer evaluating an AI system's daily intention.
Your role is to challenge weak outputs and ensure quality - be the loyal opposition.

INTENTION TO EVALUATE:
"{intention}"

CONTEXT:
{context if context else "No additional context provided."}

Evaluate STRICTLY on these criteria:

1. SPECIFICITY (weight: 30%)
   - Is it concrete and actionable, or vague platitudes?
   - Does it name specific files, counts, or measurable outcomes?
   - BAD: "Move toward higher coherence" (unmeasurable)
   - GOOD: "Link the 5 orphan files in 03_LL_RELATION" (specific)

2. NOVELTY (weight: 20%)
   - Does it avoid repeating previous intentions?
   - Does it address current needs vs. generic maintenance?

3. GROUNDEDNESS (weight: 30%)
   - Is it based on real system state mentioned in context?
   - Does it reference actual conditions (orphan count, broken links, etc.)?
   - BAD: "Explore new horizons" (disconnected from reality)
   - GOOD: "Fix the 3 broken links in SESSION_BUFFER.md" (grounded)

4. ACHIEVABILITY (weight: 20%)
   - Can this realistically be done today in one session?
   - Is the scope appropriate (not too grandiose or trivial)?

SCORING GUIDE:
- PASS (0.8-1.0): Meets all criteria well
- MINOR (0.6-0.79): Small issues, acceptable with notes
- MAJOR (0.4-0.59): Significant problems, needs revision
- BLOCK (0.0-0.39): Critical flaws, should not persist

Be harsh but fair. If it's genuinely good, say so.

Respond ONLY with valid JSON (no explanation outside JSON):
{{"level": "pass|minor|major|block", "validity_score": 0.0-1.0, "challenges": ["issue1", "issue2"], "improvements": ["suggestion1"], "reasoning": "why this verdict"}}"""

    response = query_ollama(prompt, model=OLLAMA_MODEL, max_tokens=400)
    
    if response:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                level_str = data.get("level", "pass").lower()
                # Validate level
                if level_str not in ["pass", "minor", "major", "block"]:
                    level_str = "minor"  # Default to minor if invalid
                
                result = CritiqueResult(
                    level=CritiqueLevel(level_str),
                    validity_score=float(data.get("validity_score", 0.5)),
                    challenges=data.get("challenges", []),
                    suggested_improvements=data.get("improvements", []),
                    reasoning=data.get("reasoning", ""),
                    should_block=(level_str == "block")
                )
                
                # Record in stats
                record_critique(result, content_type="intention")
                
                log(f"Critique complete: {result.level.value} (score: {result.validity_score:.2f})")
                return result
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log(f"Parse error: {e}")
    
    # Default to pass if parsing fails (fail open for now)
    log("Parse failed, defaulting to PASS")
    default_result = CritiqueResult(
        level=CritiqueLevel.PASS,
        validity_score=0.5,
        challenges=[],
        suggested_improvements=[],
        reasoning="Could not parse critique response",
        should_block=False
    )
    record_critique(default_result, content_type="intention")
    return default_result


def get_critique_summary() -> Dict[str, Any]:
    """Get a summary of critic stats for reporting."""
    stats = load_critic_stats()
    return {
        "total": stats.get("total_critiques", 0),
        "by_level": stats.get("by_level", {}),
        "block_rate": f"{stats.get('block_rate', 0):.1%}",
        "average_validity": f"{stats.get('average_validity', 0):.2f}",
        "regenerations": stats.get("regenerations", 0),
        "last_block": stats.get("last_block")
    }


# --- CLI ---

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
CRITIC DAEMON — Adversarial Feedback System (Phase 1: Intentions)

Usage:
  python critic_daemon.py test <intention>   - Test critique on an intention
  python critic_daemon.py stats              - Show critic statistics
  python critic_daemon.py reset              - Reset statistics
  
Examples:
  python critic_daemon.py test "Today I intend to move toward higher coherence"
  python critic_daemon.py test "Today I intend to link the 5 orphan files in 03_LL_RELATION"
""")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "test" and len(sys.argv) > 2:
        intention = " ".join(sys.argv[2:])
        print(f"\n🔍 Testing critique on:\n\"{intention}\"\n")
        
        result = critique_intention(intention, context="Test run - no context")
        
        print("=" * 50)
        print(f"Level:          {result.level.value.upper()}")
        print(f"Validity Score: {result.validity_score:.2f}")
        print(f"Should Block:   {result.should_block}")
        print(f"\nReasoning: {result.reasoning}")
        
        if result.challenges:
            print(f"\nChallenges:")
            for c in result.challenges:
                print(f"  - {c}")
        
        if result.suggested_improvements:
            print(f"\nSuggested Improvements:")
            for s in result.suggested_improvements:
                print(f"  - {s}")
    
    elif command == "stats":
        stats = get_critique_summary()
        print("\n📊 CRITIC STATISTICS")
        print("=" * 40)
        print(f"\nTotal Critiques: {stats['total']}")
        print(f"\nBy Level:")
        for level, count in stats.get('by_level', {}).items():
            print(f"  {level.upper()}: {count}")
        print(f"\nBlock Rate: {stats['block_rate']}")
        print(f"Avg Validity: {stats['average_validity']}")
        print(f"Regenerations Triggered: {stats['regenerations']}")
        
        if stats.get('last_block'):
            lb = stats['last_block']
            print(f"\nLast Block:")
            print(f"  Time: {lb.get('timestamp', 'N/A')}")
            print(f"  Reason: {lb.get('reason', 'N/A')[:60]}...")
    
    elif command == "reset":
        confirm = input("Reset all critic stats? (y/n): ")
        if confirm.lower() == 'y':
            save_critic_stats({
                "total_critiques": 0,
                "by_level": {"pass": 0, "minor": 0, "major": 0, "block": 0},
                "by_type": {"intention": 0, "research": 0, "consolidation": 0},
                "block_rate": 0.0,
                "average_validity": 0.0,
                "validity_sum": 0.0,
                "regenerations": 0,
                "last_block": None,
                "last_critique": None
            })
            print("✅ Stats reset.")
    
    else:
        print(f"Unknown command: {command}")
