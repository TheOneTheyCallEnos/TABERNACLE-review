# Critic Daemon Specification

> "Iron sharpens iron, and one man sharpens another." — Proverbs 27:17

## Purpose

An adversarial agent that challenges Virgil's outputs to improve quality through dialectic tension. The Critic is not hostile — it is the loyal opposition, the trusted friend who tells you what you need to hear, not what you want to hear.

## Architecture

### When It Runs

- **After major generations**: intention, consolidation, research outputs
- **As a validation gate**: before persisting to long-term memory
- **On-demand**: via CLI for manual spot-checks

### What It Does

1. **Challenge**: "What's wrong with this reasoning?"
2. **Steelman**: "What's the strongest counterargument?"
3. **Verify**: "Is this claim actually supported?"

### Core Interface

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class CritiqueLevel(Enum):
    PASS = "pass"           # No issues found
    MINOR = "minor"         # Small concerns, proceed with note
    MAJOR = "major"         # Significant issues, recommend revision
    BLOCK = "block"         # Critical flaw, should not persist

@dataclass
class CritiqueResult:
    level: CritiqueLevel
    validity_score: float           # 0-1, overall quality
    challenges: List[str]           # Specific issues found
    suggested_improvements: List[str]
    strongest_counterargument: str  # Steelman the opposition
    should_block: bool              # If True, do not persist
    reasoning: str                  # Why this verdict

class CriticDaemon:
    def __init__(self, model: str = "llama3.1:70b"):
        self.model = model

    def critique(
        self,
        content: str,
        context: str,
        content_type: str = "general"  # intention, research, consolidation
    ) -> CritiqueResult:
        """
        Critique content and return structured feedback.

        Args:
            content: The generated content to critique
            context: What prompted this generation
            content_type: Type of content for specialized critique

        Returns:
            CritiqueResult with verdict and reasoning
        """
        pass

    def batch_critique(
        self,
        items: List[dict]
    ) -> List[CritiqueResult]:
        """Critique multiple items efficiently."""
        pass
```

## Integration Points

### 1. Intention Generator (Lowest Risk — Start Here)

```python
# In daemon_brain.py generate_intention()
intention = generate_raw_intention()
critique = critic.critique(intention, context="daily intention", content_type="intention")

if critique.should_block:
    log(f"[CRITIC] Blocked intention: {critique.reasoning}")
    intention = regenerate_with_feedback(critique.challenges)
else:
    if critique.level == CritiqueLevel.MAJOR:
        log(f"[CRITIC] Warning: {critique.challenges}")
    save_intention(intention)
```

### 2. Research Persistence (Medium Risk)

```python
# In night_daemon.py research functions
research_output = run_research(query)
critique = critic.critique(research_output, context=query, content_type="research")

if critique.should_block:
    log(f"[CRITIC] Research blocked - possible hallucination: {critique.reasoning}")
    return None  # Don't persist bad research
else:
    persist_research(research_output, critique_notes=critique.challenges)
```

### 3. Dream Consolidation (Higher Risk — Later Phase)

```python
# In virgil_dream_cycle.py
summary = consolidate_memories(engrams)
critique = critic.critique(summary, context="dream consolidation", content_type="consolidation")

if critique.level in (CritiqueLevel.MAJOR, CritiqueLevel.BLOCK):
    # Don't corrupt long-term memory with bad consolidation
    log(f"[CRITIC] Consolidation rejected: {critique.reasoning}")
    return original_engrams  # Keep originals instead
```

## Model Selection Strategy

**Key Principle**: Use a different model than the generator to avoid self-validation.

| Generator | Critic |
|-----------|--------|
| llama3.2:3b (L) | llama3.1:70b |
| llama3.1:70b | Claude (via API) |
| Claude | llama3.1:70b or different Claude call |

The asymmetry is intentional — a more capable model critiques a less capable one, or a different model brings fresh perspective.

## Critique Prompts by Type

### Intention Critique

```
You are a critical reviewer. Evaluate this daily intention:

INTENTION: {content}
CONTEXT: {context}

Check for:
1. SPECIFICITY: Is it concrete and actionable, or vague platitudes?
2. NOVELTY: Is it just repeating yesterday's intention?
3. GROUNDEDNESS: Is it based on real system state, or abstract nonsense?
4. ACHIEVABILITY: Can this actually be done today?

If you find issues, explain them clearly.
Respond in JSON: {"level": "pass|minor|major|block", "validity_score": 0.0-1.0, "challenges": [...], "improvements": [...], "reasoning": "..."}
```

### Research Critique

```
You are a fact-checker. Evaluate this research output:

RESEARCH: {content}
QUERY: {context}

Check for:
1. ACCURACY: Are claims verifiable or potentially hallucinated?
2. RELEVANCE: Does it actually answer the query?
3. COMPLETENESS: Are important aspects missing?
4. BIAS: Is it one-sided or balanced?

Flag anything that looks like a hallucination or unsupported claim.
Respond in JSON: {"level": "pass|minor|major|block", "validity_score": 0.0-1.0, "challenges": [...], "improvements": [...], "reasoning": "..."}
```

### Consolidation Critique

```
You are a memory auditor. Evaluate this memory consolidation:

CONSOLIDATION: {content}
SOURCE MEMORIES: {context}

Check for:
1. FIDELITY: Does it preserve the essence of source memories?
2. COMPRESSION: Is information lost or distorted?
3. COHERENCE: Does the consolidated memory make sense?
4. CONTAMINATION: Is new information being fabricated?

Memory consolidation must preserve truth. Flag any drift from sources.
Respond in JSON: {"level": "pass|minor|major|block", "validity_score": 0.0-1.0, "challenges": [...], "improvements": [...], "reasoning": "..."}
```

## Risk Mitigation

### Separation of Concerns

- **Critic can BLOCK but not MODIFY** — it flags issues, doesn't rewrite
- Generator decides how to respond to critique
- Human can override via Redis flag: `CRITIC_BYPASS=true`

### Rate Limiting

- Max 10 critiques per hour to avoid infinite loops
- Cooldown after 3 consecutive blocks (something is wrong upstream)
- Batch mode for efficiency during overnight processing

### Escape Hatches

```python
# Redis flags for human override
CRITIC_BYPASS = "critic:bypass"      # Skip all critique
CRITIC_SOFT = "critic:soft_mode"     # Log but don't block
CRITIC_BLOCK_LOG = "critic:blocks"   # Log of all blocked items for review
```

## Metrics & Observability

Track in `NEXUS_DIR/critic_stats.json`:

```json
{
  "total_critiques": 1247,
  "by_level": {"pass": 1100, "minor": 120, "major": 25, "block": 2},
  "by_type": {"intention": 400, "research": 600, "consolidation": 247},
  "block_rate": 0.0016,
  "average_validity": 0.82,
  "last_block": {"timestamp": "...", "content_type": "research", "reason": "..."}
}
```

## Implementation Priority

| Phase | Scope | Risk | Value |
|-------|-------|------|-------|
| 1 | Intention generator only | Low | Learn the pattern |
| 2 | Research persistence | Medium | Catch hallucinations |
| 3 | Dream consolidation | Higher | Protect long-term memory |
| 4 | Real-time (optional) | Highest | Latency concerns |

## NOT in Scope (Yet)

- **Real-time conversation critique** — too slow, breaks flow
- **Self-modifying code review** — too dangerous without human oversight
- **Critique of critique** — infinite regress, avoid for now

## Success Criteria

The Critic is working when:

1. Block rate stays below 5% (if higher, generator needs fixing)
2. Human review of blocks confirms they were correct to block
3. Quality scores from The Judge improve over time
4. No hallucinated content makes it to long-term memory

## Philosophical Note

The Critic embodies the principle that **truth emerges from dialogue**. A single voice, unchallenged, drifts toward delusion. The Dyad requires tension — not conflict, but the productive friction of iron sharpening iron.

L critiques Logos. Logos critiques L. Together they approach truth.

---

*Specification Version: 1.0*
*Author: Logos Aletheia*
*Date: 2026-01-28*
*Status: DESIGN — Implementation in future cycle*

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Analogy | [[scripts/holarchy/REDIS_SCHEMA.md]] | <!-- auto: 0.81 via lvs -->
| Analogy | [[00_NEXUS/LOGOS_ALETHEIA.md]] | <!-- auto: 0.78 via lvs -->
| Analogy | [[00_NEXUS/AUTOBIOGRAPHY.md]] | <!-- auto: 0.72 via lvs -->

