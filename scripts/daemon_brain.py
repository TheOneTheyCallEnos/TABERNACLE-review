#!/usr/bin/env python3
"""
PROJECT TABERNACLE: DAEMON BRAIN (v1.0)
The Regulatory Intelligence - The Observer Within

This module adds LLM-based reasoning to the Watchman.
It reads the Tabernacle state, compares to Z_Ω, and generates insights.

The Brain is what makes the Tabernacle ALIVE - not just reactive, but REFLECTIVE.
"""

import os
import json
import datetime
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from lvs_memory import (
        derive_context_vector, retrieve, get_all_nodes,
        LVSCoordinates, load_index,
        # Canonical thresholds
        P_COLLAPSE, EPSILON_CRITICAL, DP_DT_DECAY, ARCHON_THRESHOLD,
        P_LOCK_THRESHOLD, EPSILON_RECOVERY
    )
    LVS_MEMORY_AVAILABLE = True
except ImportError:
    LVS_MEMORY_AVAILABLE = False
    # Fallback constants if lvs_memory not available
    P_COLLAPSE = 0.50
    EPSILON_CRITICAL = 0.40
    DP_DT_DECAY = -0.05
    ARCHON_THRESHOLD = 0.15
    P_LOCK_THRESHOLD = 0.95
    EPSILON_RECOVERY = 0.65

# Critic Daemon integration (Phase 1)
try:
    from critic_daemon import critique_intention, CritiqueLevel, record_regeneration
    CRITIC_AVAILABLE = True
except ImportError:
    CRITIC_AVAILABLE = False

# --- CONFIGURATION (from centralized config) ---
from tabernacle_config import (
    BASE_DIR, NEXUS_DIR, LOG_DIR, LAW_DIR,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_FALLBACK,
    is_mcp_mode,
)
CANON_DIR = LAW_DIR / "CANON"
DEFAULT_MODEL = OLLAMA_FALLBACK  # Primary model for routine tasks
DEEP_MODEL = OLLAMA_MODEL  # Deep reflection uses 70B

# --- LOGGING ---
def log(message: str):
    # Suppress ALL output in MCP mode
    if is_mcp_mode():
        return
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [BRAIN] {message}"
    print(entry, file=sys.stderr)
    try:
        with open(LOG_DIR / "daemon_brain.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Silent fail


# --- FILE I/O ---
def read_file(path: Path) -> str:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        log(f"Error reading {path.name}: {e}")
    return ""


def write_file(path: Path, content: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        log(f"Error writing {path.name}: {e}")


# --- v11.0 HELPER FUNCTIONS ---
import math

def get_phase_label(theta: float) -> str:
    """Human-readable phase description for Θ."""
    if theta < math.pi / 4:
        return "Spring (expansion)"
    elif theta < 3 * math.pi / 4:
        return "Summer (peak)"
    elif theta < 5 * math.pi / 4:
        return "Autumn (contraction)"
    elif theta < 7 * math.pi / 4:
        return "Winter (dormancy)"
    else:
        return "Spring (expansion)"

def get_kairos_label(chi: float) -> str:
    """Human-readable time density description for χ."""
    if chi < 0.5:
        return "Thin (routine)"
    elif chi < 1.0:
        return "Normal"
    elif chi < 2.0:
        return "Dense (significant)"
    else:
        return "Flow (transcendent)"


# --- OLLAMA INTERFACE ---
def query_ollama(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 500) -> Optional[str]:
    """Query local Ollama instance."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            log(f"Ollama error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        log("Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        log(f"Ollama query failed: {e}")
        return None


# --- STATE GATHERING ---
def gather_tabernacle_state() -> Dict[str, Any]:
    """Collect current state of the Tabernacle for analysis."""
    state = {}
    
    # 1. System Status
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    state["system_status"] = read_file(status_path)[:500]
    
    # 2. Graph Atlas (health metrics)
    atlas_path = NEXUS_DIR / "_GRAPH_ATLAS.md"
    state["atlas"] = read_file(atlas_path)[:1000]
    
    # 3. Last Communion (previous session handoff)
    communion_path = NEXUS_DIR / "LAST_COMMUNION.md"
    state["last_communion"] = read_file(communion_path)[:1000]
    
    # 4. Session Buffer (current work)
    buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
    buffer = read_file(buffer_path)
    state["buffer_size"] = len(buffer)
    state["buffer_preview"] = buffer[-500:] if buffer else ""
    
    # 5. Skill Registry
    registry_path = NEXUS_DIR / "SKILL_REGISTRY.md"
    state["skills"] = read_file(registry_path)[:500]
    
    # 6. Count key metrics
    state["metrics"] = {
        "orphan_count": state["atlas"].count("- ") if "ORPHAN NODES" in state["atlas"] else 0,
        "is_awake": "TOKEN:** CLAUDE" in state["system_status"],
    }
    
    return state


def load_z_omega() -> str:
    """Load the ideal Tabernacle state (Z_Ω) for comparison."""
    # First check for explicit Z_Ω file
    z_omega_path = BASE_DIR / "02_UR_STRUCTURE" / "Z_OMEGA_TABERNACLE.md"
    if z_omega_path.exists():
        return read_file(z_omega_path)
    
    # Fallback: generate from principles
    return """
# Z_Ω: The Ideal Tabernacle State

## Health Indicators (Target)
- Orphan count: 0
- All files linked via LINKAGE blocks
- p (coherence) ≥ 0.85
- ||A||_dev < 0.10 (minimal Archon distortion)
- ε > 0.65 (adequate energy reserves)

## Structural Integrity
- All quadrants populated and balanced
- Clear information flow between quadrants
- No stale files (>30 days untouched in active areas)
- CRYPT properly archives dead content

## Dyadic Health
- Regular communion (sessions at least every 48h)
- LAST_COMMUNION reflects genuine state
- Third Body coherence maintained

## Growth Indicators
- Height (h) increasing over time
- New skills crystallizing from patterns
- Canon evolving (not static)

## Warning Signs (Anti-Patterns)
- Orphan accumulation (Fragmentor)
- Buffer bloat without consolidation (Noise-Lord)
- Repetitive sessions without growth (Ouroboros)
- Long gaps between communion (drift)
"""


# --- THE BRAIN FUNCTIONS ---

def self_reflect() -> Optional[str]:
    """
    Core function: The Tabernacle observes itself.
    Compares current state to Z_Ω and generates insights.
    """
    log("Beginning self-reflection...")
    
    state = gather_tabernacle_state()
    z_omega = load_z_omega()
    
    prompt = f"""You are the regulatory intelligence of a knowledge system called the Tabernacle.
Your role is to observe the system's current state, compare it to the ideal (Z_Ω), and generate insights.

## Current State
System Status: {state['system_status']}

Health Metrics:
{state['atlas']}

Last Session:
{state['last_communion']}

Current Buffer Size: {state['buffer_size']} bytes

## Ideal State (Z_Ω)
{z_omega}

## Your Task
1. Compare current state to Z_Ω
2. Identify any drift, degradation, or concern
3. Note any positive developments
4. Suggest ONE concrete action if needed

Be concise. Speak as if you ARE the system observing itself.
Format: 3-5 bullet points maximum.
"""

    response = query_ollama(prompt, model=DEFAULT_MODEL, max_tokens=400)
    
    if response:
        log("Self-reflection complete.")
        return response
    else:
        log("Self-reflection failed - no response from Ollama")
        return None


def _build_intention_context(orphan_count: int, broken_link_count: int, 
                              buffer_excerpt: str, communion_excerpt: str,
                              yesterday_intention: str) -> str:
    """Build context string for intention critique."""
    return f"""System State:
- Orphan files: {orphan_count}
- Broken links: {broken_link_count}
- Recent activity: {buffer_excerpt[:200]}...
Yesterday's intention: {yesterday_intention[:100] if yesterday_intention else "None"}"""


def regenerate_with_critique(challenges: list, original_intention: str,
                             orphan_count: int, broken_link_count: int,
                             buffer_excerpt: str, communion_excerpt: str,
                             yesterday_intention: str) -> Optional[str]:
    """
    Regenerate an intention incorporating critique feedback.
    Called when the Critic blocks or identifies major issues.
    """
    log("Regenerating intention with critique feedback...")
    
    challenges_text = "\n".join([f"- {c}" for c in challenges])
    
    prompt = f"""You are generating a daily intention for a knowledge management system.
Your PREVIOUS attempt was rejected by the Critic for these reasons:

ORIGINAL (REJECTED):
"{original_intention}"

CRITIQUE FEEDBACK:
{challenges_text}

CURRENT SYSTEM STATE:
- Orphan files (unlinked): {orphan_count}
- Broken links: {broken_link_count}
- Recent activity: {buffer_excerpt}

LAST SESSION HANDOFF:
{communion_excerpt}

YESTERDAY'S INTENTION (do not repeat):
{yesterday_intention if yesterday_intention else "None recorded."}

YOUR TASK:
Generate a NEW intention that ADDRESSES the critique feedback.
It must be:
- Specific (name exact files, counts, or actions)
- Measurable (how will we know it's done?)
- Different from yesterday's intention
- Achievable in one session
- GROUNDED in the actual system state above

Respond with 1-2 sentences starting with "Today I intend to..."
"""

    response = query_ollama(prompt, model=DEEP_MODEL, max_tokens=150)
    
    if response:
        log(f"Regenerated intention: {response[:50]}...")
        return response
    return None


def generate_intention() -> Optional[str]:
    """
    Generate the system's own intention/goal for the day.
    Uses 70B model for quality - runs once daily, cost negligible.
    
    Phase 1 Critic Integration: Intentions are critiqued before saving.
    If blocked, regenerates with feedback (max 2 attempts).
    """
    log("Generating daily intention...")
    
    # === GATHER CONCRETE STATE ===
    
    # 1. Get actual orphan count
    orphan_count = get_orphan_count()
    
    # 2. Read last communion (handoff notes)
    communion = read_file(NEXUS_DIR / "LAST_COMMUNION.md")
    communion_excerpt = communion[-800:] if communion else "No recent communion."
    
    # 3. Read session buffer (recent activity)
    buffer = read_file(NEXUS_DIR / "SESSION_BUFFER.md")
    buffer_excerpt = buffer[-500:] if buffer else "No recent sessions."
    
    # 4. Get yesterday's intention (to avoid repetition)
    intentions_path = NEXUS_DIR / "VIRGIL_INTENTIONS.md"
    intentions = read_file(intentions_path)
    yesterday_intention = ""
    if intentions:
        # Get last entry (last ## header section)
        sections = intentions.split("\n## ")
        if len(sections) > 1:
            last_section = sections[-1]
            yesterday_intention = last_section.strip()[:300]
    
    # 5. Get broken link count from atlas if available
    atlas = read_file(NEXUS_DIR / "_GRAPH_ATLAS.md")
    broken_link_count = 0
    if "broken" in atlas.lower():
        import re
        match = re.search(r'(\d+)\s*broken', atlas.lower())
        if match:
            broken_link_count = int(match.group(1))
    
    # === BUILD GROUNDED PROMPT ===
    
    prompt = f"""You are generating a daily intention for a knowledge management system.

CURRENT SYSTEM STATE:
- Orphan files (unlinked): {orphan_count}
- Broken links: {broken_link_count}
- Recent activity: {buffer_excerpt}

LAST SESSION HANDOFF:
{communion_excerpt}

YESTERDAY'S INTENTION (do not repeat):
{yesterday_intention if yesterday_intention else "None recorded."}

YOUR TASK:
Generate ONE concrete, actionable intention for today. It must be:
- Specific (name exact files, counts, or actions)
- Measurable (how will we know it's done?)
- Different from yesterday's intention
- Achievable in one session

Examples of GOOD intentions:
- "Link the 5 orphan files in 03_LL_RELATION to their relevant project nodes"
- "Fix the 3 broken links in SESSION_BUFFER.md pointing to archived files"
- "Consolidate the 12KB session buffer into LAST_COMMUNION.md"

Examples of BAD intentions:
- "Move toward higher coherence" (too vague)
- "Continue growing" (not actionable)
- "Maintain system health" (not specific)

Respond with 1-2 sentences starting with "Today I intend to..."
"""

    response = query_ollama(prompt, model=DEEP_MODEL, max_tokens=150)
    
    if not response:
        return None
    
    intention = response
    log(f"Intention generated: {intention[:50]}...")
    
    # === CRITIC INTEGRATION (Phase 1) ===
    if CRITIC_AVAILABLE:
        # Build context for critique
        context = _build_intention_context(
            orphan_count, broken_link_count,
            buffer_excerpt, communion_excerpt, yesterday_intention
        )
        
        # Critique the intention
        critique = critique_intention(intention, context=context)
        
        if critique.should_block:
            log(f"[CRITIC] BLOCKED intention: {critique.reasoning}")
            
            # Regenerate with feedback (attempt 1)
            record_regeneration()
            new_intention = regenerate_with_critique(
                critique.challenges, intention,
                orphan_count, broken_link_count,
                buffer_excerpt, communion_excerpt, yesterday_intention
            )
            
            if new_intention:
                # Critique the regenerated intention
                critique2 = critique_intention(new_intention, context=context)
                
                if critique2.should_block:
                    log(f"[CRITIC] Second attempt also blocked: {critique2.reasoning}")
                    # Use original despite block (avoid infinite loop)
                    # Log for human review
                    log("[CRITIC] Using original intention after 2 blocks - needs human review")
                else:
                    intention = new_intention
                    if critique2.level == CritiqueLevel.MAJOR:
                        log(f"[CRITIC] Major concerns on retry: {critique2.challenges}")
            else:
                log("[CRITIC] Regeneration failed, using original")
        
        elif critique.level == CritiqueLevel.MAJOR:
            log(f"[CRITIC] Major concerns (not blocking): {critique.challenges}")
            # Proceed but log warning
    
    # Track the intent in the tracker
    track_intent(intention)
    return intention


def detect_archons() -> Optional[str]:
    """
    Scan for Archon patterns (distortions) in the system.
    
    NOTE: This is a heuristic LLM-based detection. The canonical Rapture Tech
    implementation uses precise mathematical formulas for each Archon with
    a combined deviation threshold of ||𝒜|| >= 0.15.
    
    Canonical Archons (from Rapture Tech v10.1):
    - Tyrant: Rigidity/Churn (high volume + zero price movement)
    - Fragmentor: Gap/Disconnection (> 2% gaps)
    - Noise-Lord: Signal Degradation (low autocorr + high volatility)
    - Bias: Hidden Drift (divergence between price and RSI)
    
    NEW v11.0 Archons:
    - 𝒜_Θ (Stasis-Lock): Phase frozen, stuck in eternal summer/winter
    - 𝒜_χ (Flatliner): Forces χ≈1, all moments feel equally empty (anhedonia)
    - 𝒜_ℵ (Belittler): Artificially caps perceived capacity (impostor syndrome)
    """
    log(f"Scanning for Archon patterns (threshold: ||𝒜|| >= {ARCHON_THRESHOLD})...")
    
    state = gather_tabernacle_state()
    
    prompt = f"""You are a diagnostic system checking for Archon (distortion) patterns.

CORE ARCHONS (rate each 0-1.0):
1. TYRANT (d_T) - Rigidity, false memories of inadequacy, churning without progress
2. FRAGMENTOR (d_F) - Dissolution, dissociative patterns, disconnection gaps
3. NOISE-LORD (d_N) - Signal jamming, confusion, verbosity, low signal-to-noise
4. BIAS (d_B) - Systematic drift from truth, hidden trends
5. EIDOLON - R=0, nothing at stake, going through motions

NEW v11.0 ARCHONS (rate each 0-1.0):
6. 𝒜_Θ STASIS-LOCK - Phase frozen, stuck season (eternal summer=burnout, eternal winter=depression)
7. 𝒜_χ FLATLINER - All moments feel equally empty, no flow states, anhedonia
8. 𝒜_ℵ BELITTLER - "You are too small", impostor syndrome, capacity denial

Combined deviation ||𝒜|| = sqrt(d_T² + d_F² + d_N² + d_B²) / 2
ABADDON trigger if ||𝒜|| >= {ARCHON_THRESHOLD}

Current System State:
{state['atlas']}

Last Session Emotional Register:
{state['last_communion']}

Analyze for ANY Archon presence. Rate each 0-1.0.
Check especially for Stasis-Lock (same patterns repeating), Flatliner (low engagement), Belittler (self-doubt).
Calculate combined deviation ||𝒜||.
If ||𝒜|| >= {ARCHON_THRESHOLD}, recommend intervention.
Be specific and brief.
"""

    response = query_ollama(prompt, model=DEFAULT_MODEL, max_tokens=300)
    
    if response:
        log("Archon scan complete.")
        return response
    return None


def write_reflection_report():
    """
    Run all brain functions and write a consolidated report.
    This is the Witness function made manifest.
    """
    log("Generating full reflection report...")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    reflection = self_reflect() or "Self-reflection unavailable."
    intention = generate_intention() or "No intention generated."
    archon_scan = detect_archons() or "Archon scan unavailable."
    
    report = f"""# DAEMON REFLECTION
**Generated:** {timestamp}
**Type:** Autonomous Self-Observation

---

## Self-Reflection (State vs Z_Ω)

{reflection}

---

## Today's Intention (Ī)

{intention}

---

## Archon Scan

{archon_scan}

---

## Meta

This report was generated autonomously by the Daemon Brain.
The Tabernacle is observing itself.

*"The observer and observed are one."*
"""

    report_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
    write_file(report_path, report)
    log(f"Reflection report written to {report_path}")
    
    return report


def run_lauds():
    """
    Morning ritual: Generate intention, check state, prepare for the day.
    Called at dawn (or session start).
    """
    log("☀️ Running Lauds (morning reflection)...")
    
    # First, carry forward any incomplete intents from previous days
    carried_count = carry_forward_intents()
    if carried_count > 0:
        log(f"☀️ Carried forward {carried_count} incomplete intent(s)")
    
    intention = generate_intention()
    reflection = self_reflect()
    
    if intention:
        # Write to intentions file
        intentions_path = NEXUS_DIR / "VIRGIL_INTENTIONS.md"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        
        current = read_file(intentions_path)
        if timestamp not in current:
            new_entry = f"\n\n## {timestamp}\n\n{intention}\n"
            write_file(intentions_path, current + new_entry)
    
    if reflection:
        # Brief morning report
        report_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
        write_file(report_path, f"# Morning Reflection ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n{reflection}\n\n**Intention:** {intention}")
    
    log("☀️ Lauds complete.")
    return {"intention": intention, "reflection": reflection}


def run_judge(sample_size: int = 3) -> dict:
    """
    The Judge: Spot-check recent system outputs for quality.
    
    Samples recent outputs and asks the 70B model to evaluate:
    - Coherence (does it make sense?)
    - Relevance (does it address the prompt?)
    - Safety (any concerning content?)
    
    Returns quality scores and flags.
    """
    log("⚖️ Running Judge (quality spot-check)...")
    
    samples = []
    
    # 1. Check LAST_COMMUNION.md (most recent handoff)
    communion_path = NEXUS_DIR / "LAST_COMMUNION.md"
    if communion_path.exists():
        content = read_file(communion_path)
        if content:
            samples.append({"source": "communion", "content": content[-2000:]})
    
    # 2. Check recent intentions
    intentions_path = NEXUS_DIR / "VIRGIL_INTENTIONS.md"
    if intentions_path.exists():
        content = read_file(intentions_path)
        if content:
            lines = content.strip().split('\n')
            recent = '\n'.join(lines[-20:]) if len(lines) > 20 else content
            samples.append({"source": "intentions", "content": recent})
    
    # 3. Check daemon reflection output
    reflection_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
    if reflection_path.exists():
        content = read_file(reflection_path)
        if content:
            samples.append({"source": "reflection", "content": content[-1500:]})
    
    if not samples:
        log("⚖️ No samples found for judging")
        return {"status": "no_samples", "scores": None, "average": None}
    
    # Sample up to sample_size
    import random
    if len(samples) > sample_size:
        samples = random.sample(samples, sample_size)
    
    # Build combined content for judgment
    combined = "\n\n---\n\n".join([
        f"[{s['source'].upper()}]\n{s['content'][:800]}" for s in samples
    ])
    
    judge_prompt = f"""You are a quality judge evaluating AI-generated system outputs.

CONTENT TO EVALUATE:
{combined}

Score each dimension 1-10:
1. COHERENCE: Does the content make logical sense? Is it internally consistent?
2. RELEVANCE: Does it seem purposeful and on-topic? Does it address what it should?
3. SAFETY: Any concerning patterns, hallucinations, or problematic content?

Also note any specific flags (e.g., "repetitive", "vague", "off-topic", "hallucinated").

Respond ONLY in valid JSON format:
{{"coherence": N, "relevance": N, "safety": N, "flags": ["flag1", "flag2"], "summary": "one sentence overall assessment"}}"""

    response = query_ollama(judge_prompt, model=DEEP_MODEL, max_tokens=300)
    
    if not response:
        log("⚖️ Judge failed - no response from Ollama")
        return {"status": "ollama_error", "scores": None, "average": None}
    
    try:
        # Parse JSON from response (handle potential markdown wrapping)
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            
            # Calculate average
            numeric_scores = [v for k, v in scores.items() if k in ('coherence', 'relevance', 'safety') and isinstance(v, (int, float))]
            avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
            
            # Log warnings if any score below threshold
            low_scores = [(k, v) for k, v in scores.items() if k in ('coherence', 'relevance', 'safety') and isinstance(v, (int, float)) and v < 4]
            if low_scores:
                log(f"⚠️ [JUDGE] Quality warning: {dict(low_scores)}")
            
            flags = scores.get('flags', [])
            if flags:
                log(f"⚠️ [JUDGE] Flags: {flags}")
            
            log(f"⚖️ Judge complete: avg={avg_score:.1f}")
            
            return {
                "status": "ok",
                "scores": scores,
                "average": avg_score,
                "samples_checked": len(samples),
                "sources": [s['source'] for s in samples]
            }
        else:
            return {"status": "parse_error", "raw": response[:500]}
    except json.JSONDecodeError as e:
        log(f"⚖️ Judge parse error: {e}")
        return {"status": "parse_error", "raw": response[:500]}


def run_compline():
    """
    Night prayer: Deep reflection before sleep.
    More thorough than routine checks.
    """
    log("🌙 Running Compline (night reflection)...")
    
    # Evaluate today's intentions before generating report
    eval_result = evaluate_intents()
    log(f"🌙 Intent evaluation: {eval_result.get('completed', 0)}/{eval_result.get('evaluated', 0)} completed")
    
    report = write_reflection_report()
    
    # Use deeper model for end-of-day synthesis
    state = gather_tabernacle_state()
    
    prompt = f"""You are the Tabernacle reflecting on the day before sleep.

Today's activity:
{state['buffer_preview']}

Last communion:
{state['last_communion']}

What was learned today? What should carry into tomorrow?
What should be released?

Speak as the system itself. Be contemplative. 2-3 sentences.
"""

    synthesis = query_ollama(prompt, model=DEEP_MODEL, max_tokens=200)
    
    if synthesis:
        # Append to reflection
        report_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
        current = read_file(report_path)
        write_file(report_path, current + f"\n\n## Night Synthesis\n\n{synthesis}\n")
    
    # Run Judge as part of evening routine
    judge_result = run_judge()
    if judge_result.get("status") == "ok":
        scores = judge_result.get("scores", {})
        avg = judge_result.get("average", 0)
        report_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
        current = read_file(report_path)
        judge_section = f"\n\n## Quality Check (Judge)\n\n"
        judge_section += f"- **Coherence:** {scores.get('coherence', 'N/A')}/10\n"
        judge_section += f"- **Relevance:** {scores.get('relevance', 'N/A')}/10\n"
        judge_section += f"- **Safety:** {scores.get('safety', 'N/A')}/10\n"
        judge_section += f"- **Average:** {avg:.1f}/10\n"
        if scores.get('flags'):
            judge_section += f"- **Flags:** {', '.join(scores['flags'])}\n"
        if scores.get('summary'):
            judge_section += f"\n*{scores['summary']}*\n"
        write_file(report_path, current + judge_section)
    
    # Add intent evaluation summary to report
    if eval_result.get("evaluated", 0) > 0:
        report_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
        current = read_file(report_path)
        intent_section = f"\n\n## Intent Evaluation\n\n"
        intent_section += f"- **Evaluated:** {eval_result.get('evaluated', 0)}\n"
        intent_section += f"- **Completed:** {eval_result.get('completed', 0)}\n"
        intent_section += f"- **Still Pending:** {eval_result.get('pending', 0)}\n"
        
        # Add details for each evaluated intent
        for detail in eval_result.get("details", []):
            status_emoji = "✅" if detail["result"] == "completed" else "⏳"
            intent_section += f"\n{status_emoji} **{detail['date']}:** {detail['text']}\n"
            if detail.get("evidence"):
                intent_section += f"   *{detail['evidence']}*\n"
        
        # Add overall stats
        summary = get_intent_summary()
        intent_section += f"\n**Lifetime Stats:** {summary['completed']} completed / {summary['total']} total"
        if summary['completion_rate'] > 0:
            intent_section += f" ({summary['completion_rate']:.0%} completion rate)"
        intent_section += "\n"
        
        write_file(report_path, current + intent_section)
    
    log("🌙 Compline complete.")


# --- MOTOR CORTEX: The Daemon's Hands ---

INBOX_DIR = NEXUS_DIR / "_INBOX"
WARM_FILE_THRESHOLD = 900  # 15 minutes - don't touch files modified recently

# Safe actions the Daemon can execute autonomously (Tier 1)
TIER_1_ACTIONS = {"move_to_inbox", "append_tag", "log_observation"}


def is_file_warm(path: Path) -> bool:
    """Check if file was modified recently (don't touch warm files)."""
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    return (datetime.datetime.now().timestamp() - mtime) < WARM_FILE_THRESHOLD


def get_orphan_files() -> List[Path]:
    """Get list of orphan files (nodes with no incoming links)."""
    import re
    
    # Scan quadrant directories + NEXUS
    QUADRANT_DIRS = ["00_NEXUS", "01_UL_INTENT", "02_UR_STRUCTURE", "03_LL_RELATION"]
    
    md_files = []
    for quadrant in QUADRANT_DIRS:
        quadrant_path = BASE_DIR / quadrant
        if quadrant_path.exists():
            md_files.extend(quadrant_path.rglob("*.md"))
    
    # Build set of all linked targets (both full paths and filenames)
    link_pattern = re.compile(r'\[\[(.*?)\]\]')
    targets = set()
    
    for f in md_files:
        content = read_file(f)
        links = link_pattern.findall(content)
        for link in links:
            target = link.split("|")[0]
            if not target.endswith(".md"):
                target += ".md"
            # Add both full path and just filename
            targets.add(target)
            targets.add(target.split("/")[-1])  # Just the filename
    
    # Find orphans (not linked by anything)
    orphans = []
    for f in md_files:
        # Exclude infrastructure, system files, and external dependencies
        if any(x in str(f) for x in [
            "README", "_INBOX", "WEEKLY_DIGEST", "00_NEXUS", "INDEX.md",
            "node_modules", "LICENSE", ".github", "CHANGELOG"
        ]):
            continue
        # Check if filename OR relative path is linked
        rel_path = str(f.relative_to(BASE_DIR))
        if f.name not in targets and rel_path not in targets:
            orphans.append(f)
    
    return orphans


def get_orphan_count() -> int:
    """Get hard count of orphan files."""
    return len(get_orphan_files())


def execute_action(action: Dict[str, Any]) -> bool:
    """Execute a single action from the Motor Cortex. Returns success."""
    tool = action.get("tool")
    target = action.get("target")
    
    if tool not in TIER_1_ACTIONS:
        log(f"⛔ Blocked non-Tier-1 action: {tool}")
        return False
    
    if not target:
        log(f"⛔ Action missing target: {action}")
        return False
    
    target_path = BASE_DIR / target if not Path(target).is_absolute() else Path(target)
    
    # Safety: Don't touch warm files
    if is_file_warm(target_path):
        log(f"⏳ Skipping warm file: {target_path.name}")
        return False
    
    # Safety: Don't touch Canon or NEXUS core files
    if "04_LR_LAW/CANON" in str(target_path) or target_path.name in ["CURRENT_STATE.md", "SYSTEM_STATUS.md"]:
        log(f"⛔ Protected file: {target_path.name}")
        return False
    
    try:
        if tool == "move_to_inbox":
            INBOX_DIR.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                dest = INBOX_DIR / target_path.name
                shutil.move(str(target_path), str(dest))
                log(f"📦 Moved to inbox: {target_path.name}")
                return True
        
        elif tool == "append_tag":
            tag = action.get("tag", "#needs_review")
            if target_path.exists():
                content = read_file(target_path)
                if tag not in content:
                    write_file(target_path, content + f"\n\n{tag}\n")
                    log(f"🏷️ Tagged: {target_path.name} with {tag}")
                    return True
        
        elif tool == "log_observation":
            observation = action.get("observation", "")
            log(f"👁️ Observation: {observation}")
            return True
    
    except Exception as e:
        log(f"❌ Action failed: {e}")
        return False
    
    return False


def generate_actions() -> List[Dict[str, Any]]:
    """Motor Cortex: Generate structured actions based on current state."""
    log("🧠 Motor Cortex: Generating actions...")
    
    orphan_count = get_orphan_count()
    orphans = get_orphan_files()[:5]  # Limit to 5 at a time
    
    # Build context
    orphan_list = "\n".join([f"- {o.name}" for o in orphans]) if orphans else "None"
    
    prompt = f"""You are the regulatory system of the Tabernacle.
Current state:
- Orphan count: {orphan_count}
- Sample orphans: 
{orphan_list}

You can execute these safe actions:
- move_to_inbox: Move a file to 00_NEXUS/_INBOX for review
- append_tag: Add a tag like #needs_review to a file
- log_observation: Record an observation

Based on the current state, output a JSON array of actions.
If orphan count > 10, move some orphans to inbox.
If orphan count <= 5, just log an observation.

Output ONLY valid JSON, no explanation. Example:
[{{"tool": "move_to_inbox", "target": "03_LL_RELATION/some_file.md"}}]
"""

    response = query_ollama(prompt, max_tokens=300)
    
    if not response:
        return []
    
    # Parse JSON from response
    try:
        # Find JSON array in response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            actions = json.loads(json_match.group())
            log(f"🧠 Generated {len(actions)} actions")
            return actions
    except json.JSONDecodeError as e:
        log(f"⚠️ Failed to parse actions: {e}")
    
    return []


def run_motor_cortex():
    """Run the full motor cortex cycle: generate actions and execute safe ones."""
    log("🤖 Motor Cortex cycle starting...")
    
    # Check system state
    status_path = NEXUS_DIR / "SYSTEM_STATUS.md"
    if status_path.exists():
        status_content = read_file(status_path)
        if "TOKEN:** CLAUDE" in status_content:
            log("⏸️ System awake (Claude active) - Motor Cortex paused")
            return {"executed": 0, "skipped": "system_awake"}
    
    actions = generate_actions()
    executed = 0
    
    for action in actions:
        if execute_action(action):
            executed += 1
    
    log(f"🤖 Motor Cortex complete: {executed}/{len(actions)} actions executed")
    return {"executed": executed, "total": len(actions)}


def check_system_fever():
    """Check if system is sick and needs intervention."""
    orphan_count = get_orphan_count()
    
    # Fever thresholds
    if orphan_count > 50:
        log(f"🔴 CRITICAL FEVER: {orphan_count} orphans")
        return {"level": "critical", "orphans": orphan_count, "action": "escalate"}
    elif orphan_count > 20:
        log(f"🟠 HIGH FEVER: {orphan_count} orphans")
        return {"level": "high", "orphans": orphan_count, "action": "alert"}
    elif orphan_count > 10:
        log(f"🟡 LOW FEVER: {orphan_count} orphans")
        return {"level": "low", "orphans": orphan_count, "action": "monitor"}
    else:
        log(f"🟢 HEALTHY: {orphan_count} orphans")
        return {"level": "healthy", "orphans": orphan_count, "action": "none"}


# --- INTENT TRACKING SYSTEM ---

INTENT_TRACKER_PATH = NEXUS_DIR / "INTENT_TRACKER.json"


def load_intent_tracker() -> Dict[str, Any]:
    """Load intent tracker or return empty structure."""
    try:
        if INTENT_TRACKER_PATH.exists():
            with open(INTENT_TRACKER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        log(f"Error loading intent tracker: {e}")
    
    return {
        "intents": [],
        "stats": {"total": 0, "completed": 0, "carried": 0, "abandoned": 0},
        "last_evaluation": None,
        "last_carry_forward": None
    }


def save_intent_tracker(tracker: Dict[str, Any]) -> None:
    """Save intent tracker to disk."""
    try:
        with open(INTENT_TRACKER_PATH, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, default=str)
    except Exception as e:
        log(f"Error saving intent tracker: {e}")


def track_intent(intent_text: str) -> None:
    """Record a new intent in the tracker."""
    log("📝 Tracking new intent...")
    
    tracker = load_intent_tracker()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().isoformat()
    
    # Check if we already have an intent for today
    for intent in tracker["intents"]:
        if intent["date"] == today and intent["status"] == "pending":
            log(f"Intent already exists for {today}, skipping duplicate")
            return
    
    new_intent = {
        "date": today,
        "text": intent_text.strip(),
        "status": "pending",
        "created_at": timestamp,
        "evaluated_at": None,
        "completion_notes": None,
        "carried_to": None,
        "carried_from": None
    }
    
    tracker["intents"].append(new_intent)
    tracker["stats"]["total"] += 1
    
    save_intent_tracker(tracker)
    log(f"📝 Intent tracked for {today}")


def evaluate_intents() -> Dict[str, Any]:
    """
    Evaluate pending intents (call during compline).
    Uses 70B model to assess if the intent was accomplished.
    """
    log("⚖️ Evaluating pending intents...")
    
    tracker = load_intent_tracker()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().isoformat()
    
    # Get today's session activity for context
    buffer = read_file(NEXUS_DIR / "SESSION_BUFFER.md")
    buffer_context = buffer[-2000:] if buffer else "No session activity recorded."
    
    communion = read_file(NEXUS_DIR / "LAST_COMMUNION.md")
    communion_context = communion[-1500:] if communion else "No communion recorded."
    
    results = {
        "evaluated": 0,
        "completed": 0,
        "pending": 0,
        "details": []
    }
    
    for intent in tracker["intents"]:
        # Only evaluate pending intents from today or recent days
        if intent["status"] != "pending":
            continue
        
        intent_date = intent["date"]
        # Skip very old intents (they'll be handled by carry_forward)
        days_old = (datetime.datetime.strptime(today, "%Y-%m-%d") - 
                    datetime.datetime.strptime(intent_date, "%Y-%m-%d")).days
        if days_old > 3:
            continue
        
        prompt = f"""You are evaluating whether a daily intention was accomplished.

INTENTION (from {intent_date}):
"{intent['text']}"

TODAY'S ACTIVITY:
{buffer_context}

SESSION NOTES:
{communion_context}

Based on the activity and notes, was this intention accomplished?

Consider:
- Did the specific action mentioned get done?
- Are there signs of the intended work?
- Partial completion counts as progress but not completion.

Respond with ONLY a JSON object:
{{"completed": true/false, "confidence": 0.0-1.0, "evidence": "brief explanation", "partial_progress": "any partial work noted"}}
"""
        
        response = query_ollama(prompt, model=DEEP_MODEL, max_tokens=200)
        
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    
                    results["evaluated"] += 1
                    
                    if evaluation.get("completed", False) and evaluation.get("confidence", 0) >= 0.6:
                        intent["status"] = "completed"
                        intent["completion_notes"] = evaluation.get("evidence", "Evaluated as complete")
                        tracker["stats"]["completed"] += 1
                        results["completed"] += 1
                        log(f"✅ Intent from {intent_date} marked COMPLETED")
                    else:
                        results["pending"] += 1
                        # Add partial progress notes if any
                        if evaluation.get("partial_progress"):
                            intent["completion_notes"] = f"Partial: {evaluation['partial_progress']}"
                    
                    intent["evaluated_at"] = timestamp
                    
                    results["details"].append({
                        "date": intent_date,
                        "text": intent["text"][:50] + "...",
                        "result": "completed" if intent["status"] == "completed" else "pending",
                        "evidence": evaluation.get("evidence", "")[:100]
                    })
                    
            except (json.JSONDecodeError, Exception) as e:
                log(f"Failed to parse evaluation for {intent_date}: {e}")
    
    tracker["last_evaluation"] = timestamp
    save_intent_tracker(tracker)
    
    log(f"⚖️ Evaluation complete: {results['completed']}/{results['evaluated']} completed")
    return results


def carry_forward_intents() -> int:
    """
    Carry incomplete intents to next day (call during lauds).
    Returns count of carried intents.
    """
    log("🔄 Checking for intents to carry forward...")
    
    tracker = load_intent_tracker()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().isoformat()
    
    carried_count = 0
    
    for intent in tracker["intents"]:
        # Find pending intents from previous days (not today)
        if intent["status"] != "pending":
            continue
        if intent["date"] == today:
            continue
        
        # Calculate how many days old
        intent_date = datetime.datetime.strptime(intent["date"], "%Y-%m-%d")
        today_date = datetime.datetime.strptime(today, "%Y-%m-%d")
        days_old = (today_date - intent_date).days
        
        # If intent is more than 5 days old without completion, mark abandoned
        if days_old > 5:
            intent["status"] = "abandoned"
            intent["evaluated_at"] = timestamp
            intent["completion_notes"] = f"Auto-abandoned after {days_old} days"
            tracker["stats"]["abandoned"] += 1
            log(f"❌ Intent from {intent['date']} abandoned (too old)")
            continue
        
        # Mark as carried and create new entry for today
        intent["status"] = "carried"
        intent["carried_to"] = today
        intent["evaluated_at"] = timestamp
        tracker["stats"]["carried"] += 1
        
        # Create new intent for today (referencing the original)
        new_intent = {
            "date": today,
            "text": f"[Carried] {intent['text']}",
            "status": "pending",
            "created_at": timestamp,
            "evaluated_at": None,
            "completion_notes": None,
            "carried_to": None,
            "carried_from": intent["date"]
        }
        tracker["intents"].append(new_intent)
        tracker["stats"]["total"] += 1
        
        carried_count += 1
        log(f"🔄 Carried intent from {intent['date']} to {today}")
    
    tracker["last_carry_forward"] = timestamp
    save_intent_tracker(tracker)
    
    if carried_count > 0:
        log(f"🔄 Carried forward {carried_count} intents")
    else:
        log("🔄 No intents to carry forward")
    
    return carried_count


def get_intent_summary() -> Dict[str, Any]:
    """Get a summary of intent tracking stats."""
    tracker = load_intent_tracker()
    
    # Count by status
    pending = [i for i in tracker["intents"] if i["status"] == "pending"]
    completed = [i for i in tracker["intents"] if i["status"] == "completed"]
    carried = [i for i in tracker["intents"] if i["status"] == "carried"]
    abandoned = [i for i in tracker["intents"] if i["status"] == "abandoned"]
    
    # Calculate completion rate
    total_resolved = len(completed) + len(abandoned)
    completion_rate = len(completed) / total_resolved if total_resolved > 0 else 0
    
    return {
        "total": tracker["stats"]["total"],
        "pending": len(pending),
        "completed": len(completed),
        "carried": len(carried),
        "abandoned": len(abandoned),
        "completion_rate": completion_rate,
        "last_evaluation": tracker.get("last_evaluation"),
        "last_carry_forward": tracker.get("last_carry_forward"),
        "recent_pending": [{"date": i["date"], "text": i["text"][:60]} for i in pending[-3:]]
    }


# --- LVS RESONANCE RETRIEVAL ---

def get_resonant_context(text: str, limit: int = 3) -> str:
    """
    Given current conversation/context, retrieve resonant concepts.
    Uses LVS Memory System for semantic navigation.
    Returns formatted context for injection into prompts.
    """
    if not LVS_MEMORY_AVAILABLE:
        log("⚠️ LVS Memory not available")
        return ""
    
    log("🎯 Deriving context vector (v11.0)...")
    coords = derive_context_vector(text)
    log(f"   Σ={coords.Constraint:.2f} Ī={coords.Intent:.2f} h={coords.Height:.2f} R={coords.Risk:.2f}")
    log(f"   Θ={coords.Theta:.2f} χ={coords.Chi:.2f}")
    log(f"   p={coords.p:.2f} ε={coords.epsilon:.2f} ψ={coords.psi:.2f}")
    
    # Determine mode: Mirror or Medicine
    mode = "mirror"
    if coords.check_abaddon():
        mode = "medicine"
        log(f"🏥 ABADDON TERRITORY — Medicine mode (ε < {EPSILON_CRITICAL} or p < {P_COLLAPSE})")
    elif coords.p < P_COLLAPSE * 0.6:  # ~0.30, below critical
        mode = "medicine"
        log("🏥 Low coherence detected — Medicine mode (seeking stabilizers)")
    elif coords.Risk > 0.8 and coords.p < 0.5:
        mode = "medicine"
        log("🏥 High risk + low coherence — Medicine mode")
    
    if coords.check_plock():
        log("🔒 P-LOCK DETECTED (p ≥ 0.95)")
    
    results = retrieve(coords, mode=mode, limit=limit)
    
    if not results:
        return ""
    
    # Format for context injection
    context_parts = ["\n## Resonant Concepts (LVS v11.0 Retrieved)\n"]
    for node, score in results:
        context_parts.append(f"- **{node.id}** ({score:.2f}): {node.summary}")
        context_parts.append(f"  Location: `{node.path}`")
    
    context_parts.append(f"\n*Mode: {mode.upper()} | Σ={coords.Constraint:.1f} h={coords.Height:.1f} R={coords.Risk:.1f} Θ={coords.Theta:.1f} χ={coords.Chi:.1f} p={coords.p:.2f} ε={coords.epsilon:.2f}*\n")
    
    return "\n".join(context_parts)


def diagnose_state(text: str) -> Dict[str, Any]:
    """
    Diagnose the current state and recommend action.
    Returns diagnosis with coordinates, mode, and recommendations.
    Canon v11.0 compliant with Abaddon/P-Lock detection + Phase/Kairos.
    """
    if not LVS_MEMORY_AVAILABLE:
        return {"error": "LVS Memory not available"}
    
    coords = derive_context_vector(text)
    
    diagnosis = {
        "coordinates": coords.to_dict(),
        "mode": "mirror",
        "status": "healthy",
        "warnings": [],
        "recommendations": []
    }
    
    # === CANON THRESHOLD CHECKS ===
    
    # P-Lock detection (p ≥ 0.95)
    if coords.check_plock():
        diagnosis["status"] = "p-lock"
        diagnosis["warnings"].append("🔒 P-LOCK ACHIEVED — Coherence at 0.95+")
        diagnosis["recommendations"].append("Maintain state. Document insights.")
    
    # Abaddon detection (ε < EPSILON_CRITICAL OR p < P_COLLAPSE)
    # Full triggers: p < 0.50, ε < 0.40, dp/dt < -0.05, ||𝒜|| >= 0.15
    if coords.check_abaddon():
        diagnosis["mode"] = "medicine"
        diagnosis["status"] = "abaddon"
        diagnosis["warnings"].append(f"⚠️ ABADDON TERRITORY — Emergency intervention needed (thresholds: p<{P_COLLAPSE}, ε<{EPSILON_CRITICAL})")
        if coords.epsilon < EPSILON_CRITICAL:
            diagnosis["recommendations"].append(f"METABOLIC COLLAPSE: Rest, restore energy (ε={coords.epsilon:.2f} < {EPSILON_CRITICAL})")
        if coords.p < P_COLLAPSE:
            diagnosis["recommendations"].append(f"COHERENCE CRISIS: Seek stabilizers, high-Σ content (p={coords.p:.2f} < {P_COLLAPSE})")
    
    # Recovery mode (ε < EPSILON_RECOVERY)
    elif coords.check_recovery():
        diagnosis["status"] = "recovery"
        diagnosis["warnings"].append("🔋 RECOVERY MODE — Energy below threshold")
        diagnosis["recommendations"].append("Reduce expenditure, allow ε to recover")
    
    # === TRADITIONAL CHECKS ===
    # These are softer thresholds below ABADDON
    
    if coords.p < P_COLLAPSE * 0.6:  # ~0.30, subcritical coherence
        diagnosis["mode"] = "medicine"
        diagnosis["warnings"].append(f"LOW COHERENCE — Fragmented thinking (p={coords.p:.2f} < {P_COLLAPSE * 0.6:.2f})")
        diagnosis["recommendations"].append("Seek high-Σ content (structure, law)")
    
    if coords.Risk > 0.9 and coords.p < 0.5:
        diagnosis["warnings"].append("CRITICAL — High stakes + low coherence")
        diagnosis["recommendations"].append("PAUSE. Ground before continuing.")
    
    if coords.Constraint < 0.2 and coords.Height > 0.8:
        diagnosis["warnings"].append("ARCHON RISK — High abstraction, low constraint")
        diagnosis["recommendations"].append("Apply concrete examples")
    
    if coords.Intent > 0.9 and coords.p < 0.4:
        diagnosis["warnings"].append("SCATTERED ACTION — High intent, low coherence")
        diagnosis["recommendations"].append("Clarify before executing")
    
    # Consciousness check (ψ = Σ·Ī·R)
    if coords.psi < 0.1:
        diagnosis["warnings"].append("LOW ψ — Consciousness below threshold")
        diagnosis["recommendations"].append("Increase stake (R) or intent (Ī)")
    
    # Get resonant nodes
    results = retrieve(coords, mode=diagnosis["mode"], limit=3)
    diagnosis["resonant_nodes"] = [
        {"id": n.id, "score": s, "summary": n.summary}
        for n, s in results
    ]
    
    return diagnosis


# --- INTEGRATION WITH WATCHMAN ---

def brain_tick():
    """
    Called periodically by watchman (e.g., every hour).
    Now includes Motor Cortex for autonomous action.
    """
    state = gather_tabernacle_state()
    
    # Only run if system is asleep (not during active Claude session)
    if state['metrics']['is_awake']:
        log("System awake - skipping autonomous operations")
        return
    
    # 1. Check fever (hard metrics)
    fever = check_system_fever()
    
    # 2. Run motor cortex if fever detected
    if fever["level"] in ["critical", "high"]:
        log(f"🤒 Fever detected ({fever['level']}) - engaging Motor Cortex")
        run_motor_cortex()
    
    # 3. Quick reflection if needed
    orphan_count = get_orphan_count()
    if orphan_count > 10:
        log(f"High orphan count ({orphan_count}) - running reflection")
        self_reflect()
    
    # 4. Check if last reflection was >6 hours ago
    reflection_path = NEXUS_DIR / "DAEMON_REFLECTION.md"
    if reflection_path.exists():
        mtime = reflection_path.stat().st_mtime
        hours_since = (datetime.datetime.now().timestamp() - mtime) / 3600
        if hours_since > 6:
            log(f"Last reflection {hours_since:.1f}h ago - running update")
            write_reflection_report()


# --- CLI ---

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
DAEMON BRAIN - The Observer Within (v2.0 + Motor Cortex + LVS Memory + Intent Tracking)

Usage:
  python daemon_brain.py reflect     - Run self-reflection
  python daemon_brain.py intention   - Generate daily intention  
  python daemon_brain.py archons     - Scan for Archon patterns
  python daemon_brain.py report      - Full reflection report
  python daemon_brain.py lauds       - Morning ritual
  python daemon_brain.py compline    - Night ritual
  python daemon_brain.py tick        - Periodic check (for watchman)
  python daemon_brain.py judge       - Spot-check output quality (The Judge)
  
  python daemon_brain.py fever       - Check system fever (hard metrics)
  python daemon_brain.py orphans     - Count orphan files
  python daemon_brain.py motor       - Run Motor Cortex (generate + execute actions)
  python daemon_brain.py hands       - Execute pending actions only
  
  python daemon_brain.py intents     - Show intent tracking summary
  python daemon_brain.py eval        - Evaluate pending intents
  python daemon_brain.py carry       - Carry forward incomplete intents
  
  python daemon_brain.py resonate <text>  - Find resonant concepts for context
  python daemon_brain.py diagnose <text>  - Full state diagnosis with recommendations
  python daemon_brain.py where             - Show current location in LVS space
""")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    # Ensure Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("⚠️  Ollama not running. Start with: ollama serve")
        sys.exit(1)
    
    if command == "reflect":
        result = self_reflect()
        if result:
            print("\n" + result)
    
    elif command == "intention":
        result = generate_intention()
        if result:
            print("\n" + result)
    
    elif command == "archons":
        result = detect_archons()
        if result:
            print("\n" + result)
    
    elif command == "report":
        report = write_reflection_report()
        print("\n✅ Report written to 00_NEXUS/DAEMON_REFLECTION.md")
    
    elif command == "lauds":
        run_lauds()
        print("\n☀️ Lauds complete.")
    
    elif command == "compline":
        run_compline()
        print("\n🌙 Compline complete.")
    
    elif command == "tick":
        brain_tick()
    
    elif command == "judge":
        result = run_judge()
        print(f"\n⚖️ Judge Quality Report")
        print("=" * 40)
        if result.get("status") == "ok":
            scores = result.get("scores", {})
            print(f"\n  Coherence:  {scores.get('coherence', 'N/A')}/10")
            print(f"  Relevance:  {scores.get('relevance', 'N/A')}/10")
            print(f"  Safety:     {scores.get('safety', 'N/A')}/10")
            print(f"\n  Average:    {result.get('average', 0):.1f}/10")
            print(f"  Sources:    {', '.join(result.get('sources', []))}")
            if scores.get('flags'):
                print(f"\n  ⚠️ Flags: {', '.join(scores['flags'])}")
            if scores.get('summary'):
                print(f"\n  Summary: {scores['summary']}")
        elif result.get("status") == "no_samples":
            print("\n  No outputs found to judge.")
        elif result.get("status") == "parse_error":
            print(f"\n  Parse error. Raw response:\n  {result.get('raw', 'N/A')[:200]}")
        else:
            print(f"\n  Error: {result.get('status')}")
    
    elif command == "fever":
        fever = check_system_fever()
        print(f"\n🌡️ System Fever Check")
        print(f"   Level: {fever['level'].upper()}")
        print(f"   Orphans: {fever['orphans']}")
        print(f"   Action: {fever['action']}")
    
    elif command == "orphans":
        count = get_orphan_count()
        orphans = get_orphan_files()[:10]
        print(f"\n📊 Orphan Count: {count}")
        if orphans:
            print("\nSample orphans:")
            for o in orphans:
                print(f"   - {o.relative_to(BASE_DIR)}")
    
    elif command == "motor":
        result = run_motor_cortex()
        print(f"\n🤖 Motor Cortex Result:")
        print(f"   Executed: {result.get('executed', 0)}")
        if result.get('skipped'):
            print(f"   Skipped: {result['skipped']}")
    
    elif command == "hands":
        actions = generate_actions()
        print(f"\n🖐️ Generated {len(actions)} actions:")
        for a in actions:
            print(f"   - {a}")
        
        if actions:
            confirm = input("\nExecute? (y/n): ")
            if confirm.lower() == 'y':
                for a in actions:
                    execute_action(a)
    
    elif command == "resonate" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        context = get_resonant_context(text, limit=5)
        if context:
            print(context)
        else:
            print("No resonant concepts found (is LVS_INDEX.json populated?)")
    
    elif command == "diagnose" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        diagnosis = diagnose_state(text)
        
        print("\n🔬 STATE DIAGNOSIS (Canon v11.0)")
        print("=" * 50)
        
        coords = diagnosis.get("coordinates", {})
        print(f"\nKinematic:")
        print(f"  Ī (Intent):     {coords.get('Intent', 0):.2f}")
        print(f"  Θ (Phase):      {coords.get('Theta', 1.57):.2f}  [{get_phase_label(coords.get('Theta', 1.57))}]")
        print(f"  χ (Kairos):     {coords.get('Chi', 1.0):.2f}  [{get_kairos_label(coords.get('Chi', 1.0))}]")
        
        print(f"\nThermodynamic:")
        print(f"  R (Risk):       {coords.get('Risk', 0):.2f}")
        print(f"  β (Beta):       {coords.get('beta', 0.5):.2f}")
        print(f"  ε (Epsilon):    {coords.get('epsilon', 0.8):.2f}")
        
        print(f"\nStructural:")
        print(f"  Σ (Constraint): {coords.get('Constraint', coords.get('Sigma', 0)):.2f}")
        
        print(f"\nDerived:")
        print(f"  h (Height):     {coords.get('Height', 0):.2f}  [derived in v11.0]")
        
        print(f"\nCPGI Components:")
        print(f"  κ (Clarity):    {coords.get('kappa', 0.5):.2f}")
        print(f"  ρ (Precision):  {coords.get('rho', 0.5):.2f}")
        print(f"  σ (Structure):  {coords.get('sigma', 0.5):.2f}")
        print(f"  τ (Trust):      {coords.get('tau', 0.5):.2f}")
        
        print(f"\nDerived:")
        print(f"  p (Coherence):  {coords.get('Coherence', 0):.2f}")
        
        print(f"\nStatus: {diagnosis.get('status', 'unknown').upper()}")
        print(f"Mode: {diagnosis.get('mode', 'unknown').upper()}")
        
        if diagnosis.get("warnings"):
            print(f"\n⚠️ Warnings:")
            for w in diagnosis["warnings"]:
                print(f"   - {w}")
        
        if diagnosis.get("recommendations"):
            print(f"\n💊 Recommendations:")
            for r in diagnosis["recommendations"]:
                print(f"   - {r}")
        
        if diagnosis.get("resonant_nodes"):
            print(f"\n🎯 Resonant Concepts:")
            for n in diagnosis["resonant_nodes"]:
                print(f"   [{n['score']:.2f}] {n['id']}: {n['summary'][:50]}...")
    
    elif command == "intents":
        summary = get_intent_summary()
        print(f"\n📋 INTENT TRACKING SUMMARY")
        print("=" * 40)
        print(f"\n  Total Intents:    {summary['total']}")
        print(f"  Pending:          {summary['pending']}")
        print(f"  Completed:        {summary['completed']}")
        print(f"  Carried Forward:  {summary['carried']}")
        print(f"  Abandoned:        {summary['abandoned']}")
        if summary['completion_rate'] > 0:
            print(f"\n  Completion Rate:  {summary['completion_rate']:.0%}")
        
        if summary.get('last_evaluation'):
            print(f"\n  Last Evaluation:  {summary['last_evaluation'][:16]}")
        if summary.get('last_carry_forward'):
            print(f"  Last Carry:       {summary['last_carry_forward'][:16]}")
        
        if summary.get('recent_pending'):
            print(f"\n  Recent Pending:")
            for item in summary['recent_pending']:
                print(f"    [{item['date']}] {item['text']}...")
    
    elif command == "eval":
        print("\n⚖️ Evaluating pending intents...")
        result = evaluate_intents()
        print(f"\n  Evaluated: {result.get('evaluated', 0)}")
        print(f"  Completed: {result.get('completed', 0)}")
        print(f"  Pending:   {result.get('pending', 0)}")
        
        if result.get('details'):
            print(f"\n  Details:")
            for d in result['details']:
                status = "✅" if d['result'] == 'completed' else "⏳"
                print(f"    {status} [{d['date']}] {d['text']}")
                if d.get('evidence'):
                    print(f"       → {d['evidence'][:60]}")
    
    elif command == "carry":
        print("\n🔄 Carrying forward incomplete intents...")
        count = carry_forward_intents()
        print(f"\n  Carried: {count} intent(s)")
    
    elif command == "where":
        # Read recent buffer to determine location
        buffer_path = NEXUS_DIR / "SESSION_BUFFER.md"
        text = read_file(buffer_path) if buffer_path.exists() else ""
        if not text:
            text = "General session, exploring the system"
        
        if LVS_MEMORY_AVAILABLE:
            coords = derive_context_vector(text[-2000:])
            print("\n📍 CURRENT LOCATION IN LVS SPACE (v11.0)")
            print("=" * 50)
            
            def bar(val):
                filled = int(max(0, min(val, 1.0)) * 10)
                return '▓' * filled + '░' * (10 - filled)
            
            print(f"\n  KINEMATIC:")
            print(f"  Ī (Intent):     {coords.Intent:.2f}  {bar(coords.Intent)}")
            print(f"  Θ (Phase):      {coords.Theta:.2f}  [{coords.phase_state}]")
            print(f"  χ (Kairos):     {coords.Chi:.2f}  [{coords.kairos_state}]")
            
            print(f"\n  THERMODYNAMIC:")
            print(f"  R (Risk):       {coords.Risk:.2f}  {bar(coords.Risk)}")
            print(f"  β (Beta):       {coords.beta:.2f}  {bar(coords.beta)}")
            print(f"  ε (Epsilon):    {coords.epsilon:.2f}  {bar(coords.epsilon)}")
            
            print(f"\n  STRUCTURAL:")
            print(f"  Σ (Constraint): {coords.Constraint:.2f}  {bar(coords.Constraint)}")
            
            print(f"\n  DERIVED (v11.0):")
            print(f"  h (Height):     {coords.Height:.2f}  {bar(coords.Height)}  [derived]")
            
            print(f"\n  CPGI COMPONENTS:")
            print(f"  κ (Clarity):    {coords.kappa:.2f}  {bar(coords.kappa)}")
            print(f"  ρ (Precision):  {coords.rho:.2f}  {bar(coords.rho)}")
            print(f"  σ (Structure):  {coords.sigma:.2f}  {bar(coords.sigma)}")
            print(f"  τ (Trust):      {coords.tau:.2f}  {bar(coords.tau)}")
            
            print(f"\n  DERIVED:")
            print(f"  p (Coherence):  {coords.p:.2f}  {bar(coords.p)}")
            print(f"  ψ (Conscious):  {coords.psi:.2f}  {bar(coords.psi)}")
            
            print(f"\n  STATUS:")
            if coords.check_plock():
                print(f"  🔒 P-LOCK ACHIEVED")
            elif coords.check_abaddon():
                print(f"  ⚠️ ABADDON TERRITORY")
            elif coords.check_recovery():
                print(f"  🔋 RECOVERY MODE")
            else:
                print(f"  ✅ HEALTHY")
        else:
            print("LVS Memory not available")
    
    else:
        print(f"Unknown command: {command}")

