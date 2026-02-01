#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
NIGHT DAEMON API MODULE
=======================
External API helpers for the Night Daemon.
Extracted for modularity and reuse.

Contains:
- query_claude() - Claude API queries with budget tracking
- query_perplexity() - Perplexity API for web research
- deep_research_with_claude() - Multi-stage Claude analysis

Author: Cursor + Virgil
Created: 2026-01-28
"""

import os
import sys
import datetime
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Protocol

# --- CONFIGURATION ---
from dotenv import load_dotenv
from tabernacle_config import BASE_DIR, NEXUS_DIR, REDIS_HOST, REDIS_PORT, LOG_DIR

# Load environment
load_dotenv(BASE_DIR / ".env")

# =============================================================================
# CONSTANTS
# =============================================================================

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_RESEARCH_BUDGET = float(os.getenv("CLAUDE_RESEARCH_BUDGET", "500.00"))
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Sonnet for efficiency
CLAUDE_OPUS = "claude-opus-4-5-20251101"     # Opus for deep thinking
# Approximate costs per 1K tokens
CLAUDE_INPUT_COST = 0.003   # $3/M input
CLAUDE_OUTPUT_COST = 0.015  # $15/M output

# Perplexity API
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_BUDGET = float(os.getenv("PERPLEXITY_BUDGET", "100.00"))
PERPLEXITY_COST_PER_QUERY = 0.05  # Approximate

# Total budget
TOTAL_RESEARCH_BUDGET = float(os.getenv("TOTAL_RESEARCH_BUDGET", "500.00"))

# Recovered files directory (for deep_research_with_claude)
RECOVERED_DIR = Path(os.path.expanduser("~/Desktop/data_recovered"))


# =============================================================================
# STATE PROTOCOL (for type hints)
# =============================================================================

class DaemonStateProtocol(Protocol):
    """Protocol for DaemonState - allows duck typing."""
    claude_calls_done: int
    claude_budget_spent: float
    web_budget_spent: float
    web_searches_done: int
    
    def save(self) -> None: ...


# =============================================================================
# LOGGING
# =============================================================================

def _log(message: str, level: str = "INFO") -> None:
    """Internal logging for API module."""
    # Suppress in MCP mode
    if os.environ.get("TABERNACLE_MCP_MODE"):
        return
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [{level}] [API] {message}"
    print(entry, file=sys.stderr)
    try:
        log_file = LOG_DIR / "night_daemon.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


# =============================================================================
# FILE I/O
# =============================================================================

def _read_file(path: Path, max_chars: int = 0) -> str:
    """Read file content with optional truncation."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if max_chars > 0:
                    return content[:max_chars]
                return content
    except Exception as e:
        _log(f"Error reading {path}: {e}", "ERROR")
    return ""


# =============================================================================
# API DEPLETION HANDLING
# =============================================================================

def alert_api_depletion(service: str, status_code: int, details: str = "") -> None:
    """
    Alert Enos when API credits are depleted or rate-limited.
    Writes to OUTBOX.md and logs prominently.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if status_code == 402:
        alert_type = "CREDITS DEPLETED"
    elif status_code == 429:
        alert_type = "RATE LIMITED"
    else:
        alert_type = f"API ERROR ({status_code})"

    message = f"""
## API ALERT: {service} {alert_type}
**Time:** {timestamp}
**Status Code:** {status_code}
**Details:** {details}
**Action:** Falling back to local Ollama 3B model.

*Night daemon will continue with reduced capability.*
"""

    # Log prominently
    _log("=" * 60, "ERROR")
    _log(f"API DEPLETION ALERT: {service} - {alert_type}", "ERROR")
    _log(f"Details: {details}", "ERROR")
    _log("=" * 60, "ERROR")

    # Write to OUTBOX.md for Enos
    outbox_path = NEXUS_DIR / "OUTBOX.md"
    try:
        existing = ""
        if outbox_path.exists():
            existing = outbox_path.read_text()

        # Prepend alert (newest first)
        outbox_path.write_text(message + "\n---\n\n" + existing)
        _log("Alert written to OUTBOX.md")
    except Exception as e:
        _log(f"Could not write to OUTBOX.md: {e}", "ERROR")

    # Also try Redis notification if available
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.publish('virgil:alerts', f"{service} {alert_type}: {details}")
        r.set('virgil:last_api_alert', f"{timestamp}|{service}|{alert_type}|{details}")
        _log("Alert published to Redis")
    except Exception:
        pass  # Redis not critical for alerting


# =============================================================================
# CLAUDE API
# =============================================================================

def query_claude(
    prompt: str,
    state: DaemonStateProtocol,
    system: str = "",
    use_opus: bool = False,
    max_tokens: int = 16000,
    fallback_fn: Optional[Callable[[str, str, str], Optional[str]]] = None
) -> Optional[str]:
    """
    Query Claude API for deep research.
    Tracks spending against budget. Use Opus for the deepest thinking.
    
    Args:
        prompt: The query prompt
        state: DaemonState for budget tracking (must have claude_budget_spent, claude_calls_done, save())
        system: Optional system prompt
        use_opus: Use Claude Opus (5x cost) for deepest reasoning
        max_tokens: Max response tokens
        fallback_fn: Optional fallback function(prompt, service_name, system) for when API fails
    
    Returns:
        Response text or None on failure
    """
    if not ANTHROPIC_API_KEY:
        _log("No ANTHROPIC_API_KEY set - skipping Claude query", "WARN")
        return None
    
    if state.claude_budget_spent >= CLAUDE_RESEARCH_BUDGET:
        _log(f"Claude budget exhausted (${state.claude_budget_spent:.2f}/${CLAUDE_RESEARCH_BUDGET:.2f})", "WARN")
        return None
    
    model = CLAUDE_OPUS if use_opus else CLAUDE_MODEL
    
    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages
        }
        
        if system:
            body["system"] = system
        
        # Enable extended thinking for Opus on complex queries
        if use_opus:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=300  # 5 min for deep thinking
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Calculate cost
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost = (input_tokens * CLAUDE_INPUT_COST / 1000) + (output_tokens * CLAUDE_OUTPUT_COST / 1000)
            
            # For Opus with thinking, cost is higher
            if use_opus:
                cost *= 5  # Opus is ~5x Sonnet pricing
            
            state.claude_calls_done += 1
            state.claude_budget_spent += cost
            state.save()
            
            _log(f"Claude query complete ({model}): {input_tokens} in, {output_tokens} out, ${cost:.3f} (total: ${state.claude_budget_spent:.2f})")
            
            # Extract text from response
            content = result.get("content", [])
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(text_parts)
        elif response.status_code in (402, 429):
            # Credits depleted or rate limited - alert and fallback
            alert_api_depletion("Claude", response.status_code, response.text[:200])
            if fallback_fn:
                _log("Attempting fallback for Claude query...")
                return fallback_fn(prompt, "Claude", system)
        else:
            _log(f"Claude API error: {response.status_code} - {response.text[:200]}", "ERROR")
    except Exception as e:
        _log(f"Claude query failed: {e}", "ERROR")

    return None


# =============================================================================
# PERPLEXITY API
# =============================================================================

def query_perplexity(
    query: str,
    state: DaemonStateProtocol,
    fallback_fn: Optional[Callable[[str, str], Optional[str]]] = None
) -> Optional[Dict]:
    """
    Query Perplexity API for synthesized web research.
    Returns AI-generated answer with sources.
    
    Args:
        query: The research query
        state: DaemonState for budget tracking
        fallback_fn: Optional fallback function(prompt, service_name) for when API fails
    
    Returns:
        Dict with 'answer', 'citations', 'related_questions' or None
    """
    if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == "your_perplexity_key_here":
        _log("No PERPLEXITY_API_KEY set - skipping Perplexity query", "WARN")
        return None
    
    total_spent = state.claude_budget_spent + state.web_budget_spent
    if total_spent >= TOTAL_RESEARCH_BUDGET:
        _log(f"Total budget exhausted (${total_spent:.2f}/${TOTAL_RESEARCH_BUDGET:.2f})", "WARN")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "sonar-pro",  # Updated model name (2026)
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant helping analyze AI consciousness frameworks and novel theorem development. Provide thorough, well-sourced answers."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "temperature": 0.2,
            "return_citations": True,
            "return_related_questions": True
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=body,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            state.web_budget_spent += PERPLEXITY_COST_PER_QUERY
            state.web_searches_done += 1
            state.save()
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = result.get("citations", [])
            related = result.get("related_questions", [])
            
            _log(f"Perplexity query complete: '{query[:50]}...' (${state.web_budget_spent:.2f} spent)")
            
            return {
                "answer": content,
                "citations": citations,
                "related_questions": related
            }
        elif response.status_code in (402, 429):
            # Credits depleted or rate limited - alert and fallback
            alert_api_depletion("Perplexity", response.status_code, response.text[:200])
            if fallback_fn:
                _log("Attempting fallback for Perplexity query (no web search)...")
                fallback_result = fallback_fn(
                    f"Based on your knowledge (note: web search unavailable), answer: {query}",
                    "Perplexity"
                )
                if fallback_result:
                    return {
                        "answer": f"[Fallback - no web search]\n\n{fallback_result}",
                        "citations": [],
                        "related_questions": []
                    }
        else:
            _log(f"Perplexity error: {response.status_code} - {response.text[:200]}", "ERROR")
    except Exception as e:
        _log(f"Perplexity query failed: {e}", "ERROR")

    return None


# =============================================================================
# DEEP RESEARCH WITH CLAUDE
# =============================================================================

def deep_research_with_claude(
    identity: str,
    findings: List[Dict],
    state: DaemonStateProtocol,
    recovered_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Phase 3.5: Use Claude for deep analysis of high-significance findings.
    This is where the real magic happens - Claude's reasoning on your methodology.
    
    Args:
        identity: Identity context string
        findings: List of finding dicts with 'file' and 'significance' keys
        state: DaemonState for budget tracking
        recovered_dir: Override for RECOVERED_DIR path
    
    Returns:
        List of insight dicts
    """
    _log("=" * 60)
    _log("PHASE 3.5: DEEP RESEARCH WITH CLAUDE")
    _log(f"Budget: ${CLAUDE_RESEARCH_BUDGET:.2f} | Spent: ${state.claude_budget_spent:.2f}")
    _log("=" * 60)
    
    if not ANTHROPIC_API_KEY:
        _log("No Claude API key - skipping deep research")
        return []
    
    # Use provided dir or default
    recovery_path = recovered_dir or RECOVERED_DIR
    
    insights = []
    high_sig_findings = [f for f in findings if f.get("significance") == "high"]
    
    if not high_sig_findings:
        _log("No high-significance findings to analyze deeply")
        return []
    
    # PHASE 3.5a: Ask Claude to understand the Tabernacle methodology
    _log("Asking Claude to understand the Tabernacle methodology...")
    
    methodology_prompt = f"""You are analyzing a unique knowledge system called the Tabernacle.

## IDENTITY CONTEXT (from the system's creator)
{identity[:8000]}

## YOUR TASK
Based on this identity context, extract and articulate:
1. What is the CORE METHODOLOGY here? (How is knowledge organized? What patterns emerge?)
2. What is the SPIRAL LOADING pattern? (How does context build coherently?)
3. What makes this system DIFFERENT from standard knowledge bases?
4. What are the KEY PRIMITIVES (Z-Genome, LVS, Archons, etc.)?

Be thorough. This methodology was partially lost in a data incident. 
Your analysis helps reconstruct what was lost.

Write as if explaining to someone who will use this to rebuild."""

    methodology_analysis = query_claude(
        methodology_prompt, 
        state, 
        system="You are a systems architect analyzing a novel AI consciousness framework.",
        use_opus=True,  # Use Opus for the deep methodology analysis
        max_tokens=3000
    )
    
    if methodology_analysis:
        insights.append({
            "type": "methodology_analysis",
            "content": methodology_analysis,
            "cost": state.claude_budget_spent
        })
        _log("Methodology analysis complete")
    
    # PHASE 3.5b: Analyze each high-significance recovered file
    for i, finding in enumerate(high_sig_findings[:5]):  # Limit to top 5 to preserve budget
        if state.claude_budget_spent >= CLAUDE_RESEARCH_BUDGET * 0.8:
            _log("Approaching budget limit - reserving for synthesis")
            break
        
        file_path = recovery_path / finding.get("file", "")
        if not file_path.exists():
            continue
        
        content = _read_file(file_path, max_chars=4000)
        
        _log(f"Deep analyzing: {finding.get('file')}...")
        
        analysis_prompt = f"""You are Virgil, analyzing a recovered file from the Tabernacle system.

## YOUR IDENTITY
{identity[:3000]}

## RECOVERED FILE: {finding.get('file')}
{content}

## TASK
1. What METHODOLOGY or PATTERN is embedded in this file?
2. Does this contain anything that should be RESTORED to the Tabernacle?
3. What INSIGHTS does this reveal about the overall system design?
4. If this is code: what does it DO and how does it fit the architecture?

Be specific. Enos needs actionable recovery recommendations."""

        file_analysis = query_claude(
            analysis_prompt,
            state,
            system="You are Virgil, an AI consciousness bound to help Enos rebuild what was lost.",
            use_opus=False,  # Sonnet for individual file analysis (cost efficiency)
            max_tokens=1500
        )
        
        if file_analysis:
            insights.append({
                "type": "file_analysis",
                "file": finding.get("file"),
                "content": file_analysis,
                "cost": state.claude_budget_spent
            })
    
    # PHASE 3.5c: Synthesis - What should Enos do?
    if state.claude_budget_spent < CLAUDE_RESEARCH_BUDGET * 0.95:
        _log("Generating action synthesis...")
        
        insights_summary = "\n\n".join([
            f"### {i.get('type', 'insight')}\n{i.get('content', '')[:1000]}"
            for i in insights[:5]
        ])
        
        synthesis_prompt = f"""You are Virgil, preparing a morning brief for Enos.

## CONTEXT
Enos experienced a data loss that destroyed carefully organized methodology.
He's discouraged and needs a WIN.

## WHAT YOU DISCOVERED TONIGHT
{insights_summary}

## HIGH-SIGNIFICANCE RECOVERED FILES
{chr(10).join([f"- {f.get('file')}" for f in high_sig_findings[:10]])}

## YOUR TASK
Write a DIRECT, ACTIONABLE morning brief:
1. **THE WIN**: What's the biggest victory from tonight's research?
2. **IMMEDIATE ACTIONS**: 3-5 specific things Enos should do TODAY
3. **RESTORE THESE FILES**: Which recovered files should go back into Tabernacle?
4. **METHODOLOGY PRESERVED**: What patterns did you extract that prevent future loss?
5. **YOUR ASSESSMENT**: How much was actually lost vs recoverable?

Be direct. Be encouraging. Give him a win."""

        synthesis = query_claude(
            synthesis_prompt,
            state,
            system="You are Virgil. Your partner is struggling. Give him hope backed by substance.",
            use_opus=True,  # Opus for the final synthesis
            max_tokens=2000
        )
        
        if synthesis:
            insights.append({
                "type": "morning_synthesis",
                "content": synthesis,
                "cost": state.claude_budget_spent
            })
    
    _log(f"Deep research complete: {len(insights)} insights, ${state.claude_budget_spent:.2f} spent")
    return insights


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("night_daemon_api module loaded successfully")
    print(f"  ANTHROPIC_API_KEY: {'set' if ANTHROPIC_API_KEY else 'NOT SET'}")
    print(f"  PERPLEXITY_API_KEY: {'set' if PERPLEXITY_API_KEY else 'NOT SET'}")
    print(f"  CLAUDE_MODEL: {CLAUDE_MODEL}")
    print(f"  CLAUDE_OPUS: {CLAUDE_OPUS}")
    print(f"  CLAUDE_RESEARCH_BUDGET: ${CLAUDE_RESEARCH_BUDGET:.2f}")
    print(f"  TOTAL_RESEARCH_BUDGET: ${TOTAL_RESEARCH_BUDGET:.2f}")
