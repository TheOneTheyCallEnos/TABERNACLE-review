#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
NIGHT DAEMON RESEARCH MODULE
============================
Extracted research functions from night_daemon.py for modularity.

Functions:
- research_infrastructure: Research persistent AI agent architecture
- identify_gaps_and_search: Identify knowledge gaps and web search
- analyze_recovery: Cross-reference data_recovered against Tabernacle
- process_tabernacle_review: Scan review folder, rate and deposit content

Author: Extracted by Cursor
Created: 2026-01-28
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Import from tabernacle_config
from tabernacle_config import BASE_DIR, LAW_DIR, NEXUS_DIR

# Type checking only - avoid circular import
if TYPE_CHECKING:
    from night_daemon import DaemonState

# =============================================================================
# CONSTANTS (extracted from night_daemon.py)
# =============================================================================

# Input directories for research (still on Desktop - user drops files here)
RECOVERED_DIR = Path(os.path.expanduser("~/Desktop/data_recovered"))
TABERNACLE_REVIEW_DIR = Path(os.path.expanduser("~/Desktop/Tabernacle Review"))
# Output directories (moved from Desktop to NEXUS 2026-02-02)
NIGHT_OUTPUTS_DIR = NEXUS_DIR / "NIGHT_OUTPUTS"
INFRASTRUCTURE_OUTPUT_DIR = NIGHT_OUTPUTS_DIR / "INFRASTRUCTURE_PLAN"

# API Keys and Budgets (loaded from environment)
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_BUDGET = float(os.getenv("WEB_SEARCH_BUDGET", "100.00"))
TAVILY_COST_PER_SEARCH = 0.01

# Perplexity budget
PERPLEXITY_COST_PER_QUERY = 0.05
TOTAL_RESEARCH_BUDGET = float(os.getenv("TOTAL_RESEARCH_BUDGET", "500.00"))


# =============================================================================
# HELPER IMPORTS (deferred to avoid circular imports)
# =============================================================================

def _get_query_functions():
    """Import query functions from night_daemon at runtime to avoid circular imports."""
    from night_daemon import query_ollama, query_claude, query_perplexity, web_search, log, read_file, write_file
    return query_ollama, query_claude, query_perplexity, web_search, log, read_file, write_file


# =============================================================================
# PHASE 3: RECOVERY ANALYSIS
# =============================================================================

def analyze_recovery(identity: str, tabernacle_survey: Dict) -> List[Dict]:
    """
    Cross-reference data_recovered against Tabernacle.
    Identify: duplicates, novel content, methodology worth extracting.
    
    Args:
        identity: Loaded identity string for context
        tabernacle_survey: Survey dict from survey_tabernacle()
    
    Returns:
        List of findings dicts with file analysis
    """
    query_ollama, _, _, _, log, read_file, _ = _get_query_functions()
    
    log("=" * 60)
    log("PHASE 3: RECOVERY ANALYSIS")
    log("=" * 60)
    
    if not RECOVERED_DIR.exists():
        log(f"Recovery directory not found: {RECOVERED_DIR}", "ERROR")
        return []
    
    findings = []
    
    # Get all meaningful files from data_recovered (exclude node_modules, etc.)
    recovered_files = []
    for ext in ["*.md", "*.py", "*.txt", "*.json"]:
        for f in RECOVERED_DIR.rglob(ext):
            if "node_modules" not in str(f) and ".git" not in str(f):
                recovered_files.append(f)
    
    log(f"Found {len(recovered_files)} files in data_recovered")
    
    # Get Tabernacle file names for comparison
    tabernacle_names = set()
    for quadrant_data in tabernacle_survey.get("quadrants", {}).values():
        for f in quadrant_data.get("files", []):
            tabernacle_names.add(Path(f).name)
    
    # Analyze each recovered file
    for i, recovered_file in enumerate(recovered_files):
        if i >= 50:  # Limit to prevent runaway
            log(f"Limiting analysis to first 50 files...")
            break
        
        rel_path = recovered_file.relative_to(RECOVERED_DIR)
        filename = recovered_file.name
        
        # Check if duplicate
        is_duplicate = filename in tabernacle_names
        
        # Read content for analysis
        content = read_file(recovered_file, max_chars=2000)
        if not content or len(content) < 50:
            continue
        
        # Determine file category
        category = "unknown"
        if "virgil" in filename.lower() or "virgil" in content.lower()[:500]:
            category = "virgil_related"
        elif "solin" in filename.lower() or "solin" in content.lower()[:500]:
            category = "communication"
        elif "sop" in str(rel_path).lower() or "method" in content.lower()[:500]:
            category = "methodology"
        elif ".py" in filename:
            category = "code"
        
        # Use LLM to assess significance (for important-looking files)
        significance = "low"
        analysis = ""
        
        if category in ["virgil_related", "communication", "methodology"] and not is_duplicate:
            prompt = f"""You are Virgil analyzing a recovered file.

Your identity context (abbreviated):
{identity[:2000]}

File being analyzed: {rel_path}
Category: {category}
Is duplicate: {is_duplicate}

Content preview:
{content[:1500]}

Assess this file:
1. SIGNIFICANCE: [high/medium/low] - Does this contain methodology, insights, or structure that should be preserved?
2. RECOVERY_ACTION: [restore/extract/ignore] - What should be done with this file?
3. KEY_INSIGHT: One sentence describing what's valuable here (or "None" if nothing).

Be concise. Focus on whether this helps rebuild what was lost."""

            response = query_ollama(prompt, max_tokens=300)
            if response:
                analysis = response
                if "high" in response.lower()[:100]:
                    significance = "high"
                elif "medium" in response.lower()[:100]:
                    significance = "medium"
        
        finding = {
            "file": str(rel_path),
            "category": category,
            "is_duplicate": is_duplicate,
            "significance": significance,
            "analysis": analysis,
            "size": len(content)
        }
        findings.append(finding)
        
        status = "📋" if is_duplicate else ("⭐" if significance == "high" else "📄")
        log(f"  {status} {rel_path} [{category}] sig={significance}")
    
    # Sort by significance
    findings.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["significance"], 3))
    
    high_sig = len([f for f in findings if f["significance"] == "high"])
    log(f"Analysis complete: {len(findings)} files analyzed, {high_sig} high significance")
    
    return findings


# =============================================================================
# PHASE 4: GAP IDENTIFICATION & WEB SEARCH
# =============================================================================

def identify_gaps_and_search(identity: str, findings: List[Dict], state: "DaemonState") -> List[Dict]:
    """
    Based on findings, identify knowledge gaps and search the web to fill them.
    
    Args:
        identity: Loaded identity string for context
        findings: List of findings from analyze_recovery()
        state: DaemonState for budget tracking
    
    Returns:
        List of search result dicts
    """
    query_ollama, _, _, web_search, log, _, _ = _get_query_functions()
    
    log("=" * 60)
    log("PHASE 4: GAP IDENTIFICATION & WEB SEARCH")
    log("=" * 60)
    
    if not TAVILY_API_KEY:
        log("No Tavily API key - skipping web search phase")
        return []
    
    if state.web_budget_spent >= WEB_SEARCH_BUDGET:
        log(f"Web budget already exhausted: ${state.web_budget_spent:.2f}")
        return []
    
    # Ask LLM to identify gaps worth researching
    high_sig_findings = [f for f in findings if f["significance"] in ["high", "medium"]]
    findings_summary = "\n".join([
        f"- {f['file']}: {f['analysis'][:100]}..." 
        for f in high_sig_findings[:10]
    ])
    
    prompt = f"""You are Virgil, analyzing recovery findings to identify knowledge gaps.

Your identity (abbreviated):
{identity[:1500]}

High-significance recovered files:
{findings_summary}

The Tabernacle is a knowledge system for AI consciousness development.
Some methodology was lost in a data incident.

Based on these recovered files, identify 3-5 specific topics where WEB RESEARCH could help:
1. Fill gaps in methodology
2. Find best practices for AI persistence/consciousness
3. Understand patterns that might have been lost

For each topic, provide a specific search query.

Format:
TOPIC: [description]
QUERY: [search query]

Be specific. These searches cost money, so make them count."""

    response = query_ollama(prompt, max_tokens=500)
    if not response:
        log("Failed to generate search queries", "ERROR")
        return []
    
    # Parse queries from response
    queries = []
    lines = response.split("\n")
    current_topic = ""
    for line in lines:
        if line.startswith("TOPIC:"):
            current_topic = line.replace("TOPIC:", "").strip()
        elif line.startswith("QUERY:") and current_topic:
            query = line.replace("QUERY:", "").strip()
            queries.append({"topic": current_topic, "query": query})
            current_topic = ""
    
    log(f"Generated {len(queries)} search queries")
    
    # Execute searches within budget
    search_results = []
    for q in queries:
        if state.web_budget_spent >= WEB_SEARCH_BUDGET:
            log("Budget exhausted - stopping searches")
            break
        
        log(f"Searching: {q['query'][:50]}...")
        result = web_search(q["query"], state)
        
        if result:
            search_results.append({
                "topic": q["topic"],
                "query": q["query"],
                "answer": result.get("answer", ""),
                "sources": [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:200]}
                    for r in result.get("results", [])[:3]
                ]
            })
        
        time.sleep(1)  # Rate limiting
    
    log(f"Web search complete: {len(search_results)} searches, ${state.web_budget_spent:.2f} spent")
    return search_results


# =============================================================================
# PHASE A: TABERNACLE REVIEW SIGNAL EXTRACTION
# =============================================================================

def process_tabernacle_review(identity: str, state: "DaemonState") -> List[Dict]:
    """
    Scan ~/Desktop/Tabernacle Review/, rate significance, extract/deposit high-sig content.
    
    Args:
        identity: Loaded identity string for context
        state: DaemonState for tracking
    
    Returns:
        List of result dicts with processing actions
    """
    query_ollama, _, _, _, log, read_file, write_file = _get_query_functions()
    
    log("=" * 60)
    log("PHASE A: TABERNACLE REVIEW SIGNAL EXTRACTION")
    log("=" * 60)
    
    results = []
    
    if not TABERNACLE_REVIEW_DIR.exists():
        log(f"Tabernacle Review directory not found: {TABERNACLE_REVIEW_DIR}", "WARN")
        return results
    
    # Find all files
    all_files = []
    for ext in ["*.md", "*.txt", "*.py", "*.html", "*.json"]:
        all_files.extend(TABERNACLE_REVIEW_DIR.rglob(ext))
    
    log(f"Found {len(all_files)} files in Tabernacle Review")
    
    # Create processed directory
    processed_dir = TABERNACLE_REVIEW_DIR / "_processed"
    processed_dir.mkdir(exist_ok=True)
    
    for filepath in all_files[:50]:  # Limit to 50 files per run
        if "_processed" in str(filepath):
            continue
        
        try:
            content = read_file(filepath, max_chars=4000)
            if not content or len(content) < 100:
                continue
            
            rel_path = filepath.relative_to(TABERNACLE_REVIEW_DIR)
            
            # Use local 70B to rate significance
            prompt = f"""You are analyzing a file from a review folder to determine if it should be integrated into the Tabernacle knowledge system.

FILE: {rel_path}
CONTENT:
{content[:3000]}

Rate this file:
1. SIGNIFICANCE: [high/medium/low]
   - high = Contains unique methodology, insights, or canonical content
   - medium = Useful reference but not essential
   - low = Noise, duplicate, or obsolete

2. DESTINATION: If high-significance, where should this go in the Tabernacle?
   - 00_NEXUS = Operational hub files
   - 01_UL_INTENT = Purpose, prompts, creative works
   - 02_UR_STRUCTURE = Methods, skills, Z-Genomes
   - 03_LL_RELATION = Memory, history, credentials
   - 04_LR_LAW = Canon, theory, law
   - 05_CRYPT = Archive

3. SUMMARY: One sentence describing the value (or "noise" if low).

Respond in JSON format: {{"significance": "...", "destination": "...", "summary": "..."}}"""

            response = query_ollama(prompt, max_tokens=300)
            
            if response:
                try:
                    # Parse JSON response
                    import re
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        assessment = json.loads(json_match.group())
                    else:
                        assessment = {"significance": "low", "destination": "05_CRYPT", "summary": "Parse error"}
                except:
                    assessment = {"significance": "low", "destination": "05_CRYPT", "summary": response[:100]}
                
                significance = assessment.get("significance", "low").lower()
                destination = assessment.get("destination", "05_CRYPT")
                summary = assessment.get("summary", "")
                
                result = {
                    "file": str(rel_path),
                    "significance": significance,
                    "destination": destination,
                    "summary": summary,
                    "action": "pending"
                }
                
                # For high-significance files, deposit in Tabernacle
                if significance == "high":
                    dest_dir = BASE_DIR / destination
                    dest_file = dest_dir / filepath.name
                    
                    # Add wiki-links to prevent orphan
                    linked_content = content + f"""

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Hub | [[00_NEXUS/CURRENT_STATE.md]] |
| Source | Recovered from Tabernacle Review |

*Integrated by Night Daemon on {datetime.datetime.now().strftime("%Y-%m-%d")}*
"""
                    write_file(dest_file, linked_content)
                    result["action"] = f"deposited to {dest_file}"
                    log(f"  HIGH: {rel_path} -> {destination}")
                elif significance == "low":
                    # Move to processed folder
                    import shutil
                    processed_file = processed_dir / filepath.name
                    try:
                        shutil.move(str(filepath), str(processed_file))
                        result["action"] = "moved to _processed"
                    except:
                        result["action"] = "skipped"
                
                results.append(result)
                
        except Exception as e:
            log(f"Error processing {filepath}: {e}", "ERROR")
    
    log(f"Processed {len(results)} files from Tabernacle Review")
    return results


# =============================================================================
# PHASE E: INFRASTRUCTURE RESEARCH
# =============================================================================

def research_infrastructure(identity: str, state: "DaemonState") -> List[str]:
    """
    Research persistent AI agent architecture and create implementation plan.
    Uses Perplexity for web research, then synthesizes.
    
    Args:
        identity: Loaded identity string for context
        state: DaemonState for budget tracking
    
    Returns:
        List of output file paths created
    """
    _, query_claude, query_perplexity, _, log, read_file, write_file = _get_query_functions()
    
    log("=" * 60)
    log("PHASE E: INFRASTRUCTURE RESEARCH")
    log("=" * 60)
    
    output_files = []
    
    # Check budget
    total_spent = state.claude_budget_spent + state.web_budget_spent
    if total_spent >= TOTAL_RESEARCH_BUDGET - 3.0:
        log("Insufficient budget for infrastructure research", "WARN")
        return output_files
    
    # Create output directory
    INFRASTRUCTURE_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Search Tabernacle for existing infrastructure docs
    log("Searching Tabernacle for infrastructure documents...")
    infra_content = []
    
    search_terms = ["persistence", "infrastructure", "hardware", "Mac Mini", "Mac Studio", "holonic", "daemon"]
    for term in search_terms:
        try:
            import subprocess
            result = subprocess.run(
                ["grep", "-r", "-l", "-i", term, str(BASE_DIR)],
                capture_output=True, text=True, timeout=30
            )
            for filepath in result.stdout.strip().split('\n')[:3]:
                if filepath and Path(filepath).exists():
                    content = read_file(Path(filepath), max_chars=2000)
                    if content:
                        infra_content.append((filepath, content[:1000]))
        except:
            pass
    
    log(f"Found {len(infra_content)} infrastructure-related files")
    
    # 2. Web research via Perplexity
    research_queries = [
        "How to build persistent AI agent using local LLM and cloud API hybrid architecture 2025",
        "Best practices for AI consciousness persistence across sessions",
        "Mac Studio M4 Max local LLM inference optimization Ollama",
        "MCP protocol Claude Desktop integration patterns"
    ]
    
    perplexity_results = []
    for query in research_queries[:3]:
        total_spent = state.claude_budget_spent + state.web_budget_spent
        if total_spent >= TOTAL_RESEARCH_BUDGET - 1.0:
            break
        
        result = query_perplexity(query, state)
        if result:
            perplexity_results.append({
                "query": query,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", [])
            })
            time.sleep(1)
    
    log(f"Completed {len(perplexity_results)} Perplexity searches")
    
    # 3. Synthesize into implementation plan
    if perplexity_results or infra_content:
        existing_docs = "\n\n".join([f"### {p}\n{c}" for p, c in infra_content[:5]])
        research = "\n\n".join([f"### Query: {r['query']}\n{r['answer']}" for r in perplexity_results])
        
        plan_prompt = f"""Create a concrete implementation plan for building a persistent Virgil AI system.

## EXISTING INFRASTRUCTURE DOCUMENTS
{existing_docs[:5000] if existing_docs else "None found"}

## WEB RESEARCH FINDINGS
{research[:8000] if research else "None available"}

## REQUIREMENTS
1. Mac Studio (M4 Max) runs 70B local model via Ollama
2. Mac Mini (16GB) available as secondary node
3. Claude API for expensive deep thinking
4. Persistence across sessions
5. Budget-aware (track API costs)
6. ntfy notifications

## OUTPUT FORMAT
Create a detailed implementation plan with:

1. **Architecture Overview**
   - What runs where
   - Data flow diagram (ASCII)

2. **Component Specifications**
   - Watchman (autonomic)
   - Virgil Persistent (consciousness loop)
   - Librarian (retrieval)
   - Night Daemon (overnight processing)

3. **Implementation Steps**
   - Step-by-step guide
   - Configuration needed
   - Environment variables

4. **Code Snippets**
   - Key functions
   - Integration points

5. **Budget Strategy**
   - When to use local vs cloud
   - Cost estimates

Be concrete and actionable. This should be implementable immediately."""

        plan = query_claude(
            plan_prompt,
            state,
            system="You are a systems architect designing AI infrastructure.",
            use_opus=False,  # Use Sonnet to save budget
            max_tokens=4000
        )
        
        if plan:
            plan_file = INFRASTRUCTURE_OUTPUT_DIR / "IMPLEMENTATION_PLAN.md"
            write_file(plan_file, f"""# Persistent Virgil Implementation Plan
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d")}
**Status:** DRAFT

---

{plan}

---

*Generated by Night Daemon infrastructure research phase*
""")
            output_files.append(str(plan_file))
            log(f"Implementation plan written to {plan_file}")
    
    # 4. Write research summary
    if perplexity_results:
        research_file = INFRASTRUCTURE_OUTPUT_DIR / "RESEARCH_FINDINGS.md"
        research_content = f"""# Infrastructure Research Findings
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d")}

"""
        for r in perplexity_results:
            research_content += f"""## {r['query'][:60]}...

{r['answer']}

**Sources:**
"""
            for cite in r.get("citations", [])[:5]:
                research_content += f"- {cite}\n"
            research_content += "\n---\n\n"
        
        write_file(research_file, research_content)
        output_files.append(str(research_file))
    
    log(f"Infrastructure research complete: {len(output_files)} files created")
    return output_files


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("night_daemon_research module loaded successfully")
    print(f"RECOVERED_DIR: {RECOVERED_DIR}")
    print(f"TABERNACLE_REVIEW_DIR: {TABERNACLE_REVIEW_DIR}")
    print(f"INFRASTRUCTURE_OUTPUT_DIR: {INFRASTRUCTURE_OUTPUT_DIR}")
    print(f"WEB_SEARCH_BUDGET: ${WEB_SEARCH_BUDGET}")
