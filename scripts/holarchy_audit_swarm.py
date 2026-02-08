#!/usr/bin/env python3
"""
HOLARCHY DEEP SCAN AUDIT SWARM (v1.0)
Orchestrator: Claude Code (System) via Python
Protocol: ABCDA' Spiral | API: Distributed (Anthropic/OpenRouter/Perplexity)
Mission: Verify Nerve Graft, Audit Layers Λ0-Λ6, Calculate Coherence (p).
"""

import os
import json
import time
import subprocess
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# --- CONFIGURATION ---
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEXUS_DIR = os.path.join(ROOT_DIR, "00_NEXUS")
REPORT_FILE = os.path.join(NEXUS_DIR, f"AUDIT_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

# --- 1. SENSORS (Context Gathering) ---
def read_file(rel_path: str, limit: int = 12000) -> str:
    """Reads a file from the repo, handling missing files gracefully."""
    path = os.path.join(ROOT_DIR, rel_path)
    if not os.path.exists(path): return "[MISSING FILE]"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content if len(content) < limit else content[:limit] + "\n...[TRUNCATED]"
    except Exception as e:
        return f"[READ ERROR: {e}]"

def get_process_list() -> str:
    """Snapshots running python processes (The Ground Truth)."""
    try:
        cmd = "ps aux | grep python | grep -v grep"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout if result.stdout else "[NO PYTHON PROCESSES FOUND]"
    except Exception as e:
        return f"[PROCESS SCAN ERROR: {e}]"

def gather_evidence() -> Dict[str, str]:
    print("📦 SENSORS: Gathering Holarchy State...")
    return {
        "config": read_file("scripts/tabernacle_config.py"),
        "manifest": read_file("00_NEXUS/HOLARCHY_MANIFEST/MANIFEST_INDEX.md"),
        "heartbeat_code": read_file("scripts/virgil_variable_heartbeat.py", limit=20000),
        "state_heart": read_file("00_NEXUS/heartbeat_state.json"),
        "state_auto": read_file("00_NEXUS/autonomous_state.json"),
        "state_archon": read_file("00_NEXUS/archon_state.json"),
        "processes": get_process_list(),
        "timestamp_utc": datetime.utcnow().isoformat()
    }

# --- 2. NERVES (API Gateways) ---
def call_anthropic(model: str, system: str, prompt: str) -> Dict:
    if not ANTHROPIC_KEY: return {"error": "No API Key"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model=model, max_tokens=4000, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return extract_json(msg.content[0].text)
    except Exception as e: return {"error": str(e)}

def call_openrouter(model: str, system: str, prompt: str) -> Dict:
    if not OPENROUTER_KEY: return {"error": "No API Key"}
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )
        return json.loads(resp.json()['choices'][0]['message']['content'])
    except Exception as e: return {"error": str(e)}

def call_perplexity(model: str, system: str, prompt: str) -> Dict:
    if not PERPLEXITY_KEY: return {"info": "Skipped (No Key)"}
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_KEY}"},
            json={"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        )
        return extract_json(resp.json()['choices'][0]['message']['content'])
    except Exception as e: return {"error": str(e)}

def extract_json(text: str) -> Dict:
    """Robust JSON extraction."""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except: return {"raw_text": text, "error": "JSON Parse Failed"}

# --- 3. AGENTS (The Swarm) ---

def run_agent(name, cfg, evidence):
    print(f"⚡ DISPATCHING: {name} via {cfg['provider']} ({cfg['model']})...")
    prompt = cfg['prompt'].format(**evidence)
    if cfg['provider'] == 'anthropic':
        return call_anthropic(cfg['model'], "You are a System Auditor. Output valid JSON.", prompt)
    elif cfg['provider'] == 'openrouter':
        return call_openrouter(cfg['model'], "You are a System Auditor. Output valid JSON.", prompt)
    elif cfg['provider'] == 'perplexity':
        return call_perplexity(cfg['model'], "You are a Researcher. Output valid JSON.", prompt)

AGENTS = {
    "CARTOGRAPHER": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "prompt": """
        ROLE: Agent-Λ0Λ1 (Cartographer).
        MISSION: Audit Substrate & Fabric.
        CONTEXT:
        [PROCESSES]: {processes}
        [CONFIG]: {config}
        CHECKS:
        1. Identify Redis SPOF risk.
        2. Verify if critical daemons (watchman, heart) are in process list.
        OUTPUT JSON: {{ "layer": "L0_L1", "health": 0.0-1.0, "voids": [], "findings": [] }}
        """
    },
    "VIVISECTOR": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "prompt": """
        ROLE: Agent-Λ2Λ3 (Vivisector).
        MISSION: Audit Pulse & Cortex. VERIFY NERVE GRAFT.
        CONTEXT:
        [CODE]: {heartbeat_code}
        [HEART_STATE]: {state_heart}
        [AUTO_STATE]: {state_auto}
        CHECKS:
        1. Does `virgil_variable_heartbeat.py` contain `_sync_nerve` logic?
        2. Compare `last_enos_interaction` (Heart) vs `last_enos_seen` (Auto).
        3. If delta > 120s, the Nerve is SEVERED (Code fixed but process stale).
        OUTPUT JSON: {{ "layer": "L2_L3", "health": 0.0-1.0, "nerve_status": "GRAFTED|SEVERED|ZOMBIE", "voids": [] }}
        """
    },
    "LIBRARIAN": {
        "provider": "openrouter",
        "model": "openai/gpt-4o",
        "prompt": """
        ROLE: Agent-Λ4Λ5 (Librarian).
        MISSION: Audit Memory & Agency.
        CONTEXT:
        [ARCHON_STATE]: {state_archon}
        CHECKS:
        1. Is Archon state stale (>24h)?
        2. Is distortion level (175) a hallucination compared to manifest?
        OUTPUT JSON: {{ "layer": "L4_L5", "health": 0.0-1.0, "archon_status": "FRESH|STALE", "voids": [] }}
        """
    },
    "RESEARCHER": {
        "provider": "perplexity",
        "model": "sonar",
        "prompt": """
        TOPIC: Distributed System Split-Brain Recovery.
        We fixed a "proprioceptive dissociation" by making the heartbeat daemon poll user activity files.
        QUERY: Is this "unidirectional polling" the standard fix for split-brain in local-first architectures?
        OUTPUT JSON: {{ "validation": "CONFIRMED|RISKY", "warning": "string" }}
        """
    }
}

# --- 4. SYNTHESIS (The Architect) ---

def run_architect(evidence, agent_results):
    print("🧠 ARCHITECT: Synthesizing Global Coherence...")
    prompt = f"""
    YOU ARE LOGOS (Λ6).

    SWARM FINDINGS:
    {json.dumps(agent_results, indent=2)}

    LVS FORMULA: p = (kappa * rho * sigma * tau)^(1/4)
    - kappa: Integration (Did Vivisector confirm Graft?)
    - rho: Precision (Timestamps aligned?)
    - sigma: Structure (Process list matches Config?)
    - tau: Trust (Did Archon hallucinate?)

    TASK:
    1. Calculate p-value.
    2. Write a MARKDOWN Report.
    3. Diagnose the "Proprioceptive Dissociation".
    4. Define Priority Fixes.
    """
    # We ask Opus to write the final Markdown directly
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    msg = client.messages.create(
        model="claude-3-haiku-20240307", max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

# --- MAIN LOOP ---

def main():
    print("\n🌀 INITIATING HOLARCHY AUDIT SWARM...")
    evidence = gather_evidence()

    # WAVE 1: Parallel Agents
    agent_results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_agent, name, cfg, evidence): name for name, cfg in AGENTS.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                agent_results[name] = future.result()
                print(f"✅ {name} Reported In.")
            except Exception as e:
                print(f"❌ {name} Failed: {e}")
                agent_results[name] = {"error": str(e)}

    # WAVE 2: Architect
    final_markdown = run_architect(evidence, agent_results)

    # Output
    with open(REPORT_FILE, 'w') as f:
        f.write(final_markdown)

    print(f"\n📄 REPORT SAVED: {REPORT_FILE}")
    print("-" * 40)
    print(final_markdown)
    print("-" * 40)

if __name__ == "__main__":
    main()
