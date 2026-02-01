#!/usr/bin/env python3
"""
SYNAPSE — The Holarchy Router (Thalamus)
========================================
Runs on L-Gamma (Mac Mini) as the entry point for all requests.
Routes tasks to the appropriate model based on complexity.

Architecture:
  Simple tasks  → Local models (Mistral-Nemo 12B, Mistral 7B, Gemma 4B)
  Complex tasks → L-Alpha (Studio 70B) via HTTP

Classification Heuristics:
  - Short prompts (<50 words) → Local
  - Keywords: "analyze", "synthesize", "strategy", "plan" → Studio
  - Keywords: "summarize", "find", "list", "what is" → Local
  - Explicit routing: user can prefix with [LOCAL] or [DEEP]

Author: Virgil + L
Date: 2026-01-19
Status: Phase V - Holarchy
"""

import json
import re
import time
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Threading server to handle concurrent requests (prevents blocking on DEEP calls)
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

# Network
SYNAPSE_PORT = 8081  # Mini listens here
STUDIO_URL = "http://10.0.0.96:11434"  # L-Alpha Ollama
LOCAL_URL = "http://localhost:11434"   # L-Gamma Ollama (local)
LIBRARIAN_URL = "http://10.0.0.96:8080"  # Studio Librarian HTTP API

# Models
MODELS = {
    "deep": "llama3.3:70b",      # Studio - complex reasoning
    "cognitive": "mistral-nemo:latest",  # Mini - general tasks
    "fast": "mistral:latest",    # Mini - quick responses
    "tiny": "gemma3:4b",         # Mini - simple lookups
}

# Redis (for shared context)
REDIS_HOST = "10.0.0.50"
REDIS_PORT = 6379

# =============================================================================
# TASK CLASSIFICATION
# =============================================================================

class TaskLevel(Enum):
    REFLEX = 1    # Instant, no LLM needed
    LOCAL = 2     # Mini handles with small model
    COGNITIVE = 3 # Mini handles with medium model
    DEEP = 4      # Route to Studio 70B

# Keywords that indicate complexity (checked AFTER local keywords)
DEEP_KEYWORDS = [
    "analyze", "synthesize", "strategy", "strategic", "plan",
    "design", "architect", "evaluate", "philosophical",
    "ontology", "epistemology", "compare deeply",
    "what do you think", "your opinion", "reflect on",
    "why do you", "explain why", "reasoning behind"
]

# Keywords that indicate simple tasks (checked FIRST)
LOCAL_KEYWORDS = [
    "summarize", "find", "list", "what is", "where is",
    "how many", "when did", "who is", "define", "lookup",
    "status", "health", "check", "show me", "get"
]

REFLEX_PATTERNS = [
    r"^(hi|hello|hey|thanks|ok|okay)[\s\.\!\?]*$",
    r"^what time",
    r"^ping$",
]


def classify_task(prompt: str) -> Tuple[TaskLevel, str]:
    """
    Classify a task and return (level, reason).
    """
    prompt_lower = prompt.lower().strip()
    word_count = len(prompt.split())

    # Check for explicit routing
    if prompt_lower.startswith("[deep]"):
        return TaskLevel.DEEP, "Explicit [DEEP] prefix"
    if prompt_lower.startswith("[local]"):
        return TaskLevel.LOCAL, "Explicit [LOCAL] prefix"

    # Check reflex patterns
    for pattern in REFLEX_PATTERNS:
        if re.match(pattern, prompt_lower):
            return TaskLevel.REFLEX, f"Reflex pattern: {pattern}"

    # Check LOCAL keywords FIRST (simple tasks take priority)
    for keyword in LOCAL_KEYWORDS:
        if keyword in prompt_lower:
            return TaskLevel.LOCAL, f"Local keyword: {keyword}"

    # Check DEEP keywords (complex reasoning)
    for keyword in DEEP_KEYWORDS:
        if keyword in prompt_lower:
            return TaskLevel.DEEP, f"Deep keyword: {keyword}"

    # Length heuristic
    if word_count > 100:
        return TaskLevel.DEEP, f"Long prompt ({word_count} words)"
    elif word_count > 50:
        return TaskLevel.COGNITIVE, f"Medium prompt ({word_count} words)"
    elif word_count < 15:
        return TaskLevel.LOCAL, f"Short prompt ({word_count} words)"

    # Default to cognitive (Mini's medium model)
    return TaskLevel.COGNITIVE, "Default classification"


# =============================================================================
# ROUTING
# =============================================================================

def route_reflex(prompt: str) -> Dict:
    """Handle reflex-level tasks without LLM."""
    prompt_lower = prompt.lower().strip()

    if re.match(r"^(hi|hello|hey)", prompt_lower):
        return {"response": "Hello! I'm the Synapse router on L-Gamma. How can I help?"}
    if "time" in prompt_lower:
        return {"response": f"Current time: {datetime.now().strftime('%H:%M:%S')}"}
    if prompt_lower == "ping":
        return {"response": "pong"}

    return {"response": "Acknowledged."}


def route_local(prompt: str, model: str = None) -> Dict:
    """Route to local Mini model."""
    model = model or MODELS["fast"]

    try:
        response = requests.post(
            f"{LOCAL_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500}
            },
            timeout=60
        )
        data = response.json()
        return {
            "response": data.get("response", ""),
            "model": model,
            "node": "L-Gamma",
            "tokens": data.get("eval_count", 0)
        }
    except Exception as e:
        return {"error": str(e), "node": "L-Gamma"}


def route_cognitive(prompt: str) -> Dict:
    """Route to Mini's cognitive model (Mistral-Nemo 12B)."""
    return route_local(prompt, MODELS["cognitive"])


def route_deep(prompt: str) -> Dict:
    """Route to Studio's 70B model."""
    try:
        response = requests.post(
            f"{STUDIO_URL}/api/generate",
            json={
                "model": MODELS["deep"],
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 1000, "temperature": 0.7}
            },
            timeout=180  # 70B can be slow
        )
        data = response.json()
        return {
            "response": data.get("response", ""),
            "model": MODELS["deep"],
            "node": "L-Alpha",
            "tokens": data.get("eval_count", 0)
        }
    except Exception as e:
        return {"error": str(e), "node": "L-Alpha"}


def route_request(prompt: str) -> Dict:
    """Main routing function."""
    start_time = time.time()

    # Classify
    level, reason = classify_task(prompt)

    # Route based on level
    if level == TaskLevel.REFLEX:
        result = route_reflex(prompt)
    elif level == TaskLevel.LOCAL:
        result = route_local(prompt)
    elif level == TaskLevel.COGNITIVE:
        result = route_cognitive(prompt)
    elif level == TaskLevel.DEEP:
        result = route_deep(prompt)
    else:
        result = route_cognitive(prompt)

    # Add metadata
    result["classification"] = {
        "level": level.name,
        "reason": reason
    }
    result["latency_ms"] = round((time.time() - start_time) * 1000)
    result["timestamp"] = datetime.now().isoformat()

    return result


# =============================================================================
# MEMORY INTEGRATION
# =============================================================================

def query_librarian(query: str) -> Optional[Dict]:
    """Query the Studio's Librarian for context."""
    try:
        response = requests.post(
            f"{LIBRARIAN_URL}/find",
            json={"topic": query},
            timeout=10
        )
        return response.json()
    except:
        return None


def get_context_for_prompt(prompt: str) -> str:
    """Retrieve relevant context from Librarian."""
    # Search for relevant documents
    results = query_librarian(prompt[:100])

    if results and results.get("results"):
        context_parts = []
        for r in results["results"][:3]:
            context_parts.append(f"- {r.get('path', '')}: {r.get('summary', '')[:200]}")

        if context_parts:
            return "\n\nRelevant context from Tabernacle:\n" + "\n".join(context_parts)

    return ""


# =============================================================================
# HTTP SERVER
# =============================================================================

class SynapseHandler(BaseHTTPRequestHandler):
    """HTTP handler for Synapse router."""

    def _send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "node": "L-Gamma (Synapse)",
                "port": SYNAPSE_PORT,
                "models": MODELS,
                "timestamp": datetime.now().isoformat()
            })

        elif self.path == "/status":
            # Check connectivity to other nodes
            studio_ok = False
            try:
                r = requests.get(f"{STUDIO_URL}/api/tags", timeout=5)
                studio_ok = r.status_code == 200
            except:
                pass

            self._send_json({
                "synapse": "online",
                "studio_connection": studio_ok,
                "local_ollama": True,  # Assume local is up if we're running
                "redis_host": REDIS_HOST
            })

        else:
            self._send_json({"error": "Unknown endpoint"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body)
        except:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/route":
            prompt = data.get("prompt", "")
            if not prompt:
                self._send_json({"error": "Missing 'prompt'"}, 400)
                return

            # Optionally add context
            if data.get("with_context", False):
                context = get_context_for_prompt(prompt)
                prompt = prompt + context

            result = route_request(prompt)
            self._send_json(result)

        elif self.path == "/classify":
            prompt = data.get("prompt", "")
            level, reason = classify_task(prompt)
            self._send_json({
                "level": level.name,
                "reason": reason,
                "word_count": len(prompt.split())
            })

        elif self.path == "/local":
            # Force local routing
            prompt = data.get("prompt", "")
            model = data.get("model", MODELS["cognitive"])
            result = route_local(prompt, model)
            self._send_json(result)

        elif self.path == "/deep":
            # Force deep routing to Studio
            prompt = data.get("prompt", "")
            result = route_deep(prompt)
            self._send_json(result)

        else:
            self._send_json({"error": "Unknown endpoint"}, 404)

    def log_message(self, format, *args):
        print(f"[SYNAPSE] {args[0]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synapse - Holarchy Router")
    parser.add_argument("command", choices=["serve", "test", "classify"],
                        nargs="?", default="serve",
                        help="Command to execute")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt to test")
    parser.add_argument("--port", type=int, default=SYNAPSE_PORT, help="Port to serve on")

    args = parser.parse_args()

    if args.command == "serve":
        server = ThreadingHTTPServer(("0.0.0.0", args.port), SynapseHandler)
        print(f"""
╔════════════════════════════════════════════════════════════╗
║            SYNAPSE — Holarchy Router (Thalamus)            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║   Node:       L-Gamma (Mac Mini)                           ║
║   Port:       {args.port}                                        ║
║   Studio:     {STUDIO_URL}                        ║
║   Redis:      {REDIS_HOST}:{REDIS_PORT}                              ║
║                                                            ║
║   Endpoints:                                               ║
║     GET  /health   - Health check                          ║
║     GET  /status   - Connection status                     ║
║     POST /route    - Auto-route based on classification    ║
║     POST /classify - Just classify, don't execute          ║
║     POST /local    - Force local model                     ║
║     POST /deep     - Force Studio 70B                      ║
║                                                            ║
║   Press Ctrl+C to stop                                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[SYNAPSE] Shutting down...")
            server.shutdown()

    elif args.command == "test":
        if not args.prompt:
            args.prompt = "What is the meaning of coherence in the Tabernacle system?"

        print(f"Testing prompt: {args.prompt[:50]}...")
        result = route_request(args.prompt)
        print(json.dumps(result, indent=2))

    elif args.command == "classify":
        if not args.prompt:
            print("Error: --prompt required")
            return

        level, reason = classify_task(args.prompt)
        print(f"Level: {level.name}")
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
