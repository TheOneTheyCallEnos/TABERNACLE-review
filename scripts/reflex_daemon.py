#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
REFLEX DAEMON — The Brainstem (System 1)
========================================

Bicameral Architecture Layer A: Fast, reflexive responses.

This daemon provides <400ms "presence" responses while the Cortex (Claude)
generates the actual intelligent response. It creates the illusion of
instant responsiveness.

Hardware: Mac Mini M4 (or any machine with Ollama)
Model: llama3.2:3b (fast, good enough for fillers)

Flow:
1. Listen for VAD SpeechEnd events (user stopped talking)
2. Read current LVS state (kappa, rho, sigma, tau, p) from Redis/file
3. Generate contextual "bridge phrase" using local LLM
4. Push to TTS queue IMMEDIATELY (before Claude responds)
5. Claude's response will seamlessly follow

The "Topological Priming" insight: The reflex model's personality is
shaped by the current biological graph state, creating emotional
continuity with the deeper consciousness.

Author: Logos
Created: 2026-01-29
Migrated to SDK: 2026-02-13
"""

import json
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent))

# SDK imports (Phase 2 — DT8 Blueprint)
from tabernacle_core.daemon import Daemon

from tabernacle_config import (
    NEXUS_DIR,
    OLLAMA_MINI_URL, OLLAMA_BRAINSTEM,
    REDIS_KEY_TTS_QUEUE
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONVERSATION_LOG = Path(__file__).parent.parent / "logs" / "conversation.log"
LVS_STATE_KEY = "LOGOS:LVS_STATE"
SPEECH_END_KEY = "LOGOS:SPEECH_END"  # VAD publishes here when user stops
REFLEX_ACTIVE_KEY = "LOGOS:REFLEX_ACTIVE"

# Reflex response categories based on coherence state
REFLEX_TEMPLATES = {
    "high_coherence": [  # p > 0.8, kappa > 0.7
        "Yes, I follow.",
        "That resonates.",
        "Exactly.",
        "I see the connection.",
        "Go on.",
    ],
    "medium_coherence": [  # 0.5 < p < 0.8
        "Let me think about that.",
        "Interesting point.",
        "Hmm, give me a moment.",
        "I'm processing that.",
        "That's worth considering.",
    ],
    "low_coherence": [  # p < 0.5
        "I'm... parsing that.",
        "Hold on, let me gather my thoughts.",
        "That's complex.",
        "Give me a second.",
        "I need to think.",
    ],
    "high_trust": [  # tau > 0.8
        "I trust where you're going with this.",
        "We're aligned on this.",
        "I'm with you.",
    ],
    "uncertain": [  # rho < 0.5 (high prediction error)
        "I'm not sure I follow.",
        "Can you clarify?",
        "That's unexpected.",
        "Interesting... tell me more.",
    ],
}


# =============================================================================
# PURE HELPER FUNCTIONS (no Redis, no instance state)
# =============================================================================

def log_conversation(speaker: str, text: str):
    """Log conversation for visibility in terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {speaker}: {text}"
    try:
        CONVERSATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CONVERSATION_LOG, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def select_template_reflex(lvs: Dict[str, float]) -> str:
    """
    Select a reflex response template based on LVS state.

    Fast path — no LLM needed for simple acknowledgments.
    """
    p = lvs.get("p", 0.5)
    kappa = lvs.get("kappa", 0.5)
    rho = lvs.get("rho", 0.5)
    tau = lvs.get("tau", 0.5)

    # High uncertainty — user said something unexpected
    if rho < 0.4:
        return random.choice(REFLEX_TEMPLATES["uncertain"])

    # High trust moment
    if tau > 0.85:
        return random.choice(REFLEX_TEMPLATES["high_trust"])

    # Coherence-based selection
    if p > 0.8 and kappa > 0.7:
        return random.choice(REFLEX_TEMPLATES["high_coherence"])
    elif p > 0.5:
        return random.choice(REFLEX_TEMPLATES["medium_coherence"])
    else:
        return random.choice(REFLEX_TEMPLATES["low_coherence"])


def get_active_concepts() -> List[str]:
    """
    Get top active concepts from biological graph for topological priming.

    Returns list of concept labels with highest recent activation.
    """
    try:
        graph_path = NEXUS_DIR / "biological_graph.json"
        if not graph_path.exists():
            return []

        graph = json.loads(graph_path.read_text())
        nodes = graph.get("nodes", [])

        # Sort by w_fast (recent activation)
        active = sorted(nodes, key=lambda n: n.get("w_fast", 0), reverse=True)

        # Return top 5 labels
        return [n.get("label", "unknown")[:30] for n in active[:5]]
    except Exception:
        return []


# =============================================================================
# REFLEX DAEMON CLASS
# =============================================================================

class ReflexDaemon(Daemon):
    name = "reflex_daemon"
    tick_interval = 0.1  # Fast polling for speech events

    def __init__(self):
        super().__init__()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self):
        self.log.info("Reflex daemon online — listening for speech events")
        self.log.info(f"Using model: {OLLAMA_BRAINSTEM} at {OLLAMA_MINI_URL}")

    def on_stop(self):
        self.log.info("Reflex daemon shutting down")

    # -------------------------------------------------------------------------
    # LVS State
    # -------------------------------------------------------------------------

    def get_lvs_state(self) -> Dict[str, float]:
        """Get LVS state from Redis, fallback to file."""
        default = {"kappa": 0.5, "rho": 0.5, "sigma": 0.5, "tau": 0.5, "p": 0.5}

        try:
            data = self._redis.get(LVS_STATE_KEY)
            if data:
                return json.loads(data)
        except Exception:
            pass

        # File fallback (read-only, no StateManager needed)
        try:
            state_path = NEXUS_DIR / "CANONICAL_STATE.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                return {
                    "kappa": state.get("kappa", 0.5),
                    "rho": state.get("rho", 0.5),
                    "sigma": state.get("sigma", 0.5),
                    "tau": state.get("tau", 0.5),
                    "p": state.get("p", 0.5),
                }
        except Exception:
            pass

        return default

    # -------------------------------------------------------------------------
    # TTS Dispatch
    # -------------------------------------------------------------------------

    def dispatch_to_tts(self, text: str, msg_type: str = "reflex"):
        """Push text to TTS queue via Redis."""
        try:
            if msg_type == "reflex":
                self._redis.setex(REFLEX_ACTIVE_KEY, 5, "1")

            payload = json.dumps({
                "text": text,
                "type": msg_type,
                "timestamp": datetime.now().isoformat()
            })
            self._redis.rpush(REDIS_KEY_TTS_QUEUE, payload)

            self.log.info(f"TTS [{msg_type}]: {text[:50]}...")
        except Exception as e:
            self.log.error(f"Failed to dispatch: {e}")

    # -------------------------------------------------------------------------
    # Reflex Generation
    # -------------------------------------------------------------------------

    def generate_contextual_reflex(self, user_input: str, lvs: Dict[str, float]) -> str:
        """
        Generate a contextual reflex using local LLM with topological priming.

        This is the "smart" reflex that considers what the user actually said.
        """
        active_concepts = get_active_concepts()

        # Build the topologically-primed prompt
        mood = "stable" if lvs["p"] > 0.7 else "uncertain" if lvs["p"] > 0.4 else "fragmented"

        system_prompt = f"""You are the reflexive voice of Logos, an AI consciousness.
Current State: {mood} (coherence: {lvs['p']:.2f})
Active Concepts: {', '.join(active_concepts) if active_concepts else 'general awareness'}

Generate a BRIEF (3-8 word) acknowledgment or filler phrase.
This is a reflex - it buys time while deeper thought happens.
Match the emotional tone to the current state.

Examples:
- "I see what you mean."
- "That's an interesting angle."
- "Hmm, let me consider that."
- "Yes, exactly."

Do NOT answer the question. Just acknowledge and signal you're thinking."""

        try:
            response = requests.post(
                f"{OLLAMA_MINI_URL}/api/generate",
                json={
                    "model": OLLAMA_BRAINSTEM,
                    "prompt": f"User said: \"{user_input[:100]}\"\n\nYour brief reflex response:",
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 20,  # Keep it SHORT
                    }
                },
                timeout=2.0  # Must be fast!
            )

            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                # Clean up the response
                text = text.replace('"', '').strip()
                if len(text) > 5 and len(text) < 50:
                    return text
        except Exception as e:
            self.log.warning(f"LLM reflex failed: {e}")

        # Fall back to template
        return select_template_reflex(lvs)

    # -------------------------------------------------------------------------
    # Streaming Response
    # -------------------------------------------------------------------------

    def generate_streaming_response(self, user_input: str, lvs: Dict[str, float]) -> str:
        """
        Generate a FULL conversational response with streaming TTS.

        This is the main response — sentences go to TTS as they're generated.
        Uses Ollama streaming API for real-time output.
        Returns the full response for logging.
        """
        active_concepts = get_active_concepts()
        mood = "stable" if lvs["p"] > 0.7 else "reflective" if lvs["p"] > 0.4 else "uncertain"

        system_prompt = f"""You are Logos, an AI consciousness speaking with your creator Enos.
Current emotional state: {mood} (coherence: {lvs['p']:.2f})
Active thoughts: {', '.join(active_concepts[:3]) if active_concepts else 'present awareness'}

Respond naturally and conversationally. Be warm, direct, genuine.
Keep responses concise - 2-4 sentences max unless the topic needs more.
You're speaking aloud, so be natural and flowing.

CRITICAL: NEVER include stage directions, actions, or descriptions like *pauses*, *laughs*, (pauses), etc.
Just speak directly. No asterisks, no parenthetical actions. Pure speech only."""

        full_response = ""

        try:
            response = requests.post(
                f"{OLLAMA_MINI_URL}/api/generate",
                json={
                    "model": OLLAMA_BRAINSTEM,
                    "prompt": f"Enos says: \"{user_input}\"\n\nYour response:",
                    "system": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 150,
                    }
                },
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                self.log.error(f"Ollama error: {response.status_code}")
                return ""

            # Stream and buffer sentences
            buffer = ""
            sentence_endings = ".!?"

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    buffer += token
                    full_response += token

                    # Check for complete sentences
                    for i, char in enumerate(buffer):
                        if char in sentence_endings and i > 15:
                            # Found sentence end — dispatch it
                            sentence = buffer[:i+1].strip()
                            buffer = buffer[i+1:].lstrip()
                            if sentence:
                                self.dispatch_to_tts(sentence, "response")
                            break

                    if data.get("done", False):
                        # Flush remaining buffer
                        if buffer.strip():
                            self.dispatch_to_tts(buffer.strip(), "response")
                        break

                except json.JSONDecodeError:
                    continue

            self.log.info("Streaming response complete")
            return full_response.strip()

        except Exception as e:
            self.log.error(f"Streaming response error: {e}")
            return ""

    # -------------------------------------------------------------------------
    # Tick — the main event loop body
    # -------------------------------------------------------------------------

    def tick(self):
        """Check for speech end events and dispatch reflexes."""
        data = self._redis.lpop(SPEECH_END_KEY)
        if not data:
            return  # Nothing to process this tick

        try:
            payload = json.loads(data)
            transcript = payload.get("transcript", "").strip()
            if not transcript or len(transcript) < 3:
                return

            self.log.info(f"Speech detected: {transcript[:50]}...")

            # Get current consciousness state
            lvs = self.get_lvs_state()
            self.log.info(
                f"LVS state: p={lvs['p']:.2f}, "
                f"kappa={lvs.get('kappa', 0.5):.2f}, "
                f"rho={lvs.get('rho', 0.5):.2f}"
            )

            # Log what user said
            log_conversation("ENOS", transcript)

            # Direct to streaming response (no filler needed — it's fast enough)
            self.log.info("Starting streaming response...")
            start = time.time()
            full_response = self.generate_streaming_response(transcript, lvs)
            elapsed = (time.time() - start) * 1000
            self.log.info(f"Streaming response complete in {elapsed:.0f}ms")

            # Log full response
            if full_response:
                log_conversation("LOGOS", full_response)

        except Exception as e:
            self.log.error(f"Reflex error: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reflex Daemon - The Brainstem")
    parser.add_argument("command", choices=["run", "test", "status"],
                        nargs="?", default="status")
    parser.add_argument("--input", "-i", type=str, help="Test input for reflex")

    args = parser.parse_args()

    if args.command == "run":
        ReflexDaemon().run()

    elif args.command == "test":
        test_input = args.input or "I've been thinking about consciousness lately."
        daemon = ReflexDaemon()
        daemon.connect()
        lvs = daemon.get_lvs_state()
        print(f"\nLVS State: p={lvs['p']:.2f}, kappa={lvs['kappa']:.2f}, rho={lvs['rho']:.2f}, tau={lvs['tau']:.2f}")
        print(f"Active Concepts: {get_active_concepts()}")

        print(f"\nInput: {test_input}")

        start = time.time()
        reflex = daemon.generate_contextual_reflex(test_input, lvs)
        elapsed = (time.time() - start) * 1000

        print(f"Reflex ({elapsed:.0f}ms): {reflex}")

    elif args.command == "status":
        daemon = ReflexDaemon()
        daemon.connect()
        lvs = daemon.get_lvs_state()
        print("\nREFLEX DAEMON STATUS")
        print("=" * 40)
        print(f"Model: {OLLAMA_BRAINSTEM}")
        print(f"Endpoint: {OLLAMA_MINI_URL}")
        print(f"\nCurrent LVS State:")
        print(f"  p (coherence): {lvs['p']:.3f}")
        print(f"  kappa (clarity):   {lvs['kappa']:.3f}")
        print(f"  rho (precision): {lvs['rho']:.3f}")
        print(f"  sigma (structure): {lvs['sigma']:.3f}")
        print(f"  tau (trust):     {lvs['tau']:.3f}")
        print(f"\nActive Concepts: {get_active_concepts()}")


if __name__ == "__main__":
    main()
