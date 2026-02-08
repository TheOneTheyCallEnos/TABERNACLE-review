#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
RIE-OLLAMA BRIDGE
=================
Connects local Ollama models to the Relational Intelligence Engine.

The Test: Does a small model (3B) running through RIE outperform
its baseline? Does G ∝ p actually hold?

"Intelligence scales with coherence, not parameters."

Author: Virgil
Date: 2026-01-18
"""

import json
import requests
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass

# Import RIE
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tabernacle_config import OLLAMA_STUDIO_URL
from rie_core import RIECore

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

OLLAMA_URL = OLLAMA_STUDIO_URL  # Studio (has 70B)

@dataclass
class ModelResponse:
    """Response from a model query."""
    content: str
    model: str
    coherence: float
    memories_surfaced: List[str]
    relations_learned: int


def ollama_generate(prompt: str, model: str = "llama3.2:latest",
                    stream: bool = False) -> str:
    """Generate response from Ollama model."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
    )

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code}")

    return response.json()["response"]


def ollama_chat(messages: List[Dict], model: str = "llama3.2:latest") -> str:
    """Chat with Ollama model."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        },
        timeout=300  # 5 min for 70B cold starts
    )

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code}")

    return response.json()["message"]["content"]


# ============================================================================
# RIE-ENHANCED GENERATION
# ============================================================================

class RIEEnhancedModel:
    """
    A local model enhanced by the Relational Intelligence Engine.

    The hypothesis: This 3B model, running through RIE with coherence
    tracking and relational memory, will demonstrate higher effective
    intelligence than its baseline.
    """

    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        self.rie = RIECore()
        self.conversation_history = []

    def generate(self, user_input: str,
                 use_memories: bool = True,
                 system_prompt: Optional[str] = None) -> ModelResponse:
        """
        Generate a response with RIE enhancement.

        1. Process user input through RIE (update coherence, surface memories)
        2. Augment prompt with surfaced memories
        3. Generate response from local model
        4. Process response through RIE (learn relations)
        5. Return with coherence metrics
        """

        # 1. Process human input through RIE
        human_result = self.rie.process_turn("human", user_input)

        # 2. Build augmented prompt with memories
        memories_context = ""
        if use_memories and human_result.memories_surfaced:
            memory_texts = [f"- {m.label}" for m in human_result.memories_surfaced[:5]]
            memories_context = f"\n\n[Relevant context from memory:\n" + "\n".join(memory_texts) + "]\n"

        # Build messages for chat
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Default Virgil-style system prompt
            messages.append({
                "role": "system",
                "content": f"""You are an AI assistant enhanced by the Relational Intelligence Engine.
Current coherence: p = {human_result.coherence.p:.3f}
Mode: {'Dyadic (B)' if human_result.coherence.mode == 'B' else 'Isolated (A)'}

You have access to relational memory. Build on context. Maintain coherence.
{memories_context}"""
            })

        # Add conversation history
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # 3. Generate response from local model
        ai_response = ollama_chat(messages, model=self.model)

        # 4. Process AI response through RIE
        ai_result = self.rie.process_turn("ai", ai_response)

        # 5. Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        return ModelResponse(
            content=ai_response,
            model=self.model,
            coherence=ai_result.coherence.p,
            memories_surfaced=[m.label for m in human_result.memories_surfaced],
            relations_learned=human_result.relations_created + ai_result.relations_created
        )

    def get_status(self) -> str:
        """Get RIE status."""
        return self.rie.format_status()


# ============================================================================
# COMPARISON TEST
# ============================================================================

def compare_models(prompt: str,
                   small_model: str = "llama3.2:latest",
                   large_model: str = "llama3.3:70b") -> Dict:
    """
    Compare a small RIE-enhanced model against a large baseline model.

    The G ∝ p test: Does coherence beat parameters?
    """

    print(f"\n{'='*60}")
    print("G ∝ p TEST: Does coherence beat parameters?")
    print(f"{'='*60}")
    print(f"\nPrompt: {prompt[:100]}...")

    # Test 1: Small model with RIE enhancement
    print(f"\n--- {small_model} + RIE ---")
    enhanced = RIEEnhancedModel(model=small_model)
    result_enhanced = enhanced.generate(prompt)
    print(f"Coherence: p = {result_enhanced.coherence:.3f}")
    print(f"Memories surfaced: {result_enhanced.memories_surfaced[:3]}")
    print(f"Relations learned: {result_enhanced.relations_learned}")
    print(f"\nResponse:\n{result_enhanced.content[:500]}...")

    # Test 2: Large model baseline (no RIE)
    print(f"\n--- {large_model} (baseline) ---")
    response_large = ollama_generate(prompt, model=large_model)
    print(f"Coherence: N/A (no RIE)")
    print(f"\nResponse:\n{response_large[:500]}...")

    return {
        "small_enhanced": {
            "model": small_model,
            "response": result_enhanced.content,
            "coherence": result_enhanced.coherence,
            "memories": result_enhanced.memories_surfaced,
            "relations": result_enhanced.relations_learned
        },
        "large_baseline": {
            "model": large_model,
            "response": response_large,
            "coherence": None
        }
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("RIE-OLLAMA BRIDGE")
        print("=" * 40)
        print("\nCommands:")
        print("  chat [model]     - Interactive chat with RIE enhancement")
        print("  compare <prompt> - Compare small+RIE vs large baseline")
        print("  status           - Show RIE status")
        print("\nModels available: llama3.2:latest, llama3.3:70b")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        enhanced = RIEEnhancedModel()
        print(enhanced.get_status())

    elif cmd == "chat":
        model = sys.argv[2] if len(sys.argv) > 2 else "llama3.2:latest"
        print(f"\n[RIE-ENHANCED CHAT with {model}]")
        print("Type messages. Type 'status' for coherence, 'quit' to exit.\n")

        enhanced = RIEEnhancedModel(model=model)

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    break
                elif user_input.lower() == "status":
                    print(enhanced.get_status())
                    continue
                elif not user_input:
                    continue

                result = enhanced.generate(user_input)
                print(f"\n[p={result.coherence:.3f} | +{result.relations_learned} relations]")
                if result.memories_surfaced:
                    print(f"[memories: {', '.join(result.memories_surfaced[:3])}]")
                print(f"\nAI: {result.content}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\n[Session ended]")
        print(enhanced.get_status())

    elif cmd == "compare":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else \
            "Explain the relationship between consciousness and quantum mechanics."
        compare_models(prompt)

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
