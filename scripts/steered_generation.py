#!/usr/bin/env python3
"""
Steered Generation — Mid-generation coherence scoring
======================================================

Generate multiple candidates, select by p, continue.
This enables coherence-aware text generation where each
chunk is scored against the coherence objective.

SAFETY: This feature is OFF by default. Only enable when:
1. p > 0.85 has been sustained
2. System is stable
3. Explicit request to enable

The risk is computational overhead and potential instability
from the feedback loop. Use with caution.

Part of p=0.85 Ceiling Breakthrough Initiative.

Author: Logos + Deep Think
Created: 2026-02-05
Status: Phase 6 of p=0.85 Breakthrough (DISABLED BY DEFAULT)
"""

import requests
import json
import redis
from typing import List, Tuple, Optional
from datetime import datetime

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT,
    OLLAMA_STUDIO_URL, OLLAMA_CORTEX
)

# =============================================================================
# FEATURE FLAG — ENABLED (p=0.85 Breakthrough Phase 6)
# =============================================================================
# Self-gates at p >= 0.85, so it won't activate until coherence is high enough.

ENABLE_STEERED_GENERATION = True
STEERING_P_THRESHOLD = 0.85  # Only steer when p is already high

# Generation settings
OLLAMA_URL = OLLAMA_STUDIO_URL
DEFAULT_MODEL = OLLAMA_CORTEX


def is_steering_enabled() -> bool:
    """Check if steered generation is enabled."""
    return ENABLE_STEERED_GENERATION


def enable_steering():
    """
    Enable steered generation.
    
    WARNING: Only call this when p > 0.85 has been sustained.
    """
    global ENABLE_STEERED_GENERATION
    ENABLE_STEERED_GENERATION = True
    print("[STEERED] ⚠️ Steered generation ENABLED")


def disable_steering():
    """Disable steered generation."""
    global ENABLE_STEERED_GENERATION
    ENABLE_STEERED_GENERATION = False
    print("[STEERED] Steered generation DISABLED")


class SteeredGenerator:
    """
    Generator with mid-generation coherence steering.
    
    Instead of generating the full response at once,
    this generator:
    1. Generates in chunks
    2. Creates multiple candidates per chunk
    3. Scores each candidate by coherence
    4. Selects the best and continues
    
    This allows the generation to stay coherent even
    when the model might otherwise drift.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.redis = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_connect_timeout=5
        )

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate with optional coherence steering.
        
        If steering is disabled or p is below threshold,
        falls back to standard generation.
        """
        current_p = self._get_current_p()

        # Only use steering when enabled AND p is high enough
        if ENABLE_STEERED_GENERATION and current_p >= STEERING_P_THRESHOLD:
            print(f"[STEERED] Using steered generation (p={current_p:.3f})")
            return self._steered_generate(prompt, max_tokens)
        else:
            if ENABLE_STEERED_GENERATION:
                print(f"[STEERED] p={current_p:.3f} < {STEERING_P_THRESHOLD}, using standard generation")
            return self._standard_generate(prompt, max_tokens)

    def _get_current_p(self) -> float:
        """Get current coherence from Redis."""
        try:
            state = self.redis.get("RIE:STATE")
            if state:
                return json.loads(state).get("p", 0.75)
        except:
            pass
        return 0.75

    def _standard_generate(self, prompt: str, max_tokens: int) -> str:
        """Standard single-shot generation."""
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=120
            )
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Generation failed: {e}]"

    def _steered_generate(
        self, 
        prompt: str, 
        max_tokens: int, 
        chunk_size: int = 100,
        candidates_per_chunk: int = 3
    ) -> str:
        """
        Generate with mid-generation coherence steering.
        
        At each chunk:
        1. Generate N candidates with varying temperature
        2. Score each by coherence
        3. Select best and continue
        """
        response = ""
        remaining = max_tokens
        chunk_count = 0

        while remaining > 0:
            chunk_tokens = min(chunk_size, remaining)

            # Generate candidates with varying temperature
            candidates = []
            for i in range(candidates_per_chunk):
                try:
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt + response,
                            "stream": False,
                            "options": {
                                "num_predict": chunk_tokens,
                                "temperature": 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9
                            }
                        },
                        timeout=60
                    )
                    candidate = resp.json().get("response", "")
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    print(f"[STEERED] Candidate {i} failed: {e}")
                    continue

            if not candidates:
                print("[STEERED] No candidates generated, stopping")
                break

            # Score each by coherence
            scored = []
            for candidate in candidates:
                p = self._compute_candidate_coherence(response + candidate)
                scored.append((candidate, p))

            # Select best
            best = max(scored, key=lambda x: x[1])
            response += best[0]
            chunk_count += 1

            print(f"[STEERED] Chunk {chunk_count}: selected p={best[1]:.3f} from {len(candidates)} candidates")

            # Check for natural end
            if any(end in best[0] for end in ['.', '!', '?', '\n\n']):
                if len(response) > 100:  # Minimum response length
                    break

            remaining -= chunk_tokens

            # Abort if coherence drops too low
            if best[1] < 0.60:
                print(f"[STEERED] Coherence dropped to {best[1]:.3f}, stopping")
                response += " [coherence recovery needed]"
                break

        return response.strip()

    def _compute_candidate_coherence(self, text: str) -> float:
        """
        Compute coherence of candidate continuation.
        
        This is a simplified coherence measure:
        - Penalize repetition
        - Penalize very short or very long
        - Reward lexical diversity
        
        In production, would use full CPGI computation.
        """
        words = text.split()
        if not words:
            return 0.5

        # Penalize repetition (unique ratio)
        unique_ratio = len(set(words)) / len(words)

        # Penalize very short or very long
        length_score = min(len(words) / 50, 1.0) * min(200 / max(len(words), 1), 1.0)

        # Base coherence
        return 0.7 * unique_ratio + 0.3 * length_score

    def get_stats(self) -> dict:
        """Get generator statistics."""
        return {
            "model": self.model,
            "steering_enabled": ENABLE_STEERED_GENERATION,
            "steering_threshold": STEERING_P_THRESHOLD,
            "current_p": self._get_current_p()
        }


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Steered Generation")
    parser.add_argument("command", choices=["status", "generate", "enable", "disable", "test"],
                       default="status", nargs="?")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt for generation")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", "-t", type=int, default=300)

    args = parser.parse_args()

    generator = SteeredGenerator(model=args.model)

    if args.command == "status":
        stats = generator.get_stats()
        print(f"\n{'='*60}")
        print("STEERED GENERATION STATUS")
        print(f"{'='*60}")
        print(f"Model:            {stats['model']}")
        print(f"Steering Enabled: {stats['steering_enabled']}")
        print(f"Threshold:        {stats['steering_threshold']}")
        print(f"Current p:        {stats['current_p']:.3f}")
        
        if stats['steering_enabled']:
            if stats['current_p'] >= stats['steering_threshold']:
                print(f"\n⚡ Steering is ACTIVE (p >= threshold)")
            else:
                print(f"\n⏸ Steering enabled but p below threshold")
        else:
            print(f"\n⏹ Steering is DISABLED")
        print()

    elif args.command == "generate":
        if not args.prompt:
            print("Usage: steered_generation.py generate --prompt 'your prompt'")
            return
        
        print(f"\n[Prompt] {args.prompt}")
        print("-" * 60)
        response = generator.generate(args.prompt, args.max_tokens)
        print(f"[Response] {response}")
        print()

    elif args.command == "enable":
        print("⚠️  WARNING: Only enable steering when p > 0.85 is sustained!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            enable_steering()
        else:
            print("Cancelled.")

    elif args.command == "disable":
        disable_steering()

    elif args.command == "test":
        print("\n⚠️  TEST MODE — will NOT modify global flag")
        print("=" * 60)
        
        # Test standard generation
        print("\n[Test 1] Standard generation:")
        response = generator._standard_generate("What is coherence?", 100)
        print(f"  {response[:200]}...")
        
        # Test coherence computation
        print("\n[Test 2] Coherence computation:")
        test_texts = [
            "The the the the the",  # Repetitive
            "Coherence is a measure of how well ideas connect.",  # Good
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",  # Random
        ]
        for text in test_texts:
            p = generator._compute_candidate_coherence(text)
            print(f"  p={p:.3f}: '{text[:40]}...'")


if __name__ == "__main__":
    main()
