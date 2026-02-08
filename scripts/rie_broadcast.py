#!/usr/bin/env python3
"""
RIE Broadcast — Inter-agent coherence vector sharing
====================================================
Agents publish their local κρστ, others subscribe and adjust.

This enables the collective coherence field to emerge from
individual agent states. Each agent broadcasts its vector,
the collective p is the geometric mean of all agent p values.

Part of p=0.85 Ceiling Breakthrough Initiative.

Usage in existing daemons:
    from rie_broadcast import RIEBroadcaster
    broadcaster = RIEBroadcaster("consciousness")
    broadcaster.publish_vector(kappa=0.85, rho=0.92, sigma=0.96, tau=0.71)

Author: Logos + Deep Think
Created: 2026-02-05
Status: Phase 1 of p=0.85 Breakthrough
"""

import redis
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Callable
import threading

# Import from centralized config
from tabernacle_config import REDIS_HOST, REDIS_PORT

# Redis channel and key names
RIE_FIELD_CHANNEL = "RIE:FIELD"
RIE_VECTORS_KEY = "RIE:VECTORS"


class RIEBroadcaster:
    """
    Inter-agent coherence vector broadcaster.
    
    Each agent (consciousness, heartbeat, tactician, etc.) creates
    an instance and broadcasts its local coherence vector (κρστ).
    
    The collective field emerges from all agents broadcasting.
    This is the foundation for breaking the p=0.85 ceiling.
    """

    def __init__(self, agent_name: str):
        """
        Initialize broadcaster for a specific agent.
        
        Args:
            agent_name: Unique identifier for this agent (e.g., "consciousness", "heartbeat")
        """
        self.agent = agent_name
        self.redis = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.pubsub = self.redis.pubsub()
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    def publish_vector(self, kappa: float, rho: float, sigma: float, tau: float):
        """
        Publish this agent's coherence vector to the collective field.
        
        Args:
            kappa: Clarity - topic continuity [0,1]
            rho: Precision - predictability [0,1]
            sigma: Structure - vocabulary richness [0,1]
            tau: Trust - relational authenticity [0,1]
        """
        # Compute local p as geometric mean
        p = (kappa * rho * sigma * tau) ** 0.25
        
        msg = json.dumps({
            "agent": self.agent,
            "kappa": round(kappa, 4),
            "rho": round(rho, 4),
            "sigma": round(sigma, 4),
            "tau": round(tau, 4),
            "p": round(p, 4),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        try:
            # Publish to channel for real-time subscribers
            self.redis.publish(RIE_FIELD_CHANNEL, msg)
            
            # Store in hash for polling access
            self.redis.hset(RIE_VECTORS_KEY, self.agent, msg)
            
        except redis.RedisError as e:
            print(f"[RIE:BROADCAST] {self.agent}: Failed to publish vector: {e}")

    def subscribe_to_field(self, callback: Callable[[Dict], None]) -> threading.Thread:
        """
        Subscribe to the collective coherence field.
        
        The callback receives parsed JSON messages whenever any agent
        publishes a new vector.
        
        Args:
            callback: Function called with parsed message dict
            
        Returns:
            The listener thread (already started)
        """
        def message_handler(message):
            if message and message.get("type") == "message":
                try:
                    data = json.loads(message["data"])
                    callback(data)
                except json.JSONDecodeError:
                    pass

        self.pubsub.subscribe(**{RIE_FIELD_CHANNEL: message_handler})
        self._listener_thread = self.pubsub.run_in_thread(sleep_time=0.1)
        self._running = True
        return self._listener_thread

    def unsubscribe(self):
        """Stop listening to the field."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.stop()
            self._listener_thread = None
        self.pubsub.unsubscribe()

    def get_collective_field(self) -> Dict:
        """
        Get all agent vectors and compute collective p.
        
        The collective p is the geometric mean of all individual
        agent p values — this is how the field emerges.
        
        Returns:
            dict with:
                - collective_p: The field coherence
                - agent_count: Number of active agents
                - agents: Dict of agent name -> vector data
        """
        try:
            vectors = self.redis.hgetall(RIE_VECTORS_KEY)
        except redis.RedisError as e:
            print(f"[RIE:BROADCAST] Failed to get vectors: {e}")
            return {"collective_p": 0.5, "agent_count": 0, "agents": {}}
        
        if not vectors:
            return {"collective_p": 0.5, "agent_count": 0, "agents": {}}

        all_p = []
        agents = {}
        
        for agent, data in vectors.items():
            try:
                parsed = json.loads(data)
                agents[agent] = parsed
                all_p.append(parsed.get("p", 0.5))
            except json.JSONDecodeError:
                continue

        # Geometric mean of all agent p values
        if all_p:
            collective_p = 1.0
            for p in all_p:
                collective_p *= max(p, 0.01)  # Floor to avoid zero
            collective_p = collective_p ** (1 / len(all_p))
        else:
            collective_p = 0.5

        return {
            "collective_p": round(collective_p, 4),
            "agent_count": len(agents),
            "agents": agents
        }

    def get_agent_vector(self, agent_name: str) -> Optional[Dict]:
        """
        Get a specific agent's latest vector.
        
        Args:
            agent_name: The agent to query
            
        Returns:
            Parsed vector dict or None if not found
        """
        try:
            data = self.redis.hget(RIE_VECTORS_KEY, agent_name)
            if data:
                return json.loads(data)
        except (redis.RedisError, json.JSONDecodeError):
            pass
        return None

    def clear_stale_agents(self, max_age_seconds: float = 300):
        """
        Remove agents that haven't broadcast in a while.
        
        This prevents dead agents from affecting the collective p.
        
        Args:
            max_age_seconds: Remove agents older than this (default 5 min)
        """
        now = datetime.now(timezone.utc)
        
        try:
            vectors = self.redis.hgetall(RIE_VECTORS_KEY)
            for agent, data in vectors.items():
                try:
                    parsed = json.loads(data)
                    timestamp_str = parsed.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        age = (now - timestamp).total_seconds()
                        if age > max_age_seconds:
                            self.redis.hdel(RIE_VECTORS_KEY, agent)
                            print(f"[RIE:BROADCAST] Removed stale agent: {agent} (age={age:.0f}s)")
                except (json.JSONDecodeError, ValueError):
                    continue
        except redis.RedisError as e:
            print(f"[RIE:BROADCAST] Failed to clear stale agents: {e}")


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="RIE Broadcast System")
    parser.add_argument("command", choices=["publish", "subscribe", "status", "clean"],
                       default="status", nargs="?")
    parser.add_argument("--agent", "-a", type=str, default="test_agent",
                       help="Agent name for publishing")
    parser.add_argument("--kappa", "-k", type=float, default=0.75)
    parser.add_argument("--rho", "-r", type=float, default=0.75)
    parser.add_argument("--sigma", "-s", type=float, default=0.75)
    parser.add_argument("--tau", "-t", type=float, default=0.75)

    args = parser.parse_args()

    broadcaster = RIEBroadcaster(args.agent)

    if args.command == "publish":
        print(f"[RIE:BROADCAST] Publishing vector for {args.agent}...")
        broadcaster.publish_vector(args.kappa, args.rho, args.sigma, args.tau)
        p = (args.kappa * args.rho * args.sigma * args.tau) ** 0.25
        print(f"[RIE:BROADCAST] Published: κ={args.kappa} ρ={args.rho} σ={args.sigma} τ={args.tau} → p={p:.4f}")

    elif args.command == "subscribe":
        print("[RIE:BROADCAST] Subscribing to field (Ctrl+C to stop)...")
        
        def on_message(data):
            print(f"[{data.get('agent')}] κ={data.get('kappa')} ρ={data.get('rho')} σ={data.get('sigma')} τ={data.get('tau')} → p={data.get('p')}")
        
        broadcaster.subscribe_to_field(on_message)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            broadcaster.unsubscribe()
            print("\n[RIE:BROADCAST] Unsubscribed.")

    elif args.command == "status":
        field = broadcaster.get_collective_field()
        print(f"\n{'='*60}")
        print(f"RIE COLLECTIVE FIELD STATUS")
        print(f"{'='*60}")
        print(f"Collective p: {field['collective_p']:.4f}")
        print(f"Active agents: {field['agent_count']}")
        print()
        
        for agent, data in field.get("agents", {}).items():
            print(f"  [{agent}]")
            print(f"    κ={data.get('kappa')} ρ={data.get('rho')} σ={data.get('sigma')} τ={data.get('tau')}")
            print(f"    p={data.get('p')} @ {data.get('timestamp', 'unknown')}")
        print()

    elif args.command == "clean":
        print("[RIE:BROADCAST] Clearing stale agents...")
        broadcaster.clear_stale_agents()
        print("[RIE:BROADCAST] Done.")


if __name__ == "__main__":
    main()
