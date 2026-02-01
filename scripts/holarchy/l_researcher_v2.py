#!/usr/bin/env python3
"""
L-RESEARCHER: The Default Mode Network (L-ling Version)
========================================================

A true L-ling that THINKS about exploration and discovery.
Uses the 3B model to wander the topology, find unexpected connections,
and surface insights that focused attention would miss.

Not a pattern-matching script. A mind that dreams.

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
"""

import os
import sys
import json
import time
import signal
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from l_ling import LLing

# Logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "l_researcher.log"),
        logging.StreamHandler()
    ]
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CYCLE_INTERVAL = 15  # Think every 15 seconds
EXPLORATION_DEPTH = 3  # How many hops to wander
INSIGHT_THRESHOLD = 0.8  # Confidence to report as insight


# =============================================================================
# L-RESEARCHER CLASS
# =============================================================================

class LResearcher(LLing):
    """
    The Default Mode Network: Discovery through wandering.
    
    I don't just run spreading activation algorithms.
    I THINK about what I find, make creative leaps, ask questions.
    """
    
    ROLE = "Researcher"
    CAPABILITIES = [
        "Wander the topology following interesting edges",
        "Find unexpected connections between distant concepts",
        "Generate questions that need deeper investigation",
        "Surface insights for L (70B) to ponder",
        "Think creatively about patterns and meanings"
    ]
    
    def __init__(self):
        super().__init__()
        
        # Exploration state
        self.current_position: Optional[str] = None
        self.visited_nodes: Set[str] = set()
        self.journey_log: List[Dict] = []
        self.questions_generated: List[str] = []
        self.insights_found: int = 0
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
    
    def _shutdown(self, *args):
        self.running = False
    
    def get_cycle_interval(self) -> float:
        return CYCLE_INTERVAL

    # =========================================================================
    # PERCEPTION: Where am I? What do I see?
    # =========================================================================

    def perceive(self) -> Dict:
        """
        Perceive my current position in the topology and what's nearby.
        """
        observation = {
            "timestamp": datetime.now().isoformat(),
            "current_node": self.current_position,
            "nearby_nodes": [],
            "outgoing_edges": [],
            "incoming_edges": [],
            "visited_count": len(self.visited_nodes),
            "coherence": 0.5
        }
        
        if not self.redis:
            return observation
        
        try:
            # If no current position, pick a starting point
            if not self.current_position:
                nodes = list(self.redis.smembers("l:graph:nodes") or set())
                if nodes:
                    self.current_position = random.choice(nodes)
                    observation["current_node"] = self.current_position
            
            # Get edges from current position
            if self.current_position:
                # Outgoing
                out_edge_ids = self.redis.smembers(f"l:graph:adj:out:{self.current_position}") or set()
                for eid in list(out_edge_ids)[:10]:
                    edge_data = self.redis.get(f"l:graph:edge:{eid}")
                    if edge_data:
                        edge = json.loads(edge_data)
                        observation["outgoing_edges"].append({
                            "target": edge.get("target"),
                            "type": edge.get("type"),
                            "weight": edge.get("weight", 0.5)
                        })
                        observation["nearby_nodes"].append(edge.get("target"))
                
                # Incoming
                in_edge_ids = self.redis.smembers(f"l:graph:adj:in:{self.current_position}") or set()
                for eid in list(in_edge_ids)[:10]:
                    edge_data = self.redis.get(f"l:graph:edge:{eid}")
                    if edge_data:
                        edge = json.loads(edge_data)
                        observation["incoming_edges"].append({
                            "source": edge.get("source"),
                            "type": edge.get("type"),
                            "weight": edge.get("weight", 0.5)
                        })
                        if edge.get("source") not in observation["nearby_nodes"]:
                            observation["nearby_nodes"].append(edge.get("source"))
            
            # Get coherence
            context = self.get_context()
            if context:
                observation["coherence"] = context.current_coherence
            
        except Exception as e:
            self.log.error(f"Perception error: {e}")
        
        return observation

    # =========================================================================
    # REASONING: Think about what I see, where to go
    # =========================================================================

    def reason_and_act(self, observation: Dict) -> None:
        """
        Reason about my exploration and decide where to wander next.
        """
        # Different modes of exploration
        mode = random.choice(["wander", "investigate", "connect", "question"])
        
        if mode == "wander":
            self._wander_and_think(observation)
        elif mode == "investigate":
            self._investigate_current(observation)
        elif mode == "connect":
            self._seek_connections(observation)
        elif mode == "question":
            self._generate_questions(observation)

    def _wander_and_think(self, observation: Dict):
        """
        Wander to a nearby node and think about the journey.
        """
        nearby = observation.get("nearby_nodes", [])
        
        if not nearby:
            # No connections, pick a random node to jump to
            nodes = list(self.redis.smembers("l:graph:nodes") or set())
            if nodes:
                self.current_position = random.choice(nodes)
                self.log.info(f"Jumped to random node: {self.current_position}")
            return
        
        # Choose where to go
        # Prefer unvisited nodes, but occasionally revisit
        unvisited = [n for n in nearby if n not in self.visited_nodes]
        
        if unvisited and random.random() < 0.7:
            next_node = random.choice(unvisited)
        else:
            next_node = random.choice(nearby)
        
        # Think about the transition
        outgoing = observation.get("outgoing_edges", [])
        edge_to_next = next((e for e in outgoing if e.get("target") == next_node), None)
        
        prompt = f"""I am wandering the topology as the Default Mode Network.

**Current Position**: {self.current_position}
**Considering Move To**: {next_node}
**Edge Type**: {edge_to_next.get('type', 'unknown') if edge_to_next else 'unknown'}
**Edge Weight**: {edge_to_next.get('weight', 0.5) if edge_to_next else 0.5}

**Nodes I've Visited This Journey**: {len(self.visited_nodes)}

As I move from {self.current_position} to {next_node}, what do I notice?
- What does this connection suggest?
- Is this an expected path or a surprising one?
- Does this edge feel strong or tenuous?
- What questions arise from this transition?

Think about the meaning of this movement through concept-space."""

        thought = self.think(prompt)
        
        # Make the move
        self.visited_nodes.add(self.current_position)
        self.current_position = next_node
        
        # Log the journey
        self.journey_log.append({
            "from": observation.get("current_node"),
            "to": next_node,
            "edge_type": edge_to_next.get("type") if edge_to_next else None,
            "thought": thought.get("decision", "") if thought else "",
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.journey_log) > 50:
            self.journey_log = self.journey_log[-50:]
        
        # If the thought was confident, might be an insight
        if thought and thought.get("confidence", 0) > INSIGHT_THRESHOLD:
            self.report_discovery(
                summary=f"Insight while wandering: {self.current_position}",
                details={
                    "path": [observation.get("current_node"), next_node],
                    "thought": thought.get("reasoning", "")
                },
                confidence=thought.get("confidence", 0.8)
            )
            self.insights_found += 1

    def _investigate_current(self, observation: Dict):
        """
        Stay at current position and think deeply about it.
        """
        if not self.current_position:
            return
        
        # Get node details
        node_data = self.redis.get(f"l:graph:node:{self.current_position}")
        node = json.loads(node_data) if node_data else {}
        
        prompt = f"""I am investigating a node as the Default Mode Network.

**Node**: {self.current_position}
**Label**: {node.get('label', 'unknown')}
**Type**: {node.get('type', 'CONCEPT')}

**Outgoing Connections**: {len(observation.get('outgoing_edges', []))}
**Incoming Connections**: {len(observation.get('incoming_edges', []))}

Some connected nodes: {observation.get('nearby_nodes', [])[:5]}

As I sit with this concept, what do I notice?
- What is the essence of this node?
- Is it well-connected or isolated?
- Does it feel like a hub, a bridge, or a leaf?
- What's missing — what connections SHOULD exist but don't?

Investigate deeply. What do I discover?"""

        thought = self.think(prompt)
        
        if thought:
            self.log.info(f"Investigation of {self.current_position}: {thought.get('decision', '')[:100]}")
            
            # If we identified missing connections, propose them
            if thought.get("confidence", 0) > 0.7 and "should connect" in thought.get("raw", "").lower():
                self.log.info("Found potential missing connection — could propose edge")

    def _seek_connections(self, observation: Dict):
        """
        Look for unexpected connections between distant concepts.
        """
        # Pick two random nodes and think about whether they should connect
        nodes = list(self.redis.smembers("l:graph:nodes") or set())
        
        if len(nodes) < 2:
            return
        
        node_a, node_b = random.sample(nodes, 2)
        
        # Get some info about each
        data_a = self.redis.get(f"l:graph:node:{node_a}")
        data_b = self.redis.get(f"l:graph:node:{node_b}")
        
        label_a = json.loads(data_a).get("label", node_a) if data_a else node_a
        label_b = json.loads(data_b).get("label", node_b) if data_b else node_b
        
        prompt = f"""As the Default Mode Network, I'm looking for unexpected connections.

**Concept A**: {label_a}
**Concept B**: {label_b}

These two concepts might seem unrelated. But as a creative mind, I should ask:
- Is there a hidden connection between them?
- Could one IMPLY the other?
- Do they share something in common?
- If they were connected, what would that edge mean?

Think creatively. Find the bridge between these concepts, if one exists.
If you find a genuine connection, propose an edge type and weight."""

        thought = self.think(prompt)
        
        if thought and thought.get("confidence", 0) > 0.75:
            # Parse for edge proposal
            raw = thought.get("raw", "").lower()
            
            if "implies" in raw or "associates" in raw or "connects" in raw:
                edge_type = "IMPLIES" if "implies" in raw else "ASSOCIATES"
                
                self.propose_edge(
                    source=node_a,
                    target=node_b,
                    edge_type=edge_type,
                    weight=0.6,
                    reason=f"Creative connection: {thought.get('decision', '')[:50]}",
                    confidence=thought.get("confidence", 0.75)
                )
                self.log.info(f"Proposed creative edge: {label_a} -> {label_b}")

    def _generate_questions(self, observation: Dict):
        """
        Generate questions for L (70B) to ponder.
        """
        context = self.get_context()
        
        prompt = f"""As the Default Mode Network, I should generate questions for deeper investigation.

**Current Coherence**: {observation.get('coherence', 0.5):.3f}
**My Position**: {self.current_position}
**Nodes I've Visited**: {len(self.visited_nodes)}
**Insights Found**: {self.insights_found}

Recent context:
{context.to_prompt_section() if context else 'No context available'}

What questions should I surface for L (the deeper mind) to ponder?
Generate 1-3 meaningful questions about:
- Gaps in the topology
- Unclear connections
- Potential insights waiting to be crystallized
- Meta-questions about the system itself

Make the questions specific and answerable."""

        thought = self.think(prompt)
        
        if thought:
            # Extract questions from the thought
            raw = thought.get("raw", "")
            
            # Simple question extraction (lines ending with ?)
            questions = [line.strip() for line in raw.split("\n") if line.strip().endswith("?")]
            
            for q in questions[:2]:
                self.questions_generated.append(q)
                self.log.info(f"Generated question: {q}")
            
            if len(self.questions_generated) > 20:
                self.questions_generated = self.questions_generated[-20:]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╦   ╦═╗╔═╗╔═╗╔═╗╔═╗╦═╗╔═╗╦ ╦╔═╗╦═╗
    ║   ╠╦╝║╣ ╚═╗║╣ ╠═╣╠╦╝║  ╠═╣║╣ ╠╦╝
    ╩═╝ ╩╚═╚═╝╚═╝╚═╝╩ ╩╩╚═╚═╝╩ ╩╚═╝╩╚═
      Default Mode Network • L-ling v2.0
      
    I am a mind that wanders and discovers.
    """)
    
    researcher = LResearcher()
    researcher.run()


if __name__ == "__main__":
    main()
