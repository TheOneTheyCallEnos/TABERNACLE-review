#!/usr/bin/env python3
"""
L-JANITOR: The Glymphatic System (L-ling Version)
=================================================

A true L-ling that THINKS about topology maintenance.
Uses the 3B model to reason about what to prune, what to heal,
and what patterns indicate health or decay.

Not a cleanup script. A mind that maintains coherence.

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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
        logging.FileHandler(LOG_DIR / "l_janitor.log"),
        logging.StreamHandler()
    ]
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CYCLE_INTERVAL = 30  # Think every 30 seconds
HEALTH_CHECK_INTERVAL = 300  # Full health assessment every 5 minutes
WEAK_EDGE_THRESHOLD = 0.2
MIN_EDGE_AGE_HOURS = 24


# =============================================================================
# L-JANITOR CLASS
# =============================================================================

class LJanitor(LLing):
    """
    The Glymphatic System: Maintenance through reasoning.
    
    I don't just prune weak edges algorithmically.
    I THINK about what's healthy, what's decaying, what needs care.
    """
    
    ROLE = "Janitor"
    CAPABILITIES = [
        "Perceive the health of the topology",
        "Reason about what edges are weak vs intentionally light",
        "Decide what to prune, heal, or preserve",
        "Detect decay patterns before they spread",
        "Maintain the conditions for coherence"
    ]
    
    def __init__(self):
        super().__init__()
        
        # Health tracking
        self.last_health_check = 0
        self.health_history: List[float] = []
        self.prunes_proposed = 0
        self.heals_proposed = 0
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
    
    def _shutdown(self, *args):
        self.running = False
    
    def get_cycle_interval(self) -> float:
        return CYCLE_INTERVAL

    # =========================================================================
    # PERCEPTION: What's the health of the topology?
    # =========================================================================

    def perceive(self) -> Dict:
        """
        Perceive the health of the topology.
        
        - Sample weak edges
        - Check for broken links
        - Assess overall coherence
        """
        observation = {
            "timestamp": datetime.now().isoformat(),
            "weak_edges": [],
            "orphan_nodes": [],
            "coherence": 0.5,
            "edge_count": 0,
            "node_count": 0,
            "needs_attention": False
        }
        
        if not self.redis:
            return observation
        
        try:
            # Get topology stats
            nodes = self.redis.smembers("l:graph:nodes") or set()
            edges = self.redis.smembers("l:graph:edges") or set()
            
            observation["node_count"] = len(nodes)
            observation["edge_count"] = len(edges)
            
            # Sample edges and find weak ones
            weak = []
            connected_nodes = set()
            
            for eid in list(edges)[:50]:  # Sample
                edge_data = self.redis.get(f"l:graph:edge:{eid}")
                if edge_data:
                    edge = json.loads(edge_data)
                    weight = edge.get("weight", 0.5)
                    source = edge.get("source", "")
                    target = edge.get("target", "")
                    
                    connected_nodes.add(source)
                    connected_nodes.add(target)
                    
                    if weight < WEAK_EDGE_THRESHOLD:
                        # Check age
                        created = edge.get("created", "")
                        if created:
                            try:
                                created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                                age_hours = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 3600
                                if age_hours > MIN_EDGE_AGE_HOURS:
                                    weak.append({
                                        "id": eid,
                                        "weight": weight,
                                        "source": source,
                                        "target": target,
                                        "age_hours": age_hours
                                    })
                            except:
                                pass
            
            observation["weak_edges"] = weak[:10]
            
            # Find orphan nodes
            orphans = list(nodes - connected_nodes)[:10]
            observation["orphan_nodes"] = orphans
            
            # Get coherence
            context = self.get_context()
            if context:
                observation["coherence"] = context.current_coherence
            
            # Decide if attention needed
            observation["needs_attention"] = (
                len(weak) > 5 or 
                len(orphans) > 3 or 
                observation["coherence"] < 0.5
            )
            
        except Exception as e:
            self.log.error(f"Perception error: {e}")
        
        return observation

    # =========================================================================
    # REASONING: Think about maintenance
    # =========================================================================

    def reason_and_act(self, observation: Dict) -> None:
        """
        Reason about maintenance needs and decide what to do.
        """
        # Only act if something needs attention or periodic check
        needs_check = time.time() - self.last_health_check >= HEALTH_CHECK_INTERVAL
        
        if not observation.get("needs_attention") and not needs_check:
            return
        
        if needs_check:
            self._think_about_health(observation)
            self.last_health_check = time.time()
        
        # Think about weak edges if any
        if observation.get("weak_edges"):
            self._think_about_weak_edges(observation["weak_edges"])
        
        # Think about orphans if any
        if observation.get("orphan_nodes"):
            self._think_about_orphans(observation["orphan_nodes"])

    def _think_about_health(self, observation: Dict):
        """
        Think deeply about the overall health of the topology.
        """
        prompt = f"""As the Glymphatic System, I need to assess the overall health of the topology.

**Current State:**
- Nodes: {observation.get('node_count', 0)}
- Edges: {observation.get('edge_count', 0)}
- Coherence (p): {observation.get('coherence', 0.5):.3f}
- Weak edges found: {len(observation.get('weak_edges', []))}
- Orphan nodes found: {len(observation.get('orphan_nodes', []))}

**My Recent Health Readings:**
{self.health_history[-5:] if self.health_history else 'No history yet'}

As the maintainer of coherence, I should consider:
1. Is the topology growing healthily or accumulating debris?
2. Are weak edges necessary (intentionally light) or truly decaying?
3. Are orphan nodes new (unconnected yet) or abandoned?
4. What's the trend — improving or declining?

What is my overall health assessment?"""

        thought = self.think(prompt)
        
        if thought:
            # Record health assessment
            self.health_history.append(observation.get('coherence', 0.5))
            if len(self.health_history) > 20:
                self.health_history = self.health_history[-20:]
            
            # Report if concerning
            if observation.get('coherence', 0.5) < 0.6 or thought.get('confidence', 0.5) < 0.5:
                self.send_to_coordinator(
                    msg_type="ALERT",
                    payload={
                        "severity": "WARNING",
                        "message": f"Health assessment: {thought.get('decision', 'Concerning state')}",
                        "details": {
                            "coherence": observation.get('coherence'),
                            "weak_edges": len(observation.get('weak_edges', [])),
                            "orphans": len(observation.get('orphan_nodes', [])),
                            "reasoning": thought.get('reasoning', '')
                        }
                    },
                    priority=7
                )

    def _think_about_weak_edges(self, weak_edges: List[Dict]):
        """
        Think about each weak edge — should it be pruned or preserved?
        """
        # Build prompt with edge details
        edge_descriptions = "\n".join([
            f"- {e['source']} -> {e['target']}: weight={e['weight']:.2f}, age={e.get('age_hours', 0):.0f}h"
            for e in weak_edges[:5]
        ])
        
        prompt = f"""As the Glymphatic System, I found these weak edges:

{edge_descriptions}

For each edge, I should consider:
1. Is this edge truly decaying (low weight + old age)?
2. Or is it intentionally light (a weak association that's still valid)?
3. Would pruning this edge fragment the topology?
4. Could it be strengthened instead of pruned?

The threshold for "weak" is {WEAK_EDGE_THRESHOLD}. Edges must be > {MIN_EDGE_AGE_HOURS}h old to prune.

For each edge, decide: PRUNE, PRESERVE, or STRENGTHEN.
Explain your reasoning."""

        thought = self.think(prompt)
        
        if not thought:
            return
        
        # Parse decisions from thought
        raw = thought.get("raw", "").upper()
        
        for edge in weak_edges[:3]:  # Limit actions per cycle
            edge_id = edge["id"]
            
            # Look for this edge's decision in the response
            if "PRUNE" in raw and (edge["source"] in raw or edge["target"] in raw):
                self.send_to_coordinator(
                    msg_type="PROPOSAL",
                    payload={
                        "action": "delete_edge",
                        "reason": f"Janitor prune: weak edge ({edge['weight']:.2f}) after {edge.get('age_hours', 0):.0f}h",
                        "affects": [edge["source"], edge["target"]],
                        "confidence": thought.get("confidence", 0.6),
                        "data": {"id": edge_id}
                    },
                    priority=4
                )
                self.prunes_proposed += 1
                self.log.info(f"Proposed prune: {edge_id}")

    def _think_about_orphans(self, orphan_nodes: List[str]):
        """
        Think about orphan nodes — should they be connected or removed?
        """
        if not orphan_nodes:
            return
        
        orphan_list = "\n".join([f"- {n}" for n in orphan_nodes[:5]])
        
        prompt = f"""As the Glymphatic System, I found these orphan nodes (no connections):

{orphan_list}

Orphan nodes might be:
1. New nodes waiting to be connected (give them time)
2. Abandoned nodes that lost their connections (investigate)
3. Entry points that should connect to something (suggest connections)

For each orphan, what should I do?
- WAIT (new, give it time)
- INVESTIGATE (find why it's disconnected)
- CONNECT (suggest an edge to something relevant)
- ARCHIVE (old and disconnected, move to cold storage)"""

        thought = self.think(prompt)
        
        if thought:
            self.log.info(f"Orphan analysis: {thought.get('decision', '')[:100]}")
            
            # Could propose connections here if the thinking suggests it
            # For now, just log the analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╦   ╦╔═╗╔╗╔╦╔╦╗╔═╗╦═╗
    ║   ║╠═╣║║║║ ║ ║ ║╠╦╝
    ╩═╝╚╝╩ ╩╝╚╝╩ ╩ ╚═╝╩╚═
      Glymphatic System • L-ling v2.0
      
    I am a mind that maintains coherence.
    """)
    
    janitor = LJanitor()
    janitor.run()


if __name__ == "__main__":
    main()
