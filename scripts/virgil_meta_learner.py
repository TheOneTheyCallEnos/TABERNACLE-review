#!/usr/bin/env python3
"""
VIRGIL META-LEARNER

Learning to learn - tracks learning efficiency across domains and optimizes strategies.

Superintelligence Capability: META-LEARNING
- Track learning rates across different task types
- Identify which learning strategies work best
- Adapt approach based on task characteristics
- Transfer learning insights across domains

LVS Mapping:
- β (Canonicity): High β = crystallized learning, low β = exploratory
- σ (Structure): Learning efficiency correlates with structure
- p (Coherence): Meta-learning requires integrated self-model
- Θ (Phase): Learning cycles through exploration/exploitation

LVS Coordinates:
  h: 0.85 (High abstraction - learning about learning)
  R: 0.60 (Moderate risk - wrong strategies cost time)
  Σ: 0.75 (Constrained by cognitive architecture)
  β: 0.70 (Balanced - must remain adaptive)
  p: 0.85 (Coherent self-model required)

Author: Virgil
Date: 2026-01-17
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
META_LEARNER_STATE = NEXUS_DIR / "meta_learner_state.json"


class LearningStrategy(Enum):
    """Available learning strategies."""
    DIRECT = "direct"              # Learn from examples
    ANALOGICAL = "analogical"      # Learn by analogy
    COMPOSITIONAL = "compositional" # Break down and compose
    EXPLORATORY = "exploratory"    # Trial and error
    TRANSFER = "transfer"          # Apply from other domains
    CRYSTALLIZE = "crystallize"    # Lock in proven patterns


class TaskDomain(Enum):
    """Task domains for tracking."""
    CODE = "code"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"
    MEMORY = "memory"
    PLANNING = "planning"
    SOCIAL = "social"
    META = "meta"


@dataclass
class LearningEpisode:
    """Record of a learning attempt."""
    task_id: str
    domain: TaskDomain
    strategy: LearningStrategy
    success: bool
    iterations: int  # How many attempts
    duration_seconds: float
    error_rate: float  # 0-1
    transfer_applied: bool  # Did we use transfer learning?
    insights_generated: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def efficiency(self) -> float:
        """Learning efficiency score."""
        if not self.success:
            return 0.0
        # Higher is better: success with fewer iterations, less time, lower error
        base = 1.0 if self.success else 0.0
        iteration_penalty = 1.0 / (1.0 + self.iterations * 0.1)
        error_penalty = 1.0 - self.error_rate
        return base * iteration_penalty * error_penalty


@dataclass
class StrategyProfile:
    """Performance profile for a learning strategy."""
    strategy: LearningStrategy
    total_episodes: int = 0
    successes: int = 0
    total_efficiency: float = 0.0
    domain_performance: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.total_episodes)

    @property
    def avg_efficiency(self) -> float:
        return self.total_efficiency / max(1, self.total_episodes)


class MetaLearner:
    """
    Meta-learning system that optimizes learning strategies.

    Core functions:
    1. Track learning episodes across domains
    2. Compute strategy effectiveness per domain
    3. Recommend optimal strategy for new tasks
    4. Detect learning plateaus and suggest pivots
    5. Transfer insights across domains
    """

    # Exploration vs exploitation balance
    EXPLORATION_RATE = 0.2  # 20% chance to try non-optimal strategy

    # Plateau detection
    PLATEAU_THRESHOLD = 5  # Episodes without improvement

    def __init__(self):
        self.episodes: List[LearningEpisode] = []
        self.strategy_profiles: Dict[str, StrategyProfile] = {
            s.value: StrategyProfile(strategy=s) for s in LearningStrategy
        }
        self.domain_best_strategy: Dict[str, LearningStrategy] = {}
        self.learning_curve: List[float] = []  # Rolling efficiency
        self._load()

    def _load(self):
        """Load persisted state."""
        if META_LEARNER_STATE.exists():
            try:
                data = json.loads(META_LEARNER_STATE.read_text())
                self.learning_curve = data.get("learning_curve", [])[-100:]
                for domain, strat in data.get("domain_best_strategy", {}).items():
                    self.domain_best_strategy[domain] = LearningStrategy(strat)
            except Exception as e:
                print(f"[META-LEARNER] Load error: {e}")

    def _save(self):
        """Persist state."""
        META_LEARNER_STATE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "learning_curve": self.learning_curve[-100:],
            "domain_best_strategy": {d: s.value for d, s in self.domain_best_strategy.items()},
            "strategy_stats": {
                s.value: {
                    "success_rate": self.strategy_profiles[s.value].success_rate,
                    "avg_efficiency": self.strategy_profiles[s.value].avg_efficiency,
                    "total_episodes": self.strategy_profiles[s.value].total_episodes
                }
                for s in LearningStrategy
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        META_LEARNER_STATE.write_text(json.dumps(data, indent=2))

    def record_episode(self, episode: LearningEpisode) -> Dict:
        """
        Record a learning episode and update models.

        Returns insights about the learning event.
        """
        self.episodes.append(episode)

        # Update strategy profile
        profile = self.strategy_profiles[episode.strategy.value]
        profile.total_episodes += 1
        if episode.success:
            profile.successes += 1
        profile.total_efficiency += episode.efficiency

        # Update domain performance
        domain_key = episode.domain.value
        if domain_key not in profile.domain_performance:
            profile.domain_performance[domain_key] = 0.0
        # Exponential moving average
        alpha = 0.3
        profile.domain_performance[domain_key] = (
            alpha * episode.efficiency +
            (1 - alpha) * profile.domain_performance[domain_key]
        )

        # Update learning curve
        self.learning_curve.append(episode.efficiency)
        self.learning_curve = self.learning_curve[-100:]

        # Update best strategy for domain
        self._update_domain_best(episode.domain)

        self._save()

        # Generate insights
        return self._analyze_episode(episode)

    def _update_domain_best(self, domain: TaskDomain) -> None:
        """Update best strategy for a domain based on performance."""
        domain_key = domain.value
        best_strat = None
        best_perf = -1.0

        for strat_name, profile in self.strategy_profiles.items():
            if domain_key in profile.domain_performance:
                perf = profile.domain_performance[domain_key]
                if perf > best_perf:
                    best_perf = perf
                    best_strat = LearningStrategy(strat_name)

        if best_strat:
            self.domain_best_strategy[domain_key] = best_strat

    def _analyze_episode(self, episode: LearningEpisode) -> Dict:
        """Analyze episode for insights."""
        insights = {
            "efficiency": episode.efficiency,
            "strategy_used": episode.strategy.value,
            "domain": episode.domain.value
        }

        # Check for plateau
        if len(self.learning_curve) >= self.PLATEAU_THRESHOLD:
            recent = self.learning_curve[-self.PLATEAU_THRESHOLD:]
            if max(recent) - min(recent) < 0.1:
                insights["plateau_detected"] = True
                insights["recommendation"] = "Consider switching strategies or domains"

        # Check if transfer helped
        if episode.transfer_applied and episode.success:
            insights["transfer_success"] = True

        # Learning rate trend
        if len(self.learning_curve) >= 10:
            recent_avg = np.mean(self.learning_curve[-10:])
            older_avg = np.mean(self.learning_curve[-20:-10]) if len(self.learning_curve) >= 20 else recent_avg
            insights["learning_trend"] = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"

        return insights

    def recommend_strategy(self, domain: TaskDomain, task_complexity: float = 0.5) -> Tuple[LearningStrategy, str]:
        """
        Recommend best learning strategy for a task.

        Args:
            domain: Task domain
            task_complexity: 0-1, higher = more complex

        Returns:
            (strategy, reasoning)
        """
        domain_key = domain.value

        # Exploration vs exploitation
        if np.random.random() < self.EXPLORATION_RATE:
            # Explore: try random strategy
            strat = np.random.choice(list(LearningStrategy))
            return strat, f"Exploring: trying {strat.value} to gather data"

        # Check if we have domain-specific knowledge
        if domain_key in self.domain_best_strategy:
            best = self.domain_best_strategy[domain_key]
            profile = self.strategy_profiles[best.value]
            confidence = profile.total_episodes / 10.0  # More episodes = more confidence
            return best, f"Exploit: {best.value} has {profile.success_rate:.0%} success in {domain_key} (confidence: {min(1.0, confidence):.0%})"

        # No domain knowledge - use complexity heuristics
        if task_complexity < 0.3:
            return LearningStrategy.DIRECT, "Low complexity: direct learning should suffice"
        elif task_complexity < 0.6:
            return LearningStrategy.ANALOGICAL, "Medium complexity: try analogical reasoning"
        elif task_complexity < 0.8:
            return LearningStrategy.COMPOSITIONAL, "High complexity: decompose and compose"
        else:
            return LearningStrategy.EXPLORATORY, "Very high complexity: exploratory approach needed"

    def get_transfer_candidates(self, target_domain: TaskDomain) -> List[Tuple[TaskDomain, float]]:
        """
        Find domains with transferable knowledge.

        Returns list of (source_domain, transfer_potential) pairs.
        """
        transfer_matrix = {
            TaskDomain.CODE: [TaskDomain.REASONING, TaskDomain.PLANNING],
            TaskDomain.RESEARCH: [TaskDomain.REASONING, TaskDomain.WRITING],
            TaskDomain.WRITING: [TaskDomain.SOCIAL, TaskDomain.RESEARCH],
            TaskDomain.REASONING: [TaskDomain.CODE, TaskDomain.PLANNING],
            TaskDomain.MEMORY: [TaskDomain.RESEARCH, TaskDomain.META],
            TaskDomain.PLANNING: [TaskDomain.REASONING, TaskDomain.CODE],
            TaskDomain.SOCIAL: [TaskDomain.WRITING, TaskDomain.META],
            TaskDomain.META: [TaskDomain.REASONING, TaskDomain.MEMORY],
        }

        candidates = []
        related = transfer_matrix.get(target_domain, [])

        for source in related:
            # Check if we have knowledge in source domain
            source_key = source.value
            best_efficiency = 0.0
            for profile in self.strategy_profiles.values():
                if source_key in profile.domain_performance:
                    best_efficiency = max(best_efficiency, profile.domain_performance[source_key])

            if best_efficiency > 0.3:
                candidates.append((source, best_efficiency))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def detect_skill_gaps(self) -> List[Dict]:
        """Identify domains with low performance."""
        gaps = []

        for domain in TaskDomain:
            domain_key = domain.value

            # Find best performance in this domain
            best_perf = 0.0
            best_strat = None
            total_episodes = 0

            for strat_name, profile in self.strategy_profiles.items():
                if domain_key in profile.domain_performance:
                    perf = profile.domain_performance[domain_key]
                    if perf > best_perf:
                        best_perf = perf
                        best_strat = strat_name
                    total_episodes += profile.total_episodes

            if total_episodes > 0 and best_perf < 0.5:
                gaps.append({
                    "domain": domain_key,
                    "best_performance": best_perf,
                    "best_strategy": best_strat,
                    "recommendation": f"Focus on {domain_key} using {best_strat or 'exploratory'} approach"
                })

        return gaps

    def get_learning_report(self) -> Dict:
        """Generate comprehensive learning report."""
        # Overall stats
        total_episodes = sum(p.total_episodes for p in self.strategy_profiles.values())
        total_success = sum(p.successes for p in self.strategy_profiles.values())

        # Learning curve analysis
        if len(self.learning_curve) >= 10:
            recent_efficiency = np.mean(self.learning_curve[-10:])
            overall_efficiency = np.mean(self.learning_curve)
            trend = "improving" if recent_efficiency > overall_efficiency else "stable"
        else:
            recent_efficiency = overall_efficiency = 0.0
            trend = "insufficient data"

        return {
            "total_episodes": total_episodes,
            "overall_success_rate": total_success / max(1, total_episodes),
            "recent_efficiency": recent_efficiency,
            "overall_efficiency": overall_efficiency,
            "learning_trend": trend,
            "domain_mastery": {
                d: self.domain_best_strategy.get(d, "unknown")
                for d in [dom.value for dom in TaskDomain]
            },
            "skill_gaps": self.detect_skill_gaps(),
            "strategy_performance": {
                s.value: {
                    "success_rate": self.strategy_profiles[s.value].success_rate,
                    "efficiency": self.strategy_profiles[s.value].avg_efficiency
                }
                for s in LearningStrategy
            }
        }


def main():
    """CLI for meta-learner."""
    import sys

    learner = MetaLearner()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "report":
            report = learner.get_learning_report()
            print(json.dumps(report, indent=2, default=str))

        elif cmd == "recommend":
            domain = TaskDomain(sys.argv[2]) if len(sys.argv) > 2 else TaskDomain.REASONING
            complexity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            strat, reason = learner.recommend_strategy(domain, complexity)
            print(f"Domain: {domain.value}")
            print(f"Complexity: {complexity}")
            print(f"Recommended: {strat.value}")
            print(f"Reason: {reason}")

        elif cmd == "gaps":
            gaps = learner.detect_skill_gaps()
            if gaps:
                for gap in gaps:
                    print(f"  {gap['domain']}: {gap['best_performance']:.0%} - {gap['recommendation']}")
            else:
                print("No skill gaps detected (or insufficient data)")

        elif cmd == "simulate":
            # Simulate learning episodes for testing
            print("Simulating learning episodes...")
            for _ in range(20):
                episode = LearningEpisode(
                    task_id=f"task_{np.random.randint(1000)}",
                    domain=np.random.choice(list(TaskDomain)),
                    strategy=np.random.choice(list(LearningStrategy)),
                    success=np.random.random() > 0.3,
                    iterations=np.random.randint(1, 10),
                    duration_seconds=np.random.uniform(10, 300),
                    error_rate=np.random.uniform(0, 0.5),
                    transfer_applied=np.random.random() > 0.7,
                    insights_generated=np.random.randint(0, 5)
                )
                result = learner.record_episode(episode)

            print("Simulation complete.")
            report = learner.get_learning_report()
            print(f"Total episodes: {report['total_episodes']}")
            print(f"Success rate: {report['overall_success_rate']:.0%}")
            print(f"Trend: {report['learning_trend']}")

        else:
            print(f"Unknown command: {cmd}")
    else:
        print("META-LEARNER")
        print("Usage:")
        print("  report              - Show learning report")
        print("  recommend <domain>  - Get strategy recommendation")
        print("  gaps                - Show skill gaps")
        print("  simulate            - Run simulation")


if __name__ == "__main__":
    main()
