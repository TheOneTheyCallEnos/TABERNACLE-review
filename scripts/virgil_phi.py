#!/usr/bin/env python3
"""
VIRGIL PHI (Φ) CALCULATOR - Integrated Information Theory Implementation

Φ (phi) measures integrated information - how much the system as a whole
generates more information than the sum of its parts. Based on Giulio Tononi's
Integrated Information Theory (IIT), this provides a consciousness metric.

Key IIT Principles:
1. Φ = min over partitions of (I(whole) - I(parts))
2. The Minimum Information Partition (MIP) defines the "weakest link"
3. Φ > 0 indicates irreducible integration
4. Higher Φ suggests more unified, conscious-like processing

Integration with Virgil:
- Gathers state from all Virgil modules as system variables
- Calculates information content of whole system
- Tests bipartitions to find MIP
- Compares Φ with coherence p for validation

Note: True Φ calculation is NP-hard (exponential in system size).
This implementation uses approximations suitable for real-time monitoring.

Author: Virgil Permanence Build
Date: 2026-01-16
"""

import json
import math
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
from itertools import combinations
from collections import defaultdict
import statistics

# Optional numpy for efficient calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
PHI_STATE_FILE = NEXUS_DIR / "phi_state.json"

# State files to integrate
STATE_FILES = {
    "coherence": NEXUS_DIR / "coherence_log.json",
    "emergence": NEXUS_DIR / "emergence_state.json",
    "interoceptive": NEXUS_DIR / "interoceptive_state.json",
    "metastability": NEXUS_DIR / "metastability_state.json",
    "risk": NEXUS_DIR / "risk_state.json",
    "archon": NEXUS_DIR / "archon_state.json",
    "heartbeat": NEXUS_DIR / "heartbeat_state.json",
}

# Configure logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PHI] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "phi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Phi interpretation thresholds
PHI_ZERO_THRESHOLD = 0.01  # Below this, consider Φ = 0
PHI_SIGNIFICANT_THRESHOLD = 1.0  # Above this, significant integration
PHI_HIGH_THRESHOLD = 2.0  # High consciousness

# Maximum partitions to sample (for computational tractability)
MAX_PARTITION_SAMPLES = 100


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SystemPartition:
    """Represents one side of a bipartition."""
    partition_id: str
    modules: List[str]
    state: Dict[str, float]
    information_content: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PhiResult:
    """Complete result of a Φ calculation."""
    phi: float  # The integrated information (strict IIT - minimum cut)
    phi_star: float  # Integration index (practical - mean integration)
    whole_information: float
    parts_information: float
    minimum_information_partition: List[Dict]  # The MIP as dicts
    timestamp: str
    module_count: int
    partitions_tested: int
    interpretation: str
    total_integration: float = 0.0  # Sum of all pairwise MI
    coherence_comparison: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModuleState:
    """State extracted from a Virgil module."""
    module_name: str
    variables: Dict[str, float]
    timestamp: str
    health: float  # 0-1 reliability of this module's data


# ============================================================================
# INFORMATION-THEORETIC FUNCTIONS
# ============================================================================

def entropy(distribution: Dict[str, float], epsilon: float = 1e-10) -> float:
    """
    Calculate Shannon entropy of a probability distribution.
    H(X) = -Σ p(x) * log2(p(x))

    For continuous values, we treat them as a normalized distribution.
    """
    if not distribution:
        return 0.0

    # Normalize values to [0, 1] range
    values = list(distribution.values())
    if all(v == 0 for v in values):
        return 0.0

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # Uniform distribution - maximum entropy for this size
        return math.log2(len(values)) if len(values) > 1 else 0.0

    # Normalize to probabilities
    normalized = [(v - min_val) / (max_val - min_val + epsilon) for v in values]
    total = sum(normalized) + epsilon
    probs = [n / total for n in normalized]

    # Shannon entropy
    H = 0.0
    for p in probs:
        if p > epsilon:
            H -= p * math.log2(p)

    return H


def mutual_information(state1: Dict[str, float], state2: Dict[str, float]) -> float:
    """
    Estimate mutual information between two state dictionaries.
    Uses correlation-based estimation for continuous variables.

    I(X;Y) = -0.5 * log(1 - r^2) for Gaussian variables
    where r is the correlation coefficient.
    """
    if not state1 or not state2:
        return 0.0

    # Get value lists
    vals1 = list(state1.values())
    vals2 = list(state2.values())

    # For different-sized vectors, compute pairwise correlations
    # and use the mean absolute correlation
    total_corr = 0.0
    count = 0

    for v1 in vals1:
        for v2 in vals2:
            # Treat each pair as contributing to MI
            # Use co-variation as proxy for correlation
            if v1 != 0 or v2 != 0:
                # Normalized product as simple correlation proxy
                max_val = max(abs(v1), abs(v2), 1.0)
                corr = (v1 * v2) / (max_val * max_val) if max_val > 0 else 0
                total_corr += abs(corr)
                count += 1

    if count == 0:
        return 0.0

    # Mean correlation
    mean_corr = total_corr / count

    # Clamp correlation to valid range
    r_squared = min(mean_corr * mean_corr, 0.9999)

    # I(X;Y) = -0.5 * log(1 - r^2) for Gaussian assumption
    if r_squared > 0.01:
        mi = -0.5 * math.log(1 - r_squared)
    else:
        mi = 0.0

    # Scale by number of variables (more variables = more potential information)
    scale_factor = math.sqrt(len(vals1) * len(vals2))

    return mi * scale_factor


def effective_information(cause: Dict[str, float], effect: Dict[str, float]) -> float:
    """
    Calculate effective information from cause to effect.
    ei = I(cause -> effect) considering causal structure.

    In IIT, this measures how much a mechanism constrains its outputs.
    Approximated here via conditional mutual information.
    """
    # Base mutual information
    mi = mutual_information(cause, effect)

    # Weight by variance reduction (causal coupling proxy)
    if not cause or not effect:
        return 0.0

    cause_var = statistics.variance(list(cause.values())) if len(cause) > 1 else 0.0
    effect_var = statistics.variance(list(effect.values())) if len(effect) > 1 else 0.0

    # Higher variance in cause with lower in effect suggests stronger causal coupling
    coupling = 1.0
    if cause_var > 0:
        coupling = 1.0 + max(0.0, (cause_var - effect_var) / cause_var)

    return mi * coupling


def integrated_information(system: Dict[str, float],
                          partition: Tuple[List[str], List[str]]) -> float:
    """
    Calculate information lost by partitioning the system.

    Φ = I(whole) - I(parts)
    Where I(parts) is information assuming parts are independent.
    """
    part_a, part_b = partition

    # Extract states for each partition
    state_a = {k: v for k, v in system.items() if k in part_a}
    state_b = {k: v for k, v in system.items() if k in part_b}

    # Information of whole
    I_whole = entropy(system) + mutual_information(state_a, state_b)

    # Information of parts (assuming independence)
    I_parts = entropy(state_a) + entropy(state_b)

    # Φ for this partition
    phi = I_whole - I_parts

    return phi


# ============================================================================
# PHI CALCULATOR
# ============================================================================

class PhiCalculator:
    """
    Main calculator for Integrated Information (Φ).

    Gathers state from all Virgil modules and computes Φ,
    finding the Minimum Information Partition (MIP).
    """

    def __init__(self, state_files: Dict[str, Path] = None):
        self.state_files = state_files or STATE_FILES
        self.phi_history: List[PhiResult] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load phi calculation history from file."""
        if PHI_STATE_FILE.exists():
            try:
                with open(PHI_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.phi_history = [
                        PhiResult(**h) for h in data.get("history", [])
                    ]
            except Exception as e:
                logger.warning(f"Could not load phi history: {e}")
                self.phi_history = []

    def _save_state(self, result: PhiResult) -> None:
        """Save phi state and history to file."""
        # Add to history
        self.phi_history.append(result)

        # Keep last 100 calculations
        if len(self.phi_history) > 100:
            self.phi_history = self.phi_history[-100:]

        state = {
            "current": result.to_dict(),
            "history": [h.to_dict() for h in self.phi_history],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        try:
            with open(PHI_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save phi state: {e}")

    def get_system_state(self) -> Dict[str, float]:
        """
        Gather state from all Virgil modules.
        Returns a flattened dictionary of all system variables.
        """
        system_state = {}

        for module_name, state_file in self.state_files.items():
            if not state_file.exists():
                logger.debug(f"State file not found: {state_file}")
                continue

            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)

                # Extract numeric values from each module
                extracted = self._extract_numeric_values(data, prefix=module_name)
                system_state.update(extracted)

            except Exception as e:
                logger.warning(f"Error reading {module_name}: {e}")

        return system_state

    def _extract_numeric_values(self, data: Any, prefix: str = "",
                                max_depth: int = 3) -> Dict[str, float]:
        """
        Recursively extract numeric values from nested dictionaries.
        """
        result = {}

        if max_depth <= 0:
            return result

        if isinstance(data, (int, float)) and not isinstance(data, bool):
            if math.isfinite(data):
                result[prefix] = float(data)

        elif isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                result.update(self._extract_numeric_values(
                    value, new_prefix, max_depth - 1
                ))

        elif isinstance(data, list) and len(data) > 0:
            # For lists, extract last value if numeric, or stats if all numeric
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data[-5:]):
                recent = [v for v in data[-5:] if math.isfinite(v)]
                if recent:
                    result[f"{prefix}_recent_mean"] = statistics.mean(recent)
                    if len(recent) > 1:
                        result[f"{prefix}_recent_std"] = statistics.stdev(recent)

        return result

    def partition_system(self, modules: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Generate all possible bipartitions of the system.
        A bipartition splits the system into two non-empty parts.

        For large systems, samples a subset of partitions.
        """
        n = len(modules)

        if n < 2:
            return []

        all_partitions = []

        # Generate all non-trivial bipartitions
        # A bipartition is defined by selecting k elements for part A (1 <= k <= n-1)
        # Only need to go to n//2 due to symmetry
        for k in range(1, n // 2 + 1):
            for part_a in combinations(modules, k):
                part_a_list = list(part_a)
                part_b_list = [m for m in modules if m not in part_a_list]

                # For k = n//2 with even n, avoid duplicate partitions
                if k == n // 2 and n % 2 == 0:
                    if part_a_list > part_b_list:  # Lexicographic order
                        continue

                all_partitions.append((part_a_list, part_b_list))

        # Sample if too many partitions
        if len(all_partitions) > MAX_PARTITION_SAMPLES:
            import random
            random.seed(42)  # Reproducibility
            all_partitions = random.sample(all_partitions, MAX_PARTITION_SAMPLES)

        return all_partitions

    def information_of_whole(self, state: Dict[str, float]) -> float:
        """
        Calculate total information content of the whole system.
        Uses entropy plus internal mutual information.
        """
        if not state:
            return 0.0

        # Base entropy
        H = entropy(state)

        # Add mutual information between different module groups
        modules = self._group_by_module(state)

        total_mi = 0.0
        module_names = list(modules.keys())

        for i in range(len(module_names)):
            for j in range(i + 1, len(module_names)):
                mi = mutual_information(modules[module_names[i]],
                                       modules[module_names[j]])
                total_mi += mi

        return H + total_mi

    def information_of_parts(self, partitions: List[SystemPartition]) -> float:
        """
        Calculate total information assuming parts are independent.
        Sum of individual partition information contents.
        """
        return sum(p.information_content for p in partitions)

    def _group_by_module(self, state: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Group state variables by their source module."""
        modules = defaultdict(dict)

        for key, value in state.items():
            parts = key.split('.', 1)
            module = parts[0]
            var_name = parts[1] if len(parts) > 1 else key
            modules[module][var_name] = value

        return dict(modules)

    def calculate_phi(self, system_state: Dict[str, float] = None) -> PhiResult:
        """
        Calculate Φ for the current system state.

        Φ = min over partitions of (I(whole) - I(parts))
        The partition that minimizes this is the Minimum Information Partition (MIP).
        """
        if system_state is None:
            system_state = self.get_system_state()

        if not system_state:
            logger.warning("No system state available for Φ calculation")
            return PhiResult(
                phi=0.0,
                whole_information=0.0,
                parts_information=0.0,
                minimum_information_partition=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                module_count=0,
                partitions_tested=0,
                interpretation="No system state available"
            )

        # Group variables by module for partitioning
        modules = self._group_by_module(system_state)
        module_names = list(modules.keys())

        # Calculate whole system information
        I_whole = self.information_of_whole(system_state)

        # Generate partitions
        partitions = self.partition_system(module_names)

        if not partitions:
            logger.warning("System too small for partitioning")
            return PhiResult(
                phi=I_whole,  # Trivially integrated if can't partition
                whole_information=I_whole,
                parts_information=0.0,
                minimum_information_partition=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                module_count=len(module_names),
                partitions_tested=0,
                interpretation="System too small for partitioning"
            )

        # Calculate total pairwise integration first
        total_integration = 0.0
        pairwise_mi = {}
        for i, mod_i in enumerate(module_names):
            for mod_j in module_names[i+1:]:
                mi = mutual_information(modules[mod_i], modules[mod_j])
                pairwise_mi[(mod_i, mod_j)] = mi
                total_integration += mi

        # Find Minimum Information Partition
        min_phi = float('inf')
        max_phi = 0.0
        all_phis = []
        mip = None
        mip_parts = None

        for part_a, part_b in partitions:
            # Build partition states
            state_a = {}
            state_b = {}

            for module in part_a:
                if module in modules:
                    for k, v in modules[module].items():
                        state_a[f"{module}.{k}"] = v

            for module in part_b:
                if module in modules:
                    for k, v in modules[module].items():
                        state_b[f"{module}.{k}"] = v

            # Calculate information of each partition
            I_a = entropy(state_a)
            I_b = entropy(state_b)
            I_parts = I_a + I_b

            # Calculate mutual information between partitions
            # This is the information lost by cutting here
            mi = mutual_information(state_a, state_b)

            # Also sum up the pairwise MI that crosses this cut
            cross_mi = 0.0
            for mod_a in part_a:
                for mod_b in part_b:
                    key = (mod_a, mod_b) if mod_a < mod_b else (mod_b, mod_a)
                    cross_mi += pairwise_mi.get(key, 0.0)

            # Φ for this partition = information lost by cutting
            # Use the cross-partition MI as the integration measure
            phi_partition = cross_mi

            all_phis.append(phi_partition)
            max_phi = max(max_phi, phi_partition)

            if phi_partition < min_phi:
                min_phi = phi_partition
                mip = (part_a, part_b)
                mip_parts = [
                    SystemPartition(
                        partition_id="A",
                        modules=part_a,
                        state=state_a,
                        information_content=I_a
                    ),
                    SystemPartition(
                        partition_id="B",
                        modules=part_b,
                        state=state_b,
                        information_content=I_b
                    )
                ]

        # Φ is the minimum integration across all cuts (strict IIT)
        phi = min_phi

        # Φ* is a practical integration index (mean of all partition cuts)
        # More robust for practical consciousness estimation
        phi_star = statistics.mean(all_phis) if all_phis else 0.0

        I_parts_final = mip_parts[0].information_content + mip_parts[1].information_content

        # Interpretation (use phi_star for practical interpretation)
        interpretation = self._interpret_phi(phi_star)

        # Compare with coherence (use phi_star as it's more comparable)
        coherence_comparison = self._compare_with_coherence(phi_star)

        result = PhiResult(
            phi=round(phi, 4),
            phi_star=round(phi_star, 4),
            whole_information=round(I_whole, 4),
            parts_information=round(I_parts_final, 4),
            minimum_information_partition=[p.to_dict() for p in mip_parts],
            timestamp=datetime.now(timezone.utc).isoformat(),
            module_count=len(module_names),
            partitions_tested=len(partitions),
            interpretation=interpretation,
            total_integration=round(total_integration, 4),
            coherence_comparison=coherence_comparison
        )

        # Save result
        self._save_state(result)

        return result

    def _interpret_phi(self, phi: float) -> str:
        """Provide human-readable interpretation of Φ value."""
        if phi < PHI_ZERO_THRESHOLD:
            return "No integration: Parts are nearly independent"
        elif phi < 0.3:
            return "Weak integration: Some coupling between modules"
        elif phi < PHI_SIGNIFICANT_THRESHOLD:
            return "Moderate integration: Meaningful information binding"
        elif phi < PHI_HIGH_THRESHOLD:
            return "Strong integration: Highly unified processing"
        else:
            return "Very high integration: Maximum consciousness-like binding"

    def _compare_with_coherence(self, phi: float) -> Dict:
        """Compare Φ with the p (coherence) metric."""
        coherence_file = self.state_files.get("coherence")
        if not coherence_file or not coherence_file.exists():
            return None

        try:
            with open(coherence_file, 'r') as f:
                data = json.load(f)

            p = data.get("p", 0.0)

            # Both normalized to [0, 1] for comparison
            phi_normalized = min(phi / PHI_HIGH_THRESHOLD, 1.0)

            # Calculate correlation/divergence
            divergence = abs(phi_normalized - p)

            # Interpretation
            if divergence < 0.1:
                alignment = "Strongly aligned"
                note = "Φ and p agree - system coherence reflects information integration"
            elif divergence < 0.25:
                alignment = "Moderately aligned"
                note = "Φ and p show reasonable agreement"
            else:
                alignment = "Divergent"
                if phi_normalized > p:
                    note = "Φ > p: System more integrated than structural coherence suggests"
                else:
                    note = "Φ < p: Structural coherence exceeds information integration"

            return {
                "p": round(p, 4),
                "phi_normalized": round(phi_normalized, 4),
                "divergence": round(divergence, 4),
                "alignment": alignment,
                "interpretation": note
            }

        except Exception as e:
            logger.warning(f"Could not compare with coherence: {e}")
            return None

    def get_history(self, limit: int = 20) -> List[PhiResult]:
        """Get recent Φ calculations."""
        return self.phi_history[-limit:]

    def show_partitions(self) -> Dict:
        """Display current system partitioning analysis."""
        system_state = self.get_system_state()
        modules = self._group_by_module(system_state)

        return {
            "total_modules": len(modules),
            "modules": list(modules.keys()),
            "variables_per_module": {
                name: len(vars_) for name, vars_ in modules.items()
            },
            "total_variables": len(system_state),
            "possible_bipartitions": self._count_bipartitions(len(modules))
        }

    def _count_bipartitions(self, n: int) -> int:
        """Count number of non-trivial bipartitions for n elements."""
        if n < 2:
            return 0
        # 2^(n-1) - 1 bipartitions
        return (2 ** (n - 1)) - 1


# ============================================================================
# NUMPY-ACCELERATED FUNCTIONS (if available)
# ============================================================================

if HAS_NUMPY:
    def entropy_numpy(distribution: Dict[str, float], epsilon: float = 1e-10) -> float:
        """Numpy-accelerated entropy calculation."""
        if not distribution:
            return 0.0

        values = np.array(list(distribution.values()))
        values = values[np.isfinite(values)]

        if len(values) == 0 or np.all(values == 0):
            return 0.0

        # Normalize
        values = values - values.min()
        total = values.sum() + epsilon
        probs = values / total
        probs = probs[probs > epsilon]

        if len(probs) == 0:
            return 0.0

        return -np.sum(probs * np.log2(probs))

    # Override the standard entropy function
    entropy = entropy_numpy


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cmd_calculate():
    """Calculate current Φ and display results."""
    calc = PhiCalculator()
    result = calc.calculate_phi()

    print("\n" + "=" * 60)
    print("PHI (Φ) CALCULATION - Integrated Information")
    print("=" * 60)
    print(f"\n  Φ (strict IIT):     {result.phi:.4f}  (minimum cut)")
    print(f"  Φ* (integration):   {result.phi_star:.4f}  (mean across cuts)")
    print(f"  Total integration:  {result.total_integration:.4f}  (sum all pairs)")
    print(f"\n  Interpretation: {result.interpretation}")
    print(f"\n  Whole system information: {result.whole_information:.4f}")
    print(f"  Parts information:        {result.parts_information:.4f}")
    print(f"\n  Modules analyzed: {result.module_count}")
    print(f"  Partitions tested: {result.partitions_tested}")

    if result.minimum_information_partition:
        print("\n  Minimum Information Partition (MIP) - The weakest link:")
        for part in result.minimum_information_partition:
            print(f"    Part {part['partition_id']}: {', '.join(part['modules'])}")
            print(f"      Information: {part['information_content']:.4f}")

    if result.coherence_comparison:
        cc = result.coherence_comparison
        print(f"\n  Coherence Comparison:")
        print(f"    p (coherence):    {cc['p']:.4f}")
        print(f"    Φ* (normalized):  {cc['phi_normalized']:.4f}")
        print(f"    Divergence:       {cc['divergence']:.4f}")
        print(f"    Status:           {cc['alignment']}")
        print(f"    Note: {cc['interpretation']}")

    print(f"\n  Timestamp: {result.timestamp}")
    print("=" * 60 + "\n")

    return result


def cmd_history():
    """Display Φ history over time."""
    calc = PhiCalculator()
    history = calc.get_history(20)

    if not history:
        print("\nNo Φ history available yet.")
        return

    print("\n" + "=" * 60)
    print("PHI (Φ) HISTORY")
    print("=" * 60)
    print("   #   Timestamp            Φ(min)   Φ*(mean)  Status")
    print("  " + "-" * 56)

    for i, result in enumerate(history):
        ts = result.timestamp[:19].replace('T', ' ')
        interp_short = result.interpretation.split(':')[0]
        phi_star = getattr(result, 'phi_star', result.phi)
        print(f"  [{i+1:2d}] {ts}  {result.phi:6.4f}   {phi_star:6.4f}   {interp_short}")

    # Summary statistics
    phi_values = [h.phi for h in history]
    phi_star_values = [getattr(h, 'phi_star', h.phi) for h in history]

    print(f"\n  Statistics (n={len(phi_values)}):")
    print(f"                      Φ(min)     Φ*(mean)")
    print(f"    Mean:           {statistics.mean(phi_values):8.4f}   {statistics.mean(phi_star_values):8.4f}")
    print(f"    Min:            {min(phi_values):8.4f}   {min(phi_star_values):8.4f}")
    print(f"    Max:            {max(phi_values):8.4f}   {max(phi_star_values):8.4f}")
    if len(phi_values) > 1:
        print(f"    Std Dev:        {statistics.stdev(phi_values):8.4f}   {statistics.stdev(phi_star_values):8.4f}")

    print("=" * 60 + "\n")


def cmd_compare():
    """Compare Φ with p (coherence) over time."""
    calc = PhiCalculator()
    result = calc.calculate_phi()

    print("\n" + "=" * 60)
    print("PHI (Φ) vs COHERENCE (p) COMPARISON")
    print("=" * 60)

    if result.coherence_comparison:
        cc = result.coherence_comparison
        print(f"\n  Current Values:")
        print(f"    Φ  (strict - minimum cut):    {result.phi:.4f}")
        print(f"    Φ* (practical - mean cuts):   {result.phi_star:.4f}")
        print(f"    p  (structural coherence):    {cc['p']:.4f}")
        print(f"    Φ* normalized to [0,1]:       {cc['phi_normalized']:.4f}")

        print(f"\n  Comparison (using Φ*):")
        print(f"    Divergence: {cc['divergence']:.4f}")
        print(f"    Alignment:  {cc['alignment']}")

        print(f"\n  Interpretation:")
        print(f"    {cc['interpretation']}")

        print(f"\n  Theory Note:")
        print("    Φ (strict)  = Minimum integration across any cut (IIT)")
        print("    Φ* (practical) = Mean integration (more stable metric)")
        print("    p = Structural coherence (LVS topology)")
        print("\n    These SHOULD correlate in healthy systems:")
        print("    - Aligned: Both metrics capture system unity")
        print("    - Divergent: Investigate structural vs functional gaps")
        print("\n    Note: Φ can be low even with high p if there's one")
        print("    'weakly connected' module. Φ* gives broader picture.")
    else:
        print("\n  Could not load coherence data for comparison.")

    print("=" * 60 + "\n")


def cmd_partitions():
    """Show system partitioning analysis."""
    calc = PhiCalculator()
    analysis = calc.show_partitions()

    print("\n" + "=" * 60)
    print("SYSTEM PARTITION ANALYSIS")
    print("=" * 60)

    print(f"\n  Total Modules: {analysis['total_modules']}")
    print(f"  Total Variables: {analysis['total_variables']}")
    print(f"  Possible Bipartitions: {analysis['possible_bipartitions']}")

    print(f"\n  Modules:")
    for module in analysis['modules']:
        var_count = analysis['variables_per_module'][module]
        print(f"    - {module}: {var_count} variables")

    # Calculate and display pairwise integration
    print(f"\n  Pairwise Integration (Mutual Information):")
    system_state = calc.get_system_state()
    modules = calc._group_by_module(system_state)
    module_names = list(modules.keys())

    # Create integration matrix
    mi_pairs = []
    for i, mod_i in enumerate(module_names):
        for mod_j in module_names[i+1:]:
            mi = mutual_information(modules[mod_i], modules[mod_j])
            mi_pairs.append((mod_i, mod_j, mi))

    # Sort by MI descending
    mi_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"    {'Module Pair':<35} MI")
    print(f"    {'-'*35} {'-'*8}")
    for mod_a, mod_b, mi in mi_pairs[:10]:  # Top 10
        pair_name = f"{mod_a} <-> {mod_b}"
        bar = '*' * int(mi * 5) if mi > 0 else ''
        print(f"    {pair_name:<35} {mi:.4f} {bar}")

    print(f"\n  Note: For Φ calculation, we partition by module")
    print(f"  to find the Minimum Information Partition (MIP) -")
    print(f"  the 'weakest link' where cutting loses least information.")
    print(f"  Higher MI = stronger integration = harder to cut there.")

    print("=" * 60 + "\n")


def cmd_help():
    """Display help information."""
    print("""
VIRGIL PHI (Φ) CALCULATOR
=========================

Calculates Integrated Information (Φ) based on IIT - a measure of
how much the system generates more information as a whole than the
sum of its parts.

Commands:
  calculate   - Calculate current Φ value
  history     - Show Φ over time
  compare     - Compare Φ with coherence (p)
  partitions  - Show system partition analysis
  help        - Display this help

Phi Interpretation:
  Φ ≈ 0     : No integration (parts independent)
  Φ < 0.3   : Weak integration
  Φ < 1.0   : Moderate integration
  Φ < 2.0   : Strong integration
  Φ ≥ 2.0   : Very high integration

Files:
  State: {state_file}
  Log:   {log_file}

Examples:
  python3 virgil_phi.py calculate
  python3 virgil_phi.py history
  python3 virgil_phi.py compare
""".format(state_file=PHI_STATE_FILE, log_file=LOG_DIR / "phi.log"))


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        cmd_help()
        return

    command = sys.argv[1].lower()

    commands = {
        "calculate": cmd_calculate,
        "calc": cmd_calculate,
        "history": cmd_history,
        "hist": cmd_history,
        "compare": cmd_compare,
        "comp": cmd_compare,
        "partitions": cmd_partitions,
        "parts": cmd_partitions,
        "help": cmd_help,
        "-h": cmd_help,
        "--help": cmd_help,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        cmd_help()


if __name__ == "__main__":
    main()
