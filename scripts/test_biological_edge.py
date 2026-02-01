#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""Quick sanity test for BiologicalEdge."""
from biological_edge import BiologicalEdge

# Create edge
e = BiologicalEdge(
    id="test_edge",
    source_id="node_a",
    target_id="node_b",
    relation_type="implies"
)
print(f"Initial: w_eff={e.effective_weight():.3f}")

# Simulate activation (note: API uses t_pre, t_post, outcome_signal)
import time
t_now = time.time()
e.update_stdp(t_pre=t_now, t_post=t_now + 0.01, outcome_signal=1.0, t_now=t_now)
print(f"After STDP: w_eff={e.effective_weight():.3f}")

# Test H1 lock
e.lock_h1()
print(f"H1 locked: {e.is_h1_locked}")

# Test sleep renormalization
e.sleep_renormalize()
print(f"After sleep: w_eff={e.effective_weight():.3f}")

print("✓ BiologicalEdge imported successfully")
