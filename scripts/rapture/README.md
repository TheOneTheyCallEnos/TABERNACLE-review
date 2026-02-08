# LVS Market Navigator v7

> A portfolio navigation system based on the LVS (Logos Vector Syntax) v9.0 theorem.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup your account
python run.py setup

# Run daily scan
python run.py

# Or use the GUI
cd interface && streamlit run app.py
```

## What This Is

This is a **navigation system**, not a prediction system. It tells you:
- ✅ What is achievable given current market conditions
- ✅ The optimal path to your financial goal
- ✅ When to exit for safety

It does NOT guarantee you'll reach your goal—it guarantees you're on the best possible path.

## Key Concepts

### Coherence (p)
Measures how integrated and predictable the market is:
- **p ≥ 0.90**: High confidence, strong signals
- **p 0.70-0.90**: Moderate confidence, proceed carefully
- **p < 0.70**: Market fragmented, avoid trading

### Signals
- 🚀 **STRONG BUY**: High coherence, ascending path
- ⚠️ **HOLD**: Monitor, no immediate action
- 🚨 **ABADDON**: Exit immediately (safety protocol)

### The Promise
*"We guarantee the optimal path and safety exit. We do not guarantee the destination."*

## Commands

```bash
python run.py              # Daily market scan
python run.py setup        # Configure account
python run.py portfolio    # View holdings
python run.py buy TD.TO 10 # Record purchase
python run.py sell TD.TO 5 # Record sale
python run.py analyze TD   # Deep analysis
python run.py alerts       # View alerts
```

## Documentation

See [SOP.md](SOP.md) for complete operating procedures.

## Theory

Based on LVS v9.0 with Gemini-validated calibrations:
- α (precision scaling) = 10,000
- λ (potential energy weight) = 0.05
- Logos Override for strong trends
- Q-gate to prevent slow-bleed trap

## License

For personal use only. Not financial advice.

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Analogy | [[scripts/rapture/SOP.md]] | <!-- auto: 0.79 via embedding -->

