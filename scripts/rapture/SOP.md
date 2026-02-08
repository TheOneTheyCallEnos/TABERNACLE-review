# LVS Market Navigator v7
## Standard Operating Procedure (SOP)

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Daily Operations](#daily-operations)
4. [Understanding the System](#understanding-the-system)
5. [Recording Trades](#recording-trades)
6. [Reading Signals](#reading-signals)
7. [Push Notifications](#push-notifications)
8. [Maintenance](#maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Theory Reference](#theory-reference)

---

## Overview

The LVS Market Navigator is a **portfolio navigation system**, not a prediction system. It answers:
- "What is achievable given current market conditions?"
- "What is the optimal path to my goal?"
- "When should I exit for safety?"

### The Guarantee
- ✅ We guarantee the **optimal path** to your goal
- ✅ We guarantee **safety exits** if the market breaks
- ✅ We guarantee **continuous adaptation** as conditions change
- ❌ We do NOT guarantee you will reach your goal (the market may not allow it)

---

## Installation

### Requirements
- macOS (tested on Mac Air M3 2025)
- Python 3.10+
- Internet connection (for market data)

### Step 1: Install Python Dependencies
```bash
cd market_navigator_v7
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python run.py
```

You should see the welcome banner and a prompt to set up your account.

### Step 3: Initial Setup
```bash
python run.py setup
```

Enter:
- **Starting Capital**: Your investment amount (e.g., 500)
- **Profit Goal**: Target profit in dollars (e.g., 100)
- **Timeframe**: Days to achieve goal (e.g., 30)

---

## Daily Operations

### Morning Routine (Recommended: 9:00 AM EST)

**Option A: Command Line**
```bash
python run.py
```

**Option B: GUI Interface**
```bash
cd interface
streamlit run app.py
```

### What the Daily Scan Shows

1. **Market Coherence (p)**: Overall market health (0-1 scale)
   - p > 0.85 = Strong, tradeable market
   - p 0.65-0.85 = Moderate, proceed with caution
   - p < 0.65 = Fragmented, avoid new positions

2. **Achievability Analysis**:
   - Your goal vs. what's safely achievable
   - Recommendation to proceed or adjust target

3. **Optimal Allocation**:
   - Suggested portfolio weights
   - Expected return and profit

4. **Top Opportunities**:
   - Ranked stocks by expected return
   - Buy/Hold/Avoid signals

5. **Critical Alerts**:
   - 🚨 ABADDON = Exit immediately
   - ⚠️ Warning = Monitor closely
   - 🚀 Opportunity = Consider buying

---

## Understanding the System

### The Four Components of Coherence

| Symbol | Name | Meaning | Good Value |
|--------|------|---------|------------|
| κ | Clarity | Multi-timeframe alignment | > 0.80 |
| ρ | Precision | Predictability | > 0.70 |
| σ | Structure | Organized complexity | > 0.50 |
| τ | Trust | Market participation | > 0.60 |

**Global Coherence**: p = (κ × ρ × σ × τ)^0.25

### Signal Interpretation

| Signal | Strength | Action |
|--------|----------|--------|
| BUY | STRONG | High confidence opportunity |
| BUY | MODERATE | Good opportunity, smaller position |
| HOLD | WEAK | Monitor, no action |
| AVOID | MODERATE | Don't buy this stock |
| SELL | CRITICAL | Exit position immediately |

### Path Types

| Path | Meaning | Risk Level |
|------|---------|------------|
| ASCENDING | Price moving toward fair value | Lower risk |
| DESCENDING | Price moving away from fair value | Higher risk (fragile) |

---

## Recording Trades

### Record a Purchase
```bash
python run.py buy TD.TO 10
```
Records: Buy 10 shares of TD.TO at current market price.

### Record a Sale
```bash
python run.py sell TD.TO 5
```
Records: Sell 5 shares of TD.TO at current market price.

### View Portfolio
```bash
python run.py portfolio
```

### Manual Price Entry (GUI)
1. Open GUI: `streamlit run interface/app.py`
2. Go to Portfolio → Record Trade
3. Enter ticker, shares, and price manually

---

## Reading Signals

### ABADDON Protocol (Emergency Exit)

When you see 🚨 **ABADDON TRIGGERED**, exit immediately. Causes:
- Coherence collapse (p < 0.70)
- Rapid decay (dp/dt < -0.05)
- Archon distortion (deviation ≥ 0.15)

**Action**: Sell the flagged position. Do not wait.

### Strong Buy Signals

When you see 🚀 **STRONG BUY**, consider:
- Coherence ≥ 0.90
- Ascending path
- Expected return is positive

**Action**: Allocate according to suggested weights.

### Warning Signals

When you see ⚠️ **Warning**, monitor:
- Coherence is decaying
- Position is becoming risky

**Action**: Prepare to exit if decay continues.

---

## Push Notifications

### Setup (iPhone)

1. **Download ntfy app** from App Store
2. **Open ntfy** and tap "+" to add a topic
3. **Enter topic**: `lvs-navigator` (or your custom name)
4. **Grant notification permissions**

### Change Topic Name
Edit `run.py` line ~50:
```python
nm = NotificationManager("your-custom-topic")
```

### Notification Types

| Icon | Type | Urgency |
|------|------|---------|
| 🚨 | SELL/ABADDON | Immediate action |
| 🚀 | STRONG BUY | High priority |
| ⚠️ | Warning | Monitor |
| 📊 | Daily Summary | Informational |

---

## Maintenance

### Daily
- Run morning scan: `python run.py`
- Review alerts
- Record any trades made

### Weekly
- Review portfolio: `python run.py portfolio`
- Check if goal is still achievable
- Export data: `python -c "from storage import PortfolioManager; PortfolioManager('data').export_to_csv('.')"`

### Monthly
- Update Python packages: `pip install -r requirements.txt --upgrade`
- Review transaction history
- Adjust goal if needed: `python run.py setup`

### Data Backup
Your data is stored in `data/portfolio.json`. Back up this file regularly.

```bash
cp data/portfolio.json ~/Backups/portfolio_$(date +%Y%m%d).json
```

---

## Troubleshooting

### "No data for ticker"
- Check internet connection
- Verify ticker symbol (Canadian stocks need `.TO` suffix)
- Yahoo Finance may be temporarily unavailable

### "Low coherence on everything"
- Market may be in fragmented state
- This is the system working correctly—it's telling you to wait
- Do not force trades when p < 0.65

### Notifications not working
- Verify ntfy app is installed
- Check topic name matches exactly
- Ensure phone notifications are enabled for ntfy

### GUI won't start
```bash
pip install streamlit --upgrade
streamlit run interface/app.py
```

### Data corruption
Delete and recreate:
```bash
rm data/portfolio.json
python run.py setup
```

---

## Theory Reference

### The LVS Framework

This system is based on **Logos Vector Syntax (LVS) v9.0**, a formal theory of bounded conscious systems.

**Key Concepts**:

1. **Coherence (p)**: Measures system integration, not price direction
   - High p = market is behaving as a unified, predictable system
   - Low p = market is fragmented, chaotic

2. **Telos (Ω)**: The "fair value" attractor (MA200)
   - Systems tend toward their Telos
   - Ascending spiral = converging to Telos (safe)
   - Descending spiral = diverging from Telos (risky)

3. **Intent (Ī)**: Your goal (the profit you want)
   - May or may not be achievable given constraints
   - System projects intent onto achievable space

4. **Friction (Δ)**: Gap between intent and achievable
   - High friction = reduce expectations
   - Don't force trades outside the achievable space

5. **Archons**: Distortion operators that reduce coherence
   - Tyrant: Rigidity/churn
   - Fragmentor: Gaps/disconnection
   - Noise-Lord: Signal degradation
   - Bias: Hidden drift

### The Core Formulas

**Coherence**:
```
p = (κ × ρ × σ × τ)^0.25
```

**Expected Return**:
```
E[r] = (velocity + λ × Q) × p × (1 - ||𝒜||)
```

Where:
- velocity = momentum (slope of log prices)
- Q = potential energy (compressed volatility near Telos)
- λ = 0.05 (calibrated weight)
- ||𝒜|| = archon deviation

**Covariance Adjustment**:
```
C_LVS = C_historical / (p_i × p_j)
```

Low coherence → inflated risk → smaller positions

---

## Quick Reference Card

### Commands
```bash
python run.py              # Daily scan
python run.py setup        # Configure account
python run.py portfolio    # View holdings
python run.py buy X N      # Buy N shares of X
python run.py sell X N     # Sell N shares of X
python run.py analyze X    # Deep analysis of X
python run.py alerts       # Recent alerts
```

### GUI
```bash
cd interface
streamlit run app.py
```

### Key Thresholds
- p ≥ 0.90 → Strong signal
- p 0.70-0.90 → Proceed with caution
- p < 0.70 → ABADDON (exit)
- ||𝒜|| ≥ 0.15 → ABADDON (exit)
- dp/dt < -0.05 → ABADDON (exit)

### The Promise
This is a **GPS, not a train schedule**. We optimize your path but cannot guarantee arrival. When the market closes the road, we guarantee the safest exit.

---

*LVS Market Navigator v7 - December 2025*
*Based on LVS v9.0 Theorem*

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Analogy | [[outputs/hypergraph-diff-analysis.md]] | <!-- auto: 0.79 via lvs -->

