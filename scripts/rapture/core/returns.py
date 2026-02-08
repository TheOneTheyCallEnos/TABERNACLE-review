"""
LVS Expected Returns Engine
===========================
Implements E[r] = (velocity + λ × Q) × p × (1 - ||𝒜||)

Kinetic Energy: velocity (momentum/trend)
Potential Energy: Q (coiled spring/compression)

CRITICAL FIX: Q is gated by price position relative to MA200
to prevent "slow bleed" trap (buying stable downtrends).
"""

import numpy as np
import pandas as pd
from scipy import stats


# === CALIBRATION CONSTANTS ===
LAMBDA_POTENTIAL = 0.05  # Potential energy weight (CALIBRATED from 0.3)
EPSILON = 0.01           # Minimum volatility floor
ANNUALIZATION = 252      # Trading days per year


def compute_velocity(prices: pd.Series, window=20) -> float:
    """
    Compute annualized velocity (slope of log prices).
    
    This is the KINETIC energy component - momentum in motion.
    Uses linear regression on log prices.
    """
    if len(prices) < window:
        return 0.0
    
    recent = prices.iloc[-window:]
    log_prices = np.log(recent)
    
    # Time index
    x = np.arange(len(log_prices))
    
    try:
        slope, _, _, _, _ = stats.linregress(x, log_prices)
        # Annualize
        velocity = slope * ANNUALIZATION
        return float(velocity)
    except:
        return 0.0


def compute_volatility(prices: pd.Series, window=20) -> float:
    """
    Compute annualized volatility.
    
    Used in denominator of Q (potential energy).
    Floored at EPSILON to prevent singularity.
    """
    if len(prices) < window:
        return EPSILON
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent = log_returns.iloc[-window:]
    
    if len(recent) < 5:
        return EPSILON
    
    vol = recent.std() * np.sqrt(ANNUALIZATION)
    
    # CRITICAL: Floor to prevent division by zero
    return float(max(vol, EPSILON))


def compute_distance_to_telos(price: float, ma200: float) -> float:
    """
    Compute normalized distance to Telos (fair value).
    
    Telos = MA200 (consensus fair value proxy)
    Distance = |price - MA200| / price
    """
    if price <= 0 or ma200 <= 0:
        return 1.0
    
    dist = abs(price - ma200) / price
    return float(dist)


def compute_q_gate(drawdown_pct: float, volatility: float) -> float:
    """
    Half-Gaussian Q-gate (from Deep Think research, 2026-01-22).

    Instead of binary (Q=0 below MA200), use smooth decay:
        g = exp(-0.5 × (drawdown / σ)²)

    Where σ = 2 × asset volatility (gives ~14% cutoff at 2σ drawdown).

    Properties:
    - At telos (drawdown=0): g = 1.0 (full potential)
    - At 1σ drawdown: g ≈ 0.88
    - At 2σ drawdown: g ≈ 0.61
    - At 3σ drawdown: g ≈ 0.32
    - Asymptotes to 0 for extreme drawdowns
    """
    sigma = 2 * max(volatility, EPSILON)  # σ = 2× volatility
    z = drawdown_pct / sigma
    return float(np.exp(-0.5 * z * z))


def compute_potential_energy(prices: pd.Series, window=20) -> float:
    """
    Q (Potential Energy) - The "coiled spring" factor.

    Formula: Q = g × 1 / (volatility × (1 + dist_to_telos) + ε)

    Where g is the Half-Gaussian Q-gate that smoothly penalizes
    positions below MA200 (telos) based on how far below.

    UPDATED (2026-01-22): Replaced binary gate with smooth Half-Gaussian.
    Research showed binary gates create discontinuities that hurt optimization.
    The smooth gate allows partial potential energy when slightly below telos,
    while still protecting against deep "slow bleed" traps.
    """
    if len(prices) < 200:
        return 0.0

    current_price = prices.iloc[-1]
    ma200 = prices.iloc[-200:].mean()
    vol = compute_volatility(prices, window)

    # Compute drawdown from telos (only counts if below MA200)
    if current_price >= ma200:
        drawdown_pct = 0.0
    else:
        drawdown_pct = (ma200 - current_price) / ma200

    # Apply Half-Gaussian Q-gate
    q_gate = compute_q_gate(drawdown_pct, vol)

    dist = compute_distance_to_telos(current_price, ma200)

    # Potential energy formula with Q-gate
    raw_Q = 1 / (vol * (1 + dist) + EPSILON)
    Q = q_gate * raw_Q

    return float(Q)


def compute_expected_return(prices: pd.Series, volumes: pd.Series,
                            p: float, archon_deviation: float,
                            window=20) -> dict:
    """
    Compute expected return using kinetic + potential energy.
    
    E[r] = (velocity + λ × Q) × p × (1 - ||𝒜||)
    
    Returns dict with breakdown for transparency.
    """
    velocity = compute_velocity(prices, window)
    Q = compute_potential_energy(prices, window)
    
    # Kinetic component
    kinetic = velocity
    
    # Potential component (scaled by lambda)
    potential = LAMBDA_POTENTIAL * Q
    
    # Combined energy
    total_energy = kinetic + potential
    
    # Scale by coherence and archon safety
    safety_factor = 1 - min(archon_deviation, 1.0)
    expected_return = total_energy * p * safety_factor
    
    # Additional metrics for analysis
    ma200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
    current_price = prices.iloc[-1]
    
    return {
        'expected_return': float(expected_return),
        'velocity': float(velocity),
        'potential_Q': float(Q),
        'kinetic_component': float(kinetic * p * safety_factor),
        'potential_component': float(potential * p * safety_factor),
        'volatility': compute_volatility(prices, window),
        'dist_to_telos': compute_distance_to_telos(current_price, ma200),
        'price_vs_ma200': 'ABOVE' if current_price >= ma200 else 'BELOW',
        'ma200': float(ma200),
        'current_price': float(current_price)
    }


def rank_stocks_by_return(stock_data: dict) -> pd.DataFrame:
    """
    Rank stocks by expected return.
    
    stock_data: {ticker: {'prices': Series, 'volumes': Series, 'p': float, 'archon': float}}
    
    Returns DataFrame sorted by expected return.
    """
    results = []
    
    for ticker, data in stock_data.items():
        er = compute_expected_return(
            data['prices'],
            data['volumes'],
            data['p'],
            data['archon']
        )
        er['ticker'] = ticker
        results.append(er)
    
    df = pd.DataFrame(results)
    df = df.sort_values('expected_return', ascending=False)
    
    return df
