"""
LVS Coherence Engine
====================
Implements the core coherence formula: p = (κ × ρ × σ × τ)^0.25

CALIBRATION v7.1:
- κ (Clarity): Relaxed sigmoid center from 0.5 to 0.3
- All components floored at 0.2 to prevent total collapse
"""

import numpy as np
import pandas as pd
from scipy import stats
import zlib


# === CALIBRATION CONSTANTS ===
ALPHA_PRECISION = 10_000
LAMBDA_POTENTIAL = 0.05
EPSILON = 0.01
KAPPA_SIGMOID_SCALE = 8
KAPPA_SIGMOID_CENTER = 0.3  # Relaxed from 0.5
COMPONENT_FLOOR = 0.2  # Minimum value for any component


def compute_kappa(prices: pd.Series, window_short=5, window_med=20, window_long=60) -> float:
    """
    κ (Clarity) - Multi-timeframe alignment
    """
    if len(prices) < window_long:
        return 0.5
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    ret_short = log_returns.rolling(window_short).sum().dropna()
    ret_med = log_returns.rolling(window_med).sum().dropna()
    ret_long = log_returns.rolling(window_long).sum().dropna()
    
    min_len = min(len(ret_short), len(ret_med), len(ret_long))
    if min_len < 10:
        return 0.5
    
    ret_short = ret_short.iloc[-min_len:]
    ret_med = ret_med.iloc[-min_len:]
    ret_long = ret_long.iloc[-min_len:]
    
    # Direction agreement (simpler, more robust)
    short_dir = float(np.sign(ret_short.iloc[-1]))
    med_dir = float(np.sign(ret_med.iloc[-1]))
    long_dir = float(np.sign(ret_long.iloc[-1]))
    
    agreement = 0
    if short_dir == med_dir:
        agreement += 1
    if med_dir == long_dir:
        agreement += 1
    if short_dir == long_dir:
        agreement += 1
    
    # Base clarity from agreement (0.33 to 1.0)
    base_clarity = (agreement + 1) / 4
    
    # Also check correlation strength
    try:
        corr_sm = abs(np.corrcoef(ret_short, ret_med)[0, 1])
        corr_ml = abs(np.corrcoef(ret_med, ret_long)[0, 1])
        if np.isnan(corr_sm):
            corr_sm = 0.5
        if np.isnan(corr_ml):
            corr_ml = 0.5
        corr_avg = (corr_sm + corr_ml) / 2
    except:
        corr_avg = 0.5
    
    # Combine agreement and correlation
    kappa = (base_clarity + corr_avg) / 2
    
    # Apply sigmoid for smoothing
    kappa = 1 / (1 + np.exp(-KAPPA_SIGMOID_SCALE * (kappa - KAPPA_SIGMOID_CENTER)))
    
    return float(max(kappa, COMPONENT_FLOOR))


def compute_rho(returns: pd.Series, window=20) -> float:
    """
    ρ (Precision) - Inverse prediction error variance
    """
    if len(returns) < window + 1:
        return 0.5
    
    recent_returns = returns.iloc[-window:].dropna()
    if len(recent_returns) < 10:
        return 0.5
    
    y = recent_returns.iloc[1:].values
    x = recent_returns.iloc[:-1].values
    
    if len(x) < 5:
        return 0.5
    
    try:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        predictions = intercept + slope * x
        residuals = y - predictions
        var_epsilon = np.var(residuals)
        
        rho = 1 / (1 + ALPHA_PRECISION * var_epsilon)
        return float(max(rho, COMPONENT_FLOOR))
    except:
        return 0.5


def compute_sigma(log_returns: pd.Series, kappa: float, window=60) -> float:
    """
    σ (Structure) - Effective complexity with Logos Override
    """
    if len(log_returns) < window:
        return 0.5
    
    recent = log_returns.iloc[-window:].dropna()
    if len(recent) < 20:
        return 0.5
    
    int_returns = (recent * 10000).astype(int)
    
    data_bytes = int_returns.values.tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    
    x = len(compressed) / len(data_bytes) if len(data_bytes) > 0 else 0.5
    x = np.clip(x, 0.01, 0.99)
    
    # LOGOS OVERRIDE
    if kappa > 0.85 and x < 0.2:
        return 1.0
    
    sigma = 4 * x * (1 - x)
    
    return float(max(sigma, COMPONENT_FLOOR))


def compute_tau(prices: pd.Series, volumes: pd.Series, p_prev: float,
                window=20) -> float:
    """
    τ (Trust) - Openness gated by prior coherence
    """
    if len(prices) < 50 or len(volumes) < window:
        return 0.5
    
    close = prices.iloc[-1]
    ma50 = prices.iloc[-50:].mean()
    
    price_range = prices.iloc[-window:].max() - prices.iloc[-window:].min()
    atr20 = price_range / window if window > 0 else 1
    atr20 = max(atr20, EPSILON)
    
    avg_vol = volumes.iloc[-window:].mean()
    current_vol = volumes.iloc[-1]
    avg_vol = max(avg_vol, 1)
    
    daily_ranges = prices.diff().abs().iloc[-window:]
    daily_range = daily_ranges.iloc[-1] if len(daily_ranges) > 0 else atr20
    
    s_safe = np.tanh((close - ma50) / atr20)
    s_novel = np.tanh((current_vol - avg_vol) / avg_vol)
    s_stable = 1 - np.tanh(daily_range / atr20)
    
    g = np.tanh(0.5 * s_safe + 0.3 * s_novel + 0.2 * s_stable)
    tau_raw = (g + 1) / 2
    
    if p_prev < 0.80:
        tau = tau_raw * 0.7  # Less harsh dampening
    else:
        tau = tau_raw
    
    return float(max(tau, COMPONENT_FLOOR))


def compute_coherence(prices: pd.Series, volumes: pd.Series, 
                      p_prev: float = 0.5) -> dict:
    """
    Compute global coherence p = (κ × ρ × σ × τ)^0.25
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    kappa = compute_kappa(prices)
    rho = compute_rho(log_returns)
    sigma = compute_sigma(log_returns, kappa)
    tau = compute_tau(prices, volumes, p_prev)
    
    p = (kappa * rho * sigma * tau) ** 0.25
    
    return {
        'kappa': kappa,
        'rho': rho,
        'sigma': sigma,
        'tau': tau,
        'p': float(np.clip(p, 0, 1)),
        'components': {
            'clarity': kappa,
            'precision': rho,
            'structure': sigma,
            'trust': tau
        }
    }


def compute_coherence_history(prices: pd.Series, volumes: pd.Series,
                               lookback=30) -> pd.DataFrame:
    """
    Compute coherence over time for dp/dt calculation.
    """
    results = []
    
    for i in range(lookback, len(prices)):
        price_slice = prices.iloc[:i+1]
        vol_slice = volumes.iloc[:i+1]
        
        p_prev = results[-1]['p'] if results else 0.5
        
        coh = compute_coherence(price_slice, vol_slice, p_prev)
        coh['date'] = prices.index[i]
        results.append(coh)
    
    return pd.DataFrame(results)


def compute_dp_dt(coherence_history: pd.DataFrame, window=5) -> float:
    """
    Compute rate of change of coherence.
    """
    if len(coherence_history) < window:
        return 0.0
    
    recent_p = coherence_history['p'].iloc[-window:]
    
    if len(recent_p) < 2:
        return 0.0
    
    dp_dt = (recent_p.iloc[-1] - recent_p.iloc[0]) / window
    
    return float(dp_dt)
