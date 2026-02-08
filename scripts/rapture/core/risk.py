"""
LVS Risk Engine
===============
Implements:
- Archon Deviation ||𝒜|| detection (Tyrant, Fragmentor, Noise-Lord, Bias)
- Coherence-adjusted covariance matrix C_LVS
- ABADDON protocol triggers

ARCHON THRESHOLD: ||𝒜|| >= 0.15 triggers exit

CALIBRATION NOTES (v7.1):
- Noise-Lord detection scaled to avoid false positives
- Archon norm scaled by 1/4 to bring into realistic range
- Markets naturally have noise; only flag EXTREME distortion
"""

import numpy as np
import pandas as pd
from scipy import stats


# === THRESHOLDS ===
ARCHON_THRESHOLD = 0.15  # Exit trigger
P_COLLAPSE = 0.50        # Coherence collapse threshold
DP_DT_DECAY = -0.05      # Rapid decay threshold
EPSILON = 1e-6           # Numerical stability


def compute_tyrant(returns: pd.Series, volumes: pd.Series, window=20) -> float:
    """
    Tyrant Archon - Rigidity/Churn Detection
    
    High volume + zero price movement = churning/pinned
    """
    if len(returns) < window or len(volumes) < window:
        return 0.0
    
    recent_vol = volumes.iloc[-window:]
    recent_ret = returns.iloc[-window:]
    
    avg_vol = recent_vol.mean()
    current_vol = recent_vol.iloc[-1]
    current_ret = abs(recent_ret.iloc[-1])
    
    if avg_vol <= 0:
        return 0.0
    
    vol_ratio = current_vol / avg_vol
    
    if current_ret > 0.01:
        return 0.0
    
    d_T = vol_ratio * (0.01 - current_ret) / 0.01
    d_T = np.clip(d_T / 3, 0, 1)
    
    return float(d_T)


def compute_fragmentor(prices: pd.Series, window=20) -> float:
    """
    Fragmentor Archon - Gap/Disconnection Detection
    """
    if len(prices) < window:
        return 0.0
    
    gaps = prices.diff().abs()
    avg_price = prices.iloc[-window:].mean()
    
    recent_gap = gaps.iloc[-1] if len(gaps) > 0 else 0
    gap_pct = recent_gap / avg_price if avg_price > 0 else 0
    
    if gap_pct < 0.02:
        return 0.0
    
    d_F = np.clip((gap_pct - 0.02) / 0.05, 0, 1)
    
    return float(d_F)


def compute_noise_lord(returns: pd.Series, window=20) -> float:
    """
    Noise-Lord Archon - Signal Degradation Detection
    
    CALIBRATED: Only flag extreme noise with high volatility.
    """
    if len(returns) < window + 1:
        return 0.0
    
    recent = returns.iloc[-window:].dropna()
    
    if len(recent) < 10:
        return 0.0
    
    try:
        autocorr = recent.autocorr(lag=1)
        if np.isnan(autocorr):
            autocorr = 0
        
        vol = recent.std()
        avg_vol = returns.std()
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1
        
        if abs(autocorr) > 0.1 or vol_ratio < 1.5:
            return 0.0
        
        d_N = np.clip((vol_ratio - 1.5) / 1.5, 0, 1)
        
        return float(d_N)
    except:
        return 0.0


def compute_bias(prices: pd.Series, window=14) -> float:
    """
    Bias Archon - Hidden Drift Detection
    """
    if len(prices) < window + 10:
        return 0.0
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    
    rs = gain / (loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    
    recent_prices = prices.iloc[-window:]
    recent_rsi = rsi.iloc[-window:].dropna()
    
    if len(recent_rsi) < 5:
        return 0.0
    
    try:
        x = np.arange(len(recent_prices))
        price_slope, _, _, _, _ = stats.linregress(x, recent_prices)
        
        x_rsi = np.arange(len(recent_rsi))
        rsi_slope, _, _, _, _ = stats.linregress(x_rsi, recent_rsi)
        
        price_direction = np.sign(price_slope)
        rsi_direction = np.sign(rsi_slope)
        
        if price_direction == rsi_direction:
            return 0.0
        
        price_strength = abs(price_slope) / recent_prices.std() if recent_prices.std() > 0 else 0
        rsi_strength = abs(rsi_slope) / 10
        
        d_B = np.clip(min(price_strength, rsi_strength), 0, 1)
        
        return float(d_B)
    except:
        return 0.0


def compute_archon_deviation(prices: pd.Series, volumes: pd.Series) -> dict:
    """
    Compute total Archon deviation ||𝒜||.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    d_T = compute_tyrant(log_returns, volumes)
    d_F = compute_fragmentor(prices)
    d_N = compute_noise_lord(log_returns)
    d_B = compute_bias(prices)
    
    raw_norm = np.sqrt(d_T**2 + d_F**2 + d_N**2 + d_B**2)
    norm = raw_norm / 2
    
    archons = {'Tyrant': d_T, 'Fragmentor': d_F, 'Noise-Lord': d_N, 'Bias': d_B}
    dominant = max(archons, key=archons.get)
    
    if max(archons.values()) == 0:
        dominant = 'None'
    
    return {
        'deviation': float(norm),
        'tyrant': float(d_T),
        'fragmentor': float(d_F),
        'noise_lord': float(d_N),
        'bias': float(d_B),
        'dominant_archon': dominant,
        'alert': norm >= ARCHON_THRESHOLD,
        'threshold': ARCHON_THRESHOLD
    }


def compute_covariance_lvs(returns_df: pd.DataFrame, p_scores: np.ndarray,
                           window=60) -> np.ndarray:
    """
    Compute coherence-adjusted covariance matrix.

    CRITICAL FIX (2026-01-22): Previous formula C_hist / (p_i × p_j) destroyed
    positive semi-definiteness (PSD), making the matrix invalid for optimization.

    NEW APPROACH: Diagonal variance adjustment only.
    - Inflate variances by 1/p² (higher uncertainty → higher perceived risk)
    - Preserve off-diagonal correlations (keeps PSD intact)

    This implements the research finding that coherence should scale
    UNCERTAINTY about variance, not the covariance structure itself.
    """
    C_hist = returns_df.iloc[-window:].cov().values

    # Floor coherence scores to prevent extreme inflation
    p_floored = np.maximum(p_scores, 0.3)

    # Extract standard deviations (sqrt of diagonal)
    std_hist = np.sqrt(np.diag(C_hist))

    # Inflate standard deviations by 1/p (lower coherence = higher uncertainty)
    std_adjusted = std_hist / p_floored

    # Compute correlation matrix from historical covariance
    # corr_ij = cov_ij / (std_i × std_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        std_outer = np.outer(std_hist, std_hist)
        corr_matrix = np.where(std_outer > EPSILON, C_hist / std_outer, 0)

    # Rebuild covariance with adjusted stds but SAME correlations
    # C_LVS_ij = corr_ij × std_adjusted_i × std_adjusted_j
    C_LVS = corr_matrix * np.outer(std_adjusted, std_adjusted)

    return C_LVS


def check_abaddon_triggers(p: float, dp_dt: float, archon_deviation: float) -> dict:
    """
    Check ABADDON (emergency exit) triggers.
    """
    triggers = {
        'coherence_collapse': p < P_COLLAPSE,
        'rapid_decay': dp_dt < DP_DT_DECAY,
        'archon_distortion': archon_deviation >= ARCHON_THRESHOLD
    }
    
    abaddon_triggered = any(triggers.values())
    reasons = [k for k, v in triggers.items() if v]
    
    return {
        'triggered': abaddon_triggered,
        'triggers': triggers,
        'reasons': reasons,
        'action': 'EXIT TO CASH' if abaddon_triggered else 'HOLD',
        'metrics': {
            'p': p,
            'dp_dt': dp_dt,
            'archon_deviation': archon_deviation
        }
    }


def compute_risk_metrics(prices: pd.Series, volumes: pd.Series,
                         p: float, dp_dt: float) -> dict:
    """
    Compute all risk metrics for a single stock.
    """
    archon = compute_archon_deviation(prices, volumes)
    abaddon = check_abaddon_triggers(p, dp_dt, archon['deviation'])
    
    return {
        'archon': archon,
        'abaddon': abaddon,
        'risk_level': 'CRITICAL' if abaddon['triggered'] else (
            'HIGH' if archon['deviation'] > 0.10 else (
                'MEDIUM' if archon['deviation'] > 0.05 else 'LOW'
            )
        )
    }
