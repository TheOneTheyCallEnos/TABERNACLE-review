"""
LVS Portfolio Navigator v7.1
============================
CALIBRATED: Relaxed thresholds for realistic market conditions
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

from .coherence import compute_coherence, compute_coherence_history, compute_dp_dt
from .returns import compute_expected_return, compute_velocity
from .risk import compute_archon_deviation, compute_covariance_lvs, check_abaddon_triggers


# === CONSTRAINTS (CALIBRATED v7.1) ===
SIGMA_MAX_BASE = 0.15
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.50
CASH_MIN = 0.05

# Signal thresholds (RELAXED)
P_STRONG_BUY = 0.75    # Was 0.90
P_MODERATE_BUY = 0.60  # Was 0.80
P_HOLD = 0.50          # Was 0.65

# Kelly Criterion parameters (from Deep Research, 2026-01-22)
KELLY_ALPHA = 1.5      # Power-law shrinkage exponent (research optimal)
KELLY_MAX_FRACTION = 0.25  # Max position size per asset


def coherence_kelly(edge: float, odds: float, coherence: float,
                    alpha: float = KELLY_ALPHA) -> float:
    """
    Regime-Aware Kelly Criterion (from Deep Research, 2026-01-22).

    f* = c^α × (bp − q) / b

    Where:
    - c = coherence score ∈ [0,1]
    - α = 1.5 (power-law shrinkage, more conservative than linear)
    - b = payoff odds (win amount / loss amount)
    - p = win probability (derived from coherence and expected return)
    - q = 1 - p

    Key insight: Power-law (c^1.5) beats linear (c^1.0) because:
    - When c < 0.5, positions reduced by 70-90% (protective)
    - When c > 0.8, near-full Kelly (growth-preserving)

    Returns position fraction [0, KELLY_MAX_FRACTION].
    """
    if coherence <= 0 or odds <= 0 or edge <= 0:
        return 0.0

    # Win probability from edge (simplified: edge ≈ win_rate - loss_rate)
    p = min(0.5 + edge / 2, 0.95)  # Cap at 95% to avoid overconfidence
    q = 1 - p

    # Raw Kelly fraction
    raw_kelly = (odds * p - q) / odds

    if raw_kelly <= 0:
        return 0.0

    # Coherence-adjusted (power-law shrinkage)
    shrunk_kelly = (coherence ** alpha) * raw_kelly

    # Cap at maximum fraction
    return min(shrunk_kelly, KELLY_MAX_FRACTION)


class LVSNavigator:
    
    def __init__(self, capital: float, goal: float, timeframe_days: int):
        self.capital = capital
        self.goal = goal
        self.target_value = capital + goal
        self.timeframe = timeframe_days
        self.target_return = goal / capital
        
        self.stock_data = {}
        self.coherence_cache = {}
        self.portfolio = {}
        
    def add_stock_data(self, ticker: str, prices: pd.Series, volumes: pd.Series):
        self.stock_data[ticker] = {
            'prices': prices,
            'volumes': volumes
        }
    
    def analyze_stock(self, ticker: str) -> dict:
        if ticker not in self.stock_data:
            raise ValueError(f"No data for {ticker}")
        
        data = self.stock_data[ticker]
        prices = data['prices']
        volumes = data['volumes']
        
        p_prev = self.coherence_cache.get(ticker, {}).get('p', 0.5)
        
        coherence = compute_coherence(prices, volumes, p_prev)
        self.coherence_cache[ticker] = coherence
        
        coh_history = compute_coherence_history(prices, volumes, lookback=30)
        dp_dt = compute_dp_dt(coh_history)
        
        archon = compute_archon_deviation(prices, volumes)
        
        exp_ret = compute_expected_return(
            prices, volumes,
            coherence['p'],
            archon['deviation']
        )
        
        abaddon = check_abaddon_triggers(coherence['p'], dp_dt, archon['deviation'])
        
        ma200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
        current = prices.iloc[-1]
        path = 'ASCENDING' if current >= ma200 else 'DESCENDING'
        
        return {
            'ticker': ticker,
            'coherence': coherence,
            'dp_dt': dp_dt,
            'archon': archon,
            'expected_return': exp_ret,
            'abaddon': abaddon,
            'path': path,
            'current_price': current,
            'ma200': ma200,
            'signal': self._generate_signal(coherence['p'], dp_dt, archon, exp_ret, path)
        }
    
    def _generate_signal(self, p: float, dp_dt: float, archon: dict,
                         exp_ret: dict, path: str) -> dict:
        
        # ABADDON check (uses risk.py threshold)
        if archon['alert']:
            return {
                'action': 'SELL',
                'strength': 'CRITICAL',
                'reason': 'Archon distortion critical'
            }
        
        if p < P_HOLD:
            return {
                'action': 'AVOID',
                'strength': 'MODERATE',
                'reason': f'Low coherence ({p:.2f}), market fragmented'
            }
        
        # Strong buy: good coherence + ascending + positive momentum
        if p >= P_STRONG_BUY and path == 'ASCENDING' and exp_ret['expected_return'] > 0.20:
            return {
                'action': 'BUY',
                'strength': 'STRONG',
                'reason': f'High coherence ({p:.2f}), ascending path, strong E[r]'
            }
        
        # Moderate buy: decent coherence + positive expected return
        if p >= P_MODERATE_BUY and exp_ret['expected_return'] > 0.10:
            return {
                'action': 'BUY',
                'strength': 'MODERATE',
                'reason': f'Good coherence ({p:.2f}), positive expected return'
            }
        
        # Weak buy: above hold threshold with some upside
        if p >= P_HOLD and exp_ret['expected_return'] > 0:
            return {
                'action': 'BUY',
                'strength': 'WEAK',
                'reason': f'Moderate coherence ({p:.2f}), marginal opportunity'
            }
        
        return {
            'action': 'HOLD',
            'strength': 'WEAK',
            'reason': f'Coherence ({p:.2f}), no clear opportunity'
        }
    
    def analyze_all(self) -> pd.DataFrame:
        results = []
        for ticker in self.stock_data:
            try:
                analysis = self.analyze_stock(ticker)
                results.append({
                    'ticker': ticker,
                    'p': analysis['coherence']['p'],
                    'kappa': analysis['coherence']['kappa'],
                    'rho': analysis['coherence']['rho'],
                    'sigma': analysis['coherence']['sigma'],
                    'tau': analysis['coherence']['tau'],
                    'dp_dt': analysis['dp_dt'],
                    'expected_return': analysis['expected_return']['expected_return'],
                    'velocity': analysis['expected_return']['velocity'],
                    'potential_Q': analysis['expected_return']['potential_Q'],
                    'archon_deviation': analysis['archon']['deviation'],
                    'dominant_archon': analysis['archon']['dominant_archon'],
                    'path': analysis['path'],
                    'signal': analysis['signal']['action'],
                    'signal_strength': analysis['signal']['strength'],
                    'signal_reason': analysis['signal']['reason'],
                    'price': analysis['current_price'],
                    'ma200': analysis['ma200'],
                    'abaddon': analysis['abaddon']['triggered']
                })
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('expected_return', ascending=False)
        return df
    
    def compute_market_coherence(self) -> float:
        p_values = [self.coherence_cache[t]['p'] 
                    for t in self.coherence_cache if self.coherence_cache[t]['p'] > 0]
        
        if not p_values:
            return 0.5
        
        return np.mean(p_values)
    
    def compute_achievable_target(self) -> dict:
        market_p = self.compute_market_coherence()
        max_risk = SIGMA_MAX_BASE * market_p
        
        analyses = self.analyze_all()
        if len(analyses) == 0:
            return {
                'achievable': False,
                'max_return': 0,
                'max_profit': 0,
                'reason': 'No stocks analyzed'
            }
        
        # Filter to viable (BUY signals, no ABADDON)
        viable = analyses[
            (analyses['signal'].isin(['BUY'])) &
            (~analyses['abaddon'])
        ]
        
        if len(viable) == 0:
            # Fallback: any stock above hold threshold
            viable = analyses[
                (analyses['p'] >= P_HOLD) &
                (~analyses['abaddon'])
            ]
        
        if len(viable) == 0:
            return {
                'achievable': False,
                'max_return': 0,
                'max_profit': 0,
                'reason': 'No viable stocks found',
                'market_p': market_p
            }
        
        avg_expected = viable['expected_return'].mean()
        max_expected = viable['expected_return'].max()
        
        # Scale by time fraction and confidence
        safe_return = avg_expected * market_p * (self.timeframe / 252)
        max_return = max_expected * market_p * (self.timeframe / 252)
        
        safe_profit = self.capital * safe_return
        max_profit = self.capital * max_return
        
        achievable = self.goal <= max_profit
        
        return {
            'achievable': achievable,
            'goal': self.goal,
            'safe_target': safe_profit,
            'max_target': max_profit,
            'safe_return_pct': safe_return * 100,
            'max_return_pct': max_return * 100,
            'target_return_pct': self.target_return * 100,
            'market_p': market_p,
            'viable_stocks': len(viable),
            'recommendation': 'PROCEED' if achievable else f'REDUCE TARGET to ${safe_profit:.2f}'
        }
    
    def optimize_portfolio(self, max_stocks: int = 5, use_kelly: bool = True) -> dict:
        """
        Optimize portfolio allocation.

        UPDATED (2026-01-22): Now uses Coherence-Adjusted Kelly Criterion
        for position sizing when use_kelly=True (default).

        Kelly sizing provides:
        - Automatic position reduction in low-coherence regimes
        - Power-law shrinkage (α=1.5) for conservative risk management
        - Max 25% per position to prevent concentration risk
        """
        analyses = self.analyze_all()

        # Get buyable stocks
        buyable = analyses[
            (analyses['signal'] == 'BUY') &
            (~analyses['abaddon'])
        ].head(max_stocks)

        if len(buyable) == 0:
            # Fallback to any viable stock
            buyable = analyses[
                (analyses['p'] >= P_HOLD) &
                (~analyses['abaddon']) &
                (analyses['expected_return'] > 0)
            ].head(max_stocks)

        if len(buyable) == 0:
            return {
                'success': False,
                'weights': {'CASH': 1.0},
                'reason': 'No viable stocks',
                'sizing_method': 'none'
            }

        tickers = buyable['ticker'].tolist()
        n = len(tickers)

        exp_returns = buyable['expected_return'].values
        p_scores = buyable['p'].values

        if use_kelly:
            # Coherence-Adjusted Kelly Sizing
            kelly_fractions = []
            for i in range(n):
                # Edge: expected return (annualized)
                edge = max(exp_returns[i], 0)
                # Odds: assume 1:1 for simplicity (can be refined with actual win/loss data)
                odds = 1.0
                # Coherence: from LVS calculation
                coherence = p_scores[i]

                fraction = coherence_kelly(edge, odds, coherence)
                kelly_fractions.append(fraction)

            raw_weights = np.array(kelly_fractions)

            # Normalize if total > (1 - CASH_MIN)
            total_weight = raw_weights.sum()
            max_equity = 1 - CASH_MIN

            if total_weight > max_equity:
                raw_weights = raw_weights * (max_equity / total_weight)

            sizing_method = 'coherence_kelly'
        else:
            # Legacy method: weight by expected return * coherence
            scores = exp_returns * p_scores
            scores = np.maximum(scores, 0)

            if scores.sum() > 0:
                raw_weights = scores / scores.sum()
            else:
                raw_weights = np.ones(n) / n

            # Apply constraints
            raw_weights = raw_weights * (1 - CASH_MIN)
            raw_weights = np.minimum(raw_weights, MAX_WEIGHT)

            sizing_method = 'score_weighted'

        weights = {tickers[i]: float(raw_weights[i]) for i in range(n)}
        weights['CASH'] = max(1 - sum(weights.values()), CASH_MIN)

        # Portfolio metrics
        portfolio_return = np.dot(raw_weights, exp_returns)

        market_p = self.compute_market_coherence()

        return {
            'success': True,
            'weights': weights,
            'tickers': tickers,
            'expected_return': portfolio_return,
            'expected_profit': self.capital * portfolio_return * (self.timeframe / 252),
            'market_p': market_p,
            'sizing_method': sizing_method,
            'kelly_alpha': KELLY_ALPHA if use_kelly else None
        }
    
    def compute_portfolio_telos(self, weights: dict) -> dict:
        telos = 0
        current_value = 0
        
        for ticker, weight in weights.items():
            if ticker == 'CASH' or weight <= 0:
                continue
            
            if ticker not in self.stock_data:
                continue
            
            prices = self.stock_data[ticker]['prices']
            ma200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
            current = prices.iloc[-1]
            
            telos += weight * ma200
            current_value += weight * current
        
        cash_weight = weights.get('CASH', 0)
        telos += cash_weight
        current_value += cash_weight
        
        portfolio_telos = telos * self.capital
        portfolio_current = current_value * self.capital
        
        dist_current_to_telos = abs(portfolio_current - portfolio_telos)
        dist_target_to_telos = abs(self.target_value - portfolio_telos)
        
        if dist_target_to_telos < dist_current_to_telos:
            path = 'ASCENDING'
            path_safety = 'SAFE'
        else:
            path = 'DESCENDING'
            path_safety = 'FRAGILE'
        
        return {
            'portfolio_telos': portfolio_telos,
            'portfolio_current': portfolio_current,
            'target_value': self.target_value,
            'path': path,
            'path_safety': path_safety,
            'dist_to_telos': dist_current_to_telos,
            'converging': portfolio_current <= portfolio_telos
        }
    
    def generate_full_report(self) -> dict:
        achievable = self.compute_achievable_target()
        portfolio = self.optimize_portfolio()
        telos = self.compute_portfolio_telos(portfolio['weights']) if portfolio['success'] else {}
        analyses = self.analyze_all()
        
        return {
            'capital': self.capital,
            'goal': self.goal,
            'timeframe_days': self.timeframe,
            'achievability': achievable,
            'portfolio': portfolio,
            'telos': telos,
            'stock_analyses': analyses.to_dict('records') if len(analyses) > 0 else [],
            'alerts': self._generate_alerts(analyses)
        }
    
    def _generate_alerts(self, analyses: pd.DataFrame) -> list:
        alerts = []
        
        if len(analyses) == 0:
            return alerts
        
        abaddon_stocks = analyses[analyses['abaddon'] == True]
        for _, row in abaddon_stocks.iterrows():
            alerts.append({
                'type': 'ABADDON',
                'severity': 'CRITICAL',
                'ticker': row['ticker'],
                'message': f"EXIT {row['ticker']} IMMEDIATELY - ABADDON triggered"
            })
        
        strong_buys = analyses[
            (analyses['signal'] == 'BUY') & 
            (analyses['signal_strength'] == 'STRONG')
        ]
        for _, row in strong_buys.iterrows():
            alerts.append({
                'type': 'OPPORTUNITY',
                'severity': 'HIGH',
                'ticker': row['ticker'],
                'message': f"STRONG BUY: {row['ticker']} (p={row['p']:.2f}, E[r]={row['expected_return']:.1%})"
            })
        
        decay_stocks = analyses[analyses['dp_dt'] < -0.03]
        for _, row in decay_stocks.iterrows():
            alerts.append({
                'type': 'WARNING',
                'severity': 'MEDIUM',
                'ticker': row['ticker'],
                'message': f"Coherence decaying on {row['ticker']} (dp/dt={row['dp_dt']:.3f})"
            })
        
        return alerts
