"""Pre-defined strategy lego blocks to prevent mathematical hallucination.

Each block is a proven, parameterized fragment mapped to an underlying
strategy type in the registry. These are surfaced in the UI via a dropdown;
users (or the AI agent) pick from this constrained set to build a portfolio.
"""
from typing import List, Dict, Any

LEGO_BLOCKS: List[Dict[str, Any]] = [
    {
        "name": "RSI Mean Reversion (Short-Term)",
        "strategy_type": "mean_reversion",
        "params": {"window": 20, "num_std": 2.0},
        "description": "Buys when price deviates negatively from its 20-day mean by 2 std dev; exits on mean reversion.",
        "formula": "z_t = (P_t - MA_{20}) / (\sigma_{20}); Buy if z_t < -2; Sell if z_t > 0"
    },
    {
        "name": "Bollinger Breakout",  # maps to breakout logic
        "strategy_type": "breakout",
        "params": {"window": 40, "num_std": 2.5},
        "description": "Enters long on upper band expansion; exits or reverses on squeeze breakdown.",
        "formula": "Upper = MA_{40} + 2.5*\sigma_{40}; Lower = MA_{40} - 2.5*\sigma_{40}; Long if P_t > Upper"
    },
    {
        "name": "Momentum Crossover (21/63)",
        "strategy_type": "momentum",
        "params": {"fast_window": 21, "slow_window": 63},
        "description": "Classic dual moving-average trend capture variant.",
        "formula": "Signal = MA_{21} - MA_{63}; Long if Signal > 0; Flat otherwise"
    },
    {
        "name": "Volatility Target (ATR)",
        "strategy_type": "volatility",
        "params": {"window": 30, "vol_threshold": 0.02},
        "description": "Adjusts position size based on realized volatility compared to threshold.",
        "formula": "pos_size_t = min(1, vol_threshold / realized_vol_{30})"
    },
    {
        "name": "Regime Adaptive Blend",
        "strategy_type": "regime_based",
        "params": {"risk_on_weight": 0.7, "risk_off_weight": 0.3},
        "description": "Shifts allocation weights based on detected macro/volatility regime.",
        "formula": "w_t = RISK_ON if regime in {Growth, LowVol} else RISK_OFF"
    }
]

# Convenience index by name
LEGO_BLOCK_MAP = {b["name"]: b for b in LEGO_BLOCKS}
