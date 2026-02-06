"""
Realistic synthetic market data generator for backtesting.

Generates BTC-like 15m candle data with realistic properties:
- Trending phases (up and down)
- Ranging/consolidation phases
- Mean-reverting swings within ranges (ideal for divergence strategy)
- Realistic volatility (ATR ~1-3% of price)
- Volume patterns (higher on moves, lower in consolidation)
- Fibonacci-like retracement patterns after swings
"""

import numpy as np
import pandas as pd


def generate_btc_like_data(
    n_candles: int = 5000,
    seed: int = 42,
    base_price: float = 95000.0,
    timeframe_minutes: int = 15,
) -> pd.DataFrame:
    """
    Generate realistic BTC-like 15m OHLCV data.

    Creates a sequence of market regimes:
    - Uptrend → Retracement into golden pocket → Bounce or break
    - Downtrend → Rally into golden pocket → Rejection or break
    - Ranging → Oscillation between support/resistance

    Args:
        n_candles: Total candles to generate.
        seed: Random seed for reproducibility.
        base_price: Starting price.
        timeframe_minutes: Minutes per candle.

    Returns:
        DataFrame with timestamp, open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)

    prices = np.zeros(n_candles)
    volumes = np.zeros(n_candles)
    prices[0] = base_price

    # Market regime parameters
    regime_length = 0
    regime_type = "range"  # "uptrend", "downtrend", "range"
    regime_target_len = rng.integers(80, 200)

    # Volatility (BTC 15m: ~0.1-0.3% per candle)
    base_volatility = base_price * 0.0015
    current_volatility = base_volatility

    for i in range(1, n_candles):
        regime_length += 1

        # Switch regime
        if regime_length >= regime_target_len:
            regime_length = 0
            regime_target_len = rng.integers(80, 250)

            if regime_type == "uptrend":
                # After uptrend: often range or retrace
                regime_type = rng.choice(["range", "downtrend", "range"], p=[0.5, 0.3, 0.2])
            elif regime_type == "downtrend":
                regime_type = rng.choice(["range", "uptrend", "range"], p=[0.5, 0.3, 0.2])
            else:
                regime_type = rng.choice(["uptrend", "downtrend", "range"], p=[0.35, 0.35, 0.3])

        # Calculate drift based on regime
        if regime_type == "uptrend":
            drift = current_volatility * 0.15  # Slight upward bias
        elif regime_type == "downtrend":
            drift = -current_volatility * 0.15  # Slight downward bias
        else:
            # Mean-revert to recent average (creates swing patterns)
            lookback = min(i, 50)
            recent_avg = np.mean(prices[max(0, i - lookback):i])
            drift = (recent_avg - prices[i - 1]) * 0.03  # Mean reversion

        # Price change with regime-appropriate behavior
        noise = rng.standard_normal() * current_volatility
        change = drift + noise

        # Add occasional large moves (fat tails)
        if rng.random() < 0.02:
            change *= rng.uniform(2.0, 4.0) * rng.choice([-1, 1])

        prices[i] = prices[i - 1] + change

        # Keep price positive and realistic
        prices[i] = max(prices[i], base_price * 0.5)
        prices[i] = min(prices[i], base_price * 2.0)

        # Adaptive volatility (increases after large moves)
        recent_return = abs(change / prices[i - 1])
        current_volatility = 0.95 * current_volatility + 0.05 * (base_volatility * (1 + recent_return * 20))

        # Volume (higher on big moves, lower in consolidation)
        base_vol = 1000.0
        vol_multiplier = 1.0 + abs(change / current_volatility) * 0.5
        if regime_type != "range":
            vol_multiplier *= 1.3
        volumes[i] = base_vol * vol_multiplier * rng.uniform(0.7, 1.5)

    volumes[0] = 1000.0

    # Build OHLC from close prices
    opens = np.zeros(n_candles)
    highs = np.zeros(n_candles)
    lows = np.zeros(n_candles)

    opens[0] = prices[0] - rng.uniform(0, base_volatility * 0.3)
    for i in range(1, n_candles):
        opens[i] = prices[i - 1] + rng.standard_normal() * current_volatility * 0.1
        candle_range = abs(prices[i] - opens[i]) + rng.uniform(0, current_volatility * 0.5)
        highs[i] = max(opens[i], prices[i]) + rng.uniform(0, candle_range * 0.3)
        lows[i] = min(opens[i], prices[i]) - rng.uniform(0, candle_range * 0.3)

    highs[0] = prices[0] + base_volatility * 0.3
    lows[0] = prices[0] - base_volatility * 0.3

    # Generate timestamps
    timestamps = pd.date_range(
        "2025-01-01",
        periods=n_candles,
        freq=f"{timeframe_minutes}min",
        tz="UTC",
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })


def generate_multi_seed_data(
    seeds: list[int] | None = None,
    n_candles: int = 5000,
) -> dict[int, pd.DataFrame]:
    """
    Generate multiple datasets with different seeds for robust testing.

    Returns:
        Dict mapping seed to DataFrame.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]

    return {seed: generate_btc_like_data(n_candles=n_candles, seed=seed) for seed in seeds}
