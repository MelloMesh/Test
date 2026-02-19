"""
signals/volume.py — Volume signal module.

Three signal types:

1. Volume Spike    → VOL_SPIKE (direction-neutral momentum confirmation)
2. OBV Trend      → OBV_BULL_DIV / OBV_BEAR_DIV / OBV_CONFIRM
3. MFI (14)       → MFI_OS / MFI_OB

All math is transparent and auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_screener import config


# ── On-Balance Volume ────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    Classic OBV: cumulative sum of volume, added on up-days, subtracted on down-days.
    """
    direction = np.where(df["close"] > df["close"].shift(1), 1,
                np.where(df["close"] < df["close"].shift(1), -1, 0))
    obv = (df["volume"] * direction).cumsum()
    return pd.Series(obv, index=df.index)


def _linear_slope(series: pd.Series, window: int) -> float:
    """
    Linear regression slope over the last `window` points.
    Returns the slope (positive = uptrend, negative = downtrend).
    """
    vals = series.dropna().iloc[-window:].values
    if len(vals) < window:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    # Least squares slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    n = float(len(vals))
    slope = (n * np.dot(x, vals) - x.sum() * vals.sum()) / (n * np.dot(x, x) - x.sum() ** 2)
    return float(slope)


# ── Money Flow Index ─────────────────────────────────────────────────────────

def compute_mfi(df: pd.DataFrame, period: int = config.MFI_PERIOD) -> pd.Series:
    """
    MFI(14): combines price and volume — treated as higher-confidence RSI.

    typical_price  = (H + L + C) / 3
    raw_money_flow = typical_price * volume
    money_flow_ratio = sum(pos_mf, period) / sum(neg_mf, period)
    MFI = 100 - 100 / (1 + money_flow_ratio)
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = tp * df["volume"]

    prev_tp = tp.shift(1)
    pos_mf = raw_mf.where(tp > prev_tp, 0.0)
    neg_mf = raw_mf.where(tp < prev_tp, 0.0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return mfi


# ── Signal 1: Volume Spike ────────────────────────────────────────────────────

def signal_volume_spike(
    df: pd.DataFrame,
    window: int = config.VOL_SPIKE_WINDOW,
    mult: float = config.VOL_SPIKE_MULT,
) -> list[str]:
    """
    Current volume > 2.5x 20-period rolling average → VOL_SPIKE.
    Direction-neutral; used as multiplier to other signals.
    """
    vol = df["volume"]
    if len(vol) < window + 1:
        return []

    avg_vol = vol.iloc[-window - 1 : -1].mean()
    if avg_vol <= 0:
        return []

    if vol.iloc[-1] > mult * avg_vol:
        return ["VOL_SPIKE"]

    return []


# ── Signal 2: OBV Trend ───────────────────────────────────────────────────────

def signal_obv(
    df: pd.DataFrame,
    slope_window: int = config.OBV_SLOPE_WINDOW,
) -> list[str]:
    """
    Compare OBV slope direction to price slope direction.

    OBV slope UP,   price slope DOWN → OBV_BULL_DIV (early bullish warning)
    OBV slope DOWN, price slope UP   → OBV_BEAR_DIV (early bearish warning)
    OBV slope UP,   price slope UP   → OBV_CONFIRM (bullish confirmation)
    OBV slope DOWN, price slope DOWN → OBV_CONFIRM (bearish — handled by scorer direction)
    """
    signals: list[str] = []

    if len(df) < slope_window + 5:
        return signals

    obv = compute_obv(df)
    obv_slope = _linear_slope(obv, slope_window)
    price_slope = _linear_slope(df["close"], slope_window)

    # Normalize to sign only
    obv_up = obv_slope > 0
    price_up = price_slope > 0

    if obv_up and not price_up:
        signals.append("OBV_BULL_DIV")
    elif not obv_up and price_up:
        signals.append("OBV_BEAR_DIV")
    elif obv_up and price_up:
        signals.append("OBV_CONFIRM")
    # Both down = bearish OBV_CONFIRM — scorer uses direction to handle this

    return signals


# ── Signal 3: MFI ────────────────────────────────────────────────────────────

def signal_mfi(df: pd.DataFrame) -> list[str]:
    """
    MFI < 20 → MFI_OS (oversold, higher-confidence than RSI alone)
    MFI > 80 → MFI_OB (overbought)
    """
    signals: list[str] = []

    if len(df) < config.MFI_PERIOD + 1:
        return signals

    mfi = compute_mfi(df)
    last = mfi.iloc[-1]

    if pd.isna(last):
        return signals

    if last < config.MFI_OVERSOLD:
        signals.append("MFI_OS")
    elif last > config.MFI_OVERBOUGHT:
        signals.append("MFI_OB")

    return signals


# ── Volume illiquidity check ──────────────────────────────────────────────────

def is_illiquid(volumes: list[float], percentile: float = config.VOL_ILLIQUID_PERCENTILE) -> bool:
    """
    Returns True if the current symbol's 24h volume is below the
    given percentile of the universe — used to gate illiquid setups.
    """
    if not volumes:
        return False
    threshold = np.percentile(volumes, percentile)
    return bool(volumes[-1] < threshold)


# ── Composite Volume signals ──────────────────────────────────────────────────

def get_volume_signals(df: pd.DataFrame) -> list[str]:
    """
    Compute all volume signals for a single symbol/timeframe DataFrame.
    Returns flat list of signal keys for the scorer.
    """
    signals: list[str] = []
    signals.extend(signal_volume_spike(df))
    signals.extend(signal_obv(df))
    signals.extend(signal_mfi(df))
    return signals
