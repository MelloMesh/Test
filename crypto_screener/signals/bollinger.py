"""
signals/bollinger.py — Bollinger Band signal module.

Three signal types:

1. Mean Reversion    → BB_LOWER / BB_UPPER
2. BB Squeeze + Breakout → BB_SQUEEZE_BULL / BB_SQUEEZE_BEAR
3. %B Extremes       → BB_PCT_B_LOW / BB_PCT_B_HIGH

Squeeze detection methodology ported directly from JS computeSqueeze()
in index.html: bandwidth < 20th percentile of trailing N candles = squeeze.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_screener import config


# ── Core Bollinger Band computation ──────────────────────────────────────────

def compute_bands(
    closes: pd.Series,
    period: int = config.BB_PERIOD,
    std_mult: float = config.BB_STD,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Returns DataFrame with columns:
        mid    — simple moving average
        upper  — mid + std_mult * rolling_std
        lower  — mid - std_mult * rolling_std
        width  — (upper - lower) / mid  (normalized bandwidth)
        pct_b  — (close - lower) / (upper - lower)
    """
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)

    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid
    pct_b = (closes - lower) / (upper - lower)

    return pd.DataFrame({
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "width": width,
        "pct_b": pct_b,
    }, index=closes.index)


# ── Signal 1: Mean Reversion ──────────────────────────────────────────────────

def signal_mean_reversion(df: pd.DataFrame, bands: pd.DataFrame) -> list[str]:
    """
    Close < lower band → BB_LOWER (potential long)
    Close > upper band → BB_UPPER (potential short)
    """
    signals: list[str] = []
    close = df["close"].iloc[-1]
    upper = bands["upper"].iloc[-1]
    lower = bands["lower"].iloc[-1]

    if pd.isna(upper) or pd.isna(lower):
        return signals

    if close < lower:
        signals.append("BB_LOWER")
    elif close > upper:
        signals.append("BB_UPPER")

    return signals


# ── Signal 2: BB Squeeze + Breakout ──────────────────────────────────────────

def _is_in_squeeze(bands: pd.DataFrame, window: int = config.BB_SQUEEZE_WINDOW) -> bool:
    """
    Returns True if the current bandwidth is below the 20th percentile
    of the trailing `window` candles — identical to JS computeSqueeze() logic.
    """
    widths = bands["width"].dropna()
    if len(widths) < window:
        return False

    trailing = widths.iloc[-window:]
    threshold = np.percentile(trailing.values, config.BB_SQUEEZE_PERCENTILE)
    return bool(widths.iloc[-1] < threshold)


def signal_squeeze_breakout(
    df: pd.DataFrame,
    bands: pd.DataFrame,
    avg_vol_window: int = config.VOL_SPIKE_WINDOW,
) -> list[str]:
    """
    Detect post-squeeze breakouts.

    Conditions:
    1. Recent squeeze: bandwidth was in squeeze in the previous N candles
    2. Current candle broke out of a band
    3. Volume > 1.5x average (momentum confirmation)
    """
    signals: list[str] = []

    if len(df) < config.BB_SQUEEZE_WINDOW:
        return signals

    # Check if we were in a squeeze recently (look back ~5 candles)
    was_in_squeeze = False
    for lookback in range(1, 6):
        if len(df) < config.BB_SQUEEZE_WINDOW + lookback:
            break
        sub_bands = bands.iloc[: -(lookback)]
        if _is_in_squeeze(sub_bands, window=config.BB_SQUEEZE_WINDOW):
            was_in_squeeze = True
            break

    if not was_in_squeeze:
        return signals

    close = df["close"].iloc[-1]
    upper = bands["upper"].iloc[-1]
    lower = bands["lower"].iloc[-1]

    if pd.isna(upper) or pd.isna(lower):
        return signals

    # Volume confirmation
    vol_series = df["volume"]
    avg_vol = vol_series.iloc[-avg_vol_window - 1 : -1].mean()
    curr_vol = vol_series.iloc[-1]
    vol_confirmed = (avg_vol > 0) and (curr_vol > config.BB_BREAKOUT_VOL_MULT * avg_vol)

    if vol_confirmed:
        if close > upper:
            signals.append("BB_SQUEEZE_BULL")
        elif close < lower:
            signals.append("BB_SQUEEZE_BEAR")

    return signals


# ── Signal 3: %B Extremes ────────────────────────────────────────────────────

def signal_pct_b(bands: pd.DataFrame) -> list[str]:
    """
    %B < 0.05 → BB_PCT_B_LOW  (extreme oversold)
    %B > 0.95 → BB_PCT_B_HIGH (extreme overbought)
    """
    signals: list[str] = []
    pct_b = bands["pct_b"].iloc[-1]

    if pd.isna(pct_b):
        return signals

    if pct_b < config.BB_PERCENT_B_LOW:
        signals.append("BB_PCT_B_LOW")
    elif pct_b > config.BB_PERCENT_B_HIGH:
        signals.append("BB_PCT_B_HIGH")

    return signals


# ── Composite BB signals ──────────────────────────────────────────────────────

def get_bb_signals(df: pd.DataFrame) -> list[str]:
    """
    Compute all Bollinger Band signals for a single symbol/timeframe DataFrame.
    Returns flat list of signal keys for the scorer.
    """
    bands = compute_bands(df["close"])
    signals: list[str] = []
    signals.extend(signal_mean_reversion(df, bands))
    signals.extend(signal_squeeze_breakout(df, bands))
    signals.extend(signal_pct_b(bands))
    return signals
