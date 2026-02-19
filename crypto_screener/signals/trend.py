"""
signals/trend.py — Trend regime filter.

Provides two key inputs to the scorer's trend-alignment gate:

1. EMA-200 direction: Is price above or below the 200-period EMA?
   - BULL: close > EMA200 × (1 + buffer)  → structural uptrend
   - BEAR: close < EMA200 × (1 − buffer)  → structural downtrend
   - NEUTRAL: within buffer band          → no strong trend

2. ADX strength: Is the market trending or ranging?
   - ADX > 20 → trending; trend direction matters, mean reversion is risky
   - ADX ≤ 20 → ranging; mean reversion signals are valid in either direction

Design rationale (quant perspective):
   In a trending market, taking LONG signals when price is below EMA-200 is
   systematically buying into structural weakness — a statistically poor edge.
   RSI/BB oversold signals in downtrends are "falling knife" traps. Gating on
   EMA-200 + ADX alignment removes this entire category of false positives.

   In ranging (low-ADX) markets, mean reversion is exactly what works — so
   we allow both directions regardless of EMA position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_screener import config


# ── EMA ───────────────────────────────────────────────────────────────────────

def compute_ema(closes: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with standard span parameter."""
    return closes.ewm(span=period, adjust=False).mean()


# ── Wilder's ADX ──────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.Series:
    """
    Wilder's Average Directional Index.

    Steps:
        TR  = max(H-L, |H-prev_C|, |L-prev_C|)
        +DM = max(H - prev_H, 0) if H - prev_H > prev_L - L, else 0
        -DM = max(prev_L - L, 0) if prev_L - L > H - prev_H, else 0
        Smooth all three with Wilder's EWM (alpha = 1/period)
        +DI = 100 × smooth(+DM) / smooth(TR)
        -DI = 100 × smooth(-DM) / smooth(TR)
        DX  = 100 × |+DI - -DI| / (+DI + -DI)
        ADX = Wilder's EWM(DX)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sm_plus = dm_plus.ewm(alpha=alpha, adjust=False).mean()
    sm_minus = dm_minus.ewm(alpha=alpha, adjust=False).mean()

    di_plus = 100.0 * sm_plus / atr.replace(0, np.nan)
    di_minus = 100.0 * sm_minus / atr.replace(0, np.nan)

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


# ── Trend context ─────────────────────────────────────────────────────────────

@dataclass
class TrendContext:
    ema_trend: str       # "BULL", "BEAR", or "NEUTRAL"
    adx: float           # current ADX value (NaN-safe, defaults to 0.0)
    is_trending: bool    # ADX > ADX_TRENDING_THRESHOLD
    ema_period: int      # period used (for logging/debugging)


def get_trend_context(
    df: pd.DataFrame,
    ema_period: int = config.EMA_TREND_PERIOD,
    adx_period: int = config.ADX_PERIOD,
    ema_buffer: float = config.EMA_TREND_BUFFER,
    adx_threshold: float = config.ADX_TRENDING_THRESHOLD,
) -> TrendContext:
    """
    Compute EMA-200 trend direction and ADX regime for a symbol/timeframe DataFrame.

    Returns a TrendContext used by the scorer's trend-alignment gate.

    Note: Requires at least `ema_period` candles for a reliable EMA. Callers
    should check config.MIN_CANDLES before calling this function.
    """
    closes = df["close"]
    ema = compute_ema(closes, ema_period)
    last_close = float(closes.iloc[-1])
    last_ema = float(ema.iloc[-1])

    if pd.isna(last_ema):
        ema_trend = "NEUTRAL"
    elif last_close > last_ema * (1.0 + ema_buffer):
        ema_trend = "BULL"
    elif last_close < last_ema * (1.0 - ema_buffer):
        ema_trend = "BEAR"
    else:
        ema_trend = "NEUTRAL"

    adx_series = compute_adx(df, adx_period)
    last_adx = float(adx_series.iloc[-1])
    if pd.isna(last_adx):
        last_adx = 0.0

    is_trending = last_adx > adx_threshold

    return TrendContext(
        ema_trend=ema_trend,
        adx=last_adx,
        is_trending=is_trending,
        ema_period=ema_period,
    )
