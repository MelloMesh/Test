"""
signals/rsi.py — RSI signal module.

Three signal types (matching JS computeRSISeries / detectDivergence logic,
ported to pandas + scipy):

1. Overbought / Oversold       → RSI_OS / RSI_OB
2. Divergence                  → RSI_BULL_DIV / RSI_BEAR_DIV  (highest conviction)
3. Mid-line cross (50)         → RSI_MID_BULL / RSI_MID_BEAR

Each public function returns a list[str] of fired signal keys that map
directly to config.WEIGHTS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from crypto_screener import config


# ── Core RSI computation ──────────────────────────────────────────────────────

def compute_rsi(closes: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """
    Wilder's smoothed RSI.
    Transparent implementation — no ta library black box.

    Algorithm:
        delta  = close.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        avg_gain = EWM with alpha=1/period (Wilder smoothing)
        avg_loss = EWM with alpha=1/period
        RS   = avg_gain / avg_loss
        RSI  = 100 - 100/(1+RS)
    """
    if len(closes) < period + 1:
        return pd.Series(np.nan, index=closes.index)

    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing = EWM with adjust=False, alpha = 1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # When avg_loss = 0 (all up-candles) → RSI = 100.
    # When both = 0 (flat) → RSI = 50.
    # Avoid NaN from 0/0 division.
    rsi = pd.Series(
        np.where(
            avg_loss == 0,
            np.where(avg_gain == 0, 50.0, 100.0),
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan)),
        ),
        index=closes.index,
    )
    # First delta is always NaN → propagate to RSI warmup
    rsi[closes.diff().isna()] = np.nan
    return rsi


# ── Signal 1: Overbought / Oversold ──────────────────────────────────────────

def signal_os_ob(rsi: pd.Series) -> list[str]:
    """
    Fire RSI_OS if last RSI < 30, RSI_OB if last RSI > 70.
    Returns list of signal keys.
    """
    signals: list[str] = []
    last = rsi.iloc[-1]
    if pd.isna(last):
        return signals
    if last < config.RSI_OVERSOLD:
        signals.append("RSI_OS")
    elif last > config.RSI_OVERBOUGHT:
        signals.append("RSI_OB")
    return signals


# ── Signal 2: Mid-line cross ──────────────────────────────────────────────────

def signal_midline(rsi: pd.Series) -> list[str]:
    """
    Detect RSI crossing 50.
    Bullish: previous RSI ≤ 50, current RSI > 50 → RSI_MID_BULL
    Bearish: previous RSI ≥ 50, current RSI < 50 → RSI_MID_BEAR
    """
    signals: list[str] = []
    if len(rsi.dropna()) < 2:
        return signals

    clean = rsi.dropna()
    prev, curr = float(clean.iloc[-2]), float(clean.iloc[-1])

    if prev <= config.RSI_MIDLINE < curr:
        signals.append("RSI_MID_BULL")
    elif prev >= config.RSI_MIDLINE > curr:
        signals.append("RSI_MID_BEAR")
    return signals


# ── Signal 3: Divergence ─────────────────────────────────────────────────────

def _find_extrema(series: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return indices of local maxima and minima.
    Uses scipy.signal.argrelextrema — same approach as JS detectDivergence()
    which walks candles looking for swing highs/lows.
    """
    maxima = argrelextrema(series, np.greater_equal, order=order)[0]
    minima = argrelextrema(series, np.less_equal, order=order)[0]
    return maxima, minima


def signal_divergence(
    df: pd.DataFrame,
    rsi: pd.Series,
    min_window: int = config.RSI_DIV_MIN_WINDOW,
    max_window: int = config.RSI_DIV_MAX_WINDOW,
    order: int = config.RSI_DIV_ORDER,
) -> list[str]:
    """
    Detect RSI divergence over the last `max_window` candles.

    Bullish divergence:  price makes LOWER low  but RSI makes HIGHER low
    Bearish divergence:  price makes HIGHER high but RSI makes LOWER high

    Both conditions must be within the detection window and the divergence
    pair must span at least `min_window` candles apart.

    Logic mirrors JS detectDivergence() pattern from index.html.
    """
    signals: list[str] = []

    closes = df["close"].values
    rsi_vals = rsi.values

    if len(closes) < max_window + order * 2:
        return signals

    # Work on the detection window
    window_closes = closes[-max_window:]
    window_rsi = rsi_vals[-max_window:]

    if np.any(np.isnan(window_rsi)):
        return signals

    maxima, minima = _find_extrema(window_closes, order)
    rsi_maxima, rsi_minima = _find_extrema(window_rsi, order)

    # ── Bearish divergence: price higher high, RSI lower high ────────────────
    if len(maxima) >= 2 and len(rsi_maxima) >= 2:
        # Two most recent price highs
        ph1_idx, ph2_idx = int(maxima[-2]), int(maxima[-1])
        # Two most recent RSI highs
        rh1_idx, rh2_idx = int(rsi_maxima[-2]), int(rsi_maxima[-1])

        span = abs(ph2_idx - ph1_idx)
        if min_window <= span <= max_window:
            price_higher_high = window_closes[ph2_idx] > window_closes[ph1_idx]
            rsi_lower_high = window_rsi[rh2_idx] < window_rsi[rh1_idx]
            if price_higher_high and rsi_lower_high:
                signals.append("RSI_BEAR_DIV")

    # ── Bullish divergence: price lower low, RSI higher low ─────────────────
    if len(minima) >= 2 and len(rsi_minima) >= 2:
        pl1_idx, pl2_idx = int(minima[-2]), int(minima[-1])
        rl1_idx, rl2_idx = int(rsi_minima[-2]), int(rsi_minima[-1])

        span = abs(pl2_idx - pl1_idx)
        if min_window <= span <= max_window:
            price_lower_low = window_closes[pl2_idx] < window_closes[pl1_idx]
            rsi_higher_low = window_rsi[rl2_idx] > window_rsi[rl1_idx]
            if price_lower_low and rsi_higher_low:
                signals.append("RSI_BULL_DIV")

    return signals


# ── Composite RSI signals ─────────────────────────────────────────────────────

def get_rsi_signals(df: pd.DataFrame) -> list[str]:
    """
    Compute all RSI signals for a single symbol/timeframe DataFrame.
    Returns a flat list of signal keys ready for the scorer.
    """
    rsi = compute_rsi(df["close"])
    signals: list[str] = []
    signals.extend(signal_os_ob(rsi))
    signals.extend(signal_midline(rsi))
    signals.extend(signal_divergence(df, rsi))
    return signals
