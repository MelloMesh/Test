"""
screener/scorer.py — Multi-factor signal combiner and confidence scorer.

Takes the raw signal lists from rsi, bollinger, and volume modules
and produces a structured ScoreResult with composite score, confidence,
direction, and all gating checks.

Signal category mapping (for the ≥2-category diversity gate):
    RSI:      RSI_OS, RSI_OB, RSI_BULL_DIV, RSI_BEAR_DIV, RSI_MID_BULL, RSI_MID_BEAR
    BB:       BB_LOWER, BB_UPPER, BB_SQUEEZE_BULL, BB_SQUEEZE_BEAR, BB_PCT_B_LOW, BB_PCT_B_HIGH
    VOLUME:   VOL_SPIKE, OBV_BULL_DIV, OBV_BEAR_DIV, OBV_CONFIRM, MFI_OS, MFI_OB
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_screener import config

# ── Category membership ───────────────────────────────────────────────────────
_SIGNAL_CATEGORIES: dict[str, str] = {
    "RSI_OS": "RSI",
    "RSI_OB": "RSI",
    "RSI_BULL_DIV": "RSI",
    "RSI_BEAR_DIV": "RSI",
    "RSI_MID_BULL": "RSI",
    "RSI_MID_BEAR": "RSI",

    "BB_LOWER": "BB",
    "BB_UPPER": "BB",
    "BB_SQUEEZE_BULL": "BB",
    "BB_SQUEEZE_BEAR": "BB",
    "BB_PCT_B_LOW": "BB",
    "BB_PCT_B_HIGH": "BB",

    "VOL_SPIKE": "VOLUME",
    "OBV_BULL_DIV": "VOLUME",
    "OBV_BEAR_DIV": "VOLUME",
    "OBV_CONFIRM": "VOLUME",
    "MFI_OS": "VOLUME",
    "MFI_OB": "VOLUME",
}


@dataclass
class ScoreResult:
    symbol: str
    timeframe: str
    signals: list[str]
    composite_score: float
    confidence: float            # 0.0–1.0
    direction: str               # "LONG" or "SHORT"
    passes_threshold: bool
    passes_category_gate: bool
    passes_volume_gate: bool
    reasoning: str               # human-readable summary

    @property
    def surfaced(self) -> bool:
        """A setup is surfaced only when all gates pass."""
        return self.passes_threshold and self.passes_category_gate and self.passes_volume_gate


def _compute_reasoning(signals: list[str], score: float, confidence: float, direction: str) -> str:
    if not signals:
        return "No signals fired"
    fired = ", ".join(signals)
    return f"{direction} | score={score:.2f} conf={confidence:.2f} | {fired}"


def score_setup(
    symbol: str,
    timeframe: str,
    rsi_signals: list[str],
    bb_signals: list[str],
    vol_signals: list[str],
    universe_volumes: list[float] | None = None,
    symbol_volume: float | None = None,
    threshold: float = config.COMPOSITE_THRESHOLD,
    min_categories: int = config.MIN_SIGNAL_CATEGORIES,
) -> ScoreResult:
    """
    Combine signal lists into a composite score and apply all gates.

    Gates:
    1. |composite_score| >= threshold
    2. Signals came from >= min_categories distinct categories
    3. Symbol 24h volume is above 20th percentile of universe (if provided)
    """
    all_signals = rsi_signals + bb_signals + vol_signals

    # ── Composite score ───────────────────────────────────────────────────────
    composite_score = sum(config.WEIGHTS.get(s, 0.0) for s in all_signals)

    # ── Confidence: normalize by max possible score ───────────────────────────
    confidence = min(abs(composite_score) / config.MAX_POSSIBLE_SCORE, 1.0)

    # ── Direction ─────────────────────────────────────────────────────────────
    direction = "LONG" if composite_score >= 0 else "SHORT"

    # ── Gate 1: threshold ─────────────────────────────────────────────────────
    passes_threshold = abs(composite_score) >= threshold

    # ── Gate 2: category diversity ────────────────────────────────────────────
    categories_fired = {_SIGNAL_CATEGORIES.get(s) for s in all_signals if s in _SIGNAL_CATEGORIES}
    categories_fired.discard(None)
    passes_category_gate = len(categories_fired) >= min_categories

    # ── Gate 3: volume liquidity ──────────────────────────────────────────────
    passes_volume_gate = True
    if universe_volumes and symbol_volume is not None and len(universe_volumes) >= 5:
        import numpy as np
        threshold_vol = float(np.percentile(universe_volumes, config.VOL_ILLIQUID_PERCENTILE))
        passes_volume_gate = symbol_volume >= threshold_vol

    reasoning = _compute_reasoning(all_signals, composite_score, confidence, direction)

    return ScoreResult(
        symbol=symbol,
        timeframe=timeframe,
        signals=all_signals,
        composite_score=composite_score,
        confidence=confidence,
        direction=direction,
        passes_threshold=passes_threshold,
        passes_category_gate=passes_category_gate,
        passes_volume_gate=passes_volume_gate,
        reasoning=reasoning,
    )
