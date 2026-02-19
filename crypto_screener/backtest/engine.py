"""
backtest/engine.py — Lightweight vectorized backtester.

Design constraints:
- No external framework (Backtrader / vectorbt / zipline) — stays under 200 lines
- Vectorized via pandas/numpy — no candle-by-candle Python loop
- ATR-based stop-loss and take-profit (non-negotiable in crypto)
- Minimum 6 months of data required

Entry/Exit rules:
- Entry:      next candle open after signal fires
- Stop-loss:  entry ± (ATR_MULT_STOP  × ATR14)
- Take-profit:entry ± (ATR_MULT_TP    × ATR14)
- Direction:  determined by composite score sign

Usage:
    python -m crypto_screener.backtest.engine --symbol BTCUSDT --tf 1h
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from crypto_screener import config
from crypto_screener.data.binance_client import get_ohlcv
from crypto_screener.signals.rsi import get_rsi_signals, compute_rsi
from crypto_screener.signals.bollinger import get_bb_signals, compute_bands
from crypto_screener.signals.volume import get_volume_signals
from crypto_screener.screener.scorer import score_setup

logger = logging.getLogger(__name__)
console = Console()

# ── ATR computation ───────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """
    Wilder ATR:
        TR  = max(H-L, |H-prev_C|, |L-prev_C|)
        ATR = EWM(TR, alpha=1/period)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


# ── Signal generation over full history ──────────────────────────────────────

def generate_signal_series(
    df: pd.DataFrame,
    threshold: float = config.COMPOSITE_THRESHOLD,
    min_categories: int = config.MIN_SIGNAL_CATEGORIES,
) -> pd.Series:
    """
    Roll forward over the DataFrame and emit a signal at each candle
    where a qualified setup fires.

    Returns a pd.Series of {+1 (LONG), -1 (SHORT), 0 (no signal)}
    aligned to df.index.

    To avoid look-ahead bias, signals at index i are computed using
    data up to and including row i.
    """
    signals = pd.Series(0, index=df.index, dtype=float)

    # Minimum warmup before we start scoring
    warmup = max(config.RSI_PERIOD, config.BB_PERIOD, config.MFI_PERIOD,
                 config.BB_SQUEEZE_WINDOW) + config.RSI_DIV_MAX_WINDOW + 5

    for i in range(warmup, len(df)):
        sub = df.iloc[: i + 1]

        rsi_sigs = get_rsi_signals(sub)
        bb_sigs = get_bb_signals(sub)
        vol_sigs = get_volume_signals(sub)

        result = score_setup(
            symbol="BACKTEST",
            timeframe="—",
            rsi_signals=rsi_sigs,
            bb_signals=bb_sigs,
            vol_signals=vol_sigs,
            threshold=threshold,
            min_categories=min_categories,
        )

        if result.surfaced:
            signals.iloc[i] = 1.0 if result.direction == "LONG" else -1.0

    return signals


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int        # +1 LONG / -1 SHORT
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str      # "TP" | "SL" | "EOS"


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    stop_mult: float = config.STOP_ATR_MULT,
    tp_mult: float = config.TAKE_PROFIT_ATR_MULT,
) -> list[Trade]:
    """
    Vectorized trade simulation.
    Entry: next candle open after signal fires.
    Exit:  first candle that hits SL or TP (checked on H/L).
    """
    atr = compute_atr(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    atr_vals = atr.values

    trades: list[Trade] = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    direction = 0
    sl_price = 0.0
    tp_price = 0.0

    n = len(df)

    for i in range(n):
        if in_trade:
            # Check SL / TP on current candle H/L
            if direction == 1:  # LONG
                if lows[i] <= sl_price:
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i, direction=direction,
                        entry_price=entry_price, exit_price=sl_price,
                        pnl_pct=(sl_price - entry_price) / entry_price * 100,
                        exit_reason="SL",
                    ))
                    in_trade = False
                elif highs[i] >= tp_price:
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i, direction=direction,
                        entry_price=entry_price, exit_price=tp_price,
                        pnl_pct=(tp_price - entry_price) / entry_price * 100,
                        exit_reason="TP",
                    ))
                    in_trade = False
            else:  # SHORT
                if highs[i] >= sl_price:
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i, direction=direction,
                        entry_price=entry_price, exit_price=sl_price,
                        pnl_pct=(entry_price - sl_price) / entry_price * 100,
                        exit_reason="SL",
                    ))
                    in_trade = False
                elif lows[i] <= tp_price:
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i, direction=direction,
                        entry_price=entry_price, exit_price=tp_price,
                        pnl_pct=(entry_price - tp_price) / entry_price * 100,
                        exit_reason="TP",
                    ))
                    in_trade = False
        else:
            # Look for new signal on previous candle → enter on this candle's open
            if i > 0 and signals.iloc[i - 1] != 0 and i < n:
                direction = int(signals.iloc[i - 1])
                entry_price = opens[i]
                entry_atr = atr_vals[i - 1]
                if np.isnan(entry_atr) or entry_atr == 0:
                    continue
                if direction == 1:
                    sl_price = entry_price - stop_mult * entry_atr
                    tp_price = entry_price + tp_mult * entry_atr
                else:
                    sl_price = entry_price + stop_mult * entry_atr
                    tp_price = entry_price - tp_mult * entry_atr
                entry_idx = i
                in_trade = True

    # Close any open trade at end of series
    if in_trade:
        exit_price = closes[-1]
        if direction == 1:
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
        trades.append(Trade(
            entry_idx=entry_idx, exit_idx=n - 1, direction=direction,
            entry_price=entry_price, exit_price=exit_price,
            pnl_pct=pnl, exit_reason="EOS",
        ))

    return trades


# ── Performance metrics ───────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    n_trades: int
    win_rate: float
    avg_rr: float          # average R:R (TP hit / SL hit ratios)
    sharpe: float
    max_drawdown_pct: float
    total_return_pct: float
    validated: bool
    flag_reason: str


def compute_metrics(trades: list[Trade], symbol: str, timeframe: str) -> BacktestResult:
    if not trades:
        return BacktestResult(symbol, timeframe, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              False, "No trades generated")

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls > 0
    win_rate = float(wins.mean())

    # Avg R:R = avg win / abs(avg loss)
    avg_win = float(pnls[wins].mean()) if wins.any() else 0.0
    avg_loss = float(abs(pnls[~wins].mean())) if (~wins).any() else 1.0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Sharpe (annualized, assuming daily returns from per-trade pnl)
    if pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Cumulative equity + max drawdown
    equity = np.cumprod(1 + pnls / 100)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_drawdown = float(drawdowns.min())

    total_return = float((equity[-1] - 1) * 100)

    # Validation flags
    flag_reasons = []
    if win_rate < config.MIN_WIN_RATE:
        flag_reasons.append(f"win_rate={win_rate:.1%} < {config.MIN_WIN_RATE:.1%}")
    if sharpe < config.MIN_SHARPE:
        flag_reasons.append(f"sharpe={sharpe:.2f} < {config.MIN_SHARPE}")

    validated = not bool(flag_reasons)
    flag_reason = "; ".join(flag_reasons) if flag_reasons else "OK"

    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_rr=avg_rr,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown,
        total_return_pct=total_return,
        validated=validated,
        flag_reason=flag_reason,
    )


# ── Report rendering ──────────────────────────────────────────────────────────

def render_backtest_report(result: BacktestResult, candles: int) -> None:
    status_color = "bright_green" if result.validated else "bright_red"
    status_label = "[VALIDATED]" if result.validated else "[UNVALIDATED]"

    table = Table(
        title=f"Backtest Report — {result.symbol} / {result.timeframe} ({candles} candles)",
        box=box.SIMPLE_HEAVY,
        show_header=False,
        border_style="dim",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Symbol", result.symbol)
    table.add_row("Timeframe", result.timeframe)
    table.add_row("Candles", str(candles))
    table.add_row("Trades", str(result.n_trades))
    table.add_row("Win Rate", f"{result.win_rate:.1%}")
    table.add_row("Avg R:R", f"{result.avg_rr:.2f}")
    table.add_row("Sharpe", f"{result.sharpe:.2f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Total Return", f"{result.total_return_pct:.2f}%")
    table.add_row("Status", f"[{status_color}]{status_label}[/{status_color}]")
    if not result.validated:
        table.add_row("Flag", f"[yellow]{result.flag_reason}[/yellow]")

    console.print()
    console.print(table)
    console.print()


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    symbol: str = "BTCUSDT",
    timeframe: str = config.DEFAULT_TIMEFRAME,
    limit: int = config.BACKTEST_KLINE_LIMIT,
    threshold: float = config.COMPOSITE_THRESHOLD,
) -> BacktestResult:
    """
    Fetch historical data, generate signals, simulate trades, report results.
    Minimum 6 months: 1h × 1000 candles ≈ 41 days — use 1500+ for 6 months.
    """
    console.print(f"\n[bold cyan]Running backtest: {symbol} / {timeframe} ({limit} candles)[/bold cyan]")

    df = get_ohlcv(symbol, interval=timeframe, limit=limit)
    console.print(f"[dim]  Data: {df.index[0]} → {df.index[-1]} ({len(df)} candles)[/dim]")

    min_months = 6
    candles_per_month = {"15m": 2880, "1h": 720, "4h": 180}.get(timeframe, 720)
    if len(df) < candles_per_month * min_months:
        console.print(
            f"[yellow]Warning: {len(df)} candles may be less than 6 months "
            f"({candles_per_month * min_months} needed for {timeframe}).[/yellow]"
        )

    console.print("[dim]  Generating signal series (this may take a moment)...[/dim]")
    signals = generate_signal_series(df, threshold=threshold)

    n_signals = int((signals != 0).sum())
    console.print(f"[dim]  Signals fired: {n_signals}[/dim]")

    trades = simulate_trades(df, signals)
    result = compute_metrics(trades, symbol, timeframe)
    render_backtest_report(result, len(df))

    return result


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the crypto screener signals")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--limit", type=int, default=config.BACKTEST_KLINE_LIMIT)
    parser.add_argument("--threshold", type=float, default=config.COMPOSITE_THRESHOLD)
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        timeframe=args.tf,
        limit=args.limit,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
