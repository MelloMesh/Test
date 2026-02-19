"""
screener/runner.py — Main scan loop.

Scans the configured symbol universe across all timeframes, computes signals,
scores setups, and renders a ranked rich table to the terminal.

Usage:
    python -m crypto_screener.screener.runner [--symbols N] [--threshold T] [--tf 15m 1h 4h]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich import box

from crypto_screener import config
from crypto_screener.data.binance_client import (
    get_top_symbols, get_ohlcv, get_24h_volumes, clear_cache,
)
from crypto_screener.signals.rsi import get_rsi_signals
from crypto_screener.signals.bollinger import get_bb_signals
from crypto_screener.signals.volume import get_volume_signals
from crypto_screener.screener.scorer import score_setup, ScoreResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

console = Console()


# ── Per-symbol-per-timeframe scan ────────────────────────────────────────────

def _scan_one(
    symbol: str,
    timeframe: str,
    universe_volumes: list[float],
    symbol_volume: float,
    threshold: float,
) -> ScoreResult | None:
    """Fetch data, compute signals, and score a single symbol/timeframe pair."""
    try:
        df = get_ohlcv(symbol, interval=timeframe, limit=config.KLINE_LIMIT)
    except Exception as exc:
        logger.warning("Failed to fetch %s/%s: %s", symbol, timeframe, exc)
        return None

    rsi_sigs = get_rsi_signals(df)
    bb_sigs = get_bb_signals(df)
    vol_sigs = get_volume_signals(df)

    result = score_setup(
        symbol=symbol,
        timeframe=timeframe,
        rsi_signals=rsi_sigs,
        bb_signals=bb_sigs,
        vol_signals=vol_sigs,
        universe_volumes=universe_volumes,
        symbol_volume=symbol_volume,
        threshold=threshold,
    )
    return result


# ── Rich table rendering ─────────────────────────────────────────────────────

# Human-readable labels for each signal code
_SIGNAL_LABELS: dict[str, str] = {
    "RSI_OS":          "RSI oversold",
    "RSI_OB":          "RSI overbought",
    "RSI_BULL_DIV":    "RSI bullish divergence",
    "RSI_BEAR_DIV":    "RSI bearish divergence",
    "RSI_MID_BULL":    "RSI momentum rising",
    "RSI_MID_BEAR":    "RSI momentum falling",
    "BB_LOWER":        "price below lower band",
    "BB_UPPER":        "price above upper band",
    "BB_SQUEEZE_BULL": "band squeeze breakout ▲",
    "BB_SQUEEZE_BEAR": "band squeeze breakout ▼",
    "BB_PCT_B_LOW":    "price near lower band",
    "BB_PCT_B_HIGH":   "price near upper band",
    "VOL_SPIKE":       "volume spike",
    "OBV_BULL_DIV":    "OBV bullish divergence",
    "OBV_BEAR_DIV":    "OBV bearish divergence",
    "OBV_CONFIRM":     "volume confirming",
    "MFI_OS":          "money flow oversold",
    "MFI_OB":          "money flow overbought",
}


def _humanize_signals(signals: list[str]) -> str:
    """Convert signal code list to readable comma-separated string."""
    return ", ".join(_SIGNAL_LABELS.get(s, s) for s in signals) if signals else "-"


def _strength_stars(score: float) -> tuple[str, str]:
    """
    Return (star_markup, label) for a given abs(composite_score).

    Scale:  ≥4.0 → ★★★★★  STRONG
            ≥3.0 → ★★★★☆  HIGH
            ≥2.0 → ★★★☆☆  MEDIUM
            ≥1.5 → ★★☆☆☆  LOW
            <1.5 → ★☆☆☆☆  WEAK
    """
    abs_score = abs(score)
    if abs_score >= 4.0:
        return "[bold yellow]★★★★★[/bold yellow]", "STRONG"
    elif abs_score >= 3.0:
        return "[yellow]★★★★☆[/yellow]", "HIGH"
    elif abs_score >= 2.0:
        return "[cyan]★★★☆☆[/cyan]", "MEDIUM"
    elif abs_score >= 1.5:
        return "[dim cyan]★★☆☆☆[/dim cyan]", "LOW"
    else:
        return "[dim]★☆☆☆☆[/dim]", "WEAK"


def render_table(results: list[ScoreResult], show_all: bool = False) -> None:
    """
    Render the screener table.

    Default mode: only confirmed setups that passed all 3 gates.
    --show-all:   every setup scoring ≥ 1.5 (filters single-signal noise),
                  dimmed when gates not fully passed.
    """
    MIN_SHOW_ALL_SCORE = 1.5  # hide trivial single-signal noise in --show-all

    if show_all:
        rows = [r for r in results if abs(r.composite_score) >= MIN_SHOW_ALL_SCORE]
        rows.sort(key=lambda r: abs(r.composite_score), reverse=True)
        title = "Crypto Screener — Watchlist"
    else:
        rows = [r for r in results if r.surfaced]
        rows.sort(key=lambda r: abs(r.composite_score), reverse=True)
        title = "Crypto Screener — Trade Setups"

    if not rows:
        if show_all:
            console.print("\n[yellow]No setups scored ≥ 1.5. Try --symbols 50.[/yellow]")
        else:
            console.print("\n[yellow]No confirmed setups right now.[/yellow]")
            console.print("[dim]Run with --show-all to see developing setups.[/dim]")
        return

    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    )

    table.add_column("SYMBOL",   style="bold white", no_wrap=True)
    table.add_column("TF",       justify="center", style="white")
    table.add_column("TRADE",    justify="center")
    table.add_column("STRENGTH", justify="left")
    table.add_column("WHY",      style="dim")

    for r in rows:
        is_long = r.direction == "LONG"
        trade_markup = "[bright_green]▲ BUY[/bright_green]" if is_long else "[bright_red]▼ SELL[/bright_red]"
        stars, _ = _strength_stars(r.composite_score)
        why = _humanize_signals(r.signals)
        row_style = "" if r.surfaced else "dim"

        table.add_row(
            r.symbol,
            r.timeframe,
            trade_markup,
            stars,
            why,
            style=row_style,
        )

    console.print()
    console.print(table)

    passed = sum(1 for r in rows if r.surfaced)
    ts = __import__("datetime").datetime.utcnow().strftime("%H:%M UTC")

    if show_all:
        console.print(
            f"[dim]{len(rows)} developing setups | "
            f"{passed} confirmed (full strength, bright rows) | "
            f"★★★★★ = score ≥ 4.0  ★★★★☆ = ≥ 3.0  ★★★☆☆ = ≥ 2.0 | "
            f"Scanned at {ts}[/dim]\n"
        )
    else:
        console.print(
            f"[dim]{passed} confirmed setup(s) | "
            f"★★★★★ = STRONG  ★★★★☆ = HIGH  ★★★☆☆ = MEDIUM | "
            f"Scanned at {ts}[/dim]\n"
        )


# ── Main scan loop ────────────────────────────────────────────────────────────

def run_scan(
    n_symbols: int = config.TOP_N_SYMBOLS,
    timeframes: list[str] | None = None,
    threshold: float = config.COMPOSITE_THRESHOLD,
    show_all: bool = False,
) -> list[ScoreResult]:
    """
    Fetch top symbols, scan all timeframes in parallel, return all results.

    show_all=True bypasses gate filtering in the output — every setup that
    fired at least one signal is shown, with GATES column for diagnostics.
    """
    if timeframes is None:
        timeframes = config.TIMEFRAMES

    console.print(f"\n[bold cyan]Fetching top {n_symbols} symbols by volume...[/bold cyan]")
    symbols = get_top_symbols(n_symbols)

    vol_map = get_24h_volumes(symbols)
    universe_volumes = list(vol_map.values())

    total_jobs = len(symbols) * len(timeframes)
    console.print(f"[dim]Scanning {len(symbols)} symbols × {len(timeframes)} timeframes "
                  f"= {total_jobs} jobs...[/dim]")

    all_results: list[ScoreResult] = []

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _scan_one,
                symbol,
                tf,
                universe_volumes,
                vol_map.get(symbol, 0.0),
                threshold,
            ): (symbol, tf)
            for symbol in symbols
            for tf in timeframes
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                all_results.append(result)

            if completed % 20 == 0 or completed == total_jobs:
                console.print(f"[dim]  {completed}/{total_jobs} scanned...[/dim]",
                              end="\r")

    console.print()
    render_table(all_results, show_all=show_all)
    return all_results


# ── Watch mode ────────────────────────────────────────────────────────────────

def _countdown(seconds: int) -> None:
    """Print a live countdown, overwriting the same line."""
    for remaining in range(seconds, 0, -1):
        console.print(
            f"[dim]  Next scan in {remaining:>3}s — Press Ctrl+C to stop[/dim]",
            end="\r",
        )
        time.sleep(1)
    console.print(" " * 60, end="\r")  # clear the line


def watch_loop(
    n_symbols: int,
    timeframes: list[str],
    threshold: float,
    show_all: bool,
    interval: int,
) -> None:
    """
    Run continuous scan cycles, clearing the screen between each cycle.
    Clears the in-memory cache before each cycle so data is always fresh.
    """
    scan_number = 0

    while True:
        scan_number += 1
        clear_cache()  # force fresh data every cycle
        console.clear()

        import datetime
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        console.rule(
            f"[bold cyan]LIVE SCREENER[/bold cyan]  "
            f"[dim]Scan #{scan_number} · {n_symbols} symbols · {ts}[/dim]"
        )

        run_scan(
            n_symbols=n_symbols,
            timeframes=timeframes,
            threshold=threshold,
            show_all=show_all,
        )

        _countdown(interval)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Screener — Binance multi-factor signal scanner"
    )
    parser.add_argument("--symbols", type=int, default=config.TOP_N_SYMBOLS,
                        help=f"Number of top-volume symbols to scan (default: {config.TOP_N_SYMBOLS})")
    parser.add_argument("--threshold", type=float, default=config.COMPOSITE_THRESHOLD,
                        help=f"Minimum composite score to surface (default: {config.COMPOSITE_THRESHOLD})")
    parser.add_argument("--tf", nargs="+", default=config.TIMEFRAMES,
                        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
                        help=f"Timeframes to scan (default: {' '.join(config.TIMEFRAMES)})")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all scored setups scoring ≥ 1.5 (developing watchlist)")
    parser.add_argument("--watch", action="store_true",
                        help="Run continuously, refreshing every --interval seconds")
    parser.add_argument("--interval", type=int, default=config.WATCH_INTERVAL_SECONDS,
                        help=f"Seconds between scan cycles in watch mode "
                             f"(default: {config.WATCH_INTERVAL_SECONDS})")

    args = parser.parse_args()

    try:
        if args.watch:
            watch_loop(
                n_symbols=args.symbols,
                timeframes=args.tf,
                threshold=args.threshold,
                show_all=args.show_all,
                interval=args.interval,
            )
        else:
            run_scan(
                n_symbols=args.symbols,
                timeframes=args.tf,
                threshold=args.threshold,
                show_all=args.show_all,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Screener stopped.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
