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
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich import box

from crypto_screener import config
from crypto_screener.data.binance_client import get_top_symbols, get_ohlcv, get_24h_volumes
from crypto_screener.signals.rsi import get_rsi_signals
from crypto_screener.signals.bollinger import get_bb_signals
from crypto_screener.signals.volume import get_volume_signals
from crypto_screener.screener.scorer import score_setup, ScoreResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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

def _direction_color(direction: str) -> str:
    return "bright_green" if direction == "LONG" else "bright_red"


def _gate_str(r: ScoreResult) -> str:
    """Compact gate status: green label = passed, red = failed."""
    parts = []
    parts.append("[green]SCORE[/green]" if r.passes_threshold else "[red]SCORE[/red]")
    parts.append("[green]CAT[/green]" if r.passes_category_gate else "[red]CAT[/red]")
    parts.append("[green]VOL[/green]" if r.passes_volume_gate else "[red]VOL[/red]")
    return " ".join(parts)


def render_table(results: list[ScoreResult], show_all: bool = False) -> None:
    """
    Render the ranked screener table using rich.

    show_all=False (default): only surface setups that passed all 3 gates.
    show_all=True:            show every scored setup with a GATES column
                              indicating which gates passed (green) or failed (red).
                              Rows that failed any gate are dimmed.
    """
    if show_all:
        rows = [r for r in results if r.signals]
        rows.sort(key=lambda r: r.confidence, reverse=True)
        title = f"{config.RICH_TABLE_TITLE} — ALL SIGNALS"
    else:
        rows = [r for r in results if r.surfaced]
        rows.sort(key=lambda r: r.confidence, reverse=True)
        title = config.RICH_TABLE_TITLE

    if not rows:
        if show_all:
            console.print("\n[yellow]No signals fired at all.[/yellow]")
            console.print("[dim]Try --symbols 50 or adding more timeframes.[/dim]")
        else:
            console.print("\n[yellow]No setups crossed the confidence threshold.[/yellow]")
            console.print(
                f"[dim]Threshold: {config.COMPOSITE_THRESHOLD} | "
                f"Min categories: {config.MIN_SIGNAL_CATEGORIES} | "
                f"Try --show-all to see everything scored.[/dim]"
            )
        return

    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )

    table.add_column("SYMBOL", style="bold white", no_wrap=True)
    table.add_column("TF", justify="center", style="dim white")
    table.add_column("DIR", justify="center")
    table.add_column("SCORE", justify="right")
    table.add_column("CONF", justify="right")
    if show_all:
        table.add_column("GATES", justify="center")
    table.add_column("SIGNALS FIRED", style="dim")

    for r in rows:
        dir_color = _direction_color(r.direction)
        score_fmt = f"{r.composite_score:+.2f}"
        conf_fmt = f"{r.confidence:.2f}"
        signals_str = ", ".join(r.signals) if r.signals else "-"
        row_style = "" if r.surfaced else "dim"

        if show_all:
            table.add_row(
                r.symbol,
                r.timeframe,
                f"[{dir_color}]{r.direction}[/{dir_color}]",
                f"[bold]{score_fmt}[/bold]",
                conf_fmt,
                _gate_str(r),
                signals_str,
                style=row_style,
            )
        else:
            table.add_row(
                r.symbol,
                r.timeframe,
                f"[{dir_color}]{r.direction}[/{dir_color}]",
                f"[bold]{score_fmt}[/bold]",
                conf_fmt,
                signals_str,
            )

    console.print()
    console.print(table)

    passed = sum(1 for r in rows if r.surfaced)
    if show_all:
        console.print(
            f"[dim]{len(rows)} scored setups | "
            f"{passed} passed all gates (bright rows) | "
            f"Threshold ≥ {config.COMPOSITE_THRESHOLD} | "
            f"Ranked by confidence DESC[/dim]\n"
        )
    else:
        console.print(
            f"[dim]{passed} setup(s) passed all gates | "
            f"Threshold ≥ {config.COMPOSITE_THRESHOLD} | "
            f"Use --show-all to see full scored universe[/dim]\n"
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

    with ThreadPoolExecutor(max_workers=8) as executor:
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
                        help="Show all scored setups regardless of gate filtering "
                             "(useful for validating signal generation)")

    args = parser.parse_args()

    try:
        run_scan(
            n_symbols=args.symbols,
            timeframes=args.tf,
            threshold=args.threshold,
            show_all=args.show_all,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
