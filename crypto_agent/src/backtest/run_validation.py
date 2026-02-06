"""
EV Validation Runner.

Downloads real Binance USDT-M perpetual futures data from data.binance.vision
(official Binance bulk data — free, no API keys) and runs the full backtest
pipeline to validate EV.

Usage:
    cd crypto_agent
    python3 -m src.backtest.run_validation

Data source: https://data.binance.vision/
  - Real BTCUSDT & ETHUSDT perpetual futures klines
  - 15m and 30m candles, 12 months of history
  - Downloaded as ZIP → CSV, cached in data/historical/
"""

import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.report import generate_report, print_report
from src.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

HISTORICAL_DIR = DATA_DIR / "historical"


# ---------------------------------------------------------------------------
# Data download from data.binance.vision
# ---------------------------------------------------------------------------

def download_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    year: int = 2024,
    months: list[int] | None = None,
) -> Path:
    """
    Download monthly kline CSVs from data.binance.vision.

    These are official Binance bulk data exports — free, no API keys,
    no rate limits. Real perpetual futures OHLCV data.

    Args:
        symbol: e.g. "BTCUSDT" (no slash, no colon).
        interval: e.g. "15m", "30m".
        year: Year to download.
        months: List of months (1-12). Default: all 12.

    Returns:
        Path to directory containing the downloaded CSVs.
    """
    if months is None:
        months = list(range(1, 13))

    out_dir = HISTORICAL_DIR / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://data.binance.vision/data/futures/um/monthly/klines"

    for month in months:
        fname = f"{symbol}-{interval}-{year}-{month:02d}"
        csv_path = out_dir / f"{fname}.csv"

        if csv_path.exists():
            print(f"  Already have {csv_path.name}, skipping")
            continue

        url = f"{base_url}/{symbol}/{interval}/{fname}.zip"
        zip_path = out_dir / f"{fname}.zip"

        print(f"  Downloading {fname}.zip ...")
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            with zipfile.ZipFile(str(zip_path), "r") as z:
                z.extractall(str(out_dir))
            zip_path.unlink()
            print(f"    -> {csv_path.name}")
        except Exception as e:
            print(f"    FAILED: {e}")
            if zip_path.exists():
                zip_path.unlink()

    return out_dir


def load_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Load downloaded CSV klines into a DataFrame.

    Column mapping from Binance CSVs:
    open_time, open, high, low, close, volume, close_time,
    quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore

    Returns:
        DataFrame with: timestamp, open, high, low, close, volume
    """
    data_dir = HISTORICAL_DIR / symbol / interval
    csv_files = sorted(glob.glob(str(data_dir / f"{symbol}-{interval}-*.csv")))

    if not csv_files:
        print(f"  No CSV files found in {data_dir}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(
            f,
            header=None,
            names=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "count",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ],
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Binance CSVs may include header rows — drop any non-numeric open_time
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined = combined.dropna(subset=["open_time"])
    combined["open_time"] = combined["open_time"].astype(int)

    # Binance timestamps: milliseconds (pre-2025) or microseconds (2025+)
    # Detect: if max timestamp > year 2100 in ms, it's microseconds
    max_ts = combined["open_time"].max()
    unit = "us" if max_ts > 4_102_444_800_000 else "ms"

    combined["timestamp"] = pd.to_datetime(combined["open_time"], unit=unit, utc=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    result = combined[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    print(f"  Loaded {len(result)} candles for {symbol} {interval}")
    print(f"  Range: {result['timestamp'].iloc[0]} → {result['timestamp'].iloc[-1]}")

    return result


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def run_validation(
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    year: int = 2024,
    months: list[int] | None = None,
    initial_equity: float = 10000.0,
) -> dict:
    """
    Download real Binance futures data and run full backtests.

    Args:
        symbols: Symbols to test (default: BTCUSDT, ETHUSDT).
        intervals: Timeframes (default: 15m).
        year: Year of data to download.
        months: Months to download (default: 1-12).
        initial_equity: Starting equity per backtest.

    Returns:
        Aggregated results dict with measured EV.
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    if intervals is None:
        intervals = ["15m"]
    if months is None:
        months = list(range(1, 13))

    # Map symbol format: BTCUSDT -> BTC/USDT:USDT for the engine
    symbol_map = {
        "BTCUSDT": "BTC/USDT:USDT",
        "ETHUSDT": "ETH/USDT:USDT",
    }

    all_trades = []
    all_reports = []
    run_results = []

    print("=" * 70)
    print("EV VALIDATION: Real Binance Futures Data")
    print(f"Symbols: {symbols} | Intervals: {intervals} | Year: {year}")
    print("=" * 70)

    # Step 1: Download data
    print("\n--- Downloading Data ---")
    for symbol in symbols:
        for interval in intervals:
            print(f"\n{symbol} {interval}:")
            download_binance_klines(symbol, interval, year, months)

    # Step 2: Run backtests
    print("\n--- Running Backtests ---")
    for symbol in symbols:
        for interval in intervals:
            engine_symbol = symbol_map.get(symbol, symbol)
            print(f"\n{symbol} {interval}:")

            candles = load_historical_klines(symbol, interval)
            if candles.empty:
                print(f"  SKIPPED: No data available")
                continue

            trades, eq_curve, stats = run_backtest(
                candles=candles,
                symbol=engine_symbol,
                timeframe=interval,
                initial_equity=initial_equity,
            )

            report = generate_report(
                trades, eq_curve, initial_equity,
                stats["signals_generated"], stats["signals_taken"],
                stats["be_stop_triggers"],
            )

            all_trades.extend(trades)
            all_reports.append(report)

            r_multiples = [t.get("r_multiple", 0) for t in trades]
            avg_r = np.mean(r_multiples) if r_multiples else 0

            run_stat = {
                "symbol": symbol,
                "interval": interval,
                "trades": len(trades),
                "signals": stats["signals_generated"],
                "taken": stats["signals_taken"],
                "win_rate": report["win_rate"],
                "avg_r": avg_r,
                "profit_factor": report["profit_factor"],
                "max_dd_pct": report["max_drawdown_pct"],
                "return_pct": report["return_pct"],
                "net_pnl": report["net_pnl"],
            }
            run_results.append(run_stat)

            status = "+" if report["net_pnl"] > 0 else "-"
            print(
                f"  {len(trades):>3} trades | "
                f"WR={report['win_rate']:.0%} | "
                f"AvgR={avg_r:+.2f} | "
                f"PF={report['profit_factor']:.2f} | "
                f"DD={report['max_drawdown_pct']:.1%} | "
                f"{status}${abs(report['net_pnl']):.0f}"
            )

            # Print full report for each run
            print_report(report)

    # Step 3: Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (ALL SYMBOLS / INTERVALS)")
    print("=" * 70)

    total_trades = len(all_trades)
    if total_trades == 0:
        print("NO TRADES GENERATED.")
        print("Possible causes:")
        print("  - Data download failed (check network)")
        print("  - Strategy filters too restrictive for this data period")
        print("  - Insufficient swing patterns in data")
        print("=" * 70)
        return {"total_trades": 0, "ev": 0, "per_run": run_results}

    r_multiples = [t.get("r_multiple", 0) for t in all_trades]
    win_trades = [r for r in r_multiples if r > 0]
    loss_trades = [r for r in r_multiples if r <= 0]

    win_rate = len(win_trades) / total_trades
    avg_win_r = np.mean(win_trades) if win_trades else 0
    avg_loss_r = abs(np.mean(loss_trades)) if loss_trades else 0

    ev = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    avg_pf = np.mean([
        r["profit_factor"] for r in all_reports
        if r["profit_factor"] < float("inf")
    ]) if all_reports else 0
    avg_dd = np.mean([r["max_drawdown_pct"] for r in all_reports])
    avg_return = np.mean([r["return_pct"] for r in all_reports])
    median_r = np.median(r_multiples)

    # Exit reason breakdown
    reasons = {}
    for t in all_trades:
        reason = t.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Total Trades:        {total_trades}")
    print(f"Runs:                {len(run_results)}")
    print()
    print(f"Win Rate:            {win_rate:.1%}")
    print(f"Avg Win R:           {avg_win_r:+.2f}R")
    print(f"Avg Loss R:          {avg_loss_r:.2f}R")
    print(f"Median R:            {median_r:+.2f}R")
    print()
    print(f"*** EXPECTED VALUE:  {ev:+.3f}R ***")
    print()
    print(f"Profit Factor:       {avg_pf:.3f}")
    print(f"Avg Max Drawdown:    {avg_dd:.2%}")
    print(f"Avg Return:          {avg_return:.2%}")
    print()

    # Exit reason breakdown
    print("Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total_trades
        avg_r_for_reason = np.mean([
            t.get("r_multiple", 0) for t in all_trades if t.get("reason") == reason
        ])
        print(f"  {reason:<25} {count:>4} ({pct:>5.1%}) | AvgR={avg_r_for_reason:+.2f}")

    # R distribution
    print()
    print("R-Multiple Distribution:")
    r_arr = np.array(r_multiples)
    for threshold in [-2, -1, 0, 1, 2, 3, 5, 10]:
        if threshold < 0:
            count = np.sum(r_arr <= threshold)
            print(f"  R <= {threshold:>3}: {count:>4} ({count/total_trades:>5.1%})")
        else:
            count = np.sum(r_arr >= threshold)
            print(f"  R >= {threshold:>3}: {count:>4} ({count/total_trades:>5.1%})")

    print()
    print("=" * 70)
    if ev >= 2.0:
        print(f"TARGET MET: EV = {ev:.3f}R (target: 2-5R)")
    elif ev >= 1.0:
        print(f"PROGRESS: EV = {ev:.3f}R — getting closer to 2R target")
    elif ev > 0:
        print(f"POSITIVE EV: {ev:.3f}R — profitable but below 2R target")
    else:
        print(f"NEGATIVE EV: {ev:.3f}R — needs improvement")
    print("=" * 70)

    return {
        "total_trades": total_trades,
        "ev": round(ev, 4),
        "win_rate": round(win_rate, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "median_r": round(median_r, 4),
        "profit_factor": round(avg_pf, 4),
        "avg_max_dd": round(avg_dd, 4),
        "avg_return": round(avg_return, 4),
        "exit_reasons": reasons,
        "per_run": run_results,
    }


if __name__ == "__main__":
    results = run_validation()
