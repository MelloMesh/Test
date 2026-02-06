"""
Core backtesting engine.

Simulates the golden pocket + RSI divergence strategy on historical data
with realistic costs, risk management, and stop-to-breakeven logic.

Can be run as a module:
    python3 -m src.backtest.engine --symbol BTC/USDT:USDT --days 60 --optimize-tp
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.backtest.costs import calculate_r_multiple, cost_as_r, estimate_round_trip_cost
from src.backtest.report import generate_report, print_report, print_tp_optimization_table, save_equity_curve
from src.backtest.stop_manager import (
    calculate_trailing_stop,
    check_stop_hit,
    check_tp_hit,
    get_be_price,
    should_move_stop_to_be,
)
from src.backtest.tp_optimizer import config_to_tp_list, evaluate_config, get_tp_configs, recommend_best_config
from src.config import DEFAULT_LEVERAGE, DEFAULT_TP_CONFIG, STRATEGY_TIMEFRAMES
from src.indicators.fibonacci import calculate_fib_levels, detect_swings, get_latest_swing_pair, is_in_golden_pocket
from src.indicators.rsi import calculate_rsi, detect_divergence, detect_rsi_peaks
from src.indicators.volume import volume_confirms_divergence
from src.risk.portfolio import Portfolio, Position
from src.risk.position_sizer import calculate_position_size
from src.risk.risk_manager import RiskManager
from src.strategy.golden_pocket import analyze_timeframe, score_confluence
from src.strategy.signals import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(
    candles: pd.DataFrame,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    initial_equity: float = 10000.0,
    leverage: int = DEFAULT_LEVERAGE,
    tp_config: list[dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Run a backtest on historical candle data.

    The engine walks through candles sequentially, simulating:
    1. Signal generation (swing detection → fib → RSI divergence → golden pocket)
    2. Risk management checks
    3. Position sizing
    4. Entry execution with costs
    5. Stop loss monitoring (including move to breakeven)
    6. Take profit execution at fib levels
    7. Equity tracking

    Args:
        candles: Historical OHLCV DataFrame.
        symbol: Trading pair.
        timeframe: Candle timeframe.
        initial_equity: Starting equity.
        leverage: Leverage to use.
        tp_config: Custom TP allocation config.

    Returns:
        Tuple of (trade_results, equity_curve, stats_dict).
    """
    if tp_config is None:
        tp_config = DEFAULT_TP_CONFIG

    portfolio = Portfolio(initial_equity)
    risk_manager = RiskManager(paper_mode=True)

    trade_results: list[dict] = []
    equity_curve: list[dict] = []
    signals_generated = 0
    signals_taken = 0
    be_stop_triggers = 0

    # Track open positions in backtest
    open_positions: list[dict] = []  # {position, signal, entry_index, stop, tps_remaining}

    # We need a minimum warmup period for indicators
    warmup = 100
    if len(candles) <= warmup:
        logger.warning(f"Insufficient candles for backtest: {len(candles)} (need > {warmup})")
        return [], [], {"signals_generated": 0, "signals_taken": 0, "be_stop_triggers": 0}

    logger.info(
        f"Starting backtest: {symbol} {timeframe} | "
        f"{len(candles)} candles | equity=${initial_equity:.0f} | leverage={leverage}x"
    )

    for i in range(warmup, len(candles)):
        current_candle = candles.iloc[i]
        current_price = float(current_candle["close"])
        current_time = current_candle["timestamp"]

        # --- Manage open positions ---
        closed_indices = []
        for idx, pos_data in enumerate(open_positions):
            pos = pos_data["position"]
            signal = pos_data["signal"]
            stop = pos_data["stop"]
            entry_idx = pos_data["entry_index"]

            # Check stop loss (or trailing stop)
            if check_stop_hit(current_candle, signal.direction, stop):
                exit_price = stop  # Assume stop fills at stop price
                remaining_size = pos.size_usd * pos.remaining_size_pct
                risk_amount = signal.risk_pct * portfolio.equity
                raw_r = calculate_r_multiple(signal.direction, pos.entry_price, exit_price, signal.stop_loss)
                costs_r = cost_as_r(remaining_size, risk_amount)
                net_r = raw_r - costs_r
                pnl = net_r * risk_amount

                # Determine reason: trailing_stop if stop was moved above/below entry
                is_trailing = pos.stop_moved_to_be
                reason = "trailing_stop" if is_trailing else "stop_loss"

                trade_results.append({
                    "pnl": pnl,
                    "r_multiple": net_r,
                    "direction": signal.direction,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "stop_price": signal.stop_loss,
                    "reason": reason,
                    "timeframe": timeframe,
                    "entry_time": pos.opened_at,
                    "exit_time": current_time,
                    "cost_r": costs_r,
                    "risk_amount": risk_amount,
                })

                portfolio.close_position(pos, exit_price, reason, abs(pnl) if pnl < 0 else 0)
                closed_indices.append(idx)
                continue

            # Check take profits (tiered) — skip "trail" TPs (no fixed price)
            tps = pos_data["tps_remaining"]
            new_tps = []
            for tp in tps:
                if tp["level"] == "trail":
                    # Trail TPs have no fixed exit — kept for trailing stop exit
                    new_tps.append(tp)
                    continue
                if check_tp_hit(current_candle, signal.direction, tp["price"]):
                    partial_size = pos.size_usd * tp["pct"]
                    risk_amount = signal.risk_pct * portfolio.equity * tp["pct"]
                    raw_r = calculate_r_multiple(signal.direction, pos.entry_price, tp["price"], signal.stop_loss)
                    costs_r = cost_as_r(partial_size, risk_amount)
                    net_r = raw_r - costs_r
                    pnl = net_r * risk_amount

                    trade_results.append({
                        "pnl": pnl,
                        "r_multiple": net_r,
                        "direction": signal.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": tp["price"],
                        "stop_price": signal.stop_loss,
                        "reason": f"take_profit_{tp['level']}",
                        "timeframe": timeframe,
                        "entry_time": pos.opened_at,
                        "exit_time": current_time,
                        "cost_r": costs_r,
                        "risk_amount": risk_amount,
                    })

                    pos.partial_closes.append({"pct": tp["pct"], "price": tp["price"]})
                    portfolio.cash += pnl  # Add partial profit to cash
                else:
                    new_tps.append(tp)

            pos_data["tps_remaining"] = new_tps

            # Check if only trail TPs remain — if so, managed entirely by trailing stop
            non_trail_tps = [tp for tp in new_tps if tp["level"] != "trail"]
            trail_tps = [tp for tp in new_tps if tp["level"] == "trail"]

            # If no TPs left at all, close fully
            if not new_tps and pos.remaining_size_pct <= 0.01:
                if pos in portfolio.positions:
                    portfolio.positions.remove(pos)
                    portfolio.cash += pos.margin_used
                closed_indices.append(idx)
                continue

            # Check stop-to-breakeven (with buffer, not exact BE)
            if not pos.stop_moved_to_be:
                if should_move_stop_to_be(candles, i, signal.direction, entry_idx):
                    pos_data["stop"] = get_be_price(pos.entry_price, signal.direction)
                    pos.stop_moved_to_be = True
                    be_stop_triggers += 1
                    logger.debug(f"Stop moved to BE+buffer for {symbol} at index {i}")

            # ATR-based trailing stop (after reaching profit threshold)
            if pos.stop_moved_to_be:
                new_trail = calculate_trailing_stop(
                    candles, i, signal.direction,
                    pos.entry_price, signal.stop_loss, pos_data["stop"],
                )
                if new_trail != pos_data["stop"]:
                    pos_data["stop"] = new_trail
                    logger.debug(f"Trailing stop updated for {symbol} at index {i}: {new_trail:.2f}")

            # Update current price for unrealized PnL
            pos.current_price = current_price

        # Remove closed positions
        for idx in sorted(closed_indices, reverse=True):
            open_positions.pop(idx)

        # --- Generate signals (only if we can still open positions) ---
        if portfolio.open_position_count < 3:
            # Use a sliding window of candles for analysis
            window = candles.iloc[max(0, i - 200):i + 1].copy()
            if len(window) >= 50:
                signals = analyze_timeframe(window, symbol, timeframe, anticipatory=True)
                signals_generated += len(signals)

                for signal in signals:
                    # Only consider signals from the most recent candles
                    # (avoid re-triggering old signals)
                    if signal.timestamp < current_time - pd.Timedelta(minutes=30):
                        continue

                    # Override TP config
                    signal.take_profits = _apply_tp_config(signal.fib_levels, signal.direction, tp_config)

                    # Risk check
                    allowed, reason = risk_manager.can_open_position(signal, portfolio)
                    if not allowed:
                        logger.debug(f"Signal rejected: {reason}")
                        continue

                    # Position sizing
                    try:
                        sizing = calculate_position_size(
                            equity=portfolio.equity,
                            risk_pct=signal.risk_pct,
                            entry_price=signal.entry_price,
                            stop_loss_price=signal.stop_loss,
                            leverage=leverage,
                        )
                    except ValueError as e:
                        logger.debug(f"Position sizing failed: {e}")
                        continue

                    # Check if we can afford this position
                    if sizing["margin_required"] > portfolio.cash:
                        logger.debug("Insufficient cash for margin")
                        continue

                    # Open position (costs folded into R-multiple at exit)
                    position = Position(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        size_usd=sizing["size_usd"],
                        size_contracts=sizing["size_contracts"],
                        leverage=leverage,
                        stop_loss=signal.stop_loss,
                        take_profits=signal.take_profits,
                        margin_used=sizing["margin_required"],
                        opened_at=current_time,
                        signal_reason=signal.reason,
                        current_price=current_price,
                    )

                    portfolio.open_position(position)

                    open_positions.append({
                        "position": position,
                        "signal": signal,
                        "entry_index": i,
                        "stop": signal.stop_loss,
                        "tps_remaining": list(signal.take_profits),
                    })

                    signals_taken += 1
                    logger.info(f"BT ENTRY: {signal}")
                    break  # One entry per candle max

        # Record equity
        equity_curve.append({
            "timestamp": current_time,
            "equity": portfolio.equity,
        })

    # Close any remaining positions at last price
    last_price = float(candles["close"].iloc[-1])
    for pos_data in open_positions:
        pos = pos_data["position"]
        signal = pos_data["signal"]
        remaining = pos.remaining_size_pct
        if remaining > 0.01 and pos in portfolio.positions:
            remaining_size = pos.size_usd * remaining
            risk_amount = signal.risk_pct * initial_equity
            raw_r = calculate_r_multiple(signal.direction, pos.entry_price, last_price, signal.stop_loss)
            costs_r = cost_as_r(remaining_size, risk_amount)
            net_r = raw_r - costs_r
            pnl = net_r * risk_amount

            trade_results.append({
                "pnl": pnl,
                "r_multiple": net_r,
                "direction": signal.direction,
                "entry_price": pos.entry_price,
                "exit_price": last_price,
                "stop_price": signal.stop_loss,
                "reason": "backtest_end",
                "timeframe": timeframe,
                "entry_time": pos.opened_at,
                "exit_time": candles["timestamp"].iloc[-1],
                "cost_r": costs_r,
                "risk_amount": risk_amount,
            })
            portfolio.close_position(pos, last_price, "backtest_end", abs(pnl) if pnl < 0 else 0)

    stats = {
        "signals_generated": signals_generated,
        "signals_taken": signals_taken,
        "be_stop_triggers": be_stop_triggers,
    }

    logger.info(
        f"Backtest complete: {len(trade_results)} trades | "
        f"{signals_generated} signals generated | {signals_taken} taken | "
        f"{be_stop_triggers} BE stops"
    )

    return trade_results, equity_curve, stats


def run_tp_optimization(
    candles: pd.DataFrame,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    initial_equity: float = 10000.0,
    leverage: int = DEFAULT_LEVERAGE,
) -> list[dict]:
    """
    Run backtests with multiple TP configurations and compare results.

    Returns:
        List of config dicts with performance metrics.
    """
    configs = get_tp_configs()
    results = []

    logger.info(f"Running TP optimization with {len(configs)} configurations...")

    for i, config in enumerate(configs):
        tp_list = config_to_tp_list(config)
        trades, eq_curve, stats = run_backtest(
            candles=candles,
            symbol=symbol,
            timeframe=timeframe,
            initial_equity=initial_equity,
            leverage=leverage,
            tp_config=tp_list,
        )

        evaluated = evaluate_config(config, trades, eq_curve, initial_equity)
        results.append(evaluated)

        logger.info(
            f"Config {chr(65+i)}: {evaluated['total_trades']} trades | "
            f"WR={evaluated['win_rate']:.0%} | PF={evaluated['profit_factor']:.2f} | "
            f"MaxDD={evaluated['max_dd_pct']:.1%}"
        )

    return results


def _calculate_pnl(direction: str, entry: float, exit_price: float, size_usd: float) -> float:
    """Calculate dollar P&L for a position (before costs). Legacy helper."""
    if direction == "LONG":
        return (exit_price - entry) / entry * size_usd
    else:
        return (entry - exit_price) / entry * size_usd


def _apply_tp_config(
    fib_levels: dict,
    direction: str,
    tp_config: list[dict],
) -> list[dict]:
    """Apply TP config to fib levels to get actual TP prices.

    Supports "trail" level type — these have no fixed price and are
    exited via the trailing stop mechanism.
    """
    take_profits = []
    for tp in tp_config:
        level = tp["level"]
        if level == "trail":
            # Trailing portion: no fixed exit price, managed by trailing stop
            take_profits.append({
                "level": "trail",
                "price": None,
                "pct": tp["pct"],
            })
        elif level in fib_levels:
            take_profits.append({
                "level": level,
                "price": fib_levels[level],
                "pct": tp["pct"],
            })
    return take_profits


def main():
    """CLI entrypoint for backtesting."""
    parser = argparse.ArgumentParser(description="Run backtest for golden pocket strategy")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--timeframes", default="15m,30m", help="Comma-separated timeframes")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data")
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="Starting equity")
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE, help="Leverage")
    parser.add_argument("--optimize-tp", action="store_true", help="Run TP optimization")
    parser.add_argument("--from-db", action="store_true", help="Load data from local DB instead of fetching")

    args = parser.parse_args()
    timeframes = args.timeframes.split(",")

    # Load data
    if args.from_db:
        from src.data.storage import load_candles
        for tf in timeframes:
            candles = load_candles(args.symbol, tf, days=args.days)
            if candles.empty:
                print(f"No data in DB for {args.symbol} {tf}. Fetch first.")
                continue
            _run_and_report(candles, args, tf)
    else:
        from src.data.fetcher import fetch_candles_bulk
        for tf in timeframes:
            print(f"\nFetching {args.days} days of {tf} data for {args.symbol}...")
            candles = fetch_candles_bulk(args.symbol, tf, days=args.days)
            if candles.empty:
                print(f"Failed to fetch data for {args.symbol} {tf}")
                continue
            _run_and_report(candles, args, tf)


def _run_and_report(candles, args, tf):
    """Run backtest and print report for a single timeframe."""
    print(f"\n{'='*60}")
    print(f"Backtesting {args.symbol} {tf} ({len(candles)} candles)")
    print(f"{'='*60}")

    if args.optimize_tp:
        results = run_tp_optimization(
            candles, args.symbol, tf, args.initial_equity, args.leverage
        )
        print_tp_optimization_table(results)
        best = recommend_best_config(results)
        if best:
            print(f"\nRunning full backtest with recommended config...")
            tp_list = config_to_tp_list(best)
            trades, eq_curve, stats = run_backtest(
                candles, args.symbol, tf, args.initial_equity, args.leverage, tp_list
            )
            report = generate_report(
                trades, eq_curve, args.initial_equity,
                stats["signals_generated"], stats["signals_taken"], stats["be_stop_triggers"]
            )
            print_report(report)
            save_equity_curve(eq_curve, f"equity_curve_{tf}.csv")
    else:
        trades, eq_curve, stats = run_backtest(
            candles, args.symbol, tf, args.initial_equity, args.leverage
        )
        report = generate_report(
            trades, eq_curve, args.initial_equity,
            stats["signals_generated"], stats["signals_taken"], stats["be_stop_triggers"]
        )
        print_report(report)
        save_equity_curve(eq_curve, f"equity_curve_{tf}.csv")


if __name__ == "__main__":
    main()
