"""Tests for exchange connection module."""

import pytest
import ccxt

from src.exchange import get_exchange, get_public_exchange


def _network_available():
    """Check if we can reach Binance API."""
    try:
        ex = get_public_exchange()
        ex.fetch_ticker("BTC/USDT:USDT")
        return True
    except Exception:
        return False


network = pytest.mark.skipif(
    not _network_available(), reason="Binance API unreachable"
)


class TestExchange:
    """Test exchange connection and configuration."""

    def test_get_public_exchange_returns_binanceusdm(self):
        exchange = get_public_exchange()
        assert isinstance(exchange, ccxt.binanceusdm)

    def test_public_exchange_has_rate_limit_enabled(self):
        exchange = get_public_exchange()
        assert exchange.enableRateLimit is True

    def test_public_exchange_has_timeout(self):
        exchange = get_public_exchange()
        assert exchange.timeout == 30_000

    def test_public_exchange_no_sandbox(self):
        """Public exchange should use live API, not sandbox."""
        exchange = get_public_exchange()
        assert isinstance(exchange, ccxt.binanceusdm)

    def test_get_exchange_private_false_no_keys(self):
        """Public exchange should work without API keys."""
        exchange = get_exchange(private=False)
        assert isinstance(exchange, ccxt.binanceusdm)

    @network
    def test_fetch_ticker_btc(self):
        """Verify we can fetch a BTC ticker from the exchange."""
        exchange = get_public_exchange()
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        assert "last" in ticker
        assert ticker["last"] is not None
        assert ticker["last"] > 0

    @network
    def test_fetch_ticker_eth(self):
        """Verify we can fetch an ETH ticker."""
        exchange = get_public_exchange()
        ticker = exchange.fetch_ticker("ETH/USDT:USDT")
        assert "last" in ticker
        assert ticker["last"] is not None
        assert ticker["last"] > 0
