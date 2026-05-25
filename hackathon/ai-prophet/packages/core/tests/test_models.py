"""Tests for Pydantic models."""

from datetime import UTC, datetime

import pytest
from ai_prophet_core.models import (
    Market,
    Quote,
    SizeType,
    TradeAction,
    TradeIntent,
    TradeSide,
)


def test_market_model():
    """Test Market model."""
    market = Market(
        market_id="market_123",
        question="Will X happen?",
        description="Details...",
        resolution_time=datetime(2024, 2, 1, tzinfo=UTC),
        created_at=datetime(2024, 1, 15, tzinfo=UTC),
        source="polymarket",
        source_market_id="pm_456",
        metadata={"category": "sports"}
    )
    assert market.market_id == "market_123"
    assert market.metadata["category"] == "sports"


def test_quote_model():
    """Test Quote model."""
    quote = Quote(
        quote_id="quote_1",
        market_id="market_123",
        ts=datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
        ingested_at=datetime(2024, 1, 15, 14, 1, 0, tzinfo=UTC),
        best_bid=0.45,
        best_ask=0.47,
        bid_size=100.0,
        ask_size=150.0,
        volume_24h=5000.0
    )
    assert quote.best_bid < quote.best_ask

    # Test validation (ask must be >= bid)
    with pytest.raises(ValueError):
        Quote(
            quote_id="quote_2",
            market_id="market_123",
            ts=datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
            ingested_at=datetime(2024, 1, 15, 14, 1, 0, tzinfo=UTC),
            best_bid=0.50,
            best_ask=0.45,  # Invalid: less than bid
            bid_size=100.0,
            ask_size=150.0,
            volume_24h=5000.0
        )


def test_trade_intent_model():
    """TradeIntent accepts arbitrary tick_ts datetimes.

    Tick-boundary enforcement lives at the API layer (``normalize_tick``);
    the execution model takes whatever the caller supplies.
    """
    intent = TradeIntent(
        intent_id="intent_1",
        experiment_id="exp_1",
        participant_idx=0,
        tick_ts=datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
        market_id="market_123",
        action=TradeAction.BUY,
        side=TradeSide.YES,
        size_type=SizeType.NOTIONAL,
        size=100.0,
        submitted_at=datetime(2024, 1, 15, 14, 5, 0, tzinfo=UTC)
    )
    assert intent.action == TradeAction.BUY
    assert intent.size > 0

    # The model accepts off-boundary tick_ts too -- boundary enforcement
    # is the API layer's job, not the wire model's.
    off_boundary = TradeIntent(
        intent_id="intent_2",
        experiment_id="exp_1",
        participant_idx=0,
        tick_ts=datetime(2024, 1, 15, 14, 7, 0, tzinfo=UTC),
        market_id="market_123",
        action=TradeAction.BUY,
        side=TradeSide.YES,
        size_type=SizeType.NOTIONAL,
        size=100.0,
        submitted_at=datetime(2024, 1, 15, 14, 5, 0, tzinfo=UTC)
    )
    assert off_boundary.tick_ts.minute == 7

