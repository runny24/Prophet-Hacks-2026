from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from ai_prophet_core.client_models import PortfolioResponse, PositionData

from edge_trader_bot.aggregator import combine_signals
from edge_trader_bot.config import BotConfig
from edge_trader_bot.market_filter import deterministic_filter
from edge_trader_bot.schemas import ForecastSignals, MarketView
from edge_trader_bot.sizer import size_trade
from edge_trader_bot.stat_priors import initial_signals
from edge_trader_bot.trading_policy import decide_trade


def make_market(**overrides) -> MarketView:
    base = {
        "market_id": "m1",
        "question": "Will ACME close above $100?",
        "description": "Resolves YES if ACME closes above $100 on the resolution date.",
        "resolution_time": datetime.now(UTC) + timedelta(days=5),
        "yes_bid": 0.40,
        "yes_ask": 0.42,
        "yes_mid": 0.41,
        "no_bid": 0.58,
        "no_ask": 0.60,
        "no_mid": 0.59,
        "spread": 0.02,
        "volume_24h": 1000.0,
    }
    base.update(overrides)
    return MarketView(**base)


def make_portfolio(positions: list[PositionData] | None = None, cash: str = "10000") -> PortfolioResponse:
    return PortfolioResponse(
        experiment_id="exp",
        participant_idx=0,
        cash=cash,
        equity=cash,
        total_pnl="0",
        positions=positions or [],
        total_fills=0,
    )


class StrategyCoreTests(unittest.TestCase):
    def test_filter_rejects_wide_spread(self) -> None:
        market = make_market(yes_bid=0.30, yes_ask=0.60, yes_mid=0.45, spread=0.30)

        ok, reasons = deterministic_filter(market, None, BotConfig())

        self.assertFalse(ok)
        self.assertIn("spread_too_wide", reasons)

    def test_filter_rejects_new_position_when_open_position_budget_low(self) -> None:
        positions = [
            PositionData(
                market_id=f"held-{i}",
                side="YES",
                shares="1",
                avg_entry_price="0.5",
                current_price="0.5",
                updated_at=datetime.now(UTC),
            )
            for i in range(25)
        ]

        ok, reasons = deterministic_filter(make_market(), make_portfolio(positions), BotConfig())

        self.assertFalse(ok)
        self.assertIn("open_position_budget_low", reasons)

    def test_initial_signals_anchor_to_market_mid(self) -> None:
        market = make_market(yes_bid=0.30, yes_ask=0.34, yes_mid=0.32)

        signals = initial_signals(market)

        self.assertAlmostEqual(signals.p_market, 0.32)
        self.assertAlmostEqual(signals.p_stat, 0.32)
        self.assertEqual(signals.confidence, "low")

    def test_combination_stays_shrunk_toward_market(self) -> None:
        signals = ForecastSignals(
            market_id="m1",
            p_market=0.40,
            p_stat=0.40,
            p_2402=0.85,
            confidence="medium",
            uncertainty=0.06,
        )

        combined = combine_signals(signals)

        self.assertGreater(combined.p_final or 0, 0.40)
        self.assertLess(combined.p_final or 1, 0.60)
        self.assertIsNotNone(combined.p_final_before_blf)

    def test_blf_rag_disagreement_keeps_aggregation_conservative(self) -> None:
        signals = ForecastSignals(
            market_id="m1",
            p_market=0.40,
            p_stat=0.40,
            p_2402=0.80,
            p_blf=0.20,
            confidence="medium",
            uncertainty=0.06,
            evidence_package={"resolution_check": {"trade_blocker": False, "risk_flags": []}},
        )

        combined = combine_signals(signals)

        self.assertLess(abs((combined.p_final or 0.40) - 0.40), 0.05)
        self.assertLess(abs(combined.blf_adjustment or 0), 0.05)
        self.assertIn("shrunken verifier", combined.aggregation_reason)

    def test_sizing_respects_low_confidence_threshold_and_cap(self) -> None:
        config = BotConfig(max_new_notional_per_trade=100)

        self.assertEqual(size_trade(0.07, 0.5, "low", config), 0)
        high_conf_size = size_trade(0.25, 0.5, "high", config)
        self.assertLessEqual(high_conf_size * 0.5, 100)

    def test_policy_holds_when_edge_below_threshold(self) -> None:
        market = make_market()
        signals = ForecastSignals(
            market_id=market.market_id,
            p_market=market.yes_mid,
            p_stat=market.yes_mid,
            p_final=0.45,
            confidence="low",
            uncertainty=0.08,
        )

        decision = decide_trade(market, signals, None, BotConfig())

        self.assertIsNone(decision)

    def test_policy_buys_when_high_confidence_edge_clears(self) -> None:
        market = make_market()
        signals = ForecastSignals(
            market_id=market.market_id,
            p_market=market.yes_mid,
            p_stat=market.yes_mid,
            p_final=0.56,
            confidence="high",
            uncertainty=0.03,
        )

        decision = decide_trade(market, signals, None, BotConfig())

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "BUY")
        self.assertEqual(decision.side, "YES")

    def test_policy_holds_on_resolution_trade_blocker(self) -> None:
        market = make_market()
        signals = ForecastSignals(
            market_id=market.market_id,
            p_market=market.yes_mid,
            p_stat=market.yes_mid,
            p_final=0.80,
            confidence="high",
            uncertainty=0.02,
            evidence_package={
                "resolution_check": {
                    "trade_blocker": True,
                    "official_source_required": True,
                    "official_source_found": False,
                }
            },
        )

        decision = decide_trade(market, signals, None, BotConfig())

        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
