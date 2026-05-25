from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.run_guarded_live_6h import (
    build_live_tick_command,
    determine_next_freeze_scope,
    guarded_live_env,
    parse_args,
    unsafe_trade_size,
)


class GuardedLiveScriptTests(unittest.TestCase):
    def test_guarded_live_env_enables_price_trading_and_guard_mode(self) -> None:
        args = SimpleNamespace(
            diagnostic_profile="live_guarded",
            max_markets=None,
            rag_budget=None,
            blf_budget=None,
            blf_max_steps=2,
            max_trade_size=1,
            max_position_per_market=3,
            max_total_open_positions=12,
            max_session_loss=100.0,
        )

        env = guarded_live_env(args)

        self.assertEqual(env["EDGE_TRADER_ENABLE_PRICE_TRADING"], "1")
        self.assertEqual(env["EDGE_TRADER_DRY_RUN"], "0")
        self.assertEqual(env["EDGE_TRADER_THRESHOLD_DEBUG"], "0")
        self.assertEqual(env["EDGE_TRADER_LIVE_GUARD_MODE"], "1")
        self.assertEqual(env["EDGE_TRADER_MAX_TRADE_SIZE"], "1")
        self.assertEqual(env["EDGE_TRADER_MAX_POSITION_PER_MARKET"], "3")
        self.assertEqual(env["EDGE_TRADER_MAX_TOTAL_OPEN_POSITIONS"], "12")
        self.assertEqual(env["EDGE_TRADER_ENABLE_CANDIDATE_RERANK"], "1")
        self.assertTrue(env["EDGE_TRADER_SLUG"].startswith("rag-blf-edge-trader-live-"))

    def test_live_tick_command_allows_live_submit_but_not_threshold_debug(self) -> None:
        command = build_live_tick_command("python", max_markets=100)

        self.assertIn("--allow-live-submit", command)
        self.assertNotIn("--dry-run", command)
        self.assertNotIn("--threshold-debug", command)

    def test_unsafe_trade_size_detects_large_decision(self) -> None:
        self.assertTrue(unsafe_trade_size({"decisions": [{"shares": 2}]}, max_trade_size=1))
        self.assertFalse(unsafe_trade_size({"decisions": [{"shares": 1}]}, max_trade_size=1))

    def test_parses_24_tick_live_argument(self) -> None:
        with patch("sys.argv", ["run_guarded_live_6h.py", "--ticks", "24"]):
            args = parse_args()

        self.assertEqual(args.ticks, 24)

    def test_freeze_scope_is_tick_local_and_can_reset(self) -> None:
        self.assertEqual(
            determine_next_freeze_scope(
                rate_limit_ratio=0.75,
                fallback_blocks=0,
                consecutive_high_rate_limit_ticks=1,
            ),
            ("forecast_dependent_entries_only", "high_rate_limit_ratio"),
        )
        self.assertEqual(
            determine_next_freeze_scope(
                rate_limit_ratio=0.0,
                fallback_blocks=0,
                consecutive_high_rate_limit_ticks=0,
            ),
            ("none", ""),
        )

    def test_repeated_high_rate_limits_trigger_all_entry_freeze(self) -> None:
        self.assertEqual(
            determine_next_freeze_scope(
                rate_limit_ratio=0.75,
                fallback_blocks=0,
                consecutive_high_rate_limit_ticks=3,
            ),
            ("all_entries", "consecutive_high_rate_limit_ticks"),
        )


if __name__ == "__main__":
    unittest.main()
