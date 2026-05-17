from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from edge_trader_bot.blf_verifier import BlfError, BlfVerifier
from edge_trader_bot.schemas import ForecastSignals, MarketView


class GoodBlfSummarizer:
    def __init__(self, probability: float = 0.70, confidence: str = "medium", uncertainty: float = 0.08) -> None:
        self.probability = probability
        self.confidence = confidence
        self.uncertainty = uncertainty

    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict:
        return {
            "current_probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "evidence_for_yes": ["Relevant evidence supports YES."],
            "evidence_for_no": [],
            "key_uncertainties": ["Need exact resolution confirmation."],
            "missing_information": [],
            "resolution_risks": [],
            "reasoning_summary": "Evidence nudges the belief state toward YES.",
            "risk_flags": [],
        }


class BrokenBlfSummarizer:
    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict:
        raise BlfError("invalid_json", "bad json", json_parse_error="line 1")


class NoDirectFutureBlfSummarizer:
    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict:
        return {
            "current_probability": 0.02,
            "confidence": "low",
            "uncertainty": 0.25,
            "evidence_for_yes": [],
            "evidence_for_no": [],
            "key_uncertainties": ["No direct announcement evidence yet."],
            "missing_information": ["No announcement yet, but the deadline is far away."],
            "resolution_risks": [],
            "reasoning_summary": "No direct evidence of an announcement yet.",
            "risk_flags": ["No direct evidence yet."],
        }


class CastingDeclineBlfSummarizer:
    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict:
        return {
            "current_probability": 0.05,
            "confidence": "medium",
            "uncertainty": 0.10,
            "evidence_for_yes": [],
            "evidence_for_no": ["Credible reporting says the actor declined and is not part of the cast."],
            "key_uncertainties": [],
            "missing_information": [],
            "resolution_risks": [],
            "reasoning_summary": "The actor declined and is not part of the show.",
            "risk_flags": ["credible_decline"],
        }


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


def make_signals(**overrides) -> ForecastSignals:
    base = {
        "market_id": "m1",
        "p_market": 0.41,
        "p_stat": 0.41,
        "p_2402": 0.62,
        "p_final": 0.45,
        "confidence": "medium",
        "uncertainty": 0.06,
        "evidence_quality": 4,
        "evidence_package": {
            "resolution_check": {"trade_blocker": False, "risk_flags": []},
            "evidence_for_yes": [{"summary": "Evidence supports yes."}],
            "evidence_for_no": [],
        },
    }
    base.update(overrides)
    return ForecastSignals(**base)


class BlfVerifierTests(unittest.TestCase):
    def test_disabled_returns_noop_package_not_attempted(self) -> None:
        market = make_market()
        signals = make_signals()

        updated = BlfVerifier(enabled=False).verify(market, signals)

        self.assertIsNotNone(updated.blf_package)
        self.assertFalse(updated.blf_package["enabled"])
        self.assertFalse(updated.blf_package["attempted"])
        self.assertIn("blf_disabled", updated.risk_flags)
        self.assertIsNone(updated.p_blf)

    def test_invalid_json_falls_back_to_rag_probability(self) -> None:
        market = make_market()
        signals = make_signals()

        updated = BlfVerifier(enabled=True, summarizer=BrokenBlfSummarizer()).verify(market, signals)

        self.assertIsNone(updated.p_blf)
        self.assertTrue(updated.blf_package["attempted"])
        self.assertFalse(updated.blf_package["succeeded"])
        self.assertEqual(updated.blf_package["fallback_reason"], "blf_invalid_json")
        self.assertEqual(updated.blf_package["p_blf_final_after_shrinkage"], signals.p_2402)

    def test_probability_clamping(self) -> None:
        market = make_market()
        signals = make_signals()

        updated = BlfVerifier(enabled=True, summarizer=GoodBlfSummarizer(probability=1.5)).verify(market, signals)

        self.assertEqual(updated.blf_package["p_blf_raw"], 0.98)
        self.assertLessEqual(updated.p_blf or 0, 0.98)

    def test_low_confidence_shrinks_toward_market(self) -> None:
        market = make_market(yes_mid=0.41)
        signals = make_signals(p_market=0.41, p_2402=0.62)

        updated = BlfVerifier(
            enabled=True,
            summarizer=GoodBlfSummarizer(probability=0.90, confidence="low", uncertainty=0.25),
        ).verify(market, signals)

        self.assertLess(updated.p_blf or 1, 0.50)

    def test_blf_rag_disagreement_shrinks_toward_market(self) -> None:
        market = make_market(yes_mid=0.41)
        signals = make_signals(p_market=0.41, p_2402=0.20)

        updated = BlfVerifier(
            enabled=True,
            summarizer=GoodBlfSummarizer(probability=0.90, confidence="high", uncertainty=0.05),
        ).verify(market, signals)

        self.assertLess(updated.p_blf or 1, 0.50)

    def test_resolution_trade_blocker_limits_blf_optimism(self) -> None:
        market = make_market(yes_mid=0.41)
        signals = make_signals(
            evidence_package={
                "resolution_check": {"trade_blocker": True, "risk_flags": ["official_source_not_confirmed"]},
                "evidence_for_yes": [],
                "evidence_for_no": [],
            }
        )

        updated = BlfVerifier(
            enabled=True,
            summarizer=GoodBlfSummarizer(probability=0.95, confidence="high", uncertainty=0.03),
        ).verify(market, signals)

        self.assertLess(updated.p_blf or 1, 0.50)

    def test_future_candidacy_absent_evidence_anchors_blf_near_market_prior(self) -> None:
        market = make_market(
            question="Who will run for the Republican presidential nomination in 2028?",
            description="Resolves YES if Marco Rubio announces a presidential campaign before 2028.",
            resolution_time=datetime.now(UTC) + timedelta(days=600),
            yes_mid=0.64,
            yes_bid=0.62,
            yes_ask=0.66,
        )
        signals = make_signals(p_market=0.64, p_2402=0.60)

        updated = BlfVerifier(enabled=True, summarizer=NoDirectFutureBlfSummarizer()).verify(market, signals)

        self.assertEqual(updated.blf_package["market_type"], "future_candidacy")
        self.assertTrue(updated.blf_package["absence_of_evidence_penalty_detected"])
        self.assertGreater(updated.blf_package["p_blf_raw"], 0.50)
        self.assertLess(abs((updated.p_blf or 0) - market.yes_mid), 0.08)

    def test_entertainment_casting_credible_decline_can_lower_probability(self) -> None:
        market = make_market(
            question="Will Deepika Padukone be in White Lotus season 4?",
            description="Resolves YES if she is officially credited as cast.",
            resolution_time=datetime.now(UTC) + timedelta(days=300),
            yes_mid=0.16,
            yes_bid=0.14,
            yes_ask=0.18,
        )
        signals = make_signals(p_market=0.16, p_2402=0.18)

        updated = BlfVerifier(enabled=True, summarizer=CastingDeclineBlfSummarizer()).verify(market, signals)

        self.assertEqual(updated.blf_package["market_type"], "entertainment_casting")
        self.assertFalse(updated.blf_package["absence_of_evidence_penalty_detected"])
        self.assertLess(updated.p_blf or 1, market.yes_mid)


if __name__ == "__main__":
    unittest.main()
