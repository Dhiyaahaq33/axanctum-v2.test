import unittest

from axanctum.scoring.long_engine import (
    calculate_market_score,
    classify_spot_accumulation_quality,
)


class SpotAccumulationQualityTest(unittest.TestCase):
    def test_clean_spot_accumulation_when_price_confirms(self):
        quality = classify_spot_accumulation_quality(
            spot_cvd_slope_short=2.4,
            price_slope_short=0.35,
            post_rally_context=False,
            price_reclaim_or_breakout=True,
        )

        self.assertEqual(quality["spot_state"], "clean_spot_accumulation")
        self.assertFalse(quality["spot_absorption_risk"])

        _, flags, _ = calculate_market_score(
            d_vwap=0.6,
            delta_cvd_spot=2.4,
            delta_cvd_fut=0.4,
            delta_oi=0.2,
            delta_price=0.8,
            funding_rate=0.0,
            spot_state=str(quality["spot_state"]),
            spot_accum_score_cap=quality["long_score_cap"],
        )
        self.assertIn("A", flags)
        self.assertNotIn("SPOT_BUY_ABSORPTION", flags)

    def test_spot_buy_absorption_caps_long_and_removes_spot_accum(self):
        quality = classify_spot_accumulation_quality(
            spot_cvd_slope_short=2.8,
            price_slope_short=-0.25,
            post_rally_context=True,
            price_reclaim_or_breakout=False,
        )

        self.assertEqual(quality["spot_state"], "spot_buy_absorption")
        self.assertTrue(quality["spot_absorption_risk"])
        self.assertEqual(quality["long_score_cap_reason"], "spot_buy_absorption_after_rally")

        score, flags, _ = calculate_market_score(
            d_vwap=1.5,
            delta_cvd_spot=2.8,
            delta_cvd_fut=0.3,
            delta_oi=-2.0,
            delta_price=-0.2,
            funding_rate=0.0006,
            squeeze_type="short",
            allow_short_squeeze_boost=False,
            spot_state=str(quality["spot_state"]),
            spot_accum_score_cap=quality["long_score_cap"],
        )
        self.assertLessEqual(score, 58.0)
        self.assertNotIn("A", flags)
        self.assertNotIn("C_SQUEEZE", flags)
        self.assertIn("SPOT_BUY_ABSORPTION", flags)
        self.assertIn("BEARISH_DIVERGENCE_WATCH", flags)
        self.assertIn("UPSIDE_EXHAUSTION_RISK", flags)


if __name__ == "__main__":
    unittest.main()
