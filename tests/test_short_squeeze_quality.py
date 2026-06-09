import unittest

from axanctum.scoring.long_engine import (
    calculate_market_score,
    classify_short_squeeze_quality,
)


class ShortSqueezeQualityTest(unittest.TestCase):
    def test_downtrend_rebound_allows_short_squeeze_boost(self):
        quality = classify_short_squeeze_quality(
            squeeze_type="short",
            squeeze_fuel=58.0,
            funding_rate=-0.00045,
            delta_oi=-3.5,
            delta_price=1.2,
            price_change_24h=-6.5,
            delta_price_short=0.45,
            price_reclaim_or_breakout=True,
            post_rally_context=False,
        )

        self.assertTrue(quality["downtrend_context"])
        self.assertTrue(quality["allow_short_squeeze_boost"])
        self.assertEqual(quality["short_squeeze_veto_reasons"], [])

        _, flags, _ = calculate_market_score(
            d_vwap=-0.8,
            delta_cvd_spot=2.4,
            delta_cvd_fut=0.8,
            delta_oi=-3.5,
            delta_price=1.2,
            funding_rate=-0.00045,
            squeeze_type="short",
            allow_short_squeeze_boost=bool(quality["allow_short_squeeze_boost"]),
        )
        self.assertIn("C_SQUEEZE", flags)

    def test_post_rally_without_reclaim_keeps_squeeze_as_watch(self):
        quality = classify_short_squeeze_quality(
            squeeze_type="short",
            squeeze_fuel=52.0,
            funding_rate=0.00062,
            delta_oi=-2.2,
            delta_price=1.4,
            price_change_24h=8.0,
            delta_price_short=-0.15,
            price_reclaim_or_breakout=False,
            post_rally_context=True,
        )

        self.assertFalse(quality["downtrend_context"])
        self.assertFalse(quality["allow_short_squeeze_boost"])
        self.assertIn("positive_funding_not_short_squeeze_fuel", quality["short_squeeze_veto_reasons"])
        self.assertIn("no_price_reclaim_or_breakout", quality["short_squeeze_veto_reasons"])

        _, flags, _ = calculate_market_score(
            d_vwap=1.5,
            delta_cvd_spot=2.4,
            delta_cvd_fut=0.8,
            delta_oi=-2.2,
            delta_price=1.4,
            funding_rate=0.00062,
            squeeze_type="short",
            allow_short_squeeze_boost=bool(quality["allow_short_squeeze_boost"]),
        )
        self.assertNotIn("C_SQUEEZE", flags)


if __name__ == "__main__":
    unittest.main()
