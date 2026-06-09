import unittest
from types import SimpleNamespace

from axanctum.scoring.regime_engine import RegimeEngine
from axanctum.scanner import _calc_short_regime_adjustment


def _fixture(short_setups, short_flags=None, **overrides):
    base = {
        "short_setups": short_setups,
        "short_flags": short_flags or [],
        "price_change_24h": 6.0,
        "d_vwap": 2.5,
        "delta_price_short": -0.25,
        "delta_cvd_spot": -1.2,
        "delta_cvd_fut": 0.0,
        "delta_oi": 1.5,
        "funding_rate": 0.0002,
        "vol_ratio": 1.0,
        "dist_score": 45.0,
        "div_score": 32.0,
        "short_deriv_state": "neutral",
        "short_phase_state": "neutral",
        "short_rejection_score": 35.0,
        "failed_breakout": False,
        "spot_absorption_risk": False,
    }
    base.update(overrides)
    return base


class ShortRegimeAdjustmentTest(unittest.TestCase):
    def test_regime_context_short_thresholds_are_not_overly_coarse(self):
        engine = RegimeEngine()

        engine.current_regime = "TRENDING"
        trending = engine.get_context(confidence=0.65)
        self.assertEqual(trending.short_threshold, engine.min_score + 6.0)

        engine.current_regime = "CHOP"
        chop = engine.get_context(confidence=0.65)
        self.assertEqual(chop.short_threshold, engine.min_score + 6.0)

        engine.current_regime = "EUPHORIC"
        euphoric = engine.get_context(confidence=0.65)
        self.assertEqual(euphoric.short_threshold, engine.min_score - 4.0)

        engine.current_regime = "RECOVERY"
        recovery = engine.get_context(confidence=0.65)
        self.assertEqual(recovery.short_threshold, engine.min_score + 10.0)

    def test_euphoric_boosts_top_short_and_penalizes_continuation(self):
        top_adj, top_reasons = _calc_short_regime_adjustment(
            _fixture(
                ["top_reversal_short", "distribution_short"],
                short_flags=["MICRO_REVERSAL_TRIGGER", "BUY_PRESSURE_ABSORBED"],
                spot_absorption_risk=True,
                dist_score=58.0,
                div_score=36.0,
                short_deriv_state="buy_pressure_absorbed",
                short_rejection_score=42.0,
            ),
            SimpleNamespace(regime="EUPHORIC"),
            pre_euphoric_guard=False,
        )
        cont_adj, cont_reasons = _calc_short_regime_adjustment(
            _fixture(
                ["bear_continuation_short", "breakdown_short"],
                short_flags=["BEAR_CONTINUATION_SHORT", "VWAP_BELOW"],
                price_change_24h=-9.0,
                d_vwap=-4.2,
                delta_price_short=-0.9,
                delta_cvd_spot=-1.5,
                delta_cvd_fut=-1.3,
                delta_oi=-1.0,
                funding_rate=0.0,
                dist_score=0.0,
                div_score=0.0,
                short_phase_state="post_drop_exhaustion",
                short_rejection_score=12.0,
            ),
            SimpleNamespace(regime="EUPHORIC"),
            pre_euphoric_guard=False,
        )

        self.assertGreater(top_adj, cont_adj)
        self.assertIn("euphoric_top_short", top_reasons)
        self.assertIn("euphoric_trend_short_penalty", cont_reasons)

    def test_trending_keeps_continuation_tight(self):
        top_adj, _ = _calc_short_regime_adjustment(
            _fixture(
                ["top_reversal_short", "exhaustion_after_pump_short"],
                short_flags=["MICRO_REVERSAL_TRIGGER"],
                spot_absorption_risk=True,
                dist_score=52.0,
                div_score=40.0,
            ),
            SimpleNamespace(regime="TRENDING"),
            pre_euphoric_guard=False,
        )
        cont_adj, cont_reasons = _calc_short_regime_adjustment(
            _fixture(
                ["bear_continuation_short", "breakdown_short"],
                short_flags=["BEAR_CONTINUATION_SHORT"],
                price_change_24h=-7.5,
                d_vwap=-4.0,
                delta_price_short=-0.8,
                short_phase_state="post_drop_exhaustion",
            ),
            SimpleNamespace(regime="TRENDING"),
            pre_euphoric_guard=False,
        )

        self.assertGreater(top_adj, cont_adj)
        self.assertIn("trending_continuation_penalty", cont_reasons)

    def test_recovery_penalizes_all_short_heavy_and_more_for_continuation(self):
        top_adj, top_reasons = _calc_short_regime_adjustment(
            _fixture(
                ["top_reversal_short", "distribution_short"],
                short_flags=["MICRO_REVERSAL_TRIGGER"],
                spot_absorption_risk=True,
                dist_score=60.0,
                div_score=48.0,
                short_rejection_score=58.0,
                failed_breakout=True,
            ),
            SimpleNamespace(regime="RECOVERY"),
            pre_euphoric_guard=False,
        )
        cont_adj, cont_reasons = _calc_short_regime_adjustment(
            _fixture(
                ["bear_continuation_short", "breakdown_short", "long_squeeze_short"],
                short_flags=["BEAR_CONTINUATION_SHORT", "VWAP_BELOW"],
                price_change_24h=-10.5,
                d_vwap=-5.1,
                delta_price_short=-1.0,
                short_phase_state="short_trap_risk",
            ),
            SimpleNamespace(regime="RECOVERY"),
            pre_euphoric_guard=False,
        )

        self.assertLessEqual(top_adj, 0.0)
        self.assertLess(cont_adj, top_adj)
        self.assertTrue(
            {
                "recovery_top_penalty",
                "recovery_confirmed_top_soft_penalty",
            } & set(top_reasons)
        )
        self.assertIn("recovery_continuation_penalty", cont_reasons)


if __name__ == "__main__":
    unittest.main()
