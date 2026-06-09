import unittest
from types import SimpleNamespace

from axanctum.scanner import _short_gate_decision
from axanctum.scoring.short_engine import calc_integrated_short_score


class ShortReversalSensitivityTest(unittest.TestCase):
    def test_micro_top_reversal_beats_generic_continuation(self):
        score, flags, contexts, setups, deriv_state = calc_integrated_short_score(
            delta_price=0.6,
            delta_price_short=-0.35,
            price_change_24h=6.0,
            delta_cvd_spot=-1.2,
            delta_cvd_fut=0.2,
            delta_oi=2.0,
            funding_rate=0.00025,
            d_vwap=2.8,
            vol_ratio=1.1,
            squeeze_type="none",
            squeeze_stage="none",
            ls_score=0.0,
            ls_flags=[],
            ls_contexts=[],
            dist_score=44.0,
            dist_flags=["DISTRIBUTION"],
            dist_contexts=["distribution pressure"],
            div_score=36.0,
            div_flags=["BEARISH_DIVERGENCE_WATCH"],
            div_contexts=["divergence watch"],
            divergence_type="hidden_distribution",
            divergence_status="watch",
            price_structure_state="range_hold",
            broke_recent_swing_low=False,
            absorption_risk=True,
            divergence_age_candles=2,
            short_score_cap_reason="",
            short_rejection_score=42.0,
            failed_breakout=False,
            last_candle_bearish=True,
            last_close_position=0.52,
            upper_wick_pct=0.35,
            near_24h_high=True,
            spot_absorption_risk=True,
        )

        self.assertGreaterEqual(score, 55.0)
        self.assertIn("top_reversal_short", setups)
        self.assertIn("TOP_REVERSAL_SHORT", flags)
        self.assertIn("MICRO_REVERSAL_TRIGGER", flags)
        self.assertIn("BUY_PRESSURE_ABSORBED", flags)
        self.assertIn(deriv_state, {"neutral", "long_trap", "buy_pressure_absorbed"})
        self.assertTrue(any("top reversal" in ctx.lower() for ctx in contexts))

    def test_late_continuation_without_new_edge_is_watch(self):
        r = {
            "short_setups": ["bear_continuation_short"],
            "short_flags": ["BEAR_CONTINUATION_SHORT", "VWAP_BELOW", "CVD_CONFLUENCE_BEARISH"],
            "price_change_24h": -9.0,
            "delta_price": -2.5,
            "d_vwap": -4.5,
            "delta_price_short": -0.8,
            "delta_cvd_spot": -1.2,
            "delta_cvd_fut": -1.2,
            "delta_oi": -1.0,
            "funding_rate": 0.0,
            "vol_ratio": 1.0,
            "ls_score": 0.0,
            "dist_score": 0.0,
            "div_score": 0.0,
            "divergence_status": "none",
            "price_structure_state": "neutral",
            "broke_recent_swing_low": False,
            "absorption_risk": False,
            "divergence_age_candles": 0,
            "short_score_cap_reason": "",
            "short_deriv_state": "neutral",
            "short_phase_state": "neutral",
            "short_rejection_score": 0.0,
            "failed_breakout": False,
            "last_candle_bearish": True,
            "last_close_position": 0.40,
            "near_24h_high": False,
        }
        regime_ctx = SimpleNamespace(regime="TRENDING")
        breadth_ctx = SimpleNamespace(direction="BEARISH_CONFIRMED")

        status, reasons = _short_gate_decision(r, regime_ctx, breadth_ctx)

        self.assertEqual(status, "watch")
        self.assertIn("late_continuation_chase", reasons)


if __name__ == "__main__":
    unittest.main()
