import unittest
from types import SimpleNamespace

from axanctum.scanner import (
    _build_no_signal_diagnostic_message,
    _no_signal_diagnostic_telegram_enabled,
)


class NoSignalDiagnosticTest(unittest.TestCase):
    def test_telegram_diagnostic_is_disabled_by_default(self):
        self.assertFalse(_no_signal_diagnostic_telegram_enabled())

    def test_message_summarizes_thresholds_gates_and_top_candidates(self):
        msg = _build_no_signal_diagnostic_message(
            cycle_label="Siklus Test",
            results=[
                {
                    "symbol": "LONGUSDT",
                    "score": 68.0,
                    "score_regime_adj": 76.5,
                    "long_setups": ["breakout"],
                    "short_score": 12.0,
                    "short_score_regime_adj": 12.0,
                },
                {
                    "symbol": "SHORTUSDT",
                    "score": 22.0,
                    "score_regime_adj": 22.0,
                    "short_score": 79.0,
                    "short_score_regime_adj": 81.5,
                    "short_setups": ["bear_continuation_short"],
                    "short_phase_state": "late",
                },
            ],
            skipped=3,
            errors=1,
            elapsed_sec=12.3,
            regime_ctx=SimpleNamespace(
                regime="TRENDING",
                confidence=0.72,
                risk_profile="normal",
            ),
            breadth_ctx=SimpleNamespace(
                direction="BEARISH_CONFIRMED",
                strength=0.64,
                long_mode="restricted",
                short_mode="favored",
            ),
            base_long_threshold=73.0,
            base_short_threshold=76.0,
            long_threshold=84.0,
            short_threshold=82.0,
            raw_long_count=1,
            allowed_long_count=0,
            final_long_count=0,
            raw_short_count=2,
            allowed_short_count=1,
            alert_short_count=0,
            watch_short_count=1,
            blocked_short_count=0,
            short_gate_reason_counts={
                "late_continuation_chase": 2,
                "weak_flow": 1,
            },
        )

        self.assertIn("No-signal diagnostic", msg)
        self.assertIn("Regime: <b>TRENDING</b>", msg)
        self.assertIn("Threshold: L 73.0→84.0 | S 76.0→82.0", msg)
        self.assertIn("Long gate: raw=1 allowed=0 final=0 blocked=1/0", msg)
        self.assertIn(
            "Short gate: raw=2 allowed=1 alert=0 watch=1 blocked=0 setup_blocked=1",
            msg,
        )
        self.assertIn("Top L: LONGUSDT 76.5/84.0", msg)
        self.assertIn("Top S: SHORTUSDT 81.5/82.0", msg)
        self.assertIn("late_continuation_chase=2", msg)


if __name__ == "__main__":
    unittest.main()
