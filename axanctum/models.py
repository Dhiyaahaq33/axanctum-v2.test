from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeContext:
    """Full context yang dikembalikan RegimeEngine ke scan pipeline."""
    regime:     str    # TRENDING / CHOP / EUPHORIC / PANIC / RECOVERY
    confidence: float  # 0.0 – 1.0

    # ── Signal threshold (adjusted dari MIN_SCORE default) ─────────────────────
    long_threshold:  float   # MIN_SCORE untuk long alerts
    short_threshold: float   # MIN_SCORE untuk short/div alerts

    # ── Score weight modifiers (multiplier terhadap weight default = 1.0) ───────
    w_cvd_spot:  float   # CVD spot weight
    w_cvd_fut:   float   # CVD futures weight
    w_oi:        float   # OI delta weight
    w_fr:        float   # Funding rate weight
    w_vwap:      float   # VWAP deviation weight

    # ── Execution profile ───────────────────────────────────────────────────────
    aggressiveness:          float  # 0.0–1.0, rekomendasi ukuran sinyal
    confirmation_bias:       float  # multiplier kebutuhan konfirmasi
    risk_profile:            str    # "normal" / "reduced" / "minimal" / "suspended"
    allowed_setups:          list   # setup apa yang diizinkan

    # ── Signal modifier ─────────────────────────────────────────────────────────
    exhaustion_sensitivity:  float  # multiplier div_score dan ls_score
    continuation_trust:      float  # multiplier long continuation score
    score_bonus_continuation: float # bonus tambahan untuk sinyal continuation
    score_penalty_counter:   float  # penalti untuk sinyal counter-trend


@dataclass
class MarketBreadthContext:
    """Ringkasan kondisi market-wide dari hasil scan batch."""
    total: int
    price24_up_pct: float
    price24_down_pct: float
    avg_price24_change: float
    above_vwap_pct: float
    below_vwap_pct: float
    cvd_bull_pct: float
    cvd_bear_pct: float
    oi_hot_pct: float
    oi_deleveraging_pct: float
    funding_hot_pct: float
    distribution_pct: float
    divergence_pct: float
    long_squeeze_pct: float
    avg_long_score: float
    avg_short_score: float
    long_candidates: int
    short_candidates: int
    direction: str
    strength: float
    long_mode: str
    short_mode: str
    confirmations: int
