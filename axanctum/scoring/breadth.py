from __future__ import annotations

from typing import Dict, List

from ..models import MarketBreadthContext


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total * 100.0, 1)


def calculate_market_breadth(
    results: List[Dict],
    long_threshold: float,
    short_threshold: float,
) -> MarketBreadthContext:
    """
    Hitung breadth market dari hasil scan batch yang sudah tersedia.

    Layer ini observability-only: tidak mengubah score, threshold, atau gate.
    """
    total = len(results)
    if total <= 0:
        return MarketBreadthContext(
            total=0,
            above_vwap_pct=0.0,
            below_vwap_pct=0.0,
            cvd_bull_pct=0.0,
            cvd_bear_pct=0.0,
            oi_hot_pct=0.0,
            oi_deleveraging_pct=0.0,
            funding_hot_pct=0.0,
            distribution_pct=0.0,
            divergence_pct=0.0,
            long_squeeze_pct=0.0,
            avg_long_score=0.0,
            avg_short_score=0.0,
            long_candidates=0,
            short_candidates=0,
        )

    above_vwap = sum(1 for r in results if r.get("d_vwap", 0.0) > 0)
    below_vwap = sum(1 for r in results if r.get("d_vwap", 0.0) < 0)

    cvd_bull = sum(
        1 for r in results
        if r.get("delta_cvd_spot", 0.0) > 0
        and r.get("delta_cvd_fut", 0.0) > 0
    )
    cvd_bear = sum(
        1 for r in results
        if r.get("delta_cvd_spot", 0.0) < 0
        and r.get("delta_cvd_fut", 0.0) < 0
    )

    oi_hot = sum(1 for r in results if r.get("delta_oi", 0.0) > 5.0)
    oi_deleveraging = sum(1 for r in results if r.get("delta_oi", 0.0) < -3.0)
    funding_hot = sum(1 for r in results if r.get("funding_rate", 0.0) > 0.0008)

    distribution = sum(1 for r in results if r.get("dist_score", 0.0) > 0)
    divergence = sum(1 for r in results if r.get("div_score", 0.0) > 0)
    long_squeeze = sum(
        1 for r in results
        if r.get("squeeze_type") in ("long", "long_exhausted")
    )

    long_scores = [float(r.get("score_regime_adj", r.get("score", 0.0))) for r in results]
    short_scores = [
        max(
            float(r.get("ls_score", 0.0)),
            float(r.get("dist_score", 0.0)),
            float(r.get("div_score_adj", r.get("div_score", 0.0))),
        )
        for r in results
    ]

    long_candidates = sum(1 for score in long_scores if score >= long_threshold)
    short_candidates = sum(1 for score in short_scores if score >= short_threshold)

    return MarketBreadthContext(
        total=total,
        above_vwap_pct=_pct(above_vwap, total),
        below_vwap_pct=_pct(below_vwap, total),
        cvd_bull_pct=_pct(cvd_bull, total),
        cvd_bear_pct=_pct(cvd_bear, total),
        oi_hot_pct=_pct(oi_hot, total),
        oi_deleveraging_pct=_pct(oi_deleveraging, total),
        funding_hot_pct=_pct(funding_hot, total),
        distribution_pct=_pct(distribution, total),
        divergence_pct=_pct(divergence, total),
        long_squeeze_pct=_pct(long_squeeze, total),
        avg_long_score=round(sum(long_scores) / total, 1),
        avg_short_score=round(sum(short_scores) / total, 1),
        long_candidates=long_candidates,
        short_candidates=short_candidates,
    )


def format_market_breadth_log(ctx: MarketBreadthContext) -> str:
    return (
        f"[Breadth] n={ctx.total} | "
        f"VWAP+={ctx.above_vwap_pct:.1f}% VWAP-={ctx.below_vwap_pct:.1f}% | "
        f"CVD bull={ctx.cvd_bull_pct:.1f}% bear={ctx.cvd_bear_pct:.1f}% | "
        f"OI hot={ctx.oi_hot_pct:.1f}% delever={ctx.oi_deleveraging_pct:.1f}% | "
        f"FR hot={ctx.funding_hot_pct:.1f}% | "
        f"dist={ctx.distribution_pct:.1f}% div={ctx.divergence_pct:.1f}% "
        f"LS={ctx.long_squeeze_pct:.1f}% | "
        f"avgL={ctx.avg_long_score:.1f} avgS={ctx.avg_short_score:.1f} | "
        f"candL={ctx.long_candidates} candS={ctx.short_candidates}"
    )
