from __future__ import annotations

from typing import Dict, List

from ..models import MarketBreadthContext


PRICE24_UP_PCT = 1.0
PRICE24_DOWN_PCT = -1.0


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total * 100.0, 1)


def _classify_breadth_direction(
    *,
    price24_up_pct: float,
    price24_down_pct: float,
    avg_price24_change: float,
    above_vwap_pct: float,
    below_vwap_pct: float,
    cvd_bull_pct: float,
    cvd_bear_pct: float,
    oi_deleveraging_pct: float,
    funding_hot_pct: float,
    avg_long_score: float,
    long_candidates: int,
    total: int,
) -> tuple[str, float, str, str, int]:
    """
    Breadth overlay aktif:
    - candle 24h market-wide adalah sinyal arah utama
    - VWAP/CVD/OI/FR/score hanya mengonfirmasi kekuatan tape
    """
    if total <= 0:
        return "NEUTRAL", 0.0, "normal", "normal", 0

    bearish_conf = 0
    bullish_conf = 0

    if price24_down_pct >= 60.0:
        bearish_conf += 2
    elif price24_down_pct >= 50.0:
        bearish_conf += 1

    if avg_price24_change <= -1.5:
        bearish_conf += 2
    elif avg_price24_change <= -0.5:
        bearish_conf += 1

    if below_vwap_pct >= 60.0:
        bearish_conf += 1
    if cvd_bear_pct >= 20.0 or cvd_bull_pct <= 15.0:
        bearish_conf += 1
    if oi_deleveraging_pct >= 25.0:
        bearish_conf += 1
    if avg_long_score < 12.0 and long_candidates <= max(1, total // 100):
        bearish_conf += 1

    if price24_up_pct >= 60.0:
        bullish_conf += 2
    elif price24_up_pct >= 50.0:
        bullish_conf += 1

    if avg_price24_change >= 1.5:
        bullish_conf += 2
    elif avg_price24_change >= 0.5:
        bullish_conf += 1

    if above_vwap_pct >= 60.0:
        bullish_conf += 1
    if cvd_bull_pct >= 20.0 and cvd_bear_pct <= 20.0:
        bullish_conf += 1
    if funding_hot_pct <= 12.0:
        bullish_conf += 1

    net = bearish_conf - bullish_conf
    confirmations = max(bearish_conf, bullish_conf)

    if bearish_conf >= 6 and price24_down_pct >= 65.0 and avg_price24_change <= -2.5:
        return "RISK_OFF", min(confirmations / 8.0, 1.0), "blocked", "aggressive", confirmations
    if net >= 3 and bearish_conf >= 5:
        return "BEARISH_CONFIRMED", min(confirmations / 7.0, 1.0), "restricted", "favored", confirmations
    if net >= 2 and bearish_conf >= 3:
        return "BEARISH_WEAK", min(confirmations / 6.0, 1.0), "cautious", "favored", confirmations
    if bullish_conf - bearish_conf >= 3 and bullish_conf >= 5:
        return "BULLISH_CONFIRMED", min(bullish_conf / 7.0, 1.0), "normal", "cautious", bullish_conf
    if bullish_conf - bearish_conf >= 2 and bullish_conf >= 3:
        return "BULLISH_WEAK", min(bullish_conf / 6.0, 1.0), "normal", "normal", bullish_conf

    return "NEUTRAL", min(confirmations / 6.0, 1.0), "normal", "normal", confirmations


def calculate_market_breadth(
    results: List[Dict],
    long_threshold: float,
    short_threshold: float,
) -> MarketBreadthContext:
    """
    Hitung breadth market dari hasil scan batch yang sudah tersedia.

    Context ini dipakai scanner sebagai overlay aktif untuk menyesuaikan
    threshold long/short dan setup gate berdasarkan arah market 24 jam.
    """
    total = len(results)
    if total <= 0:
        return MarketBreadthContext(
            total=0,
            price24_up_pct=0.0,
            price24_down_pct=0.0,
            avg_price24_change=0.0,
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
            direction="NEUTRAL",
            strength=0.0,
            long_mode="normal",
            short_mode="normal",
            confirmations=0,
        )

    price24_changes = [float(r.get("price_change_24h", 0.0)) for r in results]
    price24_up = sum(1 for chg in price24_changes if chg > PRICE24_UP_PCT)
    price24_down = sum(1 for chg in price24_changes if chg < PRICE24_DOWN_PCT)

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
            float(r.get("short_score_regime_adj", r.get("short_score", 0.0))),
            float(r.get("ls_score", 0.0)),
            float(r.get("dist_score", 0.0)),
            float(r.get("div_score_adj", r.get("div_score", 0.0))),
        )
        for r in results
    ]

    long_candidates = sum(1 for score in long_scores if score >= long_threshold)
    short_candidates = sum(1 for score in short_scores if score >= short_threshold)

    price24_up_pct = _pct(price24_up, total)
    price24_down_pct = _pct(price24_down, total)
    avg_price24_change = round(sum(price24_changes) / total, 1)
    above_vwap_pct = _pct(above_vwap, total)
    below_vwap_pct = _pct(below_vwap, total)
    cvd_bull_pct = _pct(cvd_bull, total)
    cvd_bear_pct = _pct(cvd_bear, total)
    oi_deleveraging_pct = _pct(oi_deleveraging, total)
    funding_hot_pct = _pct(funding_hot, total)
    avg_long_score = round(sum(long_scores) / total, 1)

    direction, strength, long_mode, short_mode, confirmations = _classify_breadth_direction(
        price24_up_pct=price24_up_pct,
        price24_down_pct=price24_down_pct,
        avg_price24_change=avg_price24_change,
        above_vwap_pct=above_vwap_pct,
        below_vwap_pct=below_vwap_pct,
        cvd_bull_pct=cvd_bull_pct,
        cvd_bear_pct=cvd_bear_pct,
        oi_deleveraging_pct=oi_deleveraging_pct,
        funding_hot_pct=funding_hot_pct,
        avg_long_score=avg_long_score,
        long_candidates=long_candidates,
        total=total,
    )

    return MarketBreadthContext(
        total=total,
        price24_up_pct=price24_up_pct,
        price24_down_pct=price24_down_pct,
        avg_price24_change=avg_price24_change,
        above_vwap_pct=above_vwap_pct,
        below_vwap_pct=below_vwap_pct,
        cvd_bull_pct=cvd_bull_pct,
        cvd_bear_pct=cvd_bear_pct,
        oi_hot_pct=_pct(oi_hot, total),
        oi_deleveraging_pct=oi_deleveraging_pct,
        funding_hot_pct=funding_hot_pct,
        distribution_pct=_pct(distribution, total),
        divergence_pct=_pct(divergence, total),
        long_squeeze_pct=_pct(long_squeeze, total),
        avg_long_score=avg_long_score,
        avg_short_score=round(sum(short_scores) / total, 1),
        long_candidates=long_candidates,
        short_candidates=short_candidates,
        direction=direction,
        strength=round(strength, 2),
        long_mode=long_mode,
        short_mode=short_mode,
        confirmations=confirmations,
    )


def format_market_breadth_log(ctx: MarketBreadthContext) -> str:
    return (
        f"[Breadth] n={ctx.total} | "
        f"24h↑={ctx.price24_up_pct:.1f}% ↓={ctx.price24_down_pct:.1f}% "
        f"avg24={ctx.avg_price24_change:+.1f}% | "
        f"VWAP+={ctx.above_vwap_pct:.1f}% VWAP-={ctx.below_vwap_pct:.1f}% | "
        f"CVD bull={ctx.cvd_bull_pct:.1f}% bear={ctx.cvd_bear_pct:.1f}% | "
        f"OI hot={ctx.oi_hot_pct:.1f}% delever={ctx.oi_deleveraging_pct:.1f}% | "
        f"FR hot={ctx.funding_hot_pct:.1f}% | "
        f"dist={ctx.distribution_pct:.1f}% div={ctx.divergence_pct:.1f}% "
        f"LS={ctx.long_squeeze_pct:.1f}% | "
        f"avgL={ctx.avg_long_score:.1f} avgS={ctx.avg_short_score:.1f} | "
        f"candL={ctx.long_candidates} candS={ctx.short_candidates} | "
        f"bias={ctx.direction} str={ctx.strength:.2f} "
        f"long={ctx.long_mode} short={ctx.short_mode}"
    )
