from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import aiohttp

from .config import CONFIG, OI_PERIOD_MAP, TF_MINUTES
from .fetchers import (
    fetch_funding_rate,
    fetch_futures_klines,
    fetch_oi_history,
    fetch_spot_klines,
)
from .indicators import (
    calc_absorption,
    calc_cvd_and_momentum,
    calc_cvd_volume_consistency,
    calc_oi_momentum,
    calc_price_momentum,
    calc_price_targets,
    calc_rolling_vwap,
    calc_squeeze_stage,
    calc_volume_anomaly,
)
from .logging_setup import log
from .notifier import (
    _is_on_cooldown,
    _mark_sent,
    build_telegram_message,
    get_grade,
    send_signal_to_dashboard,
    send_telegram,
)
from .scoring.long_engine import calculate_market_score
from .scoring.breadth import calculate_market_breadth, format_market_breadth_log
from .scoring.regime_engine import get_regime_engine
from .scoring.short_engine import (
    calc_bearish_divergence_score,
    calc_distribution_score,
    calc_integrated_short_score,
    calc_long_squeeze_score,
)
from .state import _FR_HISTORY_4H, _FR_VELOCITY_INTERVAL


def _calc_short_execution_state(fut_df, candles_24h: int) -> Dict:
    """
    Proxy trigger eksekusi SHORT dari candle yang sudah tersedia.

    Ini bukan penentu arah utama. Fungsinya membedakan area resistance
    yang baru "rawan" dari resistance yang mulai gagal/reject.
    """
    state = {
        "short_rejection_score": 0.0,
        "failed_breakout": False,
        "last_candle_bearish": False,
        "last_close_position": 0.5,
        "upper_wick_pct": 0.0,
        "near_24h_high": False,
    }

    if fut_df is None or len(fut_df) < 3:
        return state

    last = fut_df.iloc[-1]
    open_ = float(last["open"])
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    if close <= 0 or high <= low:
        return state

    range_abs = high - low
    range_pct = range_abs / close * 100.0
    upper_wick_pct = max(0.0, high - max(open_, close)) / close * 100.0
    lower_wick_pct = max(0.0, min(open_, close) - low) / close * 100.0
    close_position = (close - low) / range_abs if range_abs > 0 else 0.5
    last_candle_bearish = close < open_

    lookback = max(3, min(candles_24h, len(fut_df) - 1))
    prev_tail = fut_df.iloc[-lookback - 1:-1]
    prev_high = float(prev_tail["high"].max()) if len(prev_tail) else high
    near_24h_high = bool(prev_high > 0 and (high >= prev_high * 0.995 or close >= prev_high * 0.99))
    failed_breakout = bool(prev_high > 0 and high > prev_high * 1.001 and close < prev_high * 0.998)

    rejection_score = 0.0
    if last_candle_bearish:
        rejection_score += 18.0
    if close_position <= 0.42:
        rejection_score += 18.0
    elif close_position <= 0.58:
        rejection_score += 9.0
    if range_pct > 0:
        upper_ratio = upper_wick_pct / range_pct
        if upper_wick_pct >= 0.25 and upper_ratio >= 0.30:
            rejection_score += min(upper_ratio * 35.0, 25.0)
    if failed_breakout:
        rejection_score += 30.0
    elif near_24h_high and upper_wick_pct >= 0.35 and close_position <= 0.60:
        rejection_score += 14.0
    if range_pct >= 0.5 and upper_wick_pct > lower_wick_pct:
        rejection_score += 8.0

    state.update({
        "short_rejection_score": round(min(rejection_score, 100.0), 1),
        "failed_breakout": failed_breakout,
        "last_candle_bearish": last_candle_bearish,
        "last_close_position": round(close_position, 3),
        "upper_wick_pct": round(upper_wick_pct, 3),
        "near_24h_high": near_24h_high,
    })
    return state


def _classify_long_setups(r: Dict) -> List[str]:
    """
    Klasifikasikan setup LONG aktual dari hasil scan.

    Regime allowed_setups hanya berarti aman kalau signal-nya benar-benar
    cocok dengan setup tersebut. Ini mencegah CHOP meloloskan continuation
    biasa hanya karena score tinggi.
    """
    flags = set(r.get("flags", []))
    setups = set()

    d_vwap = float(r.get("d_vwap", 0.0))
    delta_price = float(r.get("delta_price", 0.0))
    vol_ratio = float(r.get("vol_ratio", 1.0))

    has_accum = bool(flags & {"A", "A_FUTURES", "CONFLUENCE"})
    has_clean_risk = not bool(flags & {"B_SPECULATIVE", "D_EXHAUSTION", "E_DISTRIBUTION", "ABSORPTION"})

    if has_accum and has_clean_risk:
        setups.add("continuation")

        if d_vwap < 0:
            setups.add("pullback")
            setups.add("accumulation_long")

        if (
            delta_price > 0
            and 0 <= d_vwap <= CONFIG["VWAP_OVEREXT_PCT"]
            and vol_ratio >= 1.5
        ):
            setups.add("breakout")

        if d_vwap <= -2.5 and vol_ratio >= 1.0:
            setups.add("extreme_reversal")

        if (
            "CONFLUENCE" in flags
            and vol_ratio >= 2.0
            and delta_price > 0
            and d_vwap <= CONFIG["VWAP_OVEREXT_PCT"]
        ):
            setups.add("momentum_ignition")

    return sorted(setups)


def _is_long_allowed_by_regime(r: Dict, allowed_setups: List[str]) -> bool:
    setups = _classify_long_setups(r)
    r["long_setups"] = setups
    return bool(set(setups) & set(allowed_setups))


def _has_bearish_breadth_recovery_exception(r: Dict) -> bool:
    """
    Recovery long di breadth bearish tetap boleh, tapi hanya sebagai
    reversal/momentum setup dengan bukti microstructure kuat.
    """
    setups = set(r.get("long_setups") or _classify_long_setups(r))
    if not setups & {"extreme_reversal", "momentum_ignition"}:
        return False

    flags = set(r.get("flags", []))
    d_vwap = float(r.get("d_vwap", 0.0))
    price24 = float(r.get("price_change_24h", 0.0))
    delta_cvd_spot = float(r.get("delta_cvd_spot", 0.0))
    delta_cvd_fut = float(r.get("delta_cvd_fut", 0.0))
    squeeze_fuel = float(r.get("squeeze_fuel", 0.0))
    squeeze_type = r.get("squeeze_type", "none")
    vol_ratio = float(r.get("vol_ratio", 1.0))

    cvd_recovery = (
        delta_cvd_spot > 1.0
        or (
            delta_cvd_spot >= 0.0
            and delta_cvd_fut > 2.0
            and "A_FUTURES" in flags
        )
        or "CONFLUENCE" in flags
    )
    squeeze_reversal = (
        squeeze_type in ("short", "short_exhausted", "none")
        and ("C_SQUEEZE" in flags or squeeze_fuel >= 45.0)
    )
    clean_location = d_vwap <= 2.5 and price24 <= 6.0
    volume_ok = vol_ratio >= 0.8

    return bool(cvd_recovery and squeeze_reversal and clean_location and volume_ok)


def _short_gate_decision(r: Dict, regime_ctx, breadth_ctx) -> tuple[str, List[str]]:
    """
    Quality gate akhir untuk SHORT alert.

    short_score tetap boleh tinggi sebagai detector, tapi Telegram hanya kirim
    kalau setup punya edge mandiri: flow, derivatif, freshness, atau top signal.
    """
    reasons: List[str] = []
    setups = set(r.get("short_setups", []))
    flags = set(r.get("short_flags", []))

    price24 = float(r.get("price_change_24h", 0.0))
    delta_price = float(r.get("delta_price", 0.0))
    d_vwap = float(r.get("d_vwap", 0.0))
    delta_price_short = float(r.get("delta_price_short", 0.0))
    delta_cvd_spot = float(r.get("delta_cvd_spot", 0.0))
    delta_cvd_fut = float(r.get("delta_cvd_fut", 0.0))
    delta_oi = float(r.get("delta_oi", 0.0))
    funding_rate = float(r.get("funding_rate", 0.0))
    vol_ratio = float(r.get("vol_ratio", 1.0))
    ls_score = float(r.get("ls_score", 0.0))
    dist_score = float(r.get("dist_score", 0.0))
    div_score = float(r.get("div_score", 0.0))
    short_deriv_state = r.get("short_deriv_state", "neutral")
    short_rejection_score = float(r.get("short_rejection_score", 0.0))
    failed_breakout = bool(r.get("failed_breakout", False))
    last_candle_bearish = bool(r.get("last_candle_bearish", False))
    last_close_position = float(r.get("last_close_position", 0.5))
    near_24h_high = bool(r.get("near_24h_high", False))

    top_setups = {
        "top_reversal_short",
        "exhaustion_after_pump_short",
        "distribution_short",
        "bearish_divergence_short",
    }
    trend_setups = {
        "bear_continuation_short",
        "breakdown_short",
        "long_squeeze_short",
    }

    both_bearish = delta_cvd_spot < -1.0 and delta_cvd_fut < -1.0
    strong_bear_flow = (
        delta_cvd_spot <= -2.5 and delta_cvd_fut <= -1.5
    ) or (delta_cvd_spot + delta_cvd_fut <= -6.0)
    weak_bear_flow = delta_cvd_spot < -1.0 or delta_cvd_fut < -1.0
    bullish_conflict = (
        delta_cvd_spot > 2.0 and delta_cvd_fut > 1.5
    ) or (
        delta_cvd_spot > 4.0 and delta_cvd_fut >= 0.0
    )

    funding_trapped_long = funding_rate > 0.0001 and delta_price <= 0
    oi_hot_breakdown = delta_oi > 4.0 and (
        "breakdown_short" in setups or delta_price_short <= -0.8
    )
    long_squeeze_active = "long_squeeze_short" in setups and ls_score >= 55.0
    deriv_support = bool(flags & {
        "SHORT_DERIV_LONG_TRAP",
        "SHORT_DERIV_BUY_PRESSURE_ABSORBED",
        "SHORT_DERIV_BEAR_CONTINUATION_FRESH",
        "SHORT_DERIV_LONG_SQUEEZE_FUEL",
    })
    deriv_edge = (
        strong_bear_flow
        or long_squeeze_active
        or funding_trapped_long
        or oi_hot_breakdown
        or dist_score >= 72.0
        or div_score >= 72.0
        or deriv_support
    )

    late_entry = price24 <= -14.0 and d_vwap <= -7.0
    extreme_late = price24 <= -20.0 or d_vwap <= -11.0
    exhausted_deriv = (
        delta_oi <= -8.0
        and funding_rate <= 0.0
        and not funding_trapped_long
    )
    active_late_edge = (
        long_squeeze_active
        or funding_trapped_long
        or oi_hot_breakdown
        or (strong_bear_flow and delta_oi > -6.0)
    )

    top_context = price24 >= 3.0 or delta_price >= 1.2 or d_vwap >= 3.0
    rejection_confirmed = (
        short_rejection_score >= 55.0
        or failed_breakout
        or (
            last_candle_bearish
            and last_close_position <= 0.55
            and delta_price_short <= -0.35
            and weak_bear_flow
        )
    )
    breakout_risk = (
        top_context
        and d_vwap >= 1.5
        and delta_price_short >= 0.4
        and delta_cvd_fut >= 1.0
        and not rejection_confirmed
    ) or (
        last_close_position >= 0.70
        and delta_price_short > 0
        and not last_candle_bearish
        and not rejection_confirmed
    )
    top_flow = (
        delta_cvd_spot < -1.0 and delta_cvd_fut < 1.0
    ) or div_score >= 50.0 or dist_score >= 55.0
    top_deriv = (
        funding_rate > 0.0005
        or delta_oi > 4.0
        or dist_score >= 70.0
        or div_score >= 70.0
        or deriv_support
    )
    top_valid = bool(
        setups & top_setups
        and top_context
        and rejection_confirmed
        and top_flow
        and (top_deriv or strong_bear_flow)
        and vol_ratio >= 0.7
    )

    trend_context = breadth_ctx.direction in ("BEARISH_CONFIRMED", "RISK_OFF", "BEARISH_WEAK")
    breakdown_fresh = (
        "breakdown_short" in setups
        and -8.0 <= price24 <= 2.0
        and -5.0 <= d_vwap <= -0.4
        and delta_price_short <= -0.8
    )
    bear_continuation_valid = bool(
        setups & trend_setups
        and trend_context
        and deriv_edge
        and weak_bear_flow
        and vol_ratio >= 0.7
    )

    # Distribution dari area bawah bukan top alert; dia hanya memperkuat continuation.
    deep_distribution = (
        "distribution_short" in setups
        and price24 <= -10.0
        and d_vwap <= -5.0
        and not top_valid
    )

    if bullish_conflict:
        return "blocked", ["bullish_cvd_conflict"]

    if "SHORT_DERIV_BREAKOUT_FUEL" in flags:
        return "watch", ["deriv_breakout_fuel"]

    if "SHORT_DERIV_LATE_DELEVERAGING" in flags:
        return "watch", ["deriv_late_deleveraging"]

    if "SHORT_DERIV_SELL_PRESSURE_ABSORBED" in flags:
        return "watch", ["deriv_sell_pressure_absorbed"]

    if breakout_risk and setups & top_setups:
        return "watch", ["breakout_risk_no_rejection"]

    if regime_ctx.regime == "PANIC":
        return "blocked", ["panic_suspended"]

    if deep_distribution:
        reasons.append("distribution_not_top")

    if late_entry and not active_late_edge:
        reasons.append("late_entry")
        if extreme_late or exhausted_deriv:
            return "watch", reasons + ["exhausted_deriv"]
        return "watch", reasons

    if extreme_late and not active_late_edge:
        return "watch", ["extreme_late"]

    if regime_ctx.regime == "RECOVERY":
        if top_valid and (dist_score >= 75.0 or div_score >= 75.0):
            return "alert", ["top_reversal_valid", "recovery_exception"]
        if long_squeeze_active and funding_trapped_long and strong_bear_flow:
            return "alert", ["long_squeeze_valid", "recovery_exception"]
        return "watch", ["regime_recovery_strict"]

    if regime_ctx.regime == "CHOP":
        if top_valid:
            return "alert", ["top_reversal_valid"]
        if breakdown_fresh and deriv_edge and weak_bear_flow:
            return "alert", ["breakdown_fresh"]
        return "watch", ["chop_needs_breakdown_or_top"]

    if top_valid:
        reason = "failed_breakout" if failed_breakout else "top_reversal_valid"
        return "alert", [reason]

    if "bear_continuation_short" in setups and not deriv_edge:
        return "watch", ["no_deriv_edge"]

    if "bear_continuation_short" in setups and not both_bearish and not long_squeeze_active:
        return "watch", ["weak_flow"]

    if bear_continuation_valid:
        if near_24h_high and not both_bearish and not long_squeeze_active:
            return "watch", ["near_high_needs_stronger_flow"]
        if late_entry:
            return "alert", ["bear_continuation_valid", "late_deriv_exception"]
        return "alert", ["bear_continuation_valid"]

    if breakdown_fresh and deriv_edge:
        return "alert", ["breakdown_fresh"]

    if long_squeeze_active and (weak_bear_flow or funding_trapped_long):
        return "alert", ["long_squeeze_valid"]

    if setups & top_setups:
        return "watch", ["top_setup_unconfirmed"]

    return "watch", reasons or ["quality_gate_watch"]


async def scan_coin(
    session:   aiohttp.ClientSession,
    symbol:    str,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict]:
    """
    Pipeline analisis lengkap untuk satu simbol.
    Semua HTTP call dijalankan secara paralel (asyncio.gather).
    Return: dict hasil, atau None jika data tidak cukup.
    """
    async with semaphore:
        tf      = CONFIG["TIMEFRAME"]
        limit   = CONFIG["FETCH_LIMIT"]
        N       = CONFIG["LOOKBACK_N"]
        oi_per  = OI_PERIOD_MAP.get(tf, "15m")

        try:
            # ── Ambil semua data secara paralel ────────────────────────
            fut_df, spot_df, funding_data, oi_hist = await asyncio.gather(
                fetch_futures_klines(session, symbol, tf, limit),
                fetch_spot_klines(session, symbol, tf, limit),
                fetch_funding_rate(session, symbol),
                fetch_oi_history(session, symbol, oi_per, N + 5),
                return_exceptions=False,
            )
            # Unpack tuple (fr, basis_pct) dari fetch_funding_rate
            if funding_data and isinstance(funding_data, tuple):
                _raw_fr, basis_pct = funding_data
            else:
                _raw_fr, basis_pct = None, 0.0

            # ── Validasi data minimal ──────────────────────────────────
            if fut_df is None or len(fut_df) < N + 2:
                return None   # data futures tidak cukup → skip

            current_price = float(fut_df["close"].iloc[-1])

            # ── Rolling VWAP ───────────────────────────────────────────
            vwap, d_vwap = calc_rolling_vwap(fut_df, N)

            # ── CVD Futures ────────────────────────────────────────────
            _, delta_cvd_fut = calc_cvd_and_momentum(fut_df, N)

            # ── CVD Spot ───────────────────────────────────────────────
            spot_available = spot_df is not None and len(spot_df) >= N + 2
            if spot_available:
                _, delta_cvd_spot = calc_cvd_and_momentum(spot_df, N)
            else:
                # Spot tidak tersedia (perp-only atau listing baru)
                # Asumsikan nol — skenario A tidak akan aktif tanpa spot data
                delta_cvd_spot = 0.0

            # ── OI Momentum ────────────────────────────────────────────
            delta_oi = calc_oi_momentum(oi_hist) if oi_hist else 0.0

            # ── Price Momentum ─────────────────────────────────────────
            delta_price = calc_price_momentum(fut_df, N)

            # ── Short-term Price Momentum (5 candle terakhir) ──────────
            delta_price_short = calc_price_momentum(fut_df, 5)

            # ── 24h Candle Direction ───────────────────────────────────
            tf_min = TF_MINUTES.get(CONFIG["TIMEFRAME"], 60)
            candles_24h = max(1, int(round(1440 / max(tf_min, 1))))
            price_change_24h = calc_price_momentum(fut_df, candles_24h)
            short_execution = _calc_short_execution_state(fut_df, candles_24h)

            # ── Funding Rate & Velocity ────────────────────────────────
            fr = _raw_fr if _raw_fr is not None else 0.0

            # FR Velocity: selisih FR sekarang vs FR scan sebelumnya
            now_ts = time.time()
            prev_entry = _FR_HISTORY_4H.get(symbol)

            if prev_entry is None:
                fr_velocity = 0.0
                _FR_HISTORY_4H[symbol] = (fr, now_ts)
            else:
                prev_fr, prev_ts = prev_entry
                if now_ts - prev_ts >= _FR_VELOCITY_INTERVAL:
                    fr_velocity = fr - prev_fr
                    _FR_HISTORY_4H[symbol] = (fr, now_ts)
                else:
                    fr_velocity = fr - prev_fr  # gunakan data lama, tidak update dulu

            # ── FORECAST ENGINE ────────────────────────────────────────
            vol_ratio, vol_label = calc_volume_anomaly(fut_df, N, tf_min)

            # ── Flag: short-term momentum sudah berbalik ───────────────
            # Kalau 5 candle terakhir sudah turun > 1%, long signal di-skip.
            # SHORT signals (ls, dist, div) TETAP dihitung — kondisi ini
            # justru konteks yang paling relevan untuk short.
            _skip_long = delta_price_short < -1.0

            # ── Squeeze Stage ───────────────────────────────────────────
            squeeze_type, squeeze_stage, squeeze_desc, squeeze_fuel = calc_squeeze_stage(
                delta_oi     = delta_oi,
                funding_rate = fr,
                delta_price  = delta_price,
            )

            # ── SCORING MATRIX ─────────────────────────────────────────
            score, flags, contexts = calculate_market_score(
                d_vwap         = d_vwap,
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                delta_oi       = delta_oi,
                delta_price    = delta_price,
                funding_rate   = fr,
                basis_pct      = basis_pct,
                fr_velocity    = fr_velocity,
                squeeze_type   = squeeze_type,
            )
            
            # ── ABSORPTION DETECTION ───────────────────────────────────
            absorption_penalty, absorption_desc = calc_absorption(
                fut_df         = fut_df,
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                delta_price    = delta_price,
                n              = N,
            )
            if absorption_penalty < 1.0:
                score = round(score * absorption_penalty, 1)
                contexts.append(absorption_desc)
                flags.append("ABSORPTION")
                
            # ── CVD-VOLUME CONSISTENCY ─────────────────────────────────
            cvd_vol_mult, cvd_vol_desc = calc_cvd_volume_consistency(
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                vol_ratio      = vol_ratio,
            )
            if cvd_vol_mult != 1.0:
                score = round(min(100.0, score * cvd_vol_mult), 1)
            if cvd_vol_desc:
                contexts.append(cvd_vol_desc)

            targets = calc_price_targets(
                price         = current_price,
                vwap          = vwap,
                d_vwap        = d_vwap,
                df            = fut_df,
                n             = N,
                flags         = flags,
                squeeze_stage = squeeze_stage,
            )

            # ── LONG SQUEEZE SCORE ─────────────────────────────────────
            ls_score, ls_flags, ls_contexts = calc_long_squeeze_score(
                delta_price   = delta_price,
                delta_oi      = delta_oi,
                funding_rate  = fr,
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                d_vwap        = d_vwap,
                squeeze_type  = squeeze_type,
                squeeze_stage = squeeze_stage,
                squeeze_fuel  = squeeze_fuel,
            )

            # Untuk target long squeeze, pass flags yang sudah include ls_flags
            if ls_score >= CONFIG["MIN_SCORE"] and squeeze_type in ("long", "long_exhausted"):
                ls_targets = calc_price_targets(
                    price         = current_price,
                    vwap          = vwap,
                    d_vwap        = d_vwap,
                    df            = fut_df,
                    n             = N,
                    flags         = ls_flags,
                    squeeze_stage = squeeze_stage,
                )
            else:
                ls_targets = {}

            # ── DISTRIBUTION SHORT SCORE ───────────────────────────────
            dist_score, dist_flags, dist_contexts = calc_distribution_score(
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                delta_oi       = delta_oi,
                funding_rate   = fr,
                d_vwap         = d_vwap,
                delta_price    = delta_price,
            )

            # ── BEARISH DIVERGENCE SHORT SCORE (v2) ───────────────────
            # Pipeline terpisah — tidak mempengaruhi long scoring
            div_score, div_flags, div_contexts = calc_bearish_divergence_score(
                delta_price    = delta_price,
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                delta_oi       = delta_oi,
                funding_rate   = fr,
                d_vwap         = d_vwap,
                vol_ratio      = vol_ratio,
                fr_velocity    = fr_velocity,
            )

            # ── INTEGRATED SHORT SCORE ────────────────────────────────
            # Cabang short mandiri: LS/DIST/DIV lama tetap dipakai sebagai
            # sub-interpretasi, lalu digabung dengan arah 24h dan breakdown.
            short_score, short_flags, short_contexts, short_setups, short_deriv_state = calc_integrated_short_score(
                delta_price       = delta_price,
                delta_price_short = delta_price_short,
                price_change_24h  = price_change_24h,
                delta_cvd_spot    = delta_cvd_spot,
                delta_cvd_fut     = delta_cvd_fut,
                delta_oi          = delta_oi,
                funding_rate      = fr,
                d_vwap            = d_vwap,
                vol_ratio         = vol_ratio,
                squeeze_type      = squeeze_type,
                squeeze_stage     = squeeze_stage,
                ls_score          = ls_score,
                ls_flags          = ls_flags,
                ls_contexts       = ls_contexts,
                dist_score        = dist_score,
                dist_flags        = dist_flags,
                dist_contexts     = dist_contexts,
                div_score         = div_score,
                div_flags         = div_flags,
                div_contexts      = div_contexts,
                short_rejection_score = short_execution["short_rejection_score"],
                failed_breakout       = short_execution["failed_breakout"],
                last_candle_bearish   = short_execution["last_candle_bearish"],
                last_close_position   = short_execution["last_close_position"],
                upper_wick_pct        = short_execution["upper_wick_pct"],
                near_24h_high         = short_execution["near_24h_high"],
            )

            if short_score > 0:
                short_targets = calc_price_targets(
                    price         = current_price,
                    vwap          = vwap,
                    d_vwap        = d_vwap,
                    df            = fut_df,
                    n             = N,
                    flags         = short_flags,
                    squeeze_stage = squeeze_stage,
                )
            else:
                short_targets = {}

            # Volume anomaly langsung memodifikasi skor akhir
            # Sinyal kuat + volume sepi = kurang konvinsif
            # Sinyal kuat + volume anomali = lebih dipercaya
            if vol_ratio >= 2.5:
                vol_multiplier = 1.10   # +10% kepercayaan
            elif vol_ratio >= 1.5:
                vol_multiplier = 1.04
            elif vol_ratio >= 0.8:
                vol_multiplier = 1.0    # netral
            else:
                vol_multiplier = 0.88   # volume sepi → kurangi kepercayaan
            score = round(min(100.0, score * vol_multiplier), 1)

            # ── Terapkan _skip_long: nol-kan skor LONG jika momentum berbalik ──
            if _skip_long:
                score = 0.0

            grade, grade_label = get_grade(score)

            # ── Print live per koin ───────────────────────────────────
            fr_pct    = fr * 100 if fr else 0.0
            flags_str = ",".join(flags) if flags else "─"
            marker    = (
                " ★" if score >= CONFIG["MIN_SCORE"]
                else " ▼" if short_score >= CONFIG["MIN_SCORE"]
                else "  "
            )
            log.info(
                f"{marker} {symbol:<18} L:{score:>5.1f} S:{short_score:>5.1f} {grade:>3} │ "
                f"ΔP:{delta_price:>+6.2f}% "
                f"VWAP:{d_vwap:>+5.2f}% "
                f"CVDs:{delta_cvd_spot:>+5.1f}% "
                f"CVDf:{delta_cvd_fut:>+5.1f}% "
                f"OI:{delta_oi:>+5.2f}% "
                f"FR:{fr_pct:>+6.4f}% │ "
                f"{flags_str}"
            )
            
            return {
                "symbol":         symbol,
                "price":          current_price,
                "vwap":           vwap,
                "d_vwap":         d_vwap,
                "delta_cvd_spot": delta_cvd_spot,
                "delta_cvd_fut":  delta_cvd_fut,
                "delta_oi":       delta_oi,
                "delta_price":    delta_price,
                "delta_price_short": delta_price_short,
                "price_change_24h": price_change_24h,
                "funding_rate":   fr,
                "score":          score,
                "grade":          grade,
                "grade_label":    grade_label,
                "flags":          flags,
                "contexts":       contexts,
                "spot_available":    spot_available,
                "basis_pct":         basis_pct,
                "fr_velocity":       fr_velocity,
                "absorption_penalty": absorption_penalty,
                "skip_long":         _skip_long,   # True jika short-term momentum berbalik
                # ── forecast fields ────────────────────────────────────
                "vol_ratio":      vol_ratio,
                "vol_label":      vol_label,
                "squeeze_stage":  squeeze_stage,
                "squeeze_desc":   squeeze_desc,
                "squeeze_fuel":   squeeze_fuel,
                "targets":        targets,
                # ── long squeeze fields ────────────────────────────────
                "squeeze_type":   squeeze_type,
                "ls_score":       ls_score,
                "ls_flags":       ls_flags,
                "ls_contexts":    ls_contexts,
                "ls_targets":     ls_targets,
                # ── distribution short fields ──────────────────────────
                "dist_score":    dist_score,
                "dist_flags":    dist_flags,
                "dist_contexts": dist_contexts,
                # ── divergence short fields (v2) ───────────────────────
                "div_score":     div_score,
                "div_flags":     div_flags,
                "div_contexts":  div_contexts,
                # ── integrated short branch ────────────────────────────
                "short_score":    short_score,
                "short_flags":    short_flags,
                "short_contexts": short_contexts,
                "short_setups":   short_setups,
                "short_targets":  short_targets,
                "short_deriv_state": short_deriv_state,
                "short_rejection_score": short_execution["short_rejection_score"],
                "failed_breakout":       short_execution["failed_breakout"],
                "last_candle_bearish":   short_execution["last_candle_bearish"],
                "last_close_position":   short_execution["last_close_position"],
                "upper_wick_pct":        short_execution["upper_wick_pct"],
                "near_24h_high":         short_execution["near_24h_high"],
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info(f"scan_coin [{symbol}] error: {exc}")
            return None


async def run_scan_batch(
    session: aiohttp.ClientSession,
    symbols: List[str],
    cycle_label: str = "",
) -> None:
    t_start = time.time()
    log.info("═" * 60)
    log.info(f"  🔍 {cycle_label} — {len(symbols)} koin")
    log.info("═" * 60)

    if not symbols:
        log.warning("Batch kosong — skip.")
        return

    # ── Fetch BTC data untuk Regime Engine ─────────────────────────────────
    regime_ctx = None
    try:
        btc_fut_df, btc_spot_df, btc_funding, btc_oi = await asyncio.gather(
            fetch_futures_klines(session, "BTCUSDT", CONFIG["TIMEFRAME"], CONFIG["FETCH_LIMIT"]),
            fetch_spot_klines(session, "BTCUSDT", CONFIG["TIMEFRAME"], CONFIG["FETCH_LIMIT"]),
            fetch_funding_rate(session, "BTCUSDT"),
            fetch_oi_history(session, "BTCUSDT", OI_PERIOD_MAP.get(CONFIG["TIMEFRAME"], "15m"), CONFIG["LOOKBACK_N"] + 5),
            return_exceptions=True,
        )
        if btc_fut_df is not None and not isinstance(btc_fut_df, Exception):
            N = CONFIG["LOOKBACK_N"]
            btc_vwap, btc_d_vwap = calc_rolling_vwap(btc_fut_df, N)
            _, btc_cvd_fut        = calc_cvd_and_momentum(btc_fut_df, N)
            btc_cvd_spot = 0.0
            if btc_spot_df is not None and not isinstance(btc_spot_df, Exception):
                _, btc_cvd_spot = calc_cvd_and_momentum(btc_spot_df, N)
            btc_fr, _ = btc_funding if (btc_funding and isinstance(btc_funding, tuple)) else (0.0, 0.0)
            btc_fr = btc_fr or 0.0
            btc_oi_delta = 0.0
            if btc_oi is not None and not isinstance(btc_oi, Exception) and len(btc_oi) >= 2:
                btc_oi_delta = float((btc_oi[-1] - btc_oi[-2]) / btc_oi[-2] * 100) if btc_oi[-2] > 0 else 0.0
            # FR velocity BTC
            btc_fr_vel = 0.0
            fr_key = f"BTCUSDT_regime"
            now_ts = time.time()
            if fr_key in _FR_HISTORY_4H:
                old_fr, old_ts = _FR_HISTORY_4H[fr_key]
                btc_fr_vel = btc_fr - old_fr
                if now_ts - old_ts >= 1800:
                    _FR_HISTORY_4H[fr_key] = (btc_fr, now_ts)
            else:
                _FR_HISTORY_4H[fr_key] = (btc_fr, now_ts)
            # Volume ratio BTC
            btc_vol = float(btc_fut_df["volume"].iloc[-N:].mean()) if len(btc_fut_df) >= N else 1.0
            btc_vol_avg = float(btc_fut_df["volume"].mean()) if len(btc_fut_df) > 0 else 1.0
            btc_vol_ratio = btc_vol / btc_vol_avg if btc_vol_avg > 0 else 1.0
            # Update regime engine
            engine = get_regime_engine(CONFIG["MIN_SCORE"])
            regime_ctx = engine.update(
                btc_price_vs_vwap = btc_d_vwap,
                btc_cvd_spot      = btc_cvd_spot,
                btc_cvd_fut       = btc_cvd_fut,
                btc_oi_delta      = btc_oi_delta,
                btc_fr            = btc_fr,
                btc_fr_velocity   = btc_fr_vel,
                btc_vol_ratio     = btc_vol_ratio,
            )
    except Exception as e:
        log.warning(f"[Regime] Gagal fetch BTC data: {e} — pakai default TRENDING")

    if regime_ctx is None:
        engine = get_regime_engine(CONFIG["MIN_SCORE"])
        regime_ctx = engine.get_context(confidence=0.60)

    semaphore = asyncio.Semaphore(CONFIG["CONCURRENT_TASKS"])
    tasks     = [scan_coin(session, sym, semaphore) for sym in symbols]

    # Jalankan semua scan secara paralel
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    errors  = 0
    skipped = 0   # koin yang return None (data kurang / skip_long / exception)
    for r in raw_results:
        if isinstance(r, Exception):
            errors += 1
        elif r is not None:
            results.append(r)
        else:
            skipped += 1

    # ── Apply regime context ke scoring dan filtering ───────────────────────
    _long_threshold  = regime_ctx.long_threshold
    _short_threshold = regime_ctx.short_threshold
    _regime_scores = getattr(get_regime_engine(CONFIG["MIN_SCORE"]), "last_scores", {}) or {}
    _pre_euphoric_guard = (
        regime_ctx.regime != "EUPHORIC"
        and _regime_scores.get("EUPHORIC", 0.0) >= 80.0
    )

    if _pre_euphoric_guard:
        _long_threshold = max(_long_threshold, CONFIG["MIN_SCORE"] + 6.0)
        _short_threshold = min(_short_threshold, CONFIG["MIN_SCORE"] + 2.0)

    # Log regime aktif
    log.info(
        f"[Regime] Active: {regime_ctx.regime} | "
        f"conf={regime_ctx.confidence:.2f} | "
        f"long_thr={_long_threshold:.1f} | "
        f"short_thr={_short_threshold:.1f} | "
        f"risk={regime_ctx.risk_profile} | "
        f"pre_euphoric={_pre_euphoric_guard}"
    )

    # Suspend semua signal saat PANIC
    if regime_ctx.risk_profile == "suspended":
        log.warning("[Regime] PANIC — semua sinyal disuspend sementara")
        log.info(
            f"  ✅ Scan selesai — 0 alert (PANIC regime) / "
            f"{len(results)} hasil / {skipped} skip"
        )
        return

    # Apply regime ke score: bonus continuation, penalty counter-trend
    for r in results:
        adj = 0.0
        if "continuation" in regime_ctx.allowed_setups:
            if any(f in r.get("flags", []) for f in ["A", "A_FUTURES", "B_VOLUME"]):
                adj += regime_ctx.score_bonus_continuation
        if _pre_euphoric_guard and r.get("score", 0) > 0:
            adj -= 3.0
        if regime_ctx.score_penalty_counter > 0:
            if r.get("div_score", 0) > 40 and r.get("score", 0) > 0:
                adj -= regime_ctx.score_penalty_counter * 0.5
        r["score_regime_adj"] = round(r.get("score", 0) + adj, 1)

    # Apply exhaustion sensitivity sejak awal agar kandidat short terlihat di log.
    for r in results:
        if regime_ctx.exhaustion_sensitivity != 1.0:
            r["div_score_adj"] = round(r.get("div_score", 0) * regime_ctx.exhaustion_sensitivity, 1)
        else:
            r["div_score_adj"] = r.get("div_score", 0)

    # Apply regime ke cabang short mandiri.
    for r in results:
        short_base = float(r.get("short_score", 0.0))
        short_adj = 0.0
        short_setups = set(r.get("short_setups", []))

        if short_base <= 0:
            r["short_score_regime_adj"] = 0.0
            continue

        if _pre_euphoric_guard:
            short_adj += 3.0

        if regime_ctx.regime == "EUPHORIC":
            if short_setups & {
                "top_reversal_short",
                "exhaustion_after_pump_short",
                "bearish_divergence_short",
                "distribution_short",
                "long_squeeze_short",
            }:
                short_adj += 5.0
        elif regime_ctx.regime == "TRENDING":
            if short_setups & {"bear_continuation_short", "breakdown_short"}:
                short_adj -= 3.0
        elif regime_ctx.regime == "CHOP":
            if short_setups & {"bear_continuation_short", "breakdown_short"}:
                short_adj -= 4.0
        elif regime_ctx.regime == "RECOVERY":
            if short_setups & {"bear_continuation_short", "breakdown_short", "long_squeeze_short"}:
                short_adj -= 8.0
            else:
                short_adj -= 4.0

        r["short_score_regime_adj"] = round(max(0.0, min(100.0, short_base + short_adj)), 1)

    # ── Gate per tipe setup — pisah long vs short ──────────────────────
    # Sebelumnya satu gate untuk semua, menyebabkan regime EUPHORIC
    # memblokir SEMUA sinyal karena allowed_setups-nya tidak ada yang match.
    # Sekarang long dan short punya gate masing-masing.
    _LONG_SETUP_TYPES  = {
        "continuation", "breakout", "pullback",
        "accumulation_long", "extreme_reversal", "momentum_ignition",
    }
    _SHORT_SETUP_TYPES = {
        "exhaustion_short", "fade_top", "extreme_reversal",
        "bear_continuation_short", "breakdown_short",
        "long_squeeze_short", "distribution_short",
        "bearish_divergence_short", "exhaustion_after_pump_short",
        "top_reversal_short",
    }
    _long_setups_ok  = bool(_LONG_SETUP_TYPES  & set(regime_ctx.allowed_setups))
    # SHORT boleh dievaluasi di semua regime non-PANIC; keketatannya diatur
    # oleh _short_gate_decision, bukan hard-block regime allowed_setups.
    _short_setups_ok = regime_ctx.risk_profile != "suspended"

    breadth_ctx = calculate_market_breadth(
        results=results,
        long_threshold=_long_threshold,
        short_threshold=_short_threshold,
    )

    _base_long_threshold = _long_threshold
    _base_short_threshold = _short_threshold
    _breadth_allowed_long_setups = set(_LONG_SETUP_TYPES)
    _breadth_allowed_short_setups = set(_SHORT_SETUP_TYPES)
    _breadth_force_short_ok = False

    if breadth_ctx.long_mode == "blocked":
        _long_threshold = max(_long_threshold + 14.0, CONFIG["MIN_SCORE"] + 18.0)
        _short_threshold = max(_short_threshold, CONFIG["MIN_SCORE"] + 12.0)
        _breadth_allowed_long_setups = {"extreme_reversal"}
        _breadth_force_short_ok = True
    elif breadth_ctx.long_mode == "restricted":
        _long_threshold = max(_long_threshold + 8.0, CONFIG["MIN_SCORE"] + 10.0)
        _short_threshold = max(_short_threshold, CONFIG["MIN_SCORE"] + 10.0)
        _breadth_allowed_long_setups = {"extreme_reversal", "momentum_ignition"}
        _breadth_force_short_ok = True
    elif breadth_ctx.long_mode == "cautious":
        _long_threshold = max(_long_threshold + 4.0, CONFIG["MIN_SCORE"] + 4.0)
        _short_threshold = max(_short_threshold, CONFIG["MIN_SCORE"] + 8.0)
        _breadth_allowed_long_setups = {
            "breakout", "pullback", "accumulation_long",
            "extreme_reversal", "momentum_ignition",
        }
        _breadth_force_short_ok = True
    elif breadth_ctx.short_mode in ("favored", "aggressive"):
        _short_threshold = max(_short_threshold, CONFIG["MIN_SCORE"] + 8.0)
        _breadth_force_short_ok = True
    elif breadth_ctx.short_mode == "cautious":
        _short_threshold = max(_short_threshold + 4.0, CONFIG["MIN_SCORE"] + 6.0)
        _breadth_allowed_short_setups = {
            "exhaustion_short", "fade_top",
            "long_squeeze_short", "distribution_short",
            "bearish_divergence_short", "exhaustion_after_pump_short",
            "top_reversal_short",
        }

    if (
        _long_threshold != _base_long_threshold
        or _short_threshold != _base_short_threshold
    ):
        log.info(
            f"[BreadthOverlay] bias={breadth_ctx.direction} "
            f"strength={breadth_ctx.strength:.2f} conf={breadth_ctx.confirmations} | "
            f"long_thr {_base_long_threshold:.1f}->{_long_threshold:.1f} "
            f"short_thr {_base_short_threshold:.1f}->{_short_threshold:.1f} | "
            f"long_mode={breadth_ctx.long_mode} short_mode={breadth_ctx.short_mode}"
        )

    for r in results:
        short_score_adj = float(r.get("short_score_regime_adj", 0.0))
        if short_score_adj <= 0:
            continue

        setups = set(r.get("short_setups", []))
        breadth_adj = 0.0

        if breadth_ctx.short_mode == "aggressive" and setups & {
            "bear_continuation_short",
            "breakdown_short",
            "long_squeeze_short",
        }:
            breadth_adj += 2.0
        elif breadth_ctx.short_mode == "favored" and setups & {
            "bear_continuation_short",
            "breakdown_short",
            "long_squeeze_short",
        }:
            breadth_adj += 1.0
        elif breadth_ctx.short_mode == "cautious":
            breadth_adj -= 5.0

        if (
            breadth_ctx.direction in ("BULLISH_CONFIRMED", "BULLISH_WEAK")
            and setups & {"bear_continuation_short", "breakdown_short"}
        ):
            breadth_adj -= 6.0

        if (
            breadth_ctx.direction in ("BEARISH_CONFIRMED", "RISK_OFF")
            and setups & {"bear_continuation_short", "breakdown_short", "long_squeeze_short"}
        ):
            breadth_adj += 0.5

        r["short_score_regime_adj"] = round(
            max(0.0, min(100.0, short_score_adj + breadth_adj)),
            1,
        )

    breadth_ctx = calculate_market_breadth(
        results=results,
        long_threshold=_long_threshold,
        short_threshold=_short_threshold,
    )

    if _breadth_force_short_ok:
        _short_setups_ok = True

    log.info(
        f"[Regime] Gate — long_ok={_long_setups_ok} | short_ok={_short_setups_ok}"
    )
    log.info(format_market_breadth_log(breadth_ctx))

    _raw_long_candidates = [
        r for r in results
        if r.get("score_regime_adj", r["score"]) >= _long_threshold
    ]
    _setup_allowed_long_candidates = [
        r for r in _raw_long_candidates
        if _is_long_allowed_by_regime(r, regime_ctx.allowed_setups)
        and bool(set(r.get("long_setups", [])) & _breadth_allowed_long_setups)
        and (
            breadth_ctx.long_mode not in ("restricted", "blocked")
            or _has_bearish_breadth_recovery_exception(r)
        )
    ]
    _setup_blocked_long = len(_raw_long_candidates) - len(_setup_allowed_long_candidates)
    if _raw_long_candidates:
        log.info(
            f"[Regime] Long setup gate — raw={len(_raw_long_candidates)} | "
            f"allowed={len(_setup_allowed_long_candidates)} | "
            f"blocked={_setup_blocked_long}"
        )

    _raw_short_candidates = [
        r for r in results
        if r.get("short_score_regime_adj", r.get("short_score", 0.0)) >= _short_threshold
    ]
    _setup_allowed_short_candidates = [
        r for r in _raw_short_candidates
        if bool(set(r.get("short_setups", [])) & _breadth_allowed_short_setups)
    ]
    _setup_blocked_short = len(_raw_short_candidates) - len(_setup_allowed_short_candidates)

    _short_setup_counts = {
        name: sum(1 for r in _setup_allowed_short_candidates if name in r.get("short_setups", []))
        for name in sorted(_breadth_allowed_short_setups)
    }
    _short_setup_summary = " ".join(
        f"{name.replace('_short', '').upper()}={count}"
        for name, count in _short_setup_counts.items()
        if count
    ) or "none"

    log.info(
        f"[Regime] Short setup gate — raw={len(_raw_short_candidates)} | "
        f"allowed={len(_setup_allowed_short_candidates)} | "
        f"blocked={_setup_blocked_short} | {_short_setup_summary}"
    )

    _quality_alert_short_candidates = []
    _quality_watch_short_candidates = []
    _quality_blocked_short_candidates = []
    _short_gate_reason_counts: Dict[str, int] = {}

    for r in _setup_allowed_short_candidates:
        status, reasons = _short_gate_decision(r, regime_ctx, breadth_ctx)
        r["short_gate_status"] = status
        r["short_gate_reasons"] = reasons

        for reason in reasons:
            _short_gate_reason_counts[reason] = _short_gate_reason_counts.get(reason, 0) + 1

        if status == "alert":
            _quality_alert_short_candidates.append(r)
        elif status == "blocked":
            _quality_blocked_short_candidates.append(r)
        else:
            _quality_watch_short_candidates.append(r)

    _reason_summary = " ".join(
        f"{reason}={count}"
        for reason, count in sorted(_short_gate_reason_counts.items())
    ) or "none"
    log.info(
        f"[ShortGate] raw={len(_setup_allowed_short_candidates)} | "
        f"alert={len(_quality_alert_short_candidates)} | "
        f"watch={len(_quality_watch_short_candidates)} | "
        f"blocked={len(_quality_blocked_short_candidates)} | "
        f"reasons: {_reason_summary}"
    )

    alerts = [
        r for r in _setup_allowed_long_candidates
        if _long_setups_ok
        and not (
            "C_SQUEEZE" in r.get("flags", [])
            and r.get("squeeze_fuel", 100) < 45
            and "A" not in r.get("flags", [])
        )
    ]
    alerts.sort(key=lambda x: x.get("score_regime_adj", x["score"]), reverse=True)

    short_alerts = [
        r for r in _quality_alert_short_candidates
        if _short_setups_ok
    ]
    short_alerts.sort(
        key=lambda x: x.get("short_score_regime_adj", x.get("short_score", 0.0)),
        reverse=True,
    )

    t_elapsed = time.time() - t_start
    total_alerts = len(alerts) + len(short_alerts)
    log.info("═" * 60)
    log.info(
        f"  ✅ Scan selesai — {total_alerts} alert "
        f"({len(alerts)} long / {len(short_alerts)} short) / {len(results)} hasil "
        f"/ {skipped} skip / {errors} error | waktu: {t_elapsed:.1f}s"
    )
    log.info("═" * 60)

    # Ringkasan akhir setelah semua koin selesai
    log.info("  " + "─" * 115)
    log.info(
        f"  📊 Selesai — {len(results)} koin discan │ "
        f"{total_alerts} lolos (★/▼) │ "
        f"{skipped} skip │ {errors} error │ {t_elapsed:.1f}s"
    )
    if alerts:
        top = alerts[0]
        log.info(
            f"  🏆 Tertinggi: {top['symbol']} "
            f"score={top['score']:.1f} [{top['grade']}] "
            f"flags={','.join(top['flags'])}"
        )
    if short_alerts:
        top_s = short_alerts[0]
        log.info(
            f"  🏆 Short tertinggi: {top_s['symbol']} "
            f"score={top_s.get('short_score_regime_adj', top_s.get('short_score', 0.0)):.1f} "
            f"setups={','.join(top_s.get('short_setups', []))} "
            f"state={top_s.get('short_deriv_state', 'neutral')}"
        )
    log.info("  " + "─" * 115)

    # Kirim alert ke Telegram — LONG signals
    for r in alerts:
        sym = r["symbol"]
        if _is_on_cooldown(sym, "LONG"):
            log.info(f"  ⏳ {sym:<18} [LONG] cooldown aktif — skip")
            continue
        msg = build_telegram_message(r, signal_type="LONG", regime_ctx=regime_ctx)
        ok  = await send_telegram(session, msg)
        status = "✓ terkirim" if ok else "✗ gagal"
        if ok:
            _mark_sent(sym, "LONG")
        log.info(f"  📤 {sym:<18} [{r['grade']:>2}] score={r['score']:5.1f} → {status}")
        # ── Kirim ke Dashboard ──
        tgt = r.get("targets", {})
        send_signal_to_dashboard(
            symbol=sym, direction="LONG", entry=r["price"],
            tp=tgt.get("target1") or round(r["price"] * 1.02, 8),
            sl=tgt.get("invalidasi") or round(r["price"] * 0.995, 8),
            grade=r["grade"], leverage=5,
        )
        await asyncio.sleep(CONFIG["MSG_DELAY"])

    # Kirim alert ke Telegram — integrated SHORT signals
    for r in short_alerts:
        sym = r["symbol"]
        if _is_on_cooldown(sym, "SHORT"):
            log.info(f"  ⏳ {sym:<18} [SHORT] cooldown aktif — skip")
            continue

        short_score = r.get("short_score_regime_adj", r.get("short_score", 0.0))
        short_r = dict(r)
        short_r["ls_score"] = short_score
        short_r["ls_flags"] = r.get("short_flags", [])
        short_r["ls_contexts"] = r.get("short_contexts", [])
        short_r["ls_targets"] = r.get("short_targets", {})
        short_r["squeeze_type"] = ",".join(r.get("short_setups", [])) or "integrated_short"

        msg = build_telegram_message(short_r, signal_type="SHORT", regime_ctx=regime_ctx)
        ok  = await send_telegram(session, msg)
        if ok:
            _mark_sent(sym, "SHORT")

        setups = ",".join(r.get("short_setups", [])) or "short"
        state = r.get("short_deriv_state", "neutral")
        reasons = ",".join(r.get("short_gate_reasons", [])) or "gate_ok"
        log.info(
            f"  📤 {sym:<18} [SHORT] "
            f"score={short_score:5.1f} setups={setups} state={state} gate={reasons} → "
            f"{'✓ terkirim' if ok else '✗ gagal'}"
        )

        tgt = short_r.get("ls_targets", {})
        if short_score >= 92:
            grade = "A+"
        elif short_score >= 82:
            grade = "A"
        elif short_score >= 72:
            grade = "B+"
        else:
            grade = "B"
        send_signal_to_dashboard(
            symbol=sym, direction="SHORT", entry=r["price"],
            tp=tgt.get("target1") or round(r["price"] * 0.98, 8),
            sl=tgt.get("invalidasi") or round(r["price"] * 1.005, 8),
            grade=grade, leverage=5,
        )
        await asyncio.sleep(CONFIG["MSG_DELAY"])
