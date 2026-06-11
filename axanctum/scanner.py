from __future__ import annotations

import asyncio
import time
from html import escape
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
from .scoring.long_engine import (
    calculate_market_score,
    classify_spot_accumulation_quality,
    classify_short_squeeze_quality,
)
from .scoring.breadth import calculate_market_breadth, format_market_breadth_log
from .scoring.regime_engine import get_regime_engine
from .scoring.short_engine import (
    calc_bearish_divergence_score,
    calc_distribution_score,
    calc_integrated_short_score,
    calc_long_squeeze_score,
)
from .state import _FR_HISTORY_4H, _FR_VELOCITY_INTERVAL


_NO_SIGNAL_DIAG_LAST_SENT = 0.0
_NO_SIGNAL_DIAG_DEFAULT_INTERVAL_SEC = 60 * 60


def _safe_num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _top_items(counter: Dict[str, int], limit: int = 6) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{escape(str(key))}={count}" for key, count in items)


def _candidate_digest(r: Optional[Dict], direction: str, threshold: float) -> str:
    if not r:
        return "none"

    symbol = escape(str(r.get("symbol", "-")))
    if direction == "LONG":
        score = _safe_num(r.get("score_regime_adj", r.get("score", 0.0)))
        setups_raw = r.get("long_setups") or r.get("flags", [])
        phase = r.get("long_phase_state", "neutral")
    else:
        score = _safe_num(r.get("short_score_regime_adj", r.get("short_score", 0.0)))
        setups_raw = r.get("short_setups") or r.get("short_flags", [])
        phase = r.get("short_phase_state", r.get("short_deriv_state", "neutral"))

    if isinstance(setups_raw, (list, tuple, set)):
        setups = list(setups_raw)
    elif setups_raw:
        setups = [setups_raw]
    else:
        setups = []
    setup_text = ",".join(str(item) for item in setups[:3]) if setups else "none"
    return (
        f"{symbol} {score:.1f}/{threshold:.1f} "
        f"setup={escape(setup_text)} phase={escape(str(phase))}"
    )


def _should_send_no_signal_diagnostic(now: Optional[float] = None) -> bool:
    interval = _safe_num(
        CONFIG.get("NO_SIGNAL_DIAG_INTERVAL_MIN", 60),
        _NO_SIGNAL_DIAG_DEFAULT_INTERVAL_SEC / 60,
    ) * 60
    now = time.time() if now is None else now
    return now - _NO_SIGNAL_DIAG_LAST_SENT >= max(300.0, interval)


def _mark_no_signal_diagnostic_sent(now: Optional[float] = None) -> None:
    global _NO_SIGNAL_DIAG_LAST_SENT
    _NO_SIGNAL_DIAG_LAST_SENT = time.time() if now is None else now


def _build_no_signal_diagnostic_message(
    *,
    cycle_label: str,
    results: List[Dict],
    skipped: int,
    errors: int,
    elapsed_sec: float,
    regime_ctx,
    breadth_ctx,
    base_long_threshold: float,
    base_short_threshold: float,
    long_threshold: float,
    short_threshold: float,
    raw_long_count: int,
    allowed_long_count: int,
    final_long_count: int,
    raw_short_count: int,
    allowed_short_count: int,
    alert_short_count: int,
    watch_short_count: int,
    blocked_short_count: int,
    short_gate_reason_counts: Dict[str, int],
) -> str:
    top_long = max(
        results,
        key=lambda item: _safe_num(item.get("score_regime_adj", item.get("score", 0.0))),
        default=None,
    )
    top_short = max(
        results,
        key=lambda item: _safe_num(item.get("short_score_regime_adj", item.get("short_score", 0.0))),
        default=None,
    )

    long_blocked = max(0, raw_long_count - allowed_long_count)
    long_quality_blocked = max(0, allowed_long_count - final_long_count)
    short_setup_blocked = max(0, raw_short_count - allowed_short_count)

    lines = [
        "🧭 <b>No-signal diagnostic</b>",
        f"<b>{escape(cycle_label or 'Scan batch')}</b>",
        (
            f"Regime: <b>{escape(str(regime_ctx.regime))}</b> "
            f"conf={_safe_num(regime_ctx.confidence):.2f} "
            f"risk={escape(str(regime_ctx.risk_profile))}"
        ),
        (
            "Threshold: "
            f"L {_safe_num(base_long_threshold):.1f}→{_safe_num(long_threshold):.1f} | "
            f"S {_safe_num(base_short_threshold):.1f}→{_safe_num(short_threshold):.1f}"
        ),
        (
            "Breadth: "
            f"{escape(str(breadth_ctx.direction))} "
            f"strength={_safe_num(breadth_ctx.strength):.2f} "
            f"L={escape(str(breadth_ctx.long_mode))} "
            f"S={escape(str(breadth_ctx.short_mode))}"
        ),
        (
            "Long gate: "
            f"raw={raw_long_count} allowed={allowed_long_count} "
            f"final={final_long_count} blocked={long_blocked}/{long_quality_blocked}"
        ),
        (
            "Short gate: "
            f"raw={raw_short_count} allowed={allowed_short_count} "
            f"alert={alert_short_count} watch={watch_short_count} "
            f"blocked={blocked_short_count} setup_blocked={short_setup_blocked}"
        ),
        f"Top L: {_candidate_digest(top_long, 'LONG', long_threshold)}",
        f"Top S: {_candidate_digest(top_short, 'SHORT', short_threshold)}",
        f"Short reasons: {_top_items(short_gate_reason_counts)}",
        (
            f"Scanned={len(results)} skip={skipped} err={errors} "
            f"time={elapsed_sec:.1f}s"
        ),
        "<i>Diagnostic only — bukan sinyal entry.</i>",
    ]
    return "\n".join(lines)


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


def _calc_cvd_window_state(df, recent_n: int, prior_n: int) -> Dict:
    """Baca apakah CVD terbaru ekspansif atau cuma bolak-balik dalam range."""
    state = {
        "recent_pct": 0.0,
        "prior_pct": 0.0,
        "efficiency": 0.0,
        "sign_changes": 0,
        "choppy": False,
        "flat": True,
    }
    if df is None or len(df) < recent_n + prior_n + 1:
        return state

    window = df.tail(recent_n + prior_n).copy()
    delta = window["taker_buy_vol"] - window["taker_sell_vol"]
    total = window["taker_buy_vol"] + window["taker_sell_vol"]

    prior_delta = delta.iloc[:prior_n]
    prior_total = float(total.iloc[:prior_n].sum())
    recent_delta = delta.iloc[prior_n:]
    recent_total = float(total.iloc[prior_n:].sum())

    prior_pct = float(prior_delta.sum() / prior_total * 100.0) if prior_total > 1e-10 else 0.0
    recent_pct = float(recent_delta.sum() / recent_total * 100.0) if recent_total > 1e-10 else 0.0

    abs_flow = float(recent_delta.abs().sum())
    efficiency = abs(float(recent_delta.sum())) / abs_flow if abs_flow > 1e-10 else 0.0
    signed = [
        1 if value > 0 else -1
        for value in recent_delta
        if abs(float(value)) > 1e-10
    ]
    sign_changes = sum(1 for i in range(1, len(signed)) if signed[i] != signed[i - 1])
    choppy = bool(
        len(signed) >= 4
        and efficiency <= 0.35
        and sign_changes >= max(2, len(signed) // 3)
    )
    flat = abs(recent_pct) <= 1.0 or (
        prior_pct > 2.0 and recent_pct < max(0.75, prior_pct * 0.35)
    )

    state.update({
        "recent_pct": round(recent_pct, 2),
        "prior_pct": round(prior_pct, 2),
        "efficiency": round(efficiency, 3),
        "sign_changes": sign_changes,
        "choppy": choppy,
        "flat": flat,
    })
    return state


def _classify_cvd_confluence(spot_pct: float, fut_pct: float) -> Dict:
    """Label arah CVD lokal agar futures-only sell tidak dibaca konfluensi."""
    min_dir = 0.75
    strong_spot = 1.2

    spot_bull = spot_pct >= min_dir
    spot_bear = spot_pct <= -min_dir
    fut_bull = fut_pct >= min_dir
    fut_bear = fut_pct <= -min_dir

    direction = "mixed_neutral"
    divergence = ""
    if spot_bear and fut_bear:
        direction = "bearish_cvd_confluence"
    elif spot_bull and fut_bear:
        direction = "mixed_divergence"
        divergence = "spot_bid_vs_perp_sell_divergence"
    elif spot_bear and fut_bull:
        direction = "mixed_divergence"
        divergence = "spot_sell_vs_perp_bid_divergence"
    elif spot_bull and fut_bull:
        direction = "bullish_cvd_confluence"
    elif spot_pct >= strong_spot and fut_pct < 0:
        direction = "mixed_divergence"
        divergence = "spot_bid_vs_perp_sell_divergence"
    elif spot_pct <= -strong_spot and fut_pct > 0:
        direction = "mixed_divergence"
        divergence = "spot_sell_vs_perp_bid_divergence"

    return {
        "direction": direction,
        "divergence_label": divergence,
        "spot_bullish_local": spot_pct >= strong_spot,
        "spot_bearish_local": spot_pct <= -min_dir,
        "fut_bearish_local": fut_pct <= -min_dir,
    }


def _calc_bearish_divergence_state(
    fut_df,
    *,
    n: int,
    delta_price: float,
    delta_price_short: float,
    d_vwap: float,
    delta_cvd_spot: float,
    delta_cvd_fut: float,
    div_score: float,
    div_flags: List[str],
    short_execution: Dict,
) -> Dict:
    """Klasifikasi bearish divergence berbasis price structure dan TTL."""
    state = {
        "divergence_type": "none",
        "divergence_status": "none",
        "price_structure_state": "neutral",
        "recent_swing_low": 0.0,
        "recent_swing_high": 0.0,
        "broke_recent_swing_low": False,
        "absorption_risk": False,
        "divergence_age_candles": 0,
        "short_score_cap_reason": "",
        "score_cap": None,
        "score_effective": div_score,
        "flags": [],
        "contexts": [],
    }

    if fut_df is None or len(fut_df) < 8 or div_score <= 0:
        return state

    structure_n = max(5, min(12, max(4, n // 2)))
    if len(fut_df) < structure_n + 1:
        structure_n = max(4, len(fut_df) - 1)
    if structure_n < 4:
        return state

    window = fut_df.tail(structure_n + 1).copy()
    recent = window.tail(structure_n)
    if len(recent) < 4:
        return state

    close = float(window["close"].iloc[-1])
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    prev_range = window.iloc[:-1]
    recent_swing_low = float(prev_range["low"].min()) if len(prev_range) else recent_low
    recent_swing_high = float(prev_range["high"].max()) if len(prev_range) else recent_high

    recent_start = float(window["close"].iloc[0])
    recent_move = (close - recent_start) / recent_start * 100.0 if recent_start > 0 else 0.0
    prior_anchor = float(fut_df["close"].iloc[-structure_n - 1]) if len(fut_df) > structure_n else recent_start
    prior_move = (recent_start - prior_anchor) / prior_anchor * 100.0 if prior_anchor > 0 else 0.0
    recent_range = (recent_high - recent_low) / close * 100.0 if close > 0 else 0.0

    split_idx = max(2, len(recent) // 2)
    early_recent = recent.iloc[:split_idx]
    late_recent = recent.iloc[split_idx:]
    early_high = float(early_recent["high"].max()) if len(early_recent) else recent_high
    early_low = float(early_recent["low"].min()) if len(early_recent) else recent_low
    late_high = float(late_recent["high"].max()) if len(late_recent) else recent_high
    late_low = float(late_recent["low"].min()) if len(late_recent) else recent_low

    lower_high = bool(late_high < early_high * 0.998)
    lower_low = bool(late_low < early_low * 0.998)
    higher_high = bool(late_high > early_high * 1.002)
    higher_low = bool(late_low > early_low * 1.002)

    range_sideways = bool(
        abs(recent_move) <= 1.4
        and recent_range <= max(2.2, abs(prior_move) * 0.65)
    )
    broke_recent_swing_low = bool(close < recent_swing_low * 0.997)
    broke_recent_swing_high = bool(close > recent_swing_high * 1.003)

    price_structure_state = "mixed"
    if broke_recent_swing_low:
        price_structure_state = (
            "lower_high_lower_low" if (lower_high and lower_low) else "breakdown_confirmed"
        )
    elif higher_high and higher_low:
        price_structure_state = "bullish_intact"
    elif range_sideways:
        price_structure_state = "range_hold"

    peak_idx = int(recent["high"].to_numpy().argmax())
    divergence_age_candles = max(0, len(recent) - 1 - peak_idx)
    ttl_candles = max(4, min(8, structure_n // 2 + 1))
    price_structure_broken = bool(broke_recent_swing_low)
    cvd_fading = delta_cvd_spot < 0 or delta_cvd_fut < 0
    absorption_risk = bool(
        div_score > 0
        and cvd_fading
        and not price_structure_broken
        and (
            delta_price >= 0.0
            or delta_price_short >= -0.1
            or close >= recent_high * 0.997
            or bool(short_execution.get("near_24h_high", False))
        )
    )

    divergence_type = "bearish_divergence"
    if delta_cvd_spot > 0 and delta_cvd_fut < 0:
        divergence_type = "spot_bid_vs_perp_sell_divergence"
    elif delta_cvd_spot < 0 and delta_cvd_fut > 0:
        divergence_type = "spot_sell_vs_perp_bid_divergence"
    elif delta_cvd_spot < 0 and delta_cvd_fut < 0:
        divergence_type = "bearish_cvd_confluence"
    elif delta_cvd_spot > 0 and delta_cvd_fut > 0:
        divergence_type = "bullish_cvd_confluence"
    if "DIV_HIDDEN_DIST" in div_flags:
        divergence_type = "hidden_distribution"
    elif "DIV_FUTURES_DIST" in div_flags:
        divergence_type = "futures_distribution"
    elif "DIV_CROSS_MARKET" in div_flags:
        divergence_type = "cross_market"
    elif "DIV_VWAP_EXTREME" in div_flags or "DIV_VWAP_DIST" in div_flags:
        divergence_type = "vwap_premium_exhaustion"
    elif "DIV_LEVERAGE_TRAP" in div_flags or "DIV_FR_OVERLOAD" in div_flags:
        divergence_type = "leverage_trap"

    if broke_recent_swing_high or (higher_high and higher_low):
        status = "invalidated"
        score_cap = 0.0
        cap_reason = "bearish_divergence_invalidated_higher_high"
        contexts = [
            "❌ Divergence invalidated: harga reclaim / higher high muncul sebelum breakdown",
        ]
        flags = ["BEARISH_DIVERGENCE_INVALIDATED"]
    elif price_structure_broken and cvd_fading:
        status = "confirmed"
        score_cap = None
        cap_reason = "bearish_divergence_confirmed_price_break"
        contexts = [
            f"✅ Divergence confirmed: {price_structure_state} + CVD tetap bearish saat breakdown",
        ]
        flags = ["BEARISH_DIVERGENCE_CONFIRMED"]
        if broke_recent_swing_low:
            flags.append("PRICE_STRUCTURE_BROKEN")
        if short_execution.get("failed_breakout", False):
            flags.append("FAILED_RECLAIM_RESISTANCE")
    elif divergence_age_candles > ttl_candles:
        status = "stale"
        score_cap = 12.0 if absorption_risk else 8.0
        cap_reason = "bearish_divergence_stale_ttl"
        contexts = [
            f"🕰️ Divergence stale: {divergence_age_candles} candle tanpa breakdown konfirmasi",
        ]
        flags = ["BEARISH_DIVERGENCE_STALE"]
    else:
        status = "watch"
        score_cap = 28.0 if absorption_risk else 30.0
        cap_reason = "bearish_divergence_watch_no_price_confirmation"
        contexts = [
            f"🎭 Bearish divergence watch: {price_structure_state} belum breakdown",
        ]
        flags = ["BEARISH_DIVERGENCE_WATCH"]

    if absorption_risk:
        flags.append("POSSIBLE_SELLER_ABSORPTION")
        contexts.append("🧲 Price hold/rise while CVD fades — absorption risk, belum confirmed short")

    score_effective = div_score
    if score_cap is not None:
        score_effective = min(div_score, score_cap)
        if status != "confirmed" and div_score > score_effective:
            contexts.append(f"⏳ Divergence score capped di {score_effective:.1f} karena {cap_reason}")

    state.update({
        "divergence_type": divergence_type,
        "divergence_status": status,
        "price_structure_state": price_structure_state,
        "recent_swing_low": round(recent_swing_low, 6),
        "recent_swing_high": round(recent_swing_high, 6),
        "broke_recent_swing_low": broke_recent_swing_low,
        "absorption_risk": absorption_risk,
        "divergence_age_candles": divergence_age_candles,
        "short_score_cap_reason": cap_reason,
        "score_cap": score_cap,
        "score_effective": round(score_effective, 2),
        "flags": flags,
        "contexts": contexts,
    })
    return state


def _calc_short_drop_debug(fut_df, recent_n: int, close: float, d_vwap: float) -> Dict:
    debug = {
        "recent_drop_atr": 0.0,
        "distance_from_vwap_zscore": 0.0,
    }
    if fut_df is None or len(fut_df) < recent_n + 1 or close <= 0:
        return debug

    tail = fut_df.tail(recent_n + 1)
    start_close = float(tail["close"].iloc[0])
    recent_drop_pct = (
        max(0.0, (start_close - close) / start_close * 100.0)
        if start_close > 0
        else 0.0
    )

    true_ranges = []
    for i in range(1, len(tail)):
        row = tail.iloc[i]
        prev_close = float(tail["close"].iloc[i - 1])
        true_high = max(float(row["high"]), prev_close)
        true_low = min(float(row["low"]), prev_close)
        true_ranges.append(max(0.0, true_high - true_low))

    atr_pct = 0.0
    if true_ranges:
        atr_pct = sum(true_ranges) / len(true_ranges) / close * 100.0
    atr_pct = max(0.1, min(atr_pct, 10.0))

    debug.update({
        "recent_drop_atr": round(recent_drop_pct / atr_pct, 2),
        "distance_from_vwap_zscore": round(d_vwap / atr_pct, 2),
    })
    return debug


def _calc_long_phase_state(
    fut_df,
    spot_df,
    *,
    n: int,
    candles_24h: int,
    price_change_24h: float,
    delta_price: float,
    delta_price_short: float,
    d_vwap: float,
    delta_oi: float,
    funding_rate: float,
    squeeze_type: str,
    squeeze_fuel: float,
    delta_cvd_fut: float,
) -> Dict:
    """
    Deteksi fase long: fresh accumulation, squeeze watch, atau stale top.

    Fokusnya mencegah jejak rally lama dibaca sebagai demand aktif saat
    harga sudah konsolidasi di pucuk dan CVD spot hanya mondar-mandir.
    """
    recent_n = max(5, min(12, max(1, n // 2)))
    prior_n = recent_n
    state = {
        "state": "neutral",
        "flags": [],
        "contexts": [],
        "score_mult": 1.0,
        "score_cap": None,
        "spot_cvd": _calc_cvd_window_state(spot_df, recent_n, prior_n),
        "fut_cvd": _calc_cvd_window_state(fut_df, recent_n, prior_n),
        "debug": {
            "downtrend_context": False,
            "post_rally_context": False,
            "price_reclaim_or_breakout": False,
            "short_squeeze_setup_score": 0.0,
            "short_squeeze_confirmation_score": 0.0,
            "short_squeeze_veto_reasons": [],
            "allow_short_squeeze_boost": False,
        },
    }

    if fut_df is None or len(fut_df) < recent_n + prior_n + 1:
        return state

    close = float(fut_df["close"].iloc[-1])
    if close <= 0:
        return state

    recent = fut_df.tail(recent_n)
    prior_anchor = float(fut_df["close"].iloc[-recent_n - prior_n - 1])
    recent_start = float(fut_df["close"].iloc[-recent_n - 1])
    prior_move = (recent_start - prior_anchor) / prior_anchor * 100.0 if prior_anchor > 0 else 0.0
    recent_move = (close - recent_start) / recent_start * 100.0 if recent_start > 0 else 0.0
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    recent_range = (recent_high - recent_low) / close * 100.0 if close > 0 else 0.0
    split_idx = max(2, len(recent) // 2)
    early_recent = recent.iloc[:split_idx]
    late_recent = recent.iloc[split_idx:]
    early_high = float(early_recent["high"].max()) if len(early_recent) else recent_high
    early_low = float(early_recent["low"].min()) if len(early_recent) else recent_low
    late_high = float(late_recent["high"].max()) if len(late_recent) else recent_high
    late_low = float(late_recent["low"].min()) if len(late_recent) else recent_low

    lookback_24h = max(3, min(candles_24h, len(fut_df)))
    high_24h = float(fut_df.tail(lookback_24h)["high"].max())
    near_24h_high = bool(high_24h > 0 and recent_high >= high_24h * 0.97)
    downtrend_context = bool(
        price_change_24h <= -3.0
        or prior_move <= -2.5
        or delta_price <= -0.75
        or d_vwap <= -1.0
    )
    rally_context = bool(
        (price_change_24h >= 6.0 or prior_move >= 4.0 or delta_price >= 3.0)
        and near_24h_high
    )
    post_rally = bool(rally_context and d_vwap >= 1.2)
    post_rally_context = bool(rally_context or post_rally)
    range_sideways = bool(abs(recent_move) <= 2.0 and recent_range <= max(1.8, abs(prior_move) * 0.75))

    prev_range = fut_df.iloc[-recent_n - 1:-1]
    range_high = float(prev_range["high"].max()) if len(prev_range) else recent_high
    range_low = float(prev_range["low"].min()) if len(prev_range) else recent_low
    breakout_up = bool(close > range_high * 1.003)
    breakdown_down = bool(close < range_low * 0.997)
    price_reclaim_or_breakout = bool(breakout_up or (delta_price_short > 0.25 and delta_price > 0))

    spot = state["spot_cvd"]
    fut = state["fut_cvd"]
    combined_recent_cvd = (spot["recent_pct"] + fut["recent_pct"] * 1.3) / 2.3
    spot_expanding = bool(
        spot["recent_pct"] >= 1.2
        and (not spot["choppy"] or spot["efficiency"] >= 0.55)
    )
    spot_pause = bool(spot["flat"] or spot["choppy"])
    fut_confirm = bool(fut["recent_pct"] >= 1.0 or delta_cvd_fut > 1.0)
    short_closing = bool(squeeze_type == "short" and delta_oi < -1.5 and delta_price_short > 0.3)
    lower_high = bool(late_high < early_high * 0.998)
    lower_low = bool(late_low < early_low * 0.998)
    local_rollover = bool(
        not breakout_up
        and (
            delta_price_short <= -0.35
            or recent_move <= -0.6
            or (lower_high and lower_low)
        )
    )
    bearish_cvd_alignment = bool(
        (spot["recent_pct"] <= -0.75 and fut["recent_pct"] <= -0.75)
        or (
            spot["recent_pct"] < 0
            and fut["recent_pct"] < 0
            and combined_recent_cvd <= -1.0
        )
    )
    squeeze_quality = classify_short_squeeze_quality(
        squeeze_type=squeeze_type,
        squeeze_fuel=squeeze_fuel,
        funding_rate=funding_rate,
        delta_oi=delta_oi,
        delta_price=delta_price,
        price_change_24h=price_change_24h,
        delta_price_short=delta_price_short,
        price_reclaim_or_breakout=price_reclaim_or_breakout,
        post_rally_context=post_rally_context,
        downtrend_context_hint=downtrend_context,
    )
    state["debug"].update({
        "downtrend_context": downtrend_context,
        "post_rally_context": post_rally_context,
        "price_reclaim_or_breakout": price_reclaim_or_breakout,
        "short_squeeze_setup_score": squeeze_quality["short_squeeze_setup_score"],
        "short_squeeze_confirmation_score": squeeze_quality["short_squeeze_confirmation_score"],
        "short_squeeze_veto_reasons": squeeze_quality["short_squeeze_veto_reasons"],
        "allow_short_squeeze_boost": squeeze_quality["allow_short_squeeze_boost"],
    })

    if squeeze_type == "short" and squeeze_quality["allow_short_squeeze_boost"] and price_reclaim_or_breakout:
        state.update({
            "state": "confirmed_squeeze",
            "flags": ["CONFIRMED_SHORT_SQUEEZE"],
            "contexts": [
                f"✅ Confirmed short squeeze: downtrend rebound + reclaim/breakout, Spot CVD {spot['recent_pct']:+.1f}%"
            ],
            "score_mult": 1.05,
            "score_cap": None,
        })
        return state

    if squeeze_type == "short" and post_rally_context and not squeeze_quality["allow_short_squeeze_boost"]:
        state.update({
            "state": "squeeze_watch",
            "flags": ["SQUEEZE_WATCH"],
            "contexts": [
                "🔫 Short squeeze watch: post-rally context belum cukup untuk konfirmasi"
            ],
            "score_mult": 0.82,
            "score_cap": 66.0,
        })
        return state

    if rally_context and local_rollover and bearish_cvd_alignment:
        flags = [
            "POST_RALLY_ROLLOVER",
            "BEARISH_CVD_ALIGNMENT",
            "STALE_SPOT_ACCUM",
            "LONG_THESIS_INVALID",
        ]
        contexts = [
            f"⛔ Post-rally rollover: Spot CVD {spot['recent_pct']:+.1f}% "
            f"dan Futures CVD {fut['recent_pct']:+.1f}% sama-sama bearish",
            f"📉 Local structure melemah: ΔP5 {delta_price_short:+.1f}% "
            f"lowerH={lower_high} lowerL={lower_low}",
        ]
        if funding_rate < 0:
            flags.append("SQUEEZE_WATCH")
            contexts.append("🔫 Funding negatif hanya context — belum ada reclaim/breakout squeeze")
        state.update({
            "state": "post_rally_rollover",
            "flags": flags,
            "contexts": contexts,
            "score_mult": 0.25,
            "score_cap": 42.0,
        })
        return state

    if post_rally and breakdown_down:
        state.update({
            "state": "long_thesis_invalid",
            "flags": ["LONG_THESIS_INVALID", "LONG_EXHAUSTION_RISK"],
            "contexts": ["⛔ Long thesis invalid: range bawah post-rally ditembus"],
            "score_mult": 0.35,
            "score_cap": 48.0,
        })
        return state

    if post_rally and range_sideways and spot_pause:
        flags = ["POST_RALLY_DEMAND_PAUSE", "STALE_SPOT_ACCUM"]
        contexts = [
            f"⏸️ Demand pause: post-rally range, Spot CVD recent {spot['recent_pct']:+.1f}% "
            f"vs prior {spot['prior_pct']:+.1f}%, eff={spot['efficiency']:.2f}"
        ]
        score_mult = 0.62
        score_cap = 62.0

        if funding_rate < 0:
            flags.append("SQUEEZE_WATCH")
            contexts.append("🔫 Funding negatif hanya squeeze fuel — belum ada breakout confirmation")
            score_cap = min(score_cap, 60.0)

        if delta_oi > 1.5 or not fut_confirm:
            flags.append("LONG_EXHAUSTION_RISK")
            contexts.append(
                f"⚠️ Long exhaustion risk: OI {delta_oi:+.1f}% "
                f"dan Futures CVD recent {fut['recent_pct']:+.1f}% belum confirm"
            )
            score_mult = 0.50
            score_cap = min(score_cap, 56.0)

        state.update({
            "state": "long_exhaustion_risk" if "LONG_EXHAUSTION_RISK" in flags else "post_rally_demand_pause",
            "flags": flags,
            "contexts": contexts,
            "score_mult": score_mult,
            "score_cap": score_cap,
        })
        return state

    if funding_rate < 0 and post_rally and not (squeeze_type == "short" and squeeze_quality["allow_short_squeeze_boost"]):
        state.update({
            "state": "squeeze_watch",
            "flags": ["SQUEEZE_WATCH"],
            "contexts": ["🔫 Squeeze watch: funding negatif, tapi belum ada breakout + CVD/OI confirmation"],
            "score_mult": 0.75,
            "score_cap": 66.0,
        })

    return state


def _calc_oi_window_state(oi_list, recent_n: int, prior_n: int) -> Dict:
    state = {
        "recent_pct": 0.0,
        "prior_pct": 0.0,
        "flat": True,
        "deleveraging": False,
        "building": False,
    }
    if not oi_list or len(oi_list) < recent_n + prior_n + 1:
        return state

    window = [float(v) for v in oi_list[-(recent_n + prior_n + 1):]]
    prior_start = window[0]
    recent_start = window[prior_n]
    current = window[-1]

    prior_pct = ((recent_start - prior_start) / prior_start * 100.0) if prior_start > 0 else 0.0
    recent_pct = ((current - recent_start) / recent_start * 100.0) if recent_start > 0 else 0.0

    state.update({
        "recent_pct": round(recent_pct, 2),
        "prior_pct": round(prior_pct, 2),
        "flat": abs(recent_pct) <= 0.8,
        "deleveraging": recent_pct <= -1.2,
        "building": recent_pct >= 1.0,
    })
    return state


def _calc_short_phase_state(
    fut_df,
    spot_df,
    oi_hist,
    *,
    n: int,
    candles_24h: int,
    price_change_24h: float,
    delta_price: float,
    delta_price_short: float,
    d_vwap: float,
    delta_oi: float,
    funding_rate: float,
    squeeze_type: str,
    vol_ratio: float,
) -> Dict:
    """
    Deteksi apakah short continuation masih fresh atau hanya residu selloff lama.

    Bearish aggregate data tetap penting, tapi alert short harus turun kelas saat
    harga sudah range di low, CVD terbaru tidak ekspansif, dan OI sudah keluar.
    """
    recent_n = max(5, min(12, max(1, n // 2)))
    prior_n = recent_n
    state = {
        "state": "neutral",
        "flags": [],
        "contexts": [],
        "score_mult": 1.0,
        "score_cap": None,
        "spot_cvd": _calc_cvd_window_state(spot_df, recent_n, prior_n),
        "fut_cvd": _calc_cvd_window_state(fut_df, recent_n, prior_n),
        "oi": _calc_oi_window_state(oi_hist, recent_n, prior_n),
        "debug": {
            "spot_cvd_slope_short": 0.0,
            "fut_cvd_slope_short": 0.0,
            "cvd_confluence_direction": "mixed_neutral",
            "spot_fut_divergence_label": "",
            "recent_drop_atr": 0.0,
            "distance_from_vwap_zscore": 0.0,
            "oi_phase": "neutral",
            "bear_expansion_phase": "neutral",
            "short_veto_reasons": [],
        },
    }

    if fut_df is None or len(fut_df) < recent_n + prior_n + 1:
        return state

    close = float(fut_df["close"].iloc[-1])
    if close <= 0:
        return state
    state["debug"].update(_calc_short_drop_debug(fut_df, recent_n, close, d_vwap))

    recent = fut_df.tail(recent_n)
    prior_anchor = float(fut_df["close"].iloc[-recent_n - prior_n - 1])
    recent_start = float(fut_df["close"].iloc[-recent_n - 1])
    prior_move = (recent_start - prior_anchor) / prior_anchor * 100.0 if prior_anchor > 0 else 0.0
    recent_move = (close - recent_start) / recent_start * 100.0 if recent_start > 0 else 0.0
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    recent_range = (recent_high - recent_low) / close * 100.0 if close > 0 else 0.0

    prev_range = fut_df.iloc[-recent_n - 1:-1]
    range_low = float(prev_range["low"].min()) if len(prev_range) else recent_low
    breakdown_down = bool(close < range_low * 0.997 and delta_price_short <= -0.35)

    lookback_24h = max(3, min(candles_24h, len(fut_df)))
    tail_24h = fut_df.tail(lookback_24h)
    low_24h = float(tail_24h["low"].min())
    high_24h = float(tail_24h["high"].max())
    day_range = high_24h - low_24h
    low_position = (close - low_24h) / day_range if day_range > 0 else 0.5

    prior_drop = bool(price_change_24h <= -4.0 or prior_move <= -3.0 or delta_price <= -2.0)
    range_sideways = bool(
        abs(recent_move) <= 1.4
        and recent_range <= max(2.2, abs(prior_move) * 0.65)
    )
    near_low = bool(low_position <= 0.35 or d_vwap <= -2.5)

    spot = state["spot_cvd"]
    fut = state["fut_cvd"]
    oi = state["oi"]
    spot_data_available = bool(spot_df is not None and len(spot_df) >= recent_n + prior_n + 1)
    cvd_local = _classify_cvd_confluence(spot["recent_pct"], fut["recent_pct"])
    spot_bid_vs_perp_sell = cvd_local["divergence_label"] == "spot_bid_vs_perp_sell_divergence"
    state["debug"].update({
        "spot_cvd_slope_short": spot["recent_pct"],
        "fut_cvd_slope_short": fut["recent_pct"],
        "cvd_confluence_direction": cvd_local["direction"],
        "spot_fut_divergence_label": cvd_local["divergence_label"],
    })

    spot_bear_expanding = bool(
        spot["recent_pct"] <= -1.2
        and (not spot["choppy"] or spot["efficiency"] >= 0.50 or spot["recent_pct"] <= -2.5)
    )
    fut_bear_expanding = bool(
        fut["recent_pct"] <= -1.2
        and (not fut["choppy"] or fut["efficiency"] >= 0.50 or fut["recent_pct"] <= -2.5)
    )
    combined_recent_cvd = (spot["recent_pct"] + fut["recent_pct"] * 1.3) / 2.3
    combined_prior_cvd = (spot["prior_pct"] + fut["prior_pct"] * 1.3) / 2.3
    spot_bear_faded = bool(
        spot["prior_pct"] <= -2.0
        and spot["recent_pct"] < 0
        and abs(spot["recent_pct"]) <= abs(spot["prior_pct"]) * 0.45
    )
    fut_bear_faded = bool(
        fut["prior_pct"] <= -2.0
        and fut["recent_pct"] < 0
        and abs(fut["recent_pct"]) <= abs(fut["prior_pct"]) * 0.45
    )
    bear_cvd_faded = bool(
        spot_bear_faded
        or fut_bear_faded
        or (
            combined_prior_cvd <= -2.0
            and combined_recent_cvd < 0
            and abs(combined_recent_cvd) <= abs(combined_prior_cvd) * 0.45
        )
    )
    bear_cvd_expanding = bool(
        combined_recent_cvd <= -1.4
        and fut_bear_expanding
        and (spot_bear_expanding or not spot_data_available)
        and not bear_cvd_faded
        and not cvd_local["spot_bullish_local"]
    )
    stale_bear_cvd = bool(
        (
            spot["prior_pct"] <= -1.2
            or fut["prior_pct"] <= -1.2
            or spot["recent_pct"] < 0
            or fut["recent_pct"] < 0
        )
        and not bear_cvd_expanding
        and (
            (spot["flat"] or spot["choppy"])
            and (fut["flat"] or fut["choppy"])
            or bear_cvd_faded
            or abs(combined_recent_cvd) <= 1.4
            or spot_bid_vs_perp_sell
        )
    )
    oi_deleveraging = bool(delta_oi <= -2.0 or oi["deleveraging"])
    oi_building = bool(delta_oi >= 1.5 or oi["building"])
    fresh_fuel = bool(oi_building or squeeze_type == "long" or (funding_rate > 0.0001 and delta_oi > -2.0))
    local_bearish_confluence = bool(
        cvd_local["fut_bearish_local"]
        and (cvd_local["spot_bearish_local"] or not spot_data_available)
    )
    fresh_breakdown = bool(
        breakdown_down
        and bear_cvd_expanding
        and local_bearish_confluence
        and (fresh_fuel or vol_ratio >= 1.2)
    )
    recent_drop_large = bool(
        prior_drop
        or state["debug"]["recent_drop_atr"] >= 1.5
        or delta_price <= -2.0
    )
    post_dump_spot_absorption = bool(
        recent_drop_large
        and near_low
        and spot_bid_vs_perp_sell
        and oi_deleveraging
    )
    short_veto_reasons: List[str] = []
    if post_dump_spot_absorption:
        short_veto_reasons.append("post_dump_spot_absorption_or_bounce_risk")

    oi_phase = "neutral"
    if oi_building and bear_cvd_expanding:
        oi_phase = "fresh_short_build"
    elif oi_deleveraging and fresh_breakdown:
        oi_phase = "long_liquidation"
    elif oi_deleveraging:
        oi_phase = "post_deleveraging"

    bear_expansion_phase = "neutral"
    if fresh_breakdown:
        bear_expansion_phase = "early"
    elif bear_cvd_expanding and breakdown_down:
        bear_expansion_phase = "active"
    elif prior_drop and near_low and (range_sideways or oi_deleveraging or spot_bid_vs_perp_sell):
        bear_expansion_phase = "late"
    elif bear_cvd_expanding:
        bear_expansion_phase = "active"
    elif stale_bear_cvd:
        bear_expansion_phase = "late"

    state["debug"].update({
        "oi_phase": oi_phase,
        "bear_expansion_phase": bear_expansion_phase,
        "short_veto_reasons": short_veto_reasons,
    })

    if fresh_breakdown:
        state.update({
            "state": "fresh_bear_expansion",
            "flags": ["FRESH_BEAR_EXPANSION"],
            "contexts": [
                f"✅ Fresh bear expansion: range low pecah + CVD recent {combined_recent_cvd:+.1f}%"
            ],
            "score_mult": 1.04,
            "score_cap": None,
        })
        return state

    if post_dump_spot_absorption:
        state.update({
            "state": "post_dump_spot_absorption",
            "flags": [
                "POST_DROP_EXHAUSTION",
                "SHORT_TRAP_RISK",
                "SPOT_BID_PERP_SELL_DIVERGENCE",
                "POST_DUMP_SPOT_ABSORPTION",
                "SHORT_FUEL_SPENT",
            ],
            "contexts": [
                f"🧲 Spot bid vs perp sell: Spot CVD {spot['recent_pct']:+.1f}% "
                f"sementara Futures CVD {fut['recent_pct']:+.1f}%",
                f"↩️ Post-dump absorption risk: drop {state['debug']['recent_drop_atr']:.2f}x ATR, "
                f"OI {delta_oi:+.1f}% sudah deleveraging",
            ],
            "score_mult": 0.45,
            "score_cap": 52.0,
        })
        return state

    if prior_drop and range_sideways and near_low and stale_bear_cvd and oi_deleveraging:
        state.update({
            "state": "short_trap_risk",
            "flags": ["POST_DROP_EXHAUSTION", "SHORT_TRAP_RISK", "STALE_BEAR_CVD", "SHORT_FUEL_SPENT"],
            "contexts": [
                f"🪫 Post-drop exhaustion: harga range di low, CVD recent {combined_recent_cvd:+.1f}% tidak ekspansif",
                f"💧 Short fuel spent: OI {delta_oi:+.1f}% / recent {oi['recent_pct']:+.1f}% sudah deleveraging",
            ],
            "score_mult": 0.50,
            "score_cap": 56.0,
        })
        return state

    if prior_drop and range_sideways and near_low and stale_bear_cvd:
        state.update({
            "state": "post_drop_exhaustion",
            "flags": ["POST_DROP_EXHAUSTION", "STALE_BEAR_CVD"],
            "contexts": [
                f"⏸️ Bear residue: post-drop range, CVD recent {combined_recent_cvd:+.1f}% "
                f"vs prior spot/fut {spot['prior_pct']:+.1f}/{fut['prior_pct']:+.1f}%"
            ],
            "score_mult": 0.68,
            "score_cap": 64.0,
        })
        return state

    if prior_drop and near_low and oi_deleveraging and not bear_cvd_expanding:
        state.update({
            "state": "short_fuel_spent",
            "flags": ["SHORT_FUEL_SPENT"],
            "contexts": [
                f"💧 OI deleveraging tanpa CVD ekspansif — fuel continuation menipis ({delta_oi:+.1f}%)"
            ],
            "score_mult": 0.76,
            "score_cap": 68.0,
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
    has_clean_risk = not bool(flags & {
        "B_SPECULATIVE",
        "D_EXHAUSTION",
        "E_DISTRIBUTION",
        "ABSORPTION",
        "POST_RALLY_DEMAND_PAUSE",
        "POST_RALLY_ROLLOVER",
        "BEARISH_CVD_ALIGNMENT",
        "STALE_SPOT_ACCUM",
        "SQUEEZE_WATCH",
        "LONG_EXHAUSTION_RISK",
        "LONG_THESIS_INVALID",
    })

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


def _calc_short_regime_adjustment(r: Dict, regime_ctx, pre_euphoric_guard: bool) -> tuple[float, List[str]]:
    """Setup-aware short regime adjustment."""
    short_setups = set(r.get("short_setups", []))
    short_flags = set(r.get("short_flags", []))

    price24 = float(r.get("price_change_24h", 0.0))
    d_vwap = float(r.get("d_vwap", 0.0))
    delta_price_short = float(r.get("delta_price_short", 0.0))
    delta_cvd_spot = float(r.get("delta_cvd_spot", 0.0))
    delta_cvd_fut = float(r.get("delta_cvd_fut", 0.0))
    delta_oi = float(r.get("delta_oi", 0.0))
    funding_rate = float(r.get("funding_rate", 0.0))
    vol_ratio = float(r.get("vol_ratio", 1.0))
    dist_score = float(r.get("dist_score", 0.0))
    div_score = float(r.get("div_score", 0.0))
    short_deriv_state = r.get("short_deriv_state", "neutral")
    short_phase_state = r.get("short_phase_state", "neutral")
    rejection_confirmed = bool(r.get("short_rejection_score", 0.0) >= 55.0 or r.get("failed_breakout", False))
    spot_absorption_risk = bool(r.get("spot_absorption_risk", False))
    micro_reversal_trigger = "MICRO_REVERSAL_TRIGGER" in short_flags
    late_continuation_chase = bool(
        price24 <= -8.0
        and d_vwap <= -4.0
        and short_setups & {"bear_continuation_short", "breakdown_short"}
        and not (short_setups & {"top_reversal_short", "exhaustion_after_pump_short", "distribution_short", "bearish_divergence_short"})
    )

    top_family = {
        "top_reversal_short",
        "exhaustion_after_pump_short",
        "distribution_short",
        "bearish_divergence_short",
    }
    trend_family = {"bear_continuation_short", "breakdown_short", "long_squeeze_short"}
    adj = 0.0
    reasons: List[str] = []

    if pre_euphoric_guard:
        adj += 2.0
        reasons.append("pre_euphoric_guard")

    if regime_ctx.regime == "EUPHORIC":
        if short_setups & top_family:
            adj += 6.0
            reasons.append("euphoric_top_short")
        if "bearish_divergence_watch" in short_setups or micro_reversal_trigger:
            adj += 3.0
            reasons.append("euphoric_reversal_watch")
        if short_setups & trend_family:
            adj -= 3.0
            reasons.append("euphoric_trend_short_penalty")

    elif regime_ctx.regime == "TRENDING":
        if short_setups & top_family:
            adj += 2.0
            reasons.append("trending_top_short")
        if micro_reversal_trigger or spot_absorption_risk or (dist_score >= 35.0 and div_score >= 30.0):
            adj += 2.0
            reasons.append("trending_micro_reversal_support")
        if short_setups & {"bear_continuation_short", "breakdown_short"}:
            adj -= 4.0
            reasons.append("trending_continuation_penalty")
        if short_setups & {"long_squeeze_short"}:
            adj -= 2.0
            reasons.append("trending_long_squeeze_penalty")

    elif regime_ctx.regime == "CHOP":
        if short_setups & top_family:
            adj += 3.0
            reasons.append("chop_top_short")
        if short_setups & {"bearish_divergence_watch"}:
            adj += 2.0
            reasons.append("chop_divergence_watch")
        if micro_reversal_trigger or spot_absorption_risk:
            adj += 2.0
            reasons.append("chop_micro_reversal_support")
        if short_setups & {"bear_continuation_short", "breakdown_short", "long_squeeze_short"}:
            adj -= 6.0
            reasons.append("chop_continuation_penalty")

    elif regime_ctx.regime == "RECOVERY":
        if short_setups & top_family and rejection_confirmed and (dist_score >= 55.0 or div_score >= 45.0):
            adj -= 1.0
            reasons.append("recovery_confirmed_top_soft_penalty")
        elif short_setups & top_family:
            adj -= 4.0
            reasons.append("recovery_top_penalty")
        if micro_reversal_trigger or spot_absorption_risk:
            adj -= 1.0
            reasons.append("recovery_micro_reversal_penalty")
        if short_setups & {"bear_continuation_short", "breakdown_short", "long_squeeze_short"}:
            adj -= 9.0
            reasons.append("recovery_continuation_penalty")
        if short_setups & {"bearish_divergence_watch"}:
            adj -= 3.0
            reasons.append("recovery_divergence_watch_penalty")

    if late_continuation_chase:
        adj -= 4.0
        reasons.append("late_continuation_chase")

    if short_phase_state in {"post_drop_exhaustion", "short_trap_risk", "short_fuel_spent"}:
        if short_setups & trend_family:
            adj -= 3.0
            reasons.append("post_drop_trend_penalty")
    if short_deriv_state in {"breakout_fuel", "late_deleveraging", "local_bounce_risk", "sell_pressure_absorbed"}:
        if short_setups & top_family:
            adj += 1.0
            reasons.append(f"deriv_{short_deriv_state}_top_support")

    # micro reversal di area atas seharusnya membantu top family, bukan continuation
    if micro_reversal_trigger and short_setups & top_family:
        adj += 2.0
        reasons.append("micro_reversal_top_support")

    # Small clamp supaya regime tidak mendominasi total score
    adj = max(-12.0, min(10.0, adj))
    return adj, reasons


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
    divergence_status = r.get("divergence_status", "none")
    price_structure_state = r.get("price_structure_state", "neutral")
    broke_recent_swing_low = bool(r.get("broke_recent_swing_low", False))
    absorption_risk = bool(r.get("absorption_risk", False))
    divergence_age_candles = int(r.get("divergence_age_candles", 0))
    short_score_cap_reason = r.get("short_score_cap_reason", "")
    short_deriv_state = r.get("short_deriv_state", "neutral")
    short_phase_state = r.get("short_phase_state", "neutral")
    short_rejection_score = float(r.get("short_rejection_score", 0.0))
    failed_breakout = bool(r.get("failed_breakout", False))
    last_candle_bearish = bool(r.get("last_candle_bearish", False))
    last_close_position = float(r.get("last_close_position", 0.5))
    near_24h_high = bool(r.get("near_24h_high", False))
    spot_absorption_risk = bool(r.get("spot_absorption_risk", False))
    micro_reversal_trigger = "MICRO_REVERSAL_TRIGGER" in flags

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
    fresh_bear_expansion = (
        short_phase_state == "fresh_bear_expansion"
        or "FRESH_BEAR_EXPANSION" in flags
    )
    post_drop_risk = (
        short_phase_state in {
            "post_drop_exhaustion",
            "short_trap_risk",
            "short_fuel_spent",
            "post_dump_spot_absorption",
        }
        or bool(flags & {"POST_DROP_EXHAUSTION", "SHORT_TRAP_RISK", "SHORT_FUEL_SPENT"})
    )
    spot_bid_perp_sell = bool(
        "SPOT_BID_PERP_SELL_DIVERGENCE" in flags
        or r.get("spot_fut_divergence_label") == "spot_bid_vs_perp_sell_divergence"
    )
    post_dump_absorption = bool(
        "POST_DUMP_SPOT_ABSORPTION" in flags
        or "post_dump_spot_absorption_or_bounce_risk" in r.get("short_veto_reasons", [])
    )
    deriv_support = bool(flags & {
        "SHORT_DERIV_LONG_TRAP",
        "SHORT_DERIV_BUY_PRESSURE_ABSORBED",
        "SHORT_DERIV_BEAR_CONTINUATION_FRESH",
        "SHORT_DERIV_LONG_SQUEEZE_FUEL",
        "BUY_PRESSURE_ABSORBED",
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
    local_chase_risk = (
        delta_price_short <= -0.75
        and d_vwap <= -1.0
        and last_close_position <= 0.42
        and not long_squeeze_active
        and not funding_trapped_long
        and not oi_hot_breakdown
    )

    late_entry = price24 <= -14.0 and d_vwap <= -7.0
    late_continuation_chase = (
        price24 <= -8.0
        and d_vwap <= -4.0
        and setups & {"bear_continuation_short", "breakdown_short"}
        and not (setups & top_setups)
    )
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
        or (strong_bear_flow and delta_oi > -6.0 and not post_drop_risk)
        or fresh_bear_expansion
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
    ) or div_score >= 50.0 or dist_score >= 55.0 or spot_absorption_risk or micro_reversal_trigger
    top_deriv = (
        funding_rate > 0.0005
        or delta_oi > 4.0
        or dist_score >= 70.0
        or div_score >= 70.0
        or deriv_support
        or (
            micro_reversal_trigger
            and (
                funding_rate > 0.0001
                or delta_oi >= 1.0
                or dist_score >= 45.0
                or div_score >= 45.0
                or spot_absorption_risk
            )
        )
    )
    top_valid = bool(
        setups & top_setups
        and top_context
        and (rejection_confirmed or micro_reversal_trigger)
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
    continuation_trigger = bool(
        breakdown_fresh
        or fresh_bear_expansion
        or long_squeeze_active
        or ("SHORT_DERIV_BEAR_CONTINUATION_FRESH" in flags and not post_drop_risk)
        or (
            funding_trapped_long
            and strong_bear_flow
            and delta_oi > -6.0
            and not post_drop_risk
        )
        or (
            oi_hot_breakdown
            and weak_bear_flow
            and price24 > -12.0
        )
    )
    bear_continuation_valid = bool(
        setups & trend_setups
        and trend_context
        and deriv_edge
        and weak_bear_flow
        and vol_ratio >= 0.7
        and not post_drop_risk
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

    if "SHORT_DERIV_LOCAL_BOUNCE_RISK" in flags:
        return "watch", ["local_bounce_risk"]

    if "SHORT_DERIV_SELL_PRESSURE_ABSORBED" in flags:
        return "watch", ["deriv_sell_pressure_absorbed"]

    if post_dump_absorption:
        return "watch", ["post_dump_spot_absorption_or_bounce_risk"]

    if spot_bid_perp_sell and setups & trend_setups and delta_oi <= -1.2:
        return "watch", ["spot_bid_vs_perp_sell_divergence"]

    if (
        post_drop_risk
        and setups & {"bear_continuation_short", "breakdown_short"}
        and not long_squeeze_active
        and not fresh_bear_expansion
    ):
        return "watch", [short_phase_state if short_phase_state != "neutral" else "post_drop_exhaustion"]

    if breakout_risk and setups & top_setups:
        return "watch", ["breakout_risk_no_rejection"]

    if regime_ctx.regime == "PANIC":
        return "blocked", ["panic_suspended"]

    if deep_distribution:
        reasons.append("distribution_not_top")

    if late_continuation_chase and not active_late_edge:
        return "watch", reasons + ["late_continuation_chase"]

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

    if "bear_continuation_short" in setups and local_chase_risk:
        return "watch", ["continuation_chase_risk"]

    if "bear_continuation_short" in setups and not continuation_trigger:
        return "watch", ["continuation_needs_trigger"]

    if "bear_continuation_short" in setups and not both_bearish and not long_squeeze_active:
        return "watch", ["weak_flow"]

    if bear_continuation_valid:
        if near_24h_high and not both_bearish and not long_squeeze_active:
            return "watch", ["near_high_needs_stronger_flow"]
        if late_entry:
            return "alert", ["bear_continuation_valid", "late_deriv_exception"]
        return "alert", ["bear_continuation_valid"]

    divergence_confirmed_edge = bool(
        divergence_status == "confirmed"
        and "bearish_divergence_short" in setups
        and (
            broke_recent_swing_low
            or price_structure_state in {"lower_high_lower_low", "breakdown_confirmed"}
            or failed_breakout
            or rejection_confirmed
        )
        and not post_drop_risk
    )
    if divergence_confirmed_edge:
        if late_entry or extreme_late:
            return "watch", ["bearish_divergence_confirmed_late"]
        return "alert", ["bearish_divergence_confirmed"]

    if breakdown_fresh and deriv_edge:
        return "alert", ["breakdown_fresh"]

    if long_squeeze_active and (weak_bear_flow or funding_trapped_long):
        return "alert", ["long_squeeze_valid"]

    if setups & top_setups:
        return "watch", ["top_setup_unconfirmed"]

    if divergence_status in {"watch", "stale", "invalidated"} and setups & {
        "bearish_divergence_watch",
        "bearish_divergence_short",
    }:
        reasons.append(f"bearish_divergence_{divergence_status}")
        if absorption_risk:
            reasons.append("possible_seller_absorption")
        if short_score_cap_reason:
            reasons.append(short_score_cap_reason)
        return "watch", reasons

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

            long_phase = _calc_long_phase_state(
                fut_df,
                spot_df if spot_available else None,
                n=N,
                candles_24h=candles_24h,
                price_change_24h=price_change_24h,
                delta_price=delta_price,
                delta_price_short=delta_price_short,
                d_vwap=d_vwap,
                delta_oi=delta_oi,
                funding_rate=fr,
                squeeze_type=squeeze_type,
                squeeze_fuel=squeeze_fuel,
                delta_cvd_fut=delta_cvd_fut,
            )
            spot_quality = classify_spot_accumulation_quality(
                spot_cvd_slope_short=delta_cvd_spot,
                price_slope_short=delta_price_short,
                post_rally_context=bool(long_phase.get("debug", {}).get("post_rally_context", False)),
                price_reclaim_or_breakout=bool(long_phase.get("debug", {}).get("price_reclaim_or_breakout", False)),
            )
            long_phase.setdefault("debug", {}).update(spot_quality)

            short_phase = _calc_short_phase_state(
                fut_df,
                spot_df if spot_available else None,
                oi_hist,
                n=N,
                candles_24h=candles_24h,
                price_change_24h=price_change_24h,
                delta_price=delta_price,
                delta_price_short=delta_price_short,
                d_vwap=d_vwap,
                delta_oi=delta_oi,
                funding_rate=fr,
                squeeze_type=squeeze_type,
                vol_ratio=vol_ratio,
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
                vol_ratio      = vol_ratio,
                squeeze_type   = squeeze_type,
                allow_short_squeeze_boost = bool(long_phase.get("debug", {}).get("allow_short_squeeze_boost", False)),
                spot_state     = str(spot_quality.get("spot_state", "neutral_or_no_spot_accum")),
                spot_accum_score_cap = spot_quality.get("long_score_cap"),
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

            if long_phase["flags"]:
                for flag in long_phase["flags"]:
                    if flag not in flags:
                        flags.append(flag)
                contexts.extend(long_phase["contexts"])
                score = round(score * float(long_phase["score_mult"]), 1)
                if long_phase["score_cap"] is not None:
                    score = min(score, float(long_phase["score_cap"]))

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
            div_score_raw, div_flags_raw, div_contexts_raw = calc_bearish_divergence_score(
                delta_price    = delta_price,
                delta_cvd_spot = delta_cvd_spot,
                delta_cvd_fut  = delta_cvd_fut,
                delta_oi       = delta_oi,
                funding_rate   = fr,
                d_vwap         = d_vwap,
                vol_ratio      = vol_ratio,
                fr_velocity    = fr_velocity,
            )
            div_state = _calc_bearish_divergence_state(
                fut_df,
                n=N,
                delta_price=delta_price,
                delta_price_short=delta_price_short,
                d_vwap=d_vwap,
                delta_cvd_spot=delta_cvd_spot,
                delta_cvd_fut=delta_cvd_fut,
                div_score=div_score_raw,
                div_flags=div_flags_raw,
                short_execution=short_execution,
            )
            div_score = float(div_state["score_effective"])
            div_flags = list(div_flags_raw)
            for flag in div_state["flags"]:
                if flag not in div_flags:
                    div_flags.append(flag)
            div_contexts = list(div_contexts_raw) + list(div_state["contexts"])

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
                divergence_type   = div_state["divergence_type"],
                divergence_status = div_state["divergence_status"],
                price_structure_state = div_state["price_structure_state"],
                broke_recent_swing_low = div_state["broke_recent_swing_low"],
                absorption_risk   = div_state["absorption_risk"],
                divergence_age_candles = div_state["divergence_age_candles"],
                short_score_cap_reason = div_state["short_score_cap_reason"],
                short_rejection_score = short_execution["short_rejection_score"],
                failed_breakout       = short_execution["failed_breakout"],
                last_candle_bearish   = short_execution["last_candle_bearish"],
                last_close_position   = short_execution["last_close_position"],
                upper_wick_pct        = short_execution["upper_wick_pct"],
                near_24h_high         = short_execution["near_24h_high"],
                spot_absorption_risk  = bool(spot_quality.get("spot_absorption_risk", False)),
            )

            if short_score > 0 and short_phase["flags"]:
                for flag in short_phase["flags"]:
                    if flag not in short_flags:
                        short_flags.append(flag)
                short_contexts.extend(short_phase["contexts"])
                short_score = round(short_score * float(short_phase["score_mult"]), 1)
                if short_phase["score_cap"] is not None:
                    short_score = min(short_score, float(short_phase["score_cap"]))

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

            long_debug = long_phase.get("debug", {})
            long_debug_str = ""
            if squeeze_type == "short" or long_phase["state"] != "neutral":
                long_debug_str = (
                    f" │ SqDn:{long_debug.get('downtrend_context', False)}"
                    f" Reclaim:{long_debug.get('price_reclaim_or_breakout', False)}"
                    f" Sq:{long_debug.get('short_squeeze_setup_score', 0.0):.0f}/"
                    f"{long_debug.get('short_squeeze_confirmation_score', 0.0):.0f}"
                    f" Boost:{long_debug.get('allow_short_squeeze_boost', False)}"
                    f" SVeto:{','.join(long_debug.get('short_squeeze_veto_reasons', [])) or '-'}"
                    f" Spot:{long_debug.get('spot_state', 'neutral_or_no_spot_accum')}"
                    f" Resp:{long_debug.get('price_response_to_spot_cvd', 'neutral')}"
                    f" LCap:{long_debug.get('long_score_cap_reason') or '-'}"
                )

            short_debug = short_phase.get("debug", {})
            short_debug_str = ""
            if short_score > 0 or short_phase["state"] != "neutral":
                short_debug_str = (
                    f" │ CVDdir:{short_debug.get('cvd_confluence_direction', 'mixed_neutral')}"
                    f" Div:{short_debug.get('spot_fut_divergence_label') or '-'}"
                    f" OIph:{short_debug.get('oi_phase', 'neutral')}"
                    f" BExp:{short_debug.get('bear_expansion_phase', 'neutral')}"
                    f" DropATR:{short_debug.get('recent_drop_atr', 0.0):.2f}"
                    f" VWAPz:{short_debug.get('distance_from_vwap_zscore', 0.0):.2f}"
                    f" Veto:{','.join(short_debug.get('short_veto_reasons', [])) or '-'}"
                )
            div_debug_str = ""
            if div_score_raw > 0:
                div_debug_str = (
                    f" │ DivStat:{div_state['divergence_status']}"
                    f" Type:{div_state['divergence_type']}"
                    f" Struct:{div_state['price_structure_state']}"
                    f" Age:{div_state['divergence_age_candles']}"
                    f" Cap:{div_state['short_score_cap_reason'] or '-'}"
                )

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
                f"{flags_str}{long_debug_str}{short_debug_str}{div_debug_str}"
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
                "long_phase_state":  long_phase["state"],
                "spot_cvd_phase":    long_phase["spot_cvd"],
                "fut_cvd_phase":     long_phase["fut_cvd"],
                "short_squeeze_setup_score": long_debug.get("short_squeeze_setup_score", 0.0),
                "short_squeeze_confirmation_score": long_debug.get("short_squeeze_confirmation_score", 0.0),
                "short_squeeze_veto_reasons": long_debug.get("short_squeeze_veto_reasons", []),
                "allow_short_squeeze_boost": long_debug.get("allow_short_squeeze_boost", False),
                "downtrend_context": long_debug.get("downtrend_context", False),
                "post_rally_context": long_debug.get("post_rally_context", False),
                "price_reclaim_or_breakout": long_debug.get("price_reclaim_or_breakout", False),
                "spot_state": long_debug.get("spot_state", "neutral_or_no_spot_accum"),
                "price_response_to_spot_cvd": long_debug.get("price_response_to_spot_cvd", "neutral"),
                "spot_absorption_risk": long_debug.get("spot_absorption_risk", False),
                "long_score_cap_reason": long_debug.get("long_score_cap_reason", ""),
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
                "div_score_raw": div_score_raw,
                "div_score":     div_score,
                "div_flags":     div_flags,
                "div_contexts":  div_contexts,
                "divergence_status": div_state["divergence_status"],
                "divergence_type": div_state["divergence_type"],
                "price_structure_state": div_state["price_structure_state"],
                "recent_swing_low": div_state["recent_swing_low"],
                "recent_swing_high": div_state["recent_swing_high"],
                "broke_recent_swing_low": div_state["broke_recent_swing_low"],
                "absorption_risk": div_state["absorption_risk"],
                "divergence_age_candles": div_state["divergence_age_candles"],
                "short_score_cap_reason": div_state["short_score_cap_reason"],
                # ── integrated short branch ────────────────────────────
                "short_score":    short_score,
                "short_flags":    short_flags,
                "short_contexts": short_contexts,
                "short_regime_reasons": r.get("short_regime_reasons", []),
                "short_setups":   short_setups,
                "short_targets":  short_targets,
                "short_deriv_state": short_deriv_state,
                "short_phase_state": short_phase["state"],
                "short_phase_flags": short_phase["flags"],
                "short_phase_contexts": short_phase["contexts"],
                "short_spot_cvd_phase": short_phase["spot_cvd"],
                "short_fut_cvd_phase": short_phase["fut_cvd"],
                "short_oi_phase": short_phase["oi"],
                "spot_cvd_slope_short": short_debug.get("spot_cvd_slope_short", 0.0),
                "fut_cvd_slope_short": short_debug.get("fut_cvd_slope_short", 0.0),
                "cvd_confluence_direction": short_debug.get("cvd_confluence_direction", "mixed_neutral"),
                "spot_fut_divergence_label": short_debug.get("spot_fut_divergence_label", ""),
                "recent_drop_atr": short_debug.get("recent_drop_atr", 0.0),
                "distance_from_vwap_zscore": short_debug.get("distance_from_vwap_zscore", 0.0),
                "oi_phase": short_debug.get("oi_phase", "neutral"),
                "bear_expansion_phase": short_debug.get("bear_expansion_phase", "neutral"),
                "short_veto_reasons": short_debug.get("short_veto_reasons", []),
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

        if short_base <= 0:
            r["short_score_regime_adj"] = 0.0
            r["short_regime_reasons"] = []
            continue

        short_adj, short_reasons = _calc_short_regime_adjustment(
            r,
            regime_ctx,
            _pre_euphoric_guard,
        )
        r["short_score_regime_adj"] = round(max(0.0, min(100.0, short_base + short_adj)), 1)
        r["short_regime_reasons"] = short_reasons

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
        "top_reversal_short", "bearish_divergence_watch",
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
        and (
            "CONFIRMED_SHORT_SQUEEZE" in r.get("flags", [])
            or not bool(set(r.get("flags", [])) & {
                "POST_RALLY_DEMAND_PAUSE",
                "POST_RALLY_ROLLOVER",
                "BEARISH_CVD_ALIGNMENT",
                "STALE_SPOT_ACCUM",
                "SQUEEZE_WATCH",
                "LONG_EXHAUSTION_RISK",
                "LONG_THESIS_INVALID",
            })
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
            f"flags={','.join(top['flags'])} "
            f"phase={top.get('long_phase_state', 'neutral')}"
        )
    if short_alerts:
        top_s = short_alerts[0]
        log.info(
            f"  🏆 Short tertinggi: {top_s['symbol']} "
            f"score={top_s.get('short_score_regime_adj', top_s.get('short_score', 0.0)):.1f} "
            f"setups={','.join(top_s.get('short_setups', []))} "
            f"state={top_s.get('short_deriv_state', 'neutral')} "
            f"phase={top_s.get('short_phase_state', 'neutral')}"
        )
    log.info("  " + "─" * 115)

    if total_alerts == 0 and _should_send_no_signal_diagnostic():
        _mark_no_signal_diagnostic_sent()
        diag_msg = _build_no_signal_diagnostic_message(
            cycle_label=cycle_label,
            results=results,
            skipped=skipped,
            errors=errors,
            elapsed_sec=t_elapsed,
            regime_ctx=regime_ctx,
            breadth_ctx=breadth_ctx,
            base_long_threshold=_base_long_threshold,
            base_short_threshold=_base_short_threshold,
            long_threshold=_long_threshold,
            short_threshold=_short_threshold,
            raw_long_count=len(_raw_long_candidates),
            allowed_long_count=len(_setup_allowed_long_candidates),
            final_long_count=len(alerts),
            raw_short_count=len(_raw_short_candidates),
            allowed_short_count=len(_setup_allowed_short_candidates),
            alert_short_count=len(_quality_alert_short_candidates),
            watch_short_count=len(_quality_watch_short_candidates),
            blocked_short_count=len(_quality_blocked_short_candidates),
            short_gate_reason_counts=_short_gate_reason_counts,
        )
        ok = await send_telegram(session, diag_msg)
        log.info(
            f"  🧭 No-signal diagnostic Telegram → "
            f"{'✓ terkirim' if ok else '✗ gagal'}"
        )

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
