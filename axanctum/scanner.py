from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

from .config import CONFIG, OI_PERIOD_MAP, TF_MINUTES, TZ_WIB
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
    calc_long_squeeze_score,
)
from .state import _FR_HISTORY_4H, _FR_VELOCITY_INTERVAL


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
            tf_min = TF_MINUTES.get(CONFIG["TIMEFRAME"], 60)
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
            marker    = " ★" if score >= CONFIG["MIN_SCORE"] else "  "
            log.info(
                f"{marker} {symbol:<18} {score:>6.1f} {grade:>3} │ "
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
    }
    _long_setups_ok  = bool(_LONG_SETUP_TYPES  & set(regime_ctx.allowed_setups))
    _short_setups_ok = (
        bool(_SHORT_SETUP_TYPES & set(regime_ctx.allowed_setups))
        or _pre_euphoric_guard
    )

    breadth_ctx = calculate_market_breadth(
        results=results,
        long_threshold=_long_threshold,
        short_threshold=_short_threshold,
    )

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
    ]
    _setup_blocked_long = len(_raw_long_candidates) - len(_setup_allowed_long_candidates)
    if _raw_long_candidates:
        log.info(
            f"[Regime] Long setup gate — raw={len(_raw_long_candidates)} | "
            f"allowed={len(_setup_allowed_long_candidates)} | "
            f"blocked={_setup_blocked_long}"
        )

    _ls_candidates = sum(
        1 for r in results
        if r.get("ls_score", 0) >= _short_threshold
        and r.get("squeeze_type") in ("long", "long_exhausted")
    )
    _dist_candidates = sum(1 for r in results if r.get("dist_score", 0) >= _short_threshold)
    _div_candidates = sum(
        1 for r in results
        if r.get("div_score_adj", 0) >= _short_threshold
        and r.get("ls_score", 0) < CONFIG["MIN_SCORE"]
        and r.get("dist_score", 0) < CONFIG["MIN_SCORE"]
    )
    log.info(
        f"[Regime] Short candidates — LS={_ls_candidates} | "
        f"DIST={_dist_candidates} | DIV={_div_candidates}"
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

    t_elapsed = time.time() - t_start
    log.info("═" * 60)
    log.info(
        f"  ✅ Scan selesai — {len(alerts)} alert / {len(results)} hasil "
        f"/ {skipped} skip / {errors} error | waktu: {t_elapsed:.1f}s"
    )
    log.info("═" * 60)

    # Ringkasan akhir setelah semua koin selesai
    log.info("  " + "─" * 115)
    log.info(
        f"  📊 Selesai — {len(results)} koin discan │ "
        f"{len(alerts)} lolos (★) │ "
        f"{skipped} skip │ {errors} error │ {t_elapsed:.1f}s"
    )
    if alerts:
        top = alerts[0]
        log.info(
            f"  🏆 Tertinggi: {top['symbol']} "
            f"score={top['score']:.1f} [{top['grade']}] "
            f"flags={','.join(top['flags'])}"
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

    # Kirim alert SHORT (long squeeze)
    short_alerts = [
        r for r in results
        if r.get("ls_score", 0) >= _short_threshold
        and r.get("squeeze_type") in ("long", "long_exhausted")
        and _short_setups_ok
    ]
    short_alerts.sort(key=lambda x: x.get("ls_score", 0), reverse=True)

    for r in short_alerts:
        sym = r["symbol"]
        if _is_on_cooldown(sym, "SHORT"):
            log.info(f"  ⏳ {sym:<18} [SHORT-LS] cooldown aktif — skip")
            continue
        msg = build_telegram_message(r, signal_type="SHORT", regime_ctx=regime_ctx)
        ok  = await send_telegram(session, msg)
        if ok:
            _mark_sent(sym, "SHORT")
        _sq_type = r.get("squeeze_type", "none")
        log.info(
            f"  📤 {sym:<18} [{_sq_type}] "
            f"ls_score={r['ls_score']:5.1f} → {'✓ terkirim' if ok else '✗ gagal'}"
        )
        # ── Kirim ke Dashboard ──
        tgt = r.get("ls_targets", {})
        send_signal_to_dashboard(
            symbol=sym, direction="SHORT", entry=r["price"],
            tp=tgt.get("target1") or round(r["price"] * 0.98, 8),
            sl=tgt.get("invalidasi") or round(r["price"] * 1.005, 8),
            grade="A+" if r.get("ls_score", 0) >= 85 else "B",
            leverage=5,
        )
        await asyncio.sleep(CONFIG["MSG_DELAY"])

    # Kirim DISTRIBUTION SHORT alerts
    dist_alerts = [
        r for r in results
        if r.get("dist_score", 0) >= _short_threshold
        and _short_setups_ok
    ]
    dist_alerts.sort(key=lambda x: x.get("dist_score", 0), reverse=True)

    for r in dist_alerts:
        sym = r["symbol"]
        if _is_on_cooldown(sym, "SHORT"):
            log.info(f"  ⏳ {sym:<18} [DIST SHORT] cooldown aktif — skip")
            continue
        # Buat pesan distribution dengan data yang sudah ada
        dist_r = dict(r)
        dist_r["ls_score"]    = r["dist_score"]
        dist_r["ls_flags"]    = r["dist_flags"]
        dist_r["ls_contexts"] = r["dist_contexts"]
        dist_r["squeeze_type"] = "distribution"
        # Hitung target distribution: ke bawah VWAP
        dist_r["ls_targets"] = {
            "direction":  "SHORT — Distribusi Aktif",
            "target1":    round(r["vwap"], 6),
            "target2":    round(r["vwap"] * 0.985, 6),
            "invalidasi": round(r["price"] * 1.015, 6),
        }
        msg = build_telegram_message(dist_r, signal_type="SHORT", regime_ctx=regime_ctx)
        ok  = await send_telegram(session, msg)
        if ok:
            _mark_sent(sym, "SHORT")
        log.info(
            f"  📤 {sym:<18} [DIST SHORT] "
            f"dist_score={r['dist_score']:5.1f} → "
            f"{'✓ terkirim' if ok else '✗ gagal'}"
        )
        send_signal_to_dashboard(
            symbol=sym, direction="SHORT", entry=r["price"],
            tp=dist_r["ls_targets"]["target1"],
            sl=dist_r["ls_targets"]["invalidasi"],
            grade="A+" if r.get("dist_score", 0) >= 85 else "A",
            leverage=5,
        )
        await asyncio.sleep(CONFIG["MSG_DELAY"])


    # ── DIV SHORT ALERTS (Short Engine v2 — Exhaustion) ────────────────
    div_alerts = [
        r for r in results
        if r.get("div_score_adj", 0) >= _short_threshold
        and _short_setups_ok
        # Tidak dobel dengan ls atau dist
        and r.get("ls_score", 0) < CONFIG["MIN_SCORE"]
        and r.get("dist_score", 0) < CONFIG["MIN_SCORE"]
    ]
    div_alerts.sort(key=lambda x: x.get("div_score", 0), reverse=True)

    for r in div_alerts:
        sym_full = r["symbol"]
        if _is_on_cooldown(sym_full, "SHORT"):
            log.info(f"  ⏳ {sym_full:<18} [DIV SHORT] cooldown aktif — skip")
            continue
        div_score  = r.get("div_score", 0)
        div_flags  = r.get("div_flags", [])
        div_ctxs   = r.get("div_contexts", [])
        pr         = r["price"]
        sym        = sym_full.replace("USDT", "")

        # Grade
        if   div_score >= 85: grade, g_label = "A+", "🔥 PRIME SHORT"
        elif div_score >= 75: grade, g_label = "A",  "⭐ HIGH SHORT"
        elif div_score >= 65: grade, g_label = "B+", "✅ SHORT SETUP"
        else:                 grade, g_label = "B",  "🔷 WATCH SHORT"

        # Target: TP ke bawah, SL ke atas
        tp_pct  = 0.020 if "DIV_VWAP_EXTREME" in div_flags else 0.015
        sl_pct  = 0.010 if "DIV_LEVERAGE_TRAP" in div_flags else 0.008
        tp_price = round(pr * (1 - tp_pct), 8)
        sl_price = round(pr * (1 + sl_pct), 8)

        # Jenis divergence untuk label
        div_type = []
        if "DIV_HIDDEN_DIST"   in div_flags: div_type.append("🎭 Hidden Dist")
        if "DIV_LEVERAGE_TRAP" in div_flags: div_type.append("💸 Leverage Trap")
        if "DIV_VWAP_EXTREME"  in div_flags: div_type.append("📏 VWAP Extreme")
        if "DIV_CROSS_MARKET"  in div_flags: div_type.append("⚔️ Cross-Market")
        div_str = "  │  ".join(div_type) if div_type else "Exhaustion"

        now = datetime.now(TZ_WIB).strftime("%d %b %Y  %H:%M WIB")
        sep = "─" * 34
        msg = "\n".join([
            f"🎯 {g_label} — EXHAUSTION SHORT",
            "",
            f"<b>📌 #{sym}USDT</b>  │  <code>${pr:,.5g}</code>",
            f"⏰ {now}",
            sep,
            f"<b>🏆 {div_score:.1f} / 100  [ {grade} ]  ▼ SHORT</b>",
            "",
            f"🧭 {div_str}",
            sep,
            f"📈 TP  :  <code>${tp_price:,.5g}</code>  ({-tp_pct*100:.1f}%)",
            f"🛑 SL  :  <code>${sl_price:,.5g}</code>  (+{sl_pct*100:.1f}%)",
            sep,
            *[f"  {c}" for c in div_ctxs[:3]],
        ])

        ok = await send_telegram(session, msg)
        if ok:
            _mark_sent(sym_full, "SHORT")
        log.info(
            f"  📤 {sym_full:<18} [DIV SHORT] "
            f"div_score={div_score:5.1f} → {'✓ terkirim' if ok else '✗ gagal'}"
        )
        send_signal_to_dashboard(
            symbol=sym_full, direction="SHORT", entry=pr,
            tp=tp_price, sl=sl_price, grade=grade, leverage=5,
        )
        await asyncio.sleep(CONFIG["MSG_DELAY"])
