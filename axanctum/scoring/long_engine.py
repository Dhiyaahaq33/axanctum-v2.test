from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import CONFIG


def classify_short_squeeze_quality(
    *,
    squeeze_type: str,
    squeeze_fuel: float,
    funding_rate: float,
    delta_oi: float,
    delta_price: float,
    price_change_24h: float,
    delta_price_short: float,
    price_reclaim_or_breakout: bool,
    post_rally_context: bool,
    downtrend_context_hint: bool = False,
) -> Dict[str, object]:
    """Nilai short squeeze agar lebih responsif di downtrend, bukan di pucuk."""
    setup_score = 0.0
    confirmation_score = 0.0
    veto_reasons: List[str] = []

    downtrend_context = bool(
        downtrend_context_hint
        or price_change_24h <= -3.0
        or delta_price <= -0.5
        or delta_price_short <= -0.2
    )
    top_context = bool(post_rally_context and not downtrend_context)

    if squeeze_type == "short":
        setup_score += 18.0
    if downtrend_context and squeeze_type == "short":
        setup_score += 14.0
    if squeeze_fuel >= 45.0:
        setup_score += min(squeeze_fuel * 0.22, 18.0)
    if delta_oi <= -1.5:
        setup_score += 20.0
    if funding_rate < -0.0003:
        setup_score += 25.0
    elif funding_rate < 0:
        setup_score += 14.0
    elif top_context:
        veto_reasons.append("positive_funding_not_short_squeeze_fuel")

    if price_reclaim_or_breakout:
        confirmation_score += 45.0
        if downtrend_context:
            confirmation_score += 10.0
    elif downtrend_context and delta_price_short > 0:
        confirmation_score += 20.0
    else:
        veto_reasons.append("no_price_reclaim_or_breakout")

    if delta_oi <= -1.5:
        confirmation_score += 15.0
    if funding_rate < 0:
        confirmation_score += 15.0
    if top_context and delta_price_short <= 0:
        veto_reasons.append("top_context_without_reclaim")
    if top_context and funding_rate > 0:
        veto_reasons.append("top_context_positive_funding")

    setup_threshold = 42.0 if downtrend_context else 50.0
    confirmation_threshold = 45.0 if downtrend_context else 55.0
    allow_boost = bool(
        squeeze_type == "short"
        and setup_score >= setup_threshold
        and confirmation_score >= confirmation_threshold
        and not veto_reasons
    )

    return {
        "short_squeeze_setup_score": round(setup_score, 1),
        "short_squeeze_confirmation_score": round(confirmation_score, 1),
        "short_squeeze_veto_reasons": veto_reasons,
        "allow_short_squeeze_boost": allow_boost,
        "downtrend_context": downtrend_context,
    }


def classify_spot_accumulation_quality(
    *,
    spot_cvd_slope_short: float,
    price_slope_short: float,
    post_rally_context: bool,
    price_reclaim_or_breakout: bool,
) -> Dict[str, object]:
    """Pisahkan spot accumulation bersih dari spot buy absorption."""
    bullish_cvd = spot_cvd_slope_short >= 1.2
    price_confirming = price_slope_short >= 0.20 or price_reclaim_or_breakout
    price_not_responding = price_slope_short <= 0.10 and not price_reclaim_or_breakout

    if bullish_cvd and price_confirming:
        return {
            "spot_state": "clean_spot_accumulation",
            "price_response_to_spot_cvd": "confirming",
            "spot_absorption_risk": False,
            "long_score_cap": None,
            "long_score_cap_reason": "",
        }

    if bullish_cvd and price_not_responding:
        return {
            "spot_state": "spot_buy_absorption",
            "price_response_to_spot_cvd": "flat_or_down_after_spot_bid",
            "spot_absorption_risk": True,
            "long_score_cap": 58.0 if post_rally_context else 64.0,
            "long_score_cap_reason": (
                "spot_buy_absorption_after_rally"
                if post_rally_context
                else "spot_buy_absorption_without_price_response"
            ),
        }

    return {
        "spot_state": "neutral_or_no_spot_accum",
        "price_response_to_spot_cvd": "neutral",
        "spot_absorption_risk": False,
        "long_score_cap": None,
        "long_score_cap_reason": "",
    }


def calculate_market_score(
    d_vwap:         float,
    delta_cvd_spot: float,
    delta_cvd_fut:  float,
    delta_oi:       float,
    delta_price:    float,
    funding_rate:   float,
    basis_pct:      float = 0.0,
    fr_velocity:    float = 0.0,
    vol_ratio:      float = 1.0,
    squeeze_type:   str   = "none",
    allow_short_squeeze_boost: bool = True,
    spot_state: str = "neutral_or_no_spot_accum",
    spot_accum_score_cap: float | None = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Interdependency Scoring Matrix — Layer 1 Revisi.

    PERUBAHAN UTAMA dari versi sebelumnya:
      Sebelumnya: spot_dominance menghukum futures CVD
      Sekarang  : Combined Demand Strength — keduanya bisa valid

    Layer 1 — Combined Demand Strength
      Spot dan Futures CVD diukur bersama dengan bobot berbeda:
        Spot  : bobot 1.3 (lebih organik, tetap diutamakan)
        Futures: bobot 1.0 (valid, tidak dihukum kecuali leverage bermasalah)

      Confluence Bonus: kalau keduanya positif dan searah,
      sinyal mendapat booster ekstra karena dua market mengkonfirmasi.

    Layer 2 — Context Quality Multipliers (tidak berubah)
      VWAP, OI, FR sebagai pengali kepercayaan.
      OI dan FR yang menentukan apakah futures CVD "sehat" atau "overdrive".

    Layer 3 — Final Score (tidak berubah)
      quality = vwap_mod × oi_mod × fr_mod
      final   = clamp(raw × quality, 0, 100)
    """
    flags:    List[str] = []
    contexts: List[str] = []
    eps = 1e-6

    # ══════════════════════════════════════════════════════════════════
    # LAYER 1 — Combined Demand Strength
    # ══════════════════════════════════════════════════════════════════

    # Normalisasi masing-masing CVD ke rentang -1 sampai +1
    total_mag  = abs(delta_cvd_spot) + abs(delta_cvd_fut) + eps
    spot_norm  = delta_cvd_spot / total_mag
    fut_norm   = delta_cvd_fut  / total_mag

    # Weighted demand: spot lebih diutamakan tapi futures tidak dihukum
    WEIGHT_SPOT = 1.3
    WEIGHT_FUT  = 1.0
    combined_demand = (
        (spot_norm * WEIGHT_SPOT + fut_norm * WEIGHT_FUT)
        / (WEIGHT_SPOT + WEIGHT_FUT)
    )
    # range: -1.0 (keduanya bearish total) sampai +1.0 (keduanya bullish total)
    spot_buy_absorption = spot_state == "spot_buy_absorption"

    # ── Confluence Bonus ──────────────────────────────────────────────
    # Kalau spot DAN futures keduanya bergerak searah ke atas,
    # ini adalah konfirmasi terkuat — dua market bicara hal yang sama
    # Threshold minimum agar confluence bonus aktif
    # CVD keduanya harus cukup signifikan, bukan sekadar positif
    _cvd_min_confluence = 2.0   # minimal 2% normalized CVD

    if (
        delta_cvd_spot > _cvd_min_confluence
        and delta_cvd_fut > _cvd_min_confluence
        and not spot_buy_absorption
    ):
        conf_strength    = min(abs(spot_norm) * abs(fut_norm) * 4, 1.0)
        confluence_bonus = 1.0 + conf_strength * 0.18
        flags.append("CONFLUENCE")
        contexts.append(
            f"🤝 Konfluensi Spot+Futures — "
            f"kedua market bullish (bonus ×{confluence_bonus:.2f})"
        )
    elif delta_cvd_spot > 0 and delta_cvd_fut > 0:
        # Keduanya positif tapi lemah — bonus kecil, tidak ada flag
        confluence_bonus = 0.92 if spot_buy_absorption else 1.05
    elif delta_cvd_spot < 0 and delta_cvd_fut < 0:
        confluence_bonus = 0.85
    else:
        confluence_bonus = 1.0

    # ── Basis Modifier ────────────────────────────────────────────────
    # Basis menginformasikan konteks harga, bukan menghukum futures CVD
    if basis_pct > 0.3:
        # Futures jauh lebih mahal → kemungkinan leverage overdrive
        # Nanti OI + FR yang akan menentukan hukumannya
        basis_mod = 0.88
        contexts.append(f"⚠️ Basis +{basis_pct:.3f}% — Futures premium")
    elif basis_pct > 0.1:
        basis_mod = 0.94
    elif basis_pct < -0.05:
        # Spot memimpin futures → organik, perkuat sinyal
        basis_mod = 1.08
    else:
        basis_mod = 1.0   # netral

    # ── Konfirmasi arah harga ─────────────────────────────────────────
    if delta_price > 1.5:
        price_conf = 1.2
    elif delta_price > 0:
        price_conf = 1.0
    elif delta_price > -1.0:
        price_conf = 0.7
    else:
        price_conf = 0.4

    # Raw score (sebelum quality modifier Layer 2)
    raw_score = combined_demand * 60.0 * price_conf * confluence_bonus * basis_mod

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2 — Context Quality Multipliers
    # ══════════════════════════════════════════════════════════════════

    # ── VWAP Modifier — Adaptif berbasis volume ───────────────────────
    # Threshold overextended disesuaikan dengan kondisi volume:
    # Volume tinggi → harga "berhak" jauh dari VWAP (threshold longgar)
    # Volume tipis  → harga jauh dari VWAP lebih mencurigakan (threshold ketat)
    # vol_ratio dipass dari luar melalui parameter (default 1.0 jika tidak ada)

    _vol_ratio_for_vwap = vol_ratio
    # Threshold dasar dari CONFIG, disesuaikan secara proporsional
    _overext  = CONFIG["VWAP_OVEREXT_PCT"]   # 5.0%
    _danger   = CONFIG["VWAP_DANGER_PCT"]    # 8.0%
    _sideways = CONFIG["VWAP_SIDEWAYS_PCT"]  # 1.5%

    # =========================
    # VWAP ADAPTIVE THRESHOLD (BASED ON VOLUME)
    # =========================
    if _vol_ratio_for_vwap > 2.0:
        _overext *= 0.8     # lebih ketat (volume tinggi)
        _danger  *= 0.85

    elif _vol_ratio_for_vwap < 0.7:
        _overext *= 1.4     # lebih longgar (volume tipis)
        _danger  *= 1.3

    # Clamp biar nggak terlalu ekstrem
    _overext = max(2.5, min(_overext, 10))
    _danger  = max(4.0, min(_danger, 12))

    abs_vwap = abs(d_vwap)
    if d_vwap < -_overext:
        # Harga jauh di bawah VWAP → diskon dalam, akumulasi sangat valid
        vwap_mod = 1.45
    elif d_vwap < 0:
        vwap_mod = 1.2
    elif abs_vwap <= _sideways:
        vwap_mod = 1.1
    elif d_vwap <= _overext:
        # Sedikit di atas VWAP → mulai hati-hati
        vwap_mod = 0.85
    elif d_vwap <= _danger:
        # Overextended tapi belum bahaya (5-8%)
        vwap_mod = 0.5
        flags.append("E_DISTRIBUTION")
        contexts.append(f"E: Overextended +{d_vwap:.1f}% dari VWAP 🪤")
    else:
        # > 8% — sangat overextended → bahaya
        vwap_mod = 0.2
        flags.append("E_DISTRIBUTION")
        contexts.append(f"E: Sangat Overextended +{d_vwap:.1f}% 🪤 BAHAYA")

    # ── OI Modifier ───────────────────────────────────────────────────
   # ── OI Modifier — Goldilocks Range ───────────────────────────────
    # Squeeze dideteksi berdasarkan Goldilocks range, bukan threshold tunggal
    #
    # delta_oi < -OI_NOISE_PCT (< -1.5%)    → noise, tidak ada sinyal squeeze
    # -OI_IGNITION_MAX < delta_oi < -OI_IGNITION_MIN (1.5-4% turun) → IGNITION
    # delta_oi < -OI_EXHAUSTED_PCT (< -8%)  → squeeze hampir exhausted
    # delta_oi > OI_PARABOLIC_PCT (> 15%)   → overleverage
    _oi_noise    = CONFIG["OI_NOISE_PCT"]      # 1.5
    _oi_ign_min  = CONFIG["OI_IGNITION_MIN"]   # 1.5
    _oi_ign_max  = CONFIG["OI_IGNITION_MAX"]   # 4.0
    _oi_exhaust  = CONFIG["OI_EXHAUSTED_PCT"]  # 8.0
    _oi_parab    = CONFIG["OI_PARABOLIC_PCT"]  # 15.0

    if delta_oi < -_oi_exhaust:
        if squeeze_type == "long":
            oi_mod = 0.7
            contexts.append(f"⚠️ Long Squeeze Exhausted — OI {delta_oi:+.1f}% turun dalam")
        elif allow_short_squeeze_boost:
            oi_mod = 1.1
            flags.append("C_SQUEEZE")
            contexts.append(f"C: Squeeze Exhausted — OI {delta_oi:+.1f}% ⚠️")
        else:
            oi_mod = 0.95
            contexts.append(f"🔫 OI {delta_oi:+.1f}% turun — squeeze belum confirmed")
    elif delta_oi < -_oi_ign_max:
        if squeeze_type == "long":
            oi_mod = 0.55
            contexts.append(f"💀 Long Squeeze Aktif — OI {delta_oi:+.1f}% turun, harga ikut turun")
        elif allow_short_squeeze_boost:
            oi_mod = 1.35
            if "C_SQUEEZE" not in flags:
                flags.append("C_SQUEEZE")
                contexts.append(f"C: Squeeze Aktif — OI {delta_oi:+.1f}% 🔫")
        else:
            oi_mod = 1.0
            contexts.append(f"🔫 OI {delta_oi:+.1f}% turun — squeeze watch")
    elif delta_oi < -_oi_ign_min:
        if squeeze_type == "long":
            oi_mod = 0.6
            contexts.append(f"💀 Long Squeeze Ignition — OI {delta_oi:+.1f}% ⚠️")
        elif allow_short_squeeze_boost:
            oi_mod = 1.45
            if "C_SQUEEZE" not in flags:
                flags.append("C_SQUEEZE")
                contexts.append(f"C: Squeeze Ignition — OI {delta_oi:+.1f}% 🔫🔥")
        else:
            oi_mod = 1.0
            contexts.append(f"🔫 OI {delta_oi:+.1f}% turun — squeeze watch")
    elif delta_oi < 0:
        oi_mod = 1.0
    elif delta_oi <= 4.0:
        oi_mod = 1.0
    elif delta_oi <= _oi_parab:
        oi_mod = 0.75
    else:
        oi_mod = 0.3
        flags.append("D_EXHAUSTION")
        contexts.append(f"D: OI Parabolik {delta_oi:+.1f}% 💀 Overleverage")

    # ── Funding Rate Modifier ─────────────────────────────────────────
    fr = funding_rate
    if fr < -0.0010:
        fr_mod = 1.4 if allow_short_squeeze_boost else 1.10
        if allow_short_squeeze_boost and "C_SQUEEZE" not in flags:
            flags.append("C_SQUEEZE")
        contexts.append(
            f"C: FR Sangat Negatif ({fr*100:+.4f}%) 🔫"
            if allow_short_squeeze_boost
            else f"🔫 FR sangat negatif ({fr*100:+.4f}%) — squeeze watch"
        )
    elif fr < -0.0003:
        fr_mod = 1.2
    elif fr < CONFIG["FR_NEGATIVE"]:
        fr_mod = 1.08
    elif fr <= CONFIG["FR_OVERLEVERAGE"]:
        fr_mod = 1.0
    elif fr <= 0.0005:
        fr_mod = 0.82
    elif fr <= 0.0010:
        fr_mod = 0.6
        if "D_EXHAUSTION" not in flags:
            flags.append("D_EXHAUSTION")
            contexts.append(f"D: FR Tinggi ({fr*100:+.4f}%) 💀")
    else:
        fr_mod = 0.35
        if "D_EXHAUSTION" not in flags:
            flags.append("D_EXHAUSTION")
        contexts.append(f"D: FR Ekstrem ({fr*100:+.4f}%) 💀 BAHAYA")

    # ── FR Velocity Modifier ──────────────────────────────────────────
    if fr_velocity > 0.0003:
        fr_mod *= 0.55
        contexts.append(
            f"💥 FR Velocity +{fr_velocity*100:.4f}%/siklus — "
            f"FOMO akut, potensi blow-off top"
        )
    elif fr_velocity > 0.0001:
        fr_mod *= 0.80
    elif fr_velocity < -0.0003:
        fr_mod *= 1.20
        contexts.append(
            f"🔫 FR Velocity {fr_velocity*100:.4f}%/siklus — "
            f"Shorts menumpuk, squeeze fuel bertambah"
        )
    elif fr_velocity < -0.0001:
        fr_mod *= 1.08

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3 — Final Score
    # ══════════════════════════════════════════════════════════════════
    quality = vwap_mod * oi_mod * fr_mod

    # ── Capped Multiplicative ─────────────────────────────────────────
    # Quality boleh menekan skor ke bawah tanpa batas
    # Tapi tidak boleh mengangkat raw_score lebih dari 50%
    # Ini mencegah sinyal lemah meledak jadi 90+ hanya karena
    # semua multiplier kebetulan bagus
    quality_up_cap = 1.50
    if quality > quality_up_cap:
        quality = quality_up_cap
    quality = max(0.08, quality)

    final_score = max(0.0, min(100.0, raw_score * quality))

    if spot_buy_absorption:
        final_score = min(final_score * 0.72, spot_accum_score_cap or 64.0)
        flags.append("SPOT_BUY_ABSORPTION")
        if "BEARISH_DIVERGENCE_WATCH" not in flags:
            flags.append("BEARISH_DIVERGENCE_WATCH")
        if "UPSIDE_EXHAUSTION_RISK" not in flags:
            flags.append("UPSIDE_EXHAUSTION_RISK")
        contexts.append(
            "🧲 Spot buy absorption: Spot CVD naik tapi harga belum ikut respons"
        )

    # ── Identifikasi skenario utama ───────────────────────────────────
    if combined_demand > 0 and raw_score > 0:
        if spot_norm > 0.25 and spot_norm >= fut_norm and not spot_buy_absorption:
            # Spot memimpin atau setara
            flags.insert(0, "A")
            contexts.insert(
                0,
                f"A: Spot Dominan {spot_norm*100:.0f}% "
                f"(quality×{quality:.2f} → {final_score:.1f}pts)"
            )
        elif fut_norm > 0.25:
            # Futures memimpin tapi demand sehat (OI dan FR yang akan menentukan)
            flags.insert(0, "A_FUTURES")
            contexts.insert(
                0,
                f"A: Futures Memimpin {fut_norm*100:.0f}% "
                f"(quality×{quality:.2f} → {final_score:.1f}pts)"
            )
    elif combined_demand < -0.25 and delta_price > 0:
        flags.append("B_SPECULATIVE")
        contexts.append(
            f"B: Demand Lemah — harga naik tanpa CVD ({combined_demand*100:.0f}%)"
        )

    return round(final_score, 1), flags, contexts
