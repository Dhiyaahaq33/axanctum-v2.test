from __future__ import annotations

from typing import List, Tuple


def _add_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _extend_unique(items: List[str], values: List[str]) -> None:
    for value in values:
        _add_unique(items, value)


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def _cap_short_score(
    *,
    score: float,
    primary_setup: str,
    price_change_24h: float,
    d_vwap: float,
    delta_price_short: float,
    delta_cvd_spot: float,
    delta_cvd_fut: float,
    delta_oi: float,
    funding_rate: float,
    vol_ratio: float,
    dist_score: float,
    div_score: float,
    ls_score: float,
) -> tuple[float, str | None]:
    """Batasi confidence score agar detector tidak otomatis menjadi PRIME."""
    both_bearish = delta_cvd_spot < -1.0 and delta_cvd_fut < -1.0
    strong_bear_flow = (
        delta_cvd_spot <= -2.5 and delta_cvd_fut <= -1.5
    ) or (delta_cvd_spot + delta_cvd_fut <= -6.0)
    funding_trapped_long = funding_rate > 0.0001 and price_change_24h <= 0
    oi_hot = delta_oi > 4.0
    late_drop = price_change_24h <= -14.0 and d_vwap <= -7.0
    extreme_late = price_change_24h <= -20.0 or d_vwap <= -11.0
    fresh_breakdown = -8.0 <= price_change_24h <= 2.0 and -5.0 <= d_vwap <= -0.4 and delta_price_short <= -0.8
    top_prime = (
        d_vwap >= 4.0
        and price_change_24h >= 5.0
        and delta_cvd_spot <= -2.0
        and delta_cvd_fut <= 1.0
        and (funding_rate > 0.0008 or oi_hot)
        and (dist_score >= 70.0 or div_score >= 60.0)
        and vol_ratio >= 1.0
    )

    cap = 88.0
    reason = None

    if primary_setup == "bear_continuation_short":
        cap = 86.0
        if strong_bear_flow and (funding_trapped_long or oi_hot or ls_score >= 60.0):
            cap = 89.0
        if late_drop:
            cap = min(cap, 82.0)
        if extreme_late:
            cap = min(cap, 78.0)
        if not both_bearish and ls_score < 60.0:
            cap = min(cap, 80.0)

    elif primary_setup == "breakdown_short":
        cap = 87.0
        if fresh_breakdown and strong_bear_flow and (funding_trapped_long or oi_hot):
            cap = 91.0
        if late_drop:
            cap = min(cap, 80.0)

    elif primary_setup == "long_squeeze_short":
        cap = 88.0
        if ls_score >= 70.0 and (funding_trapped_long or strong_bear_flow):
            cap = 92.0
        if extreme_late and not funding_trapped_long:
            cap = min(cap, 82.0)

    elif primary_setup == "distribution_short":
        cap = 84.0
        if d_vwap >= 2.0 and price_change_24h >= -4.0:
            cap = 89.0
        if top_prime:
            cap = 93.0
        if price_change_24h <= -10.0 and d_vwap <= -5.0:
            cap = min(cap, 80.0)

    elif primary_setup in ("exhaustion_after_pump_short", "top_reversal_short"):
        cap = 88.0
        if top_prime:
            cap = 94.0
        elif d_vwap >= 3.0 and price_change_24h >= 3.0 and (dist_score >= 55.0 or div_score >= 45.0):
            cap = 90.0

    elif primary_setup == "bearish_divergence_short":
        cap = 86.0
        if d_vwap >= 2.0 and price_change_24h >= -2.0 and div_score >= 75.0:
            cap = 91.0

    if score > cap:
        reason = f"score_cap_{primary_setup}_{cap:.0f}"
        score = cap

    return score, reason


def calc_long_squeeze_score(
    delta_price:  float,
    delta_oi:     float,
    funding_rate: float,
    delta_cvd_spot: float,
    delta_cvd_fut:  float,
    d_vwap:       float,
    squeeze_type: str,
    squeeze_stage: str,
    squeeze_fuel: float,
) -> Tuple[float, List[str], List[str]]:
    """
    Hitung skor sinyal SHORT dari kondisi Long Squeeze.

    Terpisah dari scoring bullish karena logika dan targetnya berbeda.
    Skor 0–100: semakin tinggi = long squeeze semakin kuat = SHORT semakin valid.

    Returns: (short_score, flags, contexts)
    """
    if squeeze_type not in ("long", "long_exhausted"):
        return 0.0, [], []

    flags:    List[str] = []
    contexts: List[str] = []

    # ── Base score dari kekuatan long squeeze ─────────────────────────
    # Fuel tersisa = seberapa jauh squeeze masih bisa berlanjut
    base = squeeze_fuel * 0.6   # max 60 poin dari fuel

    # ── CVD Konfirmasi ────────────────────────────────────────────────
    # CVD negatif mengkonfirmasi tekanan jual agresif
    both_negative = delta_cvd_spot < 0 and delta_cvd_fut < 0
    if both_negative:
        cvd_bonus = 15.0   # konfluensi bearish
        flags.append("CVD_CONFLUENCE_BEARISH")
        contexts.append("🔴 CVD Spot+Futures keduanya negatif — tekanan jual kuat")
    elif delta_cvd_fut < 0:
        cvd_bonus = 8.0
    else:
        cvd_bonus = 0.0

    # ── VWAP Context ──────────────────────────────────────────────────
    # Harga jauh di atas VWAP saat long squeeze = distribusi dari premium
    if d_vwap > 3.0:
        vwap_bonus = 12.0
        contexts.append(f"📏 Harga +{d_vwap:.1f}% di atas VWAP — distribusi dari premium")
    elif d_vwap > 1.0:
        vwap_bonus = 6.0
    elif d_vwap < 0:
        vwap_bonus = 0.0   # harga sudah di bawah VWAP, squeeze mungkin terlambat
        base *= 0.7
    else:
        vwap_bonus = 3.0

    # ── FR Modifier ───────────────────────────────────────────────────
    if funding_rate > 0.0010:
        fr_bonus = 10.0
        contexts.append(f"💀 FR {funding_rate*100:+.4f}% — banyak longs yang belum kena")
    elif funding_rate > 0.0005:
        fr_bonus = 6.0
    elif funding_rate > 0.0001:
        fr_bonus = 3.0
    else:
        fr_bonus = 0.0

    short_score = base + cvd_bonus + vwap_bonus + fr_bonus
    short_score = max(0.0, min(100.0, short_score))

    # ── Flags ──────────────────────────────────────────────────────────
    if squeeze_type == "long":
        flags.insert(0, "LONG_SQUEEZE")
        contexts.insert(0, f"💀 Long Squeeze {squeeze_stage.title()} — SHORT signal")
    elif squeeze_type == "long_exhausted":
        flags.insert(0, "LONG_SQUEEZE_EXHAUSTED")
        contexts.insert(0, "✅ Long Squeeze Exhausted — potensi reversal LONG")

    return round(short_score, 1), flags, contexts


def calc_distribution_score(
    delta_cvd_spot: float,
    delta_cvd_fut:  float,
    delta_oi:       float,
    funding_rate:   float,
    d_vwap:         float,
    delta_price:    float,
) -> Tuple[float, List[str], List[str]]:
    """
    Deteksi Distribusi Aktif — sinyal SHORT standalone.

    Kondisi yang terdeteksi:
      CVD keduanya negatif (Spot & Futures sama-sama jual agresif)
      + OI naik (posisi baru dibuka = longs baru masuk tapi sia-sia)
      + FR positif (market condong long = orang-orang belum sadar distribusi)
      + Harga turun atau stagnan

    Ini adalah kondisi distribusi aktif paling jelas:
    "Smart money jual agresif, retail masih beli dengan leverage"

    Berbeda dari Long Squeeze (yang butuh OI TURUN),
    ini adalah distribusi awal sebelum squeeze terjadi.

    Skor: 0–100 (semakin tinggi = distribusi semakin kuat = SHORT semakin valid)
    """
    flags:    List[str] = []
    contexts: List[str] = []

    # Syarat minimum: keduanya harus bearish CVD
    both_bearish = delta_cvd_spot < 0 and delta_cvd_fut < 0
    if not both_bearish:
        return 0.0, [], []

    # Tambahan: harga tidak boleh sedang naik kencang
    # (kalau naik kencang dengan CVD negatif, itu skenario lain)
    if delta_price > 2.0:
        return 0.0, [], []

    score = 0.0

    # ── Base: kekuatan bearish CVD ──────────────────────────────────────
    # Semakin negatif combined CVD → distribusi semakin kuat
    combined_cvd = (delta_cvd_spot * 1.0 + delta_cvd_fut * 1.3) / 2.3
    cvd_strength = min(abs(combined_cvd) / 10.0, 1.0)   # normalize 0–1
    base_score   = cvd_strength * 40.0                   # max 40 poin dari CVD
    score += base_score
    flags.append("DISTRIBUTION")
    contexts.append(
        f"📉 Distribusi Aktif: CVD Spot{delta_cvd_spot:+.1f}% "
        f"Fut{delta_cvd_fut:+.1f}% — keduanya jual agresif"
    )

    # ── OI naik + harga turun = longs baru masuk sia-sia ────────────────
    if delta_oi > 2.0 and delta_price <= 0:
        oi_bonus = min((delta_oi - 2.0) / 8.0, 1.0) * 20.0
        score += oi_bonus
        contexts.append(
            f"⚠️ OI +{delta_oi:.1f}% tapi harga {delta_price:+.2f}% — "
            f"longs baru masuk sia-sia"
        )

    # ── FR positif = banyak longs yang akan kena ────────────────────────
    if funding_rate > 0.0005:
        fr_bonus = 15.0
        flags.append("D_EXHAUSTION")
        contexts.append(f"💀 FR {funding_rate*100:+.4f}% — longs menumpuk")
        score += fr_bonus
    elif funding_rate > 0.0001:
        score += 8.0

    # ── VWAP: distribusi dari premium lebih berbahaya ────────────────────
    if d_vwap > 5.0:
        vwap_bonus = 15.0
        contexts.append(f"📏 Distribusi dari +{d_vwap:.1f}% di atas VWAP — premium tinggi")
        score += vwap_bonus
    elif d_vwap > 2.0:
        score += 8.0
    elif d_vwap < -2.0:
        # Distribusi di bawah VWAP — sinyal lebih lemah
        score *= 0.7

    # ── Harga turun mengkonfirmasi distribusi ────────────────────────────
    if delta_price < -1.0:
        score *= 1.15
    elif delta_price < 0:
        score *= 1.05

    score = max(0.0, min(100.0, score))
    return round(score, 1), flags, contexts


def calc_bearish_divergence_score(
    delta_price:     float,
    delta_cvd_spot:  float,
    delta_cvd_fut:   float,
    delta_oi:        float,
    funding_rate:    float,
    d_vwap:          float,
    vol_ratio:       float,
    fr_velocity:     float,
) -> "tuple[float, list, list]":
    """
    SHORT Engine v2 — Exhaustion Biased (Pre-Distribution Intelligence)

    Mendeteksi exhaustion SEBELUM dump terjadi. Filosofi:
    "Long engine = apakah masih ada bensin untuk naik?
     Short engine = apakah bensinnya sudah habis atau sedang dibakar?"

    Pipeline ini sepenuhnya terpisah dari long scoring.
    Tidak mempengaruhi long engine sama sekali.

    4 Pola Exhaustion:
    ──────────────────────────────────────────────────────────────────
    [DIV-A] Hidden Distribution  — harga naik tapi CVD spot negatif
    [DIV-B] Leverage Trap        — FR mahal + naik cepat + OI naik
    [DIV-C] VWAP Premium Exhaust — terlalu jauh di atas VWAP + CVD fade
    [DIV-D] Cross-Market Diverg  — futures retail beli, spot smart money jual
    ──────────────────────────────────────────────────────────────────
    """
    flags:    list = []
    contexts: list = []
    score:    float = 0.0

    # Syarat dasar: tidak sedang dump keras (sudah terlambat, pakai ls_score)
    if delta_price < -3.0:
        return 0.0, [], []

    # ── [DIV-A] Hidden Distribution ─────────────────────────────────────
    # Harga naik tapi spot CVD negatif = smart money distribusi diam-diam
    if delta_price > 0.5 and delta_cvd_spot < 0 and delta_cvd_fut < 1.0:
        strength = min(abs(delta_cvd_spot) / 8.0, 1.0) * min(delta_price / 3.0, 1.0)
        score += 30.0 + strength * 15.0
        flags.append("DIV_HIDDEN_DIST")
        contexts.append(
            f"🎭 Hidden Dist: harga +{delta_price:.1f}% tapi "
            f"CVD Spot {delta_cvd_spot:+.1f}% (smart money jual)"
        )
    elif delta_price > 0.5 and delta_cvd_fut < -1.0:
        score += 18.0
        flags.append("DIV_FUTURES_DIST")
        contexts.append(
            f"📉 Futures distribusi: harga +{delta_price:.1f}% "
            f"CVD Fut {delta_cvd_fut:+.1f}%"
        )

    # ── [DIV-B] Leverage Trap ────────────────────────────────────────────
    # FR mahal + naik cepat + OI naik = blow-off top setup
    fr_expensive    = funding_rate > 0.0008
    fr_accelerating = fr_velocity  > 0.0002
    oi_building     = delta_oi     > 1.5
    price_not_dump  = delta_price  > -1.0

    if fr_expensive and fr_accelerating and oi_building and price_not_dump:
        trap = 25.0 + (10.0 if funding_rate > 0.0015 else 0.0)
        score += trap
        flags.append("DIV_LEVERAGE_TRAP")
        contexts.append(
            f"💸 Leverage Trap: FR {funding_rate*100:+.4f}% mahal & naik "
            f"+ OI +{delta_oi:.1f}% — longs menumpuk di harga tinggi"
        )
    elif fr_expensive and oi_building and price_not_dump:
        score += 15.0
        flags.append("DIV_FR_OVERLOAD")
        contexts.append(
            f"💀 FR Overload {funding_rate*100:+.4f}% + OI +{delta_oi:.1f}%"
        )

    # ── [DIV-C] VWAP Premium Exhaustion ─────────────────────────────────
    # Harga jauh di atas VWAP + CVD futures melemah = mean-reversion setup
    cvd_fading = delta_cvd_fut < 2.0
    if d_vwap > 5.0 and cvd_fading:
        score += 20.0 + min((d_vwap - 5.0) / 3.0, 1.0) * 10.0
        flags.append("DIV_VWAP_EXTREME")
        contexts.append(
            f"📏 VWAP +{d_vwap:.1f}% extreme + CVD fade — mean-reversion setup"
        )
    elif d_vwap > 3.0 and cvd_fading and delta_cvd_spot < 0:
        score += 14.0
        flags.append("DIV_VWAP_DIST")
        contexts.append(f"📏 VWAP +{d_vwap:.1f}% + spot jual {delta_cvd_spot:+.1f}%")

    # ── [DIV-D] Cross-Market Divergence ─────────────────────────────────
    # Retail futures beli, spot smart money jual = divergence paling reliable
    if delta_cvd_fut > 3.0 and delta_cvd_spot < -2.0 and delta_oi > 1.0:
        strength = min(abs(delta_cvd_spot) / 5.0, 1.0)
        score += 20.0 + strength * 15.0
        flags.append("DIV_CROSS_MARKET")
        contexts.append(
            f"⚔️ Cross-Market: Fut CVD {delta_cvd_fut:+.1f}% (retail beli) "
            f"vs Spot CVD {delta_cvd_spot:+.1f}% (smart money jual)"
        )

    if score <= 0:
        return 0.0, [], []

    # ── Volume Modifier ──────────────────────────────────────────────────
    if vol_ratio >= 2.0:
        score *= 1.12
        contexts.append(f"📊 Volume {vol_ratio:.1f}x konfirmasi distribusi")
    elif vol_ratio < 0.6:
        score *= 0.82

    score = round(max(0.0, min(100.0, score)), 1)

    # Skor minimum bermakna
    if score < 28.0:
        return 0.0, [], []

    return score, flags, contexts


def calc_integrated_short_score(
    *,
    delta_price: float,
    delta_price_short: float,
    price_change_24h: float,
    delta_cvd_spot: float,
    delta_cvd_fut: float,
    delta_oi: float,
    funding_rate: float,
    d_vwap: float,
    vol_ratio: float,
    squeeze_type: str,
    squeeze_stage: str,
    ls_score: float,
    ls_flags: List[str],
    ls_contexts: List[str],
    dist_score: float,
    dist_flags: List[str],
    dist_contexts: List[str],
    div_score: float,
    div_flags: List[str],
    div_contexts: List[str],
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Integrated SHORT decision branch.

    Fungsi ini membuat short berdiri sejajar dengan long:
    data/indikator yang sama dibaca ulang sebagai tekanan turun,
    distribusi, breakdown, long squeeze, atau exhaustion after pump.
    LS/DIST/DIV lama tetap dipakai sebagai sub-interpretasi.
    """
    setup_scores: List[Tuple[str, float, List[str], List[str]]] = []

    both_cvd_bearish = delta_cvd_spot < 0 and delta_cvd_fut < 0
    any_cvd_bearish = delta_cvd_spot < -1.0 or delta_cvd_fut < -1.0
    funding_positive = funding_rate > 0.0001
    funding_hot = funding_rate > 0.0008
    oi_hot = delta_oi > 4.0
    oi_deleveraging = delta_oi < -2.0
    below_vwap = d_vwap < 0

    # ── 1) Bear continuation: cocok untuk tape bearish sepanjang hari ──
    bear_score = 0.0
    bear_flags: List[str] = []
    bear_contexts: List[str] = []
    if price_change_24h <= -1.0:
        bear_score += 14.0 + min(abs(price_change_24h) / 8.0, 1.0) * 10.0
        _add_unique(bear_flags, "BEAR_CONTINUATION_SHORT")
        bear_contexts.append(f"📉 24h bearish {price_change_24h:+.1f}% — tekanan turun aktif")
    if delta_price <= -0.4:
        bear_score += 8.0 + min(abs(delta_price) / 4.0, 1.0) * 6.0
    if delta_price_short <= -0.6:
        bear_score += 8.0
        bear_contexts.append(f"⏬ Momentum pendek {delta_price_short:+.1f}% masih menekan")
    if below_vwap:
        bear_score += 8.0 + min(abs(d_vwap) / 5.0, 1.0) * 6.0
        _add_unique(bear_flags, "VWAP_BELOW")
    if both_cvd_bearish:
        bear_score += 18.0
        _add_unique(bear_flags, "CVD_CONFLUENCE_BEARISH")
        bear_contexts.append(
            f"🔴 CVD Spot{delta_cvd_spot:+.1f}% Fut{delta_cvd_fut:+.1f}% — jual searah"
        )
    elif any_cvd_bearish:
        bear_score += 9.0
    if funding_positive and delta_price <= 0:
        bear_score += 6.0 + (4.0 if funding_hot else 0.0)
        _add_unique(bear_flags, "FR_LONG_CROWD")
    if oi_deleveraging and delta_price <= 0:
        bear_score += 4.0
        _add_unique(bear_flags, "OI_DELEVERAGING")
    if vol_ratio >= 1.2:
        bear_score += 5.0

    if bear_score >= 48.0 and below_vwap and any_cvd_bearish:
        setup_scores.append(("bear_continuation_short", bear_score, bear_flags, bear_contexts))

    # ── 2) Breakdown: transisi dari netral ke bawah VWAP ───────────────
    breakdown_score = 0.0
    breakdown_flags: List[str] = []
    breakdown_contexts: List[str] = []
    if -5.0 <= price_change_24h <= 2.0 and delta_price_short <= -0.8 and d_vwap <= -0.4:
        breakdown_score = 34.0
        _add_unique(breakdown_flags, "BREAKDOWN_SHORT")
        breakdown_contexts.append(
            f"📉 Breakdown: 24h {price_change_24h:+.1f}% dan harga mulai lepas dari VWAP"
        )
        if both_cvd_bearish:
            breakdown_score += 18.0
            _add_unique(breakdown_flags, "CVD_CONFLUENCE_BEARISH")
        elif any_cvd_bearish:
            breakdown_score += 10.0
        if delta_oi > 1.5:
            breakdown_score += 8.0
            _add_unique(breakdown_flags, "OI_HOT_BEARISH")
        if funding_positive:
            breakdown_score += 7.0
            _add_unique(breakdown_flags, "FR_LONG_CROWD")
        if vol_ratio >= 1.0:
            breakdown_score += 5.0
        if breakdown_score >= 50.0:
            setup_scores.append(("breakdown_short", breakdown_score, breakdown_flags, breakdown_contexts))

    # ── 3) Long squeeze: longs masih punya fuel untuk dipaksa turun ─────
    if squeeze_type == "long" and ls_score > 0:
        squeeze_score = ls_score
        squeeze_flags = list(ls_flags)
        squeeze_contexts = list(ls_contexts)
        if price_change_24h < 0:
            squeeze_score += 5.0
        if delta_price_short < 0:
            squeeze_score += 4.0
        if both_cvd_bearish:
            squeeze_score += 5.0
            _add_unique(squeeze_flags, "CVD_CONFLUENCE_BEARISH")
        setup_scores.append(("long_squeeze_short", squeeze_score, squeeze_flags, squeeze_contexts))

    # ── 4) Distribution: smart money jual saat market belum sadar ──────
    if dist_score > 0:
        distribution_score = dist_score
        distribution_flags = ["DISTRIBUTION_SHORT"]
        distribution_contexts = list(dist_contexts)
        _extend_unique(distribution_flags, dist_flags)
        if price_change_24h <= 0:
            distribution_score += 5.0
        if d_vwap > 1.5:
            distribution_score += 6.0
        if funding_positive:
            distribution_score += 4.0
        setup_scores.append(("distribution_short", distribution_score, distribution_flags, distribution_contexts))

    # ── 5) Exhaustion after pump: fade pucuk, bukan trend-follow short ──
    pumpish = price_change_24h >= 3.0 or delta_price >= 1.2 or d_vwap >= 3.0
    exhaustion_evidence = dist_score > 0 or div_score > 0 or delta_cvd_spot < -1.0 or delta_cvd_fut < -1.0
    if pumpish and exhaustion_evidence:
        pump_ref = max(price_change_24h, delta_price, d_vwap)
        exhaustion_score = 26.0 + min(max(pump_ref, 0.0) / 8.0, 1.0) * 12.0
        exhaustion_flags = ["EXHAUSTION_AFTER_PUMP_SHORT"]
        exhaustion_contexts = [
            f"🪫 Exhaustion after pump: 24h {price_change_24h:+.1f}% VWAP {d_vwap:+.1f}%"
        ]
        if dist_score > 0:
            exhaustion_score += min(dist_score * 0.25, 12.0)
            _extend_unique(exhaustion_flags, dist_flags)
        if div_score > 0:
            exhaustion_score += min(div_score * 0.22, 12.0)
            _extend_unique(exhaustion_flags, div_flags)
        if funding_hot or oi_hot:
            exhaustion_score += 5.0
            _add_unique(exhaustion_flags, "D_EXHAUSTION")
        setup_scores.append(("exhaustion_after_pump_short", exhaustion_score, exhaustion_flags, exhaustion_contexts))

    # ── 5b) Top reversal: tren naik masih berjalan, tapi pucuk rapuh ───
    top_reversal_context = (
        (price_change_24h >= 5.0 or delta_price >= 2.0 or d_vwap >= 4.0)
        and d_vwap >= 2.0
    )
    top_flow_break = delta_cvd_spot < -1.0 and delta_cvd_fut < 1.5
    top_deriv_heat = funding_hot or oi_hot or dist_score >= 55.0 or div_score >= 45.0
    if top_reversal_context and top_flow_break and top_deriv_heat:
        top_score = 36.0
        top_score += min(max(d_vwap, 0.0) / 8.0, 1.0) * 10.0
        top_score += min(max(price_change_24h, delta_price, 0.0) / 12.0, 1.0) * 8.0
        top_flags = ["TOP_REVERSAL_SHORT"]
        top_contexts = [
            f"🎯 Top reversal: pump/premium mulai rapuh "
            f"(24h {price_change_24h:+.1f}%, VWAP {d_vwap:+.1f}%)"
        ]
        if delta_cvd_spot < -1.0:
            top_score += 6.0
            _add_unique(top_flags, "DIV_HIDDEN_DIST")
        if funding_hot or oi_hot:
            top_score += 5.0
            _add_unique(top_flags, "D_EXHAUSTION")
        if dist_score > 0:
            top_score += min(dist_score * 0.20, 10.0)
            _extend_unique(top_flags, dist_flags)
        if div_score > 0:
            top_score += min(div_score * 0.20, 10.0)
            _extend_unique(top_flags, div_flags)
        setup_scores.append(("top_reversal_short", top_score, top_flags, top_contexts))

    # ── 6) Bearish divergence: harga belum jatuh tapi flow sudah rusak ─
    if div_score > 0:
        divergence_score = div_score
        divergence_flags = ["BEARISH_DIVERGENCE_SHORT"]
        divergence_contexts = list(div_contexts)
        _extend_unique(divergence_flags, div_flags)
        if price_change_24h >= -2.0:
            divergence_score += 5.0
        if d_vwap > 1.0:
            divergence_score += 4.0
        if funding_hot or oi_hot:
            divergence_score += 5.0
        setup_scores.append(("bearish_divergence_short", divergence_score, divergence_flags, divergence_contexts))

    if not setup_scores:
        return 0.0, [], [], []

    setup_scores.sort(key=lambda x: x[1], reverse=True)
    primary_setup, score, flags, contexts = setup_scores[0]
    setups = [name for name, _, _, _ in setup_scores]

    if len(setups) > 1:
        score += min((len(setups) - 1) * 1.5, 4.0)
        contexts.append(f"🧩 {len(setups)} short setup selaras: {', '.join(setups[:3])}")

    _add_unique(flags, "SHORT_ENGINE")
    _add_unique(flags, primary_setup.upper())

    # ── Anti-late-entry guard: jangan short dasar yang sudah terlalu jauh ─
    late_drop = price_change_24h <= -12.0 and d_vwap <= -6.0
    if late_drop and primary_setup not in ("long_squeeze_short", "bear_continuation_short"):
        score *= 0.65
        contexts.append("⚠️ Drop 24h sudah jauh — short dipenalti agar tidak entry di dasar")
    elif price_change_24h <= -18.0:
        score *= 0.75
        contexts.append("⚠️ 24h sangat oversold — target short harus konservatif")

    # Flow conflict: CVD dua market malah beli, short harus turun kualitas.
    if delta_cvd_spot > 2.0 and delta_cvd_fut > 2.0:
        score *= 0.55
        _add_unique(flags, "SHORT_FLOW_CONFLICT")
        contexts.append("⚠️ CVD dua market bullish — short ditahan")

    if squeeze_type == "long_exhausted":
        score *= 0.60
        contexts.append("🔄 Long squeeze exhausted — rawan bounce, short ditahan")

    if vol_ratio < 0.5:
        score *= 0.85
        contexts.append("📉 Volume sepi — short kurang konfirmasi")

    score, cap_reason = _cap_short_score(
        score=score,
        primary_setup=primary_setup,
        price_change_24h=price_change_24h,
        d_vwap=d_vwap,
        delta_price_short=delta_price_short,
        delta_cvd_spot=delta_cvd_spot,
        delta_cvd_fut=delta_cvd_fut,
        delta_oi=delta_oi,
        funding_rate=funding_rate,
        vol_ratio=vol_ratio,
        dist_score=dist_score,
        div_score=div_score,
        ls_score=ls_score,
    )
    if cap_reason:
        contexts.append(f"🧯 {cap_reason} — confidence short dibatasi")

    score = round(_clamp_score(score), 1)
    if score < 25.0:
        return 0.0, [], [], []

    contexts.insert(0, f"▼ SHORT {primary_setup.replace('_', ' ')} — score {score:.1f}")
    return score, flags, contexts, setups
