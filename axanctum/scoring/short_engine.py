from __future__ import annotations

from typing import List, Tuple


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
