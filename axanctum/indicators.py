from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pandas as pd

from .config import CONFIG


def calc_rolling_vwap(df: pd.DataFrame, n: int) -> Tuple[float, float]:
    """
    Rolling VWAP berbasis N candle terakhir.

        VWAP = Σ(TP_i × Vol_i) / Σ(Vol_i)
        TP   = (High + Low + Close) / 3   ← Typical Price

        D_vwap = (Price_current − VWAP) / VWAP × 100

    Returns: (vwap, d_vwap_pct)
      d_vwap > 0 → harga di atas VWAP (premium)
      d_vwap < 0 → harga di bawah VWAP (discount)
    """
    tail = df.tail(n).copy()
    typical_price = (tail["high"] + tail["low"] + tail["close"]) / 3
    total_vol = tail["volume"].sum()

    if total_vol == 0:
        return float(df["close"].iloc[-1]), 0.0

    vwap = float((typical_price * tail["volume"]).sum() / total_vol)
    price_now = float(df["close"].iloc[-1])

    if vwap == 0:
        return price_now, 0.0

    d_vwap = (price_now - vwap) / vwap * 100
    return round(vwap, 8), round(d_vwap, 4)


def calc_cvd_and_momentum(df: pd.DataFrame, n: int) -> Tuple[float, float]:
    """
    Hitung Cumulative Volume Delta (CVD) dan momentum perubahan-nya.

        CVD = Σ(TakerBuyVol_i − TakerSellVol_i)   dari candle ke-1 s/d N

    Momentum (ΔCVD):
        Split N candle menjadi dua paruh.
        CVD_past    = CVD dari paruh pertama (candle awal)
        CVD_current = CVD dari paruh kedua  (candle terbaru)
        ΔCVD = (CVD_current − CVD_past) / |CVD_past| × 100

        Nilai positif → tekanan beli semakin kuat
        Nilai negatif → tekanan jual semakin kuat

    Returns: (cvd_total, delta_cvd_pct)
    """
    tail  = df.tail(n).copy()
    delta = tail["taker_buy_vol"] - tail["taker_sell_vol"]
    cvd_total = float(delta.sum())

    half = max(1, n // 2)
    cvd_past    = float(delta.iloc[:half].sum())
    cvd_current = float(delta.iloc[half:].sum())

    # Normalisasi by total taker volume — stabil, tidak bisa meledak
    total_vol = float(tail["taker_buy_vol"].sum() + tail["taker_sell_vol"].sum())
    if total_vol < 1e-10:
        delta_cvd_pct = 0.0
    else:
        delta_cvd_pct = (cvd_current - cvd_past) / total_vol * 100.0

    # ── CVD Noise Filter ──────────────────────────────────────────────
    # Jika total volume terlalu kecil relatif terhadap candle sebelumnya,
    # sinyal CVD tidak dapat dipercaya (market mati / debu)
    # avg_vol_per_candle = total_vol / N — bandingkan dengan threshold minimum
    avg_vol_per_candle = total_vol / max(n, 1)
    # Ambil rata-rata volume dari seluruh tail untuk referensi "normal"
    ref_vol = float(tail["volume"].mean())
    if ref_vol > 1e-10 and avg_vol_per_candle < ref_vol * CONFIG.get("CVD_MIN_VOL_RATIO", 0.15):
        # Volume terlalu kecil → set CVD ke nol (jangan beri sinyal pada debu)
        delta_cvd_pct = 0.0
        cvd_total     = 0.0

    delta_cvd_pct = max(-500.0, min(500.0, delta_cvd_pct))
    return round(cvd_total, 4), round(delta_cvd_pct, 2)


def calc_oi_momentum(oi_list: List[float]) -> float:
    """
    ΔOI = (OI_current − OI_past) / OI_past × 100

    OI_past    = nilai OI candle pertama dalam list
    OI_current = nilai OI candle terakhir

    Nilai positif → posisi terbuka bertambah (leverage naik)
    Nilai negatif → posisi terbuka berkurang (deleveraging)
    """
    if not oi_list or len(oi_list) < 2:
        return 0.0
    oi_past    = oi_list[0]
    oi_current = oi_list[-1]
    if oi_past == 0:
        return 0.0
    return round((oi_current - oi_past) / oi_past * 100, 4)


def calc_price_momentum(df: pd.DataFrame, n: int) -> float:
    """
    ΔPrice = (Price_current − Price_past) / Price_past × 100

    Price_past    = close candle ke-(len-N)
    Price_current = close candle terakhir
    """
    if len(df) < n + 1:
        n = len(df) - 1
    if n <= 0:
        return 0.0
    p_past    = float(df["close"].iloc[-n - 1])
    p_current = float(df["close"].iloc[-1])
    if p_past == 0:
        return 0.0
    return round((p_current - p_past) / p_past * 100, 4)


def calc_absorption(
    fut_df:         pd.DataFrame,
    delta_cvd_spot: float,
    delta_cvd_fut:  float,
    delta_price:    float,
    n:              int,
) -> Tuple[float, str]:
    """
    Deteksi Absorption menggunakan Combined CVD (Spot + Futures).

    Sebelumnya hanya cek CVD Spot — padahal volume futures jauh lebih
    besar di crypto. Absorption paling berbahaya justru terjadi di futures:
    ada sell wall raksasa di futures yang menyerap semua pembelian tapi
    harga tidak bergerak.

    Combined CVD = (CVD_spot × 1.0 + CVD_fut × 1.3) / 2.3
    Futures diberi bobot lebih besar karena volumenya lebih dominan.

    Kondisi absorption:
      - Combined CVD naik kuat (> 3%)
      - Tapi ΔPrice sangat kecil relatif terhadap ATR (< 0.40×ATR)
      → Ada yang menyerap semua pembelian → bukan akumulasi sehat
    """
    tail    = fut_df.tail(n).copy()
    hl_avg  = float((tail["high"] - tail["low"]).mean())
    price   = float(fut_df["close"].iloc[-1])
    atr_pct = (hl_avg / price * 100) if price > 0 else 0.5
    atr_pct = max(0.1, min(atr_pct, 10.0))

    # Combined CVD — futures diberi bobot lebih besar
    combined_cvd = (delta_cvd_spot * 1.0 + delta_cvd_fut * 1.3) / 2.3

    # Syarat 1: Combined CVD naik signifikan
    cvd_kuat = combined_cvd > 3.0

    # Syarat 2: Harga hampir tidak bergerak relatif terhadap ATR
    price_vs_atr  = abs(delta_price) / atr_pct if atr_pct > 0 else 1.0
    harga_stagnan = price_vs_atr < 0.40

    if cvd_kuat and harga_stagnan:
        strength = min(combined_cvd / 8.0, 1.0)
        penalty  = max(0.40, 1.0 - strength * 0.60)
        desc = (
            f"🧱 Absorption: CVD Combined +{combined_cvd:.1f}% "
            f"(Spot:{delta_cvd_spot:+.1f}% Fut:{delta_cvd_fut:+.1f}%) "
            f"tapi harga {delta_price:+.2f}% ({price_vs_atr:.2f}×ATR)"
        )
        return round(penalty, 3), desc

    return 1.0, ""


def calc_volume_anomaly(
    df: pd.DataFrame,
    n: int,
    tf_minutes: int = 60,
) -> Tuple[float, str]:
    """
    Bandingkan volume candle terakhir vs rata-rata N candle sebelumnya.

    FIX UNCLOSED CANDLE:
    Jika candle terakhir belum closed (baru berjalan sebagian),
    volume-nya diproyeksikan ke full candle sebelum dibandingkan.

    Proyeksi: vol_projected = vol_sekarang × (tf_minutes / menit_berjalan)
    Menit berjalan dihitung dari selisih open_time candle terakhir vs sekarang.
    """
    if len(df) < n + 1:
        return 1.0, "─ Normal"

    tail     = df.tail(n + 1)
    last_vol = float(tail["volume"].iloc[-1])
    avg_vol  = float(tail["volume"].iloc[:-1].mean())

    if avg_vol < 1e-10:
        return 1.0, "─ Normal"

    # ── Deteksi dan koreksi unclosed candle ──────────────────────────
    last_candle_open_ts = int(tail["ts"].iloc[-1])
    now_ms              = int(time.time() * 1000)
    elapsed_ms          = now_ms - last_candle_open_ts
    elapsed_min         = elapsed_ms / 60_000

    # Jika candle baru berjalan < 90% dari durasi penuh → proyeksikan
    if 0 < elapsed_min < tf_minutes * 0.9:
        projection_factor = tf_minutes / max(elapsed_min, 1.0)
        # Cap projection: max 10x untuk hindari outlier ekstrem
        projection_factor = min(projection_factor, 10.0)
        last_vol = last_vol * projection_factor

    ratio = last_vol / avg_vol

    if ratio >= 3.5:
        label = f"🔴🔴🔴🔴🔴 {ratio:.1f}x — EKSTREM"
    elif ratio >= 2.5:
        label = f"🔴🔴🔴🔴⚪ {ratio:.1f}x — Sangat Tinggi"
    elif ratio >= 1.5:
        label = f"🔴🔴🔴⚪⚪ {ratio:.1f}x — Elevated"
    elif ratio >= 0.8:
        label = f"⚪⚪⚪⚪⚪ {ratio:.1f}x — Normal"
    else:
        label = f"🔵 {ratio:.1f}x — Sepi / Low Vol"

    return round(ratio, 2), label


def calc_cvd_volume_consistency(
    delta_cvd_spot: float,
    delta_cvd_fut:  float,
    vol_ratio:      float,
) -> Tuple[float, str]:
    """
    Cek apakah volume candle terakhir konsisten dengan CVD momentum.

    Menggunakan Combined CVD (bukan hanya spot) karena:
    - Futures bisa punya CVD kuat meski spot sepi
    - Keduanya perlu dikonfirmasi oleh volume candle terkini

    Combined CVD menentukan arah sinyal, vol_ratio menentukan konfirmasi.
    """
    # Combined CVD — weighted average
    combined_cvd = (delta_cvd_spot * 1.0 + delta_cvd_fut * 1.3) / 2.3

    # Hanya relevan jika ada sinyal beli (combined positif)
    if combined_cvd <= 0:
        return 1.0, ""

    if vol_ratio < 0.3:
        mult = 0.65
        desc = (
            f"⚠️ CVD-Vol Inconsistent: CVD Combined +{combined_cvd:.1f}% "
            f"tapi volume hanya {vol_ratio:.1f}x — momentum memudar"
        )
    elif vol_ratio < 0.5:
        mult = 0.80
        desc = (
            f"🟡 CVD-Vol Lemah: CVD Combined +{combined_cvd:.1f}% "
            f"dengan volume {vol_ratio:.1f}x — konfirmasi tipis"
        )
    elif vol_ratio < 0.8:
        mult = 0.92
        desc = ""
    elif vol_ratio >= 2.0 and combined_cvd > 5.0:
        mult = 1.06
        desc = (
            f"✅ CVD-Vol Kuat: CVD Combined +{combined_cvd:.1f}% "
            f"dikonfirmasi volume {vol_ratio:.1f}x"
        )
    else:
        mult = 1.0
        desc = ""

    return round(mult, 3), desc


def calc_squeeze_stage(
    delta_oi:     float,
    funding_rate: float,
    delta_price:  float,
) -> Tuple[str, str, str, float]:
    """
    Deteksi tipe dan tahap squeeze — SHORT maupun LONG.

    SHORT SQUEEZE:
      Harga naik + OI turun + FR negatif
      → Shorts terlikuidasi paksa → harga naik lebih lanjut
      → Signal: BULLISH

    LONG SQUEEZE:
      Harga turun + OI turun + FR positif
      → Longs terlikuidasi paksa → harga turun lebih lanjut
      → Signal: SHORT

    LONG SQUEEZE EXHAUSTED:
      Harga sudah turun + OI sudah habis + FR sudah netral/negatif
      → Longs sudah habis dilikuidasi → potensi reversal/bounce
      → Signal: POTENSI REVERSAL

    Returns: (squeeze_type, stage, description, fuel_remaining_pct)
      squeeze_type : "short" | "long" | "long_exhausted" | "none"
      stage        : "early" | "mid" | "late" | "exhausted" | "none"
      fuel         : estimasi % bahan bakar yang tersisa (0–100)
    """
    # ── Short Squeeze ─────────────────────────────────────────────────
    if delta_oi < 0 and delta_price > 0:
        oi_drop = abs(delta_oi)
        if oi_drop < 2:      oi_score = 0.9
        elif oi_drop < 5:    oi_score = 0.65
        elif oi_drop < 10:   oi_score = 0.35
        else:                oi_score = 0.1

        if funding_rate < -0.0010:   fr_score = 1.0
        elif funding_rate < -0.0003: fr_score = 0.75
        elif funding_rate < 0:       fr_score = 0.5
        elif funding_rate < 0.0002:  fr_score = 0.25
        else:                        fr_score = 0.05

        if delta_price < 1:    price_score = 0.9
        elif delta_price < 3:  price_score = 0.65
        elif delta_price < 6:  price_score = 0.35
        else:                  price_score = 0.1

        fuel = (oi_score * 0.4 + fr_score * 0.35 + price_score * 0.25) * 100

        if fuel >= 70:
            stage = "early"
            desc  = f"🔫 Short Squeeze Early — ~{fuel:.0f}% fuel tersisa"
        elif fuel >= 45:
            stage = "mid"
            desc  = f"🔫 Short Squeeze Mid — ~{fuel:.0f}% fuel tersisa"
        elif fuel >= 20:
            stage = "late"
            desc  = f"⚠️ Short Squeeze Late — hampir habis (~{fuel:.0f}%)"
        else:
            stage = "exhausted"
            desc  = f"💨 Short Squeeze Exhausted — fuel ~{fuel:.0f}%, waspadai reversal"

        return "short", stage, desc, round(fuel, 1)

    # ── Long Squeeze ──────────────────────────────────────────────────
    if delta_oi < 0 and delta_price < 0 and funding_rate > 0:
        oi_drop = abs(delta_oi)

        # Seberapa banyak longs yang sudah dilikuidasi
        if oi_drop < 2:      oi_score = 0.9   # baru mulai
        elif oi_drop < 5:    oi_score = 0.65
        elif oi_drop < 10:   oi_score = 0.35
        else:                oi_score = 0.1   # sudah hampir habis

        # FR masih positif = masih ada longs yang belum kena likuidasi
        if funding_rate > 0.0010:    fr_score = 1.0
        elif funding_rate > 0.0005:  fr_score = 0.75
        elif funding_rate > 0.0001:  fr_score = 0.5
        else:                        fr_score = 0.15  # FR hampir netral

        # Seberapa jauh harga sudah turun
        drop_abs = abs(delta_price)
        if drop_abs < 1:    price_score = 0.9
        elif drop_abs < 3:  price_score = 0.65
        elif drop_abs < 6:  price_score = 0.35
        else:               price_score = 0.1

        fuel = (oi_score * 0.4 + fr_score * 0.35 + price_score * 0.25) * 100

        # Long squeeze exhausted = peluang reversal
        if fuel < 25 and funding_rate < 0.0002:
            stage = "exhausted"
            desc  = f"✅ Long Squeeze Exhausted — longs habis, potensi reversal"
            return "long_exhausted", stage, desc, round(fuel, 1)

        if fuel >= 70:
            stage = "early"
            desc  = f"💀 Long Squeeze Early — ~{fuel:.0f}% longs belum likuidasi"
        elif fuel >= 45:
            stage = "mid"
            desc  = f"💀 Long Squeeze Mid — ~{fuel:.0f}% longs tersisa"
        elif fuel >= 20:
            stage = "late"
            desc  = f"⚠️ Long Squeeze Late — hampir selesai (~{fuel:.0f}%)"
        else:
            stage = "exhausted"
            desc  = f"✅ Long Squeeze Exhausted — potensi reversal/bounce"

        return "long", stage, desc, round(fuel, 1)

    return "none", "none", "─ Tidak ada squeeze aktif", 0.0


def calc_price_targets(
    price:      float,
    vwap:       float,
    d_vwap:     float,
    df:         "pd.DataFrame",
    n:          int,
    flags:      List[str],
    squeeze_stage: str,
) -> Dict:
    """
    Hitung target harga dan level invalidasi berdasarkan kondisi microstructure.

    ATR (Average True Range) dihitung dari N candle terakhir sebagai
    proxy volatilitas — menggantikan kebutuhan akan indikator eksternal.

        ATR_candle = max(High, Close_prev) − min(Low, Close_prev)
        ATR_avg    = mean(ATR_candle, N candle terakhir)
        ATR_pct    = ATR_avg / Price × 100

    Target dan stop diekspresikan dalam % dari harga saat ini.
    """
    # ── Hitung ATR sebagai proxy volatilitas ─────────────────────────
    tail       = df.tail(n + 1).copy()
    prev_close = tail["close"].shift(1)
    # True Range = max(High, prev_Close) − min(Low, prev_Close)
    # Menggunakan prev_close (bukan current close) untuk akurasi
    true_high  = pd.concat([tail["high"], prev_close], axis=1).max(axis=1)
    true_low   = pd.concat([tail["low"],  prev_close], axis=1).min(axis=1)
    true_range = true_high - true_low
    atr_pct = float(true_range.iloc[1:].mean() / price * 100)
    atr_pct = max(0.1, min(atr_pct, 10.0))  # clamp 0.1–10%

    targets: Dict = {
        "atr_pct":    round(atr_pct, 3),
        "target1":    None,
        "target2":    None,
        "invalidasi": None,
        "direction":  "NEUTRAL",
        "horizon":    "─",
        "rationale":  "─",
    }

    # ── Skenario A: Organic Accumulation ─────────────────────────────
    if "A" in flags and "E_DISTRIBUTION" not in flags:
        targets["direction"] = "BULLISH"
        targets["horizon"]   = "3–8 candle"

        if d_vwap < 0:
            # Harga di bawah VWAP → target pertama adalah VWAP itu sendiri
            t1 = vwap
            t2 = vwap * (1 + atr_pct / 100 * 1.5)
            targets["rationale"] = "Akumulasi dari discount — target mean reversion ke VWAP"
        else:
            # Harga sudah di atas VWAP → target ekstensi
            t1 = price * (1 + atr_pct / 100 * 1.2)
            t2 = price * (1 + atr_pct / 100 * 2.2)
            targets["rationale"] = "Akumulasi di atas VWAP — target ekstensi ATR"

        targets["target1"]    = round(t1, 6)
        targets["target2"]    = round(t2, 6)
        targets["invalidasi"] = round(price * (1 - atr_pct / 100 * 0.8), 6)

    # ── Skenario C: Short Squeeze ─────────────────────────────────────
    elif "C_SQUEEZE" in flags:
        targets["direction"] = "BULLISH"

        if squeeze_stage == "early":
            mult1, mult2       = 1.8, 3.2
            targets["horizon"] = "2–6 candle"
            targets["rationale"] = "Early squeeze — momentum belum puncak"
        elif squeeze_stage == "mid":
            mult1, mult2       = 1.2, 2.0
            targets["horizon"] = "2–4 candle"
            targets["rationale"] = "Mid squeeze — target lebih konservatif"
        else:
            mult1, mult2       = 0.6, 1.0
            targets["horizon"] = "1–2 candle"
            targets["rationale"] = "Late/exhausted squeeze — target sangat ketat, waspadai reversal"

        targets["target1"]    = round(price * (1 + atr_pct / 100 * mult1), 6)
        targets["target2"]    = round(price * (1 + atr_pct / 100 * mult2), 6)
        targets["invalidasi"] = round(price * (1 - atr_pct / 100 * 0.6), 6)

    # ── Skenario E: Distribution / Mean Reversion ─────────────────────
    elif "E_DISTRIBUTION" in flags:
        targets["direction"]  = "BEARISH / AVOID LONG"
        targets["horizon"]    = "2–5 candle"
        targets["rationale"]  = "Distribusi di level tinggi — target mean reversion ke VWAP"
        targets["target1"]    = round(vwap, 6)           # kembali ke VWAP
        targets["target2"]    = round(vwap * 0.985, 6)  # sedikit di bawah VWAP
        targets["invalidasi"] = round(price * 1.015, 6) # 1.5% di atas = sinyal invalid

    # ── Integrated SHORT: Bearish Continuation / Breakdown ────────────
    elif any(f in flags for f in ["BEAR_CONTINUATION_SHORT", "BREAKDOWN_SHORT"]):
        targets["direction"] = "SHORT — Bear Continuation"
        targets["horizon"]   = "2–6 candle"

        if "BREAKDOWN_SHORT" in flags:
            mult1, mult2, inv = 1.3, 2.2, 0.75
            targets["rationale"] = "Breakdown dari area netral — target ekstensi ATR ke bawah"
        else:
            mult1, mult2, inv = 1.1, 1.9, 0.85
            targets["rationale"] = "Trend bearish aktif — target lanjutan konservatif"

        targets["target1"]    = round(price * (1 - atr_pct / 100 * mult1), 6)
        targets["target2"]    = round(price * (1 - atr_pct / 100 * mult2), 6)
        targets["invalidasi"] = round(price * (1 + atr_pct / 100 * inv), 6)

    # ── Integrated SHORT: Distribution / Divergence / Pump Exhaustion ─
    elif any(
        f in flags
        for f in [
            "DISTRIBUTION_SHORT",
            "BEARISH_DIVERGENCE_SHORT",
            "EXHAUSTION_AFTER_PUMP_SHORT",
            "TOP_REVERSAL_SHORT",
        ]
    ):
        targets["direction"] = "SHORT — Distribution / Exhaustion"
        targets["horizon"]   = "1–5 candle"
        targets["rationale"] = "Distribusi/exhaustion — target utama mean reversion"

        if d_vwap > 0:
            t1 = min(vwap, price * (1 - atr_pct / 100 * 0.9))
            t2 = min(vwap * 0.985, price * (1 - atr_pct / 100 * 1.8))
        else:
            t1 = price * (1 - atr_pct / 100 * 1.0)
            t2 = price * (1 - atr_pct / 100 * 1.7)

        targets["target1"]    = round(t1, 6)
        targets["target2"]    = round(t2, 6)
        targets["invalidasi"] = round(price * (1 + atr_pct / 100 * 0.8), 6)

    # ── Skenario B: Speculative ───────────────────────────────────────
    elif "B_SPECULATIVE" in flags:
        targets["direction"]  = "SPECULATIVE BULLISH ⚠️"
        targets["horizon"]    = "1–3 candle"
        targets["rationale"]  = "Futures-driven — target sempit, stop ketat"
        targets["target1"]    = round(price * (1 + atr_pct / 100 * 0.8), 6)
        targets["target2"]    = round(price * (1 + atr_pct / 100 * 1.4), 6)
        targets["invalidasi"] = round(price * (1 - atr_pct / 100 * 0.5), 6)

    # ── Skenario Long Squeeze: SHORT target ───────────────────────────
    elif "LONG_SQUEEZE" in flags:
        targets["direction"]  = "SHORT — Long Squeeze Aktif"
        targets["horizon"]    = "2–6 candle"
        targets["rationale"]  = "Long squeeze berlangsung — target ke bawah VWAP"
        targets["target1"]    = round(price * (1 - atr_pct / 100 * 1.5), 6)
        targets["target2"]    = round(price * (1 - atr_pct / 100 * 2.8), 6)
        targets["invalidasi"] = round(price * (1 + atr_pct / 100 * 0.8), 6)

    # ── Skenario Long Squeeze Exhausted: REVERSAL target ─────────────
    elif "LONG_SQUEEZE_EXHAUSTED" in flags:
        targets["direction"]  = "REVERSAL — Long Squeeze Selesai"
        targets["horizon"]    = "3–8 candle"
        targets["rationale"]  = "Longs sudah habis — setup bounce dari oversold"
        targets["target1"]    = round(vwap, 6)
        targets["target2"]    = round(vwap * (1 + atr_pct / 100 * 1.2), 6)
        targets["invalidasi"] = round(price * (1 - atr_pct / 100 * 0.6), 6)

    return targets
