from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Tuple

import aiohttp
import requests as _requests

from .config import CONFIG, DASHBOARD_URL, GRADE_SCALE, TELEGRAM_API, TF_MINUTES
from .logging_setup import log
from .state import ALERT_COOLDOWN_SEC, _ALERT_COOLDOWN

_DASHBOARD_WARNED = False


def _dashboard_post_sync(symbol, direction, entry, tp, sl, grade, leverage):
    """Synchronous HTTP POST — dijalankan di thread terpisah agar tidak block event loop."""
    if not DASHBOARD_URL:
        return
    try:
        resp = _requests.post(
            f"{DASHBOARD_URL}/signal",
            json={
                "symbol":    symbol.upper(),
                "direction": direction.upper(),
                "entry":     float(entry),
                "tp":        float(tp),
                "sl":        float(sl),
                "grade":     grade,
                "leverage":  int(leverage),
                "source":    "bot",
            },
            timeout=3,
        )
        if resp.status_code >= 400:
            log.warning(f"Dashboard HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        log.warning(f"Dashboard signal error: {exc}")


def send_signal_to_dashboard(symbol, direction, entry, tp, sl, grade="B", leverage=5):
    """
    Kirim sinyal ke dashboard simulasi.
    Fire-and-forget via asyncio.to_thread agar tidak memblokir event loop.
    """
    global _DASHBOARD_WARNED
    if not DASHBOARD_URL:
        if not _DASHBOARD_WARNED:
            log.warning("Dashboard URL belum diset — skip pengiriman sinyal dashboard.")
            _DASHBOARD_WARNED = True
        return

    try:
        try:
            loop = asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        if loop_running:
            asyncio.ensure_future(
                asyncio.to_thread(
                    _dashboard_post_sync,
                    symbol, direction, entry, tp, sl, grade, leverage,
                )
            )
        else:
            _dashboard_post_sync(symbol, direction, entry, tp, sl, grade, leverage)
    except Exception:
        pass


def get_grade(score: float) -> Tuple[str, str]:
    """Convert skor 0–100 ke (grade_letter, display_label)."""
    for threshold, grade, label in GRADE_SCALE:
        if score >= threshold:
            return grade, label
    return "D", "⚪ NO SIGNAL"


def _is_on_cooldown(symbol: str, signal_type: str) -> bool:
    """
    Cek apakah simbol ini masih dalam cooldown untuk tipe sinyal tertentu.
    Mencegah sinyal spam ke Telegram dalam window minimum satu candle.
    """
    now = time.time()
    entry = _ALERT_COOLDOWN.get(symbol, {})
    last_sent = entry.get(signal_type, 0.0)
    tf_minutes = TF_MINUTES.get(CONFIG.get("TIMEFRAME", "1h"), 60)
    cooldown_sec = max(ALERT_COOLDOWN_SEC, int(tf_minutes * 60))
    return (now - last_sent) < cooldown_sec


def _mark_sent(symbol: str, signal_type: str) -> None:
    """Tandai simbol + tipe sinyal sudah dikirim sekarang."""
    if symbol not in _ALERT_COOLDOWN:
        _ALERT_COOLDOWN[symbol] = {}
    _ALERT_COOLDOWN[symbol][signal_type] = time.time()


def _scenario_summary(flags: List[str]) -> str:
    """Buat ringkasan emoji dari skenario aktif."""
    parts = []
    if "A"              in flags: parts.append("🟢 Spot Accum")
    if "B_SPECULATIVE"  in flags: parts.append("🟡 Spec Rally")
    if "C_SQUEEZE"      in flags: parts.append("🔫 Squeeze")
    if "D_EXHAUSTION"   in flags: parts.append("💀 Overleveraged")
    if "E_DISTRIBUTION" in flags: parts.append("🪤 Distribution")
    return " │ ".join(parts) if parts else "─ Neutral"


async def send_telegram(
    session: aiohttp.ClientSession,
    message: str,
) -> bool:
    """Kirim pesan HTML ke Telegram chat/group yang dikonfigurasi."""
    url = f"{TELEGRAM_API}/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.post(
            url,
            json={
                "chat_id":    CONFIG["TELEGRAM_CHAT_ID"],
                "text":       message,
                "parse_mode": "HTML",
            },
            timeout=timeout,
        ) as resp:
            ok = resp.status == 200
            if not ok:
                body = await resp.text()
                log.warning(f"Telegram HTTP {resp.status}: {body[:120]}")
            return ok
    except Exception as exc:
        log.error(f"send_telegram error: {exc}")
        return False


def build_telegram_message(r: Dict, signal_type: str = "LONG", regime_ctx=None) -> str:
    """Format pesan Telegram — ringkas dan decisional."""
    from datetime import timezone, timedelta

    sym     = r["symbol"].replace("USDT", "")
    wib     = timezone(timedelta(hours=7))
    now     = datetime.now(wib).strftime("%d %b %Y  %H:%M WIB")

    # ── Pilih data berdasarkan tipe sinyal ────────────────────────────
    if signal_type == "SHORT":
        score      = r.get("ls_score", 0)
        flags      = r.get("ls_flags", [])
        contexts   = r.get("ls_contexts", [])
        tgt        = r.get("ls_targets", {})
        sq_type    = r.get("squeeze_type", "none")
        is_reversal = sq_type == "long_exhausted"
    else:
        score      = r["score"]
        flags      = r.get("flags", [])
        contexts   = r.get("contexts", [])
        tgt        = r.get("targets", {})
        is_reversal = False
        
    # ── Skor & Grade ─────────────────────────────────────────────────
    if signal_type == "SHORT":
        if is_reversal:
            grade_emoji = "🔄"
            grade_label = "🔄 REVERSAL SIGNAL"
            grade       = "REV"
        else:
            # Grade SHORT berdasarkan ls_score
            if score >= 85:   grade_emoji, grade_label, grade = "🔥", "🔥 PRIME SHORT", "A+"
            elif score >= 70: grade_emoji, grade_label, grade = "🩸", "🩸 STRONG SHORT", "A"
            elif score >= 55: grade_emoji, grade_label, grade = "🟠", "🟠 DECENT SHORT", "B+"
            else:             grade_emoji, grade_label, grade = "📊", "📊 WATCH SHORT", "B"
    else:
        grade_emoji = {
            "A+": "🔥", "A": "✅", "B+": "🟡",
            "B": "📊", "C": "⚠️", "D": "⚪",
        }.get(r["grade"], "📊")
        grade_label = r["grade_label"]
        grade       = r["grade"]

    # ── Konteks utama (satu kalimat) ──────────────────────────────────
    spot_pct = ""
    main_ctx = ""
    # Untuk SHORT, gunakan konteks yang relevan (ls_contexts/dist_contexts).
    # Untuk LONG, gunakan r["contexts"] seperti biasa.
    _ctx_source = r.get("ls_contexts", []) if signal_type == "SHORT" else r["contexts"]
    for c in _ctx_source:
        if c.startswith("A:") and "Futures Memimpin" in c:
            try:
                fut_pct = c.split("Futures Memimpin")[1].split("%")[0].strip()
                spot_pct = f"Futures Demand {fut_pct}%"
            except Exception:
                spot_pct = "Futures Demand"
            break
        elif c.startswith("A:"):
            try:
                spot_pct = c.split("Spot Dominan")[1].split("%")[0].strip()
                spot_pct = f"Spot Dominan {spot_pct}%"
            except Exception:
                spot_pct = "Spot Dominan"
            break
        elif c.startswith("B:") and "Futures Memimpin" in c:
            spot_pct = "Futures Memimpin ⚠️"
            break
        elif c.startswith("B:"):
            spot_pct = "Demand Lemah ⚠️"
            break

    # Sub-konteks: volume dan squeeze
    sub_parts = []
    vol_ratio = r.get("vol_ratio", 1.0)
    if vol_ratio >= 2.5:
        sub_parts.append(f"volume {vol_ratio:.1f}x konfirmasi")
    elif vol_ratio >= 1.5:
        sub_parts.append(f"volume {vol_ratio:.1f}x elevated")
    elif vol_ratio < 0.5:
        sub_parts.append("volume sepi")

    sq_stage = r.get("squeeze_stage", "none")
    if sq_stage == "early":
        sub_parts.append("squeeze fresh")
    elif sq_stage == "mid":
        sub_parts.append("squeeze aktif")
    elif sq_stage == "late":
        sub_parts.append("squeeze hampir habis")
    elif sq_stage == "exhausted":
        sub_parts.append("squeeze exhausted ⚠️")

    # Arah
    direction = tgt.get("direction", "NEUTRAL")
    if "BULLISH" in direction and "SPECULATIVE" not in direction:
        arah = "BULLISH"
    elif "SPECULATIVE" in direction:
        arah = "SPECULATIVE BULLISH ⚠️"
    elif "BEARISH" in direction:
        arah = "BEARISH / HINDARI LONG"
    else:
        arah = direction

    main_ctx = f"🧭 {arah} — {spot_pct}"
    sub_ctx  = f"    {', '.join(sub_parts)}" if sub_parts else ""

    # ── Target & Invalidasi ───────────────────────────────────────────
    pr = r["price"]

    def _fmt(p):
        if p is None:
            return "─"
        return f"${p:,.5g}"

    def _pct(p):
        if p is None:
            return ""
        return f"  ({(p - pr) / pr * 100:+.2f}%)"

    t1 = tgt.get("target1")
    t2 = tgt.get("target2")
    iv = tgt.get("invalidasi")

    # ── Signals ringkas ───────────────────────────────────────────────
    signal_parts = []
    if "A"                       in flags: signal_parts.append("🟢 Spot Accum")
    if "A_FUTURES"               in flags: signal_parts.append("🟡 Futures Demand")
    if "CONFLUENCE"              in flags: signal_parts.append("🤝 Konfluensi")
    if "B_SPECULATIVE"           in flags: signal_parts.append("🔴 Spec Rally")
    if "C_SQUEEZE"               in flags:
        fuel = r.get("squeeze_fuel", 0)
        signal_parts.append(f"🔫 Short Squeeze {fuel:.0f}%")
    if "LONG_SQUEEZE"            in flags:
        fuel = r.get("squeeze_fuel", 0)
        signal_parts.append(f"💀 Long Squeeze {fuel:.0f}%")
    if "LONG_SQUEEZE_EXHAUSTED"  in flags: signal_parts.append("🔄 LS Exhausted")
    if "BEAR_CONTINUATION_SHORT" in flags: signal_parts.append("📉 Bear Continuation")
    if "BREAKDOWN_SHORT"         in flags: signal_parts.append("⏬ Breakdown")
    if "EXHAUSTION_AFTER_PUMP_SHORT" in flags: signal_parts.append("🪫 Pump Exhaustion")
    if "TOP_REVERSAL_SHORT"      in flags: signal_parts.append("🎯 Top Reversal")
    if "DISTRIBUTION_SHORT"      in flags: signal_parts.append("🪤 Distribution Short")
    if "BEARISH_DIVERGENCE_SHORT" in flags: signal_parts.append("🎭 Bearish Divergence")
    if "CVD_CONFLUENCE_BEARISH"  in flags: signal_parts.append("🔴 CVD Bearish")
    if "FR_LONG_CROWD"           in flags: signal_parts.append("💸 Long Crowded")
    if "OI_HOT_BEARISH"          in flags: signal_parts.append("⚠️ OI Hot Bear")
    if "OI_DELEVERAGING"         in flags: signal_parts.append("💧 OI Deleveraging")
    if "VWAP_BELOW"              in flags: signal_parts.append("📏 Below VWAP")
    if "D_EXHAUSTION"            in flags: signal_parts.append("💀 Overleveraged")
    if "E_DISTRIBUTION"          in flags: signal_parts.append("🪤 Distribution")
    if "ABSORPTION"              in flags: signal_parts.append("🧱 Absorption")
    signals_str = "  │  ".join(signal_parts) if signal_parts else "─ Neutral"

    # ── Warnings ──────────────────────────────────────────────────────
    warnings = []
    if "B_SPECULATIVE" in flags and "A" not in flags:
        warnings.append("⚠️ Speculative — tidak ada konfirmasi spot")
    if "E_DISTRIBUTION" in flags:
        warnings.append("🪤 Distribution trap — hindari long")
    if "D_EXHAUSTION" in flags:
        warnings.append("💀 Overleverage — long squeeze risk")
    if "ABSORPTION" in flags:
        warnings.append("🧱 Absorption detected — iceberg sell")
    if "LONG_SQUEEZE" in flags:
        warnings.append("💀 Long Squeeze aktif — HINDARI LONG, pertimbangkan SHORT")
    if "LONG_SQUEEZE_EXHAUSTED" in flags:
        warnings.append("🔄 Long Squeeze selesai — setup reversal/bounce terbentuk")
    if "SHORT_FLOW_CONFLICT" in flags:
        warnings.append("⚠️ Flow konflik — CVD bullish menahan validitas short")
    # CVD-Vol warning dari contexts
    for c in r.get("contexts", []):
        if "CVD-Vol Inconsistent" in c:
            warnings.append("⚠️ CVD momentum memudar — volume tidak konfirmasi")
            break
        elif "CVD-Vol Lemah" in c:
            warnings.append("🟡 CVD momentum lemah — konfirmasi tipis")
            break
    # FR velocity warning
    for c in r.get("contexts", []):
        if "FOMO akut" in c:
            warnings.append("💥 FR naik cepat — potensi blow-off top")
            break
        elif "Shorts menumpuk" in c:
            warnings.append("🔫 FR turun cepat — squeeze fuel bertambah")
            break

    # ── Rakit pesan ───────────────────────────────────────────────────
    sep = "─" * 34

    # Baris regime adj — tampilkan hanya kalau ada perbedaan dengan score asli
    score_adj = r.get("score_regime_adj", score) if signal_type == "LONG" else score
    if regime_ctx and abs(score_adj - score) >= 0.5:
        diff     = score_adj - score
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        regime_line = (
            f"<i>┗ Regime adj: {score:.1f} {diff_str} = {score_adj:.1f} "
            f"({regime_ctx.regime})</i>"
        )
    else:
        regime_line = ""

    lines = [
        f"{grade_emoji} {grade_label}",
        "",
        f"<b>📌 #{sym}USDT</b>  │  <code>${pr:,.5g}</code>",
        f"⏰ {now}",
        sep,
        f"<b>🏆 {score:.1f} / 100  [ {grade} ]</b>",
    ]
    if regime_line:
        lines.append(regime_line)
    lines.append("")
    
    if sub_ctx:
        lines.append(sub_ctx)

    lines.append(sep)

    # Target hanya tampil kalau ada
    if t1 or t2 or iv:
        if t1:
            lines.append(f"📈 Target 1  :  <code>{_fmt(t1)}</code>{_pct(t1)}")
        if t2:
            lines.append(f"📈 Target 2  :  <code>{_fmt(t2)}</code>{_pct(t2)}")
        if iv:
            lines.append(f"🛑 Invalidasi:  <code>{_fmt(iv)}</code>{_pct(iv)}")
        lines.append(sep)

    lines.append(f"🎯 {signals_str}")

    # Warning di paling bawah
    if warnings:
        lines.append(sep)
        for w in warnings:
            lines.append(w)

    return "\n".join(lines)


def build_startup_message() -> str:
    cycle_lines = "\n".join(
        f"  {i+1}. {c['label']}: max {c['max_coins']} koin, "
        f"min vol ${c['min_volume']/1e6:.0f}M"
        for i, c in enumerate(CONFIG["CYCLES"])
    )
    dashboard_line = (
        f"🖥 Dashboard: <code>{DASHBOARD_URL}</code>\n"
        if DASHBOARD_URL
        else "🖥 Dashboard: <b>disabled</b> (DASHBOARD_URL belum diset)\n"
    )
    return (
        "🤖 <b>AKSA Microstructure Screener v2.0 aktif!</b>\n\n"
        "📐 Engine  : <b>Interdependency Scoring Matrix</b>\n"
        "📡 Data    : Spot CVD + Futures CVD + OI + FR + VWAP\n"
        f"{dashboard_line}"
        f"⏱ Interval: <b>{CONFIG['SCAN_INTERVAL_MIN']} menit per siklus</b>\n"
        f"🎯 Min Score: <b>{CONFIG['MIN_SCORE']}/100</b>\n\n"
        f"<b>🔄 Sistem 3 Siklus:</b>\n{cycle_lines}\n\n"
        "<b>Skenario Aktif:</b>\n"
        "  🟢 A — Organic Spot Accumulation (max +40)\n"
        "  🟡 B — Speculative Futures Rally  (max +15)\n"
        "  🔫 C — Short Squeeze Fuel         (+25)\n"
        "  💀 D — Exhaustion Penalty         (−35)\n"
        "  🪤 E — Distribution Trap          (cap 30)"
    )
