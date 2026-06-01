from __future__ import annotations

import os
import warnings
from datetime import timezone, timedelta
from typing import Dict

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

TZ_WIB = timezone(timedelta(hours=7))

# ── Dashboard Simulation ────────────────────────────────────────────────
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").rstrip("/")

# ═══════════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION  ─ Edit values below
# ═══════════════════════════════════════════════════════════════════════
CONFIG: Dict = {
    # ── Telegram ─────────────────────────────────────────────────────────
    # Ambil dari environment variable agar tidak hardcoded di source code.
    # Set sebelum run: export TELEGRAM_TOKEN="..." TELEGRAM_CHAT_ID="..."
    "TELEGRAM_TOKEN":   os.environ.get("TELEGRAM_TOKEN",   ""),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", ""),

    # ── 3-Cycle Settings ──────────────────────────────────────────────────
    # Setiap siklus punya batas koin dan minimum volume sendiri.
    # Koin dari siklus sebelumnya TIDAK akan muncul lagi.
    "CYCLES": [
        {"label": "Siklus 1 (Large Cap)",  "max_coins": 700, "min_volume": 10_000_000},
        {"label": "Siklus 2 (Mid Cap)",    "max_coins": 700, "min_volume":  3_000_000},
        {"label": "Siklus 3 (Small Cap)",  "max_coins": 700, "min_volume":  50_000},
    ],
    # Total pool yang di-fetch dari Binance sebelum dipartisi
    "TOTAL_POOL":         700,

    # ── Scan interval (berlaku untuk SETIAP siklus, bukan total) ─────────
    "SCAN_INTERVAL_MIN":  0.20,
    "MIN_SCORE":          70,

    # ── Lookback & Timeframe ──────────────────────────────────────────────
    "TIMEFRAME":          "1h",
    "LOOKBACK_N":         48,
    "FETCH_LIMIT":        100,

    # ── Threshold Scoring Matrix ──────────────────────────────────────────
    "VWAP_OVEREXT_PCT":   5.0,    # 3% → 5% (lebih longgar untuk crypto)
    "VWAP_DANGER_PCT":    8.0,    # > 8% = sangat overextended / bahaya
    "VWAP_SIDEWAYS_PCT":  1.5,
    "OI_PARABOLIC_PCT":   15.0,   # 10% → 15% (konfirmasi bandar baru)
    "OI_NOISE_PCT":       1.5,    # < 1.5% = noise, abaikan untuk squeeze
    "OI_IGNITION_MIN":    1.5,    # 1.5%-4% = goldilocks squeeze range
    "OI_IGNITION_MAX":    4.0,
    "OI_EXHAUSTED_PCT":   8.0,    # > 8% turun = squeeze exhausted
    "OI_SQUEEZE_PCT":    -1.5,    # threshold squeeze (dipakai di OI modifier)
    "SQUEEZE_MIN_FUEL":   45,
    "FR_OVERLEVERAGE":    0.0001,
    "FR_NEGATIVE":        0.0,
    "CVD_SPOT_FLAT":      5.0,
    "CVD_MIN_VOL_RATIO":  0.15,   # CVD diabaikan jika vol < 15% dari rata-rata

    # ── Concurrency & Network ─────────────────────────────────────────────
    "CONCURRENT_TASKS":   25,
    "REQUEST_TIMEOUT":    15,
    "POOL_RETRY_ATTEMPTS": 3,
    "POOL_EMPTY_SLEEP_SEC": 120,
    "MSG_DELAY":          0.6,
    "SSL_VERIFY":         False,
}

# ── API Base URLs ─────────────────────────────────────────────────────
SPOT_BASE    = os.environ.get("SPOT_BASE", "https://api.binance.com").rstrip("/")
FUTURES_BASE = os.environ.get("FUTURES_BASE", "https://fapi.binance.com").rstrip("/")
TELEGRAM_API = "https://api.telegram.org"

# ── Grade scale ───────────────────────────────────────────────────────
GRADE_SCALE = [
    (85, "A+", "🔥 PRIME SIGNAL"),
    (70, "A",  "✅ STRONG SIGNAL"),
    (55, "B+", "🟡 DECENT SIGNAL"),
    (40, "B",  "📊 WATCH SIGNAL"),
    (25, "C",  "⚠️ WEAK SIGNAL"),
    (0,  "D",  "⚪ NO SIGNAL"),
]

# ── Timeframe ke menit ────────────────────────────────────────────────
TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360,
    "8h": 480, "12h": 720, "1d": 1440,
}

# ── OI period mapping (timeframe → Binance OI period string) ─────────
OI_PERIOD_MAP = {
    "1m": "5m", "3m": "5m", "5m": "5m",
    "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "4h", "8h": "4h", "12h": "4h",
    "1d": "4h",
}
