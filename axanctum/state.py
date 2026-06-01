from __future__ import annotations

from typing import Dict, Tuple

# ── In-memory FR Velocity Tracker ────────────────────────────────────
# _FR_HISTORY_4H: simpan (fr_value, timestamp) per simbol
# FR Binance berubah setiap 4-8 jam — velocity diukur vs 4 jam lalu
_FR_HISTORY_4H: Dict[str, Tuple[float, float]] = {}
# Interval minimum (detik) sebelum FR history diupdate: 4 jam
_FR_VELOCITY_INTERVAL = 4 * 3600

# ── Alert Cooldown Tracker ────────────────────────────────────────────
# Hindari sinyal duplikat per simbol dalam satu window waktu.
# Format: {symbol: {"LONG": last_sent_ts, "SHORT": last_sent_ts}}
_ALERT_COOLDOWN: Dict[str, Dict[str, float]] = {}
# Cooldown minimum: 15 menit per simbol per tipe sinyal.
# Notifier menaikkan window ini menjadi minimal 1 candle aktif.
ALERT_COOLDOWN_SEC = 15 * 60
