"""Entry point untuk mode GitHub Actions: scan sekali (3 siklus), kirim alert
Telegram untuk sinyal baru, lalu exit.

Beda dari main.py (mode lama, proses nyala terus dengan sleep antar-siklus) -
script ini dipanggil berkala lewat cron (lihat .github/workflows/scan.yml),
karena GitHub Actions tidak bisa menahan proses persisten. State cooldown
alert dipertahankan lewat file JSON + actions/cache biar tidak kirim sinyal
dobel setiap run.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from axanctum.config import CONFIG
from axanctum.fetchers import create_session, fetch_full_symbol_pool, partition_symbols
from axanctum.logging_setup import log
from axanctum.notifier import build_startup_message, send_telegram
from axanctum.scanner import run_scan_batch
from axanctum.state import _ALERT_COOLDOWN

STATE_FILE = "alert_cooldown_state.json"


def load_cooldown_state() -> None:
    if not os.path.exists(STATE_FILE):
        log.info(f"[state] {STATE_FILE} belum ada — mulai dari cooldown kosong.")
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _ALERT_COOLDOWN.update(data)
        log.info(f"[state] cooldown state dimuat: {len(data)} symbol")
    except Exception as exc:
        log.warning(f"[state] gagal load {STATE_FILE}: {exc}")


def save_cooldown_state() -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_ALERT_COOLDOWN, f)
        log.info(f"[state] cooldown state disimpan: {len(_ALERT_COOLDOWN)} symbol")
    except Exception as exc:
        log.warning(f"[state] gagal simpan {STATE_FILE}: {exc}")


async def scan_once_async() -> None:
    load_cooldown_state()

    cycles_cfg = CONFIG["CYCLES"]
    async with create_session() as session:
        if os.environ.get("SEND_STARTUP_MESSAGE", "0") == "1":
            await send_telegram(session, build_startup_message())

        pool = await fetch_full_symbol_pool(session)
        if not pool:
            log.warning("Pool OKX kosong — skip run ini.")
            return
        partitions = partition_symbols(pool)

        for cycle_idx, cfg in enumerate(cycles_cfg):
            batch = partitions[cycle_idx] if cycle_idx < len(partitions) else []
            label = cfg["label"]
            if not batch:
                log.warning(f"  {label}: batch kosong, skip.")
                continue
            try:
                await run_scan_batch(session, batch, label)
            except Exception as exc:
                log.error(f"Siklus crash [{label}]: {exc}", exc_info=True)
                await send_telegram(
                    session,
                    f"Error pada {label}:\n{str(exc)[:300]}",
                )

    save_cooldown_state()


def main() -> None:
    try:
        asyncio.run(scan_once_async())
    except Exception as exc:
        log.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
