"""Entry point untuk mode GitHub Actions: loop scan per-siklus (bukan 3 siklus
sekaligus) selama job masih dalam budget waktu, lalu exit.

Kenapa per-siklus, bukan 3 siklus berurutan: tiap siklus individual selesai
di bawah 1 menit (Siklus 1 ~16s, Siklus 2 ~14s, Siklus 3 ~44s dari hasil tes),
sedangkan 3 siklus dijumlah berurutan (~75s) melebihi 1 menit. GitHub Actions
tidak bisa memicu job baru lebih cepat dari 5 menit sekali, jadi satu job yang
dipicu tiap 5 menit ini loop terus secara internal (ganti siklus tiap
putaran) sampai mendekati batas SCAN_BUDGET_SEC, supaya interval antar-scan
individual senyata mungkin di bawah 1 menit tanpa melanggar batas trigger
GitHub.

Beda dari main.py (mode lama, proses nyala terus dengan sleep antar-siklus)
- script ini dipanggil berkala lewat cron (lihat .github/workflows/scan.yml).
State cooldown alert dipertahankan lewat file JSON + actions/cache biar tidak
kirim sinyal dobel setiap run.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from axanctum.config import CONFIG
from axanctum.fetchers import create_session, fetch_full_symbol_pool, partition_symbols
from axanctum.logging_setup import log
from axanctum.notifier import build_startup_message, send_telegram
from axanctum.scanner import run_scan_batch
from axanctum.state import _ALERT_COOLDOWN

STATE_FILE = "alert_cooldown_state.json"
# Total waktu loop internal sebelum job berhenti sendiri (detik). Default
# 270s (4.5 menit) - sisakan buffer sebelum trigger berikutnya (5 menit) dan
# sebelum timeout job GitHub Actions (14 menit di scan.yml).
SCAN_BUDGET_SEC = float(os.environ.get("SCAN_BUDGET_SEC", 270))
# Refresh daftar koin dari OKX tiap berapa detik (tidak perlu tiap putaran -
# daftar top-volume tidak berubah signifikan dalam hitungan menit).
POOL_REFRESH_SEC = float(os.environ.get("POOL_REFRESH_SEC", 240))


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
    t_job_start = time.monotonic()

    async with create_session() as session:
        if os.environ.get("SEND_STARTUP_MESSAGE", "0") == "1":
            await send_telegram(session, build_startup_message())

        pool = await fetch_full_symbol_pool(session)
        if not pool:
            log.warning("Pool OKX kosong — job berhenti.")
            return
        partitions = partition_symbols(pool)
        t_pool_fetched = time.monotonic()

        cycle_idx = 0
        pass_no = 0
        while True:
            elapsed = time.monotonic() - t_job_start
            if elapsed >= SCAN_BUDGET_SEC:
                log.info(f"[budget] {elapsed:.0f}s tercapai dari {SCAN_BUDGET_SEC:.0f}s — stop loop.")
                break

            # Refresh pool kalau sudah cukup lama sejak fetch terakhir.
            if time.monotonic() - t_pool_fetched >= POOL_REFRESH_SEC:
                fresh_pool = await fetch_full_symbol_pool(session)
                if fresh_pool:
                    pool = fresh_pool
                    partitions = partition_symbols(pool)
                t_pool_fetched = time.monotonic()

            cfg = cycles_cfg[cycle_idx]
            batch = partitions[cycle_idx] if cycle_idx < len(partitions) else []
            label = cfg["label"]
            pass_no += 1

            if not batch:
                log.warning(f"  {label}: batch kosong, skip.")
            else:
                log.info(f"[loop pass {pass_no}] elapsed={elapsed:.0f}s — {label}")
                try:
                    await run_scan_batch(session, batch, label)
                except Exception as exc:
                    log.error(f"Siklus crash [{label}]: {exc}", exc_info=True)
                    await send_telegram(
                        session,
                        f"Error pada {label}:\n{str(exc)[:300]}",
                    )
                # Simpan state tiap habis 1 siklus - kalau job kepotong
                # timeout, cooldown yang sudah terupdate tidak hilang.
                save_cooldown_state()

            cycle_idx = (cycle_idx + 1) % len(cycles_cfg)

    log.info(f"[done] total {pass_no} pass dalam {time.monotonic() - t_job_start:.0f}s")


def main() -> None:
    try:
        asyncio.run(scan_once_async())
    except Exception as exc:
        log.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
