from __future__ import annotations

import asyncio
import sys

from .config import CONFIG
from .fetchers import create_session, fetch_full_symbol_pool, partition_symbols
from .logging_setup import log
from .notifier import build_startup_message, send_telegram
from .scanner import run_scan_batch


async def main_async() -> None:
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║      AKSA MICROSTRUCTURE SCREENER  v2.0                  ║")
    log.info("║      3-Cycle System │ Spot + Futures Async               ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    cycles_cfg = CONFIG["CYCLES"]
    log.info("  Konfigurasi Siklus:")
    for i, c in enumerate(cycles_cfg, 1):
        log.info(f"    Siklus {i}: max {c['max_coins']} koin, min vol ${c['min_volume']/1e6:.0f}M")

    async with create_session() as session:
        await send_telegram(session, build_startup_message())

        full_cycle_count = 0   # berapa kali 3 siklus penuh selesai
        cycle_idx        = 0   # 0, 1, 2 → berputar terus
        partitions       = []  # [list_siklus1, list_siklus2, list_siklus3]

        while True:
            # ── Refresh partisi setiap awal putaran baru (setiap 3 siklus) ──
            if cycle_idx == 0:
                full_cycle_count += 1
                log.info(f"\n{'═'*60}")
                log.info(f"  🔄 Putaran ke-{full_cycle_count} — Refresh daftar koin dari Binance")
                log.info(f"{'═'*60}")
                pool       = await fetch_full_symbol_pool(session)
                partitions = partition_symbols(pool)

            current_cfg    = cycles_cfg[cycle_idx]
            current_batch  = partitions[cycle_idx] if cycle_idx < len(partitions) else []
            cycle_label    = current_cfg["label"]

            log.info(f"\n{'─'*60}")
            log.info(f"  ▶ {cycle_label} ({len(current_batch)} koin)")
            log.info(f"{'─'*60}")

            if not current_batch:
                log.warning(f"  ⚠️ {cycle_label}: batch kosong, skip.")
            else:
                try:
                    await run_scan_batch(session, current_batch, cycle_label)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error(f"Siklus crash [{cycle_label}]: {exc}", exc_info=True)
                    await send_telegram(
                        session,
                        f"⚠️ <b>Error pada {cycle_label}:</b>\n"
                        f"<code>{str(exc)[:300]}</code>",
                    )

            # Maju ke siklus berikutnya
            cycle_idx = (cycle_idx + 1) % len(cycles_cfg)

            wait_sec = CONFIG["SCAN_INTERVAL_MIN"] * 60
            log.info(f"\n  ⏳ Jeda {CONFIG['SCAN_INTERVAL_MIN']} menit sebelum siklus berikutnya…\n")
            await asyncio.sleep(wait_sec)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log.info("\n⛔ Dihentikan oleh user (Ctrl+C).")
    except Exception as exc:
        log.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
