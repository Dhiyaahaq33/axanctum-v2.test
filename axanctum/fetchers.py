from __future__ import annotations

import asyncio
import ssl
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from .config import CONFIG
from .logging_setup import log

OKX_BASE = "https://www.okx.com"

# Binance-style "BTCUSDT" dipertahankan sebagai format symbol internal di
# seluruh codebase (scanner.py, notifier.py) - supaya tidak perlu ubah logic
# scoring sama sekali. Modul ini yang menerjemahkan ke/dari format OKX.
#
# Kenapa pindah dari Binance ke OKX: Binance (Spot + Futures) memblokir semua
# request dari IP US (HTTP 451 "restricted location"), dan runner GitHub
# Actions selalu dapat IP US. OKX terverifikasi tidak diblokir dan punya data
# setara (candle, funding rate, open interest, taker buy/sell volume).

_BAR_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H",
    "12h": "12H", "1d": "1Dutc",
}


def _base_ccy(symbol: str) -> str:
    """'BTCUSDT' -> 'BTC'"""
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _to_okx_swap(symbol: str) -> str:
    return f"{_base_ccy(symbol)}-USDT-SWAP"


def _to_okx_spot(symbol: str) -> str:
    return f"{_base_ccy(symbol)}-USDT"


def _build_ssl_ctx() -> ssl.SSLContext:
    """Buat SSL context dengan opsi bypass untuk ISP Indonesia."""
    ctx = ssl.create_default_context()
    if not CONFIG["SSL_VERIFY"]:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log.info("SSL verification disabled (ISP bypass mode)")
    return ctx


def create_session() -> aiohttp.ClientSession:
    """Buat aiohttp session dengan connector yang tepat."""
    import socket
    connector = aiohttp.TCPConnector(
        ssl=_build_ssl_ctx(),
        limit=60,
        limit_per_host=10,
        ttl_dns_cache=300,
        family=socket.AF_INET,
    )
    return aiohttp.ClientSession(connector=connector, trust_env=True)


async def _get(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict] = None,
    *,
    warn: bool = False,
) -> Optional[Dict]:
    """Generic async GET to JSON body OKX (dict dengan key code/data). None kalau error."""
    try:
        timeout = aiohttp.ClientTimeout(total=CONFIG["REQUEST_TIMEOUT"])
        async with session.get(url, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                msg = f"HTTP {resp.status} | {url} | params={params}"
                (log.warning if warn else log.debug)(msg)
                return None
            body = await resp.json(content_type=None)
            if isinstance(body, dict) and body.get("code") not in (None, "0"):
                msg = f"OKX error {body.get('code')}: {body.get('msg')} | {url} | params={params}"
                (log.warning if warn else log.debug)(msg)
                return None
            return body
    except asyncio.TimeoutError:
        (log.warning if warn else log.debug)(f"Timeout: {url}")
        return None
    except Exception as exc:
        (log.warning if warn else log.debug)(f"GET error [{url}]: {exc}")
        return None


async def fetch_full_symbol_pool(
    session: aiohttp.ClientSession,
) -> List[Tuple[str, float]]:
    """
    Fetch seluruh pool USDT-margined perpetual swap dari OKX.
    Return: list of (symbol, volume_24h_usd) format Binance-style ('BTCUSDT'),
    sudah diurutkan volume DESC, dipotong hingga TOTAL_POOL koin teratas.
    """
    try:
        attempts = int(CONFIG.get("POOL_RETRY_ATTEMPTS", 3))

        data = None
        for attempt in range(1, attempts + 1):
            data = await _get(
                session,
                f"{OKX_BASE}/api/v5/market/tickers",
                {"instType": "SWAP"},
                warn=True,
            )
            if data and data.get("data"):
                break
            if attempt < attempts:
                wait = min(10 * attempt, 30)
                log.warning(
                    f"OKX tickers kosong/timeout - retry {attempt}/{attempts} dalam {wait}s"
                )
                await asyncio.sleep(wait)

        if not data or not data.get("data"):
            log.warning("OKX tickers tidak tersedia - pool kosong.")
            return []

        min_vol_global = min(c["min_volume"] for c in CONFIG["CYCLES"])

        pool = []
        for t in data["data"]:
            inst_id = t.get("instId", "")
            if not inst_id.endswith("-USDT-SWAP"):
                continue
            base = inst_id[: -len("-USDT-SWAP")]
            symbol = f"{base}USDT"
            # volCcy24h dari OKX itu base-currency (mis. jumlah BTC), bukan USD -
            # kalikan harga terakhir dulu supaya sepadan dengan quoteVolume Binance.
            last_price = float(t.get("last", 0) or 0)
            vol_base = float(t.get("volCcy24h", 0) or 0)
            vol_usd = vol_base * last_price
            if vol_usd >= min_vol_global:
                pool.append((symbol, vol_usd))

        pool.sort(key=lambda x: x[1], reverse=True)
        pool = pool[: CONFIG["TOTAL_POOL"]]

        log.info(
            f"Pool fetched (OKX): {len(pool)} simbol "
            f"(min vol ${min_vol_global/1e6:.0f}M, max {CONFIG['TOTAL_POOL']} koin)"
        )
        return pool

    except Exception as exc:
        log.error(f"fetch_full_symbol_pool error: {exc}")
        return []


def partition_symbols(
    pool: List[Tuple[str, float]],
) -> List[List[str]]:
    """
    Partisi pool simbol menjadi 3 siklus tanpa overlap.

    Algoritma:
      - Pool sudah diurutkan volume DESC (terbesar duluan).
      - Siklus 1 ambil dari atas (volume >= min_vol_1), max N koin.
      - Siklus 2 ambil dari sisa (volume >= min_vol_2), max N koin.
      - Siklus 3 ambil dari sisa (volume >= min_vol_3), max N koin.
      - Koin yang sudah masuk siklus sebelumnya TIDAK bisa masuk lagi.

    Return: list of 3 list (masing-masing = simbol untuk satu siklus).
    """
    cycles_cfg  = CONFIG["CYCLES"]
    used        = set()
    partitions  = []

    for cfg in cycles_cfg:
        batch = []
        for sym, vol in pool:
            if sym in used:
                continue
            if vol < cfg["min_volume"]:
                continue
            batch.append(sym)
            if len(batch) >= cfg["max_coins"]:
                break

        for sym in batch:
            used.add(sym)

        partitions.append(batch)
        log.info(
            f"  {cfg['label']}: {len(batch)} koin "
            f"(min vol ${cfg['min_volume']/1e6:.0f}M, max {cfg['max_coins']})"
        )

    total = sum(len(p) for p in partitions)
    log.info(f"  Total unik ter-partisi: {total} koin dari {len(pool)} pool")
    return partitions


async def _fetch_taker_volume_map(
    session: aiohttp.ClientSession,
    ccy: str,
    inst_type: str,
) -> Dict[int, Tuple[float, float]]:
    """
    Ambil taker buy/sell volume per-jam untuk sebuah currency dari OKX
    (endpoint ini agregat per-ccy, bukan per-instId - cukup akurat untuk
    coin dengan satu pasar USDT dominan).
    Return: {timestamp_ms: (buy_vol_usd, sell_vol_usd)}
    """
    data = await _get(
        session,
        f"{OKX_BASE}/api/v5/rubik/stat/taker-volume",
        {"ccy": ccy, "instType": inst_type, "period": "1H"},
    )
    if not data or not data.get("data"):
        return {}
    out: Dict[int, Tuple[float, float]] = {}
    for row in data["data"]:
        try:
            ts = int(row[0])
            sell_vol = float(row[1])
            buy_vol = float(row[2])
            out[ts] = (buy_vol, sell_vol)
        except (IndexError, TypeError, ValueError):
            continue
    return out


async def _fetch_candles(
    session: aiohttp.ClientSession,
    inst_id: str,
    bar: str,
    limit: int,
) -> Optional[List[List[str]]]:
    data = await _get(
        session,
        f"{OKX_BASE}/api/v5/market/candles",
        {"instId": inst_id, "bar": bar, "limit": limit},
    )
    if not data or not data.get("data"):
        return None
    rows = list(data["data"])
    rows.reverse()  # OKX kirim newest-first, kita butuh oldest-first
    return rows


async def _fetch_klines_with_cvd(
    session: aiohttp.ClientSession,
    inst_id: str,
    ccy: str,
    taker_inst_type: str,
    interval: str,
    limit: int,
    *,
    base_vol_idx: int,
) -> Optional[pd.DataFrame]:
    """
    Gabungkan candle OHLCV OKX + taker buy/sell volume per-jam jadi satu
    DataFrame dengan kolom identik ke versi Binance lama (ts, open, high,
    low, close, volume, taker_buy_vol, taker_sell_vol).

    base_vol_idx: index volume base-currency di array candle OKX
      (SWAP -> volCcy = index 6, SPOT -> vol = index 5, lihat catatan di
      fetch_futures_klines/fetch_spot_klines).
    """
    bar = _BAR_MAP.get(interval, "1H")
    candles, taker_map = await asyncio.gather(
        _fetch_candles(session, inst_id, bar, limit),
        _fetch_taker_volume_map(session, ccy, taker_inst_type),
    )
    if not candles or len(candles) < 3:
        return None

    rows = []
    for c in candles:
        try:
            ts = int(c[0])
            total_vol = float(c[base_vol_idx])
        except (IndexError, TypeError, ValueError):
            continue
        buy_vol, sell_vol = taker_map.get(ts, (0.0, 0.0))
        if buy_vol == 0.0 and sell_vol == 0.0 and total_vol > 0:
            # Candle terbaru kadang belum ada di taker-volume history -
            # fallback netral (50/50) daripada di-drop.
            buy_vol = total_vol / 2
            sell_vol = total_vol / 2
        rows.append(
            {
                "ts":             ts,
                "open":           float(c[1]),
                "high":           float(c[2]),
                "low":            float(c[3]),
                "close":          float(c[4]),
                "volume":         total_vol,
                "taker_buy_vol":  buy_vol,
                "taker_sell_vol": sell_vol,
            }
        )
    if len(rows) < 3:
        return None
    return pd.DataFrame(rows)


async def fetch_futures_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """Ambil klines OKX perpetual SWAP (USDT-margined) + taker buy/sell volume."""
    inst_id = _to_okx_swap(symbol)
    ccy = _base_ccy(symbol)
    # OKX candle field index 6 = volCcy = volume base-currency untuk SWAP
    # (index 5 = vol = jumlah kontrak, bukan base-currency).
    return await _fetch_klines_with_cvd(
        session, inst_id, ccy, "CONTRACTS", interval, limit, base_vol_idx=6
    )


async def fetch_spot_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """Ambil klines OKX SPOT + taker buy/sell volume."""
    inst_id = _to_okx_spot(symbol)
    ccy = _base_ccy(symbol)
    # OKX candle field index 5 = vol = volume base-currency langsung untuk SPOT.
    return await _fetch_klines_with_cvd(
        session, inst_id, ccy, "SPOT", interval, limit, base_vol_idx=5
    )


async def fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str,
) -> Tuple[Optional[float], float]:
    """
    Ambil funding rate terkini dan hitung Spot-Futures Basis dari OKX.

    Basis = (MarkPrice - IndexPrice) / IndexPrice x 100
      Basis > 0 -> Futures lebih mahal dari Spot -> leverage driven
      Basis < 0 -> Spot memimpin -> organik

    Returns: (funding_rate, basis_pct)
    """
    inst_id_swap = _to_okx_swap(symbol)
    fr_data, mark_data, idx_data = await asyncio.gather(
        _get(session, f"{OKX_BASE}/api/v5/public/funding-rate", {"instId": inst_id_swap}),
        _get(session, f"{OKX_BASE}/api/v5/public/mark-price", {"instType": "SWAP", "instId": inst_id_swap}),
        _get(session, f"{OKX_BASE}/api/v5/market/index-tickers", {"instId": _to_okx_spot(symbol)}),
    )
    try:
        fr = None
        if fr_data and fr_data.get("data"):
            fr = float(fr_data["data"][0].get("fundingRate", 0) or 0)

        basis_pct = 0.0
        if mark_data and mark_data.get("data") and idx_data and idx_data.get("data"):
            mark_price = float(mark_data["data"][0].get("markPx", 0) or 0)
            idx_price = float(idx_data["data"][0].get("idxPx", 0) or 0)
            if idx_price > 0:
                basis_pct = (mark_price - idx_price) / idx_price * 100

        return fr, round(basis_pct, 4)
    except (ValueError, TypeError, IndexError, KeyError):
        return None, 0.0


async def fetch_oi_history(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str,
    limit: int,
) -> Optional[List[float]]:
    """
    Ambil riwayat Open Interest (dalam USD) dari OKX.
    Return: list nilai OI dari lama -> baru.
    """
    ccy = _base_ccy(symbol)
    okx_period = _BAR_MAP.get(period, "1H")
    data = await _get(
        session,
        f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume",
        {"ccy": ccy, "period": okx_period},
    )
    if not data or not data.get("data") or len(data["data"]) < 2:
        return None
    try:
        rows = list(reversed(data["data"]))  # OKX newest-first -> oldest-first
        rows = rows[-limit:] if limit > 0 else rows
        return [float(r[1]) for r in rows]
    except (IndexError, TypeError, ValueError):
        return None
