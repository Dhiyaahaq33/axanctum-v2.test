from __future__ import annotations

import asyncio
import ssl
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from .config import CONFIG, FUTURES_BASE, SPOT_BASE
from .logging_setup import log


def _build_ssl_ctx() -> ssl.SSLContext:
    """Buat SSL context dengan opsi bypass untuk ISP Indonesia."""
    ctx = ssl.create_default_context()
    if not CONFIG["SSL_VERIFY"]:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log.info("🔓 SSL verification disabled (ISP bypass mode)")
    return ctx


def create_session() -> aiohttp.ClientSession:
    """Buat aiohttp session dengan connector yang tepat."""
    import socket
    connector = aiohttp.TCPConnector(
        ssl=_build_ssl_ctx(),
        limit=60,
        limit_per_host=10,
        ttl_dns_cache=300,
        family=socket.AF_INET,   # paksa IPv4 — Binance blok IPv6 dari VPS
    )
    return aiohttp.ClientSession(connector=connector, trust_env=True)


async def _get(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict] = None,
    *,
    warn: bool = False,
) -> Optional[Any]:
    """Generic async GET → JSON. Returns None on any error."""
    try:
        timeout = aiohttp.ClientTimeout(total=CONFIG["REQUEST_TIMEOUT"])
        async with session.get(url, params=params, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            msg = f"HTTP {resp.status} | {url} | params={params}"
            if warn:
                log.warning(msg)
            else:
                log.debug(msg)
            return None
    except asyncio.TimeoutError:
        msg = f"Timeout: {url}"
        if warn:
            log.warning(msg)
        else:
            log.debug(msg)
        return None
    except Exception as exc:
        msg = f"GET error [{url}]: {exc}"
        if warn:
            log.warning(msg)
        else:
            log.debug(msg)
        return None


async def fetch_full_symbol_pool(
    session: aiohttp.ClientSession,
) -> List[Tuple[str, float]]:
    """
    Fetch seluruh pool simbol dari Binance Futures.
    Return: list of (symbol, volume_24h_usd), sudah diurutkan volume DESC.
    Diambil hingga TOTAL_POOL koin teratas.
    """
    try:
        attempts = int(CONFIG.get("POOL_RETRY_ATTEMPTS", 3))

        info = None
        for attempt in range(1, attempts + 1):
            info = await _get(
                session,
                f"{FUTURES_BASE}/fapi/v1/exchangeInfo",
                warn=True,
            )
            if info:
                break
            if attempt < attempts:
                wait = min(10 * attempt, 30)
                log.warning(
                    f"exchangeInfo kosong/timeout — retry {attempt}/{attempts} "
                    f"dalam {wait}s"
                )
                await asyncio.sleep(wait)
        if not info:
            log.warning(
                f"exchangeInfo tidak tersedia dari {FUTURES_BASE} — pool kosong."
            )
            return []

        perp_set = {
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        }

        # Jeda kecil agar Binance tidak drop request berikutnya
        # (exchangeInfo response besar → rate limit jika langsung dilanjut)
        await asyncio.sleep(3)

        tickers = None
        for attempt in range(1, attempts + 1):
            tickers = await _get(
                session,
                f"{FUTURES_BASE}/fapi/v1/ticker/24hr",
                warn=True,
            )
            if tickers:
                break
            if attempt < attempts:
                wait = min(10 * attempt, 30)
                log.warning(
                    f"ticker/24hr kosong/timeout — retry {attempt}/{attempts} "
                    f"dalam {wait}s"
                )
                await asyncio.sleep(wait)
        if not tickers:
            log.warning(f"Ticker tidak tersedia dari {FUTURES_BASE} — pool kosong.")
            return []

        # Volume minimum absolut = volume floor siklus terkecil
        min_vol_global = min(c["min_volume"] for c in CONFIG["CYCLES"])

        pool = [
            (t["symbol"], float(t.get("quoteVolume", 0) or 0))
            for t in tickers
            if t["symbol"] in perp_set
            and float(t.get("quoteVolume", 0) or 0) >= min_vol_global
        ]
        pool.sort(key=lambda x: x[1], reverse=True)
        pool = pool[: CONFIG["TOTAL_POOL"]]

        log.info(
            f"📦 Pool fetched: {len(pool)} simbol "
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
            f"  📋 {cfg['label']}: {len(batch)} koin "
            f"(min vol ${cfg['min_volume']/1e6:.0f}M, max {cfg['max_coins']})"
        )

    total = sum(len(p) for p in partitions)
    log.info(f"  ✅ Total unik ter-partisi: {total} koin dari {len(pool)} pool")
    return partitions


async def fetch_futures_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """
    Ambil klines Binance Futures (fapi) beserta taker buy volume.
    Binance kline field index:
      [0] open_time  [1] open  [2] high  [3] low  [4] close
      [5] volume     [9] taker_buy_base_asset_volume
    """
    raw = await _get(
        session,
        f"{FUTURES_BASE}/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not raw or len(raw) < 3:
        return None

    rows = []
    for k in raw:
        total_vol = float(k[5])
        taker_buy = float(k[9])
        rows.append(
            {
                "ts":             int(k[0]),
                "open":           float(k[1]),
                "high":           float(k[2]),
                "low":            float(k[3]),
                "close":          float(k[4]),
                "volume":         total_vol,
                "taker_buy_vol":  taker_buy,
                "taker_sell_vol": total_vol - taker_buy,
            }
        )
    return pd.DataFrame(rows)


async def fetch_spot_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """
    Ambil klines Binance SPOT beserta taker buy volume.
    Format kline identik dengan futures — field index sama.
    """
    raw = await _get(
        session,
        f"{SPOT_BASE}/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not raw or len(raw) < 3:
        return None

    rows = []
    for k in raw:
        total_vol = float(k[5])
        taker_buy = float(k[9])
        rows.append(
            {
                "ts":             int(k[0]),
                "open":           float(k[1]),
                "high":           float(k[2]),
                "low":            float(k[3]),
                "close":          float(k[4]),
                "volume":         total_vol,
                "taker_buy_vol":  taker_buy,
                "taker_sell_vol": total_vol - taker_buy,
            }
        )
    return pd.DataFrame(rows)


async def fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str,
) -> Tuple[Optional[float], float]:
    """
    Ambil funding rate terkini dan hitung Spot-Futures Basis.

    Basis = (MarkPrice − IndexPrice) / IndexPrice × 100
      Basis > 0 → Futures lebih mahal dari Spot → leverage driven ⚠️
      Basis < 0 → Spot memimpin → organik ✅
      Basis ≈ 0 → netral

    premiumIndex endpoint menyediakan keduanya dalam 1 request:
      markPrice  = harga mark futures
      indexPrice = harga index spot (sangat dekat dengan spot aktual)

    Returns: (funding_rate, basis_pct)
    """
    data = await _get(
        session,
        f"{FUTURES_BASE}/fapi/v1/premiumIndex",
        {"symbol": symbol},
    )
    if not data:
        return None, 0.0
    try:
        fr         = float(data.get("lastFundingRate", 0) or 0)
        mark_price = float(data.get("markPrice", 0) or 0)
        idx_price  = float(data.get("indexPrice", 0) or 0)

        if idx_price > 0:
            basis_pct = (mark_price - idx_price) / idx_price * 100
        else:
            basis_pct = 0.0

        return fr, round(basis_pct, 4)
    except (ValueError, TypeError):
        return None, 0.0


async def fetch_oi_history(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str,
    limit: int,
) -> Optional[List[float]]:
    """
    Ambil riwayat Open Interest dari Binance Futures.
    Mendukung period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    Return: list nilai OI dari lama → baru.
    """
    data = await _get(
        session,
        f"{FUTURES_BASE}/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit},
    )
    if not data or len(data) < 2:
        return None
    try:
        return [float(r["sumOpenInterest"]) for r in data]
    except (KeyError, TypeError, ValueError):
        return None
