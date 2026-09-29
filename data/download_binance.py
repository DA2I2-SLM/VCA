#!/usr/bin/env python3
"""
Download Binance 15-min spot klines for multiple symbols.

Uses Binance public REST endpoint (no API key required). Rate limit is
1200 req/min per IP. With 1500 bars/req and ~4 years of data per symbol
that's ~115 requests per symbol, so 20 symbols ≈ 2300 requests ≈ 2 min
under the limit. We use python-binance for retry / pagination.

Output: <data_dir>/<SYMBOL>_15min.csv with columns:
  timestamps, open, high, low, close, volume, amount

Usage:
    python data/download_binance.py --config configs/crypto_top20.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import yaml

try:
    from binance.client import Client
except ImportError:
    sys.exit("pip install python-binance  # then rerun")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('download')

# Binance kline column names (full schema; we keep subset).
KLINE_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'num_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore',
]

# Map our config frequency to Binance interval string.
FREQ_MAP = {
    '1m':   Client.KLINE_INTERVAL_1MINUTE,
    '5m':   Client.KLINE_INTERVAL_5MINUTE,
    '15m':  Client.KLINE_INTERVAL_15MINUTE,
    '1h':   Client.KLINE_INTERVAL_1HOUR,
    '4h':   Client.KLINE_INTERVAL_4HOUR,
    '1d':   Client.KLINE_INTERVAL_1DAY,
}


def download_symbol(client: Client, symbol: str, interval: str,
                    start: str, end: str, out_path: Path) -> int:
    """Download all klines for one symbol. Returns row count, 0 on failure."""
    if out_path.exists():
        df_existing = pd.read_csv(out_path, nrows=1)
        if len(df_existing) > 0:
            log.info(f"  {symbol}: skip (already at {out_path})")
            df_full = pd.read_csv(out_path)
            return len(df_full)

    log.info(f"  {symbol}: downloading {interval} {start} → {end}")
    t0 = time.time()
    try:
        klines = client.get_historical_klines(
            symbol, interval, start, end,
        )
    except Exception as e:
        log.error(f"  {symbol}: download failed: {e}")
        return 0

    if not klines:
        log.warning(f"  {symbol}: no data returned (possibly delisted or not active in period)")
        return 0

    df = pd.DataFrame(klines, columns=KLINE_COLS)

    # Convert types
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
        df[col] = df[col].astype(float)

    # Match Kronos schema: amount = quote_asset_volume (USDT-denominated turnover)
    df = df.rename(columns={'quote_asset_volume': 'amount'})
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    # Sanity: drop rows with zero volume (no trades, unreliable OHLC)
    n_before = len(df)
    df = df[df['volume'] > 0].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.info(f"  {symbol}: dropped {n_dropped} zero-volume rows ({n_dropped/n_before:.1%})")

    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    log.info(f"  {symbol}: {len(df)} rows saved in {elapsed:.1f}s")
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()}

    symbols: List[str] = cfg['symbols']
    interval = FREQ_MAP[cfg['frequency']]
    start = cfg['download_start']
    end = cfg['download_end']
    data_dir = Path(cfg['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {len(symbols)} symbols, {cfg['frequency']}, "
             f"{start} → {end} → {data_dir}/")

    # Public client (no API key needed for klines).
    client = Client(api_key='', api_secret='')

    results = {}
    for sym in symbols:
        out_path = data_dir / f"{sym.lower()}_{cfg['frequency']}.csv"
        n_rows = download_symbol(client, sym, interval, start, end, out_path)
        results[sym] = n_rows

    # Summary
    log.info("\n=== Download summary ===")
    successful = [s for s, n in results.items() if n > 0]
    failed = [s for s, n in results.items() if n == 0]
    log.info(f"  Successful: {len(successful)} / {len(symbols)}")
    if failed:
        log.warning(f"  Failed:     {failed}")
    log.info(f"  Total rows: {sum(results.values()):,}")

    # Save manifest
    manifest_path = data_dir / 'manifest.csv'
    pd.DataFrame([
        {'symbol': s, 'n_rows': n, 'status': 'ok' if n > 0 else 'failed'}
        for s, n in results.items()
    ]).to_csv(manifest_path, index=False)
    log.info(f"  Manifest:   {manifest_path}")


if __name__ == '__main__':
    main()