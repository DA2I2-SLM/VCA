"""
Export CSI300 / CSI500 daily OHLCV from a local Qlib bundle into the per-symbol
CSV format the evaluation pipeline already consumes (`{sym}_1d.csv`, columns:
timestamps,open,high,low,close,volume,amount), so eval/compute_baseline.py and
finetune/dataset.py need no dataset-specific loader.

This script is an EXPORTER, not a downloader: it reads an already-installed
Qlib bundle and cannot fetch anything. Install the bundle once with

    python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

Pass --config to export exactly the universe a shipped eval config declares --
that list is the reproducible one. Without --config the script falls back to
picking the --n-symbols symbols with the most complete history, which is
data-dependent: a different bundle vintage yields a different universe and
therefore different numbers.
"""
import argparse
import os

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.data import D

DEFAULT_QLIB_DIR = {
    'cn': '~/.qlib/qlib_data/cn_data',
    'us': '~/.qlib/qlib_data/us_data',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None,
                    help='Eval config whose `symbols` and `data_dir` define the export '
                         '(reproducible; overrides --n-symbols and --out)')
    ap.add_argument('--qlib-dir', default=None,
                    help='Qlib bundle provider_uri (default: ~/.qlib/qlib_data/{cn,us}_data)')
    ap.add_argument('--out', default=None, help='Output directory for the CSVs')
    ap.add_argument('--n-symbols', type=int, default=30,
                    help='Fallback when --config is absent: keep this many symbols by history length')
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2020-09-25',
                    help='Bundle-dependent: the vintage used for the paper ends here')
    ap.add_argument('--market', default='csi300')
    ap.add_argument('--region', default='cn', choices=['cn', 'us'])
    a = ap.parse_args()

    want, out_dir = None, a.out
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        want = [s.lower() for s in cfg['symbols']]
        out_dir = a.out or os.path.expandvars(cfg['data_dir'])
    if out_dir is None:
        raise SystemExit('need --out or --config (to take data_dir from it)')
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    qdir = os.path.expanduser(a.qlib_dir or DEFAULT_QLIB_DIR[a.region])
    if not os.path.isdir(qdir):
        raise SystemExit(
            f'Qlib bundle not found at {qdir}. Install it with:\n'
            f'  python -m qlib.run.get_data qlib_data --target_dir {qdir} --region {a.region}')
    qlib.init(provider_uri=qdir, region=a.region)

    insts = D.instruments(market=a.market)
    fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
    df = D.features(insts, fields, start_time=a.start, end_time=a.end, freq='day')
    print(f'loaded {df.index.get_level_values(0).nunique()} symbols, {len(df)} rows from {qdir}')

    if want is not None:
        have = {s.lower(): s for s in df.index.get_level_values(0).unique()}
        missing = [s for s in want if s not in have]
        keep = [have[s] for s in want if s in have]
        if missing:
            print(f'WARNING: {len(missing)} of {len(want)} config symbols absent from this '
                  f'bundle vintage, e.g. {missing[:5]}')
    else:
        counts = df.groupby(level=0).size().sort_values(ascending=False)
        nkeep = min(a.n_symbols, len(counts))
        keep = counts.head(nkeep).index.tolist()
        print(f'keeping {nkeep} of {len(counts)} symbols by history length '
              f'({counts.iloc[nkeep - 1]}..{counts.iloc[0]} bars)')

    written = []
    for sym in keep:
        s = df.loc[sym].copy().sort_index()
        fac = s['$factor'].fillna(1.0)
        out = pd.DataFrame({
            'timestamps': s.index.strftime('%Y-%m-%d %H:%M:%S'),
            'open':   (s['$open'] * fac).values,
            'high':   (s['$high'] * fac).values,
            'low':    (s['$low'] * fac).values,
            'close':  (s['$close'] * fac).values,
            'volume': s['$volume'].fillna(0.0).values,
            'amount': (s['$close'] * s['$volume']).fillna(0.0).values,
        })
        out = out[np.isfinite(out['close']) & (out['close'] > 0)]
        out.to_csv(os.path.join(out_dir, f'{sym.lower()}_1d.csv'), index=False)
        written.append(sym.lower())
    print(f'wrote {len(written)} CSVs to {out_dir}')


if __name__ == '__main__':
    main()
