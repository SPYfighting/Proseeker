#!/usr/bin/env python3
"""
Select Top-K candidates from predictions_with_uncertainty.csv with simple diversity constraints.
"""
import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import time

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def select_diverse_topk(df: pd.DataFrame, k: int, show_progress: bool = True) -> pd.DataFrame:
    from tqdm import tqdm
    df = df.copy()
    seqs = df.get('child', df.get('sequence'))
    if seqs is None:
        raise ValueError('Missing child/sequence column')
    df['_seq'] = seqs
    df = df.sort_values('ucb_score', ascending=False)

    if k >= len(df):
        return df.drop(columns=['_seq'])

    selected_idx = [df.index[0]]
    selected_seqs = [df.iloc[0]['_seq']]

    candidates = df.index.tolist()[1:]
    iterator = tqdm(range(k - 1), desc="Selecting candidates", disable=not show_progress) if k > 1 else range(k - 1)
    for _ in iterator:
        if not candidates:
            break
        best_i = None
        best_min_dist = -1
        for i in candidates:
            s = df.loc[i, '_seq']
            min_d = min(hamming(s, t) for t in selected_seqs)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)
        selected_seqs.append(df.loc[best_i, '_seq'])
        candidates.remove(best_i)

    return df.loc[selected_idx].drop(columns=['_seq']).sort_values('ucb_score', ascending=False)


def main():
    ap = argparse.ArgumentParser(description='Select Top-K acquisitions with diversity')
    ap.add_argument('--pred_csv', required=True, help='predictions_with_uncertainty.csv')
    ap.add_argument('--k', type=int, default=96)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print("=" * 80)
    print("Starting candidate selection (Top-K + Diversity)")
    print(f"Input file: {args.pred_csv} | K={args.k}")
    print("=" * 80)
    t0 = time.time()

    df = pd.read_csv(args.pred_csv)
    if 'ucb_score' not in df.columns:
        raise ValueError('Input missing ucb_score column')
    seq_col = 'child' if 'child' in df.columns else 'sequence' if 'sequence' in df.columns else None
    if seq_col is None:
        raise ValueError('Missing child/sequence column')
    print(f"Input samples: {len(df)} | Columns: {list(df.columns)} | Using sequence column: {seq_col}")
    print(f"Top 3 ucb_score: {df['ucb_score'].head(3).tolist()}")

    sel = select_diverse_topk(df, args.k, show_progress=True)
    print(f"\nSelected {len(sel)} candidates (Top {args.k})")
    print("Sample first 10 rows:")
    print(sel.head(10).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        sel.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")
    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.2f} minutes")


if __name__ == '__main__':
    main()
