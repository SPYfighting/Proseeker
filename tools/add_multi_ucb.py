#!/usr/bin/env python3
"""
Append UCB columns at several exploration weights (beta) to a predictions file.

UCB(c) = mean_score(c) + beta * sqrt(total_variance(c))

This is an optional convenience helper; candidate selection itself can also be
done manually (e.g. in a spreadsheet) from the `mean_score` / `total_variance`
columns produced by predict_with_uncertainty.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Append multi-beta UCB columns to a predictions CSV.")
    parser.add_argument(
        "--in_csv",
        default=str(Path("outputs") / "predictions_with_uncertainty_mutations.csv"),
        help="Input CSV (must contain 'mean_score' and 'total_variance' columns).",
    )
    parser.add_argument(
        "--out_csv",
        default=str(Path("outputs") / "predictions_with_uncertainty_mutations_with_multi_ucb.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.5, 1, 2, 3, 5, 8, 10],
        help="Exploration weights (beta) to compute UCB columns for.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)

    if "mean_score" not in df.columns or "total_variance" not in df.columns:
        raise SystemExit("Missing mean_score or total_variance column")

    total_std = np.sqrt(df["total_variance"])

    for b in args.betas:
        col_name = f"ucb_b{str(b).replace('.', '_')}"
        df[col_name] = df["mean_score"] + b * total_std

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("File written:", out_path)
    print("New columns:", [c for c in df.columns if c.startswith("ucb_b")])


if __name__ == "__main__":
    main()
