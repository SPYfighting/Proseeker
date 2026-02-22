#!/usr/bin/env python3
"""
Parse an Optuna-like hyperparameter search log and produce publication-ready plots.

Example:
    python tools/plot_hparam_search_log.py \
        --log ../20251208_hypersearch_sample0.2.log \
        --out_prefix ../outputs/plots/hparam_search
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Example line:
# [I 2025-12-08 06:04:50,732] Trial 2 finished with value: 0.0571 and parameters:
# {'learning_rate': 4.53e-05, 'dropout_rate': 0.03639}. Best is trial 2 with value: 0.0571.
TRIAL_SUMMARY_RE = re.compile(
    r"Trial\s+(\d+)\s+finished with value:\s*([0-9.eE+-]+)\s+and parameters:\s*\{([^}]+)\}"
)


def parse_param_block(block: str) -> Dict[str, float]:
    """
    Parse the inside of the {...} parameter block into a dict.
    Expect comma-separated key: value pairs, simple floats.
    """
    params: Dict[str, float] = {}
    # e.g. "'learning_rate': 1.23e-05, 'dropout_rate': 0.12"
    for part in block.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        key = k.strip().strip("'\"")
        val_str = v.strip().strip("'\"")
        try:
            val = float(val_str)
        except ValueError:
            # leave non-numeric as-is
            continue
        params[key] = val
    return params


def parse_log(log_path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = TRIAL_SUMMARY_RE.search(line)
            if not m:
                continue
            trial_idx = int(m.group(1))
            value = float(m.group(2))
            param_block = m.group(3)
            params = parse_param_block(param_block)
            rec: Dict = {"trial": trial_idx, "value": value}
            rec.update(params)
            records.append(rec)

    if not records:
        raise ValueError("No trial summary lines were parsed from the log. Check log format.")

    df = pd.DataFrame(records)
    df = df.sort_values("trial").reset_index(drop=True)
    # Cumulative best objective (assuming lower is better)
    df["best_so_far"] = df["value"].cummin()
    return df


def plot_objective(df: pd.DataFrame, out_path: str):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(df["trial"], df["value"], marker="o", linestyle="-", linewidth=2, markersize=6, label="Trial value")
    ax.plot(df["trial"], df["best_so_far"], linestyle="--", linewidth=2, color="tab:green", label="Best so far")

    ax.set_title("Hyperparameter Search Objective per Trial", fontsize=16, weight="bold")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Validation objective")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Annotate current best at the last trial
    best_idx = df["value"].idxmin()
    best_row = df.loc[best_idx]
    ax.scatter([best_row["trial"]], [best_row["value"]], color="red", zorder=5)
    ax.text(
        best_row["trial"] + 0.2,
        best_row["value"],
        f"best={best_row['value']:.4f}",
        fontsize=10,
        va="center",
        color="red",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved objective plot: {out_path}")


def plot_hp_scatter(df: pd.DataFrame, out_path: str):
    # Require at least learning_rate and dropout_rate to make this plot meaningful
    if "learning_rate" not in df.columns or "dropout_rate" not in df.columns:
        print("Warning: learning_rate or dropout_rate not found in parsed data; skip scatter plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    sc = ax.scatter(
        df["learning_rate"],
        df["dropout_rate"],
        c=df["value"],
        cmap="viridis_r",
        s=60,
        edgecolor="k",
    )

    ax.set_xscale("log")
    ax.set_title("Hyperparameter Landscape", fontsize=16, weight="bold")
    ax.set_xlabel("learning_rate (log scale)")
    ax.set_ylabel("dropout_rate")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Validation objective (lower is better)")

    # highlight best trial
    best_idx = df["value"].idxmin()
    best_row = df.loc[best_idx]
    ax.scatter(
        [best_row["learning_rate"]],
        [best_row["dropout_rate"]],
        color="red",
        s=90,
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label="Best trial",
    )
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved hyperparameter scatter: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot hyperparameter search results from log file.")
    ap.add_argument("--log", required=True, help="Path to hyperparameter search log.")
    ap.add_argument(
        "--out_prefix",
        required=False,
        default="hparam_search",
        help="Output file prefix (without extension).",
    )
    args = ap.parse_args()

    log_path = Path(args.log).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = parse_log(str(log_path))
    print(df.head())

    out_prefix = Path(args.out_prefix)
    base_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    stem = out_prefix.stem

    os.makedirs(base_dir, exist_ok=True)

    # objective vs trial
    plot_objective(df, str(base_dir / f"{stem}_objective.png"))
    # hyperparameter scatter
    plot_hp_scatter(df, str(base_dir / f"{stem}_hp_scatter.png"))

    # save raw parsed data
    csv_out = base_dir / f"{stem}.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved parsed data: {csv_out}")


if __name__ == "__main__":
    main()


