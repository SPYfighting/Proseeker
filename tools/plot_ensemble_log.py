#!/usr/bin/env python3
"""
Parse epoch-level loss for each member from ensemble training log and plot.
Usage example:
    python tools/plot_ensemble_log.py \
        --log ../20251208_ensemble.log \
        --out ../outputs/plots/ensemble_loss.png
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EPOCH_LINE = re.compile(
    r"^Member (\d+) \| Epoch (\d+)/(\d+)\s+completed \| Avg Loss: ([0-9.]+) \| Time: ([0-9.]+) minutes"
)


def parse_log(log_path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_LINE.search(line)
            if not m:
                continue
            member = int(m.group(1))
            epoch = int(m.group(2))
            epoch_total = int(m.group(3))
            loss = float(m.group(4))
            minutes = float(m.group(5))
            records.append(
                {
                    "member": member,
                    "epoch": epoch,
                    "epochs_total": epoch_total,
                    "loss": loss,
                    "minutes": minutes,
                }
            )
    if not records:
        raise ValueError("No epoch-level loss information parsed from log, please check log format.")
    df = pd.DataFrame(records)
    df.sort_values(["member", "epoch"], inplace=True)
    return df


def plot(df: pd.DataFrame, out_path: str):
    """
    Plot training curve for each member separately with English labels.
    For example, when out_path=ensemble_loss.png, will output:
      ensemble_loss_member1.png, ..., ensemble_loss_member5.png
    Also save a summary CSV.
    """
    sns.set_theme(style="whitegrid", context="talk")

    base_dir = os.path.dirname(out_path) or "."
    stem, ext = os.path.splitext(os.path.basename(out_path))
    os.makedirs(base_dir, exist_ok=True)

    for member, sub in df.groupby("member"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        ax.plot(
            sub["epoch"],
            sub["loss"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"Member {member}",
        )

        ax.set_title(f"Member {member} Training Loss", fontsize=16, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(title="Member", loc="upper right")
        ax.grid(True, alpha=0.3)

        last = sub.iloc[-1]
        ax.text(
            last["epoch"] + 0.1,
            last["loss"],
            f"{last['loss']:.4f}",
            fontsize=10,
            va="center",
        )

        fig.tight_layout()
        member_out = os.path.join(base_dir, f"{stem}_member{member}{ext}")
        fig.savefig(member_out)
        plt.close(fig)
        print(f"Saved figure: {member_out}")

    csv_out = os.path.join(base_dir, f"{stem}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved data: {csv_out}")


def main():
    ap = argparse.ArgumentParser(description="Plot training curves for each member from ensemble log")
    ap.add_argument("--log", required=True, help="Ensemble training log path, e.g., 20251208_ensemble.log")
    ap.add_argument(
        "--out",
        required=False,
        default="ensemble_loss.png",
        help="Output image path (supports relative/absolute path)",
    )
    args = ap.parse_args()

    log_path = Path(args.log).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    out_path = Path(args.out).expanduser()
    df = parse_log(str(log_path))
    print(df.head())
    plot(df, str(out_path))


if __name__ == "__main__":
    main()

