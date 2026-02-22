#!/usr/bin/env python3
"""
Convert full-length amino acid sequences in `parent` / `child` columns from
B-GNN prediction phase `predictions_with_uncertainty.csv` to standard mutation
notation: "WT amino acid + mutation position (1-based) + mutated amino acid".
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

WT_SEQUENCE = (
    "SLPLNMPALEMPAFIATKVSQYSCQRKTTLNNYNKKFTDAFEVMAENYEFKENEIFCLEFLRAASLLKSLPFSVTRMKDIQGLPCVGDQVRDIIEEIIEEGESSRVNEVLNDERYKAFKQFTSVFGVGVKTSEKWYRMGLRTVEEVKADKTLKLSKMQKAGLLYYEDLVSCVSKAEADAVSLIVKNTVCTFLPDALVTITGGFRRGKNIGHDIDFLITNPGPREDDELLHKVIDLWKKQGLLLYCDIIESTFVKEQLPSRKVDAMDHFQKCFAILKLYQPRVDNSTCNTSEQLEMAEVKDWKAIRVDLVITPFEQYPYALLGWTGSRQFGRDLRRYAAHERKMILDNHGLYDRRKRIFLKAGSEEEIFAHLGLDYVEPWERNA"
)

POSITION_OFFSET = 130


def seq_to_mutation_notation(seq: str, wt_seq: str = WT_SEQUENCE, position_offset: int = POSITION_OFFSET) -> str:
    """
    Convert a complete sequence relative to WT sequence to standard mutation notation:
        - Single point: "A200G"
        - Multiple points: "A200G;L205R"
        - No mutation: "WT"
    """
    if not isinstance(seq, str):
        return ""

    seq = seq.strip()
    if not seq:
        return ""

    if len(seq) != len(wt_seq):
        raise ValueError(
            f"Sequence length inconsistent with WT: len(seq)={len(seq)}, len(WT)={len(wt_seq)}.\n"
            "Please confirm using the same protein sequence."
        )

    mutations: list[str] = []
    for i, (wt_aa, aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt_aa != aa:
            external_pos = i + position_offset
            mutations.append(f"{wt_aa}{external_pos}{aa}")

    if not mutations:
        return "WT"
    return ";".join(mutations)


def convert_csv(in_csv: str, out_csv: str) -> None:
    """Read predictions_with_uncertainty.csv, convert parent/child columns and output new CSV."""
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input file not found: {in_csv}")

    df = pd.read_csv(in_csv)

    required_cols = {"parent", "child"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    print(f"Read {len(df)} rows, columns: {list(df.columns)}")
    print("Converting parent / child sequences to mutation notation (relative to WT)...")

    df["parent"] = df["parent"].apply(seq_to_mutation_notation)
    df["child"] = df["child"].apply(seq_to_mutation_notation)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_csv, index=False)
    print(f"Conversion completed, saved to: {out_csv}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert parent/child full-length sequences in predictions_with_uncertainty.csv "
            "to 'WT amino acid + mutation position + mutated amino acid' notation."
        )
    )
    parser.add_argument(
        "--in_csv",
        default=str(Path("outputs") / "predictions_with_uncertainty.csv"),
        help="Input predictions_with_uncertainty.csv path",
    )
    parser.add_argument(
        "--out_csv",
        default=str(Path("outputs") / "predictions_with_uncertainty_mutations.csv"),
        help="Output new CSV path (default: outputs/predictions_with_uncertainty_mutations.csv)",
    )

    args = parser.parse_args(argv)
    convert_csv(args.in_csv, args.out_csv)


if __name__ == "__main__":
    main()


