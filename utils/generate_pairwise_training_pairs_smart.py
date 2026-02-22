#!/usr/bin/env python3
"""
Smart sampling version of pairwise training pairs generator

Input (default: data/labeled_data.csv):
- Required columns: `label` (activity value), `sequence` or `child` (sequence)

Output (default: data/training_pairs.csv):
- Columns: `parent`, `child`, `label` (difference)

Usage example (in B-GNN directory):
    # Method 1: Run as module (recommended)
    python -m utils.generate_pairwise_training_pairs_smart
    
    # Method 2: Direct run (ensure in B-GNN directory)
    python utils/generate_pairwise_training_pairs_smart.py
"""

import argparse
import itertools
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def stratify_by_activity(labels, n_strata=3):
    """
    Stratify samples into n_strata layers by activity value
    
    Returns:
        dict: {stratum_id: [indices]}
    """
    n = len(labels)
    sorted_indices = np.argsort(labels)
    
    strata = {}
    samples_per_stratum = n // n_strata
    
    for i in range(n_strata):
        start_idx = i * samples_per_stratum
        if i == n_strata - 1:
            end_idx = n
        else:
            end_idx = (i + 1) * samples_per_stratum
        
        strata[i] = sorted_indices[start_idx:end_idx].tolist()
    
    return strata


def calculate_pair_info_score(labels, i, j):
    """
    Calculate information score for pair (i, j)
    
    Information = |label_j - label_i| * (cross-strata bonus + boundary bonus)
    """
    delta = abs(labels[j] - labels[i])
    
    base_score = delta
    
    median_delta = np.median(np.abs(np.diff(np.sort(labels))))
    if delta > median_delta * 1.5:
        base_score *= 1.5
    
    return base_score


def smart_sample_pairs(
    seqs, labels, max_pairs, 
    min_pairs_per_sequence=5,
    n_strata=3,
    cross_strata_ratio=0.4,
    seed=42
):
    """
    Smart sampling of pairs
    
    Args:
        seqs: List of sequences
        labels: Array of activity values
        max_pairs: Maximum number of pairs
        min_pairs_per_sequence: Minimum number of pairs each sequence participates in
        n_strata: Number of strata
        cross_strata_ratio: Ratio of cross-strata pairs
        seed: Random seed
    
    Returns:
        list of (i, j) tuples
    """
    n = len(seqs)
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    
    strata = stratify_by_activity(labels, n_strata)
    print(f"Activity stratification: ", end="")
    for stratum_id, indices in strata.items():
        stratum_labels = labels[indices]
        print(f"Stratum {stratum_id}: {len(indices)} samples (activity range: {stratum_labels.min():.3f} - {stratum_labels.max():.3f})", end=" | ")
    print()
    
    all_pairs = []
    pair_scores = []
    
    for i, j in itertools.combinations(range(n), 2):
        score = calculate_pair_info_score(labels, i, j)
        all_pairs.append((i, j))
        pair_scores.append(score)
    
    pair_scores = np.array(pair_scores)
    all_pairs = np.array(all_pairs)
    
    cross_strata_pairs = []
    within_strata_pairs = []
    
    for idx, (i, j) in enumerate(all_pairs):
        i_stratum = None
        j_stratum = None
        for s_id, s_indices in strata.items():
            if i in s_indices:
                i_stratum = s_id
            if j in s_indices:
                j_stratum = s_id
        
        if i_stratum != j_stratum:
            cross_strata_pairs.append((idx, pair_scores[idx]))
        else:
            within_strata_pairs.append((idx, pair_scores[idx]))
    
    n_cross = min(int(max_pairs * cross_strata_ratio), len(cross_strata_pairs))
    n_within = max_pairs - n_cross
    
    cross_strata_pairs.sort(key=lambda x: x[1], reverse=True)
    within_strata_pairs.sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = set()
    
    for idx, _ in cross_strata_pairs[:n_cross]:
        selected_indices.add(idx)
    
    if len(selected_indices) < max_pairs:
        remaining = max_pairs - len(selected_indices)
        for idx, _ in within_strata_pairs[:min(remaining, len(within_strata_pairs))]:
            selected_indices.add(idx)
    
    selected_pairs = all_pairs[list(selected_indices)]
    
    sequence_pair_count = defaultdict(int)
    for i, j in selected_pairs:
        sequence_pair_count[i] += 1
        sequence_pair_count[j] += 1
    
    under_represented = [seq_idx for seq_idx in range(n) 
                         if sequence_pair_count[seq_idx] < min_pairs_per_sequence]
    
    if under_represented:
        print(f"Warning: Found {len(under_represented)} sequences with insufficient pair participation, supplementing...")
        
        for seq_idx in under_represented:
            needed = min_pairs_per_sequence - sequence_pair_count[seq_idx]
            
            candidate_pairs = []
            for idx, (i, j) in enumerate(all_pairs):
                if idx in selected_indices:
                    continue
                if i == seq_idx or j == seq_idx:
                    candidate_pairs.append((idx, pair_scores[idx], i, j))
            
            candidate_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _, i, j in candidate_pairs[:needed]:
                if len(selected_indices) >= max_pairs:
                    break
                selected_indices.add(idx)
                sequence_pair_count[i] += 1
                sequence_pair_count[j] += 1
    
    sorted_by_activity = np.argsort(labels)
    top_k = 10
    bottom_k = 10
    
    top_indices = set(sorted_by_activity[-top_k:])
    bottom_indices = set(sorted_by_activity[:bottom_k])
    
    for seq_idx in list(top_indices) + list(bottom_indices):
        if sequence_pair_count[seq_idx] < min_pairs_per_sequence * 2:
            needed = min_pairs_per_sequence * 2 - sequence_pair_count[seq_idx]
            
            candidate_pairs = []
            for idx, (i, j) in enumerate(all_pairs):
                if idx in selected_indices:
                    continue
                if i == seq_idx or j == seq_idx:
                    candidate_pairs.append((idx, pair_scores[idx], i, j))
            
            candidate_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _, i, j in candidate_pairs[:needed]:
                if len(selected_indices) >= int(max_pairs * 1.1):
                    break
                selected_indices.add(idx)
                sequence_pair_count[i] += 1
                sequence_pair_count[j] += 1
    
    final_pairs = all_pairs[list(selected_indices)]
    
    return final_pairs.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Smart sampling version of pairwise training pairs generator"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help="Data directory (default: use config.DATA_DIR)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="labeled_data.csv",
        help="Input filename (relative to data_dir), default: labeled_data.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_pairs.csv",
        help="Output filename (relative to data_dir), default: training_pairs.csv",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=50000,
        help="Maximum number of pairs to keep (default: 50000)",
    )
    parser.add_argument(
        "--min_pairs_per_seq",
        type=int,
        default=5,
        help="Minimum number of pairs each sequence participates in (default: 5)",
    )
    parser.add_argument(
        "--n_strata",
        type=int,
        default=3,
        help="Number of activity strata (default: 3, i.e., high/medium/low)",
    )
    parser.add_argument(
        "--cross_strata_ratio",
        type=float,
        default=0.4,
        help="Ratio of cross-strata pairs (default: 0.4, i.e., 40%% of pairs are cross-activity-strata)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: use config.RANDOM_SEED)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    in_path = data_dir / args.input
    out_path = data_dir / args.output

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Reading labeled data: {in_path}")
    df = pd.read_csv(in_path)

    if "label" not in df.columns:
        raise ValueError(f"Input file {in_path} is missing required column 'label'")

    if "sequence" in df.columns:
        seq_col = "sequence"
    elif "child" in df.columns:
        seq_col = "child"
    else:
        raise ValueError(f"Input file {in_path} is missing 'sequence' or 'child' column")

    seqs = df[seq_col].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=float)
    n = len(seqs)

    if n < 2:
        raise ValueError(f"Input sample count is {n}, insufficient to construct any pairs.")

    print(f"Input mutant count: {n}")
    print(f"Activity value range: {labels.min():.3f} - {labels.max():.3f} (mean: {labels.mean():.3f})")
    
    total_possible_pairs = n * (n - 1) // 2
    print(f"Theoretical total pair count: {total_possible_pairs:,}")
    print(f"Target pair count: {args.max_pairs:,} ({args.max_pairs/total_possible_pairs*100:.1f}%)")

    seed = args.seed if args.seed is not None else getattr(config, "RANDOM_SEED", 42)
    selected_pairs = smart_sample_pairs(
        seqs, labels, args.max_pairs,
        min_pairs_per_sequence=args.min_pairs_per_seq,
        n_strata=args.n_strata,
        cross_strata_ratio=args.cross_strata_ratio,
        seed=seed
    )

    print(f"Smart sampling completed: selected {len(selected_pairs):,} pairs from {total_possible_pairs:,}")

    parents = [seqs[i] for i, _ in selected_pairs]
    children = [seqs[j] for _, j in selected_pairs]
    label_vals = [float(labels[j] - labels[i]) for i, j in selected_pairs]

    out_df = pd.DataFrame(
        {
            "parent": parents,
            "child": children,
            "label": label_vals,
        }
    )

    label_deltas = np.array(label_vals)
    print(f"Post-sampling statistics:")
    print(f"   - Pair count: {len(out_df):,}")
    print(f"   - Activity difference range: {label_deltas.min():.3f} - {label_deltas.max():.3f}")
    print(f"   - Activity difference mean: {label_deltas.mean():.3f}")
    print(f"   - Activity difference std: {label_deltas.std():.3f}")

    os.makedirs(data_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Generated smart-sampled training_pairs file: {out_path}")
    print(f"   Label semantics: difference of child relative to parent (label_child - label_parent)")


if __name__ == "__main__":
    main()
