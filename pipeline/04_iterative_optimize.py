#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import json
import random
import torch
import pandas as pd
import itertools
from torch.utils.data import DataLoader
from transformers import EsmTokenizer
import argparse
import time

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

import config
from utils.dataset_utils import PairDataset
from utils.model_utils import load_model_for_finetune, load_model_bundle
from utils.random_utils import set_global_seed
from torch.utils.data.dataloader import default_collate

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

def collate_fn_with_graph(batch):
    """
    Custom collate_fn to prevent edge_index from being merged by DataLoader (all samples share the same graph).
    Only used in GNN mode; not enabled in pure sequence mode.
    """
    edge_index = batch[0].get("edge_index")
    batch_dict = {}
    for key in batch[0]:
        if key == "edge_index":
            continue
        batch_dict[key] = default_collate([item[key] for item in batch])
    if edge_index is not None:
        batch_dict["edge_index"] = edge_index
    return batch_dict

def get_all_single_mutants(seq: str) -> list[str]:
    """Generate all single-point mutation sequences (Saturation Mutagenesis)"""
    mutants = []
    seq_list = list(seq)
    for i, original_aa in enumerate(seq_list):
        for new_aa in STANDARD_AA:
            if new_aa != original_aa:
                # Create mutation
                new_seq_list = seq_list.copy()
                new_seq_list[i] = new_aa
                mutants.append("".join(new_seq_list))
    return mutants


def load_manual_parents(round_id: int, explicit_path: str | None = None) -> list[str] | None:
    candidate_paths = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    data_dir = getattr(config, "DATA_DIR", "data")
    candidate_paths.append(os.path.join(data_dir, f"parents_manual_round{round_id}.csv"))
    candidate_paths.append(os.path.join(data_dir, "parents_manual.csv"))

    for path in candidate_paths:
        if not path:
            continue
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"WARNING: Unable to read manual parent file {path}, reason: {e}, will ignore this file.")
                continue
            seq_col = None
            for cand in ["sequence", "parent"]:
                if cand in df.columns:
                    seq_col = cand
                    break
            if not seq_col:
                print(f"WARNING: Manual parent file {path} does not contain 'sequence' or 'parent' column, will ignore this file.")
                continue
            parents = [s for s in df[seq_col].astype(str).tolist() if isinstance(s, str) and s.strip()]
            if not parents:
                print(f"WARNING: Manual parent file {path} is empty or contains no valid sequences, will ignore this file.")
                continue
            print(f"Using user-specified parent list from: {path}, total {len(parents)} entries.")
            return parents
    print("INFO: No valid manual parent CSV file detected, will use model prediction results to select parents.")
    return None

def get_random_mutants(seq: str, num_mutants: int = 50, num_edits: int = 1) -> list[str]:
    """Generate random mutation sequences"""
    mutants = set()
    seq_list = list(seq)
    attempts = 0
    while len(mutants) < num_mutants and attempts < num_mutants * 5:
        attempts += 1
        curr_seq = seq_list.copy()
        for _ in range(num_edits):
            pos = random.randint(0, len(curr_seq) - 1)
            original_aa = curr_seq[pos]
            new_aa = random.choice(STANDARD_AA)
            while new_aa == original_aa:
                new_aa = random.choice(STANDARD_AA)
            curr_seq[pos] = new_aa
        mutants.add("".join(curr_seq))
    return list(mutants)

def batch_score(model, tokenizer, parent_seq, candidate_seqs, batch_size=32, use_gnn=False, edge_index=None):
    """Helper function: batch scoring for candidate sequences"""
    model.eval()
    scores = []
    device = next(model.parameters()).device
    edge_index_device = edge_index.to(device) if (use_gnn and edge_index is not None) else None
    
    # Build simple dataset
    data = []
    for c_seq in candidate_seqs:
        data.append({'parent': parent_seq, 'child': c_seq, 'label': 0})  # Label placeholder
    
    # Temporary DataFrame
    df = pd.DataFrame(data)
    
    with torch.no_grad():
        for i in range(0, len(candidate_seqs), batch_size):
            batch_seqs = candidate_seqs[i : i+batch_size]
            
            p_enc = tokenizer([parent_seq] * len(batch_seqs), max_length=config.MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
            c_enc = tokenizer(batch_seqs, max_length=config.MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
            
            p_in = {k: v.to(device) for k, v in p_enc.items()}
            c_in = {k: v.to(device) for k, v in c_enc.items()}
            
            if use_gnn and edge_index_device is not None:
                preds = model(p_in, c_in, edge_index_device).cpu().numpy().flatten()
            else:
                preds = model(p_in, c_in).cpu().numpy().flatten()
            scores.extend(preds)
            
    return scores


def main():
    parser = argparse.ArgumentParser(description="Iterative optimization: fine-tune model and generate new generation of candidate mutants")
    parser.add_argument('--round', type=int, default=1, help='Current iteration round')
    parser.add_argument('--top_k', type=int, default=1, help='Select top K best parents for evolution (default: 1)')
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['saturation', 'greedy_stack', 'random'],
        default='saturation',
        help='Generation strategy: saturation (full single-point scan), greedy_stack (greedy stacking-double mutation), random',
    )
    parser.add_argument('--random_count', type=int, default=100, help='Number of sequences to generate under random strategy')
    parser.add_argument('--stack_top_n', type=int, default=20, help='Under greedy stacking strategy, select top N beneficial single-point mutations for combination')
    parser.add_argument(
        '--manual_parents_csv',
        type=str,
        default=None,
        help='Optional: user-provided parent list CSV path (if not provided, will try to auto-find parents_manual_round{round}.csv or parents_manual.csv in DATA_DIR)',
    )

    args = parser.parse_args()

    # --- 1. Configuration and Initialization ---
    set_global_seed(getattr(config, 'RANDOM_SEED', 42))
    print("=" * 80)
    print("Entering iterative optimization pipeline (B-GNN)")
    print(f"Round: {args.round} | Parent TopK: {args.top_k} | Strategy: {args.strategy}")
    print(f"Device: {config.DEVICE} | USE_GNN: {getattr(config, 'USE_GNN', False)}")
    print(f"ITER_MICRO_STEPS: {getattr(config, 'ITER_MICRO_STEPS', 0)}")
    print("=" * 80)
    out_dir = os.path.join(config.OUTPUTS_DIR, "iter_opt", f"round_{args.round}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "iter_ckpt.pt")

    # Parameter source hints
    argv = " ".join(sys.argv)
    if "--strategy" not in argv:
        print(f"INFO: --strategy not explicitly specified, using default value: {args.strategy}")
    if "--top_k" not in argv:
        print(f"INFO: --top_k not explicitly specified, using default value: {args.top_k}")
    if "--random_count" not in argv:
        print(f"INFO: --random_count not explicitly specified, using default value: {args.random_count}")
    if "--stack_top_n" not in argv:
        print(f"INFO: --stack_top_n not explicitly specified, using default value: {args.stack_top_n}")
    if args.manual_parents_csv is None:
        print("INFO: --manual_parents_csv not explicitly specified, will try to auto-find manual parent file.")

    print(f"Starting round {args.round} iterative optimization")
    print(f"Generation strategy: {args.strategy} | Parent Top-K: {args.top_k}")

    # --- 2. Load Data and Model ---
    measured_base = config.PATH_MEASURED_DATA
    # Simple path inference logic
    measured_path = measured_base.replace('round1', f'round{args.round}')
    if not os.path.exists(measured_path):
        print(f"WARNING: Current round measurement file {measured_path} not found, will fallback to base file {measured_base}.")
        measured_path = measured_base  # Fallback

    print(f"Reading measurement data: {measured_path}")
    measured_df = pd.read_csv(measured_path)
    print(f"Measurement data rows: {len(measured_df)} | columns: {list(measured_df.columns)}")

    if os.path.exists(config.DIR_MLM_TUNED_MODEL):
        print(f"INFO: Loading fine-tuned MLM model as tokenizer source: {config.DIR_MLM_TUNED_MODEL}")
        tokenizer = EsmTokenizer.from_pretrained(config.DIR_MLM_TUNED_MODEL)
    else:
        print(f"INFO: Fine-tuned MLM model not found, will use base model {config.BASE_ESM_MODEL} tokenizer.")
        tokenizer = EsmTokenizer.from_pretrained(config.BASE_ESM_MODEL)

    # Load model (prioritize first ensemble member, fallback to base fine-tuned version)
    try:
        ens_dir = config.DIR_ENSEMBLE_MODELS
        member_dirs = sorted([d for d in os.listdir(ens_dir) if d.startswith('member_')]) if os.path.exists(ens_dir) else []
        if member_dirs:
            print(f"Loading ensemble model member for fine-tuning: {member_dirs[0]}")
            model, _, bundle_cfg = load_model_bundle(os.path.join(ens_dir, member_dirs[0]), dropout_rate=0.1, device=config.DEVICE)
            use_gnn = bundle_cfg.get("use_gnn", getattr(config, "USE_GNN", False))
        else:
            print("No ensemble members detected, will directly load base model for fine-tuning.")
            use_gnn = getattr(config, "USE_GNN", False)
            model = load_model_for_finetune(dropout_rate=0.1, use_gnn=use_gnn).to(config.DEVICE)
    except Exception as e:
        print(f"WARNING: Failed to load ensemble model, will fallback to base model. Reason: {e}")
        use_gnn = getattr(config, "USE_GNN", False)
        model = load_model_for_finetune(dropout_rate=0.1, use_gnn=use_gnn).to(config.DEVICE)
    print(f"Model ready | Using GNN: {use_gnn} | Device: {config.DEVICE}")

    # If using GNN, load graph structure
    edge_index = None
    if use_gnn:
        graph_path = getattr(config, "PATH_WT_GRAPH", os.path.join(config.DATA_DIR, "wt_graph.pt"))
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"GNN mode requires graph structure file, but not found: {graph_path}")
        graph_data = torch.load(graph_path)
        edge_index = graph_data["edge_index"]
        print(f"Loaded graph structure: {graph_path} | edge_index shape: {tuple(edge_index.shape)}")

    # --- 3. Fine-tuning Model ---
    print("Starting model fine-tuning with current round experimental data (lightweight micro-step fine-tuning)...")
    dataset = PairDataset(measured_path, tokenizer, for_training=True, use_graph=use_gnn)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_with_graph if use_gnn else None,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    print(f"Fine-tuning settings | batch_size=16 | lr=2e-5 | max_steps={config.ITER_MICRO_STEPS} | data_size={len(dataset)}")
    
    model.train()
    # Fine-tuning loop
    steps = 0
    max_steps = config.ITER_MICRO_STEPS
    fine_tune_start = time.time()
    while steps < max_steps:
        for batch in loader:
            p_in = {k: v.to(config.DEVICE) for k, v in batch['parent_input'].items()}
            c_in = {k: v.to(config.DEVICE) for k, v in batch['child_input'].items()}
            y = batch['label'].float().to(config.DEVICE)
            if use_gnn and "edge_index" in batch:
                e_idx = batch["edge_index"].to(config.DEVICE)
                pred = model(p_in, c_in, e_idx)
            else:
                pred = model(p_in, c_in)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            steps += 1
            if steps % 20 == 0 or steps == max_steps:
                elapsed = time.time() - fine_tune_start
                eta = (elapsed / steps) * (max_steps - steps) if steps else 0
                print(f"  step {steps}/{max_steps} | loss={loss.item():.4f} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")
            if steps >= max_steps: break
    
    # Save fine-tuned model
    final_model_path = os.path.join(out_dir, 'pytorch_model_iter_final.bin')
    torch.save(model.state_dict(), final_model_path)
    print("Fine-tuning completed")

    # --- 4. Evaluate Existing Data and Select Top-K Parents ---
    print("Evaluating current best sequences for next round candidate design...")
    model.eval()
    
    # Score all measured sequences in current round (find best as parents)
    unique_candidates = list(measured_df['child'].unique())
    # Supplement: also compatible if 'sequence' column exists
    if not unique_candidates and 'sequence' in measured_df.columns:
        unique_candidates = list(measured_df['sequence'].unique())

    if not unique_candidates:
        raise ValueError("Current round measurement data has neither 'child' nor 'sequence' column, cannot perform parent evaluation.")

    # Use default parent from current round for relative scoring
    if 'parent' in measured_df.columns:
        default_parent = measured_df.iloc[0]['parent']
        print("INFO: Will use parent from first row of measurement data as default reference parent for model scoring.")
    else:
        default_parent = config.ITER_PARENT_SEQUENCE
        print("WARNING: 'parent' column not found in measurement data, will use config.ITER_PARENT_SEQUENCE as default reference parent.")

    scores = batch_score(model, tokenizer, default_parent, unique_candidates, use_gnn=use_gnn, edge_index=edge_index)
    scored_candidates = sorted(list(zip(unique_candidates, scores)), key=lambda x: x[1], reverse=True)

    # Save scoring results for reference
    pd.DataFrame(scored_candidates, columns=['sequence', 'score']).to_csv(os.path.join(out_dir, 'current_round_ranking.csv'), index=False)

    # Select Top-K parents
    k = min(args.top_k, len(scored_candidates))


    manual_parents = load_manual_parents(args.round, args.manual_parents_csv)
    if manual_parents is not None:
        # If user provides more parents than top_k, truncate; if less, use all
        if len(manual_parents) > k:
            print(f"INFO: User-provided parent count {len(manual_parents)} exceeds top_k={k}, will only use first {k} entries.")
        selected_parents = manual_parents[:k]
        print(f"This round's parents are fully user-specified: total {len(selected_parents)} entries.")
    else:
        # Fallback to model-based Top-K logic from current round measurement data
        selected_parents = [item[0] for item in scored_candidates[:k]]
        print("INFO: No valid manual parent list provided, this round's parents will be automatically selected by model based on prediction scores from current round measurement data.")
        print(f"Model-selected Top-{k} parents: {[s[:10]+'...' for s in selected_parents]}")

    # --- 5. Generate New Generation Candidates ---
    all_new_candidates = []

    for parent_seq in selected_parents:
        generated_seqs = []
        
        if args.strategy == 'saturation':
            print(f"[Parent {parent_seq[:10]}...] Performing full single-point mutation scan")
            generated_seqs = get_all_single_mutants(parent_seq)
            
        elif args.strategy == 'random':
            print(f"[Parent {parent_seq[:10]}...] Performing random mutation")
            generated_seqs = get_random_mutants(parent_seq, num_mutants=args.random_count)
            
        elif args.strategy == 'greedy_stack':
            print(f"[Parent {parent_seq[:10]}...] Performing greedy stacking (single-point scan -> Top-{args.stack_top_n} -> combination)")

            singles = get_all_single_mutants(parent_seq)
            s_scores = batch_score(model, tokenizer, parent_seq, singles, batch_size=64, use_gnn=use_gnn, edge_index=edge_index)
            # Select Top N beneficial mutations (score > 0 or top ranked)
            zipped = sorted(list(zip(singles, s_scores)), key=lambda x: x[1], reverse=True)
            top_singles = [z[0] for z in zipped[:args.stack_top_n]]
            
            # Parse mutation sites and combine
            def extract_mutation(p, c):
                # Return (index, new_aa)
                diffs = []
                for i, (aa_p, aa_c) in enumerate(zip(p, c)):
                    if aa_p != aa_c:
                        diffs.append((i, aa_c))
                return diffs  # list of tuples
            
            mut_infos = []
            for s in top_singles:
                diffs = extract_mutation(parent_seq, s)
                if len(diffs) == 1:  # Ensure single-point
                    mut_infos.append(diffs[0])
            
            # Combination
            combos = itertools.combinations(mut_infos, 2)
            for (pos1, aa1), (pos2, aa2) in combos:
                if pos1 == pos2: continue  # Same position cannot be stacked
                
                # Build double mutation
                new_l = list(parent_seq)
                new_l[pos1] = aa1
                new_l[pos2] = aa2
                generated_seqs.append("".join(new_l))
                
            print(f"   -> Generated {len(generated_seqs)} double mutation combinations")
            generated_seqs.extend(top_singles)

        for seq in generated_seqs:
            all_new_candidates.append({'parent': parent_seq, 'child': seq})

    # --- 6. Save Final Candidate List ---
    out_csv = os.path.join(out_dir, 'new_candidates.csv')
    df_out = pd.DataFrame(all_new_candidates)
    len_before = len(df_out)
    df_out = df_out.drop_duplicates(subset=['child'])
    print(f"Saving candidate sequences: {out_csv}")
    print(f"Total generated: {len_before} -> After deduplication: {len(df_out)}")
    df_out.to_csv(out_csv, index=False)

if __name__ == '__main__':
    main()