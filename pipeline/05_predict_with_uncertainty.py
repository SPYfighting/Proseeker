#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import EsmTokenizer
import time
from torch.utils.data.dataloader import default_collate

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

import config
from utils.dataset_utils import PairDataset
from utils.model_utils import load_model_bundle, load_model_for_finetune
from utils.random_utils import set_global_seed


def collate_fn_with_graph(batch):
    """
    Ensure edge_index is not merged by DataLoader (all samples share the same graph).
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


def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def main(cfg_path: str | None = None):
    set_global_seed(getattr(config, 'RANDOM_SEED', 42))
    outputs_dir = config.OUTPUTS_DIR
    ensemble_dir = config.DIR_ENSEMBLE_MODELS
    print("=" * 80)
    print("Prediction and Uncertainty Estimation (B-GNN)")
    print(f"Device: {config.DEVICE} | USE_GNN: {getattr(config, 'USE_GNN', False)}")
    print(f"MC_DROPOUT: {config.USE_MC_DROPOUT} | passes={config.MC_DROPOUT_PASSES}")
    print(f"Batch size: {config.PREDICTION_BATCH_SIZE} | Candidate file: {config.PATH_CANDIDATE_SEQUENCES}")
    print("=" * 80)

    member_dirs = [
        os.path.join(ensemble_dir, d)
        for d in os.listdir(ensemble_dir)
        if d.startswith('member_') and os.path.isdir(os.path.join(ensemble_dir, d))
    ] if os.path.exists(ensemble_dir) else []
    if not member_dirs:
        raise FileNotFoundError(f"No ensemble members found: {ensemble_dir}")

    tokenizer = EsmTokenizer.from_pretrained(member_dirs[0])

    best_path = config.PATH_BEST_HYPERPARAMS
    try:
        with open(best_path, 'r', encoding='utf-8') as f:
            best = json.load(f)
        dropout = best.get('dropout_rate', 0.1)
    except FileNotFoundError:
        dropout = 0.1
    print(f"Best hyperparameters file: {best_path if os.path.exists(best_path) else 'Not found, using default dropout=0.1'}")
    print(f"Ensemble members: {len(member_dirs)}")

    all_member_predictions = []
    for i, member_dir in enumerate(member_dirs):
        member_start = time.time()
        try:
            model, _, bundle_cfg = load_model_bundle(member_dir, dropout_rate=dropout, device=config.DEVICE)
            use_gnn = bundle_cfg.get("use_gnn", getattr(config, "USE_GNN", False))
        except (FileNotFoundError, OSError, KeyError) as e:
            print(f"Warning [Member {i+1}]: Bundle loading failed, trying traditional method: {e}")
            use_gnn = getattr(config, "USE_GNN", False)
            model = load_model_for_finetune(dropout_rate=dropout, use_gnn=use_gnn).to(config.DEVICE)
            mp = os.path.join(member_dir, 'model.pt')
            if os.path.exists(mp):
                model.load_state_dict(torch.load(mp, map_location=config.DEVICE), strict=False)
            else:
                raise FileNotFoundError(f"model.pt does not exist for member {member_dir}")
        print(f"\nMember {i+1}/{len(member_dirs)} | Path: {member_dir} | use_gnn={use_gnn}")

        tokenizer = EsmTokenizer.from_pretrained(member_dir)
        dataset = PairDataset(
            config.PATH_CANDIDATE_SEQUENCES,
            tokenizer,
            for_training=False,
            use_graph=use_gnn,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.PREDICTION_BATCH_SIZE,
            collate_fn=collate_fn_with_graph if use_gnn else None,
        )
        print(f"Candidate samples: {len(dataset)} | batch={config.PREDICTION_BATCH_SIZE}")
        model.eval()
        if config.USE_MC_DROPOUT:
            enable_mc_dropout(model)
            print(f"MC Dropout enabled, passes={config.MC_DROPOUT_PASSES}")
        member_preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Member {i+1}"):
                p_in = {k: v.to(config.DEVICE) for k, v in batch['parent_input'].items()}
                c_in = {k: v.to(config.DEVICE) for k, v in batch['child_input'].items()}
                mc = []
                passes = config.MC_DROPOUT_PASSES if config.USE_MC_DROPOUT else 1
                for _ in range(passes):
                    if use_gnn and "edge_index" in batch:
                        edge_index = batch["edge_index"].to(config.DEVICE)
                        pred = model(p_in, c_in, edge_index).cpu().numpy()
                    else:
                        pred = model(p_in, c_in).cpu().numpy()
                    mc.append(pred)
                member_preds.append(np.stack(mc, axis=0))
        all_member_predictions.append(np.concatenate(member_preds, axis=1))
        member_time = time.time() - member_start
        print(f"Member {i+1} prediction completed | Time: {member_time/60:.1f} minutes")
        del model
        if config.DEVICE == 'cuda':
            torch.cuda.empty_cache()

    tensor = np.stack(all_member_predictions, axis=0)
    mean_pred = tensor.mean(axis=(0,1))
    aleatoric_var = tensor.var(axis=1).mean(axis=0)
    epistemic_var = tensor.mean(axis=1).var(axis=0)
    total_var = aleatoric_var + epistemic_var
    total_std = np.sqrt(total_var)

    df_candidates = pd.read_csv(config.PATH_CANDIDATE_SEQUENCES)
    df = pd.DataFrame({
        'parent': df_candidates.get('parent', df_candidates.get('sequence', [''] * len(mean_pred))),
        'child': df_candidates.get('child', df_candidates.get('sequence', [''] * len(mean_pred))),
        'mean_score': mean_pred,
        'aleatoric_variance': aleatoric_var,
        'epistemic_variance': epistemic_var,
        'total_variance': total_var,
        'ucb_score': mean_pred + config.ACQ_TEMPERATURE * total_std,
    }).sort_values('ucb_score', ascending=False)

    os.makedirs(outputs_dir, exist_ok=True)
    out = os.path.join(outputs_dir, 'predictions_with_uncertainty.csv')
    df.to_csv(out, index=False)
    print(f"Predictions saved: {out}")
    print(f"Samples: {len(df)} | Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
