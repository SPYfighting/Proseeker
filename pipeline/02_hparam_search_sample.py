#!/usr/bin/env python3
"""
Hyperparameter Search (Data Sampling Version, B-GNN)

Extended from 02_hparam_search.py with added support for sample_ratio (default 0.1, using 10% of the data).
The sampling logic operates on unique child sequences to maintain protection against data leakage.
No DDP support - designed for single GPU quick hyperparameter search.
"""

import os
import sys
from pathlib import Path
import json
import torch
import optuna
import time
from torch.utils.data import DataLoader, Subset
from transformers import EsmTokenizer
import yaml
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

import config
from utils.dataset_utils import PairDataset
from utils.model_utils import load_model_for_finetune
from utils.random_utils import set_global_seed


def load_config_yaml(path: str | None) -> dict:
    defaults = {
        "paths": {
            "mlm_model_dir": config.DIR_MLM_TUNED_MODEL,
            "outputs_dir": config.OUTPUTS_DIR,
        },
        "training_pairs": config.PATH_TRAINING_PAIRS,
        "hparam_search": {
            "n_trials": config.HSEARCH_N_TRIALS,
            "epochs_per_trial": config.HSEARCH_EPOCHS_PER_TRIAL,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "val_ratio": 0.2,
            "sample_ratio": 0.1,
        },
        "device": config.DEVICE,
        "random_seed": config.RANDOM_SEED,
    }
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)

            def merge(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                        merge(a[k], v)
                    else:
                        a[k] = v

            merge(defaults, user)
    return defaults


def collate_fn_with_graph(batch):
    """
    Custom collate_fn to ensure edge_index is not batched (all samples share the same graph).
    """
    edge_index = None
    if "edge_index" in batch[0]:
        edge_index = batch[0]["edge_index"]
    
    from torch.utils.data.dataloader import default_collate
    batch_dict = {}
    for key in batch[0].keys():
        if key == "edge_index":
            continue
        batch_dict[key] = default_collate([item[key] for item in batch])
    
    if edge_index is not None:
        batch_dict["edge_index"] = edge_index
    
    return batch_dict


def split_by_unique_child_sequences(
    df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    if "child" not in df.columns:
        raise ValueError("Column 'child' is missing in training_pairs; unable to split dataset by unique child sequences.")

    unique_children = df["child"].unique()
    print(f"Dataset Statistics: Total samples={len(df)}, Unique child sequences={len(unique_children)}")

    np.random.seed(seed)
    shuffled_children = unique_children.copy()
    np.random.shuffle(shuffled_children)

    n_val_children = max(1, int(len(unique_children) * val_ratio))
    val_children = set(shuffled_children[:n_val_children])
    train_children = set(shuffled_children[n_val_children:])

    print(f"Sequence-level Split: Validation child sequences={len(val_children)}, Training child sequences={len(train_children)}")

    train_indices: list[int] = []
    val_indices: list[int] = []

    for idx, row in df.iterrows():
        child_seq = row["child"]
        if child_seq in val_children:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    print(
        f"Split Complete: Training samples={len(train_indices)}, Validation samples={len(val_indices)} | "
        f"{'Pass - No overlap' if len(val_children & train_children)==0 else 'Failed - Overlap exists!'}"
    )
    return train_indices, val_indices


def objective(trial, cfg: dict):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    dropout = trial.suggest_uniform("dropout_rate", 0.0, 0.3)

    mlm_dir = cfg["paths"]["mlm_model_dir"]
    if os.path.exists(mlm_dir):
        tokenizer = EsmTokenizer.from_pretrained(mlm_dir)
    else:
        tokenizer = EsmTokenizer.from_pretrained(config.BASE_ESM_MODEL)

    df = pd.read_csv(cfg["training_pairs"])
    sample_ratio = cfg["hparam_search"].get("sample_ratio", 1.0)
    temp_csv_path = None
    if sample_ratio < 1.0:
        unique_children = df["child"].unique()
        np.random.seed(cfg.get("random_seed", 42))
        n_sample = max(1, int(len(unique_children) * sample_ratio))
        sampled_children = set(np.random.choice(unique_children, size=n_sample, replace=False))
        df = df[df["child"].isin(sampled_children)].copy().reset_index(drop=True)
        print(
            f"Data Sampling: Sampled {len(sampled_children)} unique children from {len(unique_children)} "
            f"({sample_ratio*100:.1f}%), corresponding to {len(df)} samples"
        )

    val_ratio = cfg["hparam_search"].get("val_ratio", 0.2)
    train_indices, val_indices = split_by_unique_child_sequences(
        df, val_ratio=val_ratio, seed=cfg.get("random_seed", 42)
    )

    if sample_ratio < 1.0:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=cfg["paths"]["outputs_dir"]
        )
        temp_csv_path = temp_file.name
        temp_file.close()
        df.to_csv(temp_csv_path, index=False)
        dataset_csv = temp_csv_path
    else:
        dataset_csv = cfg["training_pairs"]

    use_gnn = getattr(config, "USE_GNN", False)
    dataset = PairDataset(dataset_csv, tokenizer, for_training=True, use_graph=use_gnn)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    collate_fn = collate_fn_with_graph if use_gnn else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["hparam_search"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["hparam_search"]["batch_size"],
        collate_fn=collate_fn,
    )

    model = load_model_for_finetune(dropout_rate=dropout, use_gnn=use_gnn).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["device"] == "cuda"))

    grad_accum = cfg["hparam_search"].get("gradient_accumulation_steps", 1)
    epoch_start_time = time.time()
    for epoch_idx in range(cfg["hparam_search"]["epochs_per_trial"]):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            p_in = {k: v.to(cfg["device"]) for k, v in batch["parent_input"].items()}
            c_in = {k: v.to(cfg["device"]) for k, v in batch["child_input"].items()}
            y = batch["label"].float().to(cfg["device"])

            with torch.cuda.amp.autocast(enabled=(cfg["device"] == "cuda")):
                if use_gnn and "edge_index" in batch:
                    edge_index = batch["edge_index"].to(cfg["device"])
                    pred = model(p_in, c_in, edge_index)
                else:
                    pred = model(p_in, c_in)
                loss = torch.nn.functional.mse_loss(pred, y) / grad_accum

            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum
            num_batches += 1
            
            progress_interval = max(50, len(train_loader) // 20)
            if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                avg_time_per_batch = elapsed / ((epoch_idx * len(train_loader)) + batch_idx + 1)
                remaining_batches = (cfg["hparam_search"]["epochs_per_trial"] - epoch_idx - 1) * len(train_loader) + (len(train_loader) - batch_idx - 1)
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                current_loss = epoch_loss / num_batches
                progress_pct = ((epoch_idx * len(train_loader)) + batch_idx + 1) / (cfg["hparam_search"]["epochs_per_trial"] * len(train_loader)) * 100
                
                print(
                    f"  Trial {trial.number} | Epoch {epoch_idx+1}/{cfg['hparam_search']['epochs_per_trial']} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} ({progress_pct:.1f}%) | "
                    f"Loss: {current_loss:.6f} | "
                    f"ETA: {eta_min}m{eta_sec}s",
                    flush=True
                )

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            p_in = {k: v.to(cfg["device"]) for k, v in batch["parent_input"].items()}
            c_in = {k: v.to(cfg["device"]) for k, v in batch["child_input"].items()}
            y = batch["label"].float().to(cfg["device"])
            if use_gnn and "edge_index" in batch:
                edge_index = batch["edge_index"].to(cfg["device"])
                pred = model(p_in, c_in, edge_index)
            else:
                pred = model(p_in, c_in)
            losses.append(torch.nn.functional.mse_loss(pred, y).item())

    if temp_csv_path and os.path.exists(temp_csv_path):
        try:
            os.unlink(temp_csv_path)
        except OSError:
            pass

    val_loss = sum(losses) / len(losses)
    return val_loss


def main(cfg_path: str | None):
    cfg = load_config_yaml(cfg_path)
    set_global_seed(cfg.get("random_seed", 42))
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    
    mlm_dir = cfg["paths"]["mlm_model_dir"]
    use_finetuned_model = os.path.exists(mlm_dir)
    use_gnn = getattr(config, "USE_GNN", False)
    
    print("=" * 80, flush=True)
    print("Hyperparameter Search Configuration", flush=True)
    print("=" * 80, flush=True)
    print(f"Model Type: {'Using fine-tuned model' if use_finetuned_model else 'Using base model (ESM2)'}", flush=True)
    if use_finetuned_model:
        print(f"  Model Path: {mlm_dir}", flush=True)
    else:
        print(f"  Base Model: {config.BASE_ESM_MODEL}", flush=True)
    print(f"Graph File Usage: {'Yes (GNN mode)' if use_gnn else 'No (Pure sequence mode)'}", flush=True)
    print(f"Search Configuration:", flush=True)
    print(f"  - n_trials: {cfg['hparam_search']['n_trials']}", flush=True)
    print(f"  - epochs_per_trial: {cfg['hparam_search']['epochs_per_trial']}", flush=True)
    print(f"  - batch_size: {cfg['hparam_search']['batch_size']}", flush=True)
    print(f"  - sample_ratio: {cfg['hparam_search'].get('sample_ratio', 1.0)}", flush=True)
    print("=" * 80, flush=True)
    
    study = optuna.create_study(direction="minimize")
    search_start_time = time.time()
    study.optimize(lambda t: objective(t, cfg), n_trials=cfg["hparam_search"]["n_trials"])
    search_time = time.time() - search_start_time
    
    best = study.best_params
    out = os.path.join(cfg["paths"]["outputs_dir"], "best_hparams.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80, flush=True)
    print(f"Best hyperparameters saved: {out}", flush=True)
    print(f"Total time: {search_time/3600:.2f} hours ({search_time/60:.1f} minutes)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    a = ap.parse_args()
    main(a.config)
