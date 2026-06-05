#!/usr/bin/env python3
"""
Ensemble Model Training Script (sequence-only DeltaRanker)
"""
import os
import sys
from pathlib import Path
import json
import torch
import time
from torch.utils.data import DataLoader
from transformers import EsmTokenizer
import yaml

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

import config
from utils.dataset_utils import PairDataset
from utils.model_utils import load_model_for_finetune, save_model_bundle
from utils.random_utils import set_global_seed


def load_config_yaml(path: str | None) -> dict:
    defaults = {
        'paths': {
            'mlm_model_dir': config.DIR_MLM_TUNED_MODEL,
            'ensemble_dir': config.DIR_ENSEMBLE_MODELS,
            'outputs_dir': config.OUTPUTS_DIR,
        },
        'training_pairs': config.PATH_TRAINING_PAIRS,
        'training': {
            'ranker': {
                'epochs': config.FINETUNE_FINAL_EPOCHS,
                'batch_size': 4,
                'learning_rate': 2e-4,
                'dropout_rate': 0.1,
                'gradient_accumulation_steps': 1,
            }
        },
        'ensemble': {'n_members': config.FINETUNE_N_ENSEMBLE},
        'device': config.DEVICE,
        'random_seed': config.RANDOM_SEED,
    }
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user = yaml.safe_load(f)
            def merge(a,b):
                for k,v in b.items():
                    if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                        merge(a[k], v)
                    else:
                        a[k] = v
            merge(defaults, user)
    return defaults


def main(cfg_path: str | None):
    cfg = load_config_yaml(cfg_path)
    set_global_seed(cfg.get('random_seed', 42))
    os.makedirs(cfg['paths']['ensemble_dir'], exist_ok=True)

    best_path = os.path.join(cfg['paths']['outputs_dir'], 'best_hparams.json')
    try:
        with open(best_path, 'r', encoding='utf-8') as f:
            best = json.load(f)
        lr = best.get('learning_rate', 2e-4)
        dr = best.get('dropout_rate', 0.1)
    except FileNotFoundError:
        lr = 2e-4
        dr = 0.1

    # tokenizer
    if os.path.exists(cfg['paths']['mlm_model_dir']):
        tokenizer = EsmTokenizer.from_pretrained(cfg['paths']['mlm_model_dir'])
    else:
        tokenizer = EsmTokenizer.from_pretrained(config.BASE_ESM_MODEL)

    ds = PairDataset(cfg['training_pairs'], tokenizer, for_training=True)
    total_samples = len(ds)
    batch_size = cfg['training']['ranker']['batch_size']
    epochs = cfg['training']['ranker']['epochs']
    n_members = cfg['ensemble']['n_members']
    total_batches_per_epoch = (total_samples + batch_size - 1) // batch_size
    
    print("=" * 80, flush=True)
    print(f"Starting Ensemble Model Training (sequence-only DeltaRanker)", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset Statistics: Total samples = {total_samples:,}", flush=True)
    print(f"Training Configuration:", flush=True)
    print(f"  - batch_size: {batch_size}", flush=True)
    print(f"  - epochs: {epochs}", flush=True)
    print(f"  - ensemble_members: {n_members}", flush=True)
    print(f"  - Approx {total_batches_per_epoch:,} batches per epoch", flush=True)
    print(f"  - learning_rate: {lr:.2e}", flush=True)
    print(f"  - dropout_rate: {dr:.3f}", flush=True)
    print("=" * 80, flush=True)

    use_ddp = str(cfg['device']).startswith('cuda') and torch.cuda.device_count() > 1
    if use_ddp:
        print(f"Detected {torch.cuda.device_count()} GPUs", flush=True)

    overall_start_time = time.time()

    for idx in range(n_members):
        member_start_time = time.time()
        member_dir = os.path.join(cfg['paths']['ensemble_dir'], f"member_{idx+1}")
        os.makedirs(member_dir, exist_ok=True)
        torch.manual_seed(cfg['random_seed'] + idx)

        print("\n" + "=" * 80, flush=True)
        print(f"Starting Training Ensemble Member {idx+1}/{n_members}", flush=True)
        print("=" * 80, flush=True)

        model = load_model_for_finetune(dropout_rate=dr).to(cfg['device'])

        if use_ddp:
            model = torch.nn.DataParallel(model)
            print(f"DataParallel enabled, using {torch.cuda.device_count()} GPUs", flush=True)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg['device'] == 'cuda'))

        model.train()
        grad_accum = cfg['training']['ranker'].get('gradient_accumulation_steps', 1)
        
        for ep in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\nEpoch {ep+1}/{epochs} starting training...", flush=True)
            
            opt.zero_grad()
            for batch_idx, batch in enumerate(loader):
                p_in = {k: v.to(cfg['device']) for k, v in batch['parent_input'].items()}
                c_in = {k: v.to(cfg['device']) for k, v in batch['child_input'].items()}
                y = batch['label'].float().to(cfg['device'])
                
                with torch.cuda.amp.autocast(enabled=(cfg['device'] == 'cuda')):
                    pred = model(p_in, c_in)
                    loss = torch.nn.functional.mse_loss(pred, y) / grad_accum
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                
                epoch_loss += loss.item() * grad_accum
                num_batches += 1
                
                progress_interval = max(50, len(loader) // 20)
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == len(loader):
                    elapsed = time.time() - epoch_start_time
                    avg_time_per_batch = elapsed / (batch_idx + 1)
                    remaining_batches = len(loader) - (batch_idx + 1)
                    eta_seconds = avg_time_per_batch * remaining_batches
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    current_loss = epoch_loss / num_batches
                    progress_pct = (batch_idx + 1) / len(loader) * 100
                    
                    print(
                        f"  Member {idx+1} | Epoch {ep+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{len(loader)} ({progress_pct:.1f}%) | "
                        f"Loss: {current_loss:.6f} | "
                        f"ETA: {eta_min}m{eta_sec}s",
                        flush=True
                    )
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(
                f"Member {idx+1} | Epoch {ep+1}/{epochs} completed | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Time: {epoch_time/60:.1f} minutes",
                flush=True
            )

        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        save_model_bundle(model_to_save, tokenizer, member_dir, {
            'member_id': idx + 1,
            'learning_rate': lr,
            'dropout_rate': dr,
        })
        
        member_time = time.time() - member_start_time
        print(f"\nMember {idx+1} saved: {member_dir}", flush=True)
        print(f"Member {idx+1} total time: {member_time/60:.1f} minutes ({member_time/3600:.2f} hours)", flush=True)
        
        if idx < n_members - 1:
            avg_time_per_member = member_time / (idx + 1)
            remaining_members = n_members - (idx + 1)
            estimated_remaining = avg_time_per_member * remaining_members
            print(f"Estimated remaining time: {estimated_remaining/3600:.2f} hours ({estimated_remaining/60:.1f} minutes)", flush=True)
        
        overall_elapsed = time.time() - overall_start_time
        print(f"Cumulative total time: {overall_elapsed/3600:.2f} hours", flush=True)

    overall_time = time.time() - overall_start_time
    print("\n" + "=" * 80, flush=True)
    print(f"All Ensemble members training completed!", flush=True)
    print(f"Total time: {overall_time/3600:.2f} hours ({overall_time/60:.1f} minutes)", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None)
    a = ap.parse_args()
    main(a.config)
