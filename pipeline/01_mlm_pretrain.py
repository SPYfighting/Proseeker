#!/usr/bin/env python3
"""
MLM Domain-Specific Fine-tuning Script 
Fine-tune ESM-2 using LoRA for Masked Language Modeling (MLM) and save the merged model.
"""
import os
import sys
from pathlib import Path
import torch
from transformers import EsmTokenizer, EsmForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import yaml

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

import config
from utils.model_utils import detect_lora_target_modules
from utils.random_utils import set_global_seed


def parse_fasta(file_path: str) -> list[str]:
    sequences = []
    current_sequence = ""
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line
    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def load_config_yaml(config_path: str = None) -> dict:
    defaults = {
        'model': {
            'esm_model_name': config.BASE_ESM_MODEL,
            'max_len': config.MAX_LEN,
        },
        'training': {
            'mlm': {
                'epochs': 10,
                'batch_size': 8,
                'learning_rate': 5e-4,
                'freeze_layers': 10,
            },
        },
        'paths': {
            'mlm_model_dir': config.DIR_MLM_TUNED_MODEL,
            'data_dir': config.DATA_DIR,
        },
        'random_seed': config.RANDOM_SEED,
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user = yaml.safe_load(f)
            def merge(a,b):
                for k,v in b.items():
                    if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                        merge(a[k], v)
                    else:
                        a[k] = v
            merge(defaults, user)
    return defaults


def main(cfg_path: str = None):
    cfg = load_config_yaml(cfg_path)
    model_cfg = cfg['model']
    train_cfg = cfg['training']['mlm']
    paths = cfg['paths']

    set_global_seed(cfg.get('random_seed', 42))

    os.makedirs(paths['mlm_model_dir'], exist_ok=True)
    fasta_path = os.path.join(paths['data_dir'], 'homologous_sequences.fasta')
    sequences = parse_fasta(fasta_path)
    if not sequences:
        raise ValueError("FASTA file is empty or does not exist.")

    dataset = Dataset.from_dict({'sequence': sequences})
    tokenizer = EsmTokenizer.from_pretrained(model_cfg['esm_model_name'])
    model = EsmForMaskedLM.from_pretrained(model_cfg['esm_model_name'])

    targets = detect_lora_target_modules(model)
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=targets,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.MASKED_LM,
    )
    model = get_peft_model(model, lora_config)

    if train_cfg.get('freeze_layers', 0) > 0:
        n = train_cfg['freeze_layers']
        for p in model.base_model.model.esm.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(model.base_model.model.esm.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def tokenize_fn(examples):
        return tokenizer(
            examples['sequence'],
            truncation=True,
            padding='max_length',
            max_length=model_cfg['max_len']
        )
    tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=os.path.join(paths['mlm_model_dir'], 'checkpoints'),
        overwrite_output_dir=True,
        num_train_epochs=train_cfg['epochs'],
        per_device_train_batch_size=train_cfg['batch_size'],
        learning_rate=train_cfg['learning_rate'],
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=50,
        seed=cfg['random_seed'],
        fp16=torch.cuda.is_available(),
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=collator,
        tokenizer=tokenizer
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(paths['mlm_model_dir'])
    tokenizer.save_pretrained(paths['mlm_model_dir'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None)
    a = ap.parse_args()
    main(a.config)
