import os
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import get_peft_model, LoraConfig
import config


class DeltaRanker(nn.Module):
    """
    Sequence-only pairwise Delta Ranker.

    Parent and child sequences are encoded by a shared ESM-2 backbone (a
    twin / Siamese encoder). The difference of their [CLS] (first-token)
    embeddings is passed through a dropout + linear head to predict the
    normalized activity gain of the child relative to the parent.
    """
    def __init__(self, esm_backbone: EsmModel, dropout_rate: float = 0.1):
        super().__init__()
        self.esm = esm_backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.esm.config.hidden_size, 1)

    def forward(self, parent_input, child_input):
        p_emb = self.esm(**parent_input).last_hidden_state[:, 0, :]
        c_emb = self.esm(**child_input).last_hidden_state[:, 0, :]
        delta_emb = c_emb - p_emb
        out = self.regressor(self.dropout(delta_emb))
        return out.squeeze(-1)


def detect_lora_target_modules(model: nn.Module) -> list[str]:
    """Scan module names and return target name substrings usable for PEFT (e.g., 'q_proj', 'v_proj')."""
    names = set()
    for name, module in model.named_modules():
        base = name.split('.')[-1]
        if 'q_proj' in base:
            names.add('q_proj')
        if 'k_proj' in base:
            names.add('k_proj')
        if 'v_proj' in base:
            names.add('v_proj')
        if 'out_proj' in base:
            names.add('out_proj')
    if not names and hasattr(config, 'LORA_TARGET_MODULES'):
        return list(config.LORA_TARGET_MODULES)
    cand = [m for m in ['q_proj', 'v_proj'] if m in names]
    return cand or list(names)


def load_model_for_finetune(dropout_rate: float):
    """
    Load the sequence-only DeltaRanker for fine-tuning.

    Parameters:
        dropout_rate: Dropout rate for the ranker head.

    Returns:
        model: DeltaRanker

    If an MLM-finetuned ESM-2 backbone exists (produced by mlm_pretrain.py) it is
    used as the encoder; otherwise the base ESM-2 model is used. LoRA adapters are
    injected into the attention query/value projections when enabled.
    """
    tuned_dir = getattr(config, "DIR_MLM_TUNED_MODEL", "")
    if tuned_dir and os.path.exists(tuned_dir):
        print(f"[ESM] Loading fine-tuned MLM model as backbone: {tuned_dir}")
        esm_backbone = EsmModel.from_pretrained(tuned_dir)
    else:
        print(f"[ESM] Fine-tuned MLM model not found, using base model: {config.BASE_ESM_MODEL}")
        esm_backbone = EsmModel.from_pretrained(config.BASE_ESM_MODEL)

    model = DeltaRanker(esm_backbone, dropout_rate=dropout_rate)

    if getattr(config, "LORA_ENABLED", True):
        targets = detect_lora_target_modules(model.esm)
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=targets,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
        )
        model.esm = get_peft_model(model.esm, lora_config)
        print(f"[LoRA] injected into modules: {targets}")

    return model


def save_model_bundle(model: nn.Module, tokenizer: EsmTokenizer, out_dir: str, extra_config: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception:
        pass
    if hasattr(model.esm, 'merge_and_unload'):
        try:
            model.esm = model.esm.merge_and_unload()
            print("[LoRA] merged into base model for export")
        except Exception:
            pass
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    meta = {
        "base_model": config.BASE_ESM_MODEL,
        "export_format": "merged_full_model",
        "lora_used_in_training": bool(getattr(config, "LORA_ENABLED", True)),
    }
    if extra_config:
        meta.update(extra_config)
    import json
    with open(os.path.join(out_dir, "bundle_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model_bundle(bundle_dir: str, dropout_rate: float = 0.1, device: str | None = None):
    tokenizer = EsmTokenizer.from_pretrained(bundle_dir, use_fast=False)

    bundle_cfg = {}
    cfg_path = os.path.join(bundle_dir, "bundle_config.json")
    if os.path.exists(cfg_path):
        try:
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                bundle_cfg = json.load(f)
        except Exception:
            bundle_cfg = {}

    dropout_rate = bundle_cfg.get("dropout_rate", dropout_rate)

    if os.path.exists(os.path.join(bundle_dir, "config.json")):
        base = EsmModel.from_pretrained(bundle_dir)
    else:
        base = EsmModel.from_pretrained(config.BASE_ESM_MODEL)

    model = DeltaRanker(base, dropout_rate=dropout_rate)

    state_path = os.path.join(bundle_dir, "model.pt")
    if os.path.exists(state_path):
        sd = torch.load(state_path, map_location=device or config.DEVICE)
        model.load_state_dict(sd, strict=False)

    if device:
        model.to(device)
    return model, tokenizer, bundle_cfg
