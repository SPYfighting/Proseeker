import os
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import get_peft_model, LoraConfig
import config

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed, GNN functionality will be unavailable")
    print("   Install command: pip install torch-geometric")


class DeltaRanker(nn.Module):
    """
    Basic sequence-only Delta Ranker (original version)
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


class EsmGnnRanker(nn.Module):
    """
    ESM-GNN Fusion Model (versionB)
    
    Architecture:
        1. ESM2 extracts sequence embeddings (per-residue)
        2. GNN layers process graph structure information
        3. Global pooling to obtain graph-level representation
        4. Differential comparison + MLP regression
    
    Parameters:
        esm_backbone: ESM2 pretrained model
        num_gnn_layers: Number of GNN layers
        gnn_hidden_dim: GNN hidden layer dimension
        num_heads: Number of GAT attention heads
        dropout_rate: Dropout rate
        use_residual: Whether to use residual connections
    """
    def __init__(self, esm_backbone: EsmModel, num_gnn_layers: int = 3, 
                 gnn_hidden_dim: int = 256, num_heads: int = 4,
                 dropout_rate: float = 0.1, use_residual: bool = True):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("EsmGnnRanker requires torch_geometric, please install it first")
        
        self.esm = esm_backbone
        self.esm_dim = esm_backbone.config.hidden_size
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_residual = use_residual
        
        self.esm_to_gnn_proj = nn.Linear(self.esm_dim, gnn_hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            in_dim = gnn_hidden_dim if i == 0 else gnn_hidden_dim * num_heads
            
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=gnn_hidden_dim,
                    heads=num_heads,
                    dropout=dropout_rate,
                    concat=True,
                )
            )
            self.gnn_norms.append(nn.LayerNorm(gnn_hidden_dim * num_heads))
        
        final_gnn_dim = gnn_hidden_dim * num_heads
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.regressor = nn.Sequential(
            nn.Linear(final_gnn_dim, final_gnn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_gnn_dim // 2, 1)
        )
    
    def encode_sequence_with_graph(self, seq_input, edge_index):
        """
        Encode a single protein sequence using ESM + GNN
        
        Parameters:
            seq_input: Dictionary output from ESM tokenizer {input_ids, attention_mask, ...}
            edge_index: Edge index of shape (2, E)
        
        Returns:
            graph_embedding: Graph-level embedding of shape (batch_size, final_gnn_dim)
        """
        esm_output = self.esm(**seq_input).last_hidden_state
        
        batch_size, seq_len, _ = esm_output.shape
        
        residue_embs = esm_output[:, 1:-1, :]
        
        node_features = self.esm_to_gnn_proj(residue_embs)
        
        graph_embeddings = []
        
        for b in range(batch_size):
            x = node_features[b]
            
            for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
                x_in = x
                x = gnn_layer(x, edge_index)
                x = norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
                
                if self.use_residual and i > 0:
                    x = x + x_in
            
            graph_emb = x.mean(dim=0)
            graph_embeddings.append(graph_emb)
        
        graph_embeddings = torch.stack(graph_embeddings, dim=0)
        
        return graph_embeddings
    
    def forward(self, parent_input, child_input, edge_index):
        p_emb = self.encode_sequence_with_graph(parent_input, edge_index)
        c_emb = self.encode_sequence_with_graph(child_input, edge_index)
        
        delta_emb = c_emb - p_emb
        
        out = self.regressor(delta_emb)
        
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
    cand = [m for m in ['q_proj','v_proj'] if m in names]
    return cand or list(names)


def load_model_for_finetune(dropout_rate: float, use_gnn: bool = None):
    """
    Load model for fine-tuning
    
    Parameters:
        dropout_rate: Dropout rate
        use_gnn: Whether to use GNN model (default: read from config)
    
    Returns:
        model: DeltaRanker or EsmGnnRanker
    """
    if use_gnn is None:
        use_gnn = getattr(config, "USE_GNN", False)
    
    tuned_dir = getattr(config, "DIR_MLM_TUNED_MODEL", "")
    if tuned_dir and os.path.exists(tuned_dir):
        print(f"[ESM] Loading fine-tuned MLM model as backbone: {tuned_dir}")
        esm_backbone = EsmModel.from_pretrained(tuned_dir)
    else:
        print(f"[ESM] Fine-tuned MLM model not found, using base model: {config.BASE_ESM_MODEL}")
        esm_backbone = EsmModel.from_pretrained(config.BASE_ESM_MODEL)
    
    if use_gnn and TORCH_GEOMETRIC_AVAILABLE:
        print("Using ESM-GNN fusion model (versionB)")
        model = EsmGnnRanker(
            esm_backbone, 
            num_gnn_layers=getattr(config, "GNN_NUM_LAYERS", 3),
            gnn_hidden_dim=getattr(config, "GNN_HIDDEN_DIM", 256),
            num_heads=getattr(config, "GNN_NUM_HEADS", 4),
            dropout_rate=dropout_rate
        )
    else:
        if use_gnn:
            print("Warning: GNN unavailable, falling back to sequence-only model")
        print("Using sequence-only DeltaRanker model")
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

    use_gnn = bundle_cfg.get("use_gnn", getattr(config, "USE_GNN", False))
    gnn_num_layers = bundle_cfg.get("gnn_num_layers", getattr(config, "GNN_NUM_LAYERS", 3))
    gnn_hidden_dim = bundle_cfg.get("gnn_hidden_dim", getattr(config, "GNN_HIDDEN_DIM", 256))
    gnn_num_heads = bundle_cfg.get("gnn_num_heads", getattr(config, "GNN_NUM_HEADS", 4))
    dropout_rate = bundle_cfg.get("dropout_rate", dropout_rate)

    if os.path.exists(os.path.join(bundle_dir, "config.json")):
        base = EsmModel.from_pretrained(bundle_dir)
    else:
        base = EsmModel.from_pretrained(config.BASE_ESM_MODEL)

    if use_gnn:
        model = EsmGnnRanker(
            base,
            num_gnn_layers=gnn_num_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            dropout_rate=dropout_rate,
        )
    else:
        model = DeltaRanker(base, dropout_rate=dropout_rate)

    state_path = os.path.join(bundle_dir, "model.pt")
    if os.path.exists(state_path):
        sd = torch.load(state_path, map_location=device or config.DEVICE)
        model.load_state_dict(sd, strict=False)

    if device:
        model.to(device)
    return model, tokenizer, bundle_cfg
