import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import config

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(seq: str, allow_gaps: bool = False, max_len: int | None = None) -> tuple[bool, str | None]:
    if not seq or not isinstance(seq, str):
        return False, "Sequence is empty or not a string"
    if max_len and len(seq) > max_len:
        return False, f"Sequence length {len(seq)} exceeds maximum length {max_len}"
    chars = set(seq.upper())
    if not allow_gaps and '-' in chars:
        return False, "Sequence contains gap character '-'"
    invalid = chars - STANDARD_AA - {'-'} if allow_gaps else chars - STANDARD_AA
    if invalid:
        return False, f"Contains invalid characters: {invalid}"
    return True, None


class PairDataset(Dataset):
    def __init__(self, csv_path, tokenizer, for_training=True, validate=True, max_len=None, 
                 sequence_col=None, child_col=None, parent_col=None, label_col=None,
                 graph_path=None, use_graph=False):
        """
        Protein sequence pair dataset (versionB with graph data support)
        
        Args:
            csv_path: CSV file path
            tokenizer: ESM tokenizer
            for_training: Whether for training (determines if labels are returned)
            validate: Whether to validate sequences
            max_len: Maximum sequence length
            sequence_col, child_col, parent_col, label_col: Column names
            graph_path: Pre-computed graph structure file path (.pt format)
            use_graph: Whether to use graph structure (versionB GNN mode)
        """
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        self.for_training = for_training
        self.max_len = max_len or config.MAX_LEN
        self.use_graph = use_graph
        self.edge_index = None
        
        if self.use_graph:
            if graph_path is None:
                graph_path = getattr(config, 'PATH_WT_GRAPH', os.path.join(config.DATA_DIR, 'wt_graph.pt'))
            
            if os.path.exists(graph_path):
                graph_data = torch.load(graph_path)
                self.edge_index = graph_data['edge_index']
                print(f"Loaded graph structure: {graph_path}")
                print(f"  Nodes: {graph_data['num_nodes']}, Edges: {self.edge_index.shape[1]}")
            else:
                print(f"Warning: Graph file does not exist: {graph_path}")
                print(f"  Please run first: python utils/process_pdb_to_graph.py")
                print(f"  GNN mode will be disabled, falling back to pure sequence mode")
                self.use_graph = False

        seq_col = sequence_col or 'sequence'
        child_col_name = child_col or 'child'
        parent_col_name = parent_col or 'parent'
        label_col_name = label_col or 'label'

        if parent_col_name not in self.df.columns:
            self.df[parent_col_name] = config.ITER_PARENT_SEQUENCE
        if child_col_name not in self.df.columns and seq_col in self.df.columns:
            self.df[child_col_name] = self.df[seq_col]
        if child_col_name not in self.df.columns:
            raise ValueError(f"Missing {child_col_name} or {seq_col} column")

        if validate:
            for idx, row in self.df.iterrows():
                for seq_name in [parent_col_name, child_col_name]:
                    seq = row[seq_name]
                    ok, msg = validate_sequence(seq, max_len=self.max_len)
                    if not ok:
                        raise ValueError(f"行 {idx} 列 {seq_name}: {msg}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parent_seq = row["parent"]
        child_seq = row["child"]

        p_enc = self.tokenizer(
            parent_seq,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        c_enc = self.tokenizer(
            child_seq,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        p_enc = {k: v.squeeze(0) for k, v in p_enc.items()}
        c_enc = {k: v.squeeze(0) for k, v in c_enc.items()}

        result = {
            "parent_input": p_enc,
            "child_input": c_enc,
        }
        
        if self.use_graph and self.edge_index is not None:
            result["edge_index"] = self.edge_index
        
        if self.for_training:
            label = torch.tensor(row["label"], dtype=torch.float)
            result["label"] = label
        
        return result
