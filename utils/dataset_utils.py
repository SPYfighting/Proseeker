import pandas as pd
import torch
from torch.utils.data import Dataset
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
                 sequence_col=None, child_col=None, parent_col=None, label_col=None):
        """
        Protein sequence pair dataset (sequence-only).

        Args:
            csv_path: CSV file path
            tokenizer: ESM tokenizer
            for_training: Whether for training (determines if labels are returned)
            validate: Whether to validate sequences
            max_len: Maximum sequence length
            sequence_col, child_col, parent_col, label_col: Column names
        """
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        self.for_training = for_training
        self.max_len = max_len or config.MAX_LEN

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
                        raise ValueError(f"Row {idx} column {seq_name}: {msg}")

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

        if self.for_training:
            label = torch.tensor(row["label"], dtype=torch.float)
            result["label"] = label

        return result
