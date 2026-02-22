import pandas as pd
import numpy as np
from pathlib import Path

in_path = Path(r"B-GNN/outputs/predictions_with_uncertainty_mutations.csv")
out_path = Path(r"B-GNN/outputs/predictions_with_uncertainty_mutations_with_multi_ucb.csv")

df = pd.read_csv(in_path)

if "mean_score" not in df.columns or "total_variance" not in df.columns:
    raise SystemExit("Missing mean_score or total_variance column")

total_std = np.sqrt(df["total_variance"])

temps = [0.5, 1, 2, 3, 5, 8,10]

for t in temps:
    col_name = f"ucb_T{str(t).replace('.', '_')}"
    df[col_name] = df["mean_score"] + t * total_std

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print("File written:", out_path)
print("New columns:", [c for c in df.columns if c.startswith("ucb_T")])