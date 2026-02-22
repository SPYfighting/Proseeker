import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_pred)[:k]
    ideal = np.argsort(-y_true)[:k]

    def dcg(idx):
        return sum((y_true[i] / np.log2(r + 2)) for r, i in enumerate(idx))

    denom = dcg(ideal)
    return (dcg(order) / (denom + 1e-12)) if denom > 0 else 0.0


def topk_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, ratio: float = 0.1) -> float:
    n = len(y_true)
    top_true = set(np.argsort(-y_true)[: max(1, int(n * ratio))])
    top_pred = set(np.argsort(-y_pred)[:k])
    denom = max(1, len(top_pred))
    return len(top_true & top_pred) / denom


def evaluate_frame(
    df: pd.DataFrame,
    label_col: str = "label",
    pred_col: str = "pred",
    group_col: str | None = None,
    k: int = 10,
) -> pd.DataFrame:
    def _eval_one(g: pd.DataFrame) -> pd.Series:
        y = g[label_col].to_numpy(dtype=float)
        p = g[pred_col].to_numpy(dtype=float)
        metrics = {}
        if len(y) > 1:
            metrics["pearson"] = float(pearsonr(y, p)[0])
            metrics["spearman"] = float(spearmanr(y, p)[0])
            metrics["kendall"] = float(kendalltau(y, p)[0])
        else:
            metrics["pearson"] = np.nan
            metrics["spearman"] = np.nan
            metrics["kendall"] = np.nan
        metrics["mse"] = float(mean_squared_error(y, p))
        metrics["rmse"] = float(mean_squared_error(y, p, squared=False))
        metrics["mae"] = float(mean_absolute_error(y, p))
        metrics[f"top{k}_hit@10pct"] = float(topk_hit_rate(y, p, k=k, ratio=0.10))
        metrics[f"ndcg@{k}"] = float(ndcg_at_k(y, p, k=k))
        return pd.Series(metrics)

    if group_col and group_col in df.columns:
        res = df.groupby(group_col).apply(_eval_one).reset_index()
        macro = res.drop(columns=[group_col]).mean(numeric_only=True).to_dict()
        macro[group_col] = "__macro_avg__"
        res = pd.concat([res, pd.DataFrame([macro])], ignore_index=True)
        return res
    else:
        return _eval_one(df).to_frame().T
