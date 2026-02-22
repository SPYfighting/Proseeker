import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def plot_prediction_scatter(y_true, y_pred, title="Prediction vs Truth", save_path=None, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residuals(y_true, y_pred, save_path=None, figsize=(8, 6)):
    residuals = y_pred - y_true
    plt.figure(figsize=figsize)
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residual (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(y_true, y_pred, n_bins=10, save_path=None, figsize=(8, 6)):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    plt.figure(figsize=figsize)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_topk_curve(y_true, y_pred, k_max=100, save_path=None, figsize=(8, 6)):
    from src.evaluation.proteingym_metrics import topk_hit_rate
    ks = range(1, min(k_max, len(y_true)) + 1)
    hit_rates = [topk_hit_rate(y_true, y_pred, k=k, ratio=0.1) for k in ks]
    plt.figure(figsize=figsize)
    plt.plot(ks, hit_rates, marker='o')
    plt.xlabel('Top-K')
    plt.ylabel('Hit Rate (Top 10% True)')
    plt.title('Top-K Hit Rate Curve')
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_report(y_true, y_pred, sequences=None, output_dir='outputs/reports'):
    from src.evaluation.proteingym_metrics import evaluate_frame
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame({'label': y_true, 'pred': y_pred})
    if sequences is not None:
        df['sequence'] = sequences
    metrics_df = evaluate_frame(df)
    
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Evaluation Report\n\n')
        f.write('## Metrics\n\n')
        f.write(metrics_df.to_markdown(index=False))
        f.write('\n\n')
        f.write('## Visualizations\n\n')
    
    plot_prediction_scatter(y_true, y_pred, save_path=os.path.join(output_dir, 'scatter.png'))
    plot_residuals(y_true, y_pred, save_path=os.path.join(output_dir, 'residuals.png'))
    plot_topk_curve(y_true, y_pred, save_path=os.path.join(output_dir, 'topk_curve.png'))
    
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write('- `scatter.png`: Prediction vs Truth scatter plot\n')
        f.write('- `residuals.png`: Residual distribution\n')
        f.write('- `topk_curve.png`: Top-K hit rate curve\n')
    
    return report_path

