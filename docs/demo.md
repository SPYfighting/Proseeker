# TdT demo

The examples contain four training pairs and two candidate pairs from the TdT
data used with PROseeker. All parent and child sequences are 383 amino acids long.
The sequence strings and training labels are retained from the source pair
tables.

| File | Rows | Columns |
|---|---:|---|
| `data/example_training_pairs.csv` | 4 | `parent`, `child`, `label` |
| `data/example_candidates.csv` | 2 | `parent`, `child` |

Each training label is the child's log-activity value minus the parent's
log-activity value. Candidate pairs have no activity labels. Predicted differences
therefore use the training labels' log-activity scale.

The small demo exercises training, model saving and prediction; it does not
estimate predictive accuracy.

## Run the demo

Complete the [environment setup](../README.md#installation), then run this block
from the repository root in the activated environment. Use a fresh output
directory for each run. The parentheses keep the demo settings within this shell
block.

```bash
(
  set -e
  export DATA_DIR=data/demo_run
  export OUTPUTS_DIR=outputs/demo_run
  export DIR_ENSEMBLE_MODELS="$OUTPUTS_DIR/ensemble"
  export DIR_MLM_TUNED_MODEL="$OUTPUTS_DIR/mlm_not_used"
  export BASE_ESM_MODEL=facebook/esm2_t33_650M_UR50D

  mkdir -p "$DATA_DIR"
  cp data/example_training_pairs.csv "$DATA_DIR/training_pairs.csv"
  cp data/example_candidates.csv "$DATA_DIR/candidates.csv"

  export DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export RANDOM_SEED=42
  export FINETUNE_N_ENSEMBLE=2
  export FINETUNE_FINAL_EPOCHS=1
  export PREDICTION_BATCH_SIZE=1
  export USE_MC_DROPOUT=1
  export MC_DROPOUT_PASSES=2
  export ACQ_TEMPERATURE=0.5

  python pipeline/train_ensemble.py
  python pipeline/predict_with_uncertainty.py
)
```

Training and prediction together took less than 5 minutes on the
[tested CPU](verification.md#reference-environment), with the ESM-2 weights
already cached. The first run downloads the weights if they are not available
locally.

## Configuration

The demo uses these settings:

| Setting | Value |
|---|---|
| Backbone | ESM-2 650M |
| LoRA | Rank 8, alpha 16, dropout 0.05, query/value projections |
| Ensemble members | 2 |
| Epochs per member | 1 |
| Training batch size | 4 |
| Learning rate | 0.0002 |
| Ranker dropout | 0.1 |
| Prediction batch size | 1 |
| MC passes per member | 2 |
| Random seed | 42, incremented for each training member |
| UCB coefficient | 0.5 |
| Device | CPU, four threads |
| Token length | 512, including padding |

The full default configuration uses five ensemble members, ten training epochs
and ten MC passes per member. MLM adaptation, hyperparameter search and iterative
optimization are separate stages.

## Output

A successful run creates:

```text
outputs/demo_run/ensemble/member_1/
outputs/demo_run/ensemble/member_2/
outputs/demo_run/predictions_with_uncertainty.csv
```

The prediction file contains two rows, sorted by `ucb_score`:

| Column | Meaning |
|---|---|
| `parent`, `child` | Input amino acid sequences |
| `mean_score` | Mean predicted child-minus-parent label difference |
| `aleatoric_variance` | Variance across MC passes within each member, averaged over members |
| `epistemic_variance` | Variance of the member mean predictions |
| `total_variance` | Sum of the two variance terms |
| `ucb_score` | `mean_score + beta * sqrt(total_variance)`; beta is 0.5 in the demo |

The variance column names refer to these computations. MC dropout dispersion
does not separately estimate experimental measurement noise.

Prediction reloads the two saved models. The mean is taken across both members
and their MC passes. See the [verification results](verification.md#demo).
