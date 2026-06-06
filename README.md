# PROseeker: Active Learning Pipeline for Protein Engineering

PROseeker is a active-learning framework for the virtual directed
evolution of proteins. 
## Model overview

PROseeker ranks candidate variants by their predicted activity gain relative to
a parent sequence:

1. Backbone: a pre-trained protein language model, ESM-2 650M
   (`facebook/esm2_t33_650M_UR50D`), optionally domain-adapted by masked-language
   modeling (MLM) on TdT-family sequences.
2. Adapter: LoRA fine-tuning (rank 8, alpha 16, dropout 0.05) on the attention
   query/value projections.
3. Ranker (`DeltaRanker`): a twin (Siamese) head. Parent and child sequences are
   encoded by the shared backbone, and the difference of their first-token
   (`[CLS]`) embeddings is passed through a dropout + linear layer to predict the
   normalized activity gain `A_norm(child) - A_norm(parent)`.
4. Uncertainty: an ensemble of 5 independently seeded rankers, each with
   Monte-Carlo dropout (10 stochastic passes), gives a predictive mean and
   variance (epistemic + aleatoric). An upper-confidence-bound (UCB) score,
   `mean + beta * std`, is reported for ranking candidates.

No structural, graph-based, or external annotation features are used.

## Quick Start

```bash
python -m venv .venv && .\.venv\Scripts\activate   # Windows
# source .venv/bin/activate                        # Linux/macOS
pip install -r requirements.txt

# (0) Build pairwise training data from labeled mutants
python -m utils.generate_pairwise_training_pairs_smart \
    --input labeled_data.csv --output training_pairs.csv

# Run the full pipeline with one command
python pipeline/run_all.py --config configs/default.yaml

# Or run step by step (see "Pipeline steps" below)
```

`run_all.py` runs these stages in order: mlm, hparam, ensemble, predict, iter.
Use `--skip` to skip any of them, e.g. `--skip mlm hparam`.

## Pipeline steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0. Build pairs | `utils/generate_pairwise_training_pairs_smart.py` | `data/labeled_data.csv` | `data/training_pairs.csv` |
| 1. MLM fine-tune (optional) | `pipeline/mlm_pretrain.py` | `data/homologous_sequences.fasta` | `outputs/mlm_finetune_lora/` |
| 2. Hyperparameter search | `pipeline/hparam_search.py` | `data/training_pairs.csv` | `outputs/best_hparams.json` |
| 3. Train ensemble | `pipeline/train_ensemble.py` | `data/training_pairs.csv` | `outputs/ensemble/member_*/` |
| 4. Predict + uncertainty | `pipeline/predict_with_uncertainty.py` | `data/candidates.csv` | `outputs/predictions_with_uncertainty.csv` |
| 5. Select candidates | (manual) | predictions CSV | shortlist to test in the lab |
| 6. Iterative optimization | `pipeline/iterative_optimize.py --round N` | `data/measured_pairs_round{N}.csv` | `outputs/iter_opt/round_N/new_candidates.csv` |

Step 5 is manual: inspect `predictions_with_uncertainty.csv` (which contains
`mean_score`, the variance terms, and `ucb_score`) and pick the variants to
assay. The optional helpers `tools/convert_predictions_to_mutations.py` and
`tools/add_multi_ucb.py` can convert sequences to mutation notation and append UCB
columns at several `beta` values to assist this choice.

Example single-step commands:

```bash
python pipeline/mlm_pretrain.py --config configs/default.yaml
python pipeline/hparam_search.py --config configs/default.yaml
python pipeline/train_ensemble.py --config configs/default.yaml
python pipeline/predict_with_uncertainty.py
python pipeline/iterative_optimize.py --config configs/default.yaml --round 1
```

## Input data formats

Place these files under `data/` (paths configurable in `config.py`). See
`data/example_training_pairs.csv` and `data/example_candidates.csv` for the
expected column layout.

- `labeled_data.csv`: columns `sequence` (or `child`) and `label` (activity).
- `training_pairs.csv`: columns `parent`, `child`, `label` (activity difference).
- `candidates.csv`: columns `parent`, `child` (variants to score).
- `measured_pairs_round{N}.csv`: columns `parent`, `child`, `label` for round N.
- `homologous_sequences.fasta`: TdT-family sequences for MLM fine-tuning.

## Directory structure

```
Proseeker/
  configs/          # YAML configuration files (default.yaml)
  pipeline/         # Pipeline scripts (mlm_pretrain, hparam_search,
                    #   train_ensemble, predict_with_uncertainty,
                    #   iterative_optimize, run_all)
  src/              # Evaluation and visualization modules
  utils/            # Dataset, model, data-generation utilities
  tools/            # Post-processing helpers (mutation notation, UCB columns)
  data/             # Example / input data
  config.py         # Default configuration (environment-variable overridable)
  requirements.txt  # Pinned dependencies
  README.md         # This file
```

## Environment requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA optional; set `DEVICE=cpu` to run on CPU)
- Other dependencies: see `requirements.txt`

Configuration can be overridden via environment variables (see `config.py`),
e.g. `DEVICE`, `BASE_ESM_MODEL`, `RANDOM_SEED`.

## License

MIT License
