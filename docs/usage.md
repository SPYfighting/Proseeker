# Using PROseeker

Run the commands below from the repository root after completing the
[environment setup](../README.md#installation). For a small working example,
see the [TdT demo](demo.md).

## Model

1. **Backbone:** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`).
2. **Fine-tuning:** LoRA on the attention query and value projections, with rank 8,
   alpha 16 and dropout 0.05.
3. **Ranker:** `DeltaRanker` encodes parent and child sequences with the same
   backbone. A dropout and linear layer map the difference between their
   first-token embeddings to the child-minus-parent difference in training labels.
   The TdT examples use log-activity differences.
4. **Uncertainty:** the default ensemble contains five independently seeded
   models, with ten Monte Carlo dropout passes per model. The resulting 50
   predictions per pair provide a mean and empirical predictive dispersion.

The model uses sequence inputs. UCB scores (`mean + beta * std`) assist candidate
ranking. Final experimental candidates are selected manually using the predicted
mean, predictive dispersion, library constraints, experimental feasibility and
the objectives of each round.

## Input data

Place input files under `data/`, or set `DATA_DIR` to another directory.
Set `OUTPUTS_DIR` to a separate output directory for each dataset or run.

| File | Required contents |
|---|---|
| `labeled_data.csv` | `sequence` (or `child`) and `label`: one sequence and its activity value per row |
| `training_pairs.csv` | `parent`, `child`, `label`: label is `activity(child) - activity(parent)` |
| `candidates.csv` | `parent`, `child`: sequence pairs to score |
| `measured_pairs_roundN.csv` | `parent`, `child`, `label`: measured pair differences for round N |
| `homologous_sequences.fasta` | Protein sequences for optional MLM adaptation |

Use the same activity scale throughout training and interpretation. The pairing
script subtracts the supplied labels; it does not normalize or log-transform
them. If activity values are stored in a column named `log_activity`, name that
column `label` before running the pairing script.

Provide amino acid sequences in `parent` and `child`, not mutation notation. The
pair dataset accepts the 20 standard amino acid letters and rejects sequences
longer than `MAX_LEN` in `config.py`. The tokenizer also uses this length limit,
including special tokens; with the default of 512, use at most 510 residues to
retain the full sequence. The demo's 383 residues occupy 385 tokens before
padding.

Create pairwise data from labeled sequences:

```bash
python -m utils.generate_pairwise_training_pairs_smart \
  --input labeled_data.csv --output training_pairs.csv
```

These filenames are relative to `DATA_DIR`. The examples in `data/` show the pair
and candidate formats.

## Pipeline steps

| Step | Script | Output |
|---|---|---|
| Build pairs | `utils/generate_pairwise_training_pairs_smart.py` | `data/training_pairs.csv` |
| Hyperparameter search | `pipeline/hparam_search.py` | `outputs/best_hparams.json` |
| Train ensemble | `pipeline/train_ensemble.py` | `outputs/ensemble/member_*/` |
| Predict | `pipeline/predict_with_uncertainty.py` | `outputs/predictions_with_uncertainty.csv` |
| Select experimental candidates | Manual selection from predictions | Candidates for experimental testing |
| Update and generate candidates | `pipeline/iterative_optimize.py --round N` | `outputs/iter_opt/round_N/new_candidates.csv` |

`config.py` provides the common paths and model settings. Training and
hyperparameter search also read selected fields from `configs/default.yaml`.
Prediction and iterative optimization use `config.py` and its environment
variables; they do not load YAML settings.

```bash
python pipeline/hparam_search.py --config configs/default.yaml
python pipeline/train_ensemble.py --config configs/default.yaml
python pipeline/predict_with_uncertainty.py
```

For iterative optimization, provide manually chosen parent sequences in a CSV
with a `sequence` column and set `--top_k` to the number to use:

```bash
python pipeline/iterative_optimize.py --round 1 \
  --manual_parents_csv data/parents_manual_round1.csv --top_k 1
```

Without a manual parent file, the script selects parents by predicted score.
Candidate generation and final experimental selection are separate steps.
The helpers `tools/convert_predictions_to_mutations.py` and
`tools/add_multi_ucb.py` convert sequences to mutation notation and add UCB scores
at different beta values.

### Optional MLM adaptation

`pipeline/mlm_pretrain.py` adapts the backbone to homologous sequences. This stage
requires `datasets`, which is not listed in `requirements.txt`, and its
`TaskType.MASKED_LM` setting is unavailable in PEFT 0.12.0. The minimal demo starts
from the pretrained ESM-2 backbone and omits this stage.

See the [demo output reference](demo.md#output) for prediction column definitions
and the [verification record](verification.md) for tested configurations.
