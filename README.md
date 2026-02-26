# ESM2+LoRA Active Learning Pipeline for Protein Engineering

This is a standalone, fully functional version that does not depend on external modules. It includes: configuration, dependencies, source code modules, complete pipeline scripts, evaluation and testing.

## Core Features

- **Automatic LoRA Detection**: Automatically identifies ESM2 modules and injects LoRA with confirmation messages
- **Unified Export Format**: Automatically merges weights after training to avoid mixed states
- **Active Learning Loop**: Prediction → Sampling (TopK + diversity) → Iterative optimization
- **Complete Evaluation System**: ProteinGym metrics + visualization reports (scatter plots, residuals, TopK curves)
- **Data Validation**: Sequence validity checking, column name mapping, length control
- **Streamlit UI**: Batch scoring, single mutation scanning
- **Gradient Accumulation**: Supports training on low-memory GPUs (configurable)
- **Reproducibility**: Unified random seed control ensures consistent results
- **Robustness**: Supports running from any directory with comprehensive error handling
- **CI Support**: GitHub Actions automated testing

## Quick Start
```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline with one command
python pipeline/run_all.py --config configs/default.yaml

# Or run step by step (see OPERATIONS.md)
python pipeline/01_mlm_pretrain.py --config configs/default.yaml
python pipeline/02_hparam_search.py --config configs/default.yaml
python pipeline/03_train_ensemble.py --config configs/default.yaml
python pipeline/04_predict_with_uncertainty.py --config configs/default.yaml
python pipeline/06_select_acquisitions.py --pred_csv outputs/predictions_with_uncertainty.csv --k 96 --out outputs/to_measure_round1.csv
python pipeline/05_iterative_optimize.py --config configs/default.yaml --round 1

# Evaluation + visualization report
python pipeline/benchmark_proteingym.py --csv outputs/predictions_with_uncertainty.csv --label_col label --pred_col mean_score --report_dir outputs/reports

# UI (optional)
streamlit run ui/app.py -- --model_dir outputs/ensemble/member_1
```

## Directory Structure
```
standalone/
  configs/          # YAML configuration files
  pipeline/         # Complete pipeline scripts (01-06)
  src/              # Evaluation and visualization modules
  utils/            # Dataset and model utilities
  ui/               # Streamlit UI
  tests/            # Unit tests
  data/             # Example data
  .github/workflows/ # CI configuration
  requirements.txt  # Dependency lock
  Dockerfile        # Docker image
  config.py         # Default configuration (environment variable override)
  README.md         # This file
  OPERATIONS.md     # Detailed operation guide
```


## Environment Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA optional)
- Other dependencies see `requirements.txt`

## License

MIT License
