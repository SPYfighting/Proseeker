# Active Learning Pipeline for Protein Engineering

Proseeker is an active learning model for virtual directed evolution of proteins.


## Quick Start
```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline with one command
python pipeline/run_all.py --config configs/default.yaml

# Or run step by step (see OPERATIONS.md)
python pipeline/train_ensemble.py --config configs/default.yaml
python pipeline/predict_with_uncertainty.py --config configs/default.yaml
python pipeline/select_acquisitions.py --pred_csv outputs/predictions_with_uncertainty.csv --k 96 --out outputs/to_measure_round1.csv
python pipeline/iterative_optimize.py --config configs/default.yaml --round 1


```

## Directory Structure
```
standalone/
  configs/          # YAML configuration files
  pipeline/         # Complete pipeline scripts (01-06)
  src/              # Evaluation and visualization modules
  utils/            # Dataset and model utilities
  tests/            # Unit tests
  data/             # Example data
  .github/workflows/ # CI configuration
  requirements.txt  # Dependency lock
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
