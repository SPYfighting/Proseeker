# Software verification

## Reference environment

| Component | Tested configuration |
|---|---|
| Operating system | Ubuntu 20.04 LTS, x86_64 |
| Python | 3.10.14 |
| CPU | Intel Xeon Silver 4316 @ 2.30 GHz |
| System memory | 125 GiB |
| PyTorch | 2.3.1+cu121 |
| Transformers | 4.43.3 |
| PEFT | 0.12.0 |
| Accelerate | 0.33.0 |
| fair-esm | 2.0.0 |
| Other dependencies | Versions specified in [requirements.txt](../requirements.txt) |

The demo used four CPU threads. The server also contained two NVIDIA GeForce RTX
3090 GPUs with driver 550.54.14; GPU runtime was not benchmarked. All 21 pinned
dependencies matched the installed versions, and `python -m pip check` completed
without dependency conflicts.

## Environment setup

The timed installation used:

```bash
python -m pip install -r requirements.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Dependency installation in a fresh Python environment took less than 5 minutes
using the Tsinghua TUNA PyPI mirror. The measurement includes package download
and installation and excludes repository cloning, environment creation and ESM-2
weight download. The default pip cache was available.

## Demo

The [demo commands](demo.md#run-the-demo) train two models for one epoch on
four TdT sequence pairs, then predict two candidate pairs with two MC passes per
model. Training and prediction together took less than 5 minutes on the reference
CPU with cached ESM-2 weights.

The cached model was `facebook/esm2_t33_650M_UR50D`, revision
`08e4846e537177426273712802403f7ba8261b6c`. The demo settings and data formats are
listed in the [demo guide](demo.md).

| Check | Result |
|---|---|
| Training process | Completed successfully |
| Model export | Two member directories, each with model weights and tokenizer files |
| Model reload and prediction | Completed successfully |
| Output CSV | Two rows and seven columns |
| Candidate correspondence | Output pairs match the two input pairs |
| Numerical values | All scores and variances are finite |
| Variance | Nonnegative; total equals within-member plus between-member variance |
| UCB | Equals mean plus 0.5 times the square root of total variance |

Expected output: `outputs/demo_run/predictions_with_uncertainty.csv`.

## Scope

The verified workflow covers dependency setup, short LoRA ensemble training,
model saving, model reloading and uncertainty prediction. GPU execution,
hyperparameter search and iterative optimization were not benchmarked. The
optional MLM stage requires the dependency and PEFT interface changes described
in the [usage guide](usage.md#optional-mlm-adaptation).

Source code and examples are distributed under the [MIT License](../LICENSE).
