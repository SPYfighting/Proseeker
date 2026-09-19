# Software verification

## Reference environment

| Component | Tested configuration |
|---|---|
| Operating system | Ubuntu 20.04 LTS, x86_64 |
| Python | 3.10.14 |
| CPU | Intel Xeon Silver 4316 |
| System memory | 125 GiB |
| PyTorch | 2.3.1+cu121 |
| Transformers | 4.43.3 |
| PEFT | 0.12.0 |
| Accelerate | 0.33.0 |
| fair-esm | 2.0.0 |
| Other dependencies | Versions specified in [requirements.txt](../requirements.txt) |

The demo used four CPU threads. Installed package versions matched
[requirements.txt](../requirements.txt).

## Environment setup

Dependency installation took less than 5 minutes in a fresh Python environment
on the reference server, excluding the ESM-2 weight download.

## Demo

The [TdT demo](demo.md#run-the-demo) completed training and prediction in less
than 5 minutes on the reference CPU with cached ESM-2 weights. Settings and
expected output are described in the [demo guide](demo.md).

| Check | Result |
|---|---|
| Training and prediction | Two models trained, saved and reloaded successfully |
| Output CSV | Two rows and seven columns; sequence pairs match the input |
| Numerical values | Finite scores and nonnegative variances |
| Calculations | Total variance and UCB match the formulas in the demo guide |

## Scope

Verification covered dependency setup and the CPU demo. GPU execution,
hyperparameter search and iterative optimization were not benchmarked. See the
[usage guide](usage.md#optional-mlm-adaptation) for optional MLM requirements.
