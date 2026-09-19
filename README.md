# PROseeker: Active learning for protein engineering

PROseeker ranks protein variants by their predicted activity difference from a
parent sequence. The examples use terminal deoxynucleotidyl transferase (TdT)
variants.

## Overview

PROseeker combines the ESM-2 650M protein language model with LoRA fine-tuning
and a ranker that compares parent and child sequence embeddings. It predicts
the child-minus-parent difference on the scale of the supplied training labels;
the TdT examples use log-activity differences.

An ensemble with Monte Carlo dropout provides a mean prediction and predictive
dispersion. UCB scores assist ranking. Final experimental candidates are selected
manually, considering the predictions, library constraints and experimental
feasibility.

See the [model and workflow guide](docs/usage.md) for details.

## Installation

The reference environment uses Ubuntu 20.04 and Python 3.10.14. With Python 3.10
installed, run:

```bash
git clone https://github.com/xmuzhanglab/Proseeker.git
cd Proseeker
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run PROseeker from the cloned directory. The commands below use a Linux shell.

Dependency installation took less than 5 minutes on the tested server, excluding
the initial ESM-2 weight download. See the
[reference environment](docs/verification.md) for software versions and hardware.

## Minimal demo

The [TdT demo](docs/demo.md#run-the-demo) trains two ensemble members on four
sequence pairs and predicts two candidate pairs. The guide includes complete
commands, settings and expected output.

Training and prediction together took less than 5 minutes on the reference CPU
with cached ESM-2 weights. The first run downloads these weights if needed.

The result is written to:

```text
outputs/demo_run/predictions_with_uncertainty.csv
```

The CSV contains two rows with predicted differences, variance estimates and UCB
scores.

## Documentation

| Guide | Contents |
|---|---|
| [TdT demo](docs/demo.md) | Example data, commands, settings and output |
| [Using your own data](docs/usage.md) | Input formats, model, pipeline and candidate selection |
| [Software verification](docs/verification.md) | Tested environment, installation time and validation results |

## License

[MIT License](LICENSE).
