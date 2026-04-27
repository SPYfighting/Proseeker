#!/usr/bin/env python3
import subprocess
import argparse
import sys


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run complete pipeline (standalone)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--skip", nargs="+", choices=["mlm", "hparam", "ensemble", "predict", "select", "iter"], default=[])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--select_k", type=int, default=96)
    args = ap.parse_args()

    if "hparam" not in args.skip:
        run([sys.executable, "pipeline/hparam_search.py", "--config", args.config])
    if "ensemble" not in args.skip:
        run([sys.executable, "pipeline/train_ensemble.py", "--config", args.config])
    if "predict" not in args.skip:
        run([sys.executable, "pipeline/predict_with_uncertainty.py", "--config", args.config])
    if "select" not in args.skip:
        run([sys.executable, "pipeline/select_acquisitions.py", "--pred_csv", "outputs/predictions_with_uncertainty.csv", "--k", str(args.select_k), "--out", "outputs/to_measure_round{}.csv".format(args.round)])
    if "iter" not in args.skip:
        run([sys.executable, "pipeline/iterative_optimize.py", "--config", args.config, "--round", str(args.round)])


if __name__ == "__main__":
    main()
