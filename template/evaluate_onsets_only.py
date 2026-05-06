#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import mir_eval

def opts_parser():
    parser = ArgumentParser(description="Evaluate onset predictions only.")
    parser.add_argument("gt_dir", type=str, help="Directory with .onsets.gt files")
    parser.add_argument("pred_json", type=str, help="Prediction JSON file")
    return parser

def read_onsets_file(path):
    with open(path, "r") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values, dtype=float)

def main():
    parser = opts_parser()
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)

    with open(args.pred_json, "r") as f:
        preds = json.load(f)

    scores = []

    for gt_file in sorted(gt_dir.glob("*.onsets.gt")):
        stem = gt_file.stem.replace(".onsets", "")
        gt_onsets = read_onsets_file(gt_file)

        pred_onsets = preds.get(stem, {}).get("onsets", [])
        pred_onsets = np.array(pred_onsets, dtype=float)

        f_measure = mir_eval.onset.f_measure(gt_onsets, pred_onsets, window=0.05)
        scores.append(f_measure)

    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"Onsets F-score: {mean_score:.4f}")

if __name__ == "__main__":
    main()