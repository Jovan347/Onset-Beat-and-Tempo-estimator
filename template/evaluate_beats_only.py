#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import mir_eval


def opts_parser():
    parser = ArgumentParser(description="Evaluate beat predictions only.")
    parser.add_argument("gt_dir", type=str, help="Directory with .beats.gt files")
    parser.add_argument("pred_json", type=str, help="Prediction JSON file")
    return parser


def read_beats_file(path):
    with open(path, "r") as f:
        values = [float(line.strip().split()[0]) for line in f if line.strip()]
    return np.array(values, dtype=float)


def main():
    parser = opts_parser()
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)

    with open(args.pred_json, "r") as f:
        preds = json.load(f)

    scores = []

    for gt_file in sorted(gt_dir.glob("*.beats.gt")):
        stem = gt_file.stem.replace(".beats", "")

        gt_beats = read_beats_file(gt_file)
        pred_beats = preds.get(stem, {}).get("beats", [])
        pred_beats = np.array(pred_beats, dtype=float)

        score = mir_eval.beat.f_measure(gt_beats, pred_beats, f_measure_threshold=0.07)
        scores.append(score)

    mean_score = float(np.mean(scores)) if scores else 0.0

    print(f"Files evaluated: {len(scores)}")
    print(f"Beats F-score: {mean_score:.4f}")


if __name__ == "__main__":
    main()