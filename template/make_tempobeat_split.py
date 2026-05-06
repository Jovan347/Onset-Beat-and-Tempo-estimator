#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import random

SOURCE_DIR = Path("../train_extra_tempobeats")
TRAIN_DIR = Path("../tempobeats_train")
VAL_DIR = Path("../tempobeats_val")

VAL_RATIO = 0.20
SEED = 0


def copy_related_files(stem, source_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    for suffix in [".wav", ".beats.gt", ".tempo.gt"]:
        src = source_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, target_dir / src.name)


def main():
    random.seed(SEED)

    wav_files = sorted(SOURCE_DIR.glob("*.wav"))
    stems = [p.stem for p in wav_files]

    usable_stems = []
    for stem in stems:
        if (SOURCE_DIR / f"{stem}.beats.gt").exists():
            usable_stems.append(stem)

    random.shuffle(usable_stems)

    n_val = int(round(len(usable_stems) * VAL_RATIO))

    val_stems = set(usable_stems[:n_val])
    train_stems = set(usable_stems[n_val:])

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    for stem in train_stems:
        copy_related_files(stem, SOURCE_DIR, TRAIN_DIR)

    for stem in val_stems:
        copy_related_files(stem, SOURCE_DIR, VAL_DIR)

    print(f"Source usable files: {len(usable_stems)}")
    print(f"Train files: {len(train_stems)}")
    print(f"Val files: {len(val_stems)}")
    print(f"Created: {TRAIN_DIR}")
    print(f"Created: {VAL_DIR}")


if __name__ == "__main__":
    main()