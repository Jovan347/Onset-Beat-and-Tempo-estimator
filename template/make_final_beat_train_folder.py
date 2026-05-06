#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil

SOURCE_DIRS = [
    Path("../train"),
    Path("../tempobeats_train"),
    Path("../tempobeats_val"),
]

OUT_DIR = Path("../beat_train_final")


def copy_related_files(stem, source_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in [".wav", ".beats.gt", ".tempo.gt", ".onsets.gt"]:
        src = source_dir / f"{stem}{suffix}"
        if src.exists():
            dst = out_dir / src.name

            # Avoid accidental overwrite if same stem appears in different folders.
            if dst.exists():
                dst = out_dir / f"{source_dir.name}_{src.name}"

            shutil.copy2(src, dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    for source_dir in SOURCE_DIRS:
        wav_files = sorted(source_dir.glob("*.wav"))

        for wav_file in wav_files:
            stem = wav_file.stem

            if not (source_dir / f"{stem}.beats.gt").exists():
                continue

            copy_related_files(stem, source_dir, OUT_DIR)
            total += 1

    print(f"Copied beat-labeled files: {total}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()