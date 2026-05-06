#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creates a frame-level beat dataset for training a small CNN beat detector.

Input:
    Directory containing .wav files and .beats.gt files.

Output:
    beat_dataset.npz containing:
        features: object array of [T, F] feature matrices
        targets: object array of [T] beat target vectors
        stems: file stems
        fps: frame rate
"""

from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from scipy.io import wavfile
import librosa


def opts_parser():
    parser = ArgumentParser()
    parser.add_argument("indir", type=str, help="Directory with .wav and .beats.gt files")
    parser.add_argument("outfile", type=str, help="Output .npz file")
    return parser


def read_wav(filename):
    sample_rate, signal = wavfile.read(filename)

    if signal.dtype.kind == "i":
        signal = signal / np.iinfo(signal.dtype).max

    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    return sample_rate, signal.astype(np.float32)


def read_beats(filename):
    beats = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                beats.append(float(line.split()[0]))

    return np.asarray(beats, dtype=np.float32)


def compute_features(sample_rate, signal, fps=70):
    """
    Computes simple features:
    - 80-bin log-mel spectrogram
    - positive spectral difference / ODF-like feature
    """

    hop_length = sample_rate // fps

    spect = librosa.stft(
        signal,
        n_fft=2048,
        hop_length=hop_length,
        window="hann"
    )

    magspect = np.abs(spect)

    melspect = librosa.feature.melspectrogram(
        S=magspect,
        sr=sample_rate,
        n_mels=80,
        fmin=27.5,
        fmax=8000
    )

    melspect = np.log1p(100 * melspect)

    # Shape now: [mel_bins, time]
    # Compute positive frame differences as an ODF-like feature.
    diff = np.diff(melspect, axis=1)
    diff = np.maximum(diff, 0.0)
    odf = diff.sum(axis=0)

    if odf.size > 0:
        odf = odf - np.median(odf)
        odf = np.maximum(odf, 0.0)
        maxv = odf.max()
        if maxv > 0:
            odf = odf / maxv

    # Because diff has one fewer frame, align mel spectrogram with it.
    melspect = melspect[:, 1:]

    # Normalize mel features per file
    mean = np.mean(melspect)
    std = np.std(melspect) + 1e-8
    melspect = (melspect - mean) / std

    # Feature matrix: [time, features]
    features = melspect.T

    # Add ODF as one extra feature
    odf = odf.reshape(-1, 1)
    features = np.concatenate([features, odf], axis=1)

    return features.astype(np.float32), fps


def make_beat_targets(num_frames, beats, fps):
    """
    Creates soft frame-level beat labels.

    Exact beat frame gets 1.0.
    Neighboring frames get smaller values.
    """

    targets = np.zeros(num_frames, dtype=np.float32)

    for beat_time in beats:
        center = int(round(beat_time * fps))

        for offset, value in [
            (0, 1.0),
            (1, 0.75),
            (2, 0.40),
            (3, 0.15),
        ]:
            for idx in [center - offset, center + offset]:
                if 0 <= idx < num_frames:
                    targets[idx] = max(targets[idx], value)

    return targets


def main():
    parser = opts_parser()
    args = parser.parse_args()

    indir = Path(args.indir)

    wav_files = sorted(indir.glob("*.wav"))

    features_list = []
    targets_list = []
    stems = []

    skipped = 0

    for wav_file in wav_files:
        beat_file = indir / f"{wav_file.stem}.beats.gt"

        if not beat_file.exists():
            skipped += 1
            continue

        sample_rate, signal = read_wav(wav_file)
        beats = read_beats(beat_file)

        features, fps = compute_features(sample_rate, signal, fps=70)
        targets = make_beat_targets(len(features), beats, fps)

        min_len = min(len(features), len(targets))
        features = features[:min_len]
        targets = targets[:min_len]

        features_list.append(features)
        targets_list.append(targets)
        stems.append(wav_file.stem)

        print(f"Loaded {wav_file.stem}: features={features.shape}, beats={len(beats)}")

    np.savez(
        args.outfile,
        features=np.asarray(features_list, dtype=object),
        targets=np.asarray(targets_list, dtype=object),
        stems=np.asarray(stems),
        fps=np.asarray([70], dtype=np.int32),
    )

    print()
    print(f"Saved dataset to {args.outfile}")
    print(f"Files used: {len(features_list)}")
    print(f"Files skipped because no .beats.gt existed: {skipped}")


if __name__ == "__main__":
    main()