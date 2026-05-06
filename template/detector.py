#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from scipy.io import wavfile
import librosa

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

try:
    import tqdm
except ImportError:
    tqdm = None

BEAT_MODEL = None
BEAT_MODEL_LOADED = False


class BeatCNN(nn.Module):
    def __init__(self, num_features=81):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits[:, 0, :]


def load_beat_model():
    """
    Loads the trained CNN model once.
    If beat_cnn_model.pt is missing, returns None and the detector falls back
    to the lecture-based baseline.
    """
    global BEAT_MODEL, BEAT_MODEL_LOADED

    if BEAT_MODEL_LOADED:
        return BEAT_MODEL

    BEAT_MODEL_LOADED = True

    if torch is None:
        print("PyTorch not available; using baseline beat tracker.")
        BEAT_MODEL = None
        return None

    model_path = Path("beat_cnn_model.pt")

    if not model_path.exists():
        print("beat_cnn_model.pt not found; using baseline beat tracker.")
        BEAT_MODEL = None
        return None

    device = torch.device("cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model = BeatCNN(num_features=checkpoint.get("num_features", 81))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    BEAT_MODEL = model
    print("Loaded beat_cnn_model.pt")

    return BEAT_MODEL


def predict_beat_activation(melspect, odf):
    """
    Uses the trained CNN to predict frame-level beat probabilities.
    Returns a 1D array with one value per frame.
    """

    model = load_beat_model()

    if model is None:
        return None

    odf = np.asarray(odf, dtype=np.float32)

    # Training used melspect[:, 1:] because the ODF comes from np.diff().
    mel = melspect[:, 1:]

    min_len = min(mel.shape[1], len(odf))
    mel = mel[:, :min_len]
    odf = odf[:min_len]

    # Normalize mel features per file, same as training.
    mean = np.mean(mel)
    std = np.std(mel) + 1e-8
    mel = (mel - mean) / std

    # Normalize ODF
    odf_norm = odf - np.median(odf)
    odf_norm = np.maximum(odf_norm, 0.0)

    maxv = odf_norm.max()
    if maxv > 0:
        odf_norm = odf_norm / maxv

    features = mel.T.astype(np.float32)
    odf_norm = odf_norm.reshape(-1, 1).astype(np.float32)

    features = np.concatenate([features, odf_norm], axis=1)

    # Shape for Conv1D: [batch, features, time]
    x = torch.tensor(features.T[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    return probs.astype(np.float32)

def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    # we only have a dumb dummy implementation here.
    # it returns every 1000th absolute sample value of the input signal.
    # this is not a useful solution at all, just a placeholder.
    # frame-to-frame difference along time axis
    diff = np.diff(melspect, axis=1)

    # keep only positive changes
    diff = np.maximum(diff, 0.0)

    # sum across frequency bins -> one value per frame
    values = diff.sum(axis=0)

    # optional light normalization for stability
    if values.size > 0:
        values = values - np.median(values)
        values = np.maximum(values, 0.0)
        maxv = values.max()
        if maxv > 0:
            values = values / maxv

    # because diff reduces the number of frames by 1
    values_per_second = fps
    return values, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns the timestamps of the 100 strongest values.
    # this is not a useful solution at all, just a placeholder.
    sodf = np.asarray(odf, dtype=float)

    if odf.size == 0:
        return np.array([], dtype=float)

    # normalize to a stable range
    odf = odf - np.median(odf)
    odf = np.maximum(odf, 0.0)
    maxv = odf.max()
    if maxv > 0:
        odf = odf / maxv

    # smooth slightly to reduce jitter
    smooth_len = max(1, int(0.05 * odf_rate))   # ~30 ms
    smooth_kernel = np.ones(smooth_len) / smooth_len
    odf_smooth = np.convolve(odf, smooth_kernel, mode='same')

    # local adaptive threshold
    thresh_len = max(1, int(0.20 * odf_rate))   # ~100 ms
    thresh_kernel = np.ones(thresh_len) / thresh_len
    local_mean = np.convolve(odf_smooth, thresh_kernel, mode='same')

    # threshold offset; likely needs later tuning
    threshold = local_mean + 0.03

    # minimum distance between detections
    min_distance = max(1, int(0.06 * odf_rate))   # ~50 ms

    strongest_indices = []
    last_idx = -min_distance

    for i in range(1, len(odf_smooth) - 1):
        is_peak = (odf_smooth[i] > odf_smooth[i - 1] and
                   odf_smooth[i] >= odf_smooth[i + 1])
        above_threshold = odf_smooth[i] > threshold[i]

        if is_peak and above_threshold:
            if i - last_idx >= min_distance:
                strongest_indices.append(i)
                last_idx = i
            elif odf_smooth[i] > odf_smooth[strongest_indices[-1]]:
                # if two peaks are too close, keep the stronger one
                strongest_indices[-1] = i
                last_idx = i

    strongest_indices = np.array(strongest_indices, dtype=float)
    return strongest_indices / odf_rate


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Estimate tempo from the onset detection function using autocorrelation.

    Lecture idea:
    - The onset detection function is treated as a noisy pulse train.
    - Autocorrelation finds repeated periodicities.
    - We search only lags corresponding to reasonable tempi, e.g. 60–200 bpm.
    """

    odf = np.asarray(odf, dtype=float)

    if odf.size < 4:
        return np.array([120.0])

    # Normalize ODF
    odf = odf - np.mean(odf)
    odf = np.maximum(odf, 0.0)

    maxv = odf.max()
    if maxv > 0:
        odf = odf / maxv
    else:
        return np.array([120.0])

    # Tempo search range from lecture heuristic
    min_bpm = 60.0
    max_bpm = 200.0

    # Convert BPM range to lag range in frames
    # bpm = 60 * odf_rate / lag
    min_lag = int(round(60.0 * odf_rate / max_bpm))
    max_lag = int(round(60.0 * odf_rate / min_bpm))

    min_lag = max(1, min_lag)
    max_lag = min(max_lag, len(odf) - 1)

    if max_lag <= min_lag:
        return np.array([120.0])

    # Autocorrelation only for allowed lags
    scores = []
    lags = np.arange(min_lag, max_lag + 1)

    for lag in lags:
        score = np.sum(odf[:-lag] * odf[lag:])
        scores.append(score)

    scores = np.asarray(scores)

    if scores.size == 0 or scores.max() <= 0:
        return np.array([120.0])

    # Best lag = strongest periodicity
    best_lag = lags[np.argmax(scores)]
    tempo = 60.0 * odf_rate / best_lag

    # Also return the octave-related alternative if reasonable.
    # The challenge allows one or two tempo guesses.
    candidates = [tempo]

    if tempo * 2 <= 240:
        candidates.append(tempo * 2)
    elif tempo / 2 >= 30:
        candidates.append(tempo / 2)

    # Sort lower first, as required when giving two tempi
    candidates = sorted(candidates)

    return np.array(candidates[:2], dtype=float)


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Beat tracking using:
    1. estimated tempo / beat period,
    2. phase search over possible first-beat offsets,
    3. beat grid,
    4. snapping each beat to a nearby ODF peak.

    This follows the lecture idea:
    periodicity estimation + phase/beat location.
    """

    odf = np.asarray(odf, dtype=float)
    tempo = np.asarray(tempo, dtype=float)

    if odf.size == 0:
        return np.array([], dtype=float)

    # Normalize ODF for scoring
    baseline_activation = odf - np.median(odf)
    baseline_activation = np.maximum(baseline_activation, 0.0)

    maxv = baseline_activation.max()
    if maxv > 0:
        baseline_activation = baseline_activation / maxv

    # Try AI beat activation model.
    beat_activation = predict_beat_activation(melspect, odf)

    if beat_activation is not None:
        min_len = min(len(baseline_activation), len(beat_activation))

        baseline_activation = baseline_activation[:min_len]
        beat_activation = beat_activation[:min_len]

        # First test: mostly AI, with a little ODF stability.
        odf_norm = beat_activation
    else:
        odf_norm = baseline_activation

    duration = len(odf_norm) / odf_rate
    # Use the first/main tempo candidate.
    # Since detect_tempo returns sorted candidates, choose the one in 60–200 if possible.
    tempo_candidates = tempo[(tempo >= 60.0) & (tempo <= 200.0)]

    if tempo_candidates.size == 0:
        if tempo.size > 0 and tempo[0] > 0:
            tempo_candidates = tempo
        else:
            tempo_candidates = np.array([120.0])

    # Try all tempo candidates and keep the one whose beat grid best matches the ODF.
    candidate_bpms = []
    for t in tempo_candidates:
        candidate_bpms.append(float(t))

    # Also try octave variants because half/double tempo errors are common.
    #for t in list(candidate_bpms):
        #if 60.0 <= t * 2 <= 200.0:
          #  candidate_bpms.append(float(t * 2))
        #if 60.0 <= t / 2 <= 200.0:
           # candidate_bpms.append(float(t / 2))

    candidate_bpms = sorted(set(round(t, 6) for t in candidate_bpms))

    best_cleaned = np.array([], dtype=float)
    best_total_score = -np.inf

    for bpm in candidate_bpms:
        beat_period = 60.0 / bpm

        if beat_period <= 0:
            continue


        # Phase search
        num_phase_tests = max(8, int(round(beat_period * odf_rate)))
        phase_offsets = np.linspace(0, beat_period, num_phase_tests, endpoint=False)

        best_offset = 0.0
        best_score = -np.inf

        for offset in phase_offsets:
            beat_times = np.arange(offset, duration, beat_period)

            if beat_times.size == 0:
                continue

            frame_indices = np.round(beat_times * odf_rate).astype(int)
            frame_indices = frame_indices[
                (frame_indices >= 0) & (frame_indices < len(odf_norm))
            ]

            if frame_indices.size == 0:
                continue

            # Mean score avoids always preferring faster tempi just because they create more beats.
            score = np.mean(odf_norm[frame_indices])

            if score > best_score:
                best_score = score
                best_offset = offset

        beat_times = np.arange(best_offset, duration, beat_period)

        snap_window_sec = 0.17
        snap_window = max(1, int(round(snap_window_sec * odf_rate)))

        snapped_beats = []

        for beat_time in beat_times:
            center = int(round(beat_time * odf_rate))

            left = max(0, center - snap_window)
            right = min(len(odf_norm), center + snap_window + 1)

            if right <= left:
                continue

            local_region = odf_norm[left:right]

            if local_region.size == 0:
                continue

            local_best = left + int(np.argmax(local_region))

            if odf_norm[local_best] > 0.005:
                final_time = local_best / odf_rate
            else:
                final_time = beat_time

            if 0 <= final_time <= duration:
                snapped_beats.append(final_time)

        snapped_beats = np.asarray(snapped_beats, dtype=float)

        if snapped_beats.size == 0:
            continue

        snapped_beats = np.sort(snapped_beats)

        cleaned = [snapped_beats[0]]
        min_distance = 0.27

        for b in snapped_beats[1:]:
            if b - cleaned[-1] >= min_distance:
                cleaned.append(b)
            else:
                old_idx = int(round(cleaned[-1] * odf_rate))
                new_idx = int(round(b * odf_rate))

                old_idx = np.clip(old_idx, 0, len(odf_norm) - 1)
                new_idx = np.clip(new_idx, 0, len(odf_norm) - 1)

                if odf_norm[new_idx] > odf_norm[old_idx]:
                    cleaned[-1] = b

        cleaned = np.asarray(cleaned, dtype=float)

        cleaned_indices = np.round(cleaned * odf_rate).astype(int)
        cleaned_indices = cleaned_indices[
            (cleaned_indices >= 0) & (cleaned_indices < len(odf_norm))
        ]

        if cleaned_indices.size == 0:
            continue

        # Combined score:
        # - high average ODF at beat positions
        # - enough beats, but not too many
        avg_strength = np.mean(odf_norm[cleaned_indices])
        beat_count = len(cleaned)
        expected_count = duration / beat_period
        count_penalty = abs(beat_count - expected_count) / max(expected_count, 1)

        total_score = avg_strength

        if total_score > best_total_score:
            best_total_score = total_score
            best_cleaned = cleaned

    return np.asarray(best_cleaned, dtype=float)

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

