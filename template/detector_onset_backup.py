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
    import tqdm
except ImportError:
    tqdm = None


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
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    # we only have a dumb dummy implementation here.
    # it uses the time difference between the first two onsets to
    # define the tempo, and returns half of that as a second guess.
    # this is not a useful solution at all, just a placeholder.
    onsets = np.asarray(onsets, dtype=float)

    if onsets.size < 2:
        return np.array([120.0])

    intervals = np.diff(onsets)
    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        return np.array([120.0])

    median_interval = np.median(intervals)

    if median_interval <= 0:
        return np.array([120.0])

    tempo = 60.0 / median_interval
    return np.array([tempo, 2 * tempo])


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.
    onsets = np.asarray(onsets, dtype=float)
    tempo = np.asarray(tempo, dtype=float)

    if onsets.size == 0:
        return np.array([], dtype=float)

    if tempo.size == 0 or tempo[0] <= 0:
        return np.array([onsets[0]])

    beat_interval = 60.0 / tempo[0]

    if beat_interval <= 0:
        return np.array([onsets[0]])

    start = onsets[0]
    end = onsets[-1]

    return np.arange(start, end + beat_interval, beat_interval)

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

