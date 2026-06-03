# Onset, Beat and Tempo Estimator

This repository contains an audio processing project for estimating **musical onsets**, **beats**, and **tempo** from `.wav` files. The project was developed for an Audio and Music Processing challenge and extends the original template with a working onset detector, tempo estimator, beat tracker, and a small CNN-based beat activation model.

The final system combines signal-processing methods with a lightweight neural model for beat detection. Onsets and tempo are estimated from an onset detection function, while beat tracking can optionally use a trained CNN model if `beat_cnn_model.pt` is available.

## Project Overview

The task is to process a folder of audio files and produce a JSON file containing predictions for each track:

```json
{
  "track_name": {
    "onsets": [0.123, 0.456],
    "beats": [0.500, 1.000],
    "tempo": [120.0, 60.0]
  }
}
```

The project includes three main subtasks:

- **Onset detection**: detect the time positions where new musical events begin.
- **Tempo estimation**: estimate the main tempo of the track in beats per minute.
- **Beat detection**: estimate the sequence of beat positions over time.

## Main Features

The implemented detector includes:

- log-mel spectrogram feature extraction
- positive spectral flux onset detection function
- adaptive thresholding and peak picking for onset detection
- autocorrelation-based tempo estimation
- beat grid generation using estimated tempo and phase search
- beat snapping to nearby activation peaks
- optional CNN-based beat activation prediction
- evaluation scripts for onset, beat, and tempo metrics

## Repository Structure

```text
.
├── template/
│   ├── detector.py                    # Main detector for onsets, beats and tempo
│   ├── detector_baseline_v13.py        # Earlier/baseline detector version
│   ├── detector_onset_backup.py        # Backup onset detector version
│   ├── evaluate.py                     # Full evaluation script
│   ├── evaluate_beats_only.py          # Beat-only evaluation
│   ├── evaluate_onsets_only.py         # Onset-only evaluation
│   ├── make_beat_dataset.py            # Creates beat CNN training dataset
│   ├── make_final_beat_train_folder.py # Combines beat-labeled training folders
│   ├── make_tempobeat_split.py         # Creates train/validation split
│   └── train_beat_cnn.py               # Trains the CNN beat activation model
│
├── .gitignore
└── README.md
```

Large datasets, generated `.npz` files, prediction JSON files, and trained model checkpoints are not meant to be committed to the repository.

## Method Summary

### 1. Feature Extraction

The detector reads each `.wav` file, converts it to mono if needed, and computes a short-time Fourier transform. From this, it creates an 80-bin log-mel spectrogram using:

```text
fps = 70
n_fft = 2048
n_mels = 80
fmin = 27.5
fmax = 8000
```

The mel spectrogram is compressed using:

```python
melspect = np.log1p(100 * melspect)
```

### 2. Onset Detection

The onset detection function is based on positive frame-to-frame changes in the log-mel spectrogram. Negative changes are removed, and the positive differences are summed over frequency bins.

The onset detector then applies:

- median-based normalization
- light smoothing
- local adaptive thresholding
- peak picking
- a minimum-distance constraint between detected onsets

The main onset settings are:

```text
smoothing window:       0.05 * odf_rate
threshold window:       0.20 * odf_rate
threshold offset:       0.03
minimum onset distance: 0.06 * odf_rate
```

### 3. Tempo Estimation

Tempo is estimated from the onset detection function using autocorrelation. The algorithm searches over lags corresponding to a musically reasonable tempo range:

```text
minimum tempo: 60 BPM
maximum tempo: 200 BPM
```

The strongest autocorrelation lag is converted into BPM. The detector also returns an octave-related alternative when useful, because half-tempo and double-tempo ambiguities are common in tempo estimation.

### 4. Beat Detection

Beat tracking uses the estimated tempo to create beat grids. The algorithm tests possible phase offsets and chooses the beat grid that best aligns with the beat activation signal.

The beat detector then snaps each beat to a nearby activation peak and removes beats that are too close together.

Important final beat-tracking settings include:

```text
snap window:            0.17 seconds
minimum beat distance:  0.27 seconds
activation threshold:   0.005
beat-grid score:        average activation strength
```

### 5. CNN Beat Activation Model

The project also includes a small CNN model for frame-level beat activation prediction. The model uses 81 input features per frame:

- 80 normalized log-mel features
- 1 normalized onset detection function feature

The CNN architecture consists of three 1D convolution blocks followed by a final 1D convolution output layer.

The detector automatically tries to load:

```text
beat_cnn_model.pt
```

If the model file is available, the CNN beat activation is used for beat tracking. If the file is missing, the detector falls back to the signal-processing baseline.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>/template
```

Install the required Python packages:

```bash
pip install numpy scipy librosa mir_eval tqdm matplotlib torch
```

PyTorch can also be installed separately from the official PyTorch installation page depending on whether CPU or GPU support is needed.

## Usage

Run the detector on a folder of `.wav` files:

```bash
python detector.py path/to/wav_folder predictions.json
```

For example:

```bash
python detector.py ../test final_predictions.json
```

To visualize features and predictions while processing files, use:

```bash
python detector.py path/to/wav_folder predictions.json --plot
```

## Evaluation

Evaluate all available tasks:

```bash
python evaluate.py path/to/groundtruth predictions.json
```

The ground truth folder may contain:

```text
*.onsets.gt
*.beats.gt
*.tempo.gt
```

Evaluate only beats:

```bash
python evaluate_beats_only.py path/to/groundtruth predictions.json
```

Evaluate only onsets:

```bash
python evaluate_onsets_only.py path/to/groundtruth predictions.json
```

## Training the Beat CNN

First create a beat dataset from a folder containing `.wav` files and `.beats.gt` files:

```bash
python make_beat_dataset.py path/to/beat_train_folder beat_dataset.npz
```

Then train the CNN model:

```bash
python train_beat_cnn.py beat_dataset.npz beat_cnn_model.pt
```

Optional training arguments include:

```bash
python train_beat_cnn.py beat_dataset.npz beat_cnn_model.pt --epochs 20 --batch_size 16 --chunk_size 512 --lr 1e-3
```

The resulting `beat_cnn_model.pt` should be placed in the same working directory from which `detector.py` is run, so that it can be loaded automatically.

## Data Preparation Scripts

The repository includes helper scripts for preparing beat training data.

Create a train/validation split for tempo-beat data:

```bash
python make_tempobeat_split.py
```

Create a final combined beat training folder:

```bash
python make_final_beat_train_folder.py
```

The final beat training setup combines the original training data and the extra tempo/beat data into one folder before creating the final `.npz` beat dataset.

## Reported Development Results

During development, the onset detector reached a strong validation result on the held-out onset split:

```text
Onset validation F-score: 0.8152
```

The best beat-tracking setup on the held-out `tempobeats_val` split achieved:

```text
Beat validation F-score: 0.6971
```

These results were obtained during local development and may differ slightly depending on the exact dataset split, available training files, and whether the CNN checkpoint is present.

## Notes

- The detector writes one JSON entry per audio file stem.
- File names in the output JSON should not include the `.wav` extension.
- `beat_cnn_model.pt` is optional, but improves the intended final beat-tracking pipeline.
- If no CNN checkpoint is found, the code falls back to the baseline activation.
- Generated model files, datasets, and prediction files should not be committed to GitHub.

## Acknowledgement

The original challenge template was provided for the Audio and Music Processing onset, beat, and tempo estimation tasks. This repository extends that template with implemented signal-processing methods and a CNN-based beat activation model.
