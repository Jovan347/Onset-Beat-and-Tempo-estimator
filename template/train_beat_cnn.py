#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a small CNN model for frame-level beat activation prediction.

Input:
    beat_dataset_train.npz

Output:
    beat_cnn_model.pt
"""

from argparse import ArgumentParser
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def opts_parser():
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str, help="Input .npz dataset")
    parser.add_argument("outfile", type=str, help="Output model .pt file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


class BeatChunkDataset(Dataset):
    def __init__(self, features, targets, indices, chunk_size=512, augment=False):
        self.features = features
        self.targets = targets
        self.indices = indices
        self.chunk_size = chunk_size
        self.augment = augment

        self.items = []

        for file_idx in self.indices:
            length = len(self.features[file_idx])

            if length <= chunk_size:
                self.items.append((file_idx, 0))
            else:
                step = chunk_size // 2
                for start in range(0, length - chunk_size + 1, step):
                    self.items.append((file_idx, start))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_idx, start = self.items[idx]

        x = self.features[file_idx]
        y = self.targets[file_idx]

        end = start + self.chunk_size

        x_chunk = x[start:end]
        y_chunk = y[start:end]

        # Pad short examples
        if len(x_chunk) < self.chunk_size:
            pad_len = self.chunk_size - len(x_chunk)

            x_pad = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
            y_pad = np.zeros((pad_len,), dtype=np.float32)

            x_chunk = np.concatenate([x_chunk, x_pad], axis=0)
            y_chunk = np.concatenate([y_chunk, y_pad], axis=0)

        if self.augment:
            # tiny feature noise for robustness
            noise = np.random.normal(0, 0.01, size=x_chunk.shape).astype(np.float32)
            x_chunk = x_chunk + noise

        # PyTorch Conv1D expects [features, time]
        x_tensor = torch.tensor(x_chunk.T, dtype=torch.float32)
        y_tensor = torch.tensor(y_chunk, dtype=torch.float32)

        return x_tensor, y_tensor


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
        # x: [batch, features, time]
        logits = self.net(x)
        # output: [batch, time]
        return logits[:, 0, :]


def evaluate_model(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    parser = opts_parser()
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    data = np.load(args.dataset, allow_pickle=True)

    features = data["features"]
    targets = data["targets"]

    num_files = len(features)
    all_indices = list(range(num_files))

    random.shuffle(all_indices)

    split = int(0.8 * num_files)
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    print(f"Files total: {num_files}")
    print(f"Train files: {len(train_indices)}")
    print(f"Val files: {len(val_indices)}")

    train_dataset = BeatChunkDataset(
        features,
        targets,
        train_indices,
        chunk_size=args.chunk_size,
        augment=True,
    )

    val_dataset = BeatChunkDataset(
        features,
        targets,
        val_indices,
        chunk_size=args.chunk_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BeatCNN(num_features=81).to(device)

    # Beat frames are rare, so use positive weighting.
    all_train_targets = np.concatenate([targets[i] for i in train_indices])
    positive = np.sum(all_train_targets > 0)
    negative = np.sum(all_train_targets == 0)

    pos_weight_value = negative / max(positive, 1)
    pos_weight_value = min(pos_weight_value, 20.0)

    print(f"Positive frames: {positive}")
    print(f"Negative frames: {negative}")
    print(f"pos_weight: {pos_weight_value:.3f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_features": 81,
                    "fps": 70,
                    "chunk_size": args.chunk_size,
                    "val_loss": best_val_loss,
                },
                args.outfile,
            )

            print(f"  saved best model to {args.outfile}")

    print()
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Done. Model saved to {args.outfile}")


if __name__ == "__main__":
    main()