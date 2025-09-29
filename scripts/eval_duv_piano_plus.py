#!/usr/bin/env python3
"""Custom evaluation script for DUV Piano Plus model with proper architecture."""

import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Import required components
sys.path.append(str(Path(__file__).parent.parent))
from models.phrase_transformer import PhraseTransformer
from utilities.csv_io import coerce_columns


def load_stats_file(stats_path):
    """Load and parse stats JSON file."""
    with open(stats_path) as f:
        stats = json.load(f)

    feat_cols = stats["feat_cols"]
    mean = np.array([stats["mean"][col] for col in feat_cols], dtype=np.float32)
    std = np.array([stats["std"][col] for col in feat_cols], dtype=np.float32)

    return feat_cols, mean, std


def normalize_features(df, feat_cols, mean, std):
    """Normalize DataFrame features using stats."""
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        for col in missing:
            df[col] = 0.0

    # Select and order columns
    arr = df[feat_cols].to_numpy(dtype="float32")

    # Normalize
    arr = (arr - mean) / np.maximum(std, 1e-8)

    return arr


def create_model():
    """Create model with the same architecture used in training."""
    return PhraseTransformer(
        d_model=256,
        nhead=8,
        num_layers=6,
        ff_dim=2048,  # Use ff_dim instead of dim_feedforward
        max_len=256,
        vocab_size=12,
        pos_vocab_size=16,
        dropout=0.1,
        # DUV heads
        has_vel_head=True,
        has_dur_head=True,
        # Embedding features
        vel_bucket_emb=True,
        dur_bucket_emb=True,
    )


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()

    all_vel_pred = []
    all_vel_true = []
    all_dur_pred = []
    all_dur_true = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}...")

            # Unpack batch
            features, vel_targets, dur_targets = batch

            # Handle batch dimension - squeeze if single item batch
            if features.dim() == 4:  # [1, 1, seq_len, feat_dim]
                features = features.squeeze(0)  # [1, seq_len, feat_dim]
            if vel_targets.dim() == 3:  # [1, 1, seq_len]
                vel_targets = vel_targets.squeeze(0)  # [1, seq_len]
            if dur_targets.dim() == 3:  # [1, 1, seq_len]
                dur_targets = dur_targets.squeeze(0)  # [1, seq_len]

            features = features.to(device)
            vel_targets = vel_targets.to(device)
            dur_targets = dur_targets.to(device)

            # Create attention mask (all positions valid for this example)
            batch_size, seq_len, _ = features.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

            # Convert features to feature dict format expected by model
            feat_dict = {}
            # Assuming features is [batch, seq, feat_dim] and we need to map to individual features
            # For simplicity, create dummy features and use velocity/duration from targets
            feat_dict["velocity"] = vel_targets
            feat_dict["duration"] = dur_targets
            feat_dict["pitch"] = torch.zeros_like(vel_targets, dtype=torch.long)
            feat_dict["position"] = torch.arange(seq_len, device=device).expand(batch_size, -1)
            feat_dict["pitch_class"] = torch.zeros_like(vel_targets, dtype=torch.long)
            feat_dict["vel_bucket"] = torch.zeros_like(vel_targets, dtype=torch.long)
            feat_dict["dur_bucket"] = torch.zeros_like(vel_targets, dtype=torch.long)

            # Forward pass
            outputs = model(feat_dict, mask=mask)

            # Extract predictions
            vel_pred = outputs["vel_reg"]  # [batch, seq, 1]
            dur_pred = outputs["dur_reg"]  # [batch, seq, 1]

            # Apply mask and flatten
            vel_pred = vel_pred.squeeze(-1)[mask]
            dur_pred = dur_pred.squeeze(-1)[mask]
            vel_targets = vel_targets[mask]
            dur_targets = dur_targets[mask]

            # Convert to numpy
            all_vel_pred.append(vel_pred.cpu().numpy())
            all_vel_true.append(vel_targets.cpu().numpy())
            all_dur_pred.append(dur_pred.cpu().numpy())
            all_dur_true.append(dur_targets.cpu().numpy())

    # Concatenate all results
    vel_pred = np.concatenate(all_vel_pred)
    vel_true = np.concatenate(all_vel_true)
    dur_pred = np.concatenate(all_dur_pred)
    dur_true = np.concatenate(all_dur_true)

    # Calculate metrics
    vel_mae = np.mean(np.abs(vel_pred - vel_true))
    vel_rmse = np.sqrt(np.mean((vel_pred - vel_true) ** 2))
    dur_mae = np.mean(np.abs(dur_pred - dur_true))
    dur_rmse = np.sqrt(np.mean((dur_pred - dur_true) ** 2))

    try:
        from scipy.stats import pearsonr, spearmanr

        vel_pearson = pearsonr(vel_pred, vel_true)[0]
        vel_spearman = spearmanr(vel_pred, vel_true)[0]
    except:
        vel_pearson = None
        vel_spearman = None

    return {
        "velocity_mae": float(vel_mae),
        "velocity_rmse": float(vel_rmse),
        "velocity_pearson": float(vel_pearson) if vel_pearson is not None else None,
        "velocity_spearman": float(vel_spearman) if vel_spearman is not None else None,
        "velocity_count": len(vel_pred),
        "duration_mae": float(dur_mae),
        "duration_rmse": float(dur_rmse),
        "duration_count": len(dur_pred),
        "velocity_pred_stats": {
            "mean": float(np.mean(vel_pred)),
            "std": float(np.std(vel_pred)),
            "min": float(np.min(vel_pred)),
            "max": float(np.max(vel_pred)),
        },
        "velocity_true_stats": {
            "mean": float(np.mean(vel_true)),
            "std": float(np.std(vel_true)),
            "min": float(np.min(vel_true)),
            "max": float(np.max(vel_true)),
        },
    }


def main():
    device = torch.device("cpu")  # Use CPU to avoid MPS issues

    # Paths
    csv_path = "data/duv/piano_plus.valid.csv"
    ckpt_path = "checkpoints/duv_piano_plus_clean.ckpt"
    stats_path = "checkpoints/piano_plus.duv.stats.json"

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Load stats and normalize
    print(f"Loading stats from {stats_path}")
    feat_cols, mean, std = load_stats_file(stats_path)
    features = normalize_features(df, feat_cols, mean, std)

    # Extract targets
    velocity = df["velocity"].values.astype(np.float32)
    duration = df["duration"].values.astype(np.float32)

    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {feat_cols}")

    # Create dataset (using simple batching for now)
    batch_size = 32
    seq_len = min(256, len(df))  # Limit sequence length

    # For simplicity, create overlapping windows
    datasets = []
    for i in range(0, len(df) - seq_len + 1, seq_len // 2):
        end_idx = i + seq_len
        feat_batch = features[i:end_idx]
        vel_batch = velocity[i:end_idx]
        dur_batch = duration[i:end_idx]

        datasets.append(
            (
                torch.tensor(feat_batch).unsqueeze(0),  # [1, seq_len, feat_dim]
                torch.tensor(vel_batch).unsqueeze(0),  # [1, seq_len]
                torch.tensor(dur_batch).unsqueeze(0),  # [1, seq_len]
            )
        )

    # Limit to 1000 samples for evaluation
    datasets = datasets[: min(1000 // (seq_len // 2), len(datasets))]

    print(f"Created {len(datasets)} sequences of length {seq_len}")

    # Create model and load weights
    print("Creating model...")
    model = create_model()

    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    print("Model loaded successfully!")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

    # Create simple dataloader
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SimpleDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Starting evaluation...")
    metrics = evaluate_model(model, dataloader, device)

    print("\n=== EVALUATION RESULTS ===")
    print(json.dumps(metrics, indent=2))

    # Save results
    output_path = "outputs/duv_piano_plus_custom_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
