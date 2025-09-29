#!/usr/bin/env python3
"""Simple DUV evaluation using the trained Lightning model directly."""

import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Import Lightning and model
import pytorch_lightning as L

sys.path.append(str(Path(__file__).parent.parent))
from scripts.train_duv import DUVModel, DUVDataModule


def load_lightning_model(ckpt_path):
    """Load trained Lightning model."""
    return DUVModel.load_from_checkpoint(ckpt_path)


def evaluate_samples(model, dataloader, device, limit=1000):
    """Evaluate model on limited samples."""
    model = model.to(device)
    model.eval()

    all_vel_pred = []
    all_vel_true = []
    all_dur_pred = []
    all_dur_true = []

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= limit:
                break

            print(f"Processing batch {batch_idx}, samples so far: {sample_count}")

            # Move batch to device
            features = {k: v.to(device) for k, v in batch["features"].items()}
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            # Forward pass
            outputs = model(features)

            # Extract predictions and targets
            mask = targets["mask"]

            if "vel_reg" in outputs:
                vel_pred = outputs["vel_reg"].squeeze(-1)  # [batch, seq]
                vel_true = targets["velocity"]

                # Apply mask
                vel_pred_masked = vel_pred[mask]
                vel_true_masked = vel_true[mask]

                all_vel_pred.append(vel_pred_masked.cpu().numpy())
                all_vel_true.append(vel_true_masked.cpu().numpy())

                sample_count += len(vel_pred_masked)

            if "dur_reg" in outputs:
                dur_pred = outputs["dur_reg"].squeeze(-1)  # [batch, seq]
                dur_true = targets["duration"]

                # Apply mask
                dur_pred_masked = dur_pred[mask]
                dur_true_masked = dur_true[mask]

                all_dur_pred.append(dur_pred_masked.cpu().numpy())
                all_dur_true.append(dur_true_masked.cpu().numpy())

    # Concatenate results
    if all_vel_pred:
        vel_pred = np.concatenate(all_vel_pred)
        vel_true = np.concatenate(all_vel_true)
    else:
        vel_pred = vel_true = np.array([])

    if all_dur_pred:
        dur_pred = np.concatenate(all_dur_pred)
        dur_true = np.concatenate(all_dur_true)
    else:
        dur_pred = dur_true = np.array([])

    return vel_pred, vel_true, dur_pred, dur_true


def calculate_metrics(vel_pred, vel_true, dur_pred, dur_true):
    """Calculate evaluation metrics."""
    metrics = {}

    # Velocity metrics
    if len(vel_pred) > 0:
        vel_mae = np.mean(np.abs(vel_pred - vel_true))
        vel_rmse = np.sqrt(np.mean((vel_pred - vel_true) ** 2))

        metrics.update(
            {
                "velocity_mae": float(vel_mae),
                "velocity_rmse": float(vel_rmse),
                "velocity_count": len(vel_pred),
                "velocity_pred_stats": {
                    "mean": float(np.mean(vel_pred)),
                    "std": float(np.std(vel_pred)),
                    "min": float(np.min(vel_pred)),
                    "max": float(np.max(vel_pred)),
                    "unique_values": len(np.unique(vel_pred)),
                },
                "velocity_true_stats": {
                    "mean": float(np.mean(vel_true)),
                    "std": float(np.std(vel_true)),
                    "min": float(np.min(vel_true)),
                    "max": float(np.max(vel_true)),
                },
            }
        )

        # Check if predictions are constant
        if len(np.unique(vel_pred)) == 1:
            metrics["velocity_warning"] = "constant_predictions"

        # Correlation if scipy available
        try:
            from scipy.stats import pearsonr, spearmanr

            if len(np.unique(vel_pred)) > 1 and len(np.unique(vel_true)) > 1:
                vel_pearson = pearsonr(vel_pred, vel_true)[0]
                vel_spearman = spearmanr(vel_pred, vel_true)[0]
                metrics["velocity_pearson"] = float(vel_pearson)
                metrics["velocity_spearman"] = float(vel_spearman)
        except ImportError:
            pass

    # Duration metrics
    if len(dur_pred) > 0:
        dur_mae = np.mean(np.abs(dur_pred - dur_true))
        dur_rmse = np.sqrt(np.mean((dur_pred - dur_true) ** 2))

        metrics.update(
            {
                "duration_mae": float(dur_mae),
                "duration_rmse": float(dur_rmse),
                "duration_count": len(dur_pred),
            }
        )

    return metrics


def main():
    device = torch.device("cpu")  # Use CPU for stability

    # Paths
    ckpt_path = "checkpoints/duv_piano_plus.best.ckpt"
    csv_path = "data/duv/piano_plus.valid.csv"
    stats_path = "checkpoints/piano_plus.duv.stats.json"

    print(f"Loading model from {ckpt_path}")

    try:
        model = load_lightning_model(ckpt_path)
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Create data module
        print(f"Setting up data from {csv_path}")
        data_module = DUVDataModule(
            csv_train="data/duv/piano_plus.train.csv",  # Need actual train file
            csv_valid=csv_path,
            stats_json=stats_path,
            batch_size=8,  # Small batch for stability
            max_len=256,
            num_workers=0,  # No parallel loading for simplicity
        )

        data_module.setup("fit")  # Need to setup both train and val
        val_dataloader = data_module.val_dataloader()

        print(f"Data loaded, {len(val_dataloader)} batches")

        # Evaluate
        print("Starting evaluation...")
        vel_pred, vel_true, dur_pred, dur_true = evaluate_samples(
            model, val_dataloader, device, limit=1000
        )

        print(f"Evaluation complete:")
        print(f"  Velocity samples: {len(vel_pred)}")
        print(f"  Duration samples: {len(dur_pred)}")

        # Calculate metrics
        metrics = calculate_metrics(vel_pred, vel_true, dur_pred, dur_true)

        print("\n=== EVALUATION RESULTS ===")
        print(json.dumps(metrics, indent=2))

        # Save results
        output_path = "outputs/duv_piano_plus_simple_metrics.json"
        Path("outputs").mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
