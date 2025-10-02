#!/usr/bin/env python3
"""
LoRA DUV Inference Utilities
Load and run inference with LoRA-adapted DUV models saved as Lightning checkpoints.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_lora_duv_model(
    lora_checkpoint_path: str,
    base_checkpoint_path: Optional[str] = None,
    device: str = "cpu",
):
    """
    Load a LoRA-adapted DUV model from Lightning checkpoint.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint (duv_lora_final.ckpt)
        base_checkpoint_path: Optional path to base model (not needed if LoRA checkpoint contains full model)
        device: Device to load model on

    Returns:
        Model ready for inference
    """
    logger.info(f"Loading LoRA model from: {lora_checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(lora_checkpoint_path, map_location=device)

    # Extract model state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Import model architecture from train_duv_lora
    # We need to reconstruct the model with LoRA layers
    import sys
    from pathlib import Path as PathLib

    # Add project root to path
    project_root = PathLib(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.train_duv_lora import DUVLoRAModel
    from models.phrase_transformer import PhraseTransformer

    # Get hyperparameters from checkpoint if available
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        logger.info("Found hyperparameters in checkpoint")
    else:
        # Use default hyperparameters
        hparams = {
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "max_len": 128,
            "num_features": 8,
            "w_vel": 1.0,
            "w_dur": 1.0,
            "huber_delta": 1.0,
        }
        logger.warning("No hyperparameters found, using defaults")

    # Extract model parameters (handle different naming conventions)
    # Note: hyperparameters in checkpoint might be incorrect, infer from weights
    # Check actual model dimensions from state_dict
    pointer_keys = [k for k in state_dict.keys() if "pointer" in k and state_dict[k].dim() == 2]
    if pointer_keys:
        actual_d_model = state_dict[pointer_keys[0]].shape[0]
        logger.info(f"Inferred d_model={actual_d_model} from weights")
        d_model = actual_d_model
    else:
        d_model = hparams.get("d_model", 128)

    n_heads = hparams.get("n_heads", hparams.get("nhead", 4))
    n_layers = hparams.get("n_layers", hparams.get("num_encoder_layers", 4))
    ff_dim = hparams.get("ff_dim", hparams.get("dim_feedforward", 512))
    dropout = hparams.get("dropout", 0.1)
    max_len = hparams.get("max_len", 128)

    logger.info(f"Creating model: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # Create model architecture
    base_model = PhraseTransformer(
        d_model=d_model,
        nhead=n_heads,
        num_encoder_layers=n_layers,
        dim_feedforward=ff_dim,
        dropout=dropout,
        max_len=max_len,
        num_features=8,  # Fixed for CSV format
    )

    # Create Lightning model wrapper
    model = DUVLoRAModel(
        model=base_model,
        w_vel=hparams.get("w_vel", 1.0),
        w_dur=hparams.get("w_dur", 1.0),
        huber_delta=hparams.get("huber_delta", 1.0),
    )

    # Load state dict (remove "model." prefix if present)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove "model." prefix from keys
        new_key = key.replace("model.", "", 1) if key.startswith("model.") else key
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded model state dict")
    except Exception as e:
        logger.warning(f"Could not load full state dict: {e}")
        logger.info("Attempting to load model submodule only...")
        model.model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    model.to(device)

    logger.info(f"Model loaded successfully on {device}")
    return model


def predict_with_lora_model(
    model,
    features: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run inference with LoRA model.

    Args:
        model: Loaded LoRA model
        features: Input features (N, num_features)
        device: Device to run on

    Returns:
        Predictions (N, 2) where columns are [velocity, duration]
    """
    model.eval()

    with torch.no_grad():
        # Convert to tensor
        x_tensor = torch.from_numpy(features.astype(np.float32)).to(device)

        # Add batch dimension if needed
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)  # (1, seq_len, num_features)

        # Forward pass
        outputs = model.model({"features": x_tensor})

        # Extract velocity and duration predictions
        vel_pred = outputs["vel_reg"].cpu().numpy()
        dur_pred = outputs["dur_reg"].cpu().numpy()

        # Stack into (N, 2) array
        predictions = np.stack([vel_pred.flatten(), dur_pred.flatten()], axis=1)

    return predictions


class LoRADUVModel:
    """Wrapper to make LoRA model compatible with duv_infer.py interface."""

    def __init__(self, lora_checkpoint_path: str, device: str = "cpu"):
        self.model = load_lora_duv_model(lora_checkpoint_path, device=device)
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using LoRA model."""
        return predict_with_lora_model(self.model, X, self.device)


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lora_duv_infer.py <lora_checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    logging.basicConfig(level=logging.INFO)

    try:
        model = load_lora_duv_model(checkpoint_path)
        print(f"✅ Successfully loaded model from {checkpoint_path}")
        print(f"Model: {type(model)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
