#!/usr/bin/env python3
"""Test script to verify DUV head fix in PhraseTransformer."""

import sys

sys.path.insert(0, "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3")

import torch
import numpy as np
from models.phrase_transformer import PhraseTransformer


def test_duv_heads():
    """Test that DUV heads are properly initialized and can be called."""
    print("Testing DUV heads in PhraseTransformer...")

    # Create model
    model = PhraseTransformer(d_model=256, max_len=16, ff_dim=2048)
    model.eval()

    # Check if DUV heads exist and are not None
    print(f"head_vel_reg exists: {hasattr(model, 'head_vel_reg')}")
    print(f"head_vel_reg is not None: {getattr(model, 'head_vel_reg', None) is not None}")
    print(f"head_dur_reg exists: {hasattr(model, 'head_dur_reg')}")
    print(f"head_dur_reg is not None: {getattr(model, 'head_dur_reg', None) is not None}")

    if hasattr(model, "head_vel_reg") and model.head_vel_reg is not None:
        print(f"head_vel_reg type: {type(model.head_vel_reg)}")
        print(f"head_vel_reg: {model.head_vel_reg}")

    if hasattr(model, "head_dur_reg") and model.head_dur_reg is not None:
        print(f"head_dur_reg type: {type(model.head_dur_reg)}")
        print(f"head_dur_reg: {model.head_dur_reg}")

    # Create dummy input
    feats = {
        "position": torch.randint(0, 16, (1, 8)),
        "pitch_class": torch.randint(0, 12, (1, 8)),
        "velocity": torch.rand(1, 8),
        "duration": torch.rand(1, 8),
    }

    # Debug: check torch state in forward
    print(f"torch module available: {torch is not None}")

    # Test forward pass
    with torch.no_grad():
        outputs = model(feats)

    print(f"Model output type: {type(outputs)}")

    if isinstance(outputs, dict):
        print("✅ Model returns dictionary (DUV mode)")
        for key, value in outputs.items():
            print(f"  {key}: shape={tuple(value.shape)}")
            print(
                f"    min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}"
            )
    else:
        print("❌ Model returns tensor (boundary only mode)")
        print(f"  Output shape: {tuple(outputs.shape)}")
        print(
            f"  Output stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}"
        )

        # Debug: Check if the model has the right methods called
        print("Debugging forward logic...")
        # Check if DUV logic is correctly triggered
        has_vel = hasattr(model, "head_vel_reg") and model.head_vel_reg is not None
        has_dur = hasattr(model, "head_dur_reg") and model.head_dur_reg is not None
        print(f"  has_vel: {has_vel}, has_dur: {has_dur}")
        print(f"  should return dict: {has_vel or has_dur}")

    return isinstance(outputs, dict)


if __name__ == "__main__":
    success = test_duv_heads()
    if success:
        print("\n✅ DUV heads test PASSED")
    else:
        print("\n❌ DUV heads test FAILED")
    sys.exit(0 if success else 1)
