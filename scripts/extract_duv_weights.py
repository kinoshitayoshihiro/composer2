#!/usr/bin/env python
"""
Extract DUV model weights from Lightning checkpoint and save as standard PyTorch state_dict.
"""

import torch
import argparse
from pathlib import Path


def extract_lightning_weights(ckpt_path: str, output_path: str):
    """Extract model weights from Lightning checkpoint."""
    print(f"Loading Lightning checkpoint: {ckpt_path}")

    # Load Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract model state_dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]

        # Remove 'model.' prefix from keys
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                clean_key = key[6:]  # Remove 'model.' prefix
                clean_state_dict[clean_key] = value
            else:
                clean_state_dict[key] = value

        print(f"Extracted {len(clean_state_dict)} parameters")
        print("Sample keys:", list(clean_state_dict.keys())[:5])

        # Save as standard checkpoint
        torch.save(clean_state_dict, output_path)
        print(f"Saved weights to: {output_path}")

    else:
        print("ERROR: No 'state_dict' found in checkpoint")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract DUV weights from Lightning checkpoint")
    parser.add_argument("ckpt_path", help="Path to Lightning checkpoint")
    parser.add_argument("--output", "-o", help="Output path for extracted weights", default=None)

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        ckpt_path = Path(args.ckpt_path)
        args.output = str(ckpt_path.parent / f"{ckpt_path.stem}_extracted.ckpt")

    success = extract_lightning_weights(args.ckpt_path, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
