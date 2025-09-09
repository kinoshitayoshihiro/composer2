from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.segment_phrase import load_model


class Wrapper(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pitch_class: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        feats = {
            "pitch_class": pitch_class,
            "velocity": velocity,
            "duration": duration,
            "position": position,
        }
        return self.model(feats, mask)


def export_model(arch: str, ckpt: Path, out: Path) -> None:
    model = load_model(arch, ckpt)
    wrapper = Wrapper(model)
    dummy = (
        torch.zeros(1, 32, dtype=torch.long),
        torch.zeros(1, 32),
        torch.zeros(1, 32),
        torch.zeros(1, 32, dtype=torch.long),
        torch.ones(1, 32, dtype=torch.bool),
    )
    torch.onnx.export(wrapper, dummy, out, opset_version=13)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=["transformer", "lstm"], default="transformer"
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export_model(args.arch, args.ckpt, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
