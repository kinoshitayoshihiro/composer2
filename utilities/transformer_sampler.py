from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from .groove_transformer import GrooveTransformer


def load(path: Path | str) -> GrooveTransformer:
    if torch is None:
        raise RuntimeError("torch not available")
    ckpt = torch.load(str(path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    model = GrooveTransformer(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def sample_multi(
    model: GrooveTransformer,
    history: Dict[str, Sequence[int]] | None,
    length: int,
    temperature: Dict[str, float] | None = None,
) -> Dict[str, list[dict[str, Any]]]:
    if torch is None:
        raise RuntimeError("torch not available")
    device = next(model.parameters()).device
    history = history or {p: [] for p in model.parts}
    temp = temperature or {}
    tokens = {
        p: torch.tensor(history.get(p, []) or [0], dtype=torch.long, device=device).unsqueeze(0)
        for p in model.parts
    }
    events = {p: [] for p in model.parts}
    for i in range(length):
        with torch.no_grad():
            out = model(tokens)
        next_ids = {}
        for p in model.parts:
            logits = out[p][0, -1]
            t = temp.get(p, 1.0)
            if t <= 0:
                idx = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / t, dim=-1)
                idx = int(torch.multinomial(probs, 1).item())
            next_ids[p] = idx
            tokens[p] = torch.cat([tokens[p], torch.tensor([[idx]], device=device)], dim=1)
            events[p].append({
                "instrument": str(idx),
                "offset": (len(tokens[p][0]) - 1) / model.hparams.resolution,
                "duration": 1 / model.hparams.resolution,
            })
    return events


__all__ = ["load", "sample_multi"]


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="transformer_sampler")
    sub = ap.add_subparsers(dest="cmd")
    sample_p = sub.add_parser("sample")
    sample_p.add_argument("model", type=Path)
    sample_p.add_argument("--parts", type=str, default="drums,bass,piano,perc")
    sample_p.add_argument("--length", type=int, default=16)
    ns = ap.parse_args(argv)

    if ns.cmd == "sample":
        model = load(ns.model)
        parts = [p.strip() for p in ns.parts.split(",")]
        events = sample_multi(model, {p: [] for p in parts}, ns.length)
        print(events)
    else:
        ap.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
