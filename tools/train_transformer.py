from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    pl = object  # type: ignore

from utilities.groove_transformer import GrooveTransformer, MultiPartDataset, collate_multi_part


def main(argv: Sequence[str] | None = None) -> None:
    if torch is None:
        raise RuntimeError("torch not available")

    ap = argparse.ArgumentParser(prog="train_transformer")
    ap.add_argument("data", type=Path, help="directory with multi-part loops")
    ap.add_argument("--parts", type=str, default="drums,bass,piano,perc")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ns = ap.parse_args(argv)

    parts = [p.strip() for p in ns.parts.split(",") if p.strip()]

    # load synthetic dataset for now: expecting <data>/<part>.txt with token ids
    sequences: list[dict[str, list[int]]] = []
    for p in parts:
        part_file = ns.data / f"{p}.txt"
        tokens = []
        if part_file.is_file():
            for line in part_file.read_text().splitlines():
                seq = [int(x) for x in line.strip().split()] if line.strip() else []
                while len(sequences) <= len(tokens):
                    sequences.append({pp: [] for pp in parts})
                sequences[len(tokens)][p] = seq
                tokens.append(seq)
    vocab_sizes = {p: 1 + max((max(s[p]) if s[p] else 0) for s in sequences) for p in parts}

    dataset = MultiPartDataset(sequences, parts)
    dl = DataLoader(dataset, batch_size=ns.batch, shuffle=True, collate_fn=lambda b: {"input": collate_multi_part(b, parts), "target": collate_multi_part(b, parts)})

    model = GrooveTransformer(vocab_sizes, d_model=ns.d_model, nhead=ns.heads, num_layers=ns.layers)

    logger = pl.loggers.TensorBoardLogger("runs", name="groove_transformer")
    trainer = pl.Trainer(max_epochs=ns.epochs, logger=logger, enable_progress_bar=False)
    trainer.fit(model, dl)
    Path("models").mkdir(exist_ok=True)
    trainer.save_checkpoint("models/groove_transformer.ckpt")


if __name__ == "__main__":
    main()
