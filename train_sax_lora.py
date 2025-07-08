from __future__ import annotations

"""Train :class:`SaxTransformer` on a JSONL corpus with LoRA.

This script supports automatic hyper-parameter scaling via ``--auto-hparam``
and pads variable-length sequences in ``collate_fn``.
"""

import argparse
import json
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore

from transformer.sax_transformer import SaxTransformer
from transformer.sax_tokenizer import SaxTokenizer


class JsonlDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.items: list[list[int]] = []
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            tokens = obj.get("ids") or obj.get("tokens")
            if tokens is not None:
                self.items.append(tokens)

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover - simple container
        return {"input_ids": torch.tensor(self.items[idx], dtype=torch.long)}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:  # pragma: no cover - simple
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, x in enumerate(batch):
        seq = x["input_ids"]
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    if torch is None:
        raise RuntimeError("Install torch and transformers to run training")

    parser = argparse.ArgumentParser(description="Train SaxTransformer with LoRA")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=800, help="Training steps")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument(
        "--auto-hparam",
        action="store_true",
        help="auto scale LoRA rank and steps based on dataset size",
    )
    args = parser.parse_args()

    n_samples = sum(1 for _ in open(args.data))
    if args.auto_hparam:
        if n_samples < 10_000:
            args.rank = 4
            args.steps = 800
        elif n_samples < 30_000:
            args.rank = 8
            args.steps = 1_200
        else:
            args.rank = 16
            args.steps = 2_000

    if args.epochs is not None:
        args.steps = args.epochs * n_samples

    dataset = JsonlDataset(args.data)
    tokenizer = SaxTokenizer()
    model = SaxTransformer(vocab_size=len(tokenizer.vocab), rank=args.rank)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        max_steps=args.steps,
        logging_steps=10,
        save_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    model.model.save_pretrained(str(args.out))


if __name__ == "__main__":
    main()
