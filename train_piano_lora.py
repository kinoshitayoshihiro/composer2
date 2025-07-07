from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore

from transformer.piano_transformer import PianoTransformer
from transformer.tokenizer_piano import PianoTokenizer


class JsonlDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.items: List[List[int]] = []
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            tokens = obj.get("ids") or obj.get("tokens")
            if tokens is not None:
                self.items.append(tokens)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.items[idx], dtype=torch.long)}


def collate_fn(batch: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    if torch is None:
        raise RuntimeError("Install torch and transformers to run training")

    parser = argparse.ArgumentParser(description="Train PianoTransformer with LoRA")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = JsonlDataset(args.data)
    tokenizer = PianoTokenizer()
    model = PianoTransformer(vocab_size=len(tokenizer.vocab))

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate_fn)
    trainer.train()
    model.model.save_pretrained(str(args.out))


if __name__ == "__main__":
    main()
