from __future__ import annotations

import argparse
import json
from pathlib import Path
from functools import partial

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    IterableDataset = object  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore

from transformer.piano_transformer import PianoTransformer
from transformer.tokenizer_piano import PianoTokenizer


class JsonlDataset(IterableDataset):
    """1 行 1 サンプルの JSONL をトークナイズして返す軽量データセット。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._len: int | None = None  # 初回だけ行数を数え、キャッシュ

    def __iter__(self):
        with self.path.open() as f:
            for line in f:
                obj = json.loads(line)
                tokens = obj.get("ids") or obj.get("tokens")
                if tokens is not None:
                    yield {"input_ids": torch.tensor(tokens, dtype=torch.long)}

    def __len__(self) -> int:  # type: ignore[override]
        if self._len is None:
            with self.path.open() as f:
                self._len = sum(1 for _ in f)
        return self._len


def collate_fn(
    batch: list[dict[str, torch.Tensor]], *, tokenizer: PianoTokenizer
) -> dict[str, torch.Tensor]:
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    pad_id = getattr(tokenizer, "pad_id", 0)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = input_ids.clone()
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    if torch is None:
        raise RuntimeError("Install torch and transformers to run training")

    parser = argparse.ArgumentParser(description="Train PianoTransformer with LoRA")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA scaling factor (default: rank*2)")
    parser.add_argument("--steps", type=int, default=800, help="Training steps")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument(
        "--auto-hparam",
        action="store_true",
        help=(
            "Auto‑scale LoRA rank and steps based on dataset size\n"
            "(<10k: rank=4, steps=800; <30k: rank=8, steps=1200; else: rank=16, steps=2000)"
        ),
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    # 行数を数えてハイパーパラメータの自動調整に使う
    with args.data.open() as f:
        n_samples = sum(1 for _ in f)

    if args.auto_hparam:
        if n_samples < 10_000:
            args.rank, args.steps = 4, 800
        elif n_samples < 30_000:
            args.rank, args.steps = 8, 1_200
        else:
            args.rank, args.steps = 16, 2_000

    if args.epochs is not None:
        args.steps = args.epochs * n_samples

    dataset = JsonlDataset(args.data)
    tokenizer = PianoTokenizer()
    model = PianoTransformer(vocab_size=len(tokenizer.vocab), rank=args.rank, lora_alpha=args.lora_alpha)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        max_steps=args.steps,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )
    trainer.train()

    # 重みを保存（safe_serialization=True で pytorch_model‑00001.safetensors）
    model.model.save_pretrained(str(args.out), safe_serialization=True)

    if args.eval:
        from scripts.evaluate_piano_model import evaluate_dirs

        evaluate_dirs(args.data.parent, args.out, out_dir=args.out / "eval")


if __name__ == "__main__":
    main()
