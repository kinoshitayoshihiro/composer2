from __future__ import annotations

"""Train :class:`SaxTransformer` on a JSONL corpus with LoRA.

This script supports automatic hyper‑parameter scaling via ``--auto-hparam``
and pads variable‑length sequences in ``collate_fn``.
"""

import argparse
import json
from functools import partial
from pathlib import Path

try:
    import torch
    from torch.utils.data import IterableDataset
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover – optional deps
    torch = None  # type: ignore
    IterableDataset = object  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore

from transformer.sax_tokenizer import SaxTokenizer
from transformer.sax_transformer import SaxTransformer


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class JsonlDataset(IterableDataset):  # type: ignore[misc]
    """Stream JSONL lines and yield token tensors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._len: int | None = None  # cache file length

    # streaming iterator
    def __iter__(self):
        with self.path.open() as f:
            for line in f:
                obj = json.loads(line)
                tokens = obj.get("ids") or obj.get("tokens")
                if tokens is not None:
                    yield {"input_ids": torch.tensor(tokens, dtype=torch.long)}

    # cached length
    def __len__(self) -> int:  # pragma: no cover – simple container
        if self._len is None:
            with self.path.open() as f:
                self._len = sum(1 for _ in f)
        return self._len


# --------------------------------------------------------------------------- #
# Collate ‑ pad to max‑length in mini‑batch
# --------------------------------------------------------------------------- #
def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    *,
    tokenizer: SaxTokenizer,
) -> dict[str, torch.Tensor]:  # pragma: no cover – simple
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    pad_id = getattr(tokenizer, "pad_id", tokenizer.vocab.get("<pad>", 0))

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["input_ids"]
        input_ids[i, : len(seq)] = seq

    labels = input_ids.clone()
    attention_mask = (input_ids != pad_id).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# --------------------------------------------------------------------------- #
# Entry‑point
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: C901 – top‑level script
    if torch is None:
        raise RuntimeError("Install torch and transformers to run training.")

    # ---------- CLI ----------
    parser = argparse.ArgumentParser(description="Train SaxTransformer with LoRA")
    parser.add_argument("--data", type=Path, required=True, help="JSONL corpus")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint dir")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=None, help="LoRA α (default: rank*2)"
    )
    parser.add_argument(
        "--steps", type=int, default=800, help="training steps (overridden by --epochs)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="epochs (overrides --steps)"
    )
    parser.add_argument(
        "--auto-hparam",
        action="store_true",
        help=(
            "auto‑scale rank & steps by dataset size "
            "(<10k: rank=4/800; <30k: rank=8/1200; else: rank=16/2000)"
        ),
    )
    parser.add_argument(
        "--eval", action="store_true", help="run piano‑style evaluation"
    )
    parser.add_argument(
        "--eval-ref", type=Path, default=None, help="reference MIDI dir"
    )
    parser.add_argument(
        "--eval-gen", type=Path, default=None, help="generated MIDI dir"
    )
    parser.add_argument(
        "--safe", action="store_true", help="save adapters with .safetensors"
    )
    args = parser.parse_args()

    # ---------- Hyper‑param autoscale ----------
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

    # ---------- Objects ----------
    dataset = JsonlDataset(args.data)
    tokenizer = SaxTokenizer()
    model = SaxTransformer(
        vocab_size=len(tokenizer.vocab),
        rank=args.rank,
        lora_alpha=args.lora_alpha,
    )

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

    # ---------- Train ----------
    trainer.train()

    # ---------- Save ----------
    model.model.save_pretrained(str(args.out), safe_serialization=args.safe)

    # ---------- Eval ----------
    if args.eval:
        from scripts.evaluate_piano_model import evaluate_dirs as eval_piano

        if args.eval_ref is not None and args.eval_gen is not None:
            # user‑specified directories
            eval_piano(args.eval_ref, args.eval_gen, out_dir=args.out / "eval")
        else:
            # default: reference = data.parent, generated = out
            eval_piano(args.data.parent, args.out, out_dir=args.out / "eval")


if __name__ == "__main__":
    main()
