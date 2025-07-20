from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data.articulation_datamodule import ArticulationDataModule
from ml_models.tagger_module import TaggerModule
from utilities.settings import settings


class EvalCLI(LightningCLI):
    def __init__(self) -> None:
        super().__init__(
            TaggerModule,
            ArticulationDataModule,
            run=False,
            save_config_callback=None,
        )
        self.evaluate()

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--schema_path",
            type=Path,
            default=settings.schema_path,
        )

    def evaluate(self) -> None:
        model = TaggerModule.load_from_checkpoint(self.config.ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.datamodule.setup("validate")
        loader = self.datamodule.val_dataloader()

        preds: list[int] = []
        labels: list[int] = []
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model.decode_batch(batch)[0]
            mask = batch["pad_mask"][0].bool()
            lab = batch["labels"][0][mask].tolist()
            preds.extend(pred[: len(lab)])
            labels.extend(lab)

        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "micro_f1": float(f1_score(labels, preds, average="micro")),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        cm = confusion_matrix(labels, preds).astype(float)
        cm /= cm.sum(axis=1, keepdims=True)
        schema_path = self.config.schema_path
        labels_map = yaml.safe_load(Path(schema_path).read_text())
        ticks = list(labels_map.keys())
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(ticks)))
        ax.set_yticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=45, ha="right")
        ax.set_yticklabels(ticks)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.savefig("cm_normalized.svg")
        plt.savefig("cm_normalized.png", dpi=300)


def main() -> None:  # pragma: no cover - CLI
    EvalCLI()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
