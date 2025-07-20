from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data.articulation_data import ArticulationDataModule
from ml_models.articulation_tagger import ArticulationTagger


def main() -> None:
    ap = ArgumentParser(description="Evaluate articulation model")
    ap.add_argument("data", type=Path)
    ap.add_argument("schema", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("metrics.json"))
    args = ap.parse_args()

    dm = ArticulationDataModule(args.data, args.schema, batch_size=8)
    dm.setup()
    model = ArticulationTagger.load_from_checkpoint(str(args.checkpoint))
    trainer = pl.Trainer(logger=False)
    preds = trainer.predict(model, dm)

    all_y = []
    all_p = []
    for emission, label, mask in preds:
        y = label[mask.bool()]
        p = model.crf.decode(emission, mask=mask)[0]
        all_y.extend(y.tolist())
        all_p.extend(p)
    acc = accuracy_score(all_y, all_p)
    f1_macro = f1_score(all_y, all_p, average="macro")
    f1_micro = f1_score(all_y, all_p, average="micro")
    cm = confusion_matrix(all_y, all_p)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = list(dm.label_map.keys())
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title("confusion (normalized)")
    plt.tight_layout()
    plt.savefig(args.out.with_suffix(".svg"))
    plt.savefig(args.out.with_name("cm_normalized.svg"))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y, all_p, labels=list(range(len(labels)))
    )
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "per_label": {
            lbl: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for lbl, p, r, f in zip(labels, precision, recall, f1)
        },
    }
    with args.out.open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
