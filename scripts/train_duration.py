from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import pytorch_lightning as pl
    import torch
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    torch = None  # type: ignore

try:
    from hydra import compose, initialize
    import hydra
    from omegaconf import DictConfig
except Exception:  # pragma: no cover - optional
    hydra = None  # type: ignore
    compose = initialize = None  # type: ignore
    DictConfig = object  # type: ignore

from utilities.duration_datamodule import DurationDataModule
from utilities.ml_duration import DurationTransformer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def run(cfg: DictConfig) -> int:
    if pl is None or torch is None:
        print("PyTorch Lightning required", file=sys.stderr)
        return 1

    dm = DurationDataModule(cfg)
    model = DurationTransformer(cfg.d_model, cfg.max_len)
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5), ckpt_cb],
    )
    trainer.fit(model, dm)
    ckpt = Path(cfg.model.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(ckpt))
    return 0


@hydra.main(
    config_path="../configs", config_name="duration_model.yaml", version_base="1.3"
)
def hydra_main(cfg: DictConfig) -> int:  # pragma: no cover - entry point
    return run(cfg)


def main(argv: list[str] | None = None) -> int:
    if hydra is None:
        print("hydra-core required", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(prog="train_duration.py")
    parser.add_argument("--data", type=Path, help="CSV file")
    parser.add_argument(
        "--out", type=Path, default=Path("model.ckpt"), help="Checkpoint output path"
    )
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument(
        "--batch-size", "--batch", dest="batch", type=int, help="Batch size"
    )
    parser.add_argument("--d_model", type=int, help="Transformer hidden size (d_model)")
    parser.add_argument("--max-len", type=int, help="Maximum notes per bar")
    args, overrides = parser.parse_known_args(argv)

    override_list = [f"data.csv={args.data}", f"model.checkpoint={args.out}"]
    if args.epochs is not None:
        override_list.append(f"epochs={args.epochs}")
    if args.batch is not None:
        override_list.append(f"batch_size={args.batch}")
    if args.d_model is not None:
        override_list.append(f"d_model={args.d_model}")
    if args.max_len is not None:
        override_list.append(f"max_len={args.max_len}")
    override_list += overrides
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="duration_model.yaml", overrides=override_list)
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

# 推論サンプル: 推論用データ1バーパートでduration予測
import torch
import pandas as pd
from utilities.ml_duration import DurationTransformer

# 1. モデルの準備
model = DurationTransformer(d_model=256, max_len=16)
ckpt = torch.load("checkpoints/duration_earlystop.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 2. 推論用データの準備（例: bar=0のデータを使う）
df = pd.read_csv("data/duration_fixed.csv")
sample = df[df["bar"] == 0].sort_values("position")
pad = 16 - len(sample)

# 3. 入力テンソルの作成
feats = {
    "duration": torch.tensor(sample["duration"].tolist() + [0.0]*pad, dtype=torch.float32).unsqueeze(0),
    "velocity": torch.tensor(sample["velocity"].tolist() + [0.0]*pad, dtype=torch.float32).unsqueeze(0),
    "pitch_class": torch.tensor((sample["pitch"] % 12).tolist() + [0]*pad, dtype=torch.long).unsqueeze(0),
    "position_in_bar": torch.tensor(sample["position"].tolist() + [0]*pad, dtype=torch.long).unsqueeze(0),
}
mask = torch.zeros(1, 16, dtype=torch.bool)
mask[0, :len(sample)] = 1

# 4. 推論実行
with torch.no_grad():
    pred = model(feats, mask)

print("予測duration:", pred[0, :len(sample)].numpy())
