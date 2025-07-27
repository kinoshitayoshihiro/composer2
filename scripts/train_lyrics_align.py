from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig
except Exception:  # pragma: no cover - optional
    hydra = None  # type: ignore
    compose = initialize = None  # type: ignore
    DictConfig = object  # type: ignore

import numpy as np
import pretty_midi
import soundfile as sf

from models.lyrics_alignment import LyricsAligner


class AlignDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_rate: int,
        hop: int,
        vocab: list[str] | None = None,
        blank: str = "<blank>",
    ) -> None:
        self.items = []
        for wav in sorted(root.glob("*.wav")):
            mid = wav.with_suffix(".mid")
            phn = wav.with_suffix(".phn")
            if mid.exists() and phn.exists():
                self.items.append((wav, mid, phn))
        self.sr = sample_rate
        self.hop = hop
        phonemes = set()
        for _, _, p in self.items:
            for line in Path(p).read_text().splitlines():
                _, ph = line.split()
                phonemes.add(ph)
        self.vocab = sorted(vocab or phonemes)
        self.blank = blank
        self.ph2id = {p: i for i, p in enumerate(self.vocab)}
        self.blank_id = len(self.vocab)

    def __len__(self) -> int:
        return len(self.items)

    def _midi_feat(self, mid_path: Path, n_frames: int) -> torch.Tensor:
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        arr = torch.zeros(n_frames, dtype=torch.long)
        for note in pm.instruments[0].notes:
            idx = int(note.start * 1000 / self.hop)
            if idx < n_frames:
                val = int(note.start * 1000)
                arr[idx] = min(val, 511)
        return arr

    def __getitem__(self, idx: int):
        wav, mid, phn = self.items[idx]
        audio, _ = sf.read(wav, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio.shape) == 0:
            audio = np.array([0.0], dtype=np.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        n_frames = int(round(len(audio) / self.sr / (self.hop / 1000)))
        midi = self._midi_feat(mid, n_frames)
        times: list[float] = []
        labels: list[int] = []
        for line in Path(phn).read_text().splitlines():
            t, ph = line.split()
            times.append(float(t) * 1000)
            labels.append(self.ph2id[ph])
        return (
            audio,
            midi,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(times, dtype=torch.float32),
        )


def collate(batch):
    audios, midis, labels, times = zip(*batch)
    max_a = max(a.shape[0] for a in audios)
    max_m = max(m.shape[0] for m in midis)
    audio_pad = torch.stack(
        [nn.functional.pad(a, (0, max_a - a.shape[0])) for a in audios]
    )
    midi_pad = torch.stack(
        [nn.functional.pad(m, (0, max_m - len(m)), value=0) for m in midis]
    )
    input_lens = torch.tensor([len(m) for m in midis], dtype=torch.long)
    target = torch.cat(labels)
    target_lens = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    times_pad = nn.utils.rnn.pad_sequence(times, batch_first=True)
    return audio_pad, midi_pad, input_lens, target, target_lens, times_pad


def decode(seq: torch.Tensor, blank_id: int, hop: int) -> list[float]:
    out: list[float] = []
    prev = blank_id
    for i, p in enumerate(seq):
        pi = int(p)
        if pi != prev and pi != blank_id:
            out.append(i * hop)
        prev = pi
    return out


def evaluate(
    model: nn.Module, loader: DataLoader, crit: nn.CTCLoss, hop: int, blank_id: int
):
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for audio, midi, input_len, target, target_len, times in loader:
            logp = model(audio, midi)
            loss = crit(logp, target, input_len, target_len)
            total_loss += float(loss)
            preds = logp.argmax(-1).transpose(0, 1)  # (T,N)->(N,T)
            for b in range(preds.size(0)):
                seq = preds[b, : input_len[b]]
                pred_times = decode(seq, blank_id, hop)
                true_times = times[b, : target_len[b]].tolist()
                if true_times:
                    mae = sum(
                        abs(pt - tt) for pt, tt in zip(pred_times, true_times)
                    ) / len(true_times)
                else:
                    mae = 0.0
                total_mae += mae
                count += 1
    return total_loss / max(count, 1), total_mae / max(count, 1)


class AlignModule(pl.LightningModule):
    def __init__(self, vocab: list[str], cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = LyricsAligner(
            len(vocab),
            cfg.midi_feature_dim,
            cfg.hidden_size,
            cfg.dropout,
            cfg.ctc_blank,
            freeze_encoder=cfg.freeze_encoder,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        self.crit = nn.CTCLoss(blank=len(vocab), zero_infinity=True)
        self.vocab = vocab
        self.hop = cfg.hop_length_ms
        self.lr = cfg.lr

    def forward(self, audio: torch.Tensor, midi: torch.Tensor) -> torch.Tensor:
        return self.model(audio, midi)

    def training_step(self, batch, batch_idx):
        audio, midi, input_len, target, target_len, _ = batch
        logp = self(audio, midi)
        loss = self.crit(logp, target, input_len, target_len)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, midi, input_len, target, target_len, times = batch
        logp = self(audio, midi)
        loss = self.crit(logp, target, input_len, target_len)
        preds = logp.argmax(-1).transpose(0, 1)  # (T,N)->(N,T)
        mae = 0.0
        for b in range(preds.size(0)):
            seq = preds[b, : input_len[b]]
            pred_times = decode(seq, len(self.vocab), self.hop)
            true_times = times[b, : target_len[b]].tolist()
            if true_times:
                mae += sum(
                    abs(pt - tt) for pt, tt in zip(pred_times, true_times)
                ) / len(true_times)
        mae = mae / preds.size(0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


def run(cfg: DictConfig) -> float:
    train_ds = AlignDataset(
        Path(cfg.train_dir), cfg.sample_rate, cfg.hop_length_ms, blank=cfg.ctc_blank
    )
    val_ds = AlignDataset(
        Path(cfg.val_dir),
        cfg.sample_rate,
        cfg.hop_length_ms,
        vocab=train_ds.vocab,
        blank=cfg.ctc_blank,
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)

    module = AlignModule(train_ds.vocab, cfg)
    logger = TensorBoardLogger(save_dir=cfg.trainer.default_root_dir)
    es = EarlyStopping(
        monitor="val_mae", patience=2, mode="min", stopping_threshold=50.0
    )
    trainer = pl.Trainer(max_epochs=cfg.epochs, logger=logger, callbacks=[es])
    trainer.fit(module, train_dl, val_dl)

    best_mae = es.best_score.item() if es.best_score is not None else float("inf")
    if best_mae < 50.0:
        state = {
            "model": module.model.state_dict(),
            "vocab": train_ds.vocab,
            "config": {
                "sample_rate": cfg.sample_rate,
                "hop_length_ms": cfg.hop_length_ms,
                "midi_feature_dim": cfg.midi_feature_dim,
                "hidden_size": cfg.hidden_size,
                "dropout": cfg.dropout,
                "ctc_blank": cfg.ctc_blank,
                "freeze_encoder": cfg.freeze_encoder,
                "gradient_checkpointing": cfg.gradient_checkpointing,
            },
        }
        Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cfg.checkpoint)

    return best_mae


@hydra.main(
    config_path="../configs", config_name="lyrics_align.yaml", version_base="1.3"
)
def hydra_main(cfg: DictConfig) -> float:  # pragma: no cover - entry
    return run(cfg)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    if hydra is None:
        print("hydra-core required")
        return 1
    parser = argparse.ArgumentParser(prog="train_lyrics_align.py")
    parser.add_argument("--train_dir", type=Path)
    parser.add_argument("--val_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("model.ckpt"))
    parser.add_argument("--epochs", type=int)
    args, overrides = parser.parse_known_args(argv)
    override = [
        f"train_dir={args.train_dir}",
        f"val_dir={args.val_dir}",
        f"checkpoint={args.out}",
    ]
    if args.epochs is not None:
        override.append(f"epochs={args.epochs}")
    override += overrides
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="lyrics_align.yaml", overrides=override)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
