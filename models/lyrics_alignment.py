from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCH_FSDP", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_DYNAMO", "1")

import torch
from torch import nn
from types import SimpleNamespace

# Use the lightweight stand-in only when explicitly requested or when
# ``transformers`` is unavailable. This avoids hiding real import errors.
USE_DUMMY = os.getenv("COMPOSER_USE_DUMMY_TRANSFORMERS", "0") == "1"

_HF_Wav2Vec2Model = None
if not USE_DUMMY:
    try:  # pragma: no cover - optional dependency
        from transformers import Wav2Vec2Model as _HF_Wav2Vec2Model
    except Exception:  # pragma: no cover - any failure switches to dummy
        USE_DUMMY = True

if USE_DUMMY or _HF_Wav2Vec2Model is None:  # pragma: no cover - fallback path
    class _HF_Wav2Vec2Model(nn.Module):
        """Minimal stand-in for :class:`transformers.Wav2Vec2Model`.

        Only implements the methods used within this project. The forward pass
        preserves the input length although the real model downsamples.
        """

        def __init__(self, hidden_size: int = 32) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - simple stub
            return cls(hidden_size=kwargs.get("hidden_size", 32))

        def freeze_feature_extractor(self):  # pragma: no cover - no-op
            pass

        def gradient_checkpointing_enable(self):  # pragma: no cover - no-op
            pass

        def forward(self, audio: torch.Tensor, **kwargs):  # pragma: no cover - stubbed
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            b, t = audio.shape[:2]
            zeros = audio.new_zeros((b, t, self.config.hidden_size))
            return SimpleNamespace(last_hidden_state=zeros)

Wav2Vec2Model = _HF_Wav2Vec2Model


class LyricsAligner(nn.Module):
    """Simple CTC-based lyrics aligner."""

    def __init__(
        self,
        vocab_size: int,
        midi_feature_dim: int = 64,
        hidden_size: int = 256,
        dropout: float = 0.1,
        ctc_blank: str = "<blank>",
        *,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if USE_DUMMY:
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base", hidden_size=hidden_size
            )
        else:
            self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        enc_dim = self.wav2vec.config.hidden_size
        if freeze_encoder:
            self.wav2vec.freeze_feature_extractor()
        if gradient_checkpointing:
            self.wav2vec.gradient_checkpointing_enable()
        self.audio_proj = nn.Linear(enc_dim, hidden_size)
        self.audio_norm = nn.LayerNorm(hidden_size)
        self.midi_embed = nn.Embedding(512, midi_feature_dim)
        self.lstm = nn.LSTM(
            hidden_size + midi_feature_dim,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pre_fc_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size + 1)
        self.blank_id = vocab_size
        self.ctc_blank = ctc_blank

    def forward(self, audio: torch.Tensor, midi: torch.Tensor) -> torch.Tensor:
        """Return log probabilities for CTC."""
        feats = self.wav2vec(audio).last_hidden_state
        feats = self.audio_proj(feats)
        feats = self.audio_norm(feats)
        midi_emb = self.midi_embed(midi)
        if midi_emb.shape[1] < feats.shape[1]:
            pad = feats.shape[1] - midi_emb.shape[1]
            # pad time dimension; (B, T, C) -> (B, C, T)
            midi_emb = torch.nn.functional.pad(
                midi_emb.transpose(1, 2), (0, pad)
            ).transpose(1, 2)
        elif midi_emb.shape[1] > feats.shape[1]:
            midi_emb = midi_emb[:, : feats.shape[1]]
        x = torch.cat([feats, midi_emb], dim=-1)
        x, _ = self.lstm(x)
        x = self.pre_fc_norm(x)
        logits = self.fc(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        # CTCLoss expects (T, N, C), so transpose here for consistency
        return log_probs.transpose(0, 1)


__all__ = ["LyricsAligner"]
