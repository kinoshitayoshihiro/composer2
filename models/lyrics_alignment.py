from __future__ import annotations

import torch
from torch import nn
from transformers import Wav2Vec2Model


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
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_encoder:
            self.wav2vec.freeze_feature_extractor()
        if gradient_checkpointing:
            self.wav2vec.gradient_checkpointing_enable()
        self.audio_proj = nn.Linear(self.wav2vec.config.hidden_size, hidden_size)
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
        x = torch.cat([feats, midi_emb], dim=-1)
        x, _ = self.lstm(x)
        x = self.pre_fc_norm(x)
        logits = self.fc(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)


__all__ = ["LyricsAligner"]
