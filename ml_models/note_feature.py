import torch
from torch import nn


class NoteFeatureEmbedder(nn.Module):
    """Embed categorical note features and pass through scalars."""

    def __init__(
        self,
        pitch_dim: int = 16,
        bucket_dim: int = 4,
        pedal_dim: int = 2,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pitch = nn.Embedding(128, pitch_dim)
        self.bucket = nn.Embedding(8, bucket_dim)
        self.pedal = nn.Embedding(3, pedal_dim)
        self.norm = (
            nn.LayerNorm(pitch_dim + bucket_dim + pedal_dim + 2) if layer_norm else None
        )

    def forward(
        self,
        pitch: torch.Tensor,
        bucket: torch.Tensor,
        pedal: torch.Tensor,
        velocity: torch.Tensor,
        qlen: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat(
            [
                self.pitch(pitch),
                self.bucket(bucket),
                self.pedal(pedal),
                velocity.unsqueeze(-1),
                qlen.unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.norm is not None:
            feat = self.norm(feat)
        return feat
