"""Minimal Transformer encoder for phrase-level embeddings with legacy compat."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Create sinusoidal positional encodings of shape ``(max_len, d_model)``."""

    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PhraseTransformer(nn.Module):
    """Lightweight Transformer encoder used for phrase embedding."""

    def __init__(
        self,
        d_model: int,
        max_len: int,
        *,
        section_vocab_size: int = 0,
        mood_vocab_size: int = 0,
        vel_bucket_size: int = 0,
        dur_bucket_size: int = 0,
        use_bar_beat: bool = False,
        duv_mode: str = "reg",
        vel_bins: int = 0,
        dur_bins: int = 0,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_sinusoidal_posenc: bool = False,
        pitch_vocab_size: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.section_vocab_size = section_vocab_size
        self.mood_vocab_size = mood_vocab_size
        self.vel_bucket_size = vel_bucket_size
        self.dur_bucket_size = dur_bucket_size
        self.use_bar_beat = use_bar_beat
        self.duv_mode = duv_mode
        self.vel_bins = vel_bins
        self.dur_bins = dur_bins
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.use_sinusoidal_posenc = use_sinusoidal_posenc
        self.pitch_vocab_size = pitch_vocab_size

        self.token_embed = nn.Embedding(128, d_model, padding_idx=0)
        self.section_embed = (
            nn.Embedding(section_vocab_size, d_model) if section_vocab_size > 0 else None
        )
        self.mood_embed = (
            nn.Embedding(mood_vocab_size, d_model) if mood_vocab_size > 0 else None
        )
        self.vel_embed = (
            nn.Embedding(vel_bucket_size, d_model) if vel_bucket_size > 0 else None
        )
        self.dur_embed = (
            nn.Embedding(dur_bucket_size, d_model) if dur_bucket_size > 0 else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:  # pragma: no cover - older torch
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        if use_sinusoidal_posenc:
            pos_enc = _build_positional_encoding(max_len + 1, d_model)
        else:
            pos_enc = torch.zeros(max_len + 1, d_model, dtype=torch.float32)
        self.register_buffer("positional_encoding", pos_enc, persistent=False)

        self.in_norm = nn.Identity()
        self.feat_proj = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)
        self.head_boundary = nn.Linear(d_model, 1)
        self.head_vel_reg = (
            nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        )
        self.head_dur_reg = (
            nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        )
        self.head_vel_cls = (
            nn.Linear(d_model, vel_bins)
            if duv_mode in {"cls", "both"} and vel_bins > 0
            else None
        )
        self.head_dur_cls = (
            nn.Linear(d_model, dur_bins)
            if duv_mode in {"cls", "both"} and dur_bins > 0
            else None
        )
        self.head_pitch = nn.Linear(d_model, pitch_vocab_size)

        self.use_cls = True
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def _first_device(self, *tensors: torch.Tensor | None) -> torch.device:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def _infer_shape(self, batch: dict[str, Any]) -> tuple[int, int]:
        for key in (
            "section",
            "mood",
            "velocity_bucket",
            "vel_bucket",
            "duration_bucket",
            "dur_bucket",
            "tokens",
            "pitch",
            "pitch_class",
            "position",
            "velocity",
            "duration",
            "bar_phase",
            "beat_phase",
        ):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return value.size(0), value.size(1)
        length_hint = batch.get("lengths")
        if isinstance(length_hint, torch.Tensor) and length_hint.numel() > 0:
            max_len = max(int(length_hint.max().item()), 1)
            return int(length_hint.size(0)), max_len
        if isinstance(length_hint, (list, tuple)) and len(length_hint) > 0:
            return len(length_hint), max(int(max(length_hint)), 1)
        single_length = batch.get("length")
        if isinstance(single_length, torch.Tensor):
            return 1, max(int(single_length.item()), 1)
        if isinstance(single_length, int):
            return 1, max(single_length, 1)
        return 1, 1

    def _prepare_ids(
        self, ids: Optional[torch.Tensor], target_shape: torch.Size, vocab_size: int
    ) -> Optional[torch.Tensor]:
        if ids is None:
            return None
        ids = ids.to(torch.long)
        if ids.dim() == 0:
            ids = ids.view(1, 1)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(0) == 1 and target_shape[0] > 1:
            ids = ids.expand(target_shape[0], -1)
        if ids.shape != target_shape:
            if ids.size(0) != target_shape[0]:
                return None
            if ids.size(1) < target_shape[1]:
                pad = torch.zeros(
                    ids.size(0),
                    target_shape[1] - ids.size(1),
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, pad], dim=1)
            else:
                ids = ids[:, : target_shape[1]]
        return ids.clamp_(min=0, max=max(vocab_size - 1, 0))

    def _select_mask(
        self, batch: dict[str, Any], shape: torch.Size
    ) -> Optional[torch.Tensor]:
        for key in (
            "src_key_padding_mask",
            "key_padding_mask",
            "attention_mask",
            "padding_mask",
            "mask",
        ):
            mask = batch.get(key)
            if isinstance(mask, torch.Tensor):
                mask = mask.to(torch.bool)
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if mask.size(0) == 1 and shape[0] > 1:
                    mask = mask.expand(shape[0], -1)
                if mask.shape != shape:
                    if mask.size(0) != shape[0]:
                        continue
                    if mask.size(1) < shape[1]:
                        pad = torch.ones(
                            mask.size(0),
                            shape[1] - mask.size(1),
                            dtype=torch.bool,
                            device=mask.device,
                        )
                        mask = torch.cat([mask, pad], dim=1)
                    else:
                        mask = mask[:, : shape[1]]
                return mask
        return None

    def forward(
        self, *args: Any, **batch: Any
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode token sequences or legacy feature dicts."""

        if args and isinstance(args[0], dict):
            feats = args[0]
            mask = args[1] if len(args) > 1 else None
            return self._forward_legacy(feats, mask, **batch)

        return self._forward_tokens(*args, **batch)

    def _resolve_tokens(
        self,
        *args: Any,
        **batch: Any,
    ) -> tuple[torch.Tensor, torch.device]:
        tokens: Optional[torch.Tensor] = None
        if args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                tokens = first
        if tokens is None:
            tokens = batch.get("tokens")
        if tokens is None:
            tokens = batch.get("pitch")

        tensor_values = tuple(value for value in batch.values() if isinstance(value, torch.Tensor))

        if tokens is None or not isinstance(tokens, torch.Tensor):
            device = self._first_device(None, *tensor_values)
            batch_size, seq_len = self._infer_shape(batch)
            tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        else:
            device = self._first_device(tokens, *tensor_values)
            tokens = tokens.to(device=device, dtype=torch.long)
            vocab_mask = (tokens >= 0) & (tokens < self.token_embed.num_embeddings)
            tokens = torch.where(
                vocab_mask, tokens, torch.zeros_like(tokens, dtype=tokens.dtype)
            )
        return tokens, device

    def _forward_tokens(self, *args: Any, **batch: Any) -> torch.Tensor:
        tokens, device = self._resolve_tokens(*args, **batch)
        batch_size, seq_len = tokens.size(0), tokens.size(1)

        base = self.token_embed(tokens)

        if self.section_embed is not None:
            section_ids = self._prepare_ids(
                batch.get("section"), tokens.shape, self.section_vocab_size
            )
            if section_ids is not None:
                base = base + self.section_embed(section_ids.to(device))
        if self.mood_embed is not None:
            mood_ids = self._prepare_ids(
                batch.get("mood"), tokens.shape, self.mood_vocab_size
            )
            if mood_ids is not None:
                base = base + self.mood_embed(mood_ids.to(device))
        if self.vel_embed is not None:
            vel_ids = batch.get("vel_bucket")
            if vel_ids is None:
                vel_ids = batch.get("velocity_bucket")
            vel_ids_prepared = self._prepare_ids(vel_ids, tokens.shape, self.vel_bucket_size)
            if vel_ids_prepared is not None:
                base = base + self.vel_embed(vel_ids_prepared.to(device))
        if self.dur_embed is not None:
            dur_ids = batch.get("dur_bucket")
            if dur_ids is None:
                dur_ids = batch.get("duration_bucket")
            dur_ids_prepared = self._prepare_ids(dur_ids, tokens.shape, self.dur_bucket_size)
            if dur_ids_prepared is not None:
                base = base + self.dur_embed(dur_ids_prepared.to(device))

        pos_enc = self._ensure_posenc(seq_len + 1, device)
        base = base + pos_enc[:seq_len].unsqueeze(0)
        base = self.dropout(base)

        # NOTE: src_key_padding_mask expects True where tokens are padding.
        mask = self._select_mask(batch, tokens.shape)
        hidden = self.encoder(base, src_key_padding_mask=mask)
        return hidden

    def _forward_legacy(
        self,
        feats: dict[str, Any],
        mask: Optional[torch.Tensor] = None,
        **extra: Any,
    ) -> dict[str, torch.Tensor]:
        tokens = feats.get("pitch")
        if tokens is None:
            tokens = feats.get("pitch_class")

        padding_mask = None
        if isinstance(mask, torch.Tensor):
            padding_mask = mask
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask != 0
            padding_mask = ~padding_mask

        hidden = self._forward_tokens(
            tokens=tokens,
            section=feats.get("section"),
            mood=feats.get("mood"),
            vel_bucket=feats.get("vel_bucket"),
            velocity_bucket=feats.get("velocity_bucket"),
            dur_bucket=feats.get("dur_bucket"),
            duration_bucket=feats.get("duration_bucket"),
            src_key_padding_mask=padding_mask,
            lengths=feats.get("lengths"),
            length=feats.get("length"),
            **extra,
        )

        if hidden.size(1) == 0:
            hidden = torch.zeros(
                hidden.size(0),
                1,
                self.d_model,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        h = self.out_norm(hidden)

        out: dict[str, torch.Tensor] = {
            "boundary": self.head_boundary(h).squeeze(-1),
            "pitch_logits": self.head_pitch(h),
        }
        if self.head_vel_reg is not None:
            out["vel_reg"] = self.head_vel_reg(h).squeeze(-1)
        if self.head_dur_reg is not None:
            out["dur_reg"] = self.head_dur_reg(h).squeeze(-1)
        if self.head_vel_cls is not None:
            out["vel_cls"] = self.head_vel_cls(h)
        if self.head_dur_cls is not None:
            out["dur_cls"] = self.head_dur_cls(h)
        return out

    def _ensure_posenc(self, need_len: int, device: torch.device) -> torch.Tensor:
        if need_len > self.positional_encoding.size(0):
            if self.use_sinusoidal_posenc:
                return _build_positional_encoding(need_len, self.d_model).to(device)
            return torch.zeros(need_len, self.d_model, device=device)
        return self.positional_encoding[:need_len].to(device)

    def __repr__(self) -> str:
        return (
            "PhraseTransformer("
            f"d_model={self.d_model}, "
            f"layers={self.num_layers}, "
            f"nhead={self.nhead}, "
            f"sin_posenc={self.use_sinusoidal_posenc}, "
            f"sec={self.section_vocab_size}, "
            f"mood={self.mood_vocab_size}, "
            f"vel={self.vel_bucket_size}, "
            f"dur={self.dur_bucket_size}, "
            f"duv_mode='{self.duv_mode}'"
            ")"
        )

    def encode_seed(self, seq: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Return a minimal generation state stub."""

        return {"seq": list(seq or [])}

    def step(self, state: Optional[dict[str, Any]] = None) -> dict[str, torch.Tensor]:
        """Return dummy logits for generation workflows."""

        if state is None:
            state = self.encode_seed([])
        device = self.cls.device
        h = torch.zeros(1, 1, self.d_model, device=device)
        out = {
            "boundary": self.head_boundary(h).view(1),
            "pitch_logits": self.head_pitch(h).view(1, -1),
        }
        if self.head_vel_reg is not None:
            out["vel_reg"] = self.head_vel_reg(h).view(1)
        if self.head_dur_reg is not None:
            out["dur_reg"] = self.head_dur_reg(h).view(1)
        if self.head_vel_cls is not None:
            out["vel_cls"] = self.head_vel_cls(h).view(1, -1)
        if self.head_dur_cls is not None:
            out["dur_cls"] = self.head_dur_cls(h).view(1, -1)
        return out

    def update_state(
        self, state: dict[str, Any], event: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Append the provided event to the generation state."""

        if event is not None:
            state.setdefault("seq", []).append(event)
        return state


__all__ = ["PhraseTransformer"]
