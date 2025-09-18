from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = object  # type: ignore

try:  # pragma: no cover - optional during documentation builds
    from models.phrase_transformer import PhraseTransformer  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for environments without model deps
    PhraseTransformer = None  # type: ignore

_script_cache: dict[str, Any] = {}


def _strip_prefix(sd: Dict[str, Any], prefix: str = "model.") -> Dict[str, Any]:
    if not prefix:
        return dict(sd)
    out: Dict[str, Any] = {}
    for key, value in sd.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def _lru_script(model: nn.Module, key: str) -> Any:
    if torch is None:
        raise RuntimeError("torch required")
    cached = _script_cache.get(key)
    if cached is None:
        cached = torch.jit.script(model.eval())
        _script_cache[key] = cached
        if len(_script_cache) > 16:
            _script_cache.pop(next(iter(_script_cache)))
    return cached


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1) * diff))


def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred, target)
    q = quantile_loss(pred, target, 0.1) + quantile_loss(pred, target, 0.9)
    return huber + q


class ModelV1_LSTM(nn.Module if torch is not None else object):
    def __init__(self, input_dim: int = 3, hidden: int = 64) -> None:
        if torch is None:
            raise RuntimeError("torch required")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        nn.init.constant_(self.fc.bias, 64.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


class MLVelocityModel(nn.Module if torch is not None else object):
    def __init__(self, input_dim: int = 3) -> None:
        self.input_dim = input_dim
        if torch is None:
            self._dummy = True
        else:
            super().__init__()
            self._dummy = False
            self.fc_in = nn.Linear(input_dim, 256)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
            self.fc_out = nn.Linear(256, 1)
            nn.init.constant_(self.fc_out.bias, 64.0)
            # Ensure that an untrained model predicts the default velocity
            # (64) by zeroing the output weights. This keeps the unit tests
            # deterministic and provides a sensible starting point for
            # fine-tuning.
            nn.init.zeros_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = self.encoder(h)
        return self.fc_out(h).squeeze(-1)

    @staticmethod
    def load(path: str) -> nn.Module:
        """Load a velocity model checkpoint as an ``nn.Module`` ready for eval.

        TorchScript files are loaded via :func:`torch.jit.load`; other
        checkpoints use :func:`torch.load` and must still yield an ``nn.Module``.
        """
        if torch is None:
            raise RuntimeError("torch required")

        p = str(path)
        if p.endswith((".ts", ".torchscript")):
            model = torch.jit.load(p, map_location="cpu").eval()
            try:
                setattr(model, "_duv_loader", "ts")
            except Exception:  # pragma: no cover - best effort on ScriptModule
                pass
            return model

        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, nn.Module):
            return obj.eval()

        if isinstance(obj, dict):
            for key in ("model", "net", "module"):
                mod = obj.get(key)
                if isinstance(mod, nn.Module):
                    return mod.eval()
            model_state = obj.get("model")
            if isinstance(model_state, dict):
                if PhraseTransformer is None:
                    raise RuntimeError("PhraseTransformer unavailable; install model dependencies")
                meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}

                def _get_int(name: str, fallback: int) -> int:
                    try:
                        return int(meta.get(name, fallback))
                    except Exception:
                        return fallback

                state_dict = _strip_prefix(model_state, "model.")
                pitch_emb = state_dict.get("pitch_emb.weight")
                pos_emb = state_dict.get("pos_emb.weight")
                if pitch_emb is None or pos_emb is None:
                    raise ValueError("state_dict missing pitch_emb/pos_emb tensors")
                d_model = _get_int("d_model", int(pitch_emb.shape[1] * 4))
                max_len = _get_int("max_len", int(pos_emb.shape[0]))

                def _rows(key: str) -> int:
                    tensor = state_dict.get(key)
                    try:
                        return int(tensor.shape[0]) if tensor is not None else 0
                    except Exception:
                        return 0

                def _has_param(key: str) -> bool:
                    tensor = state_dict.get(key)
                    if tensor is None:
                        return False
                    try:
                        return int(tensor.numel()) > 0  # type: ignore[call-arg]
                    except Exception:
                        return False

                section_vocab_size = _get_int("section_vocab_size", _rows("section_emb.weight"))
                mood_vocab_size = _get_int("mood_vocab_size", _rows("mood_emb.weight"))
                vel_bucket_size = _get_int("vel_bucket_size", _rows("vel_bucket_emb.weight"))
                dur_bucket_size = _get_int("dur_bucket_size", _rows("dur_bucket_emb.weight"))
                vel_bins = _get_int("vel_bins", _rows("head_vel_cls.weight"))
                dur_bins = _get_int("dur_bins", _rows("head_dur_cls.weight"))

                has_reg_head = _has_param("head_vel_reg.weight") or _has_param(
                    "head_dur_reg.weight"
                )
                has_cls_head = vel_bins > 0 or dur_bins > 0
                duv_mode_meta = str(meta.get("duv_mode", "")).strip()
                if duv_mode_meta in {"reg", "cls", "both"}:
                    duv_mode = duv_mode_meta
                elif has_reg_head and has_cls_head:
                    duv_mode = "both"
                elif has_cls_head:
                    duv_mode = "cls"
                else:
                    duv_mode = "reg"

                use_bar_beat = bool(
                    meta.get("use_bar_beat")
                    if isinstance(meta.get("use_bar_beat"), bool)
                    else _has_param("barpos_proj.weight") and _has_param("beatpos_proj.weight")
                )
                core = PhraseTransformer(
                    d_model=d_model,
                    max_len=max_len,
                    section_vocab_size=section_vocab_size,
                    mood_vocab_size=mood_vocab_size,
                    vel_bucket_size=vel_bucket_size,
                    dur_bucket_size=dur_bucket_size,
                    vel_bins=vel_bins,
                    dur_bins=dur_bins,
                    use_bar_beat=use_bar_beat,
                    duv_mode=duv_mode,
                )
                missing, unexpected = core.load_state_dict(state_dict, strict=False)
                miss_list = sorted(missing) if isinstance(missing, (list, tuple)) else list(missing)
                unexp_list = (
                    sorted(unexpected) if isinstance(unexpected, (list, tuple)) else list(unexpected)
                )
                if miss_list or unexp_list:
                    logging.warning(
                        {"missing_keys": miss_list, "unexpected_keys": unexp_list}
                    )

                class PhraseDUVModule(nn.Module):
                    def __init__(self, inner: nn.Module, *, meta: dict[str, Any]) -> None:
                        super().__init__()
                        self.core = inner
                        self.meta = dict(meta)
                        self.requires_duv_feats = True
                        self.has_vel_head = bool(getattr(inner, "head_vel_reg", None))
                        self.has_dur_head = bool(getattr(inner, "head_dur_reg", None))
                        self.d_model = int(getattr(inner, "d_model", d_model))
                        self.max_len = int(getattr(inner, "max_len", max_len))
                        self.heads = {
                            "vel_reg": self.has_vel_head,
                            "dur_reg": self.has_dur_head,
                        }
                        self._duv_loader = "ckpt"

                    def forward(
                        self,
                        feats: dict[str, torch.Tensor],
                        mask: torch.Tensor | None = None,
                    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                        if mask is None:
                            raise ValueError("mask tensor required for DUV transformer")
                        outputs = self.core(feats, mask)
                        if isinstance(outputs, dict):
                            vel = outputs.get("vel_reg")
                            dur = outputs.get("dur_reg")
                            return vel, dur
                        if isinstance(outputs, tuple):
                            out0 = outputs[0] if len(outputs) > 0 else None
                            out1 = outputs[1] if len(outputs) > 1 else None
                            return out0, out1
                        return outputs, None

                return PhraseDUVModule(core.eval(), meta=meta).eval()
            if "state_dict" in obj:
                raise RuntimeError(
                    "DUV ckpt is state_dict-only. Export to TorchScript or restore model class to load state_dict."
                )

        raise RuntimeError(f"Unsupported ckpt type: {type(obj).__name__}")

    def predict(self, ctx, *, cache_key: str | None = None) -> "np.ndarray":
        if torch is None or getattr(self, "_dummy", False):
            import numpy as np
            return np.full((ctx.shape[0],), 64.0, dtype=np.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
        model: Any = self
        if cache_key is not None:
            model = _lru_script(self, cache_key)
            model = model.to(device)
        else:
            model = self.to(device).eval()
        with torch.no_grad():
            out = model(x).squeeze(0).cpu().clamp(0, 127).to(torch.float32)
        return out.numpy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Velocity model inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())
    model = MLVelocityModel.load(args.model)
    result = model.predict(data, cache_key=args.model)
    print(json.dumps(result.tolist()))


if __name__ == "__main__":
    main()

__all__ = [
    "MLVelocityModel",
    "ModelV1_LSTM",
    "velocity_loss",
]
