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
    def __init__(self, core=None, input_dim: int = 3) -> None:
        if torch is None:
            self.input_dim = input_dim
            self._dummy = True
            self.core = core
        else:
            super().__init__()
            self._dummy = False

            if core is not None:
                # DUVモデルとして使用（PhraseTransformerコア付き）
                self.core = core
                self.input_dim = getattr(core, "d_model", 256)
                # DUV関連の属性を設定
                self.requires_duv_feats = True
                # 明示的にTrueに設定（実際のヘッドの存在は外部で確認済み）
                self.has_vel_head = True
                self.has_dur_head = True
                self.d_model = int(getattr(core, "d_model", 256))
                self.max_len = int(getattr(core, "max_len", 256))
                self.heads = {
                    "vel_reg": self.has_vel_head,
                    "dur_reg": self.has_dur_head,
                }
                self._duv_loader = "ckpt"
                # Velocity scaling configuration
                self.vel_scale = 127.0  # 0-1 output -> 1-127 MIDI velocity
                self.vel_offset_min = 1.0  # Minimum MIDI velocity
            else:
                # 従来のMLVelocityModelとして使用
                self.input_dim = input_dim
                self.core = None
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

    def forward(self, x, mask=None):
        if self.core is not None:
            # DUVモードでの推論
            if isinstance(x, dict):
                # DUV featureの場合
                if mask is None:
                    first = next(iter(x.values())) if x else None
                    if first is not None and hasattr(first, "shape"):
                        B, T = first.shape[:2]
                        if torch is not None:
                            mask = torch.ones((B, T), dtype=torch.bool, device=first.device)
                outputs = self.core(x, mask)
                if isinstance(outputs, dict):
                    vel = outputs.get("vel_reg")
                    dur = outputs.get("dur_reg")
                    return vel, dur
                if isinstance(outputs, tuple):
                    out0 = outputs[0] if len(outputs) > 0 else None
                    out1 = outputs[1] if len(outputs) > 1 else None
                    return out0, out1
                return outputs, None
            else:
                # レガシー形式の場合はエラー
                raise ValueError("DUV model requires dict input format")
        else:
            # 従来のMLVelocityModelでの推論
            if torch is None:
                raise RuntimeError("torch required")
            h = self.fc_in(x)
            h = self.encoder(h)
            return self.fc_out(h).squeeze(-1)

    @classmethod
    def _infer_hparams_from_sd(cls, sd: dict) -> dict:
        """
        ckpt の state_dict 形状からハイパラを推定:
          - d_model: encoder.layers.0.linear1.weight の形状 [d_ff, d_model] の第2軸
          - d_ff   : 同行列の第1軸
          - max_len: pos_emb.weight の第1軸（あれば）
          - 追加埋め込みの有無: *_bucket_emb.weight などのキー有無
        """
        d_model = None
        d_ff = None
        for k, v in sd.items():
            if k.endswith("encoder.layers.0.linear1.weight"):
                d_ff, d_model = int(v.shape[0]), int(v.shape[1])
                break
        if d_model is None:
            # フォールバック：in_proj_weight から埋め込み次元を推定
            for k, v in sd.items():
                if k.endswith("self_attn.in_proj_weight"):
                    d_model = int(v.shape[1])
                    d_ff = int(max(512, 4 * d_model))  # 仮に4*d_model
                    break
        max_len = 256
        if "pos_emb.weight" in sd:
            max_len = int(sd["pos_emb.weight"].shape[0])
        # オプション埋め込みの有無
        vel_in_sd = "vel_bucket_emb.weight" in sd or "model.vel_bucket_emb.weight" in sd
        dur_in_sd = "dur_bucket_emb.weight" in sd or "model.dur_bucket_emb.weight" in sd
        sec_in_sd = "section_emb.weight" in sd or "model.section_emb.weight" in sd
        mood_in_sd = "mood_emb.weight" in sd or "model.mood_emb.weight" in sd

        # DUVヘッドの有無をチェック
        vel_head_in_sd = "head_vel_reg.weight" in sd or "model.head_vel_reg.weight" in sd
        dur_head_in_sd = "head_dur_reg.weight" in sd or "model.head_dur_reg.weight" in sd

        extras = {
            "vel_bucket_emb": vel_in_sd,
            "dur_bucket_emb": dur_in_sd,
            "section_emb": sec_in_sd,
            "mood_emb": mood_in_sd,
            "has_vel_head": vel_head_in_sd,
            "has_dur_head": dur_head_in_sd,
        }
        return {
            "d_model": d_model or 256,
            "d_ff": d_ff or 512,
            "max_len": max_len,
            "extras": extras,
        }

    @classmethod
    def load(cls, path: str) -> "MLVelocityModel":
        if torch is None:
            raise RuntimeError("torch required")

        p = Path(path)
        if p.suffix == ".ts":
            return torch.jit.load(p, map_location="cpu").eval()

        obj = torch.load(p, map_location="cpu")
        # state_dict の場所を素直に解決
        sd = obj
        if isinstance(obj, dict) and "model" in obj:
            maybe = obj["model"]
            sd = maybe.get("state_dict", maybe) if isinstance(maybe, dict) else maybe
        if isinstance(sd, dict) and any(k.startswith("model.") for k in sd.keys()):
            # "model." プレフィクスを除去
            sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}

        # 形状からハイパラを推定し、その寸法で PhraseTransformer を構築
        hp = cls._infer_hparams_from_sd(sd if isinstance(sd, dict) else {})

        if PhraseTransformer is None:
            raise RuntimeError("PhraseTransformer unavailable")

        core = PhraseTransformer(
            d_model=hp["d_model"],
            max_len=hp["max_len"],
            ff_dim=hp["d_ff"],
            section_emb=hp["extras"]["section_emb"],
            mood_emb=hp["extras"]["mood_emb"],
            vel_bucket_emb=hp["extras"]["vel_bucket_emb"],
            dur_bucket_emb=hp["extras"]["dur_bucket_emb"],
        )
        missing, unexpected = core.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logging.warning({"missing_keys": missing, "unexpected_keys": unexpected})

        # Model configuration summary
        model_info = {
            "d_model": hp["d_model"],
            "max_len": hp["max_len"],
            "d_ff": hp["d_ff"],
            "vel_scale": 127.0,
            "extras": hp["extras"],
        }
        logging.info(f"MLVelocityModel loaded: {model_info}")

        model = cls(core)
        return model.eval()

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
