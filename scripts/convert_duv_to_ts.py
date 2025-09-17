import argparse
from pathlib import Path
import sys
import torch

# Run as a script without installing the package:
# add repo root so "utilities" can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.ml_duration import DurationTransformer


def _get_tensor(sd: dict, *names: str) -> torch.Tensor:
    for name in names:
        tensor = sd.get(name)
        if tensor is not None:
            return tensor
    raise KeyError(f"Missing expected keys {names} in state dict")


def infer_hparams(sd: dict) -> dict:
    pitch_emb = _get_tensor(sd, "model.pitch_emb.weight", "pitch_emb.weight")
    pos_emb = _get_tensor(sd, "model.pos_emb.weight", "pos_emb.weight")
    d_model = int(pitch_emb.shape[1] * 4)
    max_len = int(pos_emb.shape[0])
    return {"d_model": d_model, "max_len": max_len}


def _strip_prefix(sd: dict, prefix: str = "model.") -> dict:
    if not prefix:
        return dict(sd)
    out = {}
    for key, value in sd.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--d-model", type=int, dest="d_model")
    parser.add_argument("--max-len", type=int, dest="max_len")
    args = parser.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit("ckpt is not a dict; expected training checkpoint")

    state_dict = None
    prefix = ""
    if "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
        prefix = "model."
    elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    if state_dict is None:
        raise SystemExit("ckpt に 'model' dict が見つかりません。学習時の ckpt を指定してください。")

    hparams = infer_hparams(state_dict)
    if args.d_model:
        hparams["d_model"] = args.d_model
    if args.max_len:
        hparams["max_len"] = args.max_len

    model = DurationTransformer(d_model=hparams["d_model"], max_len=hparams["max_len"])

    cleaned = _strip_prefix(state_dict, prefix)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("[load_state_dict] missing:", missing)
    print("[load_state_dict] unexpected:", unexpected)

    model.eval()
    in_dim = hparams["max_len"]
    feats = {
        "duration": torch.zeros(1, in_dim, dtype=torch.float32),
        "velocity": torch.zeros(1, in_dim, dtype=torch.float32),
        "pitch_class": torch.zeros(1, in_dim, dtype=torch.long),
        "position_in_bar": torch.zeros(1, in_dim, dtype=torch.long),
    }
    mask = torch.zeros(1, in_dim, dtype=torch.bool)
    traced = torch.jit.trace(model, (feats, mask), strict=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print("saved TorchScript →", args.out)


if __name__ == "__main__":
    main()
