import argparse, json, importlib, os, torch
from torch import nn
from typing import Any, Dict

SUFFIXES = (".ts", ".pt", ".pth", ".ckpt", ".zip")


def try_import(cp: str):
    m, c = cp.rsplit(".", 1)
    return getattr(__import__(m, fromlist=[c]), c)


def iter_paths(obj: Any):
    """再帰で文字列パス候補を拾う"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                yield from iter_paths(v)
            elif isinstance(v, str) and v.endswith(SUFFIXES):
                yield v


def resolve(path: str, ckpt_dir: str):
    """存在しなければ ckpt ディレクトリ起点で補完"""
    cands = [path, os.path.join(ckpt_dir, path), os.path.join(ckpt_dir, os.path.basename(path))]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def find_module(obj: Any):
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "net", "module"):
            v = obj.get(k)
            if isinstance(v, nn.Module):
                return v
    return None


def maybe_build_from_state_dict(
    pack: Dict[str, Any], class_path: str | None, init_args: Dict[str, Any] | None
):
    sd = None
    if isinstance(pack, dict):
        if "state_dict" in pack and isinstance(pack["state_dict"], dict):
            sd = pack["state_dict"]
        else:
            for key in ("model", "weights", "params"):
                v = pack.get(key)
                if isinstance(v, dict):
                    if "state_dict" in v and isinstance(v["state_dict"], dict):
                        sd = v["state_dict"]
                        break
                    # Check if v itself looks like a state_dict (has tensor values)
                    elif all(isinstance(val, torch.Tensor) for val in list(v.values())[:5]):
                        sd = v
                        break
    if sd is None:
        return None
    hp = pack.get("hyper_parameters") or pack.get("hparams") or {}
    cands = [
        class_path,
        hp.get("class_path"),
        hp.get("target"),
        hp.get("model_class"),
        hp.get("cls"),
        hp.get("import_path"),
    ]
    cands = [c for c in cands if isinstance(c, str)]
    last = None
    for cp in cands:
        try:
            cls = try_import(cp)
            kwargs = {}
            for key in ("init_args", "model_kwargs", "kwargs", "hparams", "config"):
                if key in hp and isinstance(hp[key], dict):
                    kwargs.update(hp[key])
            if isinstance(init_args, dict):
                kwargs.update(init_args)
            m = cls(**kwargs) if kwargs else cls()
            try:
                m.load_state_dict(sd, strict=True)
            except Exception:
                from collections import OrderedDict

                od = OrderedDict()
                for k, v in sd.items():
                    for pref in ("model.", "net.", "module."):
                        if k.startswith(pref):
                            k = k[len(pref) :]
                            break
                    od[k] = v
                m.load_state_dict(od, strict=False)
            return m
        except Exception as e:
            last = e
    raise SystemExit(
        f"state_dictは見つかったがクラス復元に失敗。--class/--init を指定してください。最後のエラー: {last}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class", dest="class_path", default=None)
    ap.add_argument("--init", dest="init_json", default=None)
    args = ap.parse_args()

    init_args = json.loads(args.init_json) if args.init_json else None
    ckpt_path = os.path.abspath(args.ckpt)
    ckpt_dir = os.path.dirname(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu")

    # 1) どこかに Module がそのまま入っていないか
    m = find_module(obj)
    if m is None and isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        m = find_module(obj["model"])

    # 2) TorchScript パスが埋め込まれていないか（再帰探索＋相対補完）
    if m is None:
        for pth in iter_paths(obj):
            rp = resolve(pth, ckpt_dir)
            if not rp:
                continue
            try:
                m = torch.jit.load(rp, map_location="cpu").eval()
                print(f"found TorchScript at: {rp}")
                break
            except Exception:
                pass

    # 3) state_dict から復元できないか
    if m is None:
        pack = obj.get("model") if isinstance(obj, dict) else obj
        if pack is None:
            pack = obj
        m = maybe_build_from_state_dict(pack, args.class_path, init_args)

    if m is None:
        raise SystemExit("モデルの復元に失敗（Module/state_dict/TS-path すべて不発）")

    m.eval()
    try:
        ts = torch.jit.script(m)
    except Exception as e:
        raise SystemExit(f"torch.jit.script に失敗: {e}（trace 切替はダミー入力形状が必要）")
    ts.save(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
