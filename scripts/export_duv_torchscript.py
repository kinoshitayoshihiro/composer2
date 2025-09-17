import argparse, json, importlib, torch, os
from torch import nn
from typing import Any, Dict

def try_import(class_path: str):
    mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)

def find_module_in_obj(obj: Any):
    """obj から nn.Module を探して返す（なければ None）"""
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        # ありがちなキー
        for k in ("model", "net", "module"):
            v = obj.get(k)
            if isinstance(v, nn.Module):
                return v
        # TorchScript のパスが書いてあるならそれを使う
        for k in ("ts_path", "torchscript", "jit_path"):
            p = obj.get(k)
            if isinstance(p, str) and os.path.exists(p):
                return torch.jit.load(p)
    return None

def maybe_build_from_state_dict(pack: Dict[str, Any], meta: Dict[str, Any], class_path: str|None, init_args: Dict[str,Any]|None):
    """state_dict からモデルを復元（クラスが必要）"""
    # state_dict を入れ子も含めて探索
    sd = None
    if "state_dict" in pack and isinstance(pack["state_dict"], dict):
        sd = pack["state_dict"]
    else:
        # よくある入れ子 'model' / 'weights' / 'params'
        for key in ("model", "weights", "params"):
            v = pack.get(key)
            if isinstance(v, dict) and "state_dict" in v and isinstance(v["state_dict"], dict):
                sd = v["state_dict"]; break
    if sd is None:
        return None  # 復元対象がない

    # クラスパス候補を収集
    hp = pack.get("hyper_parameters") or pack.get("hparams") or meta or {}
    cands = [class_path,
             hp.get("class_path"), hp.get("target"), hp.get("model_class"),
             hp.get("cls"), hp.get("import_path")]
    cands = [c for c in cands if isinstance(c, str)]

    last_err = None
    for cp in cands:
        try:
            cls = try_import(cp)
            kwargs = {}
            for key in ("init_args","model_kwargs","kwargs","hparams","config"):
                if key in hp and isinstance(hp[key], dict):
                    kwargs.update(hp[key])
            if isinstance(init_args, dict):
                kwargs.update(init_args)
            m = cls(**kwargs) if kwargs else cls()
            try:
                m.load_state_dict(sd, strict=True)
            except Exception:
                # 'model.' / 'net.' / 'module.' をはがして再試行
                from collections import OrderedDict
                od = OrderedDict()
                for k, v in sd.items():
                    for pref in ("model.", "net.", "module."):
                        if k.startswith(pref):
                            k = k[len(pref):]
                            break
                    od[k] = v
                m.load_state_dict(od, strict=False)
            return m
        except Exception as e:
            last_err = e

    raise SystemExit(
        "state_dict は見つかりましたが、モデルクラスの復元に失敗しました。\n"
        "  対処: --class でクラスの import パスを与えるか、--init で初期化引数を渡してください。\n"
        f"  最後のエラー: {last_err}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class", dest="class_path", default=None,
                    help="state_dict-only の場合のモデルクラス import パス")
    ap.add_argument("--init", dest="init_json", default=None,
                    help="上記クラスの初期化引数（JSON 文字列）")
    args = ap.parse_args()

    init_args = json.loads(args.init_json) if args.init_json else None

    obj = torch.load(args.ckpt, map_location="cpu")
    meta = {}
    if isinstance(obj, dict):
        meta = obj.get("meta") or obj.get("hyper_parameters") or obj.get("hparams") or {}

    # 1) どこかに Module がそのまま入っていないか探す
    m = find_module_in_obj(obj)
    if m is None and isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        m = find_module_in_obj(obj["model"])

    # 2) Module が見つからない → state_dict から復元を試す
    if m is None:
        pack = obj["model"] if isinstance(obj, dict) and isinstance(obj.get("model"), dict) else obj
        m = maybe_build_from_state_dict(pack, meta, args.class_path, init_args)

    if m is None:
        raise SystemExit("モデルの復元に失敗しました（Module/state_dict 共に見つからず）")

    # 3) TorchScript 化（script が難しければ trace に切替を検討）
    m.eval()
    try:
        ts = torch.jit.script(m)
    except Exception as e:
        raise SystemExit(
            f"torch.jit.script に失敗しました: {e}\n"
            "必要なら trace に変更可能です（ダミー入力テンソル形状が必要）。"
        )
    ts.save(args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
