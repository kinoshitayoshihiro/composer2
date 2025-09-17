import argparse, json, importlib, torch
from torch import nn

def try_import(class_path: str):
    mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)

def build_from_ckpt_obj(obj, class_path=None, init_args=None):
    """
    ckpt から nn.Module を復元:
      1) そのまま Module → 返す
      2) dict 内の 'model'/'net'/'module' に Module → 返す
      3) state_dict のみ → class_path と init_args でインスタンス化して load
         （class_path は --class or ckpt に埋まっている 'hyper_parameters' などから推測）
    """
    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, dict):
        for k in ("model", "net", "module"):
            m = obj.get(k)
            if isinstance(m, nn.Module):
                return m

        if "state_dict" in obj:
            hp = obj.get("hyper_parameters") or obj.get("hparams") or {}
            cand = [
                class_path,
                hp.get("class_path"),
                hp.get("target"),
                hp.get("model_class"),
            ]
            cand = [c for c in cand if isinstance(c, str)]
            last_err = None
            for cp in cand:
                try:
                    cls = try_import(cp)
                    kwargs = {}
                    # hydra/lightning 由来の入れ子にありがちな候補
                    for key in ("init_args", "model_kwargs", "kwargs", "hparams"):
                        if key in hp and isinstance(hp[key], dict):
                            kwargs.update(hp[key])
                    if isinstance(init_args, dict):
                        kwargs.update(init_args)
                    m = cls(**kwargs) if kwargs else cls()
                    # state_dict のキーに prefix が付いている場合に備えて緩く対応
                    sd = obj["state_dict"]
                    try:
                        m.load_state_dict(sd, strict=True)
                    except Exception:
                        # 'model.' や 'net.' などを剥がして再挑戦
                        def strip_prefix(d):
                            from collections import OrderedDict
                            od = OrderedDict()
                            for k, v in d.items():
                                if k.startswith("model."):
                                    od[k[len("model."):]] = v
                                elif k.startswith("net."):
                                    od[k[len("net."):]] = v
                                elif k.startswith("module."):
                                    od[k[len("module."):]] = v
                                else:
                                    od[k] = v
                            return od
                        m.load_state_dict(strip_prefix(sd), strict=False)
                    return m
                except Exception as e:
                    last_err = e
            raise SystemExit(
                "state_dict 形式の ckpt ですがモデルクラスを復元できませんでした。\n"
                "  対処: --class でクラスの import パスを与えるか、--init で初期化引数を渡してください。\n"
                f"  最後のエラー: {last_err}"
            )

    raise SystemExit(f"未知の ckpt 形式です: {type(obj)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class", dest="class_path", default=None,
                    help="state_dict-only の場合に使うモデルクラスの import パス")
    ap.add_argument("--init", dest="init_json", default=None,
                    help="上記クラスの初期化引数（JSON 文字列）")
    args = ap.parse_args()

    init_args = json.loads(args.init_json) if args.init_json else None
    obj = torch.load(args.ckpt, map_location="cpu")
    model = build_from_ckpt_obj(obj, class_path=args.class_path, init_args=init_args)
    model.eval()
    try:
        ts = torch.jit.script(model)
    except Exception:
        # script が難しい場合は trace（入力形状が固定ならば）
        raise SystemExit(
            "torch.jit.script に失敗しました。必要なら trace に変更しますが、"
            "ダミー入力テンソル形状が必要です。必要なら連絡してください。"
        )
    ts.save(args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
