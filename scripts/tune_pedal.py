from __future__ import annotations

"""Sweep post-processing params for sustain-pedal probabilities.

This runs a grid search over thresholding/smoothing parameters using a cached
probability array produced by ``scripts.predict_pedal --dump-prob``.
"""

import argparse
import csv
import json
import itertools
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from .predict_pedal import compute_metrics, postprocess_on, write_cc64  # type: ignore
except Exception:  # pragma: no cover - script execution
    from predict_pedal import compute_metrics, postprocess_on, write_cc64  # type: ignore


def _smooth(prob: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return prob
    sig = float(sigma)
    rad = max(1, int(round(3 * sig)))
    xk = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xk / sig) ** 2)
    k /= k.sum()
    return np.convolve(prob, k, mode="same")


def _load_prob(path: Path) -> tuple[np.ndarray, Optional[np.ndarray], float, int, int]:
    obj = np.load(path, allow_pickle=False)
    prob = obj["prob"].astype("float32")
    y_true = obj["y_true"].astype("uint8") if "y_true" in obj.files else None
    fps = float(obj["fps"]) if "fps" in obj.files else 100.0
    window = int(obj["window"]) if "window" in obj.files else 64
    hop = int(obj["hop"]) if "hop" in obj.files else 16
    return prob, y_true, fps, window, hop


def _cc_count(on: np.ndarray) -> int:
    cur = None
    cc = 0
    for v in on.tolist():
        if cur is None or v != cur:
            cc += 1
            cur = v
    return cc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--stats-json", type=Path)
    ap.add_argument("--prob", type=Path)
    ap.add_argument("--make-prob", action="store_true")
    ap.add_argument("--family", choices=["ratio", "kstd"], default="ratio")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.5])
    ap.add_argument("--k-std", nargs="+", type=float, default=[2.0])
    ap.add_argument("--min-margin", nargs="+", type=float, default=[0.0])
    ap.add_argument("--off-margin", nargs="+", type=float, default=[0.0])
    ap.add_argument("--smooth-sigma", nargs="+", type=float, default=[0.0])
    ap.add_argument("--hyst-delta", nargs="+", type=float, default=[0.002])
    ap.add_argument("--min-on-sec", nargs="+", type=float, default=[0.05])
    ap.add_argument("--min-hold-sec", nargs="+", type=float, default=[0.05])
    ap.add_argument("--off-consec-sec", nargs="+", type=float, default=[0.0])
    ap.add_argument("--average", choices=["micro", "macro"], default="micro")
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--seconds-per-file", type=float)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-f1", type=float)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--emit-best-midi", action="store_true")
    args = ap.parse_args(argv)

    if args.prob is None:
        raise SystemExit("--prob is required (use scripts.predict_pedal --dump-prob)")

    prob, y_true, fps, window, hop = _load_prob(args.prob)

    rows: List[dict] = []
    count = 0

    primary = args.ratios if args.family == "ratio" else args.k_std
    for prim, sig, hyst, min_on, min_hold, off_consec, min_m, off_m in itertools.product(
        primary,
        args.smooth_sigma,
        args.hyst_delta,
        args.min_on_sec,
        args.min_hold_sec,
        args.off_consec_sec,
        args.min_margin,
        args.off_margin,
    ):
        prob_s = _smooth(prob, sig)
        if args.family == "ratio":
            on_thr = float(np.quantile(prob_s, 1.0 - prim))
            off_thr = on_thr - (max(hyst, off_m) if off_m else hyst)
        else:
            m = float(np.median(prob_s))
            sd = float(prob_s.std())
            on_thr = float(m + prim * sd + min_m)
            off_thr = on_thr - max(hyst, off_m)
        pred = postprocess_on(
            prob_s,
            on_thr=on_thr,
            off_thr=off_thr,
            fps=fps,
            min_on_sec=min_on,
            min_hold_sec=min_hold,
            eps_on=1e-6,
            off_consec_sec=off_consec,
        )
        metrics = compute_metrics(y_true, prob_s, pred) if y_true is not None else {
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "roc_auc": None,
        }
        row = {
            ("ratio" if args.family == "ratio" else "k_std"): float(prim),
            "smooth_sigma": float(sig),
            "hyst_delta": float(hyst),
            "min_on_sec": float(min_on),
            "min_hold_sec": float(min_hold),
            "off_consec_sec": float(off_consec),
            "min_margin": float(min_m),
            "off_margin": float(off_m),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "on_ratio": float(pred.mean()),
            "cc": _cc_count(pred),
        }
        rows.append(row)
        count += 1
        if args.target_f1 is not None and metrics["f1"] is not None and metrics["f1"] >= args.target_f1:
            break
        if args.budget and count >= args.budget:
            break

    def _f1_key(r):
        f = r.get("f1")
        return f if isinstance(f, (float, int)) else -1.0

    rows_sorted = sorted(rows, key=_f1_key, reverse=True)
    top = rows_sorted[: args.topk]

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.out_dir / "tuning_results.csv"
        fieldnames = list(top[0].keys()) if top else ["f1"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

    if top:
        best = top[0]
        print(json.dumps(best, indent=2))
        if args.emit_best_midi:
            sig = best["smooth_sigma"]
            prob_b = _smooth(prob, sig)
            if args.family == "ratio":
                on_thr = float(np.quantile(prob_b, 1.0 - best["ratio"]))
                off_thr = on_thr - (max(best["hyst_delta"], best["off_margin"]) if best["off_margin"] else best["hyst_delta"])
            else:
                m = float(np.median(prob_b))
                sd = float(prob_b.std())
                on_thr = float(m + best["k_std"] * sd + best["min_margin"])
                off_thr = on_thr - max(best["hyst_delta"], best["off_margin"])
            on_b = postprocess_on(
                prob_b,
                on_thr=on_thr,
                off_thr=off_thr,
                fps=fps,
                min_on_sec=best["min_on_sec"],
                min_hold_sec=best["min_hold_sec"],
                eps_on=1e-6,
                off_consec_sec=best["off_consec_sec"],
            )
            out_mid = (args.out_dir / "best.pedal.mid") if args.out_dir else Path("best.pedal.mid")
            write_cc64(out_mid, on_b, 1.0 / fps)
    else:
        print(json.dumps({}, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

