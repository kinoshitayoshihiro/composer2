from __future__ import annotations

"""Sweep post-processing parameters for sustain-pedal probabilities.

This CLI performs a grid search over post-processing parameters for a dumped
probability array and pedal_state labels.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:  # Optional sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score  # type: ignore
except Exception:  # pragma: no cover
    accuracy_score = precision_recall_fscore_support = roc_auc_score = None  # type: ignore


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
        return gaussian_filter1d(x, sigma, mode="nearest")
    except Exception:
        k = np.array([1, 2, 4, 2, 1], dtype=np.float32) / 10.0
        return np.convolve(x, k, mode="same")


def _apply_hysteresis(
    prob: np.ndarray,
    thr_on: float,
    thr_off: float,
    min_on: int,
    min_hold: int,
    off_consec: int,
) -> np.ndarray:
    n = prob.size
    on = np.zeros(n, dtype=np.uint8)
    state = 0
    on_run = 0
    hold = 0
    off_run = 0
    for i in range(n):
        p = prob[i]
        if state == 0:
            if p >= thr_on:
                on_run += 1
            else:
                on_run = 0
            if on_run >= min_on:
                start = i - on_run + 1
                on[start : i + 1] = 1
                state = 1
                hold = max(0, min_hold - on_run)
                off_run = 0
        else:
            on[i] = 1
            if hold > 0:
                hold -= 1
                off_run = 0
            else:
                if p < thr_off:
                    off_run += 1
                    if off_run >= off_consec:
                        on[i - off_run + 1 : i + 1] = 0
                        state = 0
                        on_run = 0
                        off_run = 0
                else:
                    off_run = 0
    return on

def _load_prob(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    for key in ("prob", "probs", "p", "y_pred_proba"):
        if key in obj.files:
            prob = obj[key]
            break
    else:
        if "logits" in obj.files:
            z = obj["logits"]
            prob = 1.0 / (1.0 + np.exp(-z))
        else:
            raise KeyError("no prob array found in NPZ")
    return prob.astype(np.float32).ravel()


def _load_labels(csv_path: Path) -> np.ndarray:
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path, usecols=["pedal_state"], low_memory=False)
        arr = df["pedal_state"].to_numpy()
    except Exception:
        import csv as _csv
        vals: List[float] = []
        with csv_path.open(newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                vals.append(float(row["pedal_state"]))
        arr = np.asarray(vals, dtype=np.float32)
    return (arr > 0.5).astype(np.uint8)


def _load_fps(ckpt: Optional[Path], stats_json: Optional[Path]) -> float:
    path: Optional[Path] = None
    if stats_json and stats_json.exists():
        path = stats_json
    elif ckpt:
        cand = Path(str(ckpt) + ".stats.json")
        if cand.exists():
            path = cand
    if path:
        try:
            with path.open() as f:
                data = json.load(f)
                return float(data.get("fps", 100.0))
        except Exception:
            return 100.0
    return 100.0


def _metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, average: str) -> dict:
    if accuracy_score and precision_recall_fscore_support:
        prec, rec, f1, _ = precision_recall_fscore_support(  # type: ignore
            y_true, pred, average=average, zero_division=0
        )
        acc = accuracy_score(y_true, pred)  # type: ignore
        try:
            auc = roc_auc_score(y_true, prob) if roc_auc_score else float("nan")
        except Exception:
            auc = float("nan")
        return {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "roc_auc": float(auc),
        }
    y = y_true.astype(int)
    p = pred.astype(int)
    acc = float((p == y).mean())
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    if average == "micro":
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    else:
        def _prf(tpv, fpv, fnv):
            pr = tpv / (tpv + fpv) if (tpv + fpv) else 0.0
            rc = tpv / (tpv + fnv) if (tpv + fnv) else 0.0
            f = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            return pr, rc, f

        pr0, rc0, f0 = _prf(tn, fn, fp)
        pr1, rc1, f1 = _prf(tp, fp, fn)
        prec = (pr0 + pr1) / 2.0
        rec = (rc0 + rc1) / 2.0
        f1 = (f0 + f1) / 2.0
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
        "roc_auc": float("nan"),
    }

def write_cc64(on: np.ndarray, fps: float, out_mid: Path) -> None:
    step = 1.0 / fps
    try:
        import pretty_midi as pm  # type: ignore
        midi = pm.PrettyMIDI()
        inst = pm.Instrument(program=0)
        midi.instruments.append(inst)
        cc = inst.control_changes
        cur = None
        for i, v in enumerate(on.tolist()):
            val = 127 if v else 0
            if cur != val:
                t = float(i * step)
                cc.append(pm.ControlChange(number=64, value=val, time=t))
                cur = val
        out_mid.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(out_mid))
    except Exception:
        print("warning: pretty_midi not installed; skipping MIDI")


def _cartesian_product(args: Sequence[Iterable[float]]) -> Iterable[tuple[float, ...]]:
    from itertools import product
    return product(*args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--stats-json", type=Path)
    ap.add_argument("--prob", type=Path, required=True)
    ap.add_argument("--family", choices=["ratio", "kstd"], default="ratio")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.50])
    ap.add_argument("--k-std", nargs="+", type=float, default=[2.0])
    ap.add_argument("--smooth-sigma", nargs="+", type=float, default=[4.0])
    ap.add_argument("--hyst-delta", nargs="+", type=float, default=[0.006])
    ap.add_argument("--min-on-sec", nargs="+", type=float, default=[0.10])
    ap.add_argument("--min-hold-sec", nargs="+", type=float, default=[0.10])
    ap.add_argument("--off-consec-sec", nargs="+", type=float, default=[0.08])
    ap.add_argument("--min-margin", nargs="+", type=float, default=[0.0005])
    ap.add_argument("--off-margin", nargs="+", type=float, default=[0.0005])
    ap.add_argument("--average", choices=["micro", "macro"], default="micro")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/pedal_tuning"), help="output directory")
    ap.add_argument("--emit-best-midi", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    prob = _load_prob(args.prob)
    y_true = _load_labels(args.csv)
    if prob.shape[0] != y_true.shape[0]:
        raise SystemExit("length mismatch between prob and labels")
    fps = _load_fps(args.ckpt, args.stats_json)

    combos = _cartesian_product([
        args.ratios if args.family == "ratio" else args.k_std,
        args.smooth_sigma,
        args.hyst_delta,
        args.min_on_sec,
        args.min_hold_sec,
        args.off_consec_sec,
        args.min_margin,
        args.off_margin,
    ])

    rows: List[dict] = []
    for prim, sig, hyst, min_on_s, min_hold_s, off_consec_s, min_m, off_m in combos:
        prob_s = _smooth(prob, sig)
        if args.family == "ratio":
            base = float(np.quantile(prob_s, 1.0 - prim))
        else:
            base = float(np.median(prob_s) + prim * prob_s.std())
        thr_on = base + min_m + hyst / 2.0
        thr_off = base - off_m - hyst / 2.0
        min_on = int(round(min_on_s * fps))
        min_hold = int(round(min_hold_s * fps))
        off_consec = int(round(off_consec_s * fps))
        on = _apply_hysteresis(prob_s, thr_on, thr_off, min_on, min_hold, off_consec)
        metrics = _metrics(y_true, prob_s, on, args.average)
        row = {
            ("ratio" if args.family == "ratio" else "k_std"): float(prim),
            "smooth_sigma": float(sig),
            "hyst_delta": float(hyst),
            "min_on_sec": float(min_on_s),
            "min_hold_sec": float(min_hold_s),
            "off_consec_sec": float(off_consec_s),
            "min_margin": float(min_m),
            "off_margin": float(off_m),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "on_ratio": float(on.mean()),
        }
        rows.append(row)

    def _sort_key(r: dict) -> tuple[float, float]:
        f1 = r.get("f1")
        auc = r.get("roc_auc")
        f1v = f1 if isinstance(f1, (float, int)) else -1.0
        aucv = auc if isinstance(auc, (float, int)) and math.isfinite(auc) else -1.0
        return (f1v, aucv)

    rows.sort(key=_sort_key, reverse=True)
    top = rows[: args.topk]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "tuning_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    if top:
        best = top[0]
        print(json.dumps(best, ensure_ascii=False, indent=2))
        if args.emit_best_midi:
            prob_s = _smooth(prob, best["smooth_sigma"])
            if args.family == "ratio":
                base = float(np.quantile(prob_s, 1.0 - best["ratio"]))
            else:
                base = float(np.median(prob_s) + best["k_std"] * prob_s.std())
            thr_on = base + best["min_margin"] + best["hyst_delta"] / 2.0
            thr_off = base - best["off_margin"] - best["hyst_delta"] / 2.0
            min_on = int(round(best["min_on_sec"] * fps))
            min_hold = int(round(best["min_hold_sec"] * fps))
            off_consec = int(round(best["off_consec_sec"] * fps))
            on_best = _apply_hysteresis(prob_s, thr_on, thr_off, min_on, min_hold, off_consec)
            write_cc64(on_best, fps, args.out_dir / "val.pedal.mid")
    else:
        print(json.dumps({}, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
