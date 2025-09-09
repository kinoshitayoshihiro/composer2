from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import random
from pathlib import Path
from typing import Iterable


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "d_model",
                "seed",
                "pos_weight",
                "w_boundary",
                "duv_mode",
                "dropout",
                "reweight",
                "f1",
                "best_th",
                "ckpt",
                "cmd",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def make_key(r: dict[str, object]) -> tuple:
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return x
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return x
    return (
        str(r.get("arch")),
        _to_int(r.get("d_model")),
        _to_int(r.get("seed")),
        _to_float(r.get("pos_weight")),
        _to_float(r.get("w_boundary")),
        str(r.get("duv_mode")) if r.get("duv_mode") is not None else "",
        _to_float(r.get("dropout")),
        str(r.get("reweight", "")),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-sweep wrapper for scripts.train_phrase")
    p.add_argument("train_csv", type=Path)
    p.add_argument("val_csv", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--archs", nargs="+", default=["lstm"], choices=["lstm", "transformer"])
    p.add_argument("--d-models", nargs="+", type=int, default=[256])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--pos-weights", nargs="+", type=float, default=[1.8])
    p.add_argument("--w-boundaries", nargs="+", type=float, default=[1.5])
    # Common passthrough (defaults mirror your recent run)
    p.add_argument("--duv-modes", nargs="+", default=["both"], choices=["none", "reg", "cls", "both"])
    p.add_argument("--vel-bins", type=int, default=8)
    p.add_argument("--dur-bins", type=int, default=16)
    p.add_argument("--use-duv-embed", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--instrument", type=str, default="guitar")
    p.add_argument("--w-dur-reg", type=float, default=0.05)
    p.add_argument("--w-vel-reg", type=float, default=0.05)
    p.add_argument("--w-vel-cls", type=float, default=0.05)
    p.add_argument("--w-dur-cls", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", default="plateau")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-patience", type=int, default=1)
    p.add_argument("--f1-scan-range", nargs=3, type=float, default=(0.30, 0.95, 0.01))
    p.add_argument("--early-stopping", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    # Sweep control
    p.add_argument("--skip-done", action="store_true", help="skip runs already present in sweep_results.csv")
    p.add_argument("--max-runs", type=int, default=0, help="max number of runs to execute in this invocation")
    p.add_argument("--stop-file", type=Path, help="if this file exists, stop before launching the next run")
    p.add_argument("--random", type=int, default=0, help="if >0, sample this many random combos instead of full grid")
    p.add_argument("--sweep-seed", type=int, default=1337)
    # Random ranges (used when --random > 0)
    p.add_argument("--pos-weights-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--w-boundaries-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--dropout-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    # Transformer-specific
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.3])
    p.add_argument("--sin-posenc", action="store_true")
    # Reweighting schemes (use empty string for none)
    p.add_argument("--reweights", nargs="+", type=str, default=[""], help="list of reweight configs, e.g., tag=instrument,scheme=inv_freq; empty for none")
    # Stopping
    p.add_argument("--target-f1", type=float, default=0.60)
    p.add_argument("--stop-on-target", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "sweep_results.csv"
    results: list[dict[str, object]] = read_csv(results_path)
    done_keys = set(make_key(r) for r in results) if args.skip_done else set()

    ran = 0
    rng = random.Random(args.sweep_seed)

    def should_stop_now() -> bool:
        if args.max_runs and ran >= args.max_runs:
            print(f"[sweep] Reached max runs: {args.max_runs}")
            return True
        if args.stop_file and args.stop_file.exists():
            print(f"[sweep] Stop-file detected: {args.stop_file}")
            return True
        return False

    def iter_grid():
        for duv_mode in args.duv_modes:
            for arch in args.archs:
                for dm in args.d_models:
                    for seed in args.seeds:
                        for pw in args.pos_weights:
                            for wb in args.w_boundaries:
                                for rw in args.reweights:
                                    yield duv_mode, arch, dm, seed, float(pw), float(wb), rw

    def iter_random():
        for _ in range(max(1, args.random)):
            duv_mode = rng.choice(args.duv_modes)
            arch = rng.choice(args.archs)
            dm = rng.choice(args.d_models)
            seed = rng.choice(args.seeds)
            pw = (
                rng.uniform(float(args.pos_weights_range[0]), float(args.pos_weights_range[1]))
                if args.pos_weights_range
                else float(rng.choice(args.pos_weights))
            )
            wb = (
                rng.uniform(float(args.w_boundaries_range[0]), float(args.w_boundaries_range[1]))
                if args.w_boundaries_range
                else float(rng.choice(args.w_boundaries))
            )
            rw = rng.choice(args.reweights)
            yield duv_mode, arch, dm, seed, pw, wb, rw

    combo_iter = iter_random() if args.random and args.random > 0 else iter_grid()

    for duv_mode, arch, dm, seed, pw, wb, rw in combo_iter:
        if should_stop_now():
            break
        tag = f"{arch}_dm{dm}_seed{seed}_pw{pw:g}_wb{wb:g}_{duv_mode}"
        ckpt = out_dir / f"{tag}.ckpt"
        base_cmd = [
            sys.executable,
            "-m",
            "scripts.train_phrase",
            str(args.train_csv),
            str(args.val_csv),
            "--epochs",
            str(args.epochs),
            "--arch",
            arch,
            "--seed",
            str(seed),
            "--duv-mode",
            duv_mode,
            "--vel-bins",
            str(args.vel_bins),
            "--dur-bins",
            str(args.dur_bins),
            "--out",
            str(ckpt),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--d_model",
            str(dm),
            "--max-len",
            str(args.max_len),
            "--instrument",
            args.instrument,
            "--w-boundary",
            str(wb),
            "--w-dur-reg",
            str(args.w_dur_reg),
            "--w-vel-reg",
            str(args.w_vel_reg),
            "--w-vel-cls",
            str(args.w_vel_cls),
            "--w-dur-cls",
            str(args.w_dur_cls),
            "--pos-weight",
            str(pw),
            "--dropout",
            str(args.dropout),
            "--weight-decay",
            str(args.weight_decay),
            "--scheduler",
            args.scheduler,
            "--lr",
            str(args.lr),
            "--lr-patience",
            str(args.lr_patience),
            "--f1-scan-range",
            str(args.f1_scan_range[0]),
            str(args.f1_scan_range[1]),
            str(args.f1_scan_range[2]),
            "--early-stopping",
            str(args.early_stopping),
            "--save-best",
            "--device",
            args.device,
        ]
        if args.use_duv_embed:
            base_cmd.append("--use-duv-embed")
        if args.progress:
            base_cmd.append("--progress")
        if args.num_workers:
            base_cmd += ["--num-workers", str(args.num_workers)]
        if args.pin_memory:
            base_cmd.append("--pin-memory")
        if rw:
            base_cmd += ["--reweight", rw]

        if arch == "transformer":
            dp_list = []
            if args.random and args.dropout_range:
                dp_list = [rng.uniform(float(args.dropout_range[0]), float(args.dropout_range[1]))]
            else:
                dp_list = list(args.dropouts)
            for dp in dp_list:
                key = (arch, dm, int(seed), float(pw), float(wb), duv_mode, float(dp), rw)
                if done_keys and key in done_keys:
                    print(f"[sweep] Skipping done: {key}")
                    continue
                ckpt_dp = ckpt.with_name(f"{ckpt.stem}_dp{dp:g}")
                cmd_tf = list(base_cmd)
                idx = cmd_tf.index("--out")
                cmd_tf[idx + 1] = str(ckpt_dp)
                cmd_tf += [
                    "--nhead",
                    str(args.nhead),
                    "--layers",
                    str(args.layers),
                    "--dropout",
                    str(dp),
                ]
                if args.sin_posenc:
                    cmd_tf.append("--sin-posenc")
                f1 = run_once(
                    cmd_tf,
                    results,
                    out_dir,
                    arch,
                    dm,
                    int(seed),
                    float(pw),
                    float(wb),
                    duv_mode,
                    float(dp),
                    rw,
                    ckpt_dp,
                    args,
                )
                ran += 1
                done_keys.add(key)
                write_csv(results, results_path)
                if args.stop_on_target and f1 is not None and f1 >= args.target_f1:
                    print(f"[sweep] Target F1 {args.target_f1} reached: {f1:.3f}")
                    return 0
                if should_stop_now():
                    return 0
            continue

        key = (arch, dm, int(seed), float(pw), float(wb), duv_mode, float(args.dropout), rw)
        if done_keys and key in done_keys:
            print(f"[sweep] Skipping done: {key}")
            continue
        f1 = run_once(
            base_cmd,
            results,
            out_dir,
            arch,
            dm,
            int(seed),
            float(pw),
            float(wb),
            duv_mode,
            float(args.dropout),
            rw,
            ckpt,
            args,
        )
        ran += 1
        done_keys.add(key)
        write_csv(results, results_path)
        if args.stop_on_target and f1 is not None and f1 >= args.target_f1:
            print(f"[sweep] Target F1 {args.target_f1} reached: {f1:.3f}")
            return 0
        if should_stop_now():
            return 0
    write_csv(results, results_path)
    return 0


def run_once(
    cmd: list[str],
    results: list[dict[str, object]],
    out_dir: Path,
    arch: str,
    dm: int,
    seed: int,
    pw: float,
    wb: float,
    duv_mode: str,
    dropout: float,
    reweight: str,
    ckpt: Path,
    args: argparse.Namespace,
) -> float | None:
    print("[sweep] Running:", shlex.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Stream logs to console
    print(proc.stdout)
    f1 = -1.0
    best_th = 0.5
    # Parse trailing JSON like {"f1": 0.58, "best_th": 0.52}
    for line in proc.stdout.strip().splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "f1" in obj:
                    f1 = float(obj["f1"])
                if "best_th" in obj:
                    best_th = float(obj["best_th"])
                break
            except Exception:
                pass
    results.append(
        {
            "arch": arch,
            "d_model": dm,
            "seed": seed,
            "pos_weight": pw,
            "w_boundary": wb,
            "duv_mode": duv_mode,
            "dropout": dropout,
            "reweight": reweight,
            "f1": f1,
            "best_th": best_th,
            "ckpt": str(ckpt),
            "cmd": shlex.join(cmd),
        }
    )
    write_csv(results, out_dir / "sweep_results.csv")
    return f1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
