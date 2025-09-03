from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
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
                "pos_weight",
                "w_boundary",
                "f1",
                "best_th",
                "ckpt",
                "cmd",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-sweep wrapper for scripts.train_phrase")
    p.add_argument("train_csv", type=Path)
    p.add_argument("val_csv", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--archs", nargs="+", default=["lstm"], choices=["lstm", "transformer"])
    p.add_argument("--d-models", nargs="+", type=int, default=[256])
    p.add_argument("--pos-weights", nargs="+", type=float, default=[1.8])
    p.add_argument("--w-boundaries", nargs="+", type=float, default=[1.5])
    # Common passthrough (defaults mirror your recent run)
    p.add_argument("--duv-mode", default="both")
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
    # Transformer-specific
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--sin-posenc", action="store_true")
    # Stopping
    p.add_argument("--target-f1", type=float, default=0.60)
    p.add_argument("--stop-on-target", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results: list[dict[str, object]] = []
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arch in args.archs:
        for dm in args.d_models:
            for pw in args.pos_weights:
                for wb in args.w_boundaries:
                    tag = f"{arch}_dm{dm}_pw{pw:g}_wb{wb:g}"
                    ckpt = out_dir / f"{tag}.ckpt"
                    cmd = [
                        sys.executable,
                        "-m",
                        "scripts.train_phrase",
                        str(args.train_csv),
                        str(args.val_csv),
                        "--epochs",
                        str(args.epochs),
                        "--arch",
                        arch,
                        "--duv-mode",
                        args.duv_mode,
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
                        cmd.append("--use-duv-embed")
                    if args.progress:
                        cmd.append("--progress")
                    if args.num_workers:
                        cmd += ["--num-workers", str(args.num_workers)]
                    if args.pin_memory:
                        cmd.append("--pin-memory")
                    # Transformer extras
                    if arch == "transformer":
                        cmd += [
                            "--nhead",
                            str(args.nhead),
                            "--layers",
                            str(args.layers),
                        ]
                        if args.sin_posenc:
                            cmd.append("--sin-posenc")

                    print("[sweep] Running:", shlex.join(cmd))
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    # Stream logs to console
                    print(proc.stdout)
                    f1 = -1.0
                    best_th = 0.5
                    # Parse trailing JSON like {"f1": 0.58}
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
                            "pos_weight": pw,
                            "w_boundary": wb,
                            "f1": f1,
                            "best_th": best_th,
                            "ckpt": str(ckpt),
                            "cmd": shlex.join(cmd),
                        }
                    )
                    write_csv(results, out_dir / "sweep_results.csv")
                    # early stop if reaching target
                    if args.stop_on_target and f1 >= args.target_f1:
                        print(f"[sweep] Target F1 {args.target_f1} reached: {f1:.3f}")
                        write_csv(results, out_dir / "sweep_results.csv")
                        return 0
    write_csv(results, out_dir / "sweep_results.csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

