from __future__ import annotations

import argparse
import importlib.metadata as _md
from pathlib import Path

from utilities.groove_sampler_v2 import generate_events, load, save, train  # noqa: F401
from utilities.peak_extractor import PeakExtractorConfig, extract_peaks
from utilities.peak_synchroniser import PeakSynchroniser

try:
    __version__ = _md.version("modular_composer")
except _md.PackageNotFoundError:
    __version__ = "0.0.0"


def _cmd_demo(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose demo")
    ap.add_argument("-o", "--out", type=Path, default=Path("demo.mid"))
    ns = ap.parse_args(args)
    model = train(Path("data/loops"), n=2, auto_res=True)
    ev = generate_events(model, bars=4)
    # (PrettyMIDI export placeholder)
    print(f"[demo] generated {len(ev)} events -> {ns.out}")


def _cmd_sample(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose sample")
    ap.add_argument("model", type=Path)
    ap.add_argument("-l", "--length", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    ap.add_argument(
        "--cond-kick", choices=["four_on_floor", "sparse"], default=None
    )
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--peaks", type=Path)
    ap.add_argument("--lag", type=float, default=10.0)
    ns = ap.parse_args(args)
    model = load(ns.model)
    ev = generate_events(
        model,
        bars=ns.length,
        temperature=ns.temperature,
        seed=ns.seed,
        cond_velocity=ns.cond_velocity,
        cond_kick=ns.cond_kick,
    )
    if ns.peaks:
        import json

        with ns.peaks.open() as fh:
            peaks = json.load(fh)
        ev = PeakSynchroniser.sync_events(
            peaks,
            ev,
            tempo_bpm=120.0,
            lag_ms=ns.lag,
        )
    import json
    import sys

    if ns.out is None:
        json.dump(ev, sys.stdout)
    else:
        with ns.out.open("w") as fh:
            json.dump(ev, fh)


def _cmd_peaks(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose peaks")
    ap.add_argument("wav", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("peaks.json"))
    ap.add_argument("--cfg", type=Path)
    ap.add_argument("--plot", action="store_true")
    ns = ap.parse_args(args)

    import json

    import librosa
    import matplotlib.pyplot as plt
    import yaml  # type: ignore
    from scipy.ndimage import uniform_filter1d

    cfg = PeakExtractorConfig()
    if ns.cfg:
        with ns.cfg.open() as fh:
            cfg = PeakExtractorConfig(**yaml.safe_load(fh))
    peaks = extract_peaks(ns.wav, cfg)
    with ns.out.open("w") as fh:
        json.dump(peaks, fh)

    if ns.plot:
        y, sr = librosa.load(ns.wav, sr=cfg.sr, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        win = max(1, int(round(cfg.rms_smooth_ms / 1000 * sr / cfg.hop_length)))
        rms_db = uniform_filter1d(rms_db, win)
        times = librosa.frames_to_time(range(len(rms_db)), sr=sr, hop_length=cfg.hop_length)
        plt.step(times, rms_db, where="mid")
        for p in peaks:
            plt.axvline(p, color="r", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS (dB)")
        plt.tight_layout()
        plt.savefig(ns.out.with_suffix(".png"))
        plt.close()

    print(f"{len(peaks)} peaks -> {ns.out}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: modcompose <command> [<args>]\n\ncommands: demo, sample, peaks"
        )
        sys.exit(0)
    cmd, *rest = argv
    if cmd == "demo":
        _cmd_demo(rest)
    elif cmd == "sample":
        _cmd_sample(rest)
    elif cmd == "peaks":
        _cmd_peaks(rest)
    else:
        sys.exit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
