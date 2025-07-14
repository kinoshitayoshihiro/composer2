import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


def _classify(pitch: float, amp: float, dur: float) -> int:
    if pitch < 50 and amp < 0.5:
        return 36  # kick
    if 50 <= pitch < 65:
        return 38  # snare
    if pitch >= 65 and dur < 0.3:
        return 42  # hi-hat
    return 39  # other


def _vel(amp: float) -> int:
    return int(np.clip(np.interp(amp, [0.0, 1.0], [20, 120]), 1, 127))


def _process(path: Path, out_dir: Path, overwrite: bool) -> None:
    out = out_dir / f"{path.stem}.mid"
    if out.exists() and not overwrite:
        return
    from basic_pitch.inference import predict

    _, _midi, notes = predict(str(path))

    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for start, end, pitch, amp, *_ in notes:
        dur = end - start
        inst.notes.append(
            pretty_midi.Note(
                velocity=_vel(amp),
                pitch=_classify(pitch, amp, dur),
                start=float(start),
                end=float(end),
            )
        )
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    pm.write(str(out))


def convert(in_dir: Path, out_dir: Path, part: str, exts: Iterable[str], stem: str | None, jobs: int, overwrite: bool) -> None:
    paths: list[Path] = []
    for ext in exts:
        paths.extend(in_dir.rglob(f"*.{ext}"))
    if stem:
        paths = [p for p in paths if p.name == stem]
    out_dir.mkdir(parents=True, exist_ok=True)
    func = partial(_process, out_dir=out_dir, overwrite=overwrite)
    if jobs == 1:
        for p in paths:
            func(p)
    else:
        with mp.Pool(jobs) as pool:
            pool.map(func, paths)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch convert audio to MIDI")
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--part", default="drums", choices=["drums", "perc"])
    parser.add_argument("--ext", default="wav,mp3,flac")
    parser.add_argument("--stem")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    convert(Path(args.in_dir), Path(args.out_dir), args.part, exts, args.stem, args.jobs, args.overwrite)


if __name__ == "__main__":
    main()
