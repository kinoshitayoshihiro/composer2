import argparse
import csv
from pathlib import Path

import librosa
import pretty_midi


def wav_to_midi(wav_path: Path, midi_path: Path) -> None:
    y, sr = librosa.load(str(wav_path), sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for t in onset_times:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=60, start=float(t), end=float(t + 0.1))
        )
    pm.instruments.append(inst)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))


def build_csv(midi_paths: list[Path], csv_path: Path) -> None:
    rows = []
    for path in midi_paths:
        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            for n in inst.notes:
                rows.append((n.pitch, n.start, n.velocity))
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["note_on", "time", "velocity"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert audio loops to MIDI and CSV")
    parser.add_argument("--loops-dir", default="data/loops")
    parser.add_argument("--midi-dir", default="data/loops_converted")
    parser.add_argument("--out", default="data/velocity.csv")
    args = parser.parse_args(argv)

    loops_dir = Path(args.loops_dir)
    midi_dir = Path(args.midi_dir)
    out_csv = Path(args.out)

    midi_paths: list[Path] = []
    for wav in loops_dir.rglob("*.wav"):
        midi_path = midi_dir / f"{wav.stem}.mid"
        wav_to_midi(wav, midi_path)
        midi_paths.append(midi_path)

    build_csv(midi_paths, out_csv)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
