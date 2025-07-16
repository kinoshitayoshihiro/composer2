from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf


def main(argv: list[str] | None = None) -> int:
    """Apply ±cents pitch + formant shift to a stem."""
    parser = argparse.ArgumentParser(description="Randomize stem pitch/formant")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cents", type=float, required=True)
    parser.add_argument("--formant", type=int, required=True)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        audio, sr = librosa.load(args.input, sr=None)
        shifted = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=args.cents / 100 + args.formant
        )
        sf.write(args.out, shifted, sr)
    except Exception as exc:  # pragma: no cover - error handling
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
