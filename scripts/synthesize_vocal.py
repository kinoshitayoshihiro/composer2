import argparse
import json
from pathlib import Path

from tts_model import synthesize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthesize vocals with phoneme sequence")
    parser.add_argument("--mid", type=Path, required=True)
    parser.add_argument("--phonemes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.phonemes.open("r", encoding="utf-8") as f:
        phonemes = json.load(f)

    audio = synthesize(args.mid, phonemes)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{args.mid.stem}.wav"
    out_file.write_bytes(audio)
    print(out_file)
    return out_file


if __name__ == "__main__":
    main()
