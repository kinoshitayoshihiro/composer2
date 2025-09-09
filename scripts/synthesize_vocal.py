import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Synthesize vocals with phoneme sequence"
    )
    parser.add_argument("--mid", type=Path, required=True)
    parser.add_argument("--phonemes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    with args.phonemes.open("r", encoding="utf-8") as f:
        phonemes = json.load(f)

    try:
        logging.info("Starting synthesis for %s", args.mid)
        if args.onnx_model:
            from generator.vocal_generator import synthesize_with_onnx

            audio = synthesize_with_onnx(args.onnx_model, args.mid, phonemes)
        else:
            from tts_model import synthesize  # type: ignore

            audio = synthesize(args.mid, phonemes)
        logging.info("Finished synthesis")
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - runtime path
        logging.error("TTS synthesis failed: %s", exc, exc_info=True)
        sys.exit(1)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{args.mid.stem}.wav"
    out_file.write_bytes(audio)
    print(out_file)
    return out_file


if __name__ == "__main__":
    main()
