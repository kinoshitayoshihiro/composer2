import argparse
from pathlib import Path

try:
    from sunoai import GenerationClient
except Exception:  # pragma: no cover - missing dependency
    GenerationClient = None  # type: ignore


def synthesize(model: str) -> bytes:
    """Generate vocal audio using SunoAI."""
    if GenerationClient is None:
        raise RuntimeError("sunoai not installed")
    client = GenerationClient(model)
    return client.generate("Hello")  # minimal prompt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vocal synthesis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    audio = synthesize(args.model)
    out = Path(args.output)
    out.write_bytes(audio)
    print(out)
    return out


if __name__ == "__main__":
    main()
