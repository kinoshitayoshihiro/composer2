import argparse
from pathlib import Path
from typing import Any

import modular_composer
from utilities.config_loader import load_chordmap_yaml, load_main_cfg
from utilities.rhythm_library_loader import load_rhythm_library


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Arrange sections using emotion-aware chordmap")
    parser.add_argument(
        "--chordmap", type=Path, default=Path("data/processed_chordmap_with_emotion.yaml")
    )
    parser.add_argument("--rhythm", type=Path, default=Path("data/rhythm_library.yml"))
    parser.add_argument("--main-cfg", type=Path, default=Path("config/main_cfg.yml"))
    parser.add_argument("--output", type=Path, default=Path("emotion_output.mid"))
    args = parser.parse_args(argv)

    chordmap = load_chordmap_yaml(args.chordmap)
    rhythm_lib = load_rhythm_library(args.rhythm)
    main_cfg = load_main_cfg(args.main_cfg)

    score, _ = modular_composer.compose(main_cfg, chordmap, rhythm_lib)  # type: ignore[attr-defined]
    score.write("midi", fp=str(args.output))
    print(f"Exported MIDI: {args.output}")


if __name__ == "__main__":
    main()
