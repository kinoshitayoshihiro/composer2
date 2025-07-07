import argparse
import logging
from pathlib import Path

from utilities.convolver import render_with_ir
from utilities.tone_shaper import ToneShaper

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("dry_wav", type=Path, help="input dry WAV")
    ap.add_argument("preset", help="preset name")
    ap.add_argument("-g", "--gain-db", type=float, default=None, help="override gain dB")
    ap.add_argument("-l", "--lufs-target", type=float, default=None, help="loudness target")
    ap.add_argument("-b", "--block-size", type=int, default=16384, help="FFT block size")
    ns = ap.parse_args(argv)

    shaper = ToneShaper()
    dry_path: Path = ns.dry_wav
    preset: str = ns.preset

    ir = shaper.get_ir_file(preset, fallback_ok=True)
    if ir is None:  # graceful-fallback
        logger.warning("No IR for preset %s; skipping convolution", preset)
        return dry_path

    out = dry_path.with_name(dry_path.stem + "_ir" + dry_path.suffix)
    render_with_ir(
        str(dry_path),
        str(ir),
        str(out),
        gain_db=ns.gain_db,
        lufs_target=ns.lufs_target,
        block_size=ns.block_size,
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = main()
    print(out)
