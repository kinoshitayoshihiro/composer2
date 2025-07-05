import argparse
import logging
from pathlib import Path
import shutil

import pretty_midi

from generator.guitar_generator import GuitarGenerator
from utilities import convolver

log = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("score", type=Path)
    parser.add_argument("--ir-folder", type=Path, required=True)
    parser.add_argument("--preset", default="crunch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pm = pretty_midi.PrettyMIDI(str(args.score))
    for idx, inst in enumerate(pm.instruments):
        out = Path(f"track{idx}.wav")
        ir = args.ir_folder / f"{args.preset}.wav"
        log.info("Render %s with %s -> %s", inst.name or idx, ir, out)
        if args.dry_run:
            continue
        # Placeholder rendering: copy IR as output
        shutil.copy(ir, out)
        convolver.render_with_ir(out, ir, out, progress=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
