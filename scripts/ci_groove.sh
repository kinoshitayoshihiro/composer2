#!/usr/bin/env bash
set -e
ruff check .
mypy modular_composer utilities tests --strict
python - <<'PY'
import tempfile, time, pretty_midi
from pathlib import Path
from utilities import groove_sampler_ngram as gs

with tempfile.TemporaryDirectory() as d:
    for i in range(2):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
        pm.instruments.append(inst)
        pm.write(f"{d}/{i}.mid")
    t0 = time.time()
    model = gs.train(Path(d), order=1)
    gs.sample(model, bars=8, seed=0)
    elapsed = time.time() - t0
    print(f"elapsed {elapsed:.2f}s")
    if elapsed > 60:
        raise SystemExit("runtime >60s")
PY
