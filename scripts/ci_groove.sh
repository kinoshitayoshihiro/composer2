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
    model = gs.train(Path(d), order=1)
    t0 = time.time()
    gs.sample(model, bars=8, seed=0, no_bar_cache=True)
    uncached = time.time() - t0
    t0 = time.time()
    gs.sample(model, bars=8, seed=0)
    cached = time.time() - t0
    ratio = uncached / cached if cached else 1.0
    print(f"uncached {uncached:.2f}s cached {cached:.2f}s ratio {ratio:.2f}")
    if ratio < 1.25:
        raise SystemExit("bar-cache speed-up <25%")
PY
