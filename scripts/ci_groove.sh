#!/usr/bin/env bash
set -e
ruff check . --select I,U
mypy modular_composer utilities tests --strict --warn-unused-ignores
python - <<'PY'
import statistics, tempfile, time
from pathlib import Path
import pretty_midi
from utilities import groove_sampler_ngram as gs

with tempfile.TemporaryDirectory() as d:
    for i in range(2):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
        pm.instruments.append(inst)
        pm.write(f"{d}/{i}.mid")
    model = gs.train(Path(d), order=1)
    uncached = []
    cached = []
    for _ in range(5):
        t0 = time.time()
        gs.sample(model, bars=8, use_bar_cache=False)
        uncached.append(time.time() - t0)
        t0 = time.time()
        gs.sample(model, bars=8)
        cached.append(time.time() - t0)
    med_no = statistics.median(uncached)
    med_yes = statistics.median(cached)
    ratio = med_no / med_yes if med_yes else float('inf')
    diff = med_no - med_yes
    print(f"ratio {ratio:.2f} diff {diff:.3f}")
    if ratio < 1.25 and diff > 0.5:
        raise SystemExit(1)
PY
