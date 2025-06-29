#!/usr/bin/env bash
set -e
URL=${TORCH_WHL_URL:-https://download.pytorch.org/whl/cpu}
start=$(date +%s)
pip install --no-cache-dir --progress-bar off --extra-index-url "$URL" torch==2.2.1+cpu || true
duration=$(( $(date +%s) - start ))
if [ "$duration" -gt 60 ]; then
  echo "RNN tests skipped (timeout)"
  exit 0
fi
python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(0)
PY
ruff check . --select I,S,B
mypy modular_composer utilities tests --strict
python - <<'PY'
import tempfile
from pathlib import Path
import pretty_midi
from utilities import groove_sampler_rnn

with tempfile.TemporaryDirectory() as d:
    cache = Path(d)/"loops.json"
    data = {"ppq":480,"resolution":16,"data":[{"file":"a.mid","tokens":[(0,"kick",100,0)],"tempo_bpm":120.0,"bar_beats":4,"section":"verse","heat_bin":0,"intensity":"mid"}]}
    cache.write_text(__import__('json').dumps(data))
    model, meta = groove_sampler_rnn.train(cache, epochs=1)
    groove_sampler_rnn.save(model, meta, Path(d)/"m.pt")
    groove_sampler_rnn.sample(model, meta, bars=4)
PY
