#!/usr/bin/env bash
set -e
start=$(date +%s)
bash scripts/ci_groove.sh
python - <<'PY'
import tempfile
from pathlib import Path
from music21 import instrument
from generator.drum_generator import DrumGenerator
from generator.bass_generator import BassGenerator
from utilities import groove_sampler_ngram as gs

with tempfile.TemporaryDirectory() as d:
    # train tiny groove model
    pm_dir = Path(d)/"mid"
    pm_dir.mkdir()
    for i in range(2):
        path = pm_dir/f"{i}.mid"
        open(path, "wb").close()
    model = gs.train(pm_dir, order=1)
    events = gs.sample(model, bars=4)
    kicks = [e.offset for e in events.events if e.drum == "kick"]
    drum = DrumGenerator(part_name="drums", part_parameters={}, default_instrument=instrument.Woodblock(), global_tempo=120, global_time_signature="4/4", global_key_signature_tonic="C", global_key_signature_mode="major")
    drum.render_kick_track(4.0)
    bass = BassGenerator(part_name="bass", default_instrument=instrument.AcousticBass(), global_tempo=120, global_time_signature="4/4", global_key_signature_tonic="C", global_key_signature_mode="major", emotion_profile_path="data/emotion_profile.yaml")
    part = bass.render_part(emotion="joy", key_signature="C", tempo_bpm=120, groove_history=kicks)
    assert len(part.notes) == 4
PY
end=$(date +%s)
elapsed=$((end - start))
if [ "$elapsed" -gt 60 ]; then
  echo "ci_groove_bass.sh took ${elapsed}s" >&2
  exit 1
fi
