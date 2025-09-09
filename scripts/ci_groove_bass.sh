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
import pretty_midi

with tempfile.TemporaryDirectory() as d:
    # train tiny groove model
    pm_dir = Path(d)/"mid"
    pm_dir.mkdir()
    for i in range(2):
        path = pm_dir / f"{i}.mid"
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1)
        )
        pm.instruments.append(inst)
        pm.write(str(path))
    model = gs.train(pm_dir, order=1)
    events = gs.sample(model, bars=4)
    kicks = [e["offset"] for e in events if e["instrument"] == "kick"]
    drum = DrumGenerator(
        part_name="drums",
        part_parameters={},
        default_instrument=instrument.Woodblock(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
    )
    drum.render_kick_track(4.0)
    bass = BassGenerator(part_name="bass", default_instrument=instrument.AcousticBass(), global_tempo=120, global_time_signature="4/4", global_key_signature_tonic="C", global_key_signature_mode="major", emotion_profile_path="data/emotion_profile.yaml")
    part = bass.render_part(
        {
            "emotion": "joy",
            "key_signature": "C",
            "tempo_bpm": 120,
            "groove_kicks": kicks,
            "chord": "C",
            "melody": [],
        }
    )
    assert len(part.notes) == 4
PY
end=$(date +%s)

runtime=$((end - start))
echo "Bass CI runtime: ${runtime}s"
if [ "$runtime" -gt 60 ]; then
  echo "Runtime exceeded 60 seconds" >&2
  exit 1
fi
