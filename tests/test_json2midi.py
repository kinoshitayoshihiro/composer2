import json
import subprocess
from pathlib import Path

import pretty_midi

SCRIPT = Path(__file__).resolve().parents[1] / "json2midi.py"


def run_cli(args):
    return subprocess.run(
        [
            "python",
            str(SCRIPT),
            *args,
        ],
        capture_output=True,
        text=True,
    )


def make_events(n=1):
    return [
        {
            "instrument": "snare",
            "offset": i,
            "duration": 0.5,
            "velocity_factor": 1,
        }
        for i in range(n)
    ]


def test_version_and_help(tmp_path):
    out = run_cli(["--version"])
    assert out.stdout.strip().startswith("0.")
    help_out = run_cli(["--help"])
    assert "--split-tracks" in help_out.stdout


def test_swing_and_repeat(tmp_path):
    data = [
        {"instrument": "snare", "offset": 0.5, "duration": 0.5, "velocity_factor": 1}
    ]
    json_path = tmp_path / "a.json"
    json_path.write_text(json.dumps(data))
    midi_path = tmp_path / "a.mid"
    run_cli([str(json_path), "--swing", "0.5", "--out", str(midi_path)])
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    assert len(pm.instruments[0].notes) == 1
    start = pm.instruments[0].notes[0].start
    assert 0.18 <= start <= 0.19  # 0.375 beat at 120 BPM ~=0.1875s
    dur = pm.instruments[0].notes[0].end - start
    assert 0.24 <= dur <= 0.26  # duration stays ~0.25s


def test_progress_bar_and_quiet(tmp_path):
    events = make_events(150)
    json_path = tmp_path / "many.json"
    json_path.write_text(json.dumps(events))
    res = run_cli([str(json_path)])
    assert "100%" in res.stderr
    res_q = run_cli([str(json_path), "-q"])
    assert "100%" not in res_q.stderr


def test_multiple_inputs_and_yaml_map(tmp_path):
    events = make_events()
    j1 = tmp_path / "one.json"
    j2 = tmp_path / "two.json"
    j1.write_text(json.dumps(events))
    j2.write_text(json.dumps(events))
    yaml_map = tmp_path / "map.yaml"
    yaml_map.write_text("snare: 40\n")
    run_cli([str(j1), str(j2), "--map", str(yaml_map)])
    m1 = tmp_path / "one.mid"
    m2 = tmp_path / "two.mid"
    assert m1.exists() and m2.exists()
    p1 = pretty_midi.PrettyMIDI(str(m1))
    assert p1.instruments[0].notes[0].pitch == 40


def test_seed_reproducible(tmp_path):
    events = [
        {
            "instrument": "snare",
            "offset": 0.5,
            "duration": 0.5,
            "velocity_factor": 1,
        }
    ]
    j = tmp_path / "e.json"
    j.write_text(json.dumps(events))
    m1 = tmp_path / "a.mid"
    m2 = tmp_path / "b.mid"
    run_cli(
        [
            str(j),
            "--humanize-timing",
            "20",
            "--humanize-vel",
            "10",
            "--seed",
            "42",
            "--out",
            str(m1),
        ]
    )
    run_cli(
        [
            str(j),
            "--humanize-timing",
            "20",
            "--humanize-vel",
            "10",
            "--seed",
            "42",
            "--out",
            str(m2),
        ]
    )
    pm1 = pretty_midi.PrettyMIDI(str(m1))
    pm2 = pretty_midi.PrettyMIDI(str(m2))
    note1 = pm1.instruments[0].notes[0]
    note2 = pm2.instruments[0].notes[0]
    assert note1.start == note2.start and note1.velocity == note2.velocity
