import json
from pathlib import Path
import sys

import pretty_midi
from scripts import train_velocity


def _make_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0, end=1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_build_velocity_csv_success(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks"
    drums = tmp_path / "drums"
    tracks.mkdir()
    drums.mkdir()
    _make_midi(tracks / "a.mid")
    _make_midi(drums / "b.mid")
    csv_out = tmp_path / "out.csv"
    stats_out = tmp_path / "stats.csv"

    train_velocity.pretty_midi = pretty_midi
    rc = train_velocity.main([
        "build-velocity-csv",
        "--tracks-dir",
        str(tracks),
        "--drums-dir",
        str(drums),
        "--csv-out",
        str(csv_out),
        "--stats-out",
        str(stats_out),
    ])
    assert rc == 0
    assert csv_out.exists()
    assert stats_out.exists()


def test_build_velocity_csv_error(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    csv_out = tmp_path / "out.csv"
    stats_out = tmp_path / "stats.csv"
    train_velocity.pretty_midi = pretty_midi
    rc = train_velocity.main([
        "build-velocity-csv",
        "--tracks-dir",
        str(tmp_path / "missing"),
        "--drums-dir",
        str(drums),
        "--csv-out",
        str(csv_out),
        "--stats-out",
        str(stats_out),
    ])
    assert rc == 1


def test_csv_path_check(tmp_path: Path) -> None:
    rc = train_velocity.main(["--csv-path", str(tmp_path / "none.csv")])
    assert rc == 1


def test_dry_run_output_formats(capsys) -> None:
    train_velocity.dry_run_flag = True
    train_velocity.dry_run_json = True
    from hydra import compose, initialize_config_dir
    cfg_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), job_name="test", version_base="1.3"):
        cfg = compose(config_name="velocity_model.yaml")
        rc = train_velocity.run(cfg)
    out = capsys.readouterr().out
    assert rc == 0
    assert "input_dim:" in out
    lines = out.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith("{"):
            json_text = "\n".join(lines[idx:])
            break
    data = json.loads(json_text)
    assert "input_dim" in data
