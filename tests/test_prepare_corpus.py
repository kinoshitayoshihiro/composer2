import argparse
import json
import logging
from pathlib import Path

import pytest
yaml = pytest.importorskip("yaml")

pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")
mido = pytest.importorskip("mido")

from tools.prepare_transformer_corpus import (
    build_corpus,
    tokenize_notes,
    build_beat_map,
    gather_midi_files,
    bin_duration,
    load_tag_maps,
)
from utilities.pretty_midi_safe import pm_to_mido


def make_midi(path: Path, *, pitch: int = 60, velocity: int = 80) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # 2 seconds at 120 BPM -> 4 beats (one bar)
    inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=0, end=2))
    pm.instruments.append(inst)
    pm.write(path.as_posix())


def test_bin_duration_numpy_import() -> None:
    assert bin_duration(0.5, 4) == 2


def test_missing_tag_file_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = tmp_path / "tags.yaml"
    good.write_text(yaml.safe_dump({"song.mid": {"mood": "sad"}}))
    missing = tmp_path / "missing.yaml"
    with caplog.at_level(logging.WARNING):
        tag_map = load_tag_maps([good, missing])
    assert tag_map["song.mid"]["mood"] == "sad"
    assert "tag file" in caplog.text


def test_load_tags_without_yaml(monkeypatch, caplog):
    monkeypatch.setattr("tools.prepare_transformer_corpus.yaml", None)
    with caplog.at_level(logging.WARNING):
        tag_map = load_tag_maps([Path("dummy.yaml")])
    assert tag_map == {}
    assert "PyYAML" in caplog.text


def test_tokenize_duv_and_standard() -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0, end=1))
    pm.instruments.append(inst)
    tokens_std = tokenize_notes(inst.notes, duv=False, dur_bins=4, vel_bins=4, quant=480)
    assert any(t.startswith("D_") for t in tokens_std)
    assert any(t.startswith("V_") for t in tokens_std)
    tokens_duv = tokenize_notes(inst.notes, duv=True, dur_bins=4, vel_bins=4, quant=480)
    assert any(t.startswith("DUV_") for t in tokens_duv)
    assert not any(t.startswith("D_") for t in tokens_duv)


def test_tag_merge(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    mid_path = midi_dir / "song.mid"
    make_midi(mid_path)
    tag_file = tmp_path / "tags.yaml"
    tag_file.write_text(yaml.safe_dump({"song.mid": {"mood": "happy"}}))
    args = argparse.Namespace(
        in_dir=str(midi_dir),
        bars_per_sample=1,
        quant=480,
        min_notes=1,
        duv="off",
        dur_bins=4,
        vel_bins=4,
        tags=[str(tag_file)],
        lyric_json=None,
        split=(1.0, 0.0, 0.0),
        seed=0,
        section_tokens=False,
        mood_tokens=False,
        include_programs=None,
        drums_only=False,
        exclude_drums=False,
        max_files=None,
        max_samples_per_file=None,
        progress=False,
        num_workers=1,
    )
    files = gather_midi_files(Path(args.in_dir))
    splits, _, _ = build_corpus(args, files)
    assert splits["train"][0].meta["mood"] == "happy"


def test_deterministic_split(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    for i in range(4):
        make_midi(midi_dir / f"f{i}.mid", pitch=60 + i)
    args = argparse.Namespace(
        in_dir=str(midi_dir),
        bars_per_sample=1,
        quant=480,
        min_notes=1,
        duv="off",
        dur_bins=4,
        vel_bins=4,
        tags=[],
        lyric_json=None,
        split=(0.5, 0.25, 0.25),
        seed=123,
        section_tokens=False,
        mood_tokens=False,
        include_programs=None,
        drums_only=False,
        exclude_drums=False,
        max_files=None,
        max_samples_per_file=None,
        progress=False,
        num_workers=1,
    )
    files = gather_midi_files(Path(args.in_dir))
    s1, _, _ = build_corpus(args, files)
    s2, _, _ = build_corpus(args, files)
    assert [s.tokens for s in s1["train"]] == [s.tokens for s in s2["train"]]
    args.seed = 321
    s3, _, _ = build_corpus(args, files)
    assert [s.tokens for s in s1["train"]] != [s.tokens for s in s3["train"]]


def test_pm_to_mido_fallback() -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0, end=1))
    pm.instruments.append(inst)
    mid = pm_to_mido(pm)
    import mido  # local import for assertion

    assert isinstance(mid, mido.MidiFile)


def test_note_ordering_stable() -> None:
    notes = [
        pretty_midi.Note(velocity=80, pitch=62, start=1.0, end=2.0),
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0),
    ]
    t1 = tokenize_notes(notes, duv=False, dur_bins=4, vel_bins=4, quant=4)
    t2 = tokenize_notes(list(reversed(notes)), duv=False, dur_bins=4, vel_bins=4, quant=4)
    assert t1 == t2


def test_non_four_four_timesig(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    mid_path = midi_dir / "ts.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=6, denominator=8, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=62, velocity=80, time=960))
    track.append(mido.Message("note_off", note=62, velocity=0, time=1440))
    mid.save(mid_path.as_posix())
    args = argparse.Namespace(
        in_dir=str(midi_dir),
        bars_per_sample=1,
        quant=4,
        min_notes=1,
        duv="off",
        dur_bins=4,
        vel_bins=4,
        tags=[],
        lyric_json=None,
        split=(1.0, 0.0, 0.0),
        seed=0,
        section_tokens=False,
        mood_tokens=False,
        include_programs=None,
        drums_only=False,
        exclude_drums=False,
        max_files=None,
        max_samples_per_file=None,
        progress=False,
        num_workers=1,
    )
    files = gather_midi_files(Path(args.in_dir))
    splits, _, _ = build_corpus(args, files)
    assert len(splits["train"]) == 1
    meta = splits["train"][0].meta
    assert meta["beats_per_bar"] == 3.0
    assert meta["time_signature"] == "6/8"


def test_variable_tempo_snap(tmp_path: Path) -> None:
    mid_path = tmp_path / "tempo.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=80, time=480))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=0))
    track.append(mido.Message("note_on", note=62, velocity=80, time=480))
    track.append(mido.Message("note_off", note=62, velocity=80, time=480))
    mid.save(mid_path.as_posix())
    pm = pretty_midi.PrettyMIDI(mid_path.as_posix())
    sec_to_beats, _ = build_beat_map(pm)
    assert sec_to_beats(1.5) == pytest.approx(2.0)


def test_instrument_filters(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    mid_path = midi_dir / "multi.mid"
    pm = pretty_midi.PrettyMIDI()
    inst_piano = pretty_midi.Instrument(program=0)
    inst_piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0, end=2))
    inst_guitar = pretty_midi.Instrument(program=24)
    inst_guitar.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0, end=2))
    inst_drum = pretty_midi.Instrument(program=0, is_drum=True)
    inst_drum.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0, end=2))
    pm.instruments.extend([inst_piano, inst_guitar, inst_drum])
    pm.write(mid_path.as_posix())
    base = dict(
        in_dir=str(midi_dir),
        bars_per_sample=1,
        quant=4,
        min_notes=1,
        duv="off",
        dur_bins=4,
        vel_bins=4,
        tags=[],
        lyric_json=None,
        split=(1.0, 0.0, 0.0),
        seed=0,
        section_tokens=False,
        mood_tokens=False,
        max_files=None,
        max_samples_per_file=None,
        progress=False,
        num_workers=1,
    )
    args = argparse.Namespace(include_programs=[0], drums_only=False, exclude_drums=False, **base)
    files = gather_midi_files(Path(args.in_dir))
    splits, _, _ = build_corpus(args, files)
    toks = splits["train"][0].tokens
    assert any("NOTE_60" == t for t in toks)
    assert all("NOTE_64" != t for t in toks)
    args = argparse.Namespace(include_programs=None, drums_only=True, exclude_drums=False, **base)
    files = gather_midi_files(Path(args.in_dir))
    splits, _, _ = build_corpus(args, files)
    toks = splits["train"][0].tokens
    assert any("NOTE_36" == t for t in toks)
    args = argparse.Namespace(include_programs=None, drums_only=False, exclude_drums=True, **base)
    files = gather_midi_files(Path(args.in_dir))
    splits, _, _ = build_corpus(args, files)
    toks = splits["train"][0].tokens
    assert all("NOTE_36" != t for t in toks)


def test_offline_embed_and_duv_oov(tmp_path: Path, caplog) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    make_midi(midi_dir / "a.mid")
    make_midi(midi_dir / "b.mid", pitch=65)
    bad_map = {"a.mid": [0.0, 0.1], "b.mid": [0.2]}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad_map))
    base_args = dict(
        in_dir=str(midi_dir),
        bars_per_sample=1,
        quant=480,
        min_notes=1,
        duv="on",
        dur_bins=4,
        vel_bins=4,
        tags=[],
        lyric_json=None,
        split=(1.0, 0.0, 0.0),
        seed=0,
        section_tokens=False,
        mood_tokens=False,
        include_programs=None,
        drums_only=False,
        exclude_drums=False,
        max_files=None,
        max_samples_per_file=None,
        progress=False,
        num_workers=2,
        embed_offline=str(bad_path),
    )
    with pytest.raises(ValueError):
        files = gather_midi_files(Path(base_args["in_dir"]))
        build_corpus(argparse.Namespace(**base_args), files)
    good_map = {"a.mid": [0.0, 0.1], "b.mid": [0.2, 0.3]}
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good_map))
    base_args["embed_offline"] = str(good_path)
    base_args["duv_max"] = 0
    with caplog.at_level(logging.INFO):
        files = gather_midi_files(Path(base_args["in_dir"]))
        splits, _, _ = build_corpus(argparse.Namespace(**base_args), files)
    sample = splits["train"][0]
    assert sample.meta["text_emb"] == [0.0, 0.1]
    assert "DUV_OOV" in sample.tokens
    assert any("offline embeddings" in r.message for r in caplog.records)
    assert any("DUV kept" in r.message and "collapsed" in r.message for r in caplog.records)


def test_lyrics_normalization_and_compress(tmp_path: Path, caplog) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    a = midi_dir / "A.mid"
    b = midi_dir / "b.mid"
    make_midi(a)
    make_midi(b, pitch=65)
    lyrics = {str(a.resolve()): "la", "B.MID": "lb"}
    lyr_path = tmp_path / "lyrics.json"
    lyr_path.write_text(json.dumps(lyrics))
    out_dir = tmp_path / "out"

    args = [
        "--in",
        str(midi_dir),
        "--out",
        str(out_dir),
        "--bars-per-sample",
        "1",
        "--quant",
        "480",
        "--min-notes",
        "1",
        "--duv",
        "off",
        "--dur-bins",
        "4",
        "--vel-bins",
        "4",
        "--split",
        "0.5",
        "0.5",
        "0.0",
        "--lyric-json",
        str(lyr_path),
        "--compress",
        "gz",
    ]
    from tools.prepare_transformer_corpus import main
    with caplog.at_level(logging.INFO):
        main(args)
    assert any("files_scanned" in r.message for r in caplog.records)
    caplog.clear()

    train_path = out_dir / "train.jsonl.gz"
    valid_path = out_dir / "valid.jsonl.gz"
    assert train_path.exists() and valid_path.exists()

    import gzip

    train_lines = [json.loads(l) for l in gzip.open(train_path, "rt", encoding="utf-8")]
    valid_lines = [json.loads(l) for l in gzip.open(valid_path, "rt", encoding="utf-8")]
    total = len(train_lines) + len(valid_lines)
    meta = json.loads((out_dir / "meta.json").read_text())
    assert total == meta["stats"]["train"] + meta["stats"]["valid"] + meta["stats"]["test"]
    assert any("text" in l["meta"] for l in train_lines + valid_lines)

    out_dir2 = tmp_path / "out2"
    args_no_gz = args.copy()
    args_no_gz[3] = str(out_dir2)
    args_no_gz[-1] = "none"
    with caplog.at_level(logging.INFO):
        main(args_no_gz)
    import gzip as gz2
    with gz2.open(train_path, "rt", encoding="utf-8") as fh1, open(
        out_dir2 / "train.jsonl", "r", encoding="utf-8"
    ) as fh2:
        assert fh1.read() == fh2.read()


def test_invalid_cli_args(tmp_path: Path) -> None:
    from tools import prepare_transformer_corpus as mod

    with pytest.raises(SystemExit, match="duv-max"):
        mod.main([
            "--in",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--duv-max",
            "0",
        ])
    with pytest.raises(SystemExit, match="split"):
        mod.main([
            "--in",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--split",
            "0.3",
            "0.3",
            "0.3",
        ])
