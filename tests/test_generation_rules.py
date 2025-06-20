from pathlib import Path

import pretty_midi

from utilities import groove_sampler_v2


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=start, end=start + 0.05))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=42, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def _make_kick_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.05))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=1.0, end=1.05))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=3.0, end=3.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_kick_snare_collision(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    for off in {ev["offset"] for ev in events}:
        insts = [e["instrument"] for e in events if abs(e["offset"] - off) <= 1e-6]
        assert not ("kick" in insts and "snare" in insts)


def test_hihat_shift(tmp_path: Path) -> None:
    _make_loop(tmp_path / "b.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    for ev in events:
        if ev["instrument"] == "kick":
            hats = [
                h
                for h in events
                if h["instrument"].endswith("hh")
                and abs(h["offset"] - ev["offset"]) < 0.011
            ]
            for h in hats:
                assert h["offset"] - ev["offset"] >= 0.002 - 1e-6


def test_cond_velocity_soft(tmp_path: Path) -> None:
    _make_loop(tmp_path / "c.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0, cond_velocity="soft")
    assert max(ev["velocity_factor"] for ev in events) <= 0.8


def test_four_on_floor(tmp_path: Path) -> None:
    _make_kick_loop(tmp_path / "d.mid")
    model = groove_sampler_v2.train(tmp_path)
    bars = 5
    events = groove_sampler_v2.generate_events(model, bars=bars, seed=1, cond_kick="four_on_floor")
    starts = [i * 4 for i in range(bars)]
    kicks = {round(ev["offset"], 6) for ev in events if ev["instrument"] == "kick"}
    count = sum(1 for s in starts if any(abs(k - s) <= 1e-6 for k in kicks))
    assert count / bars >= 0.4
