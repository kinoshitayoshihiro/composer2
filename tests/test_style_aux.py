import yaml
import pretty_midi
from pathlib import Path
from utilities import groove_sampler_v2


def _make_loop(path: Path, hits: list[float], style: str) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for h in hits:
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=h * 0.25, end=h * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))
    with open(path.with_suffix(".meta.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"style": style}, fh)


def test_style_aux_sampling(tmp_path: Path) -> None:
    _make_loop(tmp_path / "lofi.mid", [0, 8], "lofi")
    _make_loop(tmp_path / "funk.mid", list(range(8)), "funk")
    model = groove_sampler_v2.train(tmp_path, aux_key="style")
    ev_lofi = groove_sampler_v2.style_aux_sampling(model, bars=1, cond={"style": "lofi"})
    ev_funk = groove_sampler_v2.style_aux_sampling(model, bars=1, cond={"style": "funk"})
    assert len(ev_lofi) <= 4
    assert len(ev_funk) >= 8
    ev_mix = groove_sampler_v2.style_aux_sampling(model, bars=1)
    assert 4 <= len(ev_mix) <= 8
