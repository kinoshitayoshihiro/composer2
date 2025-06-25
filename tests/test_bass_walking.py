from music21 import instrument, harmony
from generator.bass_generator import BassGenerator


def make_gen():
    cfg = {"global_settings": {"key_tonic": "C", "key_mode": "major"}}
    return BassGenerator(
        part_name="bass",
        part_parameters={},
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg=cfg,
    )


def test_walking_quarters_progression():
    gen = make_gen()
    chords = ["C", "F", "G", "C"]
    part_all = []
    offset = 0.0
    for i, ch in enumerate(chords):
        section = {
            "q_length": 4.0,
            "chord_symbol_for_voicing": ch,
            "part_params": {"bass": {"rhythm_key": "walking_quarters", "velocity": 70}},
        }
        next_sec = {"chord_symbol_for_voicing": chords[i + 1]} if i + 1 < len(chords) else None
        part = gen.compose(section_data=section, next_section_data=next_sec)
        for n in part.flatten().notes:
            part_all.append((offset + n.offset, n, ch))
        offset += 4.0
    # check roots on beats and walking notes
    for start in range(0, int(offset), 4):
        beat_roots = [n for o, n, c in part_all if start <= o < start + 4 and abs(o - start - round(o - start)) < 1e-3]
        assert len(beat_roots) == 4
    for o, n, ch in part_all:
        rel = o % 4
        cs = harmony.ChordSymbol(ch)
        if abs(rel - round(rel)) > 1e-3:
            valid_names = {p.name for p in [cs.root(), cs.third, cs.fifth] if p}
            bar_index = int(o // 4)
            next_root = harmony.ChordSymbol(chords[bar_index + 1]).root() if bar_index + 1 < len(chords) else cs.root()
            if n.pitch.name in valid_names:
                continue
            diff = abs(n.pitch.midi - next_root.midi)
            assert diff in (1, 2)
