import xml.etree.ElementTree as ET


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def test_guitar_harmonic_xml(_basic_gen, tmp_path):
    gen = _basic_gen(enable_harmonics=True, prob_harmonic=1.0, rng_seed=42)
    part = gen.compose(section_data=_basic_section())
    path = tmp_path / "g.xml"
    gen.export_musicxml_tab(str(path))
    tree = ET.parse(path)
    root = tree.getroot()
    assert root.findall(".//harmonic")
    strings = [int(e.text) for e in root.findall(".//string")]
    frets = [int(e.text) for e in root.findall(".//fret")]
    assert strings and frets
    first = part.flatten().notes[0]
    meta = getattr(first, "harmonic_meta", None)
    assert meta
    assert meta["string_idx"] + 1 in strings
    assert meta["touch_fret"] in frets


def test_strings_harmonic_xml(_strings_gen, tmp_path):
    gen = _strings_gen(enable_harmonics=True, prob_harmonic=1.0, rng_seed=99)
    parts = gen.compose(section_data=_basic_section())
    vio = parts["violin_i"]
    path = tmp_path / "s.xml"
    gen.export_musicxml(str(path))
    root = ET.parse(path).getroot()
    assert root.findall(".//harmonic")
    strings = [int(e.text) for e in root.findall(".//string")]
    frets = [int(e.text) for e in root.findall(".//fret")]
    assert strings and frets
    first = vio.flatten().notes[0]
    meta = getattr(first, "harmonic_meta", None)
    assert meta
    assert meta["string_idx"] + 1 in strings
    assert meta["touch_fret"] in frets

