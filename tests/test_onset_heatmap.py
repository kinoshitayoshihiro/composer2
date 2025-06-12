import tempfile
import pretty_midi
from utilities.onset_heatmap import build_heatmap


def test_build_heatmap_counts():
    # create simple MIDI with notes at 0.0s, 0.5s, and 1.0s
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(0)
    for start in [0.0, 0.5, 1.0]:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start, end=start + 0.1))
    pm.instruments.append(inst)

    with tempfile.NamedTemporaryFile(suffix=".mid") as tmp:
        pm.write(tmp.name)
        heatmap = build_heatmap(tmp.name, resolution=16)

    assert heatmap == {0: 2, 8: 1}
