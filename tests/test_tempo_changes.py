import sitecustomize  # noqa: F401  # ensure tempo patch
from pretty_midi import PrettyMIDI


def test_tempo_changes() -> None:
    pm = PrettyMIDI()
    pm.set_tempo_changes([120, 140], [0.0, 30.0])
    assert len(pm.get_tempo_changes()[0]) == 2
    assert pm.get_tempo_changes()[0][0] == 120
    assert pm.get_tempo_changes()[0][1] == 140
