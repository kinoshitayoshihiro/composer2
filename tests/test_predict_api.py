from pathlib import Path

from music21 import note, stream, expressions
import torch

from ml_models.articulation_tagger import ArticulationTagger
from utilities import ml_articulation


def test_predict_length(tmp_path: Path) -> None:
    model = ArticulationTagger(num_labels=2)
    ckpt = tmp_path / "m.ckpt"
    torch.save(model.state_dict(), ckpt)
    loaded = ml_articulation.load(ckpt, num_labels=2)
    s = stream.Score()
    p = stream.Part()
    for i in range(4):
        n = note.Note(60 + i)
        n.volume.velocity = 80
        p.append(n)
    s.append(p)
    tags = ml_articulation.predict(s, loaded)
    assert len(tags) == 4

def test_predict_with_trill_and_sustain(tmp_path: Path) -> None:
    model = ArticulationTagger(num_labels=2)
    ckpt = tmp_path / "m.ckpt"
    torch.save(model.state_dict(), ckpt)
    loaded = ml_articulation.load(ckpt, num_labels=2)
    s = stream.Score()
    p = stream.Part()
    n1 = note.Note(60, quarterLength=1.0)
    n1.expressions.append(expressions.Trill())
    n2 = note.Note(62, quarterLength=1.0)
    n2.volume.velocity = 90
    p.append([n1, n2])
    s.append(p)
    tags = ml_articulation.predict(s, loaded)
    assert len(tags) == 2
