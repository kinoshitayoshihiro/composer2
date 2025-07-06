from music21 import note, stream

from generator.counter_line import CounterLineGenerator


def test_counterline_basic():
    melody = stream.Part(id="Melody")
    melody.append(note.Note("C4", quarterLength=1.0))
    melody.append(note.Note("G4", quarterLength=1.0, offset=1.0))
    gen = CounterLineGenerator()
    counter = gen.generate(melody)
    notes = list(counter.recurse().notes)
    assert len(notes) == 3
    # First and last notes correspond to melody notes transposed down a fifth then up an octave
    assert notes[0].pitch.midi - melody.notes[0].pitch.midi == 5
    assert notes[-1].pitch.midi - melody.notes[-1].pitch.midi == 5
