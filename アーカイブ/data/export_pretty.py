# utilities/export_pretty.py
from typing import Union, Optional
import music21
import pretty_midi


def stream_to_pretty_midi(
    src: Union[music21.stream.Score, music21.stream.Part],
    default_tempo: int = 120
) -> pretty_midi.PrettyMIDI:
    """
    music21 Stream → PrettyMIDI へ変換する最小関数。
    - instrument.Program は music21 Instrument から推定
    - Drum Part（MIDI ch 10）は Percussion に強制
    """
    pm = pretty_midi.PrettyMIDI()
    tempo_changes = [(0.0, default_tempo)]
    pm._PrettyMIDI__tempo_changes = (
        [t for t, _ in tempo_changes],
        [bpm for _, bpm in tempo_changes],
    )

    # Part（または Score 全体）を列挙
    parts = src.parts if isinstance(src, music21.stream.Score) else [src]

    for idx, part in enumerate(parts):
        # --- Instrument 判定 ---
        inst_m21 = part.getInstrument(returnDefault=True)
        is_drum = isinstance(inst_m21, music21.instrument.Percussion)
        program = 0 if is_drum else inst_m21.midiProgram or 0

        pm_instrument = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=inst_m21.partName or f"Part{idx+1}"
        )

        # --- Note / Chord 変換 ---
        for n in part.recurse().notesAndRests:
            start = float(n.offset)
            dur = float(n.quarterLength)
            if dur <= 0:
                continue

            if n.isRest:
                continue

            # Chord → 各音にバラし
            pitches = n.pitches if isinstance(n, music21.chord.Chord) else [n.pitch]

            velocity = n.volume.velocity or 80

            for p in pitches:
                pm_note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(p.midi),
                    start=start,
                    end=start + dur
                )
                pm_instrument.notes.append(pm_note)

        pm.instruments.append(pm_instrument)

    return pm


def export_pretty_midi(
    src: Union[music21.stream.Score, music21.stream.Part],
    out_path: str,
    tempo: Optional[int] = None
) -> None:
    """Stream から直接 .mid ファイルを書き出すユーティリティ"""
    default_tempo = tempo or int(src.metronomeMarkBoundaries()[0][-1].number)
    pm = stream_to_pretty_midi(src, default_tempo=default_tempo)
    pm.write(out_path)
