
#!/usr/bin/env python3
# sparkle_convert.py — Convert generic MIDI to UJAM Sparkle-friendly MIDI.
#
# Features
# - Reads an input MIDI and (optionally) a chord CSV/YAML timeline.
# - Emits:
#   1) Long chord notes (triads) in Sparkle’s "Chord" range (configurable octave).
#   2) Steady "common pulse" keys (left-hand phrase trigger) at a chosen subdivision.
#
# Assumptions / Notes
# - UJAM Virtual Guitarist (Sparkle) uses "left hand" keys to trigger patterns/phrases
#   and "chord area" notes to define the chord. Exact note ranges may vary by version
#   and preset. Therefore, ALL layout values are configurable via a mapping YAML.
# - Default time signature is 4/4 if not provided. Tempo is read from the file if present,
#   otherwise --bpm is used.
# - If no chord timeline is provided, a lightweight heuristic infers major/minor triads
#   by bar from active pitch classes.
#
# CLI
#     python sparkle_convert.py IN.mid --out OUT.mid \
#         --pulse 1/8 --chord-octave 4 --phrase-note 36 \
#         --mapping sparkle_mapping.yaml
#
#     # With explicit chord timeline (CSV):
#     python sparkle_convert.py IN.mid --out OUT.mid --pulse 1/8 \
#         --chords chords.csv
#
# Chord CSV format (times in seconds; headers required):
#     start,end,root,quality
#     0.0,2.0,C,maj
#     2.0,4.0,A,min
# Supported qualities: maj, min (others are passed through if triad mapping provided).
#
# Mapping YAML example is created alongside this script as 'sparkle_mapping.example.yaml'.
#
# (c) 2025 — Utility script for MIDI preprocessing. MIT License.

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

try:
    import pretty_midi  # type: ignore
except Exception as e:
    raise SystemExit("This tool requires pretty_midi. Try: pip install pretty_midi") from e

try:
    import yaml  # optional for mapping file
except Exception:
    yaml = None

PITCH_CLASS = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

@dataclass
class ChordSpan:
    start: float
    end: float
    root_pc: int  # 0-11
    quality: str  # 'maj' or 'min' (extendable)

def parse_time_sig(default_num=4, default_den=4) -> Tuple[int,int]:
    # pretty_midi doesn't store TS per track reliably; keep configurable if needed
    return default_num, default_den

def parse_pulse(s: str) -> float:
    '''
    Parse a subdivision string like '1/8' -> 0.5 beats (if a beat is a quarter note).
    We define '1/8' as eighth-notes = 0.5 quarter-beats.
    '''
    s = s.strip()
    if '/' in s:
        num, den = s.split('/', 1)
        num = int(num)
        den = int(den)
        if num != 1:
            raise ValueError("Use forms like 1/8, 1/16, 1/4.")
        # relative to quarter-note = 1 beat
        return 4.0 / den
    else:
        # numeric beats directly
        return float(s)

def triad_pitches(root_pc: int, quality: str, octave: int, mapping: Dict) -> List[int]:
    '''Return MIDI numbers for a simple triad in the given octave based on mapping intervals.'''
    intervals = mapping.get('triad_intervals', {}).get(quality, [0,4,7])  # default maj
    base_c = (octave + 1) * 12  # C-octave base
    return [base_c + ((root_pc + iv) % 12) for iv in intervals]

def load_mapping(path: Optional[Path]) -> Dict:
    default = {
        "phrase_note": 36,          # Default left-hand "Common" phrase key (C2)
        "phrase_velocity": 96,
        "phrase_length_beats": 0.25,
        "chord_octave": 4,          # Place chord tones around C4-B4 by default
        "chord_velocity": 90,
        "triad_intervals": {
            "maj": [0,4,7],
            "min": [0,3,7]
        }
    }
    if path is None:
        return default
    if yaml is None:
        raise SystemExit("PyYAML is required to read mapping files. pip install pyyaml")
    data = yaml.safe_load(Path(path).read_text())
    default.update(data or {})
    return default

def read_chords_csv(path: Path) -> List['ChordSpan']:
    spans: List[ChordSpan] = []
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            start = float(row['start']); end = float(row['end'])
            root = row['root'].strip()
            quality = row['quality'].strip().lower()
            if root not in PITCH_CLASS:
                raise ValueError(f"Unknown root {root}")
            spans.append(ChordSpan(start, end, PITCH_CLASS[root], quality))
    return spans

def infer_chords_by_bar(pm: 'pretty_midi.PrettyMIDI', ts_num=4, ts_den=4) -> List['ChordSpan']:
    # Build a simplistic bar grid from downbeats. If absent, estimate from tempo.
    downbeats = pm.get_downbeats()
    if len(downbeats) < 2:
        # fallback: use beats to construct measures
        beats = pm.get_beats()
        if len(beats) < 2:
            raise ValueError("Cannot infer beats/downbeats from this MIDI; please provide a chord CSV.")
        # take every 4 beats as a bar (assume 4/4)
        downbeats = beats[::ts_num]

    spans: List[ChordSpan] = []
    # Aggregate pitch-class histograms per bar
    for i in range(len(downbeats)):
        start = downbeats[i]
        end = downbeats[i+1] if i+1 < len(downbeats) else pm.get_end_time()
        if end - start <= 0.0:
            continue
        pc_weights = [0.0]*12
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                ns = max(n.start, start)
                ne = min(n.end, end)
                if ne <= ns:
                    continue
                dur = ne - ns
                pc_weights[n.pitch % 12] += dur * (n.velocity/127.0)
        # choose a root candidate
        root_pc = max(range(12), key=lambda pc: pc_weights[pc]) if any(pc_weights) else 0
        # score maj vs min by template match (0,4,7) vs (0,3,7)
        def score(intervals):
            return sum(pc_weights[(root_pc + iv) % 12] for iv in intervals)
        maj_s = score([0,4,7]); min_s = score([0,3,7])
        quality = 'maj' if maj_s >= min_s else 'min'
        spans.append(ChordSpan(start, end, root_pc, quality))
    return spans

def ensure_tempo(pm: 'pretty_midi.PrettyMIDI', fallback_bpm: Optional[float]) -> float:
    tempi = pm.get_tempo_changes()[1]
    if len(tempi):
        return float(tempi[0])
    if fallback_bpm is None:
        return 120.0
    return float(fallback_bpm)

def beats_to_seconds(beats: float, bpm: float) -> float:
    # beats are quarter-notes
    return (60.0 / bpm) * beats

def build_sparkle_midi(pm_in: 'pretty_midi.PrettyMIDI',
                       chords: List['ChordSpan'],
                       mapping: Dict,
                       pulse_subdiv_beats: float,
                       bpm: float) -> 'pretty_midi.PrettyMIDI':
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    chord_inst = pretty_midi.Instrument(program=0, name="Sparkle Chords")  # program ignored by UJAM
    phrase_inst = pretty_midi.Instrument(program=0, name="Sparkle Phrase (Common Pulse)")

    chord_oct = int(mapping.get("chord_octave", 4))
    chord_vel = int(mapping.get("chord_velocity", 90))
    phrase_note = int(mapping.get("phrase_note", 36))
    phrase_vel = int(mapping.get("phrase_velocity", 96))
    phrase_len_beats = float(mapping.get("phrase_length_beats", 0.25))

    for span in chords:
        # Chord triad
        triad = triad_pitches(span.root_pc, span.quality, chord_oct, mapping)
        # make a single long note per pitch across the span
        for p in triad:
            chord_inst.notes.append(pretty_midi.Note(
                velocity=chord_vel, pitch=p, start=span.start, end=span.end
            ))
        # Phrase pulses
        t = span.start
        pulse_len_sec = beats_to_seconds(phrase_len_beats, bpm)
        step_sec = beats_to_seconds(pulse_subdiv_beats, bpm)
        while t < span.end - 1e-6:
            phrase_inst.notes.append(pretty_midi.Note(
                velocity=phrase_vel, pitch=phrase_note, start=t, end=min(t + pulse_len_sec, span.end)
            ))
            t += step_sec

    out.instruments.append(chord_inst)
    out.instruments.append(phrase_inst)
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert generic MIDI to UJAM Sparkle-friendly MIDI (chords + common pulse).")
    ap.add_argument("input_midi", type=str, help="Input MIDI file")
    ap.add_argument("--out", type=str, required=True, help="Output MIDI file")
    ap.add_argument("--pulse", type=str, default="1/8", help="Pulse subdivision (e.g., 1/8, 1/16, 1/4)")
    ap.add_argument("--bpm", type=float, default=None, help="Fallback BPM if input has no tempo")
    ap.add_argument("--chords", type=str, default=None, help="Chord CSV file (start,end,root,quality). If omitted, infer per bar.")
    ap.add_argument("--mapping", type=str, default=None, help="YAML for Sparkle mapping (phrase note, chord octave, velocities, triad intervals).")
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(args.input_midi)

    ts_num, ts_den = parse_time_sig()  # currently fixed 4/4; extend as needed
    bpm = ensure_tempo(pm, args.bpm)
    pulse_beats = parse_pulse(args.pulse)

    mapping = load_mapping(Path(args.mapping) if args.mapping else None)

    if args.chords:
        chords = read_chords_csv(Path(args.chords))
    else:
        chords = infer_chords_by_bar(pm, ts_num, ts_den)

    out_pm = build_sparkle_midi(pm, chords, mapping, pulse_beats, bpm)
    out_pm.write(args.out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
