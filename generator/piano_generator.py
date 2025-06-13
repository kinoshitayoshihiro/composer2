# generator/piano_generator.py (v2.3 最終修正版)

from __future__ import annotations
import logging
import copy
from typing import Dict, Any, Optional, List, Tuple

from music21 import (
    stream,
    note,
    chord as m21chord,
    harmony,
    meter,
    tempo,
    instrument,
    articulations,
    expressions,
    interval,
    volume as m21volume,
    pitch,
)

# Pedalクラスはexpressionsから個別にインポート
from music21.expressions import PedalMark


from .base_part_generator import BasePartGenerator
from utilities import humanizer

try:
    from utilities.core_music_utils import (
        get_time_signature_object,
        sanitize_chord_label,
        MIN_NOTE_DURATION_QL,
    )
    from utilities.override_loader import PartOverride
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Required dependencies are missing. Please run 'pip install -r requirements.txt'."
    ) from e


logger = logging.getLogger("modular_composer.piano_generator")

# (LUTは変更なし)
EMO_TO_BUCKET_PIANO = {
    "quiet_pain": "calm",
    "nascent_strength": "calm",
    "hope_dawn": "calm",
    "emotional_realization": "groove",
    "gratitude": "groove",
    "wavering_heart": "groove",
    "love_and_resolution": "energetic",
    "trial_prayer": "energetic",
    "emotional_storm": "energetic",
    "default": "calm",
}
BUCKET_TO_PATTERN_PIANO = {
    ("calm", "low"): ("piano_rh_ambient_pad", "piano_lh_roots_whole"),
    ("calm", "medium"): ("piano_rh_block_chords_quarters", "piano_lh_roots_half"),
    ("calm", "medium_low"): ("piano_rh_block_chords_quarters", "piano_lh_roots_half"),
    ("calm", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("groove", "low"): ("piano_rh_syncopated_chords_pop", "piano_lh_roots_half"),
    ("groove", "medium"): (
        "piano_rh_syncopated_chords_pop",
        "piano_lh_octaves_quarters",
    ),
    ("groove", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_alberti_bass_eighths",
    ),
    ("energetic", "low"): (
        "piano_rh_block_chords_quarters",
        "piano_lh_octaves_quarters",
    ),
    ("energetic", "medium"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_alberti_bass_eighths",
    ),
    ("energetic", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("default", "medium_low"): (
        "piano_rh_block_chords_quarters",
        "piano_lh_roots_half",
    ),
    ("default", "medium_high"): (
        "piano_rh_syncopated_chords_pop",
        "piano_lh_octaves_quarters",
    ),
    ("default", "high_to_very_high_then_fade"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("default", "default"): ("piano_rh_block_chords_quarters", "piano_lh_roots_whole"),
}


class PianoGenerator(BasePartGenerator):
    def __init__(self, *args, main_cfg=None, **kwargs):
        self.main_cfg = main_cfg
        self.part_parameters = kwargs.get("part_parameters", {})
        self.chord_voicer = None
        ts_obj = get_time_signature_object(kwargs.get("global_time_signature", "4/4"))
        self.global_time_signature_obj = ts_obj
        self.measure_duration = ts_obj.barDuration.quarterLength if ts_obj else 4.0
        super().__init__(*args, **kwargs)
        self.cfg: dict = kwargs.copy()
        # ...他の初期化処理...

    def _get_pattern_keys(
        self, musical_intent: Dict[str, Any], overrides: Optional[PartOverride]
    ) -> Tuple[str, str]:
        if (
            overrides
            and getattr(overrides, "rhythm_key_rh", None)
            and getattr(overrides, "rhythm_key_lh", None)
        ):
            return overrides.rhythm_key_rh, overrides.rhythm_key_lh
        emotion = musical_intent.get("emotion", "default")
        intensity = musical_intent.get("intensity", "medium")
        bucket = EMO_TO_BUCKET_PIANO.get(emotion, "default")
        keys = BUCKET_TO_PATTERN_PIANO.get((bucket, intensity))
        if not keys:
            base_intensity = str(intensity).split("_")[0]
            keys = BUCKET_TO_PATTERN_PIANO.get((bucket, base_intensity))
            if not keys:
                keys = BUCKET_TO_PATTERN_PIANO.get(("default", "default"))
        return keys

    def _render_hand_part(
        self,
        hand: str,
        cs: harmony.ChordSymbol,
        duration_ql: float,
        rhythm_key: str,
        params: Dict[str, Any],
    ) -> stream.Part:
        part = stream.Part(id=f"Piano_{hand}")
        pattern_data = self.part_parameters.get(rhythm_key) or {}
        pattern_events = pattern_data.get("pattern") or []
        if not pattern_events:
            logger.warning(
                f"PianoGen: pattern '{rhythm_key}' not found or empty. Using single root note fallback."
            )
            pattern_events = [
                {
                    "offset": 0.0,
                    "duration": duration_ql,
                    "type": "root",
                    "velocity_factor": 0.7,
                }
            ]

        octave = params.get(
            f"default_{hand.lower()}_target_octave", 4 if hand == "RH" else 2
        )
        num_voices = params.get(
            f"default_{hand.lower()}_num_voices", 4 if hand == "RH" else 2
        )
        voicing_style = params.get(
            f"default_{hand.lower()}_voicing_style",
            "spread_upper" if hand == "RH" else "closed_low",
        )
        voiced_pitches = self._get_voiced_pitches(cs, num_voices, octave, voicing_style)

        if not voiced_pitches:
            part.insert(0, note.Rest(quarterLength=duration_ql))
            return part

        base_velocity = params.get("velocity", 70)
        ref_duration = float(pattern_data.get("length_beats", self.measure_duration))
        scale_factor = duration_ql / ref_duration if ref_duration > 0 else 1.0

        for p_event in pattern_events:
            offset = float(p_event.get("offset", 0.0)) * scale_factor
            duration = float(p_event.get("duration", 1.0)) * scale_factor
            if offset >= duration_ql:
                continue
            duration = min(duration, duration_ql - offset)
            if duration < MIN_NOTE_DURATION_QL / 4:
                continue
            vel_factor = float(p_event.get("velocity_factor", 1.0))
            velocity = int(base_velocity * vel_factor)
            event_type = p_event.get("type", "chord")
            element_to_add = self._create_music_element(
                event_type, voiced_pitches, duration, hand
            )
            if element_to_add:
                element_to_add.volume = m21volume.Volume(velocity=velocity)
                part.insert(offset, element_to_add)
        return part

    def _get_voiced_pitches(self, cs, num_voices, octave, style) -> List[pitch.Pitch]:
        if self.chord_voicer:
            try:
                return self.chord_voicer.voice_chord(
                    cs, style=style, num_voices=num_voices, target_octave=octave
                )
            except Exception as e:
                logger.warning(
                    f"ChordVoicer failed: {e}. Falling back to basic voicing."
                )

        cs_copy = copy.deepcopy(cs)
        cs_copy.closedPosition(inPlace=True)
        if not cs_copy.pitches:
            return []
        bottom_pitch = cs_copy.pitches[0]

        target_pitch_ref = note.Note(bottom_pitch.name, octave=octave)
        transpose_interval = interval.Interval(bottom_pitch, target_pitch_ref)
        cs_copy.transpose(transpose_interval, inPlace=True)

        pitches = list(cs_copy.pitches)
        return (
            pitches[:num_voices]
            if num_voices is not None and len(pitches) > num_voices
            else pitches
        )

    def _create_music_element(
        self, event_type: str, pitches: List[pitch.Pitch], duration_ql: float, hand: str
    ) -> Optional[stream.GeneralNote]:
        if not pitches:
            return None
        if event_type in ("octave_root", "octave"):
            root_pitch = min(pitches, key=lambda p: p.ps)
            return m21chord.Chord(
                [root_pitch, root_pitch.transpose(12)], quarterLength=duration_ql
            )
        elif event_type == "root":
            root_pitch = min(pitches, key=lambda p: p.ps)
            return note.Note(root_pitch, quarterLength=duration_ql)
        elif event_type == "chord_high_voices":
            high_pitches = (
                sorted(pitches, key=lambda p: p.ps)[-3:]
                if len(pitches) >= 3
                else pitches
            )
            return m21chord.Chord(high_pitches, quarterLength=duration_ql)
        elif event_type == "chord":
            return m21chord.Chord(pitches, quarterLength=duration_ql)
        else:
            logger.warning(
                f"Unknown pattern event type: '{event_type}'. Using default for {hand}."
            )
            if hand == "RH":
                return m21chord.Chord(pitches, quarterLength=duration_ql)
            else:
                root_pitch = min(pitches, key=lambda p: p.ps)
                return note.Note(root_pitch, quarterLength=duration_ql)

    def _insert_pedal(self, part: stream.Part, section_data: Dict[str, Any]):
        intensity = section_data.get("musical_intent", {}).get("intensity", "medium")
        end_offset = section_data.get("q_length", 4.0)
        sustain_beats = {"low": 4.0, "medium": 2.0, "high": 1.0}.get(intensity, 2.0)
        if not part.flatten().notes:
            return
        t = 0.0
        while t < end_offset:
            ped = PedalMark()  # ← Pedal() から PedalMark() へ
            ped.type = "start"
            part.insert(t, ped)
            t += sustain_beats

    def _apply_weak_beat(self, part: stream.Part, style: str) -> stream.Part:
        if style in (None, "none"):
            return part
        beats = (
            self.global_time_signature_obj.beatCount
            if self.global_time_signature_obj
            else 4
        )
        beat_len = (
            self.global_time_signature_obj.beatDuration.quarterLength
            if self.global_time_signature_obj
            else 1.0
        )
        new_part = stream.Part(id=part.id)
        for elem in part.flatten().notesAndRests:
            rel = elem.offset
            beat_pos = rel / beat_len
            is_on_beat = abs(beat_pos - round(beat_pos)) < 0.001
            beat_idx = int(round(beat_pos))
            is_weak = False
            if is_on_beat:
                if beats == 4 and (beat_idx == 1 or beat_idx == 3):
                    is_weak = True
                elif beats == 3 and (beat_idx == 1 or beat_idx == 2):
                    is_weak = True
            if is_weak:
                if style == "rest":
                    continue
                elif style == "ghost":
                    base_vel = (
                        elem.volume.velocity if elem.volume and elem.volume.velocity else 64
                    )
                    elem.volume = m21volume.Volume(velocity=max(1, int(base_vel * 0.4)))
            new_part.insert(elem.offset, elem)
        return new_part

    def _add_fill(self, part: stream.Part, cs: harmony.ChordSymbol, length_beats: float) -> stream.Part:
        if length_beats <= 0:
            return part
        offset = self.measure_duration - length_beats
        if offset < 0:
            return part
        fill_chord = m21chord.Chord(cs.pitches, quarterLength=length_beats)
        fill_chord.volume = m21volume.Volume(velocity=80)
        part.insert(offset, fill_chord)
        return part

    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        piano_part = stream.Part(id=f"Piano_Part_{section_data.get('absolute_offset')}")
        chord_label = section_data.get("chord_symbol_for_voicing", "Rest")

        if not chord_label or chord_label == "Rest":
            piano_part.insert(
                0, note.Rest(quarterLength=section_data.get("q_length", 4.0))
            )
            return piano_part
        try:
            cs = harmony.ChordSymbol(chord_label)
        except Exception as e:
            logger.error(f"Failed to parse chord '{chord_label}': {e}. Inserting Rest.")
            piano_part.insert(
                0, note.Rest(quarterLength=section_data.get("q_length", 4.0))
            )
            return piano_part

        duration_ql = section_data.get("q_length", 4.0)
        musical_intent = section_data.get("musical_intent", {})
        piano_params = section_data.get("part_params", {}).get("piano", {})
        rh_key, lh_key = self._get_pattern_keys(musical_intent, self.overrides)

        rh_part = self._render_hand_part("RH", cs, duration_ql, rh_key, piano_params)
        lh_part = self._render_hand_part("LH", cs, duration_ql, lh_key, piano_params)

        ov = self.overrides.model_dump(exclude_unset=True) if self.overrides else {}
        rh_part = self._apply_weak_beat(rh_part, ov.get("weak_beat_style_rh", "none"))
        lh_part = self._apply_weak_beat(lh_part, ov.get("weak_beat_style_lh", "none"))
        if ov.get("fill_on_4th"):
            fill_len = float(ov.get("fill_length_beats", 0.5))
            rh_part = self._add_fill(rh_part, cs, fill_len)

        for element in rh_part.flatten().notesAndRests:
            element.stemDirection = "up"
            piano_part.insert(element.offset, element)
        for element in lh_part.flatten().notesAndRests:
            element.stemDirection = "down"
            piano_part.insert(element.offset, element)

        self._insert_pedal(piano_part, section_data)
        piano_part.insert(0, self.default_instrument)

        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(piano_part, profile_name)

        global_profile = self.cfg.get(
            "global_humanize_profile"
        ) or self.global_settings.get("global_humanize_profile")
        if global_profile:
            humanizer.apply(piano_part, global_profile)

        return piano_part
