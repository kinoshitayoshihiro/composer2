# --- START OF FILE generator/melody_generator.py (修正・ブラッシュアップ版) ---
from __future__ import annotations

"""melody_generator.py – *lightweight rewrite*
... (docstringは変更なし) ...
"""
import copy
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any

import music21.harmony as harmony
import music21.instrument as m21instrument  # 指摘された形式
import music21.key as key
import music21.meter as meter
import music21.note as note

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.tempo as tempo
import music21.volume as m21volume

from .base_part_generator import BasePartGenerator

# melody_utils と humanizer をインポート
try:
    from utilities.core_music_utils import (
        MIN_NOTE_DURATION_QL,
        get_time_signature_object,
        sanitize_chord_label,
    )
    from utilities.humanizer import HUMANIZATION_TEMPLATES, apply_humanization_to_part

    from .melody_utils import generate_melodic_pitches

    try:
        from cyext import insert_melody_notes as cy_insert_melody_notes
        from cyext import velocity_random_walk as cy_velocity_random_walk
    except Exception:
        cy_velocity_random_walk = None
        cy_insert_melody_notes = None
except ImportError as e:
    logger_fallback = logging.getLogger(__name__ + ".fallback_utils")
    logger_fallback.error(
        f"MelodyGenerator: Failed to import required modules (melody_utils or humanizer or core_music_utils): {e}"
    )

    def generate_melodic_pitches(*args, **kwargs) -> list[note.Note]:
        return []

    def apply_humanization_to_part(
        part, *args, **kwargs
    ) -> stream.Part:  # stream.Part を返すように修正
        # このダミー関数は stream.Part を返す必要があるため、引数の part をそのまま返すか、
        # 新しい stream.Part を生成して返す
        if isinstance(part, stream.Part):
            return part
        return stream.Part()  # 新しい空のPartを返す

    MIN_NOTE_DURATION_QL = 0.125

    def get_time_signature_object(ts_str: str | None) -> meter.TimeSignature:
        return meter.TimeSignature("4/4")

    def sanitize_chord_label(label: str | None) -> str | None:
        if not label or label.strip().lower() in ["rest", "n.c.", "nc", "none"]:
            return None
        return label.strip()

    HUMANIZATION_TEMPLATES = {}


logger = logging.getLogger(__name__)

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


class MelodyGenerator(BasePartGenerator):
    def __init__(
        self,
        *args,
        global_settings: dict = None,  # ★追加
        role: str = "melody",
        instrument_name: str = "Soprano Sax",
        apply_pedal: bool = False,
        rhythm_library: dict[str, dict] | None = None,
        default_instrument=m21instrument.Flute(),
        global_tempo: int = 100,
        global_time_signature: str = "4/4",
        global_key_signature_tonic: str = "C",
        global_key_signature_mode: str = "major",
        key: str | tuple[str, str] | None = None,
        tempo: float | None = None,
        emotion: str | None = None,
        rng: random.Random | None = None,
        **kwargs,
    ):
        if args:
            warnings.warn(
                "Positional arguments are deprecated; use keyword arguments",
                DeprecationWarning,
                stacklevel=2,
            )
            arg_names = [
                "global_settings",
                "default_instrument",
                "global_tempo",
                "global_time_signature",
                "global_key_signature_tonic",
                "global_key_signature_mode",
            ]
            for name, val in zip(arg_names, args):
                if name == "global_settings":
                    global_settings = val
                elif name == "default_instrument":
                    default_instrument = val
                elif name == "global_tempo":
                    global_tempo = val
                elif name == "global_time_signature":
                    global_time_signature = val
                elif name == "global_key_signature_tonic":
                    global_key_signature_tonic = val
                elif name == "global_key_signature_mode":
                    global_key_signature_mode = val
        from music21.instrument import fromString

        self.role = role
        self.instrument_obj = fromString(instrument_name)
        self.apply_pedal = apply_pedal
        self.rhythm_library = rhythm_library if rhythm_library else {}
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        self.global_time_signature_obj = get_time_signature_object(
            global_time_signature
        )
        self.global_key_tonic = global_key_signature_tonic
        self.global_key_mode = global_key_signature_mode
        if rng is None:
            rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()
        self.rng = rng

        # ここで親クラスの初期化（global_settingsを渡す）
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument,
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            key=key,
            tempo=tempo,
            emotion=emotion,
            rng=rng,
            **kwargs,
        )

        self.cfg: dict = kwargs.copy()
        if "default_melody_rhythm" not in self.rhythm_library:
            self.rhythm_library["default_melody_rhythm"] = {
                "description": "Default melody rhythm - quarter notes",
                "pattern": [0.0, 1.0, 2.0, 3.0],
                "note_duration_ql": 1.0,
                "reference_duration_ql": 4.0,
            }
            logger.info(
                "MelodyGenerator: Added 'default_melody_rhythm' to rhythm_library."
            )

    def _get_rhythm_details(self, rhythm_key: str) -> dict[str, Any]:
        default_rhythm = self.rhythm_library.get(
            "default_melody_rhythm",
            {
                "pattern": [0.0, 1.0, 2.0, 3.0],
                "note_duration_ql": 1.0,
                "reference_duration_ql": 4.0,
            },
        )
        details = self.rhythm_library.get(rhythm_key, default_rhythm)
        if "pattern" not in details or not isinstance(details["pattern"], list):
            logger.warning(
                f"MelodyGen: Rhythm key '{rhythm_key}' invalid or missing 'pattern' (offsets list). Using default."
            )
            return default_rhythm
        if "note_duration_ql" not in details:
            details["note_duration_ql"] = default_rhythm.get("note_duration_ql", 1.0)
        if "reference_duration_ql" not in details:
            details["reference_duration_ql"] = default_rhythm.get(
                "reference_duration_ql", 4.0
            )
        return details

    def compose(
        self, section_data=None, *, vocal_metrics: dict | None = None
    ) -> stream.Part:
        """Generate melody for the given section.

        This generator implements its own compose logic instead of calling
        :meth:`BasePartGenerator.compose`, so ``vocal_metrics`` is consumed
        directly here when needed.
        """
        # section_data を processed_blocks のように扱う
        if section_data is None:
            processed_blocks = []
        elif isinstance(section_data, list):
            processed_blocks = section_data
        else:
            processed_blocks = [section_data]

        melody_part = stream.Part(id="Melody")
        # ここで instrument_obj を必ず先頭に挿入
        melody_part.insert(0, self.instrument_obj)
        melody_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        melody_part.insert(0, copy.deepcopy(self.global_time_signature_obj))

        first_block_tonic = (
            processed_blocks[0].get("tonic_of_section", self.global_key_tonic)
            if processed_blocks
            else self.global_key_tonic
        )
        first_block_mode = (
            processed_blocks[0].get("mode", self.global_key_mode)
            if processed_blocks
            else self.global_key_mode
        )
        melody_part.insert(0, key.Key(first_block_tonic, first_block_mode))

        current_total_offset = 0.0

        for blk_idx, blk_data in enumerate(processed_blocks):
            melody_params = blk_data.get("part_params", {}).get("melody", {})
            if melody_params.get("skip", False):
                logger.debug(
                    f"MelodyGenerator: Skipping melody for block {blk_idx+1} due to 'skip' flag."
                )
                current_total_offset += blk_data.get("q_length", 0.0)
                continue

            chord_label_str = blk_data.get("chord_label", "C")
            block_q_length = blk_data.get("q_length", 4.0)

            cs_m21_obj: harmony.ChordSymbol | None = None  # 初期化
            try:
                sanitized_label = sanitize_chord_label(chord_label_str)
                if sanitized_label is None:
                    cs_m21_obj = None
                else:
                    cs_m21_obj = harmony.ChordSymbol(sanitized_label)

                if cs_m21_obj is None or not cs_m21_obj.pitches:
                    logger.warning(
                        f"MelodyGenerator: Could not parse chord '{chord_label_str}' or no pitches for block {blk_idx+1}. Skipping melody notes."
                    )
                    current_total_offset += block_q_length
                    continue
            except Exception as e_chord_parse:
                logger.warning(
                    f"MelodyGenerator: Error parsing chord '{chord_label_str}' for block {blk_idx+1}: {e_chord_parse}. Skipping."
                )
                current_total_offset += block_q_length
                continue

            # cs_m21_obj が None でないことをここで保証
            if cs_m21_obj is None:  # このチェックは論理的には不要だが、念のため
                current_total_offset += block_q_length
                continue

            tonic_for_block = blk_data.get("tonic_of_section", self.global_key_tonic)
            mode_for_block = blk_data.get("mode", self.global_key_mode)

            rhythm_key_for_block = melody_params.get(
                "rhythm_key", "default_melody_rhythm"
            )
            rhythm_details = self._get_rhythm_details(rhythm_key_for_block)

            beat_offsets_template = rhythm_details.get("pattern", [0.0, 1.0, 2.0, 3.0])
            base_note_duration_ql = rhythm_details.get(
                "note_duration_ql", melody_params.get("note_duration_ql", 0.5)
            )
            template_reference_duration = rhythm_details.get(
                "reference_duration_ql", 4.0
            )

            stretch_factor = 1.0
            if template_reference_duration > 0:
                stretch_factor = block_q_length / template_reference_duration

            final_beat_offsets_for_block = [
                tpl_off * stretch_factor
                for tpl_off in beat_offsets_template
                if (tpl_off * stretch_factor) < block_q_length
            ]
            if not final_beat_offsets_for_block and beat_offsets_template:
                if (beat_offsets_template[0] * stretch_factor) < block_q_length:
                    final_beat_offsets_for_block = [
                        beat_offsets_template[0] * stretch_factor
                    ]

            octave_range_for_block = tuple(melody_params.get("octave_range", [4, 5]))

            generated_notes = generate_melodic_pitches(
                chord=cs_m21_obj,
                tonic=tonic_for_block,
                mode=mode_for_block,
                beat_offsets=final_beat_offsets_for_block,
                octave_range=octave_range_for_block,
                rnd=self.rng,
                min_note_duration_ql=MIN_NOTE_DURATION_QL,
            )

            density_for_block = melody_params.get("density", 0.7)
            note_velocity = melody_params.get("velocity", 80)

            offsets: list[float] = []
            durs: list[float] = []
            kept: list[note.Note] = []
            for idx, n_obj in enumerate(generated_notes):
                if self.rng.random() > density_for_block:
                    continue
                note_start_offset_in_block = final_beat_offsets_for_block[idx]
                if idx < len(final_beat_offsets_for_block) - 1:
                    next_note_start_offset_in_block = final_beat_offsets_for_block[
                        idx + 1
                    ]
                    max_dur = (
                        next_note_start_offset_in_block - note_start_offset_in_block
                    )
                else:
                    max_dur = block_q_length - note_start_offset_in_block
                actual_dur = (
                    max(
                        MIN_NOTE_DURATION_QL,
                        min(
                            max_dur,
                            (
                                base_note_duration_ql * stretch_factor
                                if "stretch_factor" in locals()
                                else base_note_duration_ql
                            ),
                        ),
                    )
                    * 0.95
                )
                offsets.append(current_total_offset + note_start_offset_in_block)
                durs.append(actual_dur)
                kept.append(n_obj)

            if kept:
                if cy_insert_melody_notes is not None:
                    cy_insert_melody_notes(
                        melody_part,
                        kept,
                        offsets,
                        durs,
                        int(note_velocity),
                        density_for_block,
                        self.rng,
                    )
                else:
                    for n_obj, off, dur in zip(kept, offsets, durs):
                        n_obj.quarterLength = dur
                        n_obj.volume = m21volume.Volume(velocity=note_velocity)
                        melody_part.insert(off, n_obj)

            current_total_offset += block_q_length

        global_melody_params = (
            processed_blocks[0].get("part_params", {}).get("melody", {})
            if processed_blocks
            else {}
        )
        if global_melody_params.get(
            "melody_humanize", global_melody_params.get("humanize", False)
        ):
            h_template_mel = global_melody_params.get(
                "melody_humanize_style_template",
                global_melody_params.get("humanize_style_template", "default_subtle"),
            )
            h_custom_mel = {
                k.replace("melody_humanize_", "").replace("humanize_", ""): v
                for k, v in global_melody_params.items()
                if (k.startswith("melody_humanize_") or k.startswith("humanize_"))
                and not k.endswith("_template")
                and not k.endswith("humanize")
                and not k.endswith("_opt")
            }
            logger.info(
                f"MelodyGenerator: Applying humanization with template '{h_template_mel}' and params {h_custom_mel}"
            )
            melody_part = apply_humanization_to_part(
                melody_part, template_name=h_template_mel, custom_params=h_custom_mel
            )
            melody_part.id = "Melody"
            if not melody_part.getElementsByClass(m21instrument.Instrument).first():
                melody_part.insert(0, self.default_instrument)
            if not melody_part.getElementsByClass(tempo.MetronomeMark).first():
                melody_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
            if not melody_part.getElementsByClass(meter.TimeSignature).first():
                melody_part.insert(0, copy.deepcopy(self.global_time_signature_obj))
            if not melody_part.getElementsByClass(key.Key).first():
                melody_part.insert(0, key.Key(first_block_tonic, first_block_mode))

        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            apply_humanization_to_part(melody_part, template_name=profile_name)

        return melody_part

    def _render_part(self, *args, **kwargs):
        raise NotImplementedError("_render_part is not implemented in MelodyGenerator.")

    def write(
        self,
        part: stream.Part,
        project_root: str | Path,
        section: str,
        filename: str | None = None,
    ) -> Path:
        """Write ``part`` under ``project_root/section`` as MIDI.

        The directory is created if needed.
        """
        project_root = Path(project_root)
        section_path = project_root / section
        os.makedirs(section_path, exist_ok=True)
        fname = filename or f"{self.part_name or 'melody'}.mid"
        out_path = section_path / fname
        part.write("midi", fp=str(out_path))
        return out_path


# --- Debug / manual test (REMOVE IN PRODUCTION) -----------------
if __name__ == "__main__":
    # ここにテスト用の score, out_path の定義が必要です
    # 例:
    # score = stream.Score()
    # out_path = "test_output.mid"
    # score.write("midi", fp=out_path)
    pass  # 必要に応じてテストコードを記述

# --- END OF FILE generator/melody_generator.py ---
