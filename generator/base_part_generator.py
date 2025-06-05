# --- START OF FILE generator/base_part_generator.py (修正版) ---
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from music21 import stream, note, tempo, meter, key, instrument as m21instrument
import logging

try:
    # ★★★ prettymidi_sync からのインポート名を修正 ★★★
    from utilities.prettymidi_sync import apply_groove_pretty, load_groove_profile
    from utilities.override_loader import (
        get_part_override,
        Overrides as OverrideModelType,
        PartOverride,
    )  # ★★★ PartOverrideModel -> PartOverride ★★★
    from utilities.humanizer import apply_humanization_to_part, HUMANIZATION_TEMPLATES
except ImportError as e:
    logging.warning(
        f"BasePartGenerator: Could not import some utilities: {e}. Some features might be limited."
    )

    def apply_groove_pretty(part, groove_profile):
        return part  # type: ignore

    def load_groove_profile(path):
        return None  # type: ignore

    # ★★★ PartOverride のダミーも修正 ★★★
    class PartOverride:  # type: ignore
        model_config = {}
        model_fields = {}

        def model_dump(self, exclude_unset=True):
            return {}

    def get_part_override(
        overrides, section_label, part_name
    ) -> Optional[PartOverride]:
        return None  # type: ignore

    def apply_humanization_to_part(part, template_name=None, custom_params=None):
        return part  # type: ignore

    HUMANIZATION_TEMPLATES = {}  # type: ignore

    class OverrideModelType:
        root: Dict = {}  # type: ignore


class BasePartGenerator(ABC):
    """全楽器ジェネレーターが継承する共通基底クラス。"""

    def __init__(
        self,
        part_name: str,
        part_parameters: Dict[str, Any],  # ← これを追加
        main_cfg: Dict[str, Any],
        groove_profile: Optional[Dict[str, Any]] = None,
        global_tempo_val: int = 120,
        global_ts_str: str = "4/4",
        global_key_tonic_val: str = "C",
        global_key_mode_val: str = "maj",
    ):
        self.part_name = part_name
        self.part_parameters = part_parameters
        self.main_cfg = main_cfg
        self.groove_profile = groove_profile

        self.global_tempo = global_tempo_val
        self.global_time_signature_str = global_ts_str
        self.global_key_tonic = global_key_tonic_val
        self.global_key_mode = global_key_mode_val

        self.measure_duration = 4.0
        try:
            ts_obj_for_dur = meter.TimeSignature(self.global_time_signature_str)
            self.measure_duration = ts_obj_for_dur.barDuration.quarterLength
        except Exception:
            pass

        self.logger = self._get_logger()
        self.overrides: Optional[PartOverride] = (
            None  # ★★★ PartOverrideModel -> PartOverride ★★★
        )

    def compose(
        self,
        *,
        section_data: Dict[str, Any],
        overrides_root: Optional[OverrideModelType] = None,
        groove_profile_path: Optional[str] = None,
        next_section_data: Optional[Dict[str, Any]] = None,
        part_specific_humanize_params: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        section_label = section_data.get(
            "section_name", "UnknownSection"
        )  # section -> section_data
        if overrides_root:
            # ★★★ PartOverrideModel -> PartOverride ★★★
            self.overrides = get_part_override(
                overrides_root, section_label, self.part_name
            )
        else:
            self.overrides = None

        self.logger.info(
            f"Rendering part for section: '{section_label}' with overrides: {self.overrides.model_dump(exclude_unset=True) if self.overrides and hasattr(self.overrides, 'model_dump') else 'None'}"
        )

        part = self._render_part(section_data, next_section_data)

        if not isinstance(part, stream.Part):
            self.logger.error(
                f"_render_part for {self.part_name} did not return a valid music21.stream.Part. Returning empty part."
            )
            part = stream.Part(id=self.part_name)

        if groove_profile_path and part.flatten().notes:
            try:
                gp = load_groove_profile(groove_profile_path)
                if gp:
                    # ★★★ apply_groove_pretty が Part を返すように修正したので、代入する ★★★
                    part = apply_groove_pretty(part, gp)
                    self.logger.info(
                        f"Applied groove from '{groove_profile_path}' to {self.part_name}."
                    )
                else:
                    self.logger.warning(
                        f"Could not load groove profile from '{groove_profile_path}'."
                    )
            except Exception as e_groove:
                self.logger.error(
                    f"Error applying groove to {self.part_name}: {e_groove}",
                    exc_info=True,
                )

        if part_specific_humanize_params and part_specific_humanize_params.get(
            "humanize_opt", False
        ):
            if part.flatten().notes:
                try:
                    template = part_specific_humanize_params.get(
                        "template_name", "default_subtle"
                    )
                    custom = part_specific_humanize_params.get("custom_params", {})
                    # ★★★ apply_humanization_to_part が Part を返すように修正したので、代入する ★★★
                    part = apply_humanization_to_part(
                        part, template_name=template, custom_params=custom
                    )
                    self.logger.info(
                        f"Applied humanization (template: {template}) to {self.part_name}."
                    )
                except Exception as e_humanize_part:
                    self.logger.error(
                        f"Error during part-level humanization for {self.part_name}: {e_humanize_part}",
                        exc_info=True,
                    )
            else:
                self.logger.info(
                    f"Part {self.part_name} is empty, skipping humanization."
                )
        return part

    @abstractmethod
    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        raise NotImplementedError

    def _get_logger(self):
        return logging.getLogger(f"modular_composer.{self.part_name}_generator")


# --- END OF FILE generator/base_part_generator.py ---
