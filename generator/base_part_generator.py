# --- START OF FILE generator/base_part_generator.py (修正版) ---
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from music21 import stream
import logging
from music21 import instrument as m21instrument

try:
    from utilities.prettymidi_sync import apply_groove_pretty, load_groove_profile
    from utilities.override_loader import (
        get_part_override,
        Overrides as OverrideModelType,
        PartOverride,
    )
    from utilities.humanizer import apply_humanization_to_part
except ImportError:
    logging.warning(
        "BasePartGenerator: Could not import some utilities. Features might be limited."
    )

    def apply_groove_pretty(part, groove_profile):
        return part

    def load_groove_profile(path):
        return None

    class PartOverride:
        def model_dump(self, exclude_unset=True):
            return {}

    def get_part_override(overrides, section_label, part_name):
        return None

    def apply_humanization_to_part(part, template_name=None, custom_params=None):
        return part

    class OverrideModelType:
        root: Dict = {}


class BasePartGenerator(ABC):
    """全楽器ジェネレーターが継承する共通基底クラス。"""

    def __init__(
        self,
        part_name: str,
        part_parameters: Dict[str, Any],
        main_cfg: Dict[str, Any],
        groove_profile: Optional[Dict[str, Any]] = None,
    ):
        self.part_name = part_name
        self.part_parameters = part_parameters or {}
        self.main_cfg = main_cfg or {}
        self.groove_profile = groove_profile
        self.logger = logging.getLogger(f"modular_composer.{self.part_name}_generator")

        g_settings = self.main_cfg.get("global_settings", {})
        self.global_tempo = g_settings.get("tempo", 120)
        self.global_ts_str = g_settings.get("time_signature", "4/4")
        self.global_key_tonic = g_settings.get("key_tonic", "C")
        self.global_key_mode = g_settings.get("key_mode", "major")

        self.measure_duration = 4.0
        try:
            ts_obj = meter.TimeSignature(self.global_ts_str)
            self.measure_duration = ts_obj.barDuration.quarterLength
        except Exception:
            self.logger.warning(
                f"Could not parse time signature '{self.global_ts_str}'. Defaulting measure duration to 4.0."
            )

        part_default_cfg = self.main_cfg.get("default_part_parameters", {}).get(
            self.part_name, {}
        )
        instrument_name = part_default_cfg.get("instrument", "Piano")
        try:
            if self.part_name == "drums":
                self.default_instrument = m21instrument.Percussion()
                if hasattr(self.default_instrument, "midiChannel"):
                    self.default_instrument.midiChannel = 9
            else:
                self.default_instrument = m21instrument.fromString(instrument_name)
        except Exception:
            self.logger.warning(
                f"Could not create instrument '{instrument_name}'. Using Piano as fallback."
            )
            self.default_instrument = m21instrument.Piano()

        self.overrides: Optional[PartOverride] = None

    def compose(
        self,
        *,
        section_data: Dict[str, Any],
        overrides_root: Optional[OverrideModelType] = None,
        groove_profile_path: Optional[str] = None,
        next_section_data: Optional[Dict[str, Any]] = None,
        part_specific_humanize_params: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        section_label = section_data.get("section_name", "UnknownSection")
        if overrides_root:
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
            return stream.Part(id=self.part_name)

        if groove_profile_path and part.flatten().notes:
            try:
                gp = load_groove_profile(groove_profile_path)
                if gp:
                    part = apply_groove_pretty(part, gp)
                    self.logger.info(
                        f"Applied groove from '{groove_profile_path}' to {self.part_name}."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error applying groove to {self.part_name}: {e}", exc_info=True
                )

        # ヒューマナイズ処理で part_specific_humanize_params を使用する
        humanize_params = part_specific_humanize_params or {}
        if humanize_params.get("enable", False) and part.flatten().notes:
            try:
                template = humanize_params.get("template_name", "default_subtle")
                custom = humanize_params.get("custom_params", {})
                part = apply_humanization_to_part(
                    part, template_name=template, custom_params=custom
                )
                self.logger.info(
                    f"Applied final touch humanization (template: {template}) to {self.part_name}."
                )
            except Exception as e:
                self.logger.error(
                    f"Error during final touch humanization for {self.part_name}: {e}",
                    exc_info=True,
                )
        return part

    @abstractmethod
    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        raise NotImplementedError


# --- END OF FILE generator/base_part_generator.py ---
