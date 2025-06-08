# --- START OF FILE generator/base_part_generator.py (修正版) ---
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from music21 import stream
import logging
import random
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
        *,
        global_settings: dict = None,
        default_instrument,
        global_tempo,
        global_time_signature,
        global_key_signature_tonic,
        global_key_signature_mode,
        rng=None,
        **kwargs,
    ):
        # ここを追加
        self.global_settings = global_settings or {}
        self.part_name = kwargs.get("part_name")
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature = global_time_signature
        self.global_key_signature_tonic = global_key_signature_tonic
        self.global_key_signature_mode = global_key_signature_mode
        self.rng = rng or random.Random()
        # 各ジェネレーター固有のロガー
        name = self.part_name or self.__class__.__name__.lower()
        self.logger = logging.getLogger(f"modular_composer.{name}")

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


def prepare_main_cfg(main_cfg: dict, default_ts: str = "4/4"):
    if "global_settings" not in main_cfg:
        main_cfg["global_settings"] = {}
    if "time_signature" not in main_cfg["global_settings"]:
        main_cfg["global_settings"]["time_signature"] = default_ts
    return main_cfg


# --- END OF FILE generator/base_part_generator.py ---
