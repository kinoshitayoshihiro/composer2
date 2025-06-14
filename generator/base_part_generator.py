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
    from utilities.humanizer import (
        apply_humanization_to_part,
        apply_envelope,
        apply as humanize_apply,
        apply_swing,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing optional utilities. Please install dependencies via 'pip install -r requirements.txt'."
    ) from e


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
        shared_tracks: Dict[str, Any] | None = None,
    ) -> stream.Part:
        shared_tracks = shared_tracks or {}
        section_data.setdefault("shared_tracks", {}).update(shared_tracks)

        section_label = section_data.get("section_name", "UnknownSection")
        if overrides_root:
            self.overrides = get_part_override(
                overrides_root, section_label, self.part_name
            )
        else:
            self.overrides = None

        swing = (
            (self.overrides and getattr(self.overrides, "swing_ratio", None))
            or section_data.get("part_params", {}).get("swing_ratio")
            or 0.0
        )
        swing_rh = self.overrides.swing_ratio_rh if self.overrides else None
        swing_lh = self.overrides.swing_ratio_lh if self.overrides else None

        self.logger.info(
            f"Rendering part for section: '{section_label}' with overrides: {self.overrides.model_dump(exclude_unset=True) if self.overrides and hasattr(self.overrides, 'model_dump') else 'None'}"
        )
        parts = self._render_part(section_data, next_section_data)

        if not isinstance(parts, (stream.Part, dict)):
            self.logger.error(
                f"_render_part for {self.part_name} did not return a valid stream.Part or dict. Returning empty part."
            )
            return stream.Part(id=self.part_name)

        def process_one(p: stream.Part) -> stream.Part:
            if groove_profile_path and p.flatten().notes:
                try:
                    gp = load_groove_profile(groove_profile_path)
                    if gp:
                        p = apply_groove_pretty(p, gp)
                        self.logger.info(
                            f"Applied groove from '{groove_profile_path}' to {self.part_name}."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error applying groove to {self.part_name}: {e}", exc_info=True
                    )

            humanize_params = part_specific_humanize_params or {}
            if humanize_params.get("enable", False) and p.flatten().notes:
                try:
                    template = humanize_params.get(
                        "template_name", "default_subtle")
                    custom = humanize_params.get("custom_params", {})
                    p = apply_humanization_to_part(
                        p, template_name=template, custom_params=custom
                    )
                    self.logger.info(
                        f"Applied final touch humanization (template: {template}) to {self.part_name}."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error during final touch humanization for {self.part_name}: {e}",
                        exc_info=True,
                    )
            return p

        intensity = section_data.get(
            "musical_intent", {}).get("intensity", "medium")
        scale = {"low": 0.9, "medium": 1.0, "high": 1.1,
                 "very_high": 1.2}.get(intensity, 1.0)

        def final_process(p: stream.Part, ratio: float | None = None) -> stream.Part:
            part = process_one(p)
            humanize_apply(part, None)          # 基本ヒューマナイズ
            if ratio is not None:               # Swing が指定されていれば適用
                apply_swing(part, float(ratio))
            apply_envelope(                      # intensity → Velocity スケール
                part,
                0,
                int(section_data.get("q_length", 0)),
                scale,
            )
            return part

        if isinstance(parts, dict):
            return {
                k: final_process(
                    v,
                    (
                        swing_rh if "rh" in k.lower() and swing_rh is not None
                        else swing_lh if "lh" in k.lower() and swing_lh is not None
                        else swing
                    ),
                )
                for k, v in parts.items()
            }
        else:
            return final_process(parts, swing)

    @abstractmethod
    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part | Dict[str, stream.Part]:
        raise NotImplementedError


def prepare_main_cfg(main_cfg: dict, default_ts: str = "4/4"):
    if "global_settings" not in main_cfg:
        main_cfg["global_settings"] = {}
    if "time_signature" not in main_cfg["global_settings"]:
        main_cfg["global_settings"]["time_signature"] = default_ts
    return main_cfg


# --- END OF FILE generator/base_part_generator.py ---
