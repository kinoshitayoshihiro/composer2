# --- START OF FILE generator/base_part_generator.py (修正版) ---
import logging
import random
import statistics
from abc import ABC, abstractmethod
from typing import Any

from music21 import meter, stream
from music21 import volume as m21volume

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from utilities import fx_envelope
from utilities.cc_tools import finalize_cc_events, merge_cc_events
from utilities.tone_shaper import ToneShaper

try:
    from utilities.humanizer import apply as humanize_apply
    from utilities.humanizer import (
        apply_envelope,
        apply_humanization_to_part,
        apply_offset_profile,
        apply_swing,
    )
    from utilities.override_loader import Overrides as OverrideModelType
    from utilities.override_loader import get_part_override
    from utilities.prettymidi_sync import apply_groove_pretty, load_groove_profile
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing optional utilities. Please install dependencies via "
        "'pip install -r requirements.txt'."
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
        ml_velocity_model_path: str | None = None,
        **kwargs,
    ):
        # ここを追加
        self.global_settings = global_settings or {}
        self.part_name = kwargs.get("part_name")
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature = global_time_signature
        try:
            num, denom = map(int, str(global_time_signature).split("/"))
        except Exception:
            ts_obj = meter.TimeSignature(global_time_signature or "4/4")
            num, denom = ts_obj.numerator, ts_obj.denominator
        self.bar_length = num * (4 / denom)
        # Determine default swing subdivision
        if denom == 8 and num in (6, 12):
            self.swing_subdiv = 12
        else:
            self.swing_subdiv = 8
        self.global_key_signature_tonic = global_key_signature_tonic
        self.global_key_signature_mode = global_key_signature_mode
        self.rng = rng or random.Random()
        self.ml_velocity_model_path = ml_velocity_model_path
        self.ml_velocity_model = None
        self.ml_velocity_cache_key = (
            ml_velocity_model_path if ml_velocity_model_path and torch else None
        )
        if ml_velocity_model_path:
            try:
                from utilities.ml_velocity import MLVelocityModel

                self.ml_velocity_model = MLVelocityModel.load(ml_velocity_model_path)
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.getLogger(__name__).warning(
                    "Failed to load ML velocity model %s: %s",
                    ml_velocity_model_path,
                    exc,
                )
        # 各ジェネレーター固有のロガー
        name = self.part_name or self.__class__.__name__.lower()
        self.logger = logging.getLogger(f"modular_composer.{name}")

    # --------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------

    @property
    def measure_duration(self) -> float:
        """Return the quarterLength duration of one bar."""
        return getattr(self, "_measure_duration", self.bar_length)

    @measure_duration.setter
    def measure_duration(self, value: float) -> None:
        self._measure_duration = float(value)

    # --------------------------------------------------------------
    # Tone & Dynamics - 自動アンプ／キャビネット CC 付与
    # --------------------------------------------------------------
    def _auto_tone_shape(self, part: stream.Part, intensity: str) -> None:
        """平均 Velocity と Intensity から ToneShaper を適用し CC を追加。"""
        notes = list(part.flatten().notes)
        if not notes:  # 無音パートなら何もしない
            return

        avg_vel = statistics.mean(n.volume.velocity or 64 for n in notes)
        shaper = ToneShaper()
        preset = shaper.choose_preset(
            amp_hint=None, intensity=intensity, avg_velocity=avg_vel
        )
        tone_events = shaper.to_cc_events(
            amp_name=preset, intensity=intensity, as_dict=False
        )
        existing = [
            (e["time"], e["cc"], e["val"]) if isinstance(e, dict) else e
            for e in getattr(part, "extra_cc", [])
        ]
        part.extra_cc = merge_cc_events(set(existing), set(tone_events))

    def _apply_effect_envelope(
        self, part: stream.Part, envelope_map: dict | None
    ) -> None:
        """Helper to apply effect automation envelopes."""
        if not envelope_map:
            return
        try:
            fx_envelope.apply(part, envelope_map, bpm=float(self.global_tempo or 120.0))
            if getattr(part, "metadata", None) is not None:
                part.metadata.fx_envelope = envelope_map
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("Failed to apply effect envelope: %s", exc, exc_info=True)

    def _apply_ml_velocity(self, part: stream.Part) -> None:
        """Apply ML velocity model to generated notes if available."""
        model = getattr(self, "ml_velocity_model", None)
        if model is None or torch is None:
            return
        try:
            import numpy as np

            notes = list(part.recurse().notes)
            if not notes:
                return
            ctx = np.array(
                [
                    [
                        i / len(notes),
                        n.pitch.midi / 127.0,
                        (n.volume.velocity or 64) / 127.0,
                    ]
                    for i, n in enumerate(notes)
                ],
                dtype=np.float32,
            )
            vels = model.predict(ctx, cache_key=self.ml_velocity_cache_key)
            for n, v in zip(notes, vels):
                if n.volume is None:
                    n.volume = m21volume.Volume(velocity=64)
                n.volume.velocity = int(max(1, min(127, float(v))))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("ML velocity inference failed: %s", exc)

    def compose(
        self,
        *,
        section_data: dict[str, Any],
        overrides_root: OverrideModelType | None = None,
        groove_profile_path: str | None = None,
        next_section_data: dict[str, Any] | None = None,
        part_specific_humanize_params: dict[str, Any] | None = None,
        shared_tracks: dict[str, Any] | None = None,
        vocal_metrics: dict | None = None,
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

        offset_profile = (
            self.overrides and getattr(self.overrides, "offset_profile", None)
        ) or section_data.get("part_params", {}).get("offset_profile")
        offset_profile_rh = (
            self.overrides.offset_profile_rh if self.overrides else None
        ) or section_data.get("part_params", {}).get("offset_profile_rh")
        offset_profile_lh = (
            self.overrides.offset_profile_lh if self.overrides else None
        ) or section_data.get("part_params", {}).get("offset_profile_lh")

        overrides_dump = (
            self.overrides.model_dump(exclude_unset=True)
            if self.overrides and hasattr(self.overrides, "model_dump")
            else "None"
        )
        self.logger.info(
            f"Rendering part for section: '{section_label}' with overrides: {overrides_dump}"
        )
        parts = self._render_part(
            section_data, next_section_data, vocal_metrics=vocal_metrics
        )

        if not isinstance(parts, stream.Part | dict):
            self.logger.error(
                f"_render_part for {self.part_name} did not return a valid part."
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
                    template = humanize_params.get("template_name", "default_subtle")
                    custom = humanize_params.get("custom_params", {})
                    p = apply_humanization_to_part(
                        p, template_name=template, custom_params=custom
                    )
                    self.logger.info(
                        "Applied final touch humanization (template: %s) to %s",
                        template,
                        self.part_name,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error during final touch humanization for {self.part_name}: {e}",
                        exc_info=True,
                    )
            return p

        intensity = section_data.get("musical_intent", {}).get("intensity", "medium")
        scale = {"low": 0.9, "medium": 1.0, "high": 1.1, "very_high": 1.2}.get(
            intensity, 1.0
        )

        def final_process(
            p: stream.Part,
            ratio: float | None = None,
            profile: str | None = None,
        ) -> stream.Part:
            part = process_one(p)
            humanize_apply(part, None)  # 基本ヒューマナイズ
            apply_envelope(  # intensity → Velocity スケール
                part,
                0,
                int(section_data.get("q_length", 0)),
                scale,
            )
            if profile:
                apply_offset_profile(part, profile)
            if ratio is not None:  # Swing が指定されていれば適用
                apply_swing(part, ratio, subdiv=self.swing_subdiv)
            env_map = section_data.get("fx_envelope") or section_data.get(
                "effect_envelope"
            )
            self._apply_effect_envelope(part, env_map)
            post = getattr(self, "_post_process_generated_part", None)
            if callable(post):
                try:
                    post(part, section_data, ratio)
                except Exception:  # pragma: no cover - best effort
                    pass
            self._apply_ml_velocity(part)
            finalize_cc_events(part)
            self._last_section = section_data
            return part

        if isinstance(parts, dict):
            return {
                k: final_process(
                    v,
                    (
                        swing_rh
                        if "rh" in k.lower() and swing_rh is not None
                        else (
                            swing_lh
                            if "lh" in k.lower() and swing_lh is not None
                            else swing
                        )
                    ),
                    (
                        offset_profile_rh
                        if "rh" in k.lower() and offset_profile_rh is not None
                        else (
                            offset_profile_lh
                            if "lh" in k.lower() and offset_profile_lh is not None
                            else offset_profile
                        )
                    ),
                )
                for k, v in parts.items()
            }
        else:
            return final_process(parts, swing, offset_profile)

    @abstractmethod
    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
        vocal_metrics: dict | None = None,
    ) -> stream.Part | dict[str, stream.Part]:
        raise NotImplementedError


def prepare_main_cfg(main_cfg: dict, default_ts: str = "4/4"):
    if "global_settings" not in main_cfg:
        main_cfg["global_settings"] = {}
    if "time_signature" not in main_cfg["global_settings"]:
        main_cfg["global_settings"]["time_signature"] = default_ts
    return main_cfg


# --- END OF FILE generator/base_part_generator.py ---
