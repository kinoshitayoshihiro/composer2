# --- START OF FILE generator/base_part_generator.py (修正版) ---
import logging
import math
import random
import re
import statistics
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from music21 import meter, stream
from music21 import volume as m21volume

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from utilities import fx_envelope

try:  # pragma: no cover - optional add_cc_events
    from utilities.cc_tools import add_cc_events, finalize_cc_events, merge_cc_events
except ImportError:  # fallback if add_cc_events is absent
    from utilities.cc_tools import finalize_cc_events, merge_cc_events

    add_cc_events = None  # type: ignore
from utilities.tone_shaper import ToneShaper
from utilities.velocity_utils import scale_velocity
from data.export_pretty import stream_to_pretty_midi
from utilities import pb_math

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


@dataclass
class ControlConfig:
    enable_bend: bool = True
    bend_depth_semitones: float = 2.0
    vibrato_rate_hz: float | None = None
    vibrato_depth_semitones: float = 0.2
    portamento_ms: int = 0
    enable_cc11: bool = True
    cc11_profile: str = "auto"  # "auto" | "flat" | "adsr"
    cc11_sustain_level: int = 90
    cc11_attack_ms: int = 20
    cc11_release_ms: int = 60
    enable_cc64: bool = False


def apply_controls(inst, cfg: ControlConfig) -> None:
    """Apply simple CC11, CC64, and vibrato templates to ``inst``."""
    try:
        import pretty_midi
    except Exception:  # pragma: no cover - optional dependency
        return
    if cfg.enable_cc11 and inst.notes:
        start = float(inst.notes[0].start)
        end = float(inst.notes[-1].end)
        if cfg.cc11_profile == "flat":
            inst.control_changes.append(
                pretty_midi.ControlChange(
                    number=11, value=cfg.cc11_sustain_level, time=start
                )
            )
        else:  # "auto" or "adsr"
            atk = start + cfg.cc11_attack_ms / 1000.0
            rel = end + cfg.cc11_release_ms / 1000.0
            inst.control_changes.extend(
                [
                    pretty_midi.ControlChange(number=11, value=0, time=start),
                    pretty_midi.ControlChange(
                        number=11, value=cfg.cc11_sustain_level, time=atk
                    ),
                    pretty_midi.ControlChange(number=11, value=0, time=rel),
                ]
            )
        if not any(
            cc.number == 11 and cc.value == 0 and cc.time >= end
            for cc in inst.control_changes
        ):
            inst.control_changes.append(
                pretty_midi.ControlChange(number=11, value=0, time=end)
            )
    if cfg.enable_cc64 and inst.notes:
        start = float(inst.notes[0].start)
        end = float(inst.notes[-1].end)
        if not any(cc.number == 64 and cc.value >= 64 for cc in inst.control_changes):
            inst.control_changes.append(
                pretty_midi.ControlChange(number=64, value=127, time=start)
            )
        if not any(
            cc.number == 64 and cc.value == 0 and cc.time >= end
            for cc in inst.control_changes
        ):
            inst.control_changes.append(
                pretty_midi.ControlChange(number=64, value=0, time=end)
            )
    if cfg.enable_bend and cfg.vibrato_rate_hz and inst.notes:
        base = float(inst.notes[0].start)
        end = float(inst.notes[-1].end)
        depth = int(pb_math.semi_to_pb(cfg.vibrato_depth_semitones, cfg.bend_depth_semitones))
        t = base
        step = 1.0 / (cfg.vibrato_rate_hz * 8)
        while t <= end:
            val = int(round(depth * math.sin(2 * math.pi * (t - base) * cfg.vibrato_rate_hz)))
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=val, time=float(t)))
            t += step
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=float(end)))


class BasePartGenerator(ABC):
    """全楽器ジェネレーターが継承する共通基底クラス。"""

    # Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
    _SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"

    def __init__(
        self,
        *,
        global_settings: dict | None = None,
        default_instrument,
        global_tempo=None,
        global_time_signature=None,
        global_key_signature_tonic=None,
        global_key_signature_mode=None,
        rng=None,
        ml_velocity_model_path: str | None = None,
        duration_model=None,
        velocity_model=None,
        bend_depth_semitones: float = 2.0,
        vibrato_rate_hz: float | None = None,
        portamento_ms: float | None = None,
        vibrato_shape: str = "sine",
        controls: dict | None = None,
        control_config=None,  # SimpleNamespace-compatible (utilities.control_config) defaults for CCs
        **kwargs,
    ):
        # optional contextual parameters; shim for backwards compatibility
        key = kwargs.pop("key", None)
        tempo = kwargs.pop("tempo", None)
        emotion = kwargs.pop("emotion", None)

        self.global_settings = global_settings or {}
        if control_config is None:
            try:
                from utilities import control_config as _default_cc
            except Exception:
                from types import SimpleNamespace

                control_config = SimpleNamespace()
            else:
                control_config = _default_cc
        self.control_config = control_config
        self.part_name = kwargs.get("part_name")
        self.default_instrument = default_instrument
        self.controls = controls or {
            "enable_cc11": False,
            "cc11_shape": "pad",
            "cc11_depth": 1.0,
            "enable_sustain": False,
            "sustain_mode": "heuristic",
        }
        self.emotion = emotion
        self.global_tempo = tempo or global_tempo
        self.global_time_signature = global_time_signature or "4/4"
        try:
            num, denom = map(int, str(self.global_time_signature).split("/"))
        except Exception:
            ts_obj = meter.TimeSignature(self.global_time_signature)
            num, denom = ts_obj.numerator, ts_obj.denominator
        self.bar_length = num * (4 / denom)
        # Determine default swing subdivision
        if denom == 8 and num in (6, 12):
            self.swing_subdiv = 12
        else:
            self.swing_subdiv = 8
        if key is not None:
            if isinstance(key, (tuple, list)):
                tonic = key[0]
                mode = key[1] if len(key) > 1 else "major"
            else:
                parts = str(key).replace("_", " ").replace("-", " ").split()
                tonic = parts[0] if parts else global_key_signature_tonic
                mode = (
                    parts[1] if len(parts) > 1 else global_key_signature_mode or "major"
                )
            self.global_key_signature_tonic = tonic
            self.global_key_signature_mode = mode
        else:
            self.global_key_signature_tonic = global_key_signature_tonic
            self.global_key_signature_mode = global_key_signature_mode
        if self.global_key_signature_tonic and self.global_key_signature_mode:
            self.key = f"{self.global_key_signature_tonic} {self.global_key_signature_mode}".strip()
        else:
            self.key = None
        if rng is None:
            rng = random.Random(0) if self._SPARKLE_DETERMINISTIC else random.Random()
        self.rng = rng
        self.ml_velocity_model_path = ml_velocity_model_path
        self.velocity_model = velocity_model
        self.duration_model = duration_model
        self.bend_depth_semitones = bend_depth_semitones
        self.vibrato_rate_hz = vibrato_rate_hz
        self.portamento_ms = portamento_ms
        self.vibrato_shape = vibrato_shape
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

        super().__init__()

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
        """Apply ML velocity model to generated notes if available.

        The model should be trained using ``train_velocity.py`` and saved via
        :mod:`utilities.ml_velocity`.  It is loaded lazily based on the path
        provided at initialization.
        """
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

    def _apply_velocity_model(self, part: stream.Part) -> None:
        """Apply simple velocity model if provided."""
        model = getattr(self, "velocity_model", None)
        if model is None:
            return
        notes = list(part.recurse().notes)
        if hasattr(model, "predict"):
            try:
                import numpy as np

                ctx = np.array(
                    [[float(n.offset), float(n.pitch.midi)] for n in notes],
                    dtype=np.float32,
                )
                preds = model.predict(ctx)
            except Exception:
                preds = [None] * len(notes)
            for n, v in zip(notes, preds):
                if v is None:
                    continue
                if n.volume is None:
                    n.volume = m21volume.Volume(velocity=64)
                n.volume.velocity = scale_velocity(v, 1.0)
        else:
            for n in notes:
                pos = float(n.offset)
                try:
                    vel = model.sample(self.part_name or "part", pos)
                except Exception:
                    vel = None
                if vel is not None:
                    if n.volume is None:
                        n.volume = m21volume.Volume(velocity=vel)
                    else:
                        n.volume.velocity = scale_velocity(vel, 1.0)

    def _apply_duration_model(self, part: stream.Part) -> None:
        """Adjust note durations using an ML duration model if supplied.

        The model is expected to expose a ``predict`` method taking an array of
        note features and returning predicted quarterLength values.  This is a
        lightweight hook for models such as the Duration Transformer; training
        utilities are provided in ``scripts/train_duration.py``.
        """
        model = getattr(self, "duration_model", None)
        if model is None:
            self.logger.debug(
                "No duration model supplied; skipping duration adjustment"
            )
            return
        try:
            import numpy as np

            notes = list(part.recurse().notes)
            if not notes:
                return
            feats = np.array(
                [
                    [
                        n.pitch.midi / 127.0,
                        float(n.offset),
                        (n.volume.velocity or 64) / 127.0,
                    ]
                    for n in notes
                ],
                dtype=np.float32,
            )
            preds = model.predict(feats)
            for n, d in zip(notes, preds):
                n.quarterLength = max(0.05, float(d))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.debug("ML duration inference failed: %s", exc)

    # --------------------------------------------------------------
    # Continuous Controls
    # --------------------------------------------------------------

    def _is_piano_like(self) -> bool:
        name = str(self.default_instrument).lower()
        return bool(re.search(r"(piano|keys|ep|rhodes)", name))

    def _build_cc11_events(self, part: stream.Part) -> list[tuple[float, int, int]]:
        shape = self.controls.get("cc11_shape", "pad")
        depth = float(self.controls.get("cc11_depth", 1.0))
        params = {
            "bowed": (0.1, 0.0, 1.0, 0.1),
            "pluck": (0.01, 0.05, 0.4, 0.1),
            "pad": (0.05, 0.0, 1.0, 0.2),
        }
        attack, decay, sustain, release = params.get(shape, params["pad"])
        events: list[tuple[float, int, int]] = []
        for n in part.recurse().notes:
            start = float(n.offset)
            end = float(n.offset + n.quarterLength)
            atk = min(attack, max(0.0, end - start))
            rel = min(release, max(0.0, end - start - atk))
            dec = min(decay, max(0.0, end - start - atk - rel))
            peak = int(round(127 * depth))
            sus = int(round(127 * depth * sustain))
            events.append((start, 11, 0))
            events.append((start + atk, 11, peak))
            if dec > 0:
                events.append((start + atk + dec, 11, sus))
            events.append((end - rel, 11, sus))
            events.append((end, 11, 0))
        return events

    def _build_cc64_events(self, part: stream.Part) -> list[tuple[float, int, int]]:
        notes = sorted(part.recurse().notes, key=lambda n: float(n.offset))
        if len(notes) < 2:
            return []
        tempo = float(self.global_tempo or 120.0)
        beat = 60.0 / tempo
        thresh = beat * 0.25
        min_dwell = 0.08
        events: list[tuple[float, int, int]] = []
        for a, b in zip(notes, notes[1:]):
            a_end = float(a.offset + a.quarterLength)
            b_start = float(b.offset)
            gap = b_start - a_end
            if 0 < gap < thresh:
                on = a_end
                off = max(a_end, b_start - 1e-3)
                if off - on >= min_dwell:
                    events.append((on, 64, 127))
                    events.append((off, 64, 0))
        out: list[tuple[float, int, int]] = []
        last_val: int | None = None
        for t, cc, v in sorted(events):
            if v != last_val:
                out.append((t, cc, v))
                last_val = v
        return out

    def _apply_controls(self, part: stream.Part) -> None:
        if not self.controls:
            return
        events: list[tuple[float, int, int]] = []
        if self.controls.get("enable_cc11"):
            events.extend(self._build_cc11_events(part))
        if (
            self.controls.get("enable_sustain")
            and self.controls.get("sustain_mode", "heuristic") == "heuristic"
            and self._is_piano_like()
        ):
            events.extend(self._build_cc64_events(part))
        if events:
            if add_cc_events is not None:
                add_cc_events(part, events)
            else:  # pragma: no cover - legacy fallback
                existing = getattr(part, "extra_cc", [])
                part.extra_cc = merge_cc_events(existing, events, as_dict=True)

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
        try:
            parts = self._render_part(
                section_data, next_section_data, vocal_metrics=vocal_metrics
            )
        except TypeError:
            parts = self._render_part(section_data, next_section_data)

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
            self._apply_velocity_model(part)
            self._apply_duration_model(part)
            self._apply_controls(part)
            finalize_cc_events(part)
            cfg = self.control_config or ControlConfig()
            try:
                pm = stream_to_pretty_midi(part)
                if pm.instruments:
                    inst_pm = pm.instruments[0]
                    apply_controls(inst_pm, cfg)
                    extra = [
                        {"time": cc.time, "cc": cc.number, "val": cc.value}
                        for cc in inst_pm.control_changes
                    ]
                    if inst_pm.pitch_bends:
                        extra.extend(
                            {
                                "time": pb.time,
                                "cc": -1,
                                "val": pb.pitch,
                            }
                            for pb in inst_pm.pitch_bends
                        )
                    if extra:
                        base = getattr(part, "extra_cc", [])
                        part.extra_cc = merge_cc_events(base, extra)
            except Exception:  # pragma: no cover - best effort
                pass
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
