"""
utilities package -- 音楽生成プロジェクト全体で利用されるコアユーティリティ群
--------------------------------------------------------------------------
公開API:
    - core_music_utils:
        - MIN_NOTE_DURATION_QL
        - get_time_signature_object
        - sanitize_chord_label
    - scale_registry:
        - build_scale_object
        - ScaleRegistry (クラス)
    - humanizer:
        - generate_fractional_noise
        - apply_humanization_to_element
        - apply_humanization_to_part
        - HUMANIZATION_TEMPLATES
        - NUMPY_AVAILABLE
"""

from .core_music_utils import (
    MIN_NOTE_DURATION_QL,
    get_time_signature_object,
    sanitize_chord_label
    # get_music21_chord_object # --- この行を削除 ---
)

from .scale_registry import (
    build_scale_object,
    ScaleRegistry
)

from .humanizer import (
    generate_fractional_noise,
    apply_humanization_to_element,
    apply_humanization_to_part,
    HUMANIZATION_TEMPLATES,
    NUMPY_AVAILABLE,
)
from .drum_map import get_drum_map

from .drum_map_registry import get_drum_map

from .velocity_curve import resolve_velocity_curve, PREDEFINED_CURVES
from .tempo_curve import TempoCurve, TempoPoint, load_tempo_curve
from .tempo_utils import (
    load_tempo_curve as load_tempo_curve_simple,
    get_bpm_at,
    interpolate_bpm,
    beat_to_seconds,
    TempoVelocitySmoother,
)
from .velocity_smoother import VelocitySmoother, EMASmoother
from .midi_export import write_demo_bar
from .synth import render_midi

__all__ = [
    "MIN_NOTE_DURATION_QL", "get_time_signature_object", "sanitize_chord_label", # "get_music21_chord_object" を削除
    "build_scale_object", "ScaleRegistry",
    "generate_fractional_noise", "apply_humanization_to_element", "apply_humanization_to_part",
    "HUMANIZATION_TEMPLATES", "NUMPY_AVAILABLE",
    "resolve_velocity_curve", "PREDEFINED_CURVES",
    "TempoCurve", "TempoPoint", "load_tempo_curve", "VelocitySmoother", "EMASmoother",
    "load_tempo_curve_simple", "get_bpm_at", "interpolate_bpm", "beat_to_seconds",
    "TempoVelocitySmoother", "write_demo_bar", "render_midi",
]
