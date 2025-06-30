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

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from . import groove_sampler_ngram as groove_sampler_ngram
from .accent_mapper import AccentMapper

try:
    from .consonant_extract import (
        EssentiaUnavailable,
        detect_consonant_peaks,
        extract_to_json,
    )
except Exception:  # pragma: no cover - optional dependency
    class EssentiaUnavailable(RuntimeError):
        """Raised when librosa-based peak extraction is unavailable."""

    def detect_consonant_peaks(*_args: Any, **_kwargs: Any) -> list[float]:
        raise EssentiaUnavailable("librosa is required for peak detection")

    def extract_to_json(*_args: Any, **_kwargs: Any) -> None:
        raise EssentiaUnavailable("librosa is required for peak detection")
from .core_music_utils import (
    MIN_NOTE_DURATION_QL,
    get_time_signature_object,
    sanitize_chord_label,
    # get_music21_chord_object # --- この行を削除 ---
)
from .drum_map import get_drum_map
from .humanizer import (
    HUMANIZATION_TEMPLATES,
    NUMPY_AVAILABLE,
    apply_humanization_to_element,
    apply_humanization_to_part,
    generate_fractional_noise,
)
from .midi_export import write_demo_bar
from .scale_registry import ScaleRegistry, build_scale_object
from .synth import render_midi
from .tempo_curve import TempoCurve, TempoPoint, load_tempo_curve
from .tempo_utils import (
    TempoMap,
    TempoVelocitySmoother,
    beat_to_seconds,
    get_bpm_at,
    get_tempo_at_beat,
    interpolate_bpm,
    load_tempo_map,
)
from .tempo_utils import (
    load_tempo_curve as load_tempo_curve_simple,
)
from .velocity_curve import PREDEFINED_CURVES, resolve_velocity_curve
from .velocity_smoother import EMASmoother, VelocitySmoother
from .emotion_profile_loader import load_emotion_profile

__all__ = [
    "MIN_NOTE_DURATION_QL",
    "get_time_signature_object",
    "sanitize_chord_label",  # "get_music21_chord_object" を削除
    "build_scale_object",
    "ScaleRegistry",
    "generate_fractional_noise",
    "apply_humanization_to_element",
    "apply_humanization_to_part",
    "HUMANIZATION_TEMPLATES",
    "NUMPY_AVAILABLE",
    "resolve_velocity_curve",
    "PREDEFINED_CURVES",
    "TempoCurve",
    "TempoPoint",
    "load_tempo_curve",
    "VelocitySmoother",
    "EMASmoother",
    "load_tempo_curve_simple",
    "get_tempo_at_beat",
    "get_bpm_at",
    "interpolate_bpm",
    "beat_to_seconds",
    "TempoMap",
    "load_tempo_map",
    "TempoVelocitySmoother",
    "write_demo_bar",
    "render_midi",
    "get_drum_map",
    "AccentMapper",
    "EssentiaUnavailable",
    "detect_consonant_peaks",
    "extract_to_json",
    "load_emotion_profile",
    "groove_sampler_ngram",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name == "groove_sampler_ngram":
        module = importlib.import_module("utilities.groove_sampler_ngram")
        globals()[name] = module
        return module
    raise AttributeError(name)
