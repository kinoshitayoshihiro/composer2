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
import importlib.util as importlib_util
from typing import TYPE_CHECKING, Any

__all__: list[str] = []

_HAS_MUSIC21 = importlib_util.find_spec("music21") is not None
_HAS_YAML = importlib_util.find_spec("yaml") is not None

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from . import groove_sampler_ngram as groove_sampler_ngram
    from . import vocal_sync as vocal_sync

from .accent_mapper import AccentMapper
from .progression_templates import get_progressions
from .rest_utils import get_rest_windows
from .velocity_model import KDEVelocityModel

__all__.append("get_progressions")

from .tempo_utils import beat_to_seconds

__all__.append("beat_to_seconds")


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


if _HAS_MUSIC21:
    from .core_music_utils import (
        MIN_NOTE_DURATION_QL,
        get_time_signature_object,
        sanitize_chord_label,
    )
else:  # pragma: no cover - optional dependency
    MIN_NOTE_DURATION_QL = 0.0625

    def _missing(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )

    get_time_signature_object = _missing
    sanitize_chord_label = _missing
from .drum_map import get_drum_map

if _HAS_MUSIC21:
    from .humanizer import (
        HUMANIZATION_TEMPLATES,
        NUMPY_AVAILABLE,
        apply_humanization_to_element,
        apply_humanization_to_part,
        generate_fractional_noise,
    )
else:
    HUMANIZATION_TEMPLATES = {}
    NUMPY_AVAILABLE = False

    def _missing(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )

    apply_humanization_to_element = _missing
    apply_humanization_to_part = _missing
    generate_fractional_noise = _missing
if _HAS_MUSIC21:
    from .midi_export import write_demo_bar
else:

    def write_demo_bar(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )


if _HAS_MUSIC21:
    from .arrangement_builder import build_arrangement, score_to_pretty_midi
else:

    def build_arrangement(*_args: Any, **_kwargs: Any) -> tuple[Any, list[str]]:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )

    def score_to_pretty_midi(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )


if _HAS_MUSIC21:
    from .scale_registry import ScaleRegistry, build_scale_object
else:

    class ScaleRegistry:  # type: ignore[misc]
        pass

    def build_scale_object(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )


if _HAS_MUSIC21:
    from .synth import export_audio, render_midi
else:

    def render_midi(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )

    def export_audio(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )


if _HAS_MUSIC21 and _HAS_YAML:
    from .tempo_curve import TempoCurve, TempoPoint, load_tempo_curve
else:

    class TempoPoint:  # type: ignore[misc]
        beat: float
        bpm: float

    class TempoCurve:  # type: ignore[misc]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "PyYAML and music21 are required. Please run 'pip install -r requirements.txt'."
            )

    def load_tempo_curve(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "PyYAML and music21 are required. Please run 'pip install -r requirements.txt'."
        )


from .tempo_utils import (
    TempoMap,
    TempoVelocitySmoother,
    beat_to_seconds,
    get_bpm_at,
    get_tempo_at_beat,
    interpolate_bpm,
)
from .tempo_utils import load_tempo_curve as load_tempo_curve_simple
from .tempo_utils import (
    load_tempo_map,
)
from .velocity_curve import PREDEFINED_CURVES, resolve_velocity_curve
from .velocity_smoother import EMASmoother, VelocitySmoother

if _HAS_MUSIC21:
    from .humanizer import apply_velocity_histogram, apply_velocity_histogram_profile
else:

    def apply_velocity_histogram(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )

    def apply_velocity_histogram_profile(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "music21 is required. Please run 'pip install -r requirements.txt'."
        )


if _HAS_MUSIC21:
    from .timing_corrector import TimingCorrector
else:

    class TimingCorrector:  # type: ignore[misc]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "music21 is required. Please run 'pip install -r requirements.txt'."
            )


if _HAS_YAML:
    from .emotion_profile_loader import load_emotion_profile
else:

    def load_emotion_profile(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            "PyYAML is required. Please run 'pip install -r requirements.txt'."
        )


if importlib_util.find_spec("numpy") is not None:
    from .loudness_meter import RealtimeLoudnessMeter
else:

    class RealtimeLoudnessMeter:  # type: ignore[misc]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "numpy is required. Please run 'pip install -r requirements.txt'."
            )


from . import mix_profile
from .install_utils import run_with_retry

try:
    from .cc_map import cc_map, load_cc_map
except Exception:
    cc_map = {}

    def load_cc_map(*_a: Any, **_k: Any) -> None:
        raise ModuleNotFoundError(
            "PyYAML is required. Please run 'pip install -r requirements.txt'."
        )


try:
    importlib.import_module("utilities.ir_renderer")
    from . import ir_renderer
except ModuleNotFoundError:
    ir_renderer = None  # type: ignore
try:
    from . import vocal_sync
except Exception:
    vocal_sync = None  # type: ignore
if importlib_util.find_spec("numpy") is not None:
    from .audio_render import render_part_audio
    from .convolver import convolve_ir, load_ir, render_wav
    from .effect_preset_loader import EffectPresetLoader
else:
    import sys
    import types

    convolver_stub = types.ModuleType("utilities.convolver")

    def _missing(*_a: Any, **_k: Any) -> None:
        raise ModuleNotFoundError(
            "numpy is required. Please run 'pip install -r requirements.txt'."
        )

    convolver_stub.load_ir = _missing
    convolver_stub.convolve_ir = _missing
    convolver_stub.render_wav = _missing
    convolver_stub.render_with_ir = _missing
    sys.modules[__name__ + ".convolver"] = convolver_stub

    load_ir = convolver_stub.load_ir
    convolve_ir = convolver_stub.convolve_ir
    render_wav = convolver_stub.render_wav

    def render_part_audio(*_a: Any, **_k: Any) -> None:
        _missing()

    EffectPresetLoader = None  # type: ignore

__all__ += [
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
    "apply_velocity_histogram",
    "apply_velocity_histogram_profile",
    "TimingCorrector",
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
    "export_audio",
    "get_drum_map",
    "AccentMapper",
    "EssentiaUnavailable",
    "detect_consonant_peaks",
    "extract_to_json",
    "load_emotion_profile",
    "RealtimeLoudnessMeter",
    "run_with_retry",
    "groove_sampler_ngram",
    "groove_sampler_rnn",
    "mix_profile",
    "cc_map",
    "load_cc_map",
    "ir_renderer",
    "load_ir",
    "convolve_ir",
    "render_wav",
    "render_part_audio",
    "EffectPresetLoader",
    "KDEVelocityModel",
    "build_arrangement",
    "score_to_pretty_midi",
    "get_rest_windows",
    "get_progressions",
    "get_rest_windows",
    "vocal_sync",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name == "groove_sampler_ngram":
        module = importlib.import_module("utilities.groove_sampler_ngram")
        globals()[name] = module
        return module
    if name == "groove_sampler_rnn":
        module = importlib.import_module("utilities.groove_sampler_rnn")
        globals()[name] = module
        return module
    if name == "vocal_sync":
        module = importlib.import_module("utilities.vocal_sync")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = list(dict.fromkeys(__all__))  # de-dup
