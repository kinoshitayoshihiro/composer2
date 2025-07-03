# --- START OF FILE utilities/humanizer.py (役割特化版) ---
# ruff: noqa
import copy
import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    cast,
)

logger = logging.getLogger("otokotoba.humanizer")

try:
    from cyext.humanize import (
        apply_swing as cy_apply_swing,
        humanize_velocities as cy_humanize_velocities,
        apply_envelope as cy_apply_envelope,
        apply_velocity_histogram as cy_apply_velocity_histogram,
    )
except Exception as exc:  # pragma: no cover - optional
    logger.info(
        "Cython extensions not available (%s). Install Cython and reinstall for speed.",
        exc,
    )
    cy_apply_swing = None
    cy_humanize_velocities = None
    cy_apply_envelope = None
    cy_apply_velocity_histogram = None
from .ghost_jitter import apply_ghost_jitter  # re-export

__all__ = [
    "apply_ghost_jitter",
    "apply_velocity_histogram",
    "apply_envelope",
    "swing_offset",
]

USE_EXPR_CC11 = False
USE_AFTERTOUCH = False

def set_cc_flags(use_expr_cc11: bool, use_aftertouch: bool) -> None:
    """Configure global CC mapping flags."""
    global USE_EXPR_CC11, USE_AFTERTOUCH
    USE_EXPR_CC11 = bool(use_expr_cc11)
    USE_AFTERTOUCH = bool(use_aftertouch)

# music21 のサブモジュールを正しい形式でインポート
import music21.chord as m21chord  # check_imports.py の期待する形式 (スペースに注意)
import music21.expressions as expressions
import music21.instrument as instrument
import music21.key as key
import music21.meter as meter
import music21.note as note
import music21.stream as stream
import music21.tempo as tempo
import music21.volume as volume
from music21 import exceptions21

# MIN_NOTE_DURATION_QL は core_music_utils からインポートすることを推奨
try:
    from .core_music_utils import MIN_NOTE_DURATION_QL
except ImportError:
    MIN_NOTE_DURATION_QL = 0.125

try:
    import quantize  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _DummyQuantize:
        def setSwingRatio(self, ratio: float) -> None:
            pass

    quantize = _DummyQuantize()

class _QuantizeConfig:
    def __init__(self) -> None:
        self.swing_ratio = 0.5

    def setSwingRatio(self, ratio: float) -> None:
        self.swing_ratio = float(ratio)

# quantizeモジュールの代替として常にこのインスタンスを使う
quantize = _QuantizeConfig()

# 既存の関数があれば残しつつ、下記を追記 -----------------------
# ------------------------------------------------------------
# 1) グローバルプロファイル・レジストリ
# ------------------------------------------------------------
_PROFILE_REGISTRY: dict[str, dict[str, Any]] = {}
PROFILES = _PROFILE_REGISTRY


def load_profiles(dict_like: dict[str, Any]) -> None:
    """
    YAML から読み取った humanize_profiles セクションを登録する。
    例: cfg['humanize_profiles'] をそのまま渡す。
    """
    global _PROFILE_REGISTRY
    _PROFILE_REGISTRY = dict_like
    logger.info(f"Loaded {len(_PROFILE_REGISTRY)} humanize profiles")


def get_profile(name: str) -> dict[str, Any]:
    if name in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[name]
    raise KeyError(f"Humanize profile '{name}' not found.")


# ------------------------------------------------------------
# 2) 適用関数
# ------------------------------------------------------------
def apply(
    part_stream: stream.Part,
    profile_name: str | None = None,
    *,
    swing_ratio: float | None = None,
    kick_leak_velocity_jitter: int = 0,
    global_settings: Mapping[str, Any] | None = None,
) -> None:
    """music21.stream.Part に in-place でヒューマナイズを適用"""
    prof = get_profile(profile_name) if profile_name else {}
    off = prof.get("offset_ms", {})  # {mean, stdev}
    vel = prof.get("velocity", {})
    dur = prof.get("duration_pct", {})

    notes = list(part_stream.flatten().notes)
    for n in notes:
        # (a) onset shift
        if off:
            jitter = random.normalvariate(off.get("mean", 0.0), off.get("stdev", 0.0))
            n.offset += (
                jitter / 1000.0
            )  # ms → sec (assuming QL = 1 sec @ 60bpm; precise shiftは later pass で)

        # (b) velocity tweak
        if vel and n.volume.velocity is not None:
            tweak = random.normalvariate(vel.get("mean", 0.0), vel.get("stdev", 0.0))
            n.volume.velocity = int(max(1, min(127, n.volume.velocity + tweak)))

        # (c) duration ratio (legato / staccato 感)
        if dur:
            factor = (
                random.normalvariate(dur.get("mean", 100), dur.get("stdev", 0)) / 100.0
            )
            n.quarterLength *= factor

    if kick_leak_velocity_jitter:
        bpm_el = part_stream.recurse().getElementsByClass(tempo.MetronomeMark).first()
        bpm = bpm_el.number if bpm_el else 120
        thr = 0.06 * bpm / 60.0
        kicks = [n for n in notes if int(n.pitch.midi) == 36]
        hats = [n for n in notes if int(n.pitch.midi) in {42, 44, 46}]
        for h in hats:
            if any(abs(float(h.offset) - float(k.offset)) <= thr for k in kicks):
                if h.volume.velocity is None:
                    continue
                delta = random.randint(-kick_leak_velocity_jitter, kick_leak_velocity_jitter)
                h.volume.velocity = int(max(1, min(127, h.volume.velocity + delta)))

    if swing_ratio is None and profile_name:
        swing_ratio = cast(float | None, PROFILES.get(profile_name, {}).get("swing"))

    if swing_ratio is not None:
        try:
            quantize.setSwingRatio(swing_ratio)
        except Exception:
            pass
        swing_offset = round(swing_ratio * 0.5, 2)
        for n in part_stream.recurse().notesAndRests:
            if n.quarterLength >= 1.0:
                continue
            beat_pos = float(n.offset) % 1.0
            beat_start = n.offset - beat_pos
            if abs(beat_pos - 0.5) < 0.05:
                n.offset = beat_start + swing_offset
            elif beat_pos < 0.05 or beat_pos > 0.95:
                n.offset = beat_start

    gs = global_settings or {}
    expr = bool(gs.get("use_expr_cc11", USE_EXPR_CC11))
    aft = bool(gs.get("use_aftertouch", USE_AFTERTOUCH))
    if expr or aft:
        _humanize_velocities(
            part_stream,
            use_expr_cc11=expr,
            use_aftertouch=aft,
        )


def _validate_ratio(ratio: float) -> float:
    """Return a safe swing ratio allowing ``0`` for straight feel."""
    if ratio <= 0:
        # ``0`` disables swing; no warning necessary
        return 0.0
    if ratio >= 1:
        logger.warning("Swing ratio %.3f out of range (0,1). Using 0.5", ratio)
        return 0.5
    return ratio


def _apply_swing_py(part_stream: stream.Part, swing_ratio: float, subdiv: int = 8) -> None:
    """Shift off-beats according to ``swing_ratio`` and grid ``subdiv``.

    ``swing_ratio`` represents the relative position of the off-beat note
    within a pair (0.5 means straight). ``subdiv`` describes the number of
    divisions in a 4/4 measure. 8 results in eighth‑note swing.
    """
    swing_ratio = _validate_ratio(swing_ratio)
    if subdiv <= 0:
        return

    step = 4.0 / subdiv  # length of the smallest grid
    pair = step * 2.0    # span of an on/off pair
    tol = step * 0.1

    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        pair_start = math.floor(pos / pair) * pair
        within = pos - pair_start
        if abs(within - step) < tol:
            n.offset = pair_start + pair * swing_ratio


def swing_offset(offset: float, swing_ratio: float, subdiv: int = 8) -> float:
    """Return swung ``offset`` without modifying an element."""
    swing_ratio = _validate_ratio(swing_ratio)
    if subdiv <= 0:
        return offset
    step = 4.0 / subdiv
    pair = step * 2.0
    tol = step * 0.1
    pair_start = math.floor(offset / pair) * pair
    within = offset - pair_start
    if abs(within - step) < tol:
        return pair_start + pair * swing_ratio
    return offset


def _apply_swing(part_stream: stream.Part, swing_ratio: float, subdiv: int = 8) -> None:
    if cy_apply_swing is not None:
        cy_apply_swing(part_stream, float(swing_ratio), subdiv)
    else:
        _apply_swing_py(part_stream, swing_ratio, subdiv)


def _apply_variable_swing(
    part_stream: stream.Part, ratios: Mapping[int, float], subdiv: int = 8
) -> None:
    """Apply swing where the ratio can vary per measure index."""
    if subdiv <= 0:
        return

    step = 4.0 / subdiv
    pair = step * 2.0
    tol = step * 0.1

    ts = part_stream.recurse().getElementsByClass(meter.TimeSignature).first()
    measure_len = float(ts.barDuration.quarterLength) if ts else 4.0

    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        pair_start = math.floor(pos / pair) * pair
        within = pos - pair_start
        if abs(within - step) < tol:
            measure_idx = int(pair_start // measure_len)
            ratio = ratios.get(measure_idx, ratios.get("default", 0.5))
            ratio = _validate_ratio(ratio)
            n.offset = pair_start + pair * ratio


def _guess_subdiv(part: stream.Part) -> int:
    """Return default swing subdivision based on time signature."""
    ts = part.recurse().getElementsByClass(meter.TimeSignature).first()
    if ts:
        num, denom = ts.numerator, ts.denominator
        if denom == 4 and num in (3, 4):
            return 8
        if denom == 8 and num in (6, 12):
            return 12
    return 8


def apply_swing(
    part: stream.Part,
    ratio: float | Sequence[float] | Mapping[int, float],
    subdiv: int | None = 8,
) -> None:
    """Public API to apply swing in-place.

    Parameters
    ----------
    part : :class:`music21.stream.Part`
        Target part to modify.
    ratio : float | Sequence[float] | Mapping[int, float]
        Relative position of the off-beat note (0.5 = straight).
    subdiv : int | None
        Number of grid subdivisions per measure. ``8`` for typical eighth swing.
        When ``None``, a suitable value is inferred from the part's time signature
        (e.g. ``8`` for ``4/4`` or ``3/4`` and ``12`` for ``6/8`` or ``12/8``).
    """
    if ratio is None:
        return

    if subdiv is None:
        subdiv = _guess_subdiv(part)

    if isinstance(ratio, (int, float)):
        r = float(ratio)
        if abs(r) < 1e-6:
            return
        r = _validate_ratio(r)
        _apply_swing(part, r, subdiv=subdiv)
    else:
        if isinstance(ratio, Sequence):
            ratios = {idx: float(r) for idx, r in enumerate(ratio)}
        else:
            ratios = {int(k): float(v) for k, v in ratio.items()}
        _apply_variable_swing(part, ratios, subdiv=subdiv)


def _apply_envelope_py(part: stream.Part, start: int, end: int, scale: float) -> None:
    for n in part.recurse().notes:
        if start <= n.offset < end and n.volume and n.volume.velocity is not None:
            n.volume.velocity = int(
                max(1, min(127, round(n.volume.velocity * scale)))
            )


def apply_envelope(part: stream.Part, start: int, end: int, scale: float) -> None:
    """Scale note velocities between start and end beats."""
    if cy_apply_envelope is not None:
        cy_apply_envelope(part, start, end, scale)
    else:
        _apply_envelope_py(part, start, end, scale)


def _humanize_velocities_py(
    part: stream.Part,
    amount: int = 4,
    use_expr_cc11: bool = False,
    use_aftertouch: bool = False,
) -> None:
    for n in part.recurse().notes:
        if n.volume is None:
            n.volume = volume.Volume(velocity=64)
        vel = n.volume.velocity or 64
        delta = random.randint(-amount, amount)
        new_vel = max(1, min(127, vel + delta))
        n.volume.velocity = new_vel
        if use_expr_cc11:
            part.extra_cc = getattr(part, "extra_cc", []) + [
                {"time": float(n.offset), "number": 11, "value": new_vel}
            ]
        if use_aftertouch:
            part.extra_cc = getattr(part, "extra_cc", []) + [
                {"time": float(n.offset), "number": 74, "value": new_vel}
            ]


def _humanize_velocities(
    part: stream.Part,
    amount: int = 4,
    *,
    use_expr_cc11: bool | None = None,
    use_aftertouch: bool | None = None,
) -> None:
    """Randomise note velocities and optionally emit CC messages."""
    if use_expr_cc11 is None:
        use_expr_cc11 = USE_EXPR_CC11
    if use_aftertouch is None:
        use_aftertouch = USE_AFTERTOUCH
    if cy_humanize_velocities is not None:
        cy_humanize_velocities(part, amount, use_expr_cc11, use_aftertouch)
    else:
        _humanize_velocities_py(part, amount, use_expr_cc11, use_aftertouch)


def _apply_velocity_histogram_py(
    part: stream.Part, histogram: dict[int, float]
) -> stream.Part:
    """Assign note velocities by sampling from ``histogram``.

    Parameters
    ----------
    part:
        Target part whose note velocities will be replaced.
    histogram:
        Mapping of velocity value to weight.

    Returns
    -------
    stream.Part
        The modified part.
    """
    if not histogram:
        return part
    choices = [int(v) for v in histogram.keys()]
    weights = [float(w) for w in histogram.values()]
    for n in part.recurse().notes:
        vel = random.choices(choices, weights)[0]
        if n.volume is None:
            n.volume = volume.Volume(velocity=vel)
        else:
            n.volume.velocity = vel
    return part


def apply_velocity_histogram(
    part: stream.Part, histogram: dict[int, float]
) -> stream.Part:
    if cy_apply_velocity_histogram is not None:
        return cy_apply_velocity_histogram(part, histogram)
    return _apply_velocity_histogram_py(part, histogram)


def apply_offset_profile(part: stream.Part, profile_name: str | None) -> None:
    """Shift note offsets according to a registered profile."""
    if not profile_name:
        return
    try:
        profile = get_profile(profile_name)
    except KeyError:
        logger.warning(f"Offset profile '{profile_name}' not found.")
        return

    if "shift_ql" in profile:
        try:
            shift = float(profile["shift_ql"])
        except (TypeError, ValueError):
            shift = 0.0
        for n in part.recurse().notesAndRests:
            n.offset += shift
        for cc in getattr(part, "extra_cc", []):
            cc["time"] += shift
        return

    pattern = profile.get("offsets_ql") or profile.get("pattern")
    if not isinstance(pattern, (list, tuple)) or not pattern:
        logger.warning(
            f"Offset profile '{profile_name}' has no usable 'shift_ql' or 'offsets_ql'."
        )
        return

    shifts = [float(x) for x in pattern]
    notes = list(part.recurse().notesAndRests)
    for idx, el in enumerate(notes):
        shift = shifts[idx % len(shifts)]
        el.offset += shift
    for idx, cc in enumerate(getattr(part, "extra_cc", [])):
        shift = shifts[idx % len(shifts)]
        cc["time"] += shift


# ------------------------------------------------------------
# 3) CLI テスト用メイン（任意）
# ------------------------------------------------------------
if __name__ == "__main__":  # python utilities/humanizer.py main_cfg.yml
    import sys

    import yaml

    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    load_profiles(cfg["humanize_profiles"])
    print("Profiles loaded:", list(_PROFILE_REGISTRY))


HUMANIZATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "default_subtle": {
        "time_variation": 0.01,
        "duration_percentage": 0.03,
        "velocity_variation": 5,
        "use_fbm_time": False,
    },
    "piano_gentle_arpeggio": {
        "time_variation": 0.008,
        "duration_percentage": 0.02,
        "velocity_variation": 4,
        "use_fbm_time": True,
        "fbm_time_scale": 0.005,
        "fbm_hurst": 0.7,
    },
    "piano_block_chord": {
        "time_variation": 0.015,
        "duration_percentage": 0.04,
        "velocity_variation": 7,
        "use_fbm_time": False,
    },
    "drum_tight": {
        "time_variation": 0.005,
        "duration_percentage": 0.01,
        "velocity_variation": 3,
        "use_fbm_time": False,
    },
    "drum_loose_fbm": {
        "time_variation": 0.02,
        "duration_percentage": 0.05,
        "velocity_variation": 8,
        "use_fbm_time": True,
        "fbm_time_scale": 0.01,
        "fbm_hurst": 0.6,
    },
    "guitar_strum_loose": {
        "time_variation": 0.025,
        "duration_percentage": 0.06,
        "velocity_variation": 10,
        "use_fbm_time": True,
        "fbm_time_scale": 0.015,
    },
    "guitar_arpeggio_precise": {
        "time_variation": 0.008,
        "duration_percentage": 0.02,
        "velocity_variation": 4,
        "use_fbm_time": False,
    },
    "vocal_ballad_smooth": {
        "time_variation": 0.025,
        "duration_percentage": 0.05,
        "velocity_variation": 4,
        "use_fbm_time": True,
        "fbm_time_scale": 0.01,
        "fbm_hurst": 0.7,
    },
    "vocal_pop_energetic": {
        "time_variation": 0.015,
        "duration_percentage": 0.02,
        "velocity_variation": 8,
        "use_fbm_time": True,
        "fbm_time_scale": 0.008,
    },
    "flam_legato_ghost": {
        "time_variation": 0.005,
        "duration_percentage": 1.2,
        "velocity_variation": [-0.2, 0.2],
        "apply_to_elements": ["note"],
    },
}


try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


def generate_fractional_noise(
    length: int, hurst: float = 0.7, scale_factor: float = 1.0
) -> list[float]:
    if not NUMPY_AVAILABLE:
        logger.debug(
            f"Humanizer (FBM): NumPy not available. Using Gaussian noise for length {length}."
        )
        return [random.gauss(0, scale_factor / 3) for _ in range(length)]
    if length <= 0:
        return []
    white_noise = np.random.randn(length)
    fft_white = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(length)
    freqs[0] = 1e-6 if freqs.size > 0 and freqs[0] == 0 else freqs[0]
    filter_amplitude = np.abs(freqs) ** (-hurst)
    if freqs.size > 0:
        filter_amplitude[0] = 0
    fft_fbm = fft_white * filter_amplitude
    fbm_noise = np.fft.ifft(fft_fbm).real
    std_dev = np.std(fbm_noise)
    if std_dev != 0:
        fbm_norm = scale_factor * (fbm_noise - np.mean(fbm_noise)) / std_dev
    else:
        fbm_norm = np.zeros(length)
    return fbm_norm.tolist()


def apply_humanization_to_element(
    m21_element_obj: note.Note | m21chord.Chord,
    template_name: str | None = None,
    custom_params: dict[str, Any] | None = None,
) -> note.Note | m21chord.Chord:
    if not isinstance(m21_element_obj, (note.Note, m21chord.Chord)):
        logger.warning(
            f"Humanizer: apply_humanization_to_element received non-Note/Chord object: {type(m21_element_obj)}"
        )
        return m21_element_obj

    actual_template_name = (
        template_name
        if template_name and template_name in HUMANIZATION_TEMPLATES
        else "default_subtle"
    )
    params = HUMANIZATION_TEMPLATES.get(actual_template_name, {}).copy()

    if custom_params:
        params.update(custom_params)

    element_copy = copy.deepcopy(m21_element_obj)
    time_var = params.get("time_variation", 0.01)
    dur_perc = params.get("duration_percentage", 0.03)
    vel_var = params.get("velocity_variation", 5)
    use_fbm = params.get("use_fbm_time", False)
    fbm_scale = params.get("fbm_time_scale", 0.01)
    fbm_h = params.get("fbm_hurst", 0.6)

    if use_fbm and NUMPY_AVAILABLE:
        time_shift = generate_fractional_noise(1, hurst=fbm_h, scale_factor=fbm_scale)[
            0
        ]
    else:
        if use_fbm and not NUMPY_AVAILABLE:
            logger.debug(
                "Humanizer: FBM time shift requested but NumPy not available. Using uniform random."
            )
        time_shift = random.uniform(-time_var, time_var)

    original_offset = element_copy.offset
    element_copy.offset += time_shift
    if element_copy.offset < 0:
        element_copy.offset = 0.0

    if element_copy.duration:
        original_ql = element_copy.duration.quarterLength
        duration_change = original_ql * random.uniform(-dur_perc, dur_perc)
        new_ql = max(MIN_NOTE_DURATION_QL / 8, original_ql + duration_change)
        try:
            element_copy.duration.quarterLength = new_ql
        except exceptions21.DurationException as e:
            logger.warning(
                f"Humanizer: DurationException for {element_copy}: {e}. Skip dur change."
            )

    notes_to_affect = (
        element_copy.notes
        if isinstance(element_copy, m21chord.Chord)
        else [element_copy]
    )
    for n_obj_affect in notes_to_affect:
        if isinstance(n_obj_affect, note.Note):
            base_vel = (
                n_obj_affect.volume.velocity
                if hasattr(n_obj_affect, "volume")
                and n_obj_affect.volume
                and n_obj_affect.volume.velocity is not None
                else 64
            )
            if isinstance(vel_var, (list, tuple)) and len(vel_var) == 2:
                scale = 1.0 + random.uniform(float(vel_var[0]), float(vel_var[1]))
                final_vel = int(max(1, min(127, round(base_vel * scale))))
            else:
                vel_range = int(vel_var)
                vel_change = random.randint(-vel_range, vel_range)
                final_vel = max(1, min(127, base_vel + vel_change))

            if hasattr(n_obj_affect, "volume") and n_obj_affect.volume is not None:
                n_obj_affect.volume.velocity = final_vel
            else:
                n_obj_affect.volume = volume.Volume(velocity=final_vel)

    return element_copy


def apply_humanization_to_part(
    part_to_humanize: stream.Part,
    template_name: str | None = None,
    custom_params: dict[str, Any] | None = None,
) -> stream.Part:
    if not isinstance(part_to_humanize, stream.Part):
        logger.error(
            "Humanizer: apply_humanization_to_part expects a music21.stream.Part object."
        )
        return part_to_humanize

    # part_to_humanize.id が int の場合もあるので、文字列に変換してから連結する
    if part_to_humanize.id:
        base_id = str(part_to_humanize.id)
        new_id = f"{base_id}_humanized"
    else:
        new_id = "HumanizedPart"
    humanized_part = stream.Part(id=new_id)
    for el_class_item in [
        instrument.Instrument,
        tempo.MetronomeMark,
        meter.TimeSignature,
        key.KeySignature,
        expressions.TextExpression,
    ]:
        for item_el in part_to_humanize.getElementsByClass(el_class_item):
            humanized_part.insert(item_el.offset, copy.deepcopy(item_el))

    elements_to_process = []
    for element_item in part_to_humanize.recurse().notesAndRests:
        elements_to_process.append(element_item)

    elements_to_process.sort(
        key=lambda el_sort: el_sort.getOffsetInHierarchy(part_to_humanize)
    )

    for element_proc in elements_to_process:
        original_hierarchical_offset = element_proc.getOffsetInHierarchy(
            part_to_humanize
        )

        if isinstance(element_proc, (note.Note, m21chord.Chord)):
            humanized_element = apply_humanization_to_element(
                element_proc, template_name, custom_params
            )
            offset_shift_from_humanize = humanized_element.offset - element_proc.offset
            final_insert_offset = (
                original_hierarchical_offset + offset_shift_from_humanize
            )
            if final_insert_offset < 0:
                final_insert_offset = 0.0

            humanized_part.insert(final_insert_offset, humanized_element)
        elif isinstance(element_proc, note.Rest):
            humanized_part.insert(
                original_hierarchical_offset, copy.deepcopy(element_proc)
            )

    return humanized_part


# --- END OF FILE utilities/humanizer.py ---
