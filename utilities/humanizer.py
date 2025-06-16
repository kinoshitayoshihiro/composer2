# --- START OF FILE utilities/humanizer.py (役割特化版) ---
import music21  # name 'music21' is not defined エラー対策
import random, logging
import math
import copy
from typing import List, Dict, Any, Union, Optional, cast
from pathlib import Path

# music21 のサブモジュールを正しい形式でインポート
import music21.note as note
import music21.chord as m21chord  # check_imports.py の期待する形式 (スペースに注意)
import music21.volume as volume
import music21.duration as duration
import music21.pitch as pitch
import music21.stream as stream
import music21.instrument as instrument
import music21.tempo as tempo
import music21.meter as meter
import music21.key as key
import music21.expressions as expressions
from music21 import exceptions21

# MIN_NOTE_DURATION_QL は core_music_utils からインポートすることを推奨
try:
    from .core_music_utils import MIN_NOTE_DURATION_QL
except ImportError:
    MIN_NOTE_DURATION_QL = 0.125

logger = logging.getLogger("otokotoba.humanizer")

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
_PROFILE_REGISTRY: Dict[str, Dict[str, Any]] = {}
PROFILES = _PROFILE_REGISTRY


def load_profiles(dict_like: Dict[str, Any]) -> None:
    """
    YAML から読み取った humanize_profiles セクションを登録する。
    例: cfg['humanize_profiles'] をそのまま渡す。
    """
    global _PROFILE_REGISTRY
    _PROFILE_REGISTRY = dict_like
    logger.info(f"Loaded {len(_PROFILE_REGISTRY)} humanize profiles")


def get_profile(name: str) -> Dict[str, Any]:
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
    swing_ratio: Optional[float] = None,
) -> None:
    """music21.stream.Part に in-place でヒューマナイズを適用"""
    prof = get_profile(profile_name) if profile_name else {}
    off = prof.get("offset_ms", {})  # {mean, stdev}
    vel = prof.get("velocity", {})
    dur = prof.get("duration_pct", {})

    for n in part_stream.flatten().notes:
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


def _apply_swing(part_stream: stream.Part, swing_ratio: float, subdiv: int = 8) -> None:
    """Shift off-beats according to ``swing_ratio`` and grid ``subdiv``.

    ``swing_ratio`` represents the relative position of the off-beat note
    within a pair (0.5 means straight). ``subdiv`` describes the number of
    divisions in a 4/4 measure. 8 results in eighth‑note swing.
    """
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


def apply_swing(part: stream.Part, ratio: float, subdiv: int = 8) -> None:
    """Public API to apply swing in-place.

    Parameters
    ----------
    part : :class:`music21.stream.Part`
        Target part to modify.
    ratio : float
        Relative position of the off-beat note (0.5 = straight).
    subdiv : int
        Number of grid subdivisions per measure. ``8`` for typical eighth swing.
    """
    if ratio is None or abs(ratio) < 1e-6:
        return

    _apply_swing(part, float(ratio), subdiv=subdiv)


def apply_envelope(part: stream.Part, start: int, end: int, scale: float) -> None:
    """Scale note velocities between start and end beats."""
    for n in part.recurse().notes:
        if start <= n.offset < end and n.volume and n.volume.velocity is not None:
            n.volume.velocity = int(max(1, min(127, round(n.volume.velocity * scale))))


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
    import sys, yaml

    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    load_profiles(cfg["humanize_profiles"])
    print("Profiles loaded:", list(_PROFILE_REGISTRY))


HUMANIZATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
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
}


try:
    import numpy

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def generate_fractional_noise(
    length: int, hurst: float = 0.7, scale_factor: float = 1.0
) -> List[float]:
    if not NUMPY_AVAILABLE or np is None:
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
    m21_element_obj: Union[note.Note, m21chord.Chord],
    template_name: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Union[note.Note, m21chord.Chord]:
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
            vel_change = random.randint(-vel_var, vel_var)
            final_vel = max(1, min(127, base_vel + vel_change))
            if hasattr(n_obj_affect, "volume") and n_obj_affect.volume is not None:
                n_obj_affect.volume.velocity = final_vel
            else:
                n_obj_affect.volume = volume.Volume(velocity=final_vel)

    return element_copy


def apply_humanization_to_part(
    part_to_humanize: stream.Part,
    template_name: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None,
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
