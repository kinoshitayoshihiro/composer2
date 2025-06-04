# --- START OF FILE utilities/humanizer.py (役割特化版) ---
import music21 # name 'music21' is not defined エラー対策
import random
import math
import copy
from typing import List, Dict, Any, Union, Optional, cast 

# music21 のサブモジュールを正しい形式でインポート
import music21.note as note 
import music21.chord      as m21chord # check_imports.py の期待する形式 (スペースに注意)
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

import logging
logger = logging.getLogger(__name__)

NUMPY_AVAILABLE = False
np = None
try:
    import numpy
    np = numpy
    NUMPY_AVAILABLE = True
    logger.info("Humanizer: NumPy found. Fractional noise generation is enabled.")
except ImportError:
    logger.warning("Humanizer: NumPy not found. Fractional noise will use Gaussian fallback.")

def generate_fractional_noise(length: int, hurst: float = 0.7, scale_factor: float = 1.0) -> List[float]:
    if not NUMPY_AVAILABLE or np is None:
        logger.debug(f"Humanizer (FBM): NumPy not available. Using Gaussian noise for length {length}.")
        return [random.gauss(0, scale_factor / 3) for _ in range(length)] 
    if length <= 0: return []
    white_noise = np.random.randn(length)
    fft_white = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(length)
    freqs[0] = 1e-6 if freqs.size > 0 and freqs[0] == 0 else freqs[0]
    filter_amplitude = np.abs(freqs) ** (-hurst)
    if freqs.size > 0: filter_amplitude[0] = 0
    fft_fbm = fft_white * filter_amplitude
    fbm_noise = np.fft.ifft(fft_fbm).real
    std_dev = np.std(fbm_noise)
    if std_dev != 0: fbm_norm = scale_factor * (fbm_noise - np.mean(fbm_noise)) / std_dev
    else: fbm_norm = np.zeros(length)
    return fbm_norm.tolist()

HUMANIZATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default_subtle": {"time_variation": 0.01, "duration_percentage": 0.03, "velocity_variation": 5, "use_fbm_time": False},
    "piano_gentle_arpeggio": {"time_variation": 0.008, "duration_percentage": 0.02, "velocity_variation": 4, "use_fbm_time": True, "fbm_time_scale": 0.005, "fbm_hurst": 0.7},
    "piano_block_chord": {"time_variation": 0.015, "duration_percentage": 0.04, "velocity_variation": 7, "use_fbm_time": False},
    "drum_tight": {"time_variation": 0.005, "duration_percentage": 0.01, "velocity_variation": 3, "use_fbm_time": False},
    "drum_loose_fbm": {"time_variation": 0.02, "duration_percentage": 0.05, "velocity_variation": 8, "use_fbm_time": True, "fbm_time_scale": 0.01, "fbm_hurst": 0.6},
    "guitar_strum_loose": {"time_variation": 0.025, "duration_percentage": 0.06, "velocity_variation": 10, "use_fbm_time": True, "fbm_time_scale": 0.015},
    "guitar_arpeggio_precise": {"time_variation": 0.008, "duration_percentage": 0.02, "velocity_variation": 4, "use_fbm_time": False},
    "vocal_ballad_smooth": {"time_variation": 0.025, "duration_percentage": 0.05, "velocity_variation": 4, "use_fbm_time": True, "fbm_time_scale": 0.01, "fbm_hurst": 0.7},
    "vocal_pop_energetic": {"time_variation": 0.015, "duration_percentage": 0.02, "velocity_variation": 8, "use_fbm_time": True, "fbm_time_scale": 0.008},
}

def apply_humanization_to_element(
    m21_element_obj: Union[note.Note, m21chord.Chord], 
    template_name: Optional[str] = None, 
    custom_params: Optional[Dict[str, Any]] = None
) -> Union[note.Note, m21chord.Chord]: 
    if not isinstance(m21_element_obj, (note.Note, m21chord.Chord)): 
        logger.warning(f"Humanizer: apply_humanization_to_element received non-Note/Chord object: {type(m21_element_obj)}")
        return m21_element_obj

    actual_template_name = template_name if template_name and template_name in HUMANIZATION_TEMPLATES else "default_subtle"
    params = HUMANIZATION_TEMPLATES.get(actual_template_name, {}).copy()
    
    if custom_params: 
        params.update(custom_params)

    element_copy = copy.deepcopy(m21_element_obj)
    time_var = params.get('time_variation', 0.01)
    dur_perc = params.get('duration_percentage', 0.03)
    vel_var = params.get('velocity_variation', 5)
    use_fbm = params.get('use_fbm_time', False)
    fbm_scale = params.get('fbm_time_scale', 0.01)
    fbm_h = params.get('fbm_hurst', 0.6)

    if use_fbm and NUMPY_AVAILABLE:
        time_shift = generate_fractional_noise(1, hurst=fbm_h, scale_factor=fbm_scale)[0]
    else:
        if use_fbm and not NUMPY_AVAILABLE: logger.debug("Humanizer: FBM time shift requested but NumPy not available. Using uniform random.")
        time_shift = random.uniform(-time_var, time_var)
    
    original_offset = element_copy.offset
    element_copy.offset += time_shift
    if element_copy.offset < 0: element_copy.offset = 0.0

    if element_copy.duration: 
        original_ql = element_copy.duration.quarterLength
        duration_change = original_ql * random.uniform(-dur_perc, dur_perc)
        new_ql = max(MIN_NOTE_DURATION_QL / 8, original_ql + duration_change)
        try: element_copy.duration.quarterLength = new_ql
        except exceptions21.DurationException as e: logger.warning(f"Humanizer: DurationException for {element_copy}: {e}. Skip dur change.") 

    notes_to_affect = element_copy.notes if isinstance(element_copy, m21chord.Chord) else [element_copy] 
    for n_obj_affect in notes_to_affect: 
        if isinstance(n_obj_affect, note.Note): 
            base_vel = n_obj_affect.volume.velocity if hasattr(n_obj_affect, 'volume') and n_obj_affect.volume and n_obj_affect.volume.velocity is not None else 64 
            vel_change = random.randint(-vel_var, vel_var)
            final_vel = max(1, min(127, base_vel + vel_change))
            if hasattr(n_obj_affect, 'volume') and n_obj_affect.volume is not None: n_obj_affect.volume.velocity = final_vel 
            else: n_obj_affect.volume = volume.Volume(velocity=final_vel) 
            
    return element_copy

def apply_humanization_to_part(
    part_to_humanize: stream.Part, 
    template_name: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None
) -> stream.Part: 
    if not isinstance(part_to_humanize, stream.Part): 
        logger.error("Humanizer: apply_humanization_to_part expects a music21.stream.Part object.")
        return part_to_humanize 

    # part_to_humanize.id が int の場合もあるので、文字列に変換してから連結する
    if part_to_humanize.id:
        base_id = str(part_to_humanize.id)
        new_id = f"{base_id}_humanized"
    else:
        new_id = "HumanizedPart"
    humanized_part = stream.Part(id=new_id)    
    for el_class_item in [instrument.Instrument, tempo.MetronomeMark, meter.TimeSignature, key.KeySignature, expressions.TextExpression]: 
        for item_el in part_to_humanize.getElementsByClass(el_class_item): 
            humanized_part.insert(item_el.offset, copy.deepcopy(item_el)) 

    elements_to_process = []
    for element_item in part_to_humanize.recurse().notesAndRests:  
        elements_to_process.append(element_item)
    
    elements_to_process.sort(key=lambda el_sort: el_sort.getOffsetInHierarchy(part_to_humanize)) 


    for element_proc in elements_to_process: 
        original_hierarchical_offset = element_proc.getOffsetInHierarchy(part_to_humanize)
        
        if isinstance(element_proc, (note.Note, m21chord.Chord)): 
            humanized_element = apply_humanization_to_element(element_proc, template_name, custom_params)
            offset_shift_from_humanize = humanized_element.offset - element_proc.offset 
            final_insert_offset = original_hierarchical_offset + offset_shift_from_humanize
            if final_insert_offset < 0: final_insert_offset = 0.0
            
            humanized_part.insert(final_insert_offset, humanized_element)
        elif isinstance(element_proc, note.Rest): 
            humanized_part.insert(original_hierarchical_offset, copy.deepcopy(element_proc))
        
    return humanized_part
# --- END OF FILE utilities/humanizer.py ---
