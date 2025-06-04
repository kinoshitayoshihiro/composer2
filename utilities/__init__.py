# --- START OF FILE utilities/__init__.py (get_music21_chord_object削除版) ---
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

__all__ = [
    "MIN_NOTE_DURATION_QL", "get_time_signature_object", "sanitize_chord_label", # "get_music21_chord_object" を削除
    "build_scale_object", "ScaleRegistry",
    "generate_fractional_noise", "apply_humanization_to_element", "apply_humanization_to_part",
    "HUMANIZATION_TEMPLATES", "NUMPY_AVAILABLE",
]
# --- END OF FILE utilities/__init__.py ---