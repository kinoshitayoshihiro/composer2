# --- START OF FILE utilities/scale_registry.py (ペンタトニック互換性向上版) ---
import logging
from typing import Optional, Dict, Any, List, Tuple

# music21 のサブモジュールを個別にインポート
from music21 import pitch
from music21 import scale # music21.scale モジュールをインポート
from music21 import common # common.classTools.isNum などで使う可能性（今回は未使用）

logger = logging.getLogger(__name__)

# スケールオブジェクトをキャッシュするための辞書 (モジュールレベル)
_scale_cache: Dict[Tuple[str, str], scale.ConcreteScale] = {}

def build_scale_object(tonic_str: Optional[str], mode_str: Optional[str]) -> scale.ConcreteScale:
    """
    指定されたトニックとモードに基づいてmusic21のScaleオブジェクトを生成またはキャッシュから取得します。
    ペンタトニックやブルーススケールはAbstractScaleから派生させることで互換性を高めています。
    """
    tonic_name = (tonic_str or "C").capitalize()
    mode_name = (mode_str or "major").lower()

    cache_key = (tonic_name, mode_name)
    if cache_key in _scale_cache:
        logger.debug(f"ScaleRegistry: Returning cached scale for {tonic_name} {mode_name}.")
        return _scale_cache[cache_key]

    try:
        tonic_p = pitch.Pitch(tonic_name)
    except Exception as e_tonic:
        logger.error(f"ScaleRegistry: Invalid tonic '{tonic_name}': {e_tonic}. Defaulting to C.")
        tonic_p = pitch.Pitch("C")
    
    scl_obj: Optional[scale.ConcreteScale] = None

    # 基本的なスケールクラスのマッピング
    mode_map_basic: Dict[str, Any] = {
        "ionian": scale.MajorScale, "major": scale.MajorScale,
        "dorian": scale.DorianScale, "phrygian": scale.PhrygianScale,
        "lydian": scale.LydianScale, "mixolydian": scale.MixolydianScale,
        "aeolian": scale.MinorScale, "natural_minor": scale.MinorScale, "minor": scale.MinorScale,
        "locrian": scale.LocrianScale,
        "harmonicminor": scale.HarmonicMinorScale, "harmonic_minor": scale.HarmonicMinorScale,
        "melodicminor": scale.MelodicMinorScale, "melodic_minor": scale.MelodicMinorScale,
        "wholetone": scale.WholeToneScale, "whole_tone": scale.WholeToneScale,
        "chromatic": scale.ChromaticScale,
    }
    
    scl_cls = mode_map_basic.get(mode_name)

    if scl_cls:
        try:
            scl_obj = scl_cls(tonic_p)
        except Exception as e_create_basic:
            logger.error(f"ScaleRegistry: Error creating basic scale '{mode_name}' for {tonic_p.name}: {e_create_basic}. Fallback to MajorScale.", exc_info=True)
            scl_obj = scale.MajorScale(pitch.Pitch("C")) # 緊急フォールバック

    # 特殊なスケールをAbstractScaleから定義
    elif mode_name in ["majorpentatonic", "major_pentatonic"]:
        try:
            # メジャーペンタトニックのインターバル (半音単位でルートから): 0, 2, 4, 7, 9
            abstract_scale = scale.AbstractScale([0, 2, 4, 7, 9])
            # AbstractScaleからConcreteScaleを派生させる
            # music21のバージョンによってderiveの挙動が異なる可能性を考慮
            # deriveメソッドがConcreteScaleを返すことを期待
            derived = abstract_scale.derive(tonic_p)
            if isinstance(derived, scale.ConcreteScale):
                scl_obj = derived
            else: # deriveが予期せぬ型を返した場合 (古いバージョンなど)
                logger.warning(f"ScaleRegistry: AbstractScale.derive() did not return ConcreteScale for major pentatonic. Attempting manual pitch list creation.")
                pitches_list = [tonic_p.transpose(i) for i in abstract_scale.abstract._net] # _netは内部構造だがアクセス試行
                scl_obj = scale.ConcreteScale(pitches=pitches_list)
        except AttributeError as e_abs_attr: # AbstractScaleがない、または必要なメソッドがない場合
             logger.warning(f"ScaleRegistry: Cannot create Major Pentatonic for {tonic_name} due to missing music21.scale.AbstractScale features: {e_abs_attr}. Fallback to MajorScale.")
             scl_obj = scale.MajorScale(tonic_p)
        except Exception as e_penta:
             logger.warning(f"ScaleRegistry: Error creating Major Pentatonic for {tonic_name} {mode_name}: {e_penta}. Fallback to MajorScale.", exc_info=True)
             scl_obj = scale.MajorScale(tonic_p)

    elif mode_name in ["minorpentatonic", "minor_pentatonic"]:
        try:
            # マイナーペンタトニックのインターバル: 0, 3, 5, 7, 10
            abstract_scale = scale.AbstractScale([0, 3, 5, 7, 10])
            derived = abstract_scale.derive(tonic_p)
            if isinstance(derived, scale.ConcreteScale):
                scl_obj = derived
            else:
                logger.warning(f"ScaleRegistry: AbstractScale.derive() did not return ConcreteScale for minor pentatonic. Attempting manual pitch list creation.")
                pitches_list = [tonic_p.transpose(i) for i in abstract_scale.abstract._net]
                scl_obj = scale.ConcreteScale(pitches=pitches_list)
        except AttributeError as e_abs_attr:
             logger.warning(f"ScaleRegistry: Cannot create Minor Pentatonic for {tonic_name} due to missing music21.scale.AbstractScale features: {e_abs_attr}. Fallback to MinorScale.")
             scl_obj = scale.MinorScale(tonic_p)
        except Exception as e_minor_penta:
             logger.warning(f"ScaleRegistry: Error creating Minor Pentatonic for {tonic_name} {mode_name}: {e_minor_penta}. Fallback to MinorScale.", exc_info=True)
             scl_obj = scale.MinorScale(tonic_p)

    elif mode_name in ["blues", "blues_scale"]: # ブルーススケール (一般的なヘキサトニック)
        try:
            # ブルーススケールのインターバル (例): 0, 3, 5, 6, 7, 10
            abstract_scale = scale.AbstractScale([0, 3, 5, 6, 7, 10])
            derived = abstract_scale.derive(tonic_p)
            if isinstance(derived, scale.ConcreteScale):
                scl_obj = derived
            else:
                logger.warning(f"ScaleRegistry: AbstractScale.derive() did not return ConcreteScale for blues scale. Attempting manual pitch list creation.")
                pitches_list = [tonic_p.transpose(i) for i in abstract_scale.abstract._net]
                scl_obj = scale.ConcreteScale(pitches=pitches_list)
        except AttributeError as e_abs_attr:
            logger.warning(f"ScaleRegistry: Cannot create Blues scale for {tonic_name} due to missing music21.scale.AbstractScale features: {e_abs_attr}. Fallback to MinorScale (as approximation).")
            scl_obj = scale.MinorScale(tonic_p)
        except Exception as e_blues:
            logger.warning(f"ScaleRegistry: Error creating Blues scale for {tonic_name} {mode_name}: {e_blues}. Fallback to MinorScale.", exc_info=True)
            scl_obj = scale.MinorScale(tonic_p)
            
    # 上記のいずれにも一致しない場合、またはscl_objがNoneのままの場合
    if scl_obj is None:
        # 動的ルックアップを試みる (前回のコードより)
        try:
            scale_class_name = mode_name.capitalize().replace("_", "") + "Scale"
            if not scale_class_name.endswith("Scale"):
                 scale_class_name += "Scale"
            
            scl_cls_dynamic = getattr(scale, scale_class_name, None)
            if scl_cls_dynamic and issubclass(scl_cls_dynamic, scale.Scale):
                scl_obj = scl_cls_dynamic(tonic_p)
                logger.info(f"ScaleRegistry: Created scale {scl_obj} for {tonic_name} {mode_name} (dynamic lookup: {scale_class_name}).")
            else:
                raise AttributeError(f"Scale class '{scale_class_name}' not found via dynamic lookup or not a valid scale.")
        except (AttributeError, TypeError, Exception) as e_dyn_scale:
            logger.warning(f"ScaleRegistry: Truly unknown mode '{mode_name}' (dynamic lookup failed: {e_dyn_scale}). Using MajorScale for {tonic_name} as final fallback.")
            scl_obj = scale.MajorScale(tonic_p)

    if scl_obj: # scl_obj が正常に作成された場合のみキャッシュ
        logger.info(f"ScaleRegistry: Created and cached scale {scl_obj} for {tonic_name} {mode_name}.")
        _scale_cache[cache_key] = scl_obj
        return scl_obj
    else: # 万が一 scl_obj が None のままだった場合 (ありえないはずだが念のため)
        logger.error(f"ScaleRegistry: Failed to create any scale for {tonic_name} {mode_name}. Returning emergency C Major.")
        emergency_fallback = scale.MajorScale(pitch.Pitch("C"))
        _scale_cache[cache_key] = emergency_fallback # エラーでもキャッシュ
        return emergency_fallback


class ScaleRegistry:
    @staticmethod
    def get(tonic: str, mode: str) -> scale.ConcreteScale:
        """指定されたトニックとモードの music21.scale.ConcreteScale オブジェクトを取得します。"""
        return build_scale_object(tonic, mode)

    @staticmethod
    def get_pitches(tonic: str, mode: str, min_octave: int = 2, max_octave: int = 5) -> List[pitch.Pitch]:
        """指定された範囲のスケール構成音を取得します。"""
        scl = build_scale_object(tonic, mode)
        try:
            if not isinstance(scl.tonic, pitch.Pitch): # scl.tonic がNoneや文字列でないことを確認
                logger.error(f"ScaleRegistry (get_pitches): Tonic for scale {tonic} {mode} is not a Pitch object: {scl.tonic}. Using C.")
                effective_tonic_name = "C"
            else:
                effective_tonic_name = scl.tonic.name

            p_start = pitch.Pitch(f'{effective_tonic_name}{min_octave}')
            p_end = pitch.Pitch(f'{effective_tonic_name}{max_octave+1}') # max_octaveを含むように+1
            
            # getPitchesが空を返す場合があるのでチェック
            pitches_result = scl.getPitches(p_start, p_end)
            if not pitches_result:
                 logger.warning(f"ScaleRegistry: getPitches returned empty for {tonic} {mode} in range {min_octave}-{max_octave}. Scale might be abstract or range too narrow.")
                 # AbstractScaleの場合、deriveAll() や getPitchFromDegree() を使う必要があるかもしれない
                 if hasattr(scl, 'deriveAll'): # AbstractScaleの場合
                     return scl.deriveAll(tonicPitch=p_start, minPitch=p_start, maxPitch=p_end)

            return pitches_result

        except Exception as e_get_pitches:
            logger.error(f"ScaleRegistry: Error in get_pitches for {tonic} {mode}: {e_get_pitches}. Returning empty list.", exc_info=True)
            return []

    @staticmethod
    def mode_tensions(mode: str) -> List[int]:
        # (変更なし)
        mode_lower = mode.lower()
        if mode_lower in ["major", "ionian", "lydian"]: return [2, 6, 9, 11, 13] # 9th, 13th (6th), Lydian #11
        if mode_lower in ["minor", "aeolian", "dorian", "phrygian"]: return [2, 4, 6, 9, 11, 13] # 9th, 11th, 13th (Dorian 6th)
        if mode_lower == "mixolydian": return [2, 4, 6, 9, 11, 13] # 9th, 13th (6th)
        return [2, 4, 6] # General fallback

    @staticmethod
    def avoid_degrees(mode: str) -> List[int]:
        # (変更なし)
        mode_lower = mode.lower()
        if mode_lower == "major": return [4] # Avoid F in C major scale as a strong melodic point against tonic
        if mode_lower == "dorian": return [] # Dorian is quite stable
        if mode_lower == "phrygian": return [2, 6] # b2 and b6 can be very characteristic but also dissonant
        if mode_lower == "lydian": return [] # #4 is characteristic
        if mode_lower == "mixolydian": return [4] # Same as major, the 4th can clash with the dominant feel
        if mode_lower == "aeolian": return [6] # b6 can be dissonant
        if mode_lower == "locrian": return [1, 2, 3, 4, 5, 6, 7] # The entire scale is unstable
        return []
# --- END OF FILE utilities/scale_registry.py (ペンタトニック互換性向上版) ---
