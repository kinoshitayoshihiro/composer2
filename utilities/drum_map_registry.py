"""Registry of drum-label mappings → (description, GM-note)."""

from __future__ import annotations

from typing import Dict, Tuple, Union

# ---------------------------------------------------------------------------
# ① 共通で使う定数
# ---------------------------------------------------------------------------

HH_EDGE: Tuple[str, int] = ("closed_hi_hat_edge", 22)
HH_PEDAL: Tuple[str, int] = ("pedal_hi_hat", 44)
SNARE_RUFF: Tuple[str, int] = ("acoustic_snare", 38)

# ---------------------------------------------------------------------------
# ② GM ベース (infra/zero-green 起点)
# ---------------------------------------------------------------------------

GM_DRUM_MAP: Dict[str, Tuple[str, int]] = {
    # Hi-hat
    "chh": ("closed_hi_hat", 42),
    "hh": ("closed_hi_hat", 42),
    "hat_closed": ("closed_hi_hat", 42),
    "hh_edge": HH_EDGE,
    "ohh": ("open_hi_hat", 46),
    "hh_pedal": HH_PEDAL,
    # Kick / Snare
    "kick": ("acoustic_bass_drum", 36),
    "snare": ("acoustic_snare", 38),
    "ghost_snare": ("acoustic_snare", 38),
    "snare_ruff": SNARE_RUFF,
    # Toms
    "tom1": ("high_tom", 48),
    "tom2": ("mid_tom", 47),
    "tom3": ("low_tom", 45),
    # Cymbals
    "crash": ("crash_cymbal_1", 49),
    "crash_cymbal_soft_swell": ("crash_cymbal_1", 49),
    "ride": ("ride_cymbal_1", 51),
    "ride_cymbal_swell": ("ride_cymbal_1", 51),
    # FX
    "chimes": ("triangle", 81),
    "shaker_soft": ("shaker", 82),
    # Brushes
    "brush_kick": ("acoustic_bass_drum", 36),
    "brush_snare": ("acoustic_snare", 38),
    "snare_brush": ("electric_snare", 40),
    # Alias
    "ghost": ("closed_hi_hat", 42),
}

# ---------------------------------------------------------------------------
# ③ UJAM LEGEND (値はプラグイン note、説明は GM に倣う)
# ---------------------------------------------------------------------------

UJAM_LEGEND_MAP: Dict[str, Tuple[str, int]] = {
    **GM_DRUM_MAP,  # まず GM をベースにコピー
    # ほんの数か所だけ UJAM 独自 note に置換
    "kick": ("kick", 36),
    "snare": ("snare", 38),
    "tom1": ("high_tom", 50),   # high tom は 50
    # …
}

# ---------------------------------------------------------------------------
# ④ BeatMaker V3（main ブランチ側の追加マップ）
#     main 側は値 = int だけだったので、説明文字列を付与して tuple 化
# ---------------------------------------------------------------------------

def _mk(desc: str, note: int) -> Tuple[str, int]:
    return desc, note


BEATMAKER_V3_MAP: Dict[str, Tuple[str, int]] = {
    "kick": _mk("kick", 36),
    "snare": _mk("snare", 38),
    "closed_hi_hat": _mk("closed_hi_hat", 42),
    "chh": _mk("closed_hi_hat", 42),
    "open_hi_hat": _mk("open_hi_hat", 46),
    "ohh": _mk("open_hi_hat", 46),
    "pedal_hi_hat": _mk("pedal_hi_hat", 44),
    "phh": _mk("pedal_hi_hat", 44),
    "clap": _mk("hand_clap", 39),
    "ride": _mk("ride_cymbal_1", 51),
    "crash": _mk("crash_cymbal_1", 49),
    "tom_low": _mk("low_tom", 45),
    "tom_mid": _mk("mid_tom", 47),
    "tom_hi": _mk("high_tom", 50),
}

# ---------------------------------------------------------------------------
# ⑤ Fallback エイリアス表（infra/zero-green 起点）
# ---------------------------------------------------------------------------

MISSING_DRUM_MAP_FALLBACK = {
    "hh": "chh",
    "hat_closed": "chh",
    "ghost": "chh",
    "shaker_soft": "shaker_soft",
    "chimes": "chimes",
    "ride_cymbal_swell": "ride_cymbal_swell",
    "crash_cymbal_soft_swell": "crash_cymbal_soft_swell",
}

# ---------------------------------------------------------------------------
# ⑥ マップレジストリ & API
# ---------------------------------------------------------------------------

DRUM_MAPS: Dict[str, Dict[str, Tuple[str, int]]] = {
    "gm": GM_DRUM_MAP,
    "ujam_legend": UJAM_LEGEND_MAP,
    "beatmaker_v3": BEATMAKER_V3_MAP,
}


def _as_int_map(mapping: Dict[str, Tuple[str, int]]) -> Dict[str, int]:
    """int だけ欲しい場合に変換して返す."""
    return {k: v[1] if isinstance(v, tuple) else v for k, v in mapping.items()}


def get_drum_map(name: str, *, return_int_only: bool = False) -> Dict[str, Union[int, Tuple[str, int]]]:
    """
    Drum マップを取得するユーティリティ.

    Parameters
    ----------
    name
        `"gm" / "ujam_legend" / "beatmaker_v3"` など（大小無視）
    return_int_only
        True の場合、値を int に丸めて返す（後方互換用）

    Raises
    ------
    KeyError
        未登録の name を渡した場合
    """
    mapping = DRUM_MAPS[name.lower()]  # KeyError を自然に吐かせる
    return _as_int_map(mapping) if return_int_only else mapping
