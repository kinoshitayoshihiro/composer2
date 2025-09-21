# ui/streamlit/app.py
from __future__ import annotations
import io, sys
from pathlib import Path
import streamlit as st

# --- repo root を import パスに追加（ui/streamlit からでも動く） ---
APP_DIR = Path(__file__).resolve().parent  # ui/streamlit
REPO_ROOT = APP_DIR.parent.parent  # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 想定ファイルの存在チェック（generator 固定 / generators は見ない） ---
candidates = [
    REPO_ROOT / "generator" / "riff_generator.py",
    REPO_ROOT / "generator" / "obligato_generator.py",
    REPO_ROOT / "data" / "riff_library.yaml",
    REPO_ROOT / "data" / "obligato_library.yaml",
]
missing = [p for p in candidates if not p.exists()]

# --- import（generator 固定） ---
try:
    from generator.riff_generator import RiffGenerator
    from generator.obligato_generator import ObligatoGenerator
    from generator.riff_from_vocal import generate_riff_from_vocal
except Exception as e:
    import_err = e
else:
    import_err = None

st.set_page_config(page_title="Riff / Obligato Generator", page_icon="🎸", layout="centered")
st.title("🎸 Riff / 🎼 Obligato Generator (minimal)")

# --- Presets (A案: 王道進行) ---
# Roman numeral をキーへ射影するのが理想だが、まずは実用的な素のコード列を用意
PRESET_PROGRESSIONS = {
    "（選択してください）": "",
    "王道進行 (I–V–vi–IV) / C": "C | G | Am | F",
    "小室進行 (vi–IV–I–V) / C": "Am | F | C | G",
    "循環進行 (ii–V–I) x2 / C": "Dm | G | C | C",
    "カノン風 (C–G–Am–Em–F–C–F–G)": "C | G | Am | Em | F | C | F | G",
    "悲しげ (Am–F–C–G)": "Am | F | C | G",
    "爽やか (C–Em–Am–F)": "C | Em | Am | F",
}

with st.expander("ℹ️ ファイル配置チェック（必要なら開いてください）", expanded=False):
    for p in candidates:
        st.write(("✅" if p.exists() else "❌"), str(p))
    if import_err:
        st.error("generator/ からの import に失敗しました。")
        st.exception(import_err)

# インポート or 必須ファイルが欠けていたら停止
if import_err or missing:
    st.stop()

# --- Sidebar params ---
with st.sidebar:
    st.header("基本設定")
    gen_type = st.selectbox("生成タイプ", ["Riff (背骨)", "Obligato (彩り)", "Riff from Vocal"])
    key = st.text_input("Key（例: A minor / C major）", value="A minor")
    section = st.selectbox("Section", ["Intro", "Verse", "PreChorus", "Chorus", "Bridge"])
    emotion = st.selectbox(
        "Emotion", ["sad", "warm", "neutral", "intense", "reflective", "heroic", "tension"]
    )
    tempo = st.number_input("Tempo (BPM)", min_value=40.0, max_value=200.0, value=78.0, step=1.0)
    bars = st.number_input("Bars（小節数）", min_value=1, max_value=64, value=8, step=1)
    st.divider()
    st.subheader("スタイル（簡易）")
    style = st.selectbox("スタイル（初期は Ballad / Rock）", ["ballad", "rock"])

    st.subheader("進行プリセット（A案）")
    sel_preset = st.selectbox("王道進行を選ぶ", list(PRESET_PROGRESSIONS.keys()), index=0)
    apply_preset = st.button("⬇️ プリセットを進行欄に挿入")

    st.subheader("Humanize / Groove（常時ON・上書き可）")
    hu_profile = st.selectbox(
        "Humanize Profile",
        ["ballad_subtle", "ballad_warm", "rock_tight", "rock_drive"],
        index=0,
    )
    onset_sigma_ms = st.slider("Onset σ (ms)", 0, 20, 5, 1)
    vel_jitter = st.slider("Velocity jitter (±)", 0, 20, 5, 1)
    dur_jitter = st.slider("Duration jitter ratio", 0.0, 0.10, 0.03, 0.01)
    swing = st.slider(
        "Swing (8th)",
        0.0,
        0.3,
        0.10 if style == "ballad" else 0.00,
        0.01,
    )
    groove_name = st.selectbox("Groove Name", ["ballad_8_swing", "rock_16_loose", "(none)"])
    late_ms = st.slider("Late-humanize (ms)", 0, 20, 0, 1)
    if gen_type == "Riff from Vocal":
        st.subheader("Dials（連続つまみ）")
        intensity = st.slider("intensity（力強さ）", 0.0, 1.0, 0.55, 0.05)
        drive = st.slider("drive（推進力/細分化）", 0.0, 1.0, 0.45, 0.05)
        warmth = st.slider("warmth（柔らかさ）", 0.0, 1.0, 0.70, 0.05)
        genre_rf = st.selectbox("From-Vocal用ジャンル", ["ballad", "rock"], index=0)
        st.caption("drive↑で16分寄り、intensity↑で短く大きく、warmth↑でadd9/sus2寄り（rockはtriad少し追加）")

# --- Main: chord progression ---
st.subheader("コード進行（バーごと）")
st.caption("例: `Am | G | F | E`（4/4想定。縦棒で区切ると各バーになります）")
if "prog_text" not in st.session_state:
    st.session_state.prog_text = "Am | G | F | E"
if apply_preset and sel_preset in PRESET_PROGRESSIONS and PRESET_PROGRESSIONS[sel_preset]:
    st.session_state.prog_text = PRESET_PROGRESSIONS[sel_preset]
st.text_area("Progression", height=80, key="prog_text")

flash = st.session_state.pop("prog_text_flash", None)
if flash:
    level, message = flash
    feedback = getattr(st, level, st.info)
    feedback(message)


def parse_progression(text: str, bars: int) -> list[tuple[float, str]]:
    tokens = [t.strip() for t in text.replace("\n", " ").split("|") if t.strip()]
    if not tokens:
        tokens = ["Am"]
    return [(i * 4.0, tokens[i % len(tokens)]) for i in range(max(bars, len(tokens)))]

if gen_type == "Riff from Vocal":
    st.subheader("ボーカルMIDIアップロード（.mid）")
    up = st.file_uploader("Vocal MIDI", type=["mid", "midi"])

col1, col2 = st.columns(2)
with col1:
    if gen_type == "Riff (背骨)":
        default_name = "riff.mid"
    elif gen_type == "Obligato (彩り)":
        default_name = "obligato.mid"
    else:
        default_name = "riff_from_vocal.mid"
    out_name = st.text_input("出力ファイル名", value=default_name)
with col2:
    st.caption("B案: MIDI から進行を抽出")
    mid_src = st.file_uploader("MIDI を選択", type=["mid", "midi"], accept_multiple_files=False)
    extract_btn = st.button("🔎 MIDIから進行を抽出して挿入")
    do_generate = st.button("🎵 生成する", use_container_width=True)

if extract_btn:
    if mid_src is None:
        st.warning("MIDI ファイルを選択してください。")
    else:
        try:
            from utilities.midi_harmony import extract_progression_from_midi

            bars_extracted = extract_progression_from_midi(
                mid_src.getvalue(), key_hint=key, beats_per_bar=4.0
            )
            if bars_extracted:
                st.session_state.prog_text = " | ".join(c for _, c in bars_extracted)
                st.session_state.prog_text_flash = (
                    "success",
                    "MIDI から進行を抽出しました。編集してお使いください。",
                )
                st.experimental_rerun()
            else:
                st.warning("進行を抽出できませんでした（music21 未導入 or 解析不能）。手動で入力してください。")
        except Exception as e:
            st.error(f"抽出中にエラー: {e}")

prog_text = st.session_state.prog_text
chord_seq = parse_progression(prog_text, int(bars))

if do_generate:
    try:
        data_dir = REPO_ROOT / "data"
        if gen_type == "Riff (背骨)":
            rg = RiffGenerator(
                instrument="guitar", patterns_yaml=str(data_dir / "riff_library.yaml")
            )
            pm = rg.generate(
                key=key,
                tempo=float(tempo),
                emotion=emotion,
                section=section,
                chord_seq=chord_seq,
                bars=int(bars),
                style=style,
                humanize={
                    "onset_sigma_ms": onset_sigma_ms,
                    "velocity_jitter": vel_jitter,
                    "duration_jitter_ratio": dur_jitter,
                },
                humanize_profile=hu_profile,
                quantize={"grid": 0.25, "swing": swing},
                groove=None if groove_name == "(none)" else {"name": groove_name},
                late_humanize_ms=late_ms,
            )
        elif gen_type == "Obligato (彩り)":
            og = ObligatoGenerator(
                instrument="synth", patterns_yaml=str(data_dir / "obligato_library.yaml")
            )
            pm = og.generate(
                key=key,
                tempo=float(tempo),
                emotion=emotion,
                section=section,
                chord_seq=chord_seq,
                bars=int(bars),
                humanize={
                    "onset_sigma_ms": onset_sigma_ms,
                    "velocity_jitter": vel_jitter,
                    "duration_jitter_ratio": dur_jitter,
                },
                humanize_profile=hu_profile,
                quantize={"grid": 0.25, "swing": swing},
                groove=None if groove_name == "(none)" else {"name": groove_name},
                late_humanize_ms=late_ms,
            )
        else:
            if up is None:
                st.error("Vocal MIDI をアップロードしてください。")
                st.stop()
            dials = {"intensity": intensity, "drive": drive, "warmth": warmth}
            pm = generate_riff_from_vocal(
                up.getvalue(),
                chord_seq=chord_seq,
                tempo=float(tempo),
                genre=genre_rf,
                bars=int(bars),
                dials=dials,
                humanize={
                    "onset_sigma_ms": onset_sigma_ms,
                    "velocity_jitter": vel_jitter,
                    "duration_jitter_ratio": dur_jitter,
                },
                humanize_profile=hu_profile,
                quantize={"grid": 0.25, "swing": swing},
                groove=None if groove_name == "(none)" else {"name": groove_name},
                late_humanize_ms=late_ms,
            )

        buf = io.BytesIO()
        out_path = (REPO_ROOT / "outputs" / out_name).with_suffix(".mid")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(out_path))
        with open(out_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)

        st.success(f"生成しました → {out_path}")
        st.download_button(
            "⬇️ MIDIをダウンロード", data=buf, file_name=out_path.name, mime="audio/midi"
        )
        st.caption("生成に使ったコード（バー開始拍 / シンボル）")
        st.table({"bar_start_beat": [b for b, _ in chord_seq], "chord": [c for _, c in chord_seq]})
    except FileNotFoundError:
        st.error(
            "YAML が見つかりません。`data/riff_library.yaml` と `data/obligato_library.yaml` を配置してください。"
        )
    except Exception as e:
        st.exception(e)
