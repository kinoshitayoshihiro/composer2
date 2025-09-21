# ui/streamlit/app.py
from __future__ import annotations
import io, sys, os
from pathlib import Path
import streamlit as st

# --- make repo root importable even when running from ui/streamlit ---
APP_DIR = Path(__file__).resolve().parent  # ui/streamlit
REPO_ROOT = APP_DIR.parent.parent  # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- diagnose file placement ---
candidates = [
    REPO_ROOT / "generator" / "riff_generator.py",
    REPO_ROOT / "generator" / "obligato_generator.py",
    REPO_ROOT / "generators" / "riff_generator.py",
    REPO_ROOT / "generators" / "obligato_generator.py",
    REPO_ROOT / "data" / "riff_library.yaml",
    REPO_ROOT / "data" / "obligato_library.yaml",
]
missing = [p for p in candidates if not p.exists()]

# --- import with fallback: generator -> generators ---
RiffGenerator = ObligatoGenerator = None
import_err = None
try:
    from generator.riff_generator import RiffGenerator as RG1
    from generator.obligato_generator import ObligatoGenerator as OG1

    RiffGenerator, ObligatoGenerator = RG1, OG1
except Exception as e1:
    try:
        from generators.riff_generator import RiffGenerator as RG2
        from generators.obligato_generator import ObligatoGenerator as OG2

        RiffGenerator, ObligatoGenerator = RG2, OG2
    except Exception as e2:
        import_err = (e1, e2)

st.set_page_config(page_title="Riff / Obligato Generator", page_icon="🎸", layout="centered")
st.title("🎸 Riff / 🎼 Obligato Generator (minimal)")

# Helpful diagnostics
with st.expander("ℹ️ ファイル配置チェック（必要なら開いてください）", expanded=False):
    for p in candidates:
        st.write(("✅" if p.exists() else "❌"), str(p))
    if import_err:
        st.error(
            "インポートに失敗しました。generator/ か generators/ のいずれかに配置してください。"
        )
        st.exception(import_err[0])
        st.exception(import_err[1])

if (RiffGenerator is None) or (ObligatoGenerator is None):
    st.stop()

# --- Sidebar params ---
with st.sidebar:
    st.header("基本設定")
    gen_type = st.selectbox("生成タイプ", ["Riff (背骨)", "Obligato (彩り)"])
    key = st.text_input("Key（例: A minor / C major）", value="A minor")
    section = st.selectbox("Section", ["Verse", "PreChorus", "Chorus", "Bridge"])
    emotion = st.selectbox(
        "Emotion", ["sad", "warm", "neutral", "intense", "reflective", "heroic", "tension"]
    )
    tempo = st.number_input("Tempo (BPM)", min_value=40.0, max_value=200.0, value=78.0, step=1.0)
    bars = st.number_input("Bars（小節数）", min_value=1, max_value=64, value=8, step=1)
    st.divider()
    st.subheader("スタイル（簡易）")
    style = st.selectbox("スタイル（初期は Ballad / Rock）", ["ballad", "rock"])

# --- Main: chord progression ---
st.subheader("コード進行（バーごと）")
st.caption("例: `Am | G | F | E`（4/4想定。縦棒で区切ると各バーになります）")
prog_text = st.text_area("Progression", value="Am | G | F | E", height=80)


def parse_progression(text: str, bars: int) -> list[tuple[float, str]]:
    tokens = [t.strip() for t in text.replace("\n", " ").split("|") if t.strip()]
    if not tokens:
        tokens = ["Am"]
    return [(i * 4.0, tokens[i % len(tokens)]) for i in range(max(bars, len(tokens)))]


chord_seq = parse_progression(prog_text, int(bars))

col1, col2 = st.columns(2)
with col1:
    default_name = "riff.mid" if gen_type.startswith("Riff") else "obligato.mid"
    out_name = st.text_input("出力ファイル名", value=default_name)
with col2:
    st.write("")
    do_generate = st.button("🎵 生成する", use_container_width=True)

if do_generate:
    try:
        if gen_type.startswith("Riff"):
            # ← generator/ または generators/ の実体を使う
            rg = RiffGenerator(
                instrument="guitar", patterns_yaml=str(REPO_ROOT / "data" / "riff_library.yaml")
            )
            pm = rg.generate(
                key=key,
                tempo=float(tempo),
                emotion=emotion,
                section=section,
                chord_seq=chord_seq,
                bars=int(bars),
                style=style,
            )
        else:
            og = ObligatoGenerator(
                instrument="synth", patterns_yaml=str(REPO_ROOT / "data" / "obligato_library.yaml")
            )
            pm = og.generate(
                key=key,
                tempo=float(tempo),
                emotion=emotion,
                section=section,
                chord_seq=chord_seq,
                bars=int(bars),
            )

        buf = io.BytesIO()
        tmp = REPO_ROOT / "outputs" / Path(out_name).with_suffix(".mid")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(tmp))
        with open(tmp, "rb") as f:
            buf.write(f.read())
        buf.seek(0)

        st.success(f"生成しました → {tmp}")
        st.download_button("⬇️ MIDIをダウンロード", data=buf, file_name=tmp.name, mime="audio/midi")
        st.caption("生成に使ったコード（バー開始拍 / シンボル）")
        st.table({"bar_start_beat": [b for b, _ in chord_seq], "chord": [c for _, c in chord_seq]})
    except FileNotFoundError:
        st.error(
            "YAML が見つかりません。`data/riff_library.yaml` と `data/obligato_library.yaml` を配置してください。"
        )
    except Exception as e:
        st.exception(e)
