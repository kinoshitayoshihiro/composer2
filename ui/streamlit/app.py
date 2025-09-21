# ui/streamlit/app.py
from __future__ import annotations
import io
from pathlib import Path
import streamlit as st

# プロジェクトのジェネレータ（前にお渡しした実装）を使用
try:
    from generator.riff_generator import RiffGenerator
    from generator.obligato_generator import ObligatoGenerator
except Exception as e:
    st.error(
        "generator/ 以下に RiffGenerator / ObligatoGenerator が見つかりません。canvasのコードを配置してください。"
    )
    st.stop()

st.set_page_config(page_title="Riff / Obligato Generator", page_icon="🎸", layout="centered")

st.title("🎸 Riff / 🎼 Obligato Generator (minimal)")
st.caption("バラード優先 → 必要最小限のUI。プリセットや“つまみ”は後から拡張できます。")

# --- サイドバー：基本パラメータ ---
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

# --- メイン：コード進行入力 ---
st.subheader("コード進行（バーごと）")
st.caption("例: `Am | G | F | E`（4/4想定。縦棒で区切ると各バーになります）")
prog_text = st.text_area("Progression", value="Am | G | F | E", height=80)


def parse_progression(text: str, bars: int) -> list[tuple[float, str]]:
    # 4/4 として各バー開始拍は 0, 4, 8, ... に。
    tokens = [t.strip() for t in text.replace("\n", " ").split("|") if t.strip()]
    if not tokens:
        tokens = ["Am"]
    seq = []
    for i in range(max(bars, len(tokens))):
        chord = tokens[i % len(tokens)]
        seq.append((i * 4.0, chord))
    return seq


chord_seq = parse_progression(prog_text, int(bars))

# --- 生成ボタン ---
col1, col2 = st.columns(2)
with col1:
    default_name = "riff.mid" if gen_type.startswith("Riff") else "obligato.mid"
    out_name = st.text_input("出力ファイル名", value=default_name)
with col2:
    st.write("")  # spacer
    do_generate = st.button("🎵 生成する", use_container_width=True)

# --- 実行 ---
if do_generate:
    try:
        if gen_type.startswith("Riff"):
            # 楽器はまずギター固定。必要に応じて bass / keys もUI化可。
            rg = RiffGenerator(instrument="guitar", patterns_yaml="data/riff_library.yaml")
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
            og = ObligatoGenerator(instrument="synth", patterns_yaml="data/obligato_library.yaml")
            pm = og.generate(
                key=key,
                tempo=float(tempo),
                emotion=emotion,
                section=section,
                chord_seq=chord_seq,
                bars=int(bars),
            )

        # PrettyMIDI → バイナリに書き出してダウンロード
        buf = io.BytesIO()
        tmp_path = Path(out_name).with_suffix(".mid")
        pm.write(str(tmp_path))
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)

        st.success("生成しました。DAWにドラッグ＆ドロップできます。")
        st.download_button(
            "⬇️ MIDIをダウンロード", data=buf, file_name=str(tmp_path.name), mime="audio/midi"
        )

        # 便利: コード進行の確認用テーブル
        st.caption("生成に使ったコード（バー開始拍 / シンボル）")
        st.table({"bar_start_beat": [b for b, _ in chord_seq], "chord": [c for _, c in chord_seq]})

    except FileNotFoundError as e:
        st.error(
            "YAML が見つかりません。`data/riff_library.yaml` と `data/obligato_library.yaml` を配置してください。"
        )
    except Exception as e:
        st.exception(e)

st.divider()
st.caption(
    "※ まずは Ballad と Rock の2スタイルで運用。プリセット/“つまみ”/Jazz/演歌は後から足せます。"
)
