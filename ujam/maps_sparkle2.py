# maps_sparkle2.yaml
# UJAM Virtual Guitarist SPARKLE 2 — Common/Style keymap
ujam:
  product: VG-SPARKLE2
  # ← Amber と同じ値を入れてください（例:  c0: 24, c1: 36）
  c0: 24
  c1: 36
  # エレキの帯域に少し寄せた推奨コードレンジ
  chord_low: 43    # G2
  chord_high: 76   # E5

# 画面の並び（左→右）に完全一致させた Common Phrases
phrases_ui_order:
  - silence
  - long_chord_2_4
  - open_1_1
  - off_beat_2_4
  - open_stops_1_4
  - open_1_4
  - half_muted_1_8
  - open_1_8
  - muted_1_8
  - open_1_16
  - off_beat_1_8
  - pick_slide
  # 上段（説明にだけ出る系）もトリガ可能にしておく
  - chord_rhythm_1_16
  - generic_chord_rhythm
  - single_note_rhythm
  - generic_rhythm_sustain
  - long_chord_with_fill
  - muted_1_8_open
  - build_up_open_1_8
  - muted_1_16_riding
  - build_up_muted_1_8
  - slide_down

# C1..B1 に style_1..style_12 を割当（Amberと同様）
style_phrases:
  base: 1
  step_B: 4
  step_Build: 8
  cycle_8bars: false   # Sparkleはまず固定推奨（必要なら true）

# Sparkle向けの最初のガード（細切れ防止＆ジャキジャキ回避）
phrase_limits:
  blocklist:
    - muted_1_16_riding
    - single_note_rhythm
  fallback: "open_1_1"

# そのまま流用できる共通ルール（必要なら上書き）
rules:
  swing_mode: both
  swing_threshold: 0.12
  offbeat_threshold: 0.55
  long_med_ratio: 0.45
  dense_eighth_notes: 7
  quarter_range: [3, 6]
  section_window_bars: 4
  build_every_8bars: true
