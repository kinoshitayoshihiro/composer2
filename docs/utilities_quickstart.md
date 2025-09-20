# Utilities Quickstart

## ミックスからテンポマップを推定（ステムで微修正）
```bash
python -m utilities.tempo_from_mix path/to/mix.wav \
  --stems stems/drums.wav,stems/bass.wav \
  --tighten-ms 18 --out data/tempo_maps/{song}.json
```

## ドラムstemを曲内キャリブ付きでMIDI化
```bash
python -m utilities.drumstem_to_midi stems/drums.wav midi_out/{song}/drums.mid \
  --bpm 120 --gate -34 --humanize-ms 2 --min-sep-ms 28
```

## Base×Narrativeのマージで chordmap.final.yaml を生成
```bash
python -m utilities.chordmap_merge \
  --base data/chords/{song}/base.yaml \
  --narr data/chords/{song}/narrative.yaml \
  --context data/tempo_maps/{song}.json \
  --out data/chords/{song}/chordmap.final.yaml
```

## chordmap + テンポマップから UJAM Sparkle用MIDIを書き出す
```bash
python -m utilities.ujam_from_chordmap \
  --chordmap data/chords/{song}/chordmap.final.yaml \
  --tempo-map data/tempo_maps/{song}.json \
  --out-dir midi_out/{song}/ujam \
  --map configs/ujam/sparkle_mapping.yaml  # 任意
```
