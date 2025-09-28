# DUV Data Enrichment Workflow

このドキュメントでは、`scripts/train_phrase.py` を中心とした Duration / Velocity（DUV）学習ラインの強化に向けたデータ拡張計画を整理します。Transformer および LoRA ベースのモデルが高精度な時間構造と楽器間相互作用を学習できるよう、外部データセット導入から評価、推論パイプ統合までの実務手順をまとめています。

## スコープと位置付け
- **本線モデル**: PhraseTransformer（DUV ヘッド含む）、MLVelocityModel、DurationTransformer、LoRA GPT-2 系。
- **フォールバック**: `utilities/velocity_model.py` の KDEVelocityModel は統計ベースラインとして計測に利用。
- **追加コンポーネント**: VocalSynchro / section-aware & emotion-aware アレンジ、14-bit ピッチベンド制御、CC スプライン適用、UJAM/EZ 系プラグイン出力補助。

## 1. データソース準備
| ソース | 目的 | 必要アクション |
| --- | --- | --- |
| Lakh MIDI Dataset (LMD clean サブセット推奨) | 多様な曲構造・伴奏パターン | Todo #4 にて取得・検証・`data/lmd_clean/` へ展開 |
| Slakh2100 (MIDI のみ) | マルチトラック・実演クオリティ | Todo #5 にて MIDI 抽出・メタデータ保持・`data/slakh2100_midi/` へ配置 |
| 既存社内 MIDI / VocalSynchro 用音声 | セクション・感情タグ整合、歌詞同期 | 既存 `data/voice/` `data/lyrics/` を整理し BPM アンカーを補完 |

- 各データセットは PrettyMIDI→Music21 換装の正規化を行い、`jsonl` への書き出しを共通化する。
- バージョン管理: `data_manifests/duv/{YYYYMMDD}_*.yaml` を作成し、使用データの SHA と分割（train/val/test）を記録。

## 2. 前処理パイプライン
1. **MIDI 正規化**
   - `scripts/midi_normalize.py`（新規）で下記処理を一括化予定。
   - チャネル・プログラム番号マッピング（`configs/instrument_map.yaml`）の適用。
   - 量子化: 拍子推定後に 1/16 単位へ丸め、ペダル持続は Sustain CC64 へ変換。
2. **タグ付け強化**
   - 章タグ: `configs/sections.yaml` を参照し Verse/Chorus/Bridge/Fill を自動推定。未確定箇所は `tagger/manual_review.csv` にキュー。
   - 感情タグ: `tags.yaml` + 既存 `ml_models/tagger_module.py` を利用し、`--include-tags`/`--reweight` 互換のスコアを保存。
3. **VocalSynchro 連携**
   - 音声がある場合は `tools/vocalsynchro.detect_tempo` → `tools/vocalsynchro.align_lyrics` で拍列を生成し、`lyrics_beats.csv` を MIDI JSONL へアタッチ。

## 3. 特徴量と JSONL スキーマ更新
- 既存 `piano.jsonl` スキーマを拡張し、以下のフィールドを追加:
  - `phrase_boundary`: 0/1 フラグ。
  - `duration_class` / `velocity_class`: モデル学習用のバケット ID。
  - `velocity_cont`: 連続値（0-127 正規化）を保持し回帰ヘッドで利用。
  - `section_tag` / `emotion_tag`: アレンジ制御用。
  - `vocal_sync`: シラブルごとの拍情報。存在しない場合は `null`。
- 変換ユーティリティを `utilities/jsonl_upgrade.py`（新規）として実装し、既存データとの差分アップグレードをサポート。

## 4. トレーニング計画
### 4.1 PhraseTransformer / Duration / Velocity
- 入口: `scripts/train_phrase.py`（DUV ヘッド有効化）。
- ハイパーパラメータ:
  - `--duv-enabled`, `--velocity-regression-weight`, `--duration-class-weight`。
  - 新たに `--duv-data-manifest` で前述 YAML を読み込み可能にする（別実装タスク）。
- 学習スキーム:
  - Curriculum: 1) LMD で基本リズム、2) Slakh2100 混入で実演ニュアンス。
  - ランダム `section_tag` / `emotion_tag` ドロップアウトで汎化を確保。
- 評価指標:
  - Duration MAE、Velocity MAE。
  - Phrase boundary F1。
  - KDEVelocityModel を用いた統計ベースラインとの比較レポート（`reports/duv_baseline_vs_transformer.md`）。

### 4.2 LoRA モデルとの連携
- `train_piano_lora.py`, `train_sax_lora.py` におけるトークン列生成で、更新後 JSONL から `velocity_cont` を埋め込む。
- LoRA アダプタは DUV で得た動的ベロシティ分布を参照するため、`--duv-conditioning` フラグを追加予定。

## 5. 評価・検証フロー
1. `pytest -k duv`（新規テストスイート）で JSONL→Tensor 変換とロス計算を検証。
2. `python scripts/eval_duv.py`（新規）で以下を可視化:
   - Duration/Velocity ヒストグラム。
   - セクション別のベロシティ分布比較。
   - KDE ベースラインとの差分。
3. 生成サンプルのスモークテスト:
   - `python generator/build_song ... --duv-model checkpoints/phrase_duv.ckpt`。
   - `tools/render_audio.py` で Stems 出力後、`evaluation_metrics.py` を再利用し数値化。

## 6. デプロイ / 推論パイプ統合
- `utilities/ml_velocity.py` / `utilities/ml_duration.py` の `load_model` を JSONL フォーマット変更に合わせ更新。
- 推論 CLI: `python -m generator.build_song` に `--duv-checkpoint`, `--duv-temperature`, `--fallback-kde` を追加。
- CC スプライン・ピッチベンド制御と統合するため、`controls.md`, `pitch_bend.md` に追記予定。

## 7. マイルストーン
1. **M0** – データ取得・正規化スクリプト（Todos #4, #5）完了。バージョン付きマニフェスト作成。
2. **M1** – JSONL スキーマ拡張 + 変換ツール。旧データ資産のアップグレード完了。
3. **M2** – DUV トレーニング・評価ルーチンの更新（ハイパー・CLI 拡張含む）。
4. **M3** – LoRA 連携・推論 CLI 更新・VocalSynchro との接続テスト。
5. **M4** – 生成デモ（セクション/感情タグ＋VocalSynchro 連携）とレポート公開。

---
今後は Todo #3 「Draft instrumentation/tagging scripts」に着手し、ここで定義した前処理・タグ付けを自動化する実装を進めます。
