# Slakh2100 MIDI Extraction

このドキュメントでは、Slakh2100 データセットをダウンロードし、MIDI とメタデータのみを抽出する手順をまとめています。音声（FLAC/WAV）は削除するため、ストレージを大幅に節約できます。

## 前提条件

- Python 3.10 以上
- `requests` パッケージ（`pyproject.toml` / `requirements/base.txt` に追加済み）
- 高速な回線と十分な空き容量（ダウンロード中に 110 GB 以上のストリーム転送が発生しますが、抽出後に残るのは数 GB 程度です）

## 使い方

1. 仮想環境をアクティブ化（例）:

   ```bash
   source .venv311/bin/activate
   ```

2. 安定した回線がない場合は、まず `--download-to` でアーカイブをローカルに段階的に保存します。途中で中断してもファイルが残るため、再実行すれば自動で続きを取得します。

  ```bash
  python scripts/download_slakh_midi.py \
    --download-to downloads/slakh2100_flac_redux.tar.gz \
    --keep-metadata --quiet
  ```

  `downloads/` 配下に 105 GB のアーカイブが完成したら、同じコマンドをもう一度実行し（即時に復帰します）、続けて抽出処理が走ります。

  > 既存のダウンロードマネージャーを使う場合は、`aria2c --continue=true --split=16 --max-connection-per-server=16 URL` のように再開可能なモードで `.tar.gz` を取得し、完成後に `--archive` を指定してください。

3. スクリプトを実行:

   ```bash
   python scripts/download_slakh_midi.py --keep-metadata
   ```

   - デフォルト URL: [Zenodo: Slakh2100 Redux](https://zenodo.org/records/4599666)
   - 出力先: `data/slakh2100_midi`
   - `--keep-metadata` を付けると、各トラックの `metadata.yaml` も保存します。

4. 途中で中断した場合は `--resume` を付けて再実行すると、既存ファイルを飛ばしながら再開できます。

   ```bash
   python scripts/download_slakh_midi.py --keep-metadata --resume
   ```

5. 既存のローカルアーカイブを使いたい場合は `--archive` でパスを渡せます:

   ```bash
   python scripts/download_slakh_midi.py \
     --archive /path/to/slakh2100_flac_redux.tar.gz \
     --keep-metadata
   ```

## BabySlakh での動作確認

本番前に小さな BabySlakh でテストするには、Zenodo から `babyslakh_16k.tar.gz` を取得し、`--archive` オプションで指定してください。

```bash
python scripts/download_slakh_midi.py \
  --archive data/babyslakh_16k.tar.gz \
  --output data/babyslakh_midi \
  --keep-metadata --quiet
```

## 出力構成

```
data/slakh2100_midi/
  Track00001/
    MIDI/
      S00.mid
      S01.mid
      ...
    metadata.yaml  # --keep-metadata を指定した場合
  Track00002/
  ...
```

抽出後、音声ファイルは残らないため、モデル学習やコーパス整備の準備として直接 MIDI を利用できます。
