# 🚀 Colab用ZIP高速アップロード手順

## Step 1: MacでZIPファイル作成

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21

# 重要なファイルだけをZIP (不要なファイルを除外)
zip -r composer2-3.zip composer2-3 \
  -x "*.git/*" \
  -x "*/.venv*/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*/checkpoints/*.ckpt" \
  -x "*/lightning_logs/*"
```

予想サイズ: 約500-800MB (CSVデータ含む)

## Step 2: Colab Notebookで実行

### セル1: ZIPアップロード
```python
from google.colab import files
print("composer2-3.zip を選択してください...")
uploaded = files.upload()
print("✓ アップロード完了!")
```

### セル2: 解凍して移動
```python
import os

# 解凍
!unzip -q composer2-3.zip
print("✓ 解凍完了!")

# ディレクトリ移動
os.chdir('/content/composer2-3')
print(f"✓ 現在のディレクトリ: {os.getcwd()}")

# ファイル確認
!ls -lh scripts/train_phrase.py
!ls -lh data/phrase_csv/*_raw.csv
```

### セル3: GPU確認
```python
import torch
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

### セル4: 依存関係インストール
```python
!pip install -q pretty_midi pandas scikit-learn tqdm
print("✓ 依存関係インストール完了!")
```

### セル5: クイックテスト (3 epochs, 24分)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 3 \
  --out checkpoints/guitar_duv_raw_test \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

### セル6: 全楽器トレーニング (7-8時間)
```python
!bash scripts/train_all_colab.sh
```

## Step 3: 学習後にチェックポイントをダウンロード

```python
from google.colab import files

# 学習済みモデルをダウンロード
files.download('checkpoints/guitar_duv_raw.best.ckpt')
files.download('checkpoints/bass_duv_raw.best.ckpt')
files.download('checkpoints/piano_duv_raw.best.ckpt')
files.download('checkpoints/strings_duv_raw.best.ckpt')
files.download('checkpoints/drums_duv_raw.best.ckpt')
```

または、Google Driveに保存:
```python
from google.colab import drive
drive.mount('/content/drive')

# チェックポイントをコピー
!mkdir -p "/content/drive/MyDrive/composer2-3_checkpoints"
!cp checkpoints/*_duv_raw.best.ckpt "/content/drive/MyDrive/composer2-3_checkpoints/"
print("✓ チェックポイントをGoogle Driveに保存しました!")
```

## ⚠️ 注意事項

1. **セッション終了でファイルは削除される**
   - `/content/` のファイルは一時的
   - チェックポイントは必ずダウンロードかGoogle Driveに保存

2. **大きいファイルを除外してZIP作成**
   - 古いcheckpointsは除外 (再学習するため)
   - .venv は除外 (Colabで再インストール)
   - .git は除外 (バージョン管理不要)

3. **ZIPサイズが1GB超える場合**
   - Google Driveアップロードの方が安定
   - または、data/phrase_csv/ だけ別途アップロード
