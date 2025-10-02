# Google Colab Training Guide

## 🎯 概要

このガイドは、Google Colab GPU を使用してDUVベース#### Cell 2: 依存パッケージのインストール

```python
# PyTorch (CUDA対応) をインストール
print("📦 Installing PyTorch with CUDA support...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 必須パッケージを個別にインストール
print("📦 Installing core dependencies...")
!pip install -q pretty_midi>=0.2.9
!pip install -q pandas>=1.5.0
!pip install -q numpy>=1.23.0
!pip install -q scikit-learn>=1.2.0
!pip install -q tqdm>=4.65.0
!pip install -q PyYAML>=6.0
!pip install -q librosa>=0.10.0
!pip install -q mido>=1.3
!pip install -q pytorch-lightning>=2.0
!pip install -q torchmetrics>=1.0
!pip install -q music21>=9.1
!pip install -q scipy>=1.13
!pip install -q hydra-core>=1.3

print("✅ All packages installed!")

# バージョン確認
import torch
import pandas as pd
import numpy as np
print(f"\n📊 Installed versions:")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
```

**注意**: `requirements.txt`は依存ファイルが不完全なため使用しません。完全な手順です。

- **対象**: 5楽器 (guitar, bass, piano, strings, drums)
- **データ**: 12,041,596フレーズ、127段階ベロシティ
- **推定時間**: 4-8時間 (全5モデル、Colab GPU使用時)
- **準備**: GitHubリポジトリ、データCSVファイル (約400MB)

---

## 📋 事前準備 (ローカルで実施)

### 1. データをGoogle Driveにアップロード

ローカルの `data/phrase_csv/` フォルダにある以下のファイルをGoogle Driveにアップロード:

```
My Drive/
└── composer2_data/
    └── phrase_csv/
        ├── guitar_train_raw.csv    (261 MB)
        ├── guitar_val_raw.csv
        ├── bass_train_raw.csv
        ├── bass_val_raw.csv
        ├── piano_train_raw.csv
        ├── piano_val_raw.csv
        ├── strings_train_raw.csv
        ├── strings_val_raw.csv
        ├── drums_train_raw.csv
        └── drums_val_raw.csv
```

### 2. コードをGitHubにプッシュ

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
git add -A
git commit -m "feat: Add Colab training scripts with GPU optimization"
git push origin main
```

---

## 🚀 Colabでのトレーニング手順

### Step 1: 新しいノートブックを作成

1. [Google Colab](https://colab.research.google.com/) にアクセス
2. 「ファイル」→「ノートブックを新規作成」
3. GPUランタイムを有効化:
   - 「ランタイム」→「ランタイムのタイプを変更」
   - 「ハードウェアアクセラレータ」→「GPU」選択
   - 「保存」

---

### Step 2: 環境セットアップ

#### Cell 1: リポジトリのクローン

```python
# GitHubリポジトリをクローン
!git clone https://github.com/<YOUR_USERNAME>/composer2-3.git
%cd composer2-3

# 最新のコードを取得
!git pull origin main
```

**⚠️ 注意**: `<YOUR_USERNAME>` を実際のGitHubユーザー名に置き換えてください。

---

#### Cell 2: 依存パッケージのインストール

```python
# PyTorch (CUDA対応版) をインストール
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
!pip install -q pretty_midi pandas numpy scikit-learn tqdm PyYAML

# プロジェクトの requirements があれば
!pip install -q -r requirements.txt
```

---

#### Cell 3: GPU確認

```python
import torch

print("=" * 50)
print("🔧 GPU Configuration Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: CUDA not available! Check runtime settings.")
print("=" * 50)
```

**期待される出力**:
```
CUDA available: True
GPU Device: Tesla T4 (or similar)
```

---

### Step 3: データの準備

#### Cell 4: Google Driveのマウント

```python
from google.colab import drive
drive.mount('/content/drive')
```

画面の指示に従ってGoogleアカウントを認証してください。

---

#### Cell 5: データのコピー

```python
import os

# データディレクトリを作成
!mkdir -p data/phrase_csv

# Google Driveからデータをコピー
print("📁 Copying data from Google Drive...")
!cp -v /content/drive/MyDrive/composer2_data/phrase_csv/*.csv data/phrase_csv/

# データ確認
print("\n✓ Data files:")
!ls -lh data/phrase_csv/*.csv
```

**期待される出力**:
```
✓ Data files:
-rw-r--r-- 1 root root 261M guitar_train_raw.csv
-rw-r--r-- 1 root root  29M guitar_val_raw.csv
...
```

---

### Step 4: トレーニング実行

#### Cell 6: テストラン (Guitar 3 epochs)

まず短いテストで動作確認:

```python
# Guitarモデルで3エポックのテスト
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 3 \
  --out checkpoints/guitar_duv_raw_test \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 256 \
  --grad-accum 1 \
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

**確認ポイント**:
- ✅ `train_batches/epoch` が約50,000以上
- ✅ `val_batches/epoch` が約5,500以上
- ✅ `vel_mae`, `dur_mae` が0でなく減少していく

---

#### Cell 7: 全楽器トレーニング (本番)

テストが成功したら、全5楽器を15エポックずつトレーニング:

```python
# 全楽器のトレーニング開始 (約4-8時間)
!bash scripts/train_all_base_colab.sh
```

**進行状況の確認** (別のセルで実行可):
```python
# 最新のログを確認
!tail -n 50 checkpoints/guitar_duv_raw/train.log
```

---

### Step 5: チェックポイントの保存

#### Cell 8: Driveにバックアップ

```python
# 訓練済みモデルをGoogle Driveに保存
!mkdir -p /content/drive/MyDrive/composer2_checkpoints

print("💾 Saving checkpoints to Google Drive...")
!cp -v checkpoints/guitar_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/
!cp -v checkpoints/bass_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/
!cp -v checkpoints/piano_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/
!cp -v checkpoints/strings_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/
!cp -v checkpoints/drums_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/

print("\n✓ Checkpoints saved!")
!ls -lh /content/drive/MyDrive/composer2_checkpoints/*.ckpt
```

---

#### Cell 9: ローカルにダウンロード (オプション)

Driveからではなく直接ダウンロードする場合:

```python
from google.colab import files

# 各チェックポイントをダウンロード
files.download('checkpoints/guitar_duv_raw.best.ckpt')
files.download('checkpoints/bass_duv_raw.best.ckpt')
files.download('checkpoints/piano_duv_raw.best.ckpt')
files.download('checkpoints/strings_duv_raw.best.ckpt')
files.download('checkpoints/drums_duv_raw.best.ckpt')
```

---

## � トレーニング結果の確認

### Cell 10: メトリクスの表示

```python
import pandas as pd
import matplotlib.pyplot as plt

# Guitarのトレーニング履歴を読み込み
metrics = pd.read_csv('checkpoints/guitar_duv_raw/metrics.csv')

# プロット
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(metrics['epoch'], metrics['train_loss'], label='Train')
axes[0, 0].plot(metrics['epoch'], metrics['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].legend()

axes[0, 1].plot(metrics['epoch'], metrics['vel_mae'])
axes[0, 1].set_title('Velocity MAE')

axes[1, 0].plot(metrics['epoch'], metrics['dur_mae'])
axes[1, 0].set_title('Duration MAE')

axes[1, 1].plot(metrics['epoch'], metrics['vel_corr'])
axes[1, 1].set_title('Velocity Correlation')

plt.tight_layout()
plt.show()
```

---

## 🏠 ローカル環境での続き

### 1. チェックポイントをダウンロード

Google Driveから `composer2_checkpoints/*.ckpt` をローカルにダウンロード:

```
/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/checkpoints/
├── guitar_duv_raw.best.ckpt   (99 MB)
├── bass_duv_raw.best.ckpt     (99 MB)
├── piano_duv_raw.best.ckpt    (99 MB)
├── strings_duv_raw.best.ckpt  (99 MB)
└── drums_duv_raw.best.ckpt    (99 MB)
```

### 2. LoRAモデルの再トレーニング

新しいベースモデルを使ってLoRAアダプタを再トレーニング:

```bash
# LoRA設定ファイルのベースモデルパスを更新
# config/duv/guitarLora.yaml
# base_checkpoint: checkpoints/guitar_duv_raw.best.ckpt

# LoRAトレーニング実行
python scripts/train_guitar_lora.py
```

### 3. パイプラインテスト

全てのモデルが揃ったら、最終的なパイプラインテスト:

```bash
python scripts/test_full_pipeline.py
```

---

## ✅ トラブルシューティング

### Q1: GPUが使えない

**A**: ランタイムタイプを確認してください:
- 「ランタイム」→「ランタイムのタイプを変更」→「GPU」

### Q2: データのコピーでエラー

**A**: Driveのパスを確認:
```python
# パスの確認
!ls /content/drive/MyDrive/
```

### Q3: メモリ不足エラー

**A**: batch_size を削減:
```bash
--batch-size 128  # 256から削減
```

### Q4: トレーニングが途中で止まる

**A**: Colabは最大12時間で切断されます。チェックポイントから再開:
```bash
# 最後のチェックポイントから再開
--resume checkpoints/guitar_duv_raw/last.ckpt
```

---

## 📁 Colab用スクリプト

以下のスクリプトがGitHubリポジトリに含まれています:

- **`scripts/train_all_base_colab.sh`**: 全5楽器のGPU最適化トレーニングスクリプト
  - Batch size: 256 (GPU用)
  - Workers: 2
  - Device: cuda
  - Epochs: 15

---

## 🎉 完了後の確認

トレーニング完了後、以下を確認:

1. ✅ 5つの `.best.ckpt` ファイルが生成されている
2. ✅ 各ファイルサイズが約99MB
3. ✅ `vel_mae` が15エポック後に0.5以下に収束
4. ✅ `dur_mae` が15エポック後に0.3以下に収束

---

## 📞 サポート

問題が発生した場合は、以下を確認:

1. GPU が有効になっているか
2. データファイルが正しくコピーされているか
3. バッチカウントが正常 (50,000以上/epoch)
4. メトリクスが正常に減少しているか

Happy Training! 🚀
