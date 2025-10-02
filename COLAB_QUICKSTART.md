# Google Colab クイックスタート

## 🚀 5分でトレーニング開始

### ステップ1: 事前準備 (ローカル)

```bash
# データをGoogle Driveにアップロード
# My Drive/composer2_data/phrase_csv/ に以下をコピー:
# - guitar_train_raw.csv (261 MB)
# - guitar_val_raw.csv
# - bass_train_raw.csv
# - bass_val_raw.csv
# - piano_train_raw.csv
# - piano_val_raw.csv
# - strings_train_raw.csv
# - strings_val_raw.csv
# - drums_train_raw.csv
# - drums_val_raw.csv
```

### ステップ2: Colab Notebook

1. [Colab](https://colab.research.google.com/)で新規ノートブック作成
2. ランタイム→GPUに変更
3. 以下のセルをコピペして実行:

```python
# === Cell 1: セットアップ ===
!git clone https://github.com/kinoshitayoshihiro/composer2.git
%cd composer2
!git checkout copilot/vscode1759159549848  # 最新ブランチ

# 依存関係インストール
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pretty_midi pandas numpy scikit-learn tqdm PyYAML librosa mido pytorch-lightning torchmetrics music21 scipy hydra-core

# GPU確認
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ERROR: No GPU!'}")
```

```python
# === Cell 2: データのコピー ===
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data/phrase_csv
!cp -v /content/drive/MyDrive/composer2_data/phrase_csv/*.csv data/phrase_csv/
!ls -lh data/phrase_csv/*.csv
```

```python
# === Cell 3: テストラン (3分) ===
# Guitarモデルで動作確認 (3 epochs)
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 3 \
  --out checkpoints/guitar_test \
  --arch transformer \
  --d_model 512 --nhead 8 --layers 4 \
  --batch-size 256 --num-workers 2 \
  --lr 1e-4 --duv-mode reg \
  --device cuda --save-best --progress

# 確認: train_batches/epoch が 50,000以上なら成功!
```

```python
# === Cell 4: 本番トレーニング (4-8時間) ===
!bash scripts/train_all_base_colab.sh

# 別セルで進行確認:
# !tail -n 20 checkpoints/guitar_duv_raw/train.log
```

```python
# === Cell 5: チェックポイント保存 ===
!mkdir -p /content/drive/MyDrive/composer2_checkpoints
!cp -v checkpoints/*_duv_raw.best.ckpt /content/drive/MyDrive/composer2_checkpoints/
!ls -lh /content/drive/MyDrive/composer2_checkpoints/*.ckpt
```

### ステップ3: ローカルで続き

1. Google Driveから `composer2_checkpoints/*.ckpt` をダウンロード
2. `checkpoints/` フォルダに配置
3. LoRAモデルを再トレーニング

```bash
# ベースモデルパスを更新
# config/duv/guitarLora.yaml → base_checkpoint: checkpoints/guitar_duv_raw.best.ckpt

# LoRA再トレーニング
python scripts/train_guitar_lora.py
```

---

## 📊 成功の目安

- ✅ `train_batches/epoch`: 50,000以上
- ✅ `val_batches/epoch`: 5,500以上
- ✅ `vel_mae`: 15エポック後に0.5以下
- ✅ `dur_mae`: 15エポック後に0.3以下
- ✅ チェックポイント: 各99MB

---

## 🔗 詳細ガイド

完全な手順は `COLAB_TRAINING.md` を参照してください。

## ⚠️ トラブルシューティング

**Q**: GPUが使えない  
**A**: ランタイムタイプをGPUに変更

**Q**: データが見つからない  
**A**: Driveのパスを確認 (`/content/drive/MyDrive/composer2_data/`)

**Q**: メモリ不足  
**A**: `--batch-size 128` に削減

**Q**: 途中で切断  
**A**: 最長12時間、チェックポイントから再開可能
