import pandas as pd, numpy as np, pathlib

# 出力ディレクトリの作成
pathlib.Path('data/pedal').mkdir(parents=True, exist_ok=True)

# データの読み込みとシャッフル
df = pd.read_csv('data/pedal/all.csv').sample(frac=1, random_state=0).reset_index(drop=True)

# 訓練/検証の分割ポイント（90%）
s = int(len(df)*0.9)

# 訓練データと検証データの保存
df.iloc[:s].to_csv('data/pedal/train.csv', index=False)
df.iloc[s:].to_csv('data/pedal/val.csv', index=False)

# 結果の表示
print(len(df), 'rows ->', s, 'train /', len(df)-s, 'val')
