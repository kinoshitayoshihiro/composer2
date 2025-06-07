import yaml
from music21 import meter


def test_time_signature_parsing(ts_str):
    try:
        ts = meter.TimeSignature(ts_str)
        print(f"OK: '{ts_str}' → {ts.numerator}/{ts.denominator}")
    except Exception as e:
        print(f"NG: '{ts_str}' → {e}")


# main_cfg.yml の正しいパスを指定してください
CFG_PATH = "../config/main_cfg.yml"  # ← tools/ から見た相対パス例。必要に応じて修正

with open(CFG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# global_settings.time_signature を取得してテスト
ts_str = cfg.get("global_settings", {}).get("time_signature", "4/4")
print(f"main_cfg['global_settings']['time_signature'] = '{ts_str}'")
test_time_signature_parsing(ts_str)
