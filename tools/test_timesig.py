import yaml
from music21 import meter
from pathlib import Path

CFG_PATH = (Path(__file__).resolve().parent / '..' / 'config' / 'main_cfg.yml').resolve()


def check_time_signature(ts_str: str) -> None:
    try:
        ts = meter.TimeSignature(ts_str)
        print(f'OK: {ts_str} \u2192 {ts.numerator}/{ts.denominator}')
    except Exception as e:
        print(f'NG: {ts_str} \u2192 {e}')


def main() -> None:
    with open(CFG_PATH, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ts_str = cfg.get('global_settings', {}).get('time_signature', '4/4')
    print(f'main_cfg[global_settings][time_signature] = {ts_str}')
    check_time_signature(ts_str)


if __name__ == '__main__':
    main()
