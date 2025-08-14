import sys
import json
from pathlib import Path
import pretty_midi as pm

def check_midi_tempos(root_dir="midi_output"):
    """MIDIファイルのテンポを監査"""
    root = Path(root_dir)
    bad = []
    total_files = 0
    
    for p in sorted(root.rglob("*.mid")):
        total_files += 1
        try:
            m = pm.PrettyMIDI(str(p))
            tempi, times = m.get_tempo_changes()
            
            # ケース判定
            if len(tempi) == 0:
                bad.append({
                    "path": str(p.relative_to(root)),
                    "reason": "no_tempo_events"
                })
            elif float(tempi[0]) <= 0:
                bad.append({
                    "path": str(p.relative_to(root)),
                    "reason": f"non_positive:{float(tempi[0]):.2f}"
                })
        except Exception as e:
            bad.append({
                "path": str(p.relative_to(root)),
                "reason": f"load_error:{e.__class__.__name__}"
            })
    
    print(f"Scanned {total_files} files in {root_dir}")
    
    if bad:
        print(f"Found {len(bad)} files with missing/non-positive tempo:")
        for row in bad:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("✅ All MIDI files have positive tempo events.")
    
    return bad

if __name__ == "__main__":
    # コマンドライン引数でディレクトリ指定可能
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "midi_output"
    check_midi_tempos(target_dir)