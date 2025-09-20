import sys
from pathlib import Path
import shutil
import pretty_midi as pm

def fix_midi_tempos(src_dir="midi_output", dst_dir="midi_output_fixed", default_bpm=120.0):
    """テンポが空/0のMIDIを修復"""
    SRC = Path(src_dir)
    DST = Path(dst_dir)
    DEFAULT_BPM = default_bpm
    
    DST.mkdir(parents=True, exist_ok=True)
    
    def needs_fix(m: pm.PrettyMIDI) -> bool:
        _times, tempi = m.get_tempo_changes()
        return (len(tempi) == 0) or (float(tempi[0]) <= 0)
    
    fixed_count = 0
    copied_count = 0
    error_count = 0
    
    for src in sorted(SRC.rglob("*.mid")):
        rel = src.relative_to(SRC)
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            m = pm.PrettyMIDI(str(src))
        except Exception as e:
            # 壊れている等はそのままコピーしてスキップ
            shutil.copy2(src, dst)
            print(f"⚠️  Error loading {rel}, copied as-is: {e}")
            error_count += 1
            continue
        
        if not needs_fix(m):
            shutil.copy2(src, dst)   # 問題なければそのままコピー
            copied_count += 1
            continue
        
        # 初期テンポを持つ新しいオブジェクトに中身を移植
        fixed = pm.PrettyMIDI(initial_tempo=DEFAULT_BPM)
        
        # 可能なら解像度も合わせる
        try:
            fixed.resolution = m.resolution
        except Exception:
            pass
        
        # 各トラック/シグネチャ等をコピー（時刻は秒ベースで保たれます）
        fixed.instruments = m.instruments
        fixed.time_signature_changes = m.time_signature_changes
        fixed.key_signature_changes = m.key_signature_changes
        
        fixed.write(str(dst))
        print(f"✅ Fixed tempo ({DEFAULT_BPM} BPM) → {rel}")
        fixed_count += 1
    
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"  Fixed: {fixed_count} files")
    print(f"  Copied (already OK): {copied_count} files")
    print(f"  Errors: {error_count} files")
    print(f"  Total: {fixed_count + copied_count + error_count} files")
    print(f"Output directory: {DST}")
    print("Done.")

if __name__ == "__main__":
    # コマンドライン引数でカスタマイズ可能
    src = sys.argv[1] if len(sys.argv) > 1 else "midi_output"
    dst = sys.argv[2] if len(sys.argv) > 2 else "midi_output_fixed"
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    
    fix_midi_tempos(src, dst, bpm)