#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pretty_midi as pm


PIANO_PROGS = set(range(0,8)) # GM Acoustic/EP系を広めに許容




def is_piano_inst(inst: pm.Instrument) -> bool:
return (not inst.is_drum) and (inst.program in PIANO_PROGS)




def main():
ap = argparse.ArgumentParser()
ap.add_argument("root", type=str, help="POP909の songs ディレクトリ")
ap.add_argument("--out", required=True)
args = ap.parse_args()


root = Path(args.root)
mids = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
with open(args.out, "w", encoding="utf-8") as w:
for m in mids:
try:
obj = pm.PrettyMIDI(str(m))
picked = [inst for inst in obj.instruments if is_piano_inst(inst)]
if not picked:
# ピアノが見つからない場合は全非ドラム候補を記録
picked = [inst for inst in obj.instruments if not inst.is_drum]
for i, inst in enumerate(picked):
rec = {
"id": f"{m.stem}_ch{i}",
"path": str(m.resolve()),
"instrument": "piano",
"source": "POP909",
"meta": {"program": inst.program, "notes": len(inst.notes)}
}
w.write(json.dumps(rec, ensure_ascii=False) + "\n")
except Exception as e:
print(f"[skip] {m}: {e}")


if __name__ == "__main__":
main()
