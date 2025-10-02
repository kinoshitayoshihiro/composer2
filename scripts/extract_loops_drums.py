#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path


P = re.compile(r"(?i)(?P<prefix>\d+)_(?P<style>[a-z0-9-]+)_(?P<bpm>\d+)_?(?P<type>beat|fill)?_?(?P<ts>\d-\d)?_")




def parse_meta(name: str) -> dict:
m = P.search(name)
if not m:
return {}
d = m.groupdict()
out = {"style": d.get("style"), "bpm": int(d["bpm"]) if d.get("bpm") else None,
"type": d.get("type"), "time_sig": d.get("ts").replace("-","/") if d.get("ts") else None}
return out




def main():
ap = argparse.ArgumentParser()
ap.add_argument("root", help="loops ディレクトリ")
ap.add_argument("--out", required=True)
args = ap.parse_args()


root = Path(args.root)
mids = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
with open(args.out, "w", encoding="utf-8") as w:
for m in mids:
meta = parse_meta(m.name)
rec = {
"id": m.stem,
"path": str(m.resolve()),
"instrument": "drums",
"source": "LOOPS",
"meta": meta
}
w.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
main()