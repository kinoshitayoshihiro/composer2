from __future__ import annotations

import base64
import json
import random
import subprocess
from argparse import ArgumentParser
from pathlib import Path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.39/Tone.min.js"></script>
<button id='start'>Start</button>
<div id='questions'></div>
<script>
const trials = JSON.parse(`TRIALS`);
let current = 0;
let results = [];
function playSample(url){
  const player = new Tone.Player(url).toDestination();
  player.autostart = true;
}
function next(){
  if(current>=trials.length){alert('Done');return;}
  const t=trials[current];
  const div=document.createElement('div');
  div.innerHTML=`<button onclick="playSample('${t.a}')">A</button> <button onclick="playSample('${t.b}')">B</button> <button onclick="playSample('${t.x}')">X</button>`;
  const sel=document.createElement('select');
  sel.innerHTML='<option value="A">A</option><option value="B">B</option>';
  const btn=document.createElement('button');
  btn.textContent='Submit';
  btn.onclick=()=>{
    results.push({trial: current, choice: sel.value});
    const blob = new Blob([JSON.stringify(results)], {type:'application/json'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='results.json';
    a.click();
    current++; next();
  };
  div.appendChild(sel);div.appendChild(btn);
  document.getElementById('questions').appendChild(div);
}
document.getElementById('start').onclick=next;
</script>
</body>
</html>
"""


def main() -> None:
    ap = ArgumentParser(description="Generate ABX listening test")
    ap.add_argument("dry_dir", type=Path)
    ap.add_argument("artic_dir", type=Path)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("abx.html"))
    ap.add_argument(
        "--soundfont", type=Path, default=Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    )
    args = ap.parse_args()

    dry_files = sorted(Path(args.dry_dir).glob("*.mid"))
    art_files = sorted(Path(args.artic_dir).glob("*.mid"))
    pairs = list(zip(dry_files, art_files))
    random.shuffle(pairs)
    trials = []

    def render(m: Path) -> str:
        ogg = m.with_suffix(".ogg")
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                str(args.soundfont),
                str(m),
                "-F",
                str(ogg),
                "-r",
                "44100",
            ],
            check=True,
        )
        data = ogg.read_bytes()
        return "data:audio/ogg;base64," + base64.b64encode(data).decode()

    for a, b in pairs[: args.trials]:
        trials.append(
            {"a": render(a), "b": render(b), "x": render(random.choice([a, b]))}
        )

    html = HTML_TEMPLATE.replace("TRIALS", json.dumps(trials))
    args.out.write_text(html)
    (args.out.parent / "results.json").write_text("[]")


if __name__ == "__main__":
    main()
