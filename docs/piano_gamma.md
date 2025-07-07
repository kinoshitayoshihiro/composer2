# Piano Generator Gamma

This guide covers the LoRA-based transformer used for piano voicings.

## Quickstart

```bash
# 1. Extract events from your MIDI corpus
python scripts/extract_piano_voicings.py --midi-dir piano_midis --out piano.jsonl

# 2. Train the LoRA model
python train_piano_lora.py --data piano.jsonl --out piano_model

# 3. Sample a short accompaniment
modcompose sample dummy.pkl --backend piano_ml --model piano_model --temperature 0.9
```

## Token Table

| ID | Token |
|----|-------|
| 0  | <BAR> |
| 1  | <TS_4_4> |
| 2  | <LH> |
| 3  | <RH> |
| 4  | <REST_d8> |
| 5  | <VELO_80> |
| 6  | P60 |
| ...| ... up to P96 |

## Example Screenshot

![training placeholder](docs/img/piano_gamma_demo.png)

TODO: replace with training/demo GIF.
