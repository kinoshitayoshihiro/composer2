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


## How to fine-tune with your WAV corpus

1. **Collect WAV files** for each piece you want to use as training data.
2. **Split stems** using a tool like [spleeter](https://github.com/deezer/spleeter)
   so that piano or vocal parts can be isolated.
3. **Convert stems to MIDI** with a transcription tool (e.g. `basic-pitch` or
   `audio_to_midi.py`) and verify the timing in your DAW.
4. **Export JSONL** using
   `scripts/extract_piano_voicings.py --midi-dir MIDI_DIR --out piano.jsonl`.
5. **Train** your LoRA model with
   `python train_piano_lora.py --data piano.jsonl --out piano_model`.
