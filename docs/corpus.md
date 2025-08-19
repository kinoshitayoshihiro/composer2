# Transformer Corpus Preparation

`tools/prepare_transformer_corpus.py` converts a folder of MIDI files into a
small corpus suitable for Transformer or LoRA experiments. Files are chopped
into fixed-length bar segments, tokenised in **beats** (not seconds) and
optionally enriched with tags or lyrics. Time signatures are respected so a
``bars-per-sample`` slice spans ``beats_per_bar * bars_per_sample`` beats.
Heavy dependencies are imported lazily so ``--help`` works even in minimal
environments.

Quantisation uses the provided ``--quant`` ticks-per-beat value. For variable
tempo files the script snaps note start/end times to the nearest beat using
``pretty_midi.get_beats`` and falls back to ``estimate_tempo`` when necessary.

```bash
python -m tools.prepare_transformer_corpus \
  --in data/midi_personal \
  --out data/corpus/personal_v1 \
  --bars-per-sample 4 --quant 480 --min-notes 8 \
  --duv on --dur-bins 16 --vel-bins 8 \
  --tags sections.yaml mood.yaml \
  --section-tokens --mood-tokens \
  --include-programs 0 24 --exclude-drums \
  --split 0.9 0.05 0.05 --seed 42
```

Example `sections.yaml` and `mood.yaml`:

```yaml
song.mid:
  section: chorus
  mood: melancholic
```

Each JSONL line contains a token sequence and metadata:

```json
{"tokens": ["<SECTION=chorus>", "NOTE_60", "D_4", "V_3"], "meta": {"source_path": "song.mid", "segment_index": 0}}
```

Useful flags:

- `--section-tokens` / `--mood-tokens` to prepend tag tokens.
- `--include-programs`, `--drums-only`, `--exclude-drums` for instrument
  filtering.
- `--max-files`, `--max-samples-per-file` to subsample large collections.
- `--progress` displays a `tqdm` progress bar and `--num-workers` enables
  multiprocessing (disabled when embedding text at runtime; use
  `--embed-offline` to re-enable).

### DUV tokens and offline embeddings

`--duv on` emits combined duration–velocity tokens. Limit the vocabulary with
`--duv-max N`; rarer pairs collapse to a single `DUV_OOV` token and the
frequency table is stored in `meta.json`. An INFO log reports kept vs. collapsed
tokens, e.g. `DUV kept: 12 collapsed: 5`.

Example JSONL line with a collapsed token:

```json
{"tokens": ["DUV_0_3", "DUV_OOV"], "meta": {"source_path": "song.mid", "segment_index": 0}}
```

Keys in `--lyric-json` are normalised to lower-case paths relative to the input
directory so absolute or mixed-case entries still match. An INFO log summarizes
matches. Use `--compress gz` to write `*.jsonl.gz` instead of plain text.

Use `--embed-offline embeds.json` (or `.npz`) to supply pre-computed lyric or
title embeddings. When provided, runtime embedding is skipped and multiprocessing
remains available.
