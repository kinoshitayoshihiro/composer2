# Groove Sampler v2 Streaming Training

`groove_sampler_v2` can now ingest very large MIDI corpora without loading all
files into memory.  Training iterates over a list of files and updates the
statistics incrementally while periodically saving checkpoints.

## Streaming flags

```
python -m utilities.groove_sampler_v2 train --from-filelist FILELIST.txt \
    [--shard-index N --num-shards M] \
    [--min-bytes 800 --min-notes 8] \
    [--max-files N] [--progress/--no-progress] \
    [--log-every 200] [--save-every N] \
    [--checkpoint-dir DIR] [--resume-from PATH] \
    [--gc-every 1000] [--mem-stats] \
    [--fail-fast/--no-fail-fast]
```

* **`--from-filelist`** – text file with one MIDI path per line.
* **`--shard-index` / `--num-shards`** – round‑robin sharding for distributed runs.
* **`--min-bytes`**, **`--min-notes`** – filter tiny loops before processing.
* **`--max-files`** – optional cap on processed files.
* **`--progress/--no-progress`** – toggle tqdm progress bar.
* **`--log-every`** – emit statistics every N files.
* **`--save-every`** – dump a checkpoint after N accepted files.
* **`--checkpoint-dir`** – directory for checkpoints (defaults to output dir).
* **`--resume-from`** – resume from a previous checkpoint.
* **`--gc-every`** – force `gc.collect()` periodically.
* **`--mem-stats`** – log RSS memory if `psutil` is available.
* **`--fail-fast/--no-fail-fast`** – stop on malformed MIDI or skip (default).

Checkpoints store `{version, counts, processed, kept, skipped}` and can be
merged later.

## Merging

Partial models created by sharded training can be merged:

```
python -m utilities.groove_sampler_v2 merge shard*.pkl -o groove_model.pkl
```

## Typical workflow

1. Generate a list of MIDI files:
   ```bash
   find loops -name '*.mid' > filelist.txt
   ```
2. Train shards and periodically save checkpoints:
   ```bash
   python -m utilities.groove_sampler_v2 train \
       --from-filelist filelist.txt --shard-index 0 --num-shards 4 \
       --save-every 1000 -o shard0.pkl
   ```
3. Repeat for other shards (e.g. `shard-index 1`, etc.).
4. Merge shard models:
   ```bash
   python -m utilities.groove_sampler_v2 merge shard*.pkl -o groove_model.pkl
   ```
5. In constrained environments schedule shards over time (e.g. nightly cron).

The legacy directory-based training remains available and unchanged.

## Memory-controlled n-gram training

Standard directory training now streams n-gram counts to an on-disk SQLite
store, greatly reducing peak RAM usage:

```
python -m utilities.groove_sampler_v2 train loops/ -o model.pkl \
    --ram-budget-mb 512 --flush-every 100000 \
    --sqlite-path tmp/groove_ngrams.sqlite
```

Additional flags:

* **`--ram-budget-mb`** – approximate RAM budget for the in-memory n-gram buffer.
* **`--flush-every`** – force a flush to SQLite after N n-grams.
* **`--min-count`** – drop n-grams with counts below this value when finalising.
* **`--max-ngrams`** – safety cap on total processed n-grams.
* **`--hash-buckets`** – size of the hashing space for contexts.
* **`--sqlite-path`** – path to the SQLite database.

The resulting model references the SQLite file and loads distributions lazily
with a small LRU cache when sampling.
