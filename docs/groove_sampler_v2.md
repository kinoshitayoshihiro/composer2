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

Directory training can stream n‑gram counts directly to **numpy.memmap** shards,
avoiding large in-memory frequency tables. SQLite support may arrive in a
future release.

```
python -m utilities.groove_sampler_v2 train loops/ -o model.pkl \
    --train-mode stream --memmap-dir /fast_ssd/tmp \
    --hash-buckets 1048576 --max-rows-per-shard 1000000 \
    --n-jobs 4
```

Counts are stored in per-order shards under `--memmap-dir`.  Each shard starts as
`uint32` and automatically promotes to `uint64` if any counter would overflow.
Shards are created via atomic rename so partially written files are not left
behind.  Metadata (`meta.json`) records `schema_version=2` and the final dtype.

### Operations

* Use a fast **local SSD** for `--memmap-dir`; network filesystems may be slow
  or unreliable.
* Recommended flags for large corpora:
  `--train-mode stream --memmap-dir /fast_ssd/tmp --hash-buckets 1048576 --max-rows-per-shard 1000000 --n-jobs 4`
* `--resume` can restart unfinished runs and skips already processed files.
* Checkpoints with `--save-every` allow shard0→中断→再開のような運用。

SQLite remains a future option and is not yet supported.

## Memory & Scale

SQLite runs in WAL mode with `synchronous=NORMAL` by default, trading a small
risk of last‑commit loss for speed. Only one writer can hold the database lock
at a time; for multi‑process training use sharded databases or ensure a
single‑writer pattern.
