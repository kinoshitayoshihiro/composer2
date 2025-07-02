# Performance Tips

This project can generate music using multiple threads or processes and stream bars ahead of playback using the `LiveBuffer`.

```
+---------+     +------------+     +-------------+
| threads | --> | composers  | --> | LiveBuffer  |
+---------+     +------------+     +-------------+
       \                           /
        +-------------------------+
```

- `--threads N` enables multi-threaded part generation.
- `--process-pool` uses a process pool instead of threads.
- `--buffer-ahead N` keeps N bars ready during live streaming.
- `--parallel-bars M` sets how many bars to generate in parallel.

Tuning these options allows smooth real-time performances even on slower hardware.
