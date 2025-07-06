# Piano Beta Quickstart

Generate a simple piano backing track with pedal CCs and optional counterline.

```bash
modcompose sample dummy.pkl --backend piano_template \
  --voicing drop2 --intensity medium --counterline -o demo.mid
```

![demo gif](docs/img/piano_beta_quick.gif)

#### Intensity & Density

| intensity | RH/LH note density |
|-----------|--------------------|
| low       | 50 % (sparse)      |
| medium    | 100 % (default)    |
| high      | 110 % + anticipation|

Use ``--intensity`` to control note density. See the [test gif](docs/img/piano_beta_quick.gif).
