# Offline IR Convolution

The rendered audio can be processed with a cabinet or room impulse response offline.
Simply pass the `--ir` option to the guitar generator:

```bash
python -m generator.guitar_generator --render section.yml --ir irs/blackface.wav
```

This applies the given IR using `utilities.convolver.render_with_ir`.
