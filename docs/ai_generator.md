# AI Bass Generator

This module provides a lightweight Transformer-based bass model.
Install optional dependencies:

```bash
pip install modular-composer[ai]
```

Supported rhythm tokens:

| Token | Description |
|-------|-------------|
| `<straight8>` | Even eighths |
| `<swing16>` | Swing 16th feel |
| `<shuffle>` | Shuffle groove |

Load a model and generate a few bars:

```python
from utilities.bass_transformer import BassTransformer
model = BassTransformer("gpt2-medium")
notes = model.sample([0], top_k=8, temperature=1.0)
```

LoRA adapters can be loaded with the `lora_path` argument.

Quick sampling via CLI:

```bash
modcompose sample model.pkl --backend transformer --model-name gpt2-medium \
  --rhythm-schema <straight8>
```
