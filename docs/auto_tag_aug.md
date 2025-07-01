# Auto-Tag & Augmentation

Use the provided utilities to label your dataset and expand it with pitch or tempo shifts.

```python
from modular_composer import auto_tag
from modular_composer import augment

auto_tag('input.mid', 'tags.yaml')
augment('tags.yaml', output_dir='aug')
```
