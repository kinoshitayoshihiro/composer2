import importlib
from pathlib import Path
import pytest

# 1) ベーシック取得

def test_basic_lookup(tmp_path):
    """YAML から emotion+mode で Progression が取れる."""
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text(
        "soft_reflective:\n"
        "  major:\n"
        "    - 'I V vi IV'\n"
        "  minor:\n"
        "    - 'i VII VI VII'\n"
    )
    mod = importlib.import_module("utilities.progression_templates")
    lst = mod.get_progressions("soft_reflective", mode="major", path=yaml_file)
    assert lst == ["I V vi IV"]

# 2) キャッシュ確認 (lru_cache)

def test_cache_identity(tmp_path):
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text("dummy: {}\n")
    mod = importlib.import_module("utilities.progression_templates")
    id1 = id(mod._load(path=yaml_file))
    id2 = id(mod._load(path=yaml_file))
    assert id1 == id2, "lru_cache should return same dict instance"

# 3) エラー系
@pytest.mark.parametrize("bucket, mode", [("missing", "major"), ("soft_reflective", "dorian")])
def test_key_error(tmp_path, bucket, mode):
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text("soft_reflective:\n  major: ['I IV V']\n")
    import utilities.progression_templates as pt
    with pytest.raises(KeyError):
        pt.get_progressions(bucket, mode=mode, path=yaml_file)
