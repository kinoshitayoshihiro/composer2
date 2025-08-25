from pathlib import Path

from tools.corpus_to_phrase_csv import hash_split


def test_hash_split_deterministic(tmp_path: Path) -> None:
    files = [tmp_path / f"f{i}.mid" for i in range(6)]
    for p in files:
        p.touch()
    train1, valid1 = hash_split(files, 0.3)
    train2, valid2 = hash_split(list(reversed(files)), 0.3)
    assert {p.name for p in train1} == {p.name for p in train2}
    assert {p.name for p in valid1} == {p.name for p in valid2}

