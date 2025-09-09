def test_drum_lint_unknown_instruments():
    from pathlib import Path
    from utilities.drum_lint import check_drum_patterns

    path = Path(__file__).resolve().parents[1] / "data" / "drum_patterns.yml"
    unknown = check_drum_patterns(path)
    assert unknown == set()


def test_rhythm_library_lint_unknown_instruments():
    from pathlib import Path
    from utilities.drum_lint import check_rhythm_library

    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    unknown = check_rhythm_library(path)
    assert unknown == set()


def test_dsl_pattern_lint(tmp_path):
    from utilities.drum_lint import check_drum_patterns
    import yaml

    data = {
        "drum_patterns": {
            "demo": {
                "pattern_type": "tom_dsl_fill",
                "pattern": "(T1 K)"
            }
        }
    }
    p = tmp_path / "p.yml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    unknown = check_drum_patterns(p)
    assert unknown == set()
