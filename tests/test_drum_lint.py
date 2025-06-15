def test_drum_lint_unknown_instruments():
    from pathlib import Path
    from utilities.drum_lint import check_drum_patterns

    path = Path(__file__).resolve().parents[1] / "data" / "drum_patterns.yml"
    unknown = check_drum_patterns(path)
    assert {"chimes", "hh", "ride_cymbal_swell", "shaker_soft"} <= unknown
