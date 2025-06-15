def test_drum_lint_unknown_instruments():
    from pathlib import Path
    from utilities.drum_lint import check_drum_patterns

    path = Path(__file__).resolve().parents[1] / "data" / "drum_patterns.yml"
    unknown = check_drum_patterns(path)
    assert {"chimes", "hh", "ride_cymbal_swell", "shaker_soft"} <= unknown


def test_rhythm_library_lint_unknown_instruments():
    from pathlib import Path
    from utilities.drum_lint import check_rhythm_library

    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    unknown = check_rhythm_library(path)
    assert {
        "chimes",
        "ghost",
        "hh",
        "ride_cymbal_swell",
        "shaker_soft",
    } <= unknown
