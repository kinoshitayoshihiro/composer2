import filecmp
import pathlib
import pytest

from utilities.midi_export import write_demo_bar


def test_golden_bar(tmp_path: pathlib.Path, request: pytest.FixtureRequest) -> None:
    demo = tmp_path / "bar.mid"
    write_demo_bar(demo)
    golden = pathlib.Path(__file__).with_suffix("").parent / "golden" / "bar_simple.mid"
    if request.config.getoption("--generate-golden"):
        golden.write_bytes(demo.read_bytes())
        pytest.skip("Golden regenerated")
    if not golden.exists():
        pytest.skip("Golden MIDI missing")
    assert filecmp.cmp(demo, golden, shallow=False)
