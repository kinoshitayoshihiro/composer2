
import pytest

from modular_composer import compose

pytest.importorskip("pytest_benchmark")


def dummy_compose(num_workers=1):
    main_cfg = {
        'global_settings': {'time_signature': '4/4', 'tempo_bpm': 120},
        'sections_to_generate': [],
        'part_defaults': {},
        'paths': {'rhythm_library_path': 'data/rhythm_library.yml'},
    }
    chordmap = {'sections': {}}
    rhythm_lib = {}
    compose(main_cfg, chordmap, rhythm_lib, num_workers=num_workers)


def test_compose_benchmark(benchmark):
    benchmark(dummy_compose, num_workers=2)
    assert benchmark.stats.get('mean') <= 0.5
