import pytest
from utilities.gm_perc_map import normalize_label, label_to_number


def test_numeric_string_maps_to_sidestick():
    assert normalize_label("37") == "sidestick"
    assert label_to_number("37") == 37
