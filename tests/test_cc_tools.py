import utilities.cc_tools as cc

def test_merge_cc_events_override() -> None:
    base = [(0.0, 31, 40)]
    more = [(0.0, 31, 50), (0.5, 31, 60)]
    merged = cc.merge_cc_events(set(base), set(more))
    assert merged.count((0.0, 31, 50)) == 1
    assert merged[-1] == (0.5, 31, 60)
