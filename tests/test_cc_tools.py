import utilities.cc_tools as cc

def test_merge_cc_events_dict_support():
    base = [(0.0, 31, 40)]
    more = [{"time": 0.5, "cc": 31, "val": 50}]
    merged = cc.merge_cc_events(base, more)
    assert (0.5, 31, 50) in merged
    assert (0.0, 31, 40) in merged
    times = [e[0] for e in merged]
    assert times == sorted(times)
