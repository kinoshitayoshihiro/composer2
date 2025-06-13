import generator.drum_generator as drum_generator

def test_no_kick_only_impl():
    count = drum_generator.DrumGenerator._render_part.__code__.co_consts.count("kick")
    assert count < 3

