from utilities.phrase_filter import NGramDiversityFilter


def test_diversity_filter() -> None:
    flt = NGramDiversityFilter(n=2, max_sim=0.5)
    a = [{"instrument": "kick"}, {"instrument": "snare"}, {"instrument": "kick"}]
    b = [{"instrument": "kick"}, {"instrument": "snare"}, {"instrument": "kick"}]
    assert not flt.too_similar(a)
    assert flt.too_similar(b)
