from grade_eval.split import TEST, TRAIN, VAL, split_of


def test_split_is_deterministic_and_covers_all_sets():
    regs = [f"23BCE{n:04d}" for n in range(1000, 1400)]
    got = [split_of(r) for r in regs]
    assert got == [split_of(r) for r in regs]
    assert {TRAIN, VAL, TEST} <= set(got)
    assert 0.55 < got.count(TRAIN) / len(got) < 0.75
