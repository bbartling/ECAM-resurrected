import pandas as pd

from ecam_mv.datasets import practice_dataset


def test_reproducible_practice_data_is_explicitly_synthetic():
    a, b = practice_dataset(), practice_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["synthetic"] is True
    assert len(a) == 8760
    assert not a.index.has_duplicates
    assert str(a.index.tz) == "America/Chicago"
    assert (a.energy > 0).all()
    assert a.temperature.min() < 8 and a.temperature.max() > 20
