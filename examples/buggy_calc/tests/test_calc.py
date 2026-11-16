import pytest

from calc import is_prime, mean, median


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5
    assert mean([5]) == 5


def test_median_unsorted_input():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5


def test_is_prime():
    assert [n for n in range(12) if is_prime(n)] == [2, 3, 5, 7, 11]


def test_empty_inputs():
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        median([])
