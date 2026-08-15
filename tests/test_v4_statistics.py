"""Tests for the v4 uncertainty helpers."""

import math

import pytest

from v4_statistics import percent_with_interval, poisson_rate_interval, wilson_interval


def test_wilson_interval_validates_and_handles_empty_denominator():
    assert all(math.isnan(value) for value in wilson_interval(0, 0))
    with pytest.raises(ValueError, match="between zero and total"):
        wilson_interval(2, 1)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, rel=1e-5)
    assert high == pytest.approx(0.763407, rel=1e-5)


def test_poisson_rate_interval_validates_and_handles_zero_event():
    with pytest.raises(ValueError, match="non-negative"):
        poisson_rate_interval(-1, 10)
    assert all(math.isnan(value) for value in poisson_rate_interval(1, 0))
    assert all(math.isnan(value) for value in poisson_rate_interval(1, 10, scale=0))
    low, high = poisson_rate_interval(0, 100, scale=1_000)
    assert low == 0.0
    assert high > 0.0
    low, high = poisson_rate_interval(4, 200, scale=1_000)
    assert 0.0 < low < 20.0 < high


def test_percent_with_interval_uses_percentage_units():
    assert all(math.isnan(value) for value in percent_with_interval(0, 0))
    estimate, low, high = percent_with_interval(8, 10)
    assert estimate == 80.0
    assert low < estimate < high
