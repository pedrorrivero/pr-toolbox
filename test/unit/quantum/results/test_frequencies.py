# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for frequency results tools."""
from test import TYPES

from numpy import sqrt
from pytest import mark, raises
from qiskit.result import Counts, QuasiDistribution

from pr_toolbox.quantum.results.frequencies import (
    bitflip_counts,
    bitmask_counts,
    counts_to_quasi_dists,
    map_counts,
)


################################################################################
## TESTS
################################################################################
class TestMapCounts:
    """Test map counts."""

    @mark.parametrize(
        "counts, map, expected",
        [
            ({}, lambda _: None, {}),
            ({0: 1}, lambda _: 0, {0: 1}),
            ({0: 1}, lambda _: 1, {1: 1}),
            ({0: 1, 1: 1}, lambda _: 1, {1: 2}),
            ({0: 0, 1: 1}, lambda k: k + 1, {1: 0, 2: 1}),
        ],
    )
    def test_map_counts(self, counts, map, expected):
        """Test map counts base functionality."""
        counts = Counts(counts)
        assert map_counts(counts, map) == Counts(expected)


class TestBitflipCounts:
    """Test bitflip counts."""

    @mark.parametrize(
        "counts, bitflips, expected",
        [
            ({0b00: 0, 0b01: 1}, 0b00, {0b00: 0, 0b01: 1}),
            ({0b00: 0, 0b01: 1}, 0b01, {0b00: 1, 0b01: 0}),
            ({0b00: 0, 0b01: 1}, 0b10, {0b10: 0, 0b11: 1}),
            ({0b00: 0, 0b01: 1}, 0b11, {0b10: 1, 0b11: 0}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b00, {0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b01, {0b00: 1, 0b01: 0, 0b10: 3, 0b11: 2}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b10, {0b00: 2, 0b01: 3, 0b10: 0, 0b11: 1}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b11, {0b00: 3, 0b01: 2, 0b10: 1, 0b11: 0}),
        ],
    )
    def test_bitflip_counts(self, counts, bitflips, expected):
        """Test bitflip counts base functionality."""
        counts = Counts(counts)
        assert bitflip_counts(counts, bitflips) == Counts(expected)


class TestMaskCounts:
    """Test mask counts."""

    @mark.parametrize(
        "counts, mask, expected",
        [
            ({0b00: 0, 0b01: 1}, 0b00, {0b00: 1}),
            ({0b00: 0, 0b01: 1}, 0b01, {0b00: 0, 0b01: 1}),
            ({0b00: 0, 0b01: 1}, 0b10, {0b00: 1}),
            ({0b00: 0, 0b01: 1}, 0b11, {0b00: 0, 0b01: 1}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b00, {0b00: 6}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b01, {0b00: 2, 0b01: 4}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b10, {0b00: 1, 0b10: 5}),
            ({0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}, 0b11, {0b00: 0, 0b01: 1, 0b10: 2, 0b11: 3}),
        ],
    )
    def test_bitmask_counts(self, counts, mask, expected):
        """Test mask counts base functionality."""
        counts = Counts(counts)
        assert bitmask_counts(counts, mask) == Counts(expected)


class TestFrequencyConversion:
    """Test conversion from counts to quasi-distributions."""

    @mark.parametrize("counts", TYPES)
    def test_wrong_counts_type(self, counts):
        """Test wrong counts types upon conversion."""
        with raises(TypeError):
            counts_to_quasi_dists(counts)

    @mark.parametrize(
        "counts",
        [
            Counts({}),
            Counts({0b00: 0, 0b01: 1}),
            Counts({0: 1, 1: 1}),
            Counts({0: 0, 1: 5}),
            Counts({12: 1, 13: 5, 14: 1}),
            Counts({5: 6}),
        ],
    )
    def test_counts_to_quasi_dists(self, counts):
        """Test convert counts functionality."""
        quasi_dists = counts_to_quasi_dists(counts)
        assert isinstance(quasi_dists, QuasiDistribution)
        assert quasi_dists.shots == counts.shots()
        if quasi_dists.shots:
            assert quasi_dists.stddev_upper_bound == sqrt(1 / quasi_dists.shots)
        assert quasi_dists == {
            k: v / (quasi_dists.shots or 1) for k, v in counts.int_outcomes().items()
        }
