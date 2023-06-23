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
    bitflip_frequencies,
    bitmask_frequencies,
    convert_counts_to_quasi_dists,
    map_frequencies,
)


################################################################################
## TESTS
################################################################################
class TestMapFrequencies:
    """Test map frequencies."""

    @mark.parametrize(
        "frequencies, map, expected",
        [
            (Counts({}), lambda _: None, {}),
            (QuasiDistribution({}), lambda _: None, {}),
            (Counts({0: 1}), lambda _: 0, {0: 1}),
            (QuasiDistribution({0: 1}), lambda _: 0, {0: 1}),
            (Counts({0: 1}), lambda _: 1, {1: 1}),
            (QuasiDistribution({0: 1}), lambda _: 1, {1: 1}),
            (Counts({0: 1, 1: 1}), lambda _: 1, {1: 2}),
            (QuasiDistribution({0: 0.5, 1: 0.5}), lambda _: 1, {1: 1}),
            (Counts({0: 0, 1: 1}), lambda k: k + 1, {1: 0, 2: 1}),
            (QuasiDistribution({0: 0, 1: 1}), lambda k: k + 1, {1: 0, 2: 1}),
        ],
    )
    def test_map_frequencies(self, frequencies, map, expected):
        """Test map frequencies base functionality."""
        assert map_frequencies(frequencies, map) == type(frequencies)(expected)

    @mark.parametrize("frequencies", TYPES)
    def test_wrong_frequency_type(self, frequencies):
        """Test a non-FrequencyLike input."""
        with raises(TypeError):
            map_frequencies(frequencies, lambda _: None)


class TestBitflipFrequencies:
    """Test bitflip frequencies."""

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
    def test_bitflip_frequencies(self, counts, bitflips, expected):
        """Test bitflip frequencies base functionality."""
        counts = Counts(counts)
        assert bitflip_frequencies(counts, bitflips) == Counts(expected)


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
    def test_bitmask_frequencies(self, counts, mask, expected):
        """Test mask frequencies base functionality."""
        counts = Counts(counts)
        assert bitmask_frequencies(counts, mask) == Counts(expected)


class TestFrequencyConversion:
    """Test conversion from counts to quasi-distributions."""

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
    def test_convert_counts_to_quasi_dists(self, counts):
        """Test convert counts functionality."""
        quasi_dists = convert_counts_to_quasi_dists(counts)
        assert isinstance(quasi_dists, QuasiDistribution)
        assert quasi_dists.shots == counts.shots()
        if quasi_dists.shots:
            assert quasi_dists.stddev_upper_bound == sqrt(1 / quasi_dists.shots)
        assert quasi_dists == {
            k: v / (quasi_dists.shots or 1) for k, v in counts.int_outcomes().items()
        }
