# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for binary tools."""

from __future__ import annotations

from pytest import mark

from pr_toolbox.binary import binary_digit, parity_bit


################################################################################
## TESTS
################################################################################
class TestParityBit:
    """Test parity bit."""

    @mark.parametrize("integer", [0b000, 0b011, 0b101, 0b110])
    def test_zero_even(self, integer):
        assert parity_bit(integer, even=True) == 0
        assert parity_bit(integer, even=False) == 1

    @mark.parametrize("integer", [0b001, 0b010, 0b100, 0b111])
    def test_zero_odd(self, integer):
        assert parity_bit(integer, even=True) == 1
        assert parity_bit(integer, even=False) == 0


class TestBinaryDigit:
    """Test binary digit."""

    @mark.parametrize(
        "integer, bits", [(0b00, [0, 0]), (0b01, [1, 0]), (0b10, [0, 1]), (0b11, [1, 1])]
    )
    def test_binary_digit(self, integer, bits):
        """Test binary digit base functionality."""
        for place, expected in enumerate(bits):
            assert binary_digit(integer, place) == expected
