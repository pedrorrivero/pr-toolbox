# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Pauli operators tools."""

from __future__ import annotations

from pytest import mark
from qiskit.quantum_info.operators import Pauli

from pr_toolbox.quantum.operators.paulis import pauli_integer_mask


################################################################################
## TESTS
################################################################################
class TestPauliIntegerMask:
    """Test Pauli integer mask."""

    @mark.parametrize(
        "pauli, expected",
        [
            (Pauli("I"), 0b0),
            (Pauli("Z"), 0b1),
            (Pauli("X"), 0b1),
            (Pauli("Y"), 0b1),
            (Pauli("II"), 0b00),
            (Pauli("IZ"), 0b01),
            (Pauli("ZI"), 0b10),
            (Pauli("ZZ"), 0b11),
            (Pauli("IX"), 0b01),
            (Pauli("XI"), 0b10),
            (Pauli("ZX"), 0b11),
            (Pauli("XZ"), 0b11),
            (Pauli("XX"), 0b11),
            (Pauli("IY"), 0b01),
            (Pauli("YI"), 0b10),
            (Pauli("ZY"), 0b11),
            (Pauli("YZ"), 0b11),
            (Pauli("YY"), 0b11),
            (Pauli("XY"), 0b11),
            (Pauli("YX"), 0b11),
            (Pauli("XZIZIYIXIIXI"), 0b110101010010),
        ],
    )
    def test_pauli_integer_mask(self, pauli, expected):
        """Test Pauli integer mask base functionality."""
        assert pauli_integer_mask(pauli) == expected
