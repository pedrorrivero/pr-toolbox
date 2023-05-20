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

from collections.abc import Iterator

from pytest import mark
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Pauli

from pr_toolbox.quantum.operators.paulis import (
    build_pauli_measurement,
    pauli_integer_mask,
)


################################################################################
## CASES
################################################################################
def pauli_measurement_cases() -> Iterator[tuple[list[str], QuantumCircuit]]:
    """Generator of Paulis and corresponding measurement circuits.

    Yields:
        - Pauli
        - Quantum circuit to measure said Pauli
    """
    Z = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Z.measure(0, 0)
    yield Pauli("I"), Z
    yield Pauli("Z"), Z

    X = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    X.h(0)
    X.measure(0, 0)
    yield Pauli("X"), X

    Y = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Y.sdg(0)
    Y.h(0)
    Y.measure(0, 0)
    yield Pauli("Y"), Y

    IZ = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    IZ.measure(0, 0)
    yield Pauli("II"), IZ
    yield Pauli("IZ"), IZ

    ZI = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    ZI.measure(1, 0)
    yield Pauli("ZI"), ZI

    IX = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    IX.h(0)
    IX.measure(0, 0)
    yield Pauli("IX"), IX

    XI = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    XI.h(1)
    XI.measure(1, 0)
    yield Pauli("XI"), XI

    IY = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    IY.sdg(0)
    IY.h(0)
    IY.measure(0, 0)
    yield Pauli("IY"), IY

    YI = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    YI.sdg(1)
    YI.h(1)
    YI.measure(1, 0)
    yield Pauli("YI"), YI

    ZZ = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    ZZ.measure(range(2), range(2))
    yield Pauli("ZZ"), ZZ

    ZX = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    ZX.h(0)
    ZX.measure(range(2), range(2))
    yield Pauli("ZX"), ZX

    ZY = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    ZY.sdg(0)
    ZY.h(0)
    ZY.measure(range(2), range(2))
    yield Pauli("ZY"), ZY

    XZ = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XZ.h(1)
    XZ.measure(range(2), range(2))
    yield Pauli("XZ"), XZ

    XX = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XX.h(1)
    XX.h(0)
    XX.measure(range(2), range(2))
    yield Pauli("XX"), XX

    XY = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XY.h(1)
    XY.sdg(0)
    XY.h(0)
    XY.measure(range(2), range(2))
    yield Pauli("XY"), XY

    YZ = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    YZ.sdg(1)
    YZ.h(1)
    YZ.measure(range(2), range(2))
    yield Pauli("YZ"), YZ

    YX = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    YX.sdg(1)
    YX.h(1)
    YX.h(0)
    YX.measure(range(2), range(2))
    yield Pauli("YX"), YX

    YY = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    YY.sdg(1)
    YY.h(1)
    YY.sdg(0)
    YY.h(0)
    YY.measure(range(2), range(2))
    yield Pauli("YY"), YY

    XYZ = QuantumCircuit(3, 3)  # pylint: disable=invalid-name
    XYZ.h(2)
    XYZ.sdg(1)
    XYZ.h(1)
    XYZ.measure([0, 1, 2], [0, 1, 2])
    yield Pauli("XYZ"), XYZ

    YIX = QuantumCircuit(3, 2)  # pylint: disable=invalid-name
    YIX.sdg(2)
    YIX.h(2)
    YIX.h(0)
    YIX.measure([0, 2], [0, 1])
    yield Pauli("YIX"), YIX

    IXII = QuantumCircuit(4, 1)  # pylint: disable=invalid-name
    IXII.h(2)
    IXII.measure(2, 0)
    yield Pauli("IXII"), IXII


################################################################################
## TESTS
################################################################################
class TestBuilPauliMeasurement:
    """Test build Pauli measurement."""

    @mark.parametrize("pauli, expected", [*pauli_measurement_cases()])
    def test_build_pauli_measurement(self, pauli, expected):
        """Test build Pauli measurement circuits base functionality."""
        assert build_pauli_measurement(pauli) == expected


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
