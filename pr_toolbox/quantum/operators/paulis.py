# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pauli operators tools."""

from __future__ import annotations

from functools import reduce
from typing import Any

from numpy import bool_, dtype, ndarray, packbits
from qiskit.quantum_info.operators import Pauli


# TODO: endianess arg
def pauli_integer_mask(pauli: Pauli | str) -> int:
    """Build integer mask for input Pauli.

    This is an integer representation of the binary string with a one where
    there are Paulis, and zero where there are identities.
    """
    pauli = Pauli(pauli)
    pauli_mask: ndarray[Any, dtype[bool_]] = pauli.z | pauli.x
    packed_mask: list[int] = packbits(  # pylint: disable=no-member
        pauli_mask, bitorder="little"
    ).tolist()
    return reduce(lambda value, element: (value << 8) | element, reversed(packed_mask))
