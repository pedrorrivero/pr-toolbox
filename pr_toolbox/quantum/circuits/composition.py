# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit composition tools."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit


################################################################################
## UTILS
################################################################################
def compose_circuits_w_metadata(*circuits: QuantumCircuit, inplace: bool = False) -> QuantumCircuit:
    """Compose quantum circuits merging metadata."""
    # TODO: `circuit.compose(qc, inplace=True)` return `self` (i.e. Qiskit-Terra)
    # Note: simplified implementation after above TODO using `functools.reduce`
    # composition = reduce(lambda base, next: base.compose(next, inplace=True), circuits)
    # composition.metadata = {k: v for c in circuits for k, v in c.metadata.items()}
    composition = circuits[0] if inplace else circuits[0].copy()
    appendages = (circuits[i] for i in range(1, len(circuits)))
    for circuit in appendages:
        composition.compose(circuit, inplace=True)
        composition.metadata.update(circuit.metadata)
    return composition
