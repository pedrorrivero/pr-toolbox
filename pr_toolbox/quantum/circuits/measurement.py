# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum circuits tools."""

from __future__ import annotations

from numpy import arange
from qiskit.circuit import Measure, QuantumCircuit, Qubit
from qiskit.quantum_info.operators import Pauli


# TODO: `QuantumCircuit.measure_pauli(pauli)` (i.e. Qiskit-Terra)
def build_pauli_measurement(pauli: Pauli) -> QuantumCircuit:
    """Build measurement circuit for a given Pauli operator.

    Note: if Pauli is I for all qubits, this function generates a circuit to
    measure only the first qubit. Regardless of whether the result of that only
    measurement is zero or one, the associated expectation value will always
    evaluate to plus one. Therefore, such measurment can be interpreted as a
    constant (1) and does not need to be performed. We leave this behavior as
    default nonetheless.
    """
    measured_qubit_indices = arange(pauli.num_qubits)[pauli.z | pauli.x]
    measured_qubit_indices = set(measured_qubit_indices.tolist()) or {0}
    circuit = QuantumCircuit(pauli.num_qubits, len(measured_qubit_indices))
    for cbit, qubit in enumerate(measured_qubit_indices):
        if pauli.x[qubit]:
            if pauli.z[qubit]:
                circuit.sdg(qubit)
            circuit.h(qubit)
        circuit.measure(qubit, cbit)
    return circuit


def get_measured_qubits(circuit: QuantumCircuit) -> set[Qubit]:
    """Get qubits with at least one measurement gate in them."""
    return {qargs[0] for gate, qargs, _ in circuit if isinstance(gate, Measure)}
