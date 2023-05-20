# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit layout tools."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.layout import Layout, TranspileLayout


################################################################################
## PERMUTATIONS
################################################################################
def infer_initial_permutation(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Infer initial transpilation permutation (i.e. from virtual circuit to physical start)."""
    transpile_layout: TranspileLayout = circuit.layout
    if transpile_layout is None:
        return tuple(range(circuit.num_qubits))
    position_qubit_mapping = {p: q for q, p in transpile_layout.input_qubit_mapping.items()}
    num_qubits = len(position_qubit_mapping)
    virtual_qubits = (position_qubit_mapping[p] for p in range(num_qubits))
    return tuple(transpile_layout.initial_layout[q] for q in virtual_qubits)


def infer_final_permutation(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Infer final transpilation permutation (i.e. from physical circuit start to end)."""
    transpile_layout: TranspileLayout = circuit.layout
    if transpile_layout is None or transpile_layout.final_layout is None:
        return tuple(range(circuit.num_qubits))
    return tuple(transpile_layout.final_layout[q] for q in circuit.qubits)


def infer_total_permutation(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Infer total transpilation permutation (i.e. from virtual circuit to physical end)."""
    initial_permutation = infer_initial_permutation(circuit)
    final_permutation = infer_final_permutation(circuit)
    return tuple(final_permutation[i] for i in initial_permutation)


################################################################################
## LAYOUTS
################################################################################
def infer_total_layout(circuit: QuantumCircuit) -> Layout:
    """Infer total transpilation layout (i.e. from virtual circuit to physical end)."""
    transpile_layout: TranspileLayout = circuit.layout
    if transpile_layout is None:
        return Layout(dict(enumerate(circuit.qubits)))
    initial = transpile_layout.initial_layout
    final = transpile_layout.final_layout or Layout(dict(enumerate(circuit.qubits)))
    layout_dict = {q: final[circuit.qubits[i]] for q, i in initial.get_virtual_bits().items()}
    return Layout(layout_dict)
