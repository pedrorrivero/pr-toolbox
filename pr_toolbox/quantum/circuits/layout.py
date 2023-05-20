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
    num_qubits = len(transpile_layout.input_qubit_mapping)
    input_qubit_mapping = transpile_layout.input_qubit_mapping
    inverted_input_mapping = {i: q for q, i in input_qubit_mapping.items()}
    input_qubits = tuple(inverted_input_mapping[i] for i in range(num_qubits))
    initial_layout = transpile_layout.initial_layout
    return tuple(initial_layout[q] for q in input_qubits)


# TODO: output_qubits = circuit.qubits (simplify)
def infer_final_permutation(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Infer final transpilation permutation (i.e. from physical circuit start to end)."""
    transpile_layout: TranspileLayout = circuit.layout
    if transpile_layout is None:
        return tuple(range(circuit.num_qubits))
    num_qubits = len(transpile_layout.input_qubit_mapping)
    output_qubit_mapping = {q: i for i, q in enumerate(circuit.qubits)}
    inverted_output_mapping = {i: q for q, i in output_qubit_mapping.items()}
    output_qubits = tuple(inverted_output_mapping[i] for i in range(num_qubits))
    final_layout = transpile_layout.final_layout or Layout(output_qubit_mapping)
    return tuple(final_layout[q] for q in output_qubits)


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
    if circuit.layout is None:
        return Layout(dict(enumerate(circuit.qubits)))
    initial = circuit.layout.initial_layout
    final = circuit.layout.final_layout or Layout(dict(enumerate(circuit.qubits)))
    layout_dict = {q: final[circuit.qubits[i]] for q, i in initial.get_virtual_bits().items()}
    return Layout(layout_dict)
