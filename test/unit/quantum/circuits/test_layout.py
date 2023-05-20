# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit layout tools."""

from __future__ import annotations

from pytest import mark
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.transpiler import Layout

from pr_toolbox.quantum.circuits.layout import (
    infer_final_permutation,
    infer_initial_permutation,
    infer_total_layout,
    infer_total_permutation,
)


################################################################################
## CASES
################################################################################
def permutation_cases() -> tuple[int, list[int]]:
    """Permutation cases.

    Yields:
        - Number of qubits after transpilation (i.e. target)
        - Initial permutation
    """
    yield 2, [0]
    yield 2, [1]
    yield 2, [0, 1]
    yield 2, [1, 0]
    yield 3, [0]
    yield 3, [1]
    yield 3, [2]
    yield 3, [0, 1]
    yield 3, [0, 2]
    yield 3, [1, 0]
    yield 3, [1, 2]
    yield 3, [2, 0]
    yield 3, [2, 1]
    yield 3, [0, 1, 2]
    yield 3, [1, 2, 0]
    yield 3, [2, 0, 1]
    yield 3, [2, 1, 0]
    yield 3, [1, 0, 2]
    yield 3, [0, 2, 1]


def radial_coupling_map(num_qubits: int, center: int = 0) -> list[list[int]]:
    """Build a radial, directed, coupling map."""
    return [[center, qubit] for qubit in range(num_qubits) if qubit != center]


def radial_circuit(num_qubits: int, center: int = 0) -> QuantumCircuit:
    """Build a radial circuit."""
    circuit = QuantumCircuit(num_qubits)
    for qubit in (q for q in range(num_qubits) if q != center):
        circuit.cx(center, qubit)
    return circuit


################################################################################
## TESTS
################################################################################
class TestInferInitialPermutation:
    """Test infer initial transpilation permutation."""

    @mark.parametrize("target_num_qubits, layout", permutation_cases())
    def test_permutation(self, target_num_qubits, layout):
        """Test permutation."""
        num_qubits = len(layout)
        circuit = transpile(
            radial_circuit(num_qubits, center=0),
            initial_layout=layout,
            coupling_map=radial_coupling_map(target_num_qubits, center=0),
        )
        expected = layout + [i for i in range(target_num_qubits) if i not in layout]
        assert infer_initial_permutation(circuit) == tuple(expected)

    @mark.parametrize("num_qubits", range(1, 10))
    def test_untranspiled(self, num_qubits):
        """Test trivial permutation if circuit has not been transpiled."""
        circuit = random_circuit(num_qubits, depth=1, seed=num_qubits)
        assert infer_initial_permutation(circuit) == tuple(range(num_qubits))


class TestInferFinalPermutation:
    """Test infer final transpilation permutation."""

    @mark.parametrize("target_num_qubits, layout", permutation_cases())
    def test_permutation(self, target_num_qubits, layout):
        """Test permutation."""
        num_qubits = len(layout)
        circuit = transpile(
            radial_circuit(num_qubits, center=0),
            initial_layout=layout,
            coupling_map=radial_coupling_map(target_num_qubits, center=0),
        )
        expected = (
            range(target_num_qubits)
            if not circuit.layout.final_layout
            else (circuit.layout.final_layout[q] for q in circuit.qubits)
        )
        assert infer_final_permutation(circuit) == tuple(expected)

    @mark.parametrize("num_qubits", range(1, 10))
    def test_untranspiled(self, num_qubits):
        """Test trivial permutation if circuit has not been transpiled."""
        circuit = random_circuit(num_qubits, depth=1, seed=num_qubits)
        assert infer_final_permutation(circuit) == tuple(range(num_qubits))


class TestInferTotalPermutation:
    """Test infer total transpilation permutation."""

    @mark.parametrize("target_num_qubits, layout", permutation_cases())
    def test_permutation(self, target_num_qubits, layout):
        """Test permutation."""
        num_qubits = len(layout)
        circuit = transpile(
            radial_circuit(num_qubits, center=0),
            initial_layout=layout,
            coupling_map=radial_coupling_map(target_num_qubits, center=0),
        )
        initial = layout + [i for i in range(target_num_qubits) if i not in layout]
        final = list(
            range(target_num_qubits)
            if not circuit.layout.final_layout
            else (circuit.layout.final_layout[q] for q in circuit.qubits)
        )
        assert infer_total_permutation(circuit) == tuple(final[q] for q in initial)

    @mark.parametrize("num_qubits", range(1, 10))
    def test_untranspiled(self, num_qubits):
        """Test trivial permutation if circuit has not been transpiled."""
        circuit = random_circuit(num_qubits, depth=1, seed=num_qubits)
        assert infer_total_permutation(circuit) == tuple(range(num_qubits))


class TestInferTotalLayout:
    """Test infer total transpilation layout."""

    @mark.parametrize("target_num_qubits, layout", permutation_cases())
    def test_layout(self, target_num_qubits, layout):
        """Test layout."""
        num_qubits = len(layout)
        circuit = transpile(
            radial_circuit(num_qubits, center=0),
            initial_layout=layout,
            coupling_map=radial_coupling_map(target_num_qubits, center=0),
        )
        initial = circuit.layout.initial_layout
        final = circuit.layout.final_layout or Layout(dict(enumerate(circuit.qubits)))
        expected = {q: final[circuit.qubits[i]] for q, i in initial.get_virtual_bits().items()}
        assert infer_total_layout(circuit) == Layout(expected)

    @mark.parametrize("num_qubits", range(1, 10))
    def test_untranspiled(self, num_qubits):
        """Test trivial layout if circuit has not been transpiled."""
        circuit = random_circuit(num_qubits, depth=1, seed=num_qubits)
        assert infer_total_layout(circuit) == Layout(dict(enumerate(circuit.qubits)))
