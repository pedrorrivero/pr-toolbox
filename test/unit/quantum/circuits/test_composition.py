# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit composition tools."""

from __future__ import annotations

from functools import reduce

from numpy.random import default_rng
from pytest import mark
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit

from pr_toolbox.quantum.circuits.composition import compose_circuits_w_metadata


################################################################################
## TESTS
################################################################################
class TestComposeCircuitsWMetadata:
    """Test compose circuits."""

    @mark.parametrize("num_circuits, num_qubits, seed", zip(range(2, 5), range(1, 5), range(5)))
    def test_compose_circuits_w_metadata(self, num_circuits, num_qubits, seed):
        """Test compose circuits base functionality."""
        rng = default_rng(seed)
        seeds = tuple(rng.integers(256) for _ in range(num_circuits))
        # Case
        circuits = tuple(random_circuit(num_qubits, num_qubits, seed=s) for s in seeds)
        for circuit, seed in zip(circuits, seeds):
            circuit.metadata = {"seed": seed}
        expected = reduce(lambda base, next: base.compose(next), circuits)  # Note: deepcopies
        expected.metadata = {k: v for c in circuits for k, v in c.metadata.items()}
        # Test
        composition = compose_circuits_w_metadata(*circuits, inplace=False)
        assert composition is not circuits[0]
        assert composition == expected
        assert composition.metadata == expected.metadata
        composition = compose_circuits_w_metadata(*circuits, inplace=True)
        assert composition is circuits[0]
        assert composition == expected
        assert composition.metadata == expected.metadata

    def test_inplace_default(self):
        """Test inplace arg default (i.e. inplace=False)."""
        # Case
        qc = QuantumCircuit(3)
        qc.x(qc.qubits)
        qc.metadata = {}
        circuits = [qc] * 4
        # Test
        assert compose_circuits_w_metadata(*circuits) is not circuits[0]
