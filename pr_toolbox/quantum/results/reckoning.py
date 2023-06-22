# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Result reckoning tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Sequence

from numpy import array, dot, real_if_close, sqrt, vstack
from qiskit.opflow import PauliSumOp
from qiskit.primitives.utils import init_observable as normalize_operator
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, QuasiDistribution

from pr_toolbox.binary import parity_bit
from pr_toolbox.quantum.operators import pauli_integer_mask

from .frequencies import bitmask_frequencies, convert_counts_to_quasi_dists, FrequenciesLike

ReckoningResult = namedtuple("ReckoningResult", ("expval", "std_error"))
OperatorType = BaseOperator | PauliSumOp | str  # TODO: to types


################################################################################
## EXPECTATION VALUE RECKONER INTERFACE
################################################################################
# TODO: defer validation until necessary (e.g. `counts`)
# TODO: generalize to non pauli-based observables
class ExpvalReckoner(ABC):
    """Expectation value reckoning interface.

    Classes implementing this interface provide methods for constructing expectation values
    and associated errors out of raw frequencies and operators.
    """

    ################################################################################
    ## API
    ################################################################################
    def reckon(
            self,
            frequencies_list: Sequence[FrequenciesLike] | FrequenciesLike,
            operator_list: Sequence[OperatorType] | OperatorType,
    ) -> ReckoningResult:
        """Compute expectation value for the sum of the input operators.

        Note: the input operators need to be measurable entirely within one circuit
        execution (i.e. resulting in the one-to-one associated input frequencies). Users must
        ensure that all frequencies entries come from the appropriate circuit execution.

        args:
            frequencies: a :class:`~qiskit.result.Counts` or `~qiskit.result.QuasiDistribution`
             object from circuit execution.
            operators: a list of operators associated one-to-one to the input frequencies.

        Returns:
            The expectation value and associated std-error for the sum of the input operators.
        """
        frequencies_list = self._validate_frequencies_list(frequencies_list)
        operator_list = self._validate_operator_list(operator_list)
        self._cross_validate_lists(frequencies_list, operator_list)
        expval, std_error = self._reckon(frequencies_list, operator_list)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_operator(self, frequencies: FrequenciesLike, operator: OperatorType) -> ReckoningResult:
        """Reckon expectation value from frequencies and operator.

        Note: This function assumes that the input operators are measurable entirely
        within one circuit execution (i.e. resulting in the input frequencies), and that
        the appropriate changes of bases (i.e. rotations) were actively performed in
        the relevant qubits before readout; hence diagonalizing the input operators.

        Returns:
            A two-tuple containing the expectation value and associated std error
            for the input operator. Expectation values can have both real and
            imaginary components, which can be interpreted as corresponding to the
            hermitian and anti-hermitian components of the input operator
            respectively. Standard errors will always be real valued.
        """
        frequencies = self._validate_frequencies(frequencies)
        operator = self._validate_operator(operator)
        # TODO: cross-validation
        expval, std_error = self._reckon_operator(frequencies, operator)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_pauli(self, frequencies: FrequenciesLike, pauli: Pauli) -> ReckoningResult:
        """Reckon expectation value from frequencies and pauli.

        Note: This function treats X, Y, and Z Paulis identically, assuming that
        the appropriate changes of bases (i.e. rotations) were actively performed
        in the relevant qubits before readout; hence diagonalizing the input Pauli.

        Returns:
            A two-tuple containing the expectation value and associated std error
            for the input Pauli. Expectation values can be real or imaginary
            based on the phase associated to the input Pauli. Standard errors
            will always be real valued.
        """
        frequencies = self._validate_frequencies(frequencies)
        pauli = self._validate_pauli(pauli)
        # TODO: cross-validation
        expval, std_error = self._reckon_pauli(frequencies, pauli)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_frequencies(self, frequencies: FrequenciesLike) -> ReckoningResult:
        """Reckon expectation value and associated std error from frequencies.

        Returns:
            A two-tuple containing the expectation value and associated std error
            for the input frequencies. The measurement basis is implicit in the way the
            input frequencies were produced, therefore the resulting value can be
            regarded as coming from a multi-qubit Pauli-Z operator (i.e. a fully
            diagonal Pauli observable).
        """
        frequencies = self._validate_frequencies(frequencies)
        expval, std_error = self._reckon_frequencies(frequencies)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    ################################################################################
    ## ABSTRACT METHODS
    ################################################################################
    @abstractmethod
    def _reckon(
            self,
            frequencies_list: Sequence[QuasiDistribution],
            operator_list: Sequence[SparsePauliOp],
    ) -> ReckoningResult:
        expval = 0.0
        variance = 0.0
        for value, error in (
                self._reckon_operator(frequencies, operator)
                for frequencies, operator in zip(frequencies_list, operator_list)
        ):
            expval += value
            variance += error ** 2
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_operator(self, frequencies: QuasiDistribution, operator: SparsePauliOp) -> ReckoningResult:
        value_std_error_pairs = [self._reckon_pauli(frequencies, pauli) for pauli in operator.paulis]
        values, std_errors = vstack(value_std_error_pairs).T  # Note: like zip but array output
        coeffs = array(operator.coeffs)
        expval = dot(values, coeffs)
        variance = dot(std_errors.real ** 2, (coeffs.real ** 2 + coeffs.imag ** 2))
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_pauli(self, frequencies: QuasiDistribution, pauli: Pauli) -> ReckoningResult:
        mask = pauli_integer_mask(pauli)
        frequencies = bitmask_frequencies(frequencies, mask)
        coeff = (-1j) ** pauli.phase
        expval, std_error = self._reckon_frequencies(frequencies)
        return ReckoningResult(coeff * expval, std_error)

    @abstractmethod
    def _reckon_frequencies(self, frequencies: QuasiDistribution) -> ReckoningResult:
        expval: float = 0.0
        for readout, freq in frequencies.items():
            observation = (-1) ** parity_bit(readout, even=True)
            expval += observation * freq
        variance = 1 - expval ** 2
        std_error = sqrt(variance)
        return ReckoningResult(expval, std_error)

    ################################################################################
    ## AUXILIARY
    ################################################################################

    @classmethod
    def _validate_frequencies_list(cls, frequencies_list: Sequence[FrequenciesLike] | FrequenciesLike) -> tuple[
        QuasiDistribution, ...]:
        """Validate frequencies."""
        if isinstance(frequencies_list, (FrequenciesLike, dict)):
            frequencies_list = (frequencies_list,)
        if not isinstance(frequencies_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_frequencies(f) for f in frequencies_list)

    @staticmethod
    def _validate_frequencies(frequencies: FrequenciesLike) -> QuasiDistribution:
        """Validate frequencies."""
        if isinstance(frequencies, (Counts, dict)):
            frequencies = Counts(frequencies)
            frequencies = convert_counts_to_quasi_dists(frequencies)
        if not isinstance(frequencies, QuasiDistribution):
            raise TypeError("Expected QuasiDistribution object.")
        return frequencies

    @classmethod
    def _validate_operator_list(
            cls,
            operator_list: Sequence[OperatorType] | OperatorType,
    ) -> tuple[SparsePauliOp, ...]:
        """Validate operator list."""
        if isinstance(operator_list, (BaseOperator, PauliSumOp, str)):
            operator_list = (operator_list,)
        if not isinstance(operator_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_operator(o) for o in operator_list)

    @staticmethod
    def _validate_operator(operator: OperatorType) -> SparsePauliOp:
        """Validate operator."""
        if isinstance(operator, (BaseOperator, PauliSumOp, str)):
            return normalize_operator(operator)
        raise TypeError("Expected OperatorType object.")

    @staticmethod
    def _validate_pauli(pauli: Pauli) -> Pauli:
        """Validate Pauli."""
        if isinstance(pauli, str):
            pauli = Pauli(pauli)
        elif not isinstance(pauli, Pauli):
            raise TypeError(f"Expected Pauli, got {pauli!r} instead.")
        return pauli

    @staticmethod
    def _cross_validate_lists(
            frequencies_list: Sequence[QuasiDistribution], operator_list: Sequence[SparsePauliOp]
    ) -> None:
        """Cross validate frequencies and operator lists."""
        # TODO: validate num_bits -> Need to check every entry in frequencies (expensive)
        if len(frequencies_list) != len(operator_list):
            raise ValueError(
                f"The number of frequencies entries ({len(frequencies_list)}) does not match "
                f"the number of operators ({len(operator_list)})."
            )


################################################################################
## IMPLEMENTATION
################################################################################
# pylint: disable=useless-parent-delegation
class CanonicalReckoner(ExpvalReckoner):
    """Canonical expectation value reckoning class."""

    def _reckon(
            self, frequencies_list: Sequence[QuasiDistribution], operator_list: Sequence[SparsePauliOp]
    ) -> ReckoningResult:
        return super()._reckon(frequencies_list, operator_list)

    def _reckon_operator(self, frequencies: QuasiDistribution, operator: SparsePauliOp) -> ReckoningResult:
        return super()._reckon_operator(frequencies, operator)

    def _reckon_pauli(self, frequencies: QuasiDistribution, pauli: Pauli) -> ReckoningResult:
        return super()._reckon_pauli(frequencies, pauli)

    def _reckon_frequencies(self, frequencies: QuasiDistribution) -> ReckoningResult:
        return super()._reckon_frequencies(frequencies)
