# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Frequency results tools."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import singledispatch
from typing import Union

from numpy import sqrt
from qiskit.result import Counts, QuasiDistribution

FrequenciesLike = Union[Counts, QuasiDistribution]


################################################################################
## FREQUENCIES
################################################################################
@singledispatch
def map_frequencies(
    frequencies: FrequenciesLike | dict, mapper: Callable
) -> FrequenciesLike | dict:
    """Map frequencies by reassigning keys according to input callable.

    Args:
        frequencies: the frequencies to process.
        mapper: the callable to map readout bits (i.e. frequencies keys).

    Returns:
        New frequencies with readout bits mapped according to input callable.
    """
    raise TypeError(
        f"Invalid frequencies type. Expected `Counts` or `QuasiDistribution` or `dict'"
        f" but got {type(frequencies)} instead."
    )


@map_frequencies.register
def map_dict(frequencies: dict, mapper) -> dict:
    """Map frequency dictionary by reassigning keys according to input callable.

    Args:
        frequencies: the frequency dictionary to process.
        mapper: the callable to map readout bits (i.e. frequencies keys).

    Returns:
        New frequency dictionary with readout bits mapped according to input callable.
    """
    if not all(isinstance(k, int) for k in frequencies.keys()):
        raise TypeError("Invalid key types for frequencies. Keys must be of type `int'.")
    if not all(isinstance(v, (int, float)) for v in frequencies.values()):
        raise TypeError(
            "Invalid value types for frequencies. Values must be of type `int' or `float'."
        )
    frequencies_dict: dict[int, int | float] = defaultdict(lambda: 0)
    for readout, freq in frequencies.items():
        readout = mapper(readout)
        frequencies_dict[readout] += freq
    return frequencies_dict


@map_frequencies.register
def map_counts(counts: Counts, mapper: Callable) -> Counts:
    """Map counts by reassigning keys according to input callable.

    Args:
        counts: the counts to process.
        mapper: the callable to map readout bits (i.e. counts keys).

    Returns:
        New counts with readout bits mapped according to input callable.
    """
    frequencies: dict = counts.int_outcomes()
    frequencies = map_frequencies(frequencies, mapper)
    return Counts(frequencies)


@map_frequencies.register
def map_quasi_dist(quasi_dist: QuasiDistribution, mapper: Callable) -> QuasiDistribution:
    """Map quasi-distributions by reassigning keys according to input callable.

    Args:
        quasi_dist: the quasi-distributions to process.
        mapper: the callable to map readout bits (i.e. quasi_dist keys).

    Returns:
        New QuasiDistribution with readout bits mapped according to input callable.
    """
    frequencies: dict = dict(quasi_dist)
    frequencies = map_frequencies(frequencies, mapper)
    return QuasiDistribution(
        frequencies, shots=quasi_dist.shots, stddev_upper_bound=quasi_dist.stddev_upper_bound
    )


def bitflip_frequencies(frequencies: FrequenciesLike, bitflips: int) -> FrequenciesLike:
    """Flip readout bits in frequencies according to the input bitflips (int encoded).

    Args:
        frequencies: the frequencies to process.
        bitflips: the bitflips to be applied.

    Returns:
        New frequencies with readout bits flipped according to input.
    """
    return map_frequencies(frequencies, lambda readout: readout ^ bitflips)


def bitmask_frequencies(frequencies: FrequenciesLike, bitmask: int) -> FrequenciesLike:
    """Apply mask to readout bits in frequencies.

    Args:
        frequencies: the frequencies to process.
        bitmask: the bit mask to be applied.

    Returns:
        New frequencies with readout bits masked according to input.
    """
    return map_frequencies(frequencies, lambda readout: readout & bitmask)


def counts_to_quasi_dist(counts: Counts) -> QuasiDistribution:
    """Infers a :class:`~qiskit.result.QuasiDistribution` from :class:`~qiskit.result.Counts`.

    Args:
        counts: the counts to convert.

    Returns:
        New QuasiDistribution inferred from counts.
    """
    if not isinstance(counts, Counts):
        raise TypeError(f"Invalid counts type. Expected `Counts` but got {type(counts)} instead.")
    shots = counts.shots()
    std_dev = sqrt(1 / shots) if shots else None
    probabilities = {k: v / (shots or 1) for k, v in counts.int_outcomes().items()}
    return QuasiDistribution(probabilities, shots=shots, stddev_upper_bound=std_dev)
