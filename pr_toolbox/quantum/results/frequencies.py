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

from numpy import sqrt
from qiskit.result import QuasiDistribution, Counts

FrequenciesLike = Counts | QuasiDistribution


################################################################################
## FREQUENCIES
################################################################################
def map_frequencies(frequencies: FrequenciesLike, mapper: Callable) -> FrequenciesLike:
    """Map frequencies by reassigning keys according to input callable.

    Args:
        frequencies: the frequencies to process.
        mapper: the callable to map readout bits (i.e. counts keys).

    Returns:
        New frequencies with readout bits mapped according to input callable.
    """
    frequencies_dict: dict[int, int | float] = defaultdict(lambda: 0)
    frequency_type = type(frequencies)
    if isinstance(frequencies, Counts):
        frequencies = frequencies.int_outcomes()
    for readout, freq in frequencies.items():
        readout = mapper(readout)
        frequencies_dict[readout] += freq
    return frequency_type(frequencies_dict)


def bitflip_frequencies(frequencies: QuasiDistribution | Counts, bitflips: int) -> QuasiDistribution:
    """Flip readout bits in frequencies according to the input bitflips (int encoded).

    Args:
        frequencies: the frequencies to process.
        bitflips: the bitflips to be applied.

    Returns:
        New frequencies with readout bits flipped according to input.
    """
    return map_frequencies(frequencies, lambda readout: readout ^ bitflips)


def bitmask_frequencies(frequencies: QuasiDistribution | Counts, bitmask: int) -> QuasiDistribution:
    """Apply mask to readout bits in frequencies.

    Args:
        frequencies: the frequencies to process.
        bitmask: the bit mask to be applied.

    Returns:
        New counts with readout bits masked according to input.
    """
    return map_frequencies(frequencies, lambda readout: readout & bitmask)


def convert_counts_to_quasi_dists(counts: Counts) -> QuasiDistribution:
    """Infers a :class:`~qiskit.result.QuasiDistribution` from :class:`~qiskit.result.Counts`.

    Args:
        counts: the counts to convert.

    Returns:
        New QuasiDistribution inferred from counts.
    """
    shots = counts.shots()
    std_dev = sqrt(1 / shots) if shots else None
    probabilities = {k: v / (shots or 1) for k, v in counts.int_outcomes().items()}
    return QuasiDistribution(probabilities, shots=shots, stddev_upper_bound=std_dev)
