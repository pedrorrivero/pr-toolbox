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
from qiskit.result import Counts, QuasiDistribution


################################################################################
## COUNTS
################################################################################
def map_counts(counts: Counts, mapper: Callable) -> Counts:
    """Map counts by reassigning keys according to input callable.

    Args:
        counts: the counts to process.
        mapper: the callable to map readout bits (i.e. counts keys).

    Returns:
        New counts with readout bits mapped according to input callable.
    """
    counts_dict: dict[int, int] = defaultdict(lambda: 0)
    for readout, freq in counts.int_outcomes().items():
        readout = mapper(readout)
        counts_dict[readout] += freq
    return Counts(counts_dict)


def bitflip_counts(counts: Counts, bitflips: int) -> Counts:
    """Flip readout bits in counts according to the input bitflips (int encoded).

    Args:
        counts: the counts to process.
        bitflips: the bitflips to be applied.

    Returns:
        New counts with readout bits flipped according to input.
    """
    return map_counts(counts, lambda readout: readout ^ bitflips)


def bitmask_counts(counts: Counts, bitmask: int) -> Counts:
    """Apply mask to readout bits in counts.

    Args:
        counts: the counts to process.
        bitmask: the bit mask to be applied.

    Returns:
        New counts with readout bits masked according to input.
    """
    return map_counts(counts, lambda readout: readout & bitmask)


def counts_to_quasi_dists(counts: Counts) -> QuasiDistribution:
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
