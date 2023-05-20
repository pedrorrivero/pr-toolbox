# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum serialization tools."""

from __future__ import annotations

from qiskit.primitives import EstimatorResult

from pr_toolbox.serialization import NumPyEncoder, ReprEncoder


class EstimatorResultEncoder(NumPyEncoder, ReprEncoder):
    """JSON encoder for :class:`EstimatorResult` objects."""

    def default(self, o):
        if isinstance(o, EstimatorResult):
            return {"values": o.values, "metadata": o.metadata}
        return super().default(o)
