# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quantum serialization tools."""

from numpy import array
from pytest import mark
from qiskit.primitives import EstimatorResult

from pr_toolbox.quantum.serialization import (
    EstimatorResultEncoder,
    NumPyEncoder,
    ReprEncoder,
)


class TestEstimatorResultEncoder:
    """Test EstimatorResultEncoder class."""

    @mark.parametrize(
        "values, metadata",
        zip(
            [
                array([]),
                array([1]),
                array([1, 2]),
            ],
            [
                [],
                [{"variance": 0}],
                [{"variance": 0}, {"variance": 1}],
            ],
        ),
    )
    def test_default(self, values, metadata):
        """Test default method."""
        result = EstimatorResult(values=values, metadata=metadata)
        enc = EstimatorResultEncoder()
        assert enc.default(result) == {"values": result.values, "metadata": result.metadata}

    def test_numpy_subclass(self):
        """Test extends NumPyEncoder."""
        enc = EstimatorResultEncoder()
        assert isinstance(enc, NumPyEncoder)
        a = array([0, 1, 2])
        assert enc.default(a) == a.tolist()

    def test_repr_subclass(self):
        """Test extends ReprEncoder."""
        enc = EstimatorResultEncoder()
        assert isinstance(enc, ReprEncoder)
        obj = {"call": "super"}
        assert enc.default(obj) == repr(obj)
