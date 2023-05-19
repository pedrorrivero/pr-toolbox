# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for serialization tools."""

from os.path import join as join_path
from tempfile import TemporaryDirectory

from numpy import array
from pytest import mark, raises

from pr_toolbox.serialization import DumpEncoder, NumPyEncoder, ReprEncoder


@mark.parametrize(
    "obj, expected",
    [
        (0, "0"),
        ([0], "[0]"),
        ((0,), "[0]"),
        ([0, 1], "[0, 1]"),
        ((0, 1), "[0, 1]"),
        (["0", "1"], '["0", "1"]'),
        (("0", "1"), '["0", "1"]'),
        ([[], 1], "[[], 1]"),
        (([], 1), "[[], 1]"),
        ([[0], 1], "[[0], 1]"),
        (([0], 1), "[[0], 1]"),
        ({"key": "value"}, '{"key": "value"}'),
        ({0: 1}, '{"0": 1}'),
        ({0: (1,)}, '{"0": [1]}'),
        ({0: [1]}, '{"0": [1]}'),
    ],
)
class TestDumpEncoder:
    """Test DumpEncoder class."""

    def test_dump(self, obj, expected):
        """Test dump to file."""
        with TemporaryDirectory() as tmpdir:
            file_path = join_path(tmpdir, "zne-dump")
            DumpEncoder.dump(obj, file=file_path)
            with open(file_path) as f:
                contents = f.read()
            assert contents == expected

    def test_dumps(self, obj, expected):
        """Test dump to string."""
        assert DumpEncoder.dumps(obj) == expected


class TestReprEncoder:
    """Test ReprEncoder class."""

    @mark.parametrize(
        "repr_str",
        cases := [
            "",
            "dummy",
            "some spaces here",
            "`~!@#$%^&*()-_=+[]\\\"{}|;:',<.>/?",
        ],
        ids=cases,
    )
    def test_default(self, repr_str):
        """Test default method."""

        class DummyRepr:
            def __repr__(self):
                return repr_str

        obj = DummyRepr()
        enc = ReprEncoder()
        assert enc.default(obj) == repr(obj)


class TestNumPyEncoder:
    """Test NumPyEncoder class."""

    @mark.parametrize(
        "array_like",
        cases := [
            [0, 1, 2],
            (0, 1, 2),
        ],
        ids=[type(c) for c in cases],
    )
    def test_default(self, array_like):
        """Test default method."""
        a = array(array_like)
        enc = NumPyEncoder()
        assert enc.default(a) == a.tolist()

    def test_default_type_error(self):
        """Test default raises type error."""
        with raises(TypeError):
            _ = NumPyEncoder().default({"call": "super"})
