# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for type checking tools."""

from pytest import mark

from pr_toolbox.typing import isinteger, isreal


################################################################################
## TYPE CHECKING
################################################################################
class TestIsInteger:
    """Test is integer value."""

    @mark.parametrize("object", [0, 1, -1, 1.0, -1.0, True, False])
    def test_isinteger_true(self, object):
        """Test true."""
        assert isinteger(object)

    @mark.parametrize("object", [1.2, -2.4])
    def test_isinteger_false(self, object):
        """Test false."""
        assert not isinteger(object)


class TestIsReal:
    """Test is real value."""

    @mark.parametrize("object", [0, 1, -1, 1.2, -2.4, True, False])
    def test_isreal_true(self, object):
        """Test true."""
        assert isreal(object)

    @mark.parametrize("object", [float("nan"), float("inf"), float("-inf"), {}, (0, 1)])
    def test_isreal_false(self, object):
        """Test false."""
        assert not isreal(object)
