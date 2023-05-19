# (C) Copyright 2023 Pedro Rivero
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grouping tools."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def group_elements(elements: Iterable, group_size: int) -> Iterator[tuple]:
    """Generate groups of elements from iterable as tuples of a given size.

    Args:
        elements: An iterable of elements to be grouped
        group_size: The size of grouped tuples

    Yields:
        The next grouped tuple
    """
    if not isinstance(elements, Iterable):
        raise TypeError("Elements argument must be iterable.")
    if not isinstance(group_size, int) or group_size < 1:
        raise TypeError("Group size argument must be non-zero positive int.")
    multi_iterable = [iter(elements)] * group_size
    yield from zip(*multi_iterable)
