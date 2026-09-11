# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([1, 2, None])
    result = ak.fill_none(array, 0)
    assert result.to_list() == [1, 2, 0]
