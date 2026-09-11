# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_backend():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])

    assert ak.backend(array) == "cpu"
