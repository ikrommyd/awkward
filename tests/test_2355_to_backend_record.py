# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np  # noqa: F401

import awkward as ak
from awkward._backends.numpy import NumpyBackend


def test():
    layout = ak.to_layout({"x": 1, "y": 1})
    assert ak.backend(layout) == "cpu"
    assert layout.backend is NumpyBackend.instance()
