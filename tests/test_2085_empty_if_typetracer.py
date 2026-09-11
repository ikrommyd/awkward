# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("regulararray", [False, True])
def test_multiplier(regulararray):
    a = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)

    b = ak.from_numpy(a, regulararray=regulararray)
    assert str(b.type) == "2 * 3 * 5 * int64"

    c = ak.Array(b.layout.form.length_one_array())
    assert str(c.type) == "1 * 3 * 5 * int64"
    assert c.tolist() == [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
