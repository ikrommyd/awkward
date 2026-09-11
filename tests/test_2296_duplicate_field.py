# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import re

import awkward as ak


def test_invalid():
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray([1, 2, 3]),
            ak.contents.NumpyArray([1, 2, 3]),
        ],
        ["x", "x"],
    )
    assert re.match(r".*duplicate field 'x'.*", ak.validity_error(layout)) is not None


def test_valid():
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray([1, 2, 3]),
            ak.contents.NumpyArray([1, 2, 3]),
        ],
        ["x", "y"],
    )
    assert re.match(r".*duplicate field 'x'.*", ak.validity_error(layout)) is None
