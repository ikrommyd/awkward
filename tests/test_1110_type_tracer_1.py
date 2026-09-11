# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak


def test_getitem_at():
    concrete = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5) * 0.1)

    assert concrete.shape == (2, 3, 5)


def test_UnmaskedArray_NumpyArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    assert len(a) == 4
    assert a[2] == 2.2
    assert a[-2] == 2.2
    assert type(a[2]) is np.float64
    with pytest.raises(IndexError):
        a[4]
    with pytest.raises(IndexError):
        a[-5]
    assert isinstance(a[2:], ak.contents.unmaskedarray.UnmaskedArray)
    assert a[2:][0] == 2.2
    assert len(a[2:]) == 2
    with pytest.raises(IndexError):
        a["bad"]


def test_RegularArray_RecordArray_NumpyArray():
    b = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    assert len(b["nest"]) == 10
    assert isinstance(b["nest"][5], ak.contents.emptyarray.EmptyArray)
    assert len(b["nest"][5]) == 0
    assert isinstance(b["nest"][7:], ak.contents.regulararray.RegularArray)
    assert len(b["nest"][7:]) == 3
    assert len(b["nest"][7:100]) == 3
    with pytest.raises(IndexError):
        b["nest"]["bad"]
