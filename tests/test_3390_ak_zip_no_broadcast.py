# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak


def test_ak_zip_no_broadcast_NumpyArray_dict():
    a = ak.Array([1])
    b = ak.Array([2])
    c = ak.zip_no_broadcast({"a": a, "b": b})
    assert ak.to_list(c) == ak.to_list(ak.zip({"a": a, "b": b}))


def test_ak_zip_no_broadcast_ListOffsetArray_dict():
    a = ak.Array([[1], []])
    b = ak.Array([[2], []])
    c = ak.zip_no_broadcast({"a": a, "b": b})
    assert ak.to_list(c) == ak.to_list(ak.zip({"a": a, "b": b}))


def test_ak_zip_no_broadcast_NumpyArray_list():
    a = ak.Array([1])
    b = ak.Array([2])
    c = ak.zip_no_broadcast([a, b])
    assert ak.to_list(c) == ak.to_list(ak.zip([a, b]))


def test_ak_zip_no_broadcast_ListOffsetArray_list():
    a = ak.Array([[1], []])
    b = ak.Array([[2], []])
    c = ak.zip_no_broadcast([a, b])
    assert ak.to_list(c) == ak.to_list(ak.zip([a, b]))
