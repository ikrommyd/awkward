# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak


def test():
    conditionals = ak.Array([True, True, True, False, False, False])
    unionarray = ak.Array([1, 2, 3, [4, 5], [], [6]])
    otherarray = ak.Array(range(100, 106))
    result = ak.where(conditionals, unionarray, otherarray)
    assert result.tolist() == [1, 2, 3, 103, 104, 105]
    assert str(result.type) == "6 * union[int64, var * int64]"
