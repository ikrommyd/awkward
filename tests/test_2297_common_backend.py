# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import pytest


def test_to_rdataframe():
    pytest.importorskip("ROOT")
