# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import pickle

from awkward._nplikes.numpy import Numpy

nplike = Numpy.instance()


def test_pickle_nplike():
    assert pickle.loads(pickle.dumps(nplike)) is nplike
