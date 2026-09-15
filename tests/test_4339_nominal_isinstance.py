# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.numpy import NumpyBackend
from awkward._nplikes.array_like import ArrayLike, MaterializableArray
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray
from awkward._typing import NominalMeta

# Every class that opts into the nominal metaclass.
NOMINAL_CLASSES = [
    Backend,
    MaterializableArray,
    ArrayModuleNumpyLike,
]


@pytest.mark.parametrize("cls", NOMINAL_CLASSES, ids=lambda cls: cls.__name__)
def test_uses_nominal_metaclass(cls):
    assert type(cls) is NominalMeta


def test_real_implementations_are_still_instances():
    # nplikes
    assert isinstance(Numpy.instance(), ArrayModuleNumpyLike)
    assert issubclass(Numpy, ArrayModuleNumpyLike)

    # backends
    assert isinstance(NumpyBackend.instance(), Backend)

    # arrays
    virtual = VirtualNDArray(
        Numpy.instance(), (3,), np.dtype(np.int64), lambda: np.arange(3)
    )
    assert isinstance(virtual, MaterializableArray)

    # the ArrayLike protocol is still part of the concrete classes' interface
    assert ArrayLike in MaterializableArray.__mro__


@pytest.mark.parametrize("cls", NOMINAL_CLASSES, ids=lambda cls: cls.__name__)
def test_duck_types_are_not_instances(cls):
    """A structurally compatible duck type is *not* an instance.

    This is the behaviour change: these classes are checked nominally, so
    neither structural compatibility nor an explicit ``abc`` ``register()``
    makes an unrelated class an instance. Only real subclasses count.
    """

    class Duck:
        known_data = True
        dtype = np.dtype(np.int64)
        ndim = 1
        shape = (3,)
        strides = (8,)
        name = "duck"

        def materialize(self):
            return np.arange(3)

    assert not isinstance(Duck(), cls)
    cls.register(Duck)
    assert not isinstance(Duck(), cls)
    assert not issubclass(Duck, cls)


def test_explicit_subclasses_are_still_instances():
    class MyVirtual(VirtualNDArray):
        pass

    assert isinstance(
        MyVirtual(Numpy.instance(), (3,), np.dtype(np.int64), lambda: np.arange(3)),
        MaterializableArray,
    )

    class MyBackend(NumpyBackend):
        pass

    assert issubclass(MyBackend, Backend)
    assert isinstance(MyBackend.instance(), Backend)

    class MyNumpy(Numpy):
        pass

    assert issubclass(MyNumpy, ArrayModuleNumpyLike)


def test_operations_are_unaffected():
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.num(array).to_list() == [3, 0, 2]
    assert (array * 2).to_list() == [[2, 4, 6], [], [8, 10]]
    assert array[1:].to_list() == [[], [4, 5]]
    assert ak.sum(array, axis=-1).to_list() == [6, 0, 9]
