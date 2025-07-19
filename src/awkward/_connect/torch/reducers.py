# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._reducers import Reducer
from awkward._typing import Self, TypeVar

np = NumpyMetadata.instance()


_overloads: dict[type[Reducer], type[TorchReducer]] = {}


R = TypeVar("R", bound=Reducer)


def overloads(cls: type[Reducer]):
    def registrar(new_cls: type[R]) -> type[R]:
        _overloads[cls] = new_cls
        return new_cls

    return registrar


class TorchReducer(Reducer):
    @classmethod
    @abstractmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        raise NotImplementedError


def get_torch_reducer(reducer: Reducer) -> Reducer:
    return _overloads[type(reducer)].from_kernel_reducer(reducer)
