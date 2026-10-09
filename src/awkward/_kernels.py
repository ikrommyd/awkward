# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import ctypes
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import awkward as ak
from awkward._nplikes.array_like import maybe_materialize
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.typetracer import try_touch_data
from awkward._typing import Protocol, TypeAlias

# Tuple[str, Unpack[Tuple[metadata.dtype, ...]]]
KernelKeyType: TypeAlias = tuple


numpy = Numpy.instance()
metadata = NumpyMetadata.instance()


class KernelError(Protocol):
    filename: bytes | None
    str: bytes | None
    attempt: int
    id: int


class Kernel(Protocol):
    @property
    @abstractmethod
    def key(self) -> KernelKeyType: ...

    @abstractmethod
    def __call__(self, *args) -> KernelError | None:
        raise NotImplementedError


class BaseKernel(Kernel):
    _impl: Callable[..., Any]
    _key: KernelKeyType

    def __init__(self, impl: Callable[..., Any], key: KernelKeyType):
        self._impl = impl
        self._key = key

    @property
    def key(self) -> KernelKeyType:
        return self._key

    def __repr__(self):
        return "<{} {}{}>".format(
            type(self).__name__,
            self.key[0],
            "".join(", " + str(metadata.dtype(x)) for x in self.key[1:]),
        )


class CTypesFunc(Protocol):
    argtypes: tuple[Any, ...]

    def __call__(self, *args) -> Any: ...


class NumpyKernel(BaseKernel):
    @classmethod
    def _cast(cls, x, t):
        if issubclass(t, ctypes._Pointer):
            # Do we have a NumPy-owned array?
            if numpy.is_own_array(x):
                assert numpy.is_c_contiguous(x), "kernel expects contiguous array"
                if x.ndim > 0:
                    return ctypes.cast(numpy.memory_ptr(x), t)
                else:
                    return x
            # Or, do we have a ctypes type
            elif hasattr(x, "_b_base_"):
                return ctypes.cast(x, t)
            else:
                raise AssertionError(
                    f"Only NumPy buffers should be passed to Numpy Kernels, received {x} (ptr type={type(t).__name__})"
                )
        else:
            return x

    def __call__(self, *args) -> None:
        assert len(args) == len(self._impl.argtypes)

        args = maybe_materialize(*args)

        return self._impl(
            *(self._cast(x, t) for x, t in zip(args, self._impl.argtypes, strict=True))
        )


class TypeTracerKernelError(KernelError):
    def __init__(self):
        self.str = None
        self.filename = None
        self.attempt = ak._util.kSliceNone
        self.id = ak._util.kSliceNone


class TypeTracerKernel:
    def __init__(self, index):
        self._name_and_types = index

    def __call__(self, *args) -> TypeTracerKernelError:
        for arg in args:
            try_touch_data(arg)
        return TypeTracerKernelError()

    def __repr__(self):
        return "<{} {}{}>".format(
            type(self).__name__,
            self._name_and_types[0],
            "".join(", " + str(metadata.dtype(x)) for x in self._name_and_types[1:]),
        )
