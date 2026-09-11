# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import ctypes
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from awkward._nplikes.array_like import maybe_materialize
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
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


class CTypesKernel(BaseKernel):
    """A kernel compiled into awkward-cpp, called through ctypes.

    The NumPy and JAX backends both dispatch to these functions -- which is why
    the JAX backend requires its buffers to live on the CPU -- so they share the
    calling convention here and differ only in how a buffer's address is taken.
    """

    def __init__(self, impl: Callable[..., Any], key: KernelKeyType):
        super().__init__(impl, key)
        argtypes = impl.argtypes
        self._is_pointer = tuple(issubclass(t, ctypes._Pointer) for t in argtypes)
        # Building a typed ctypes pointer for each buffer costs several times as
        # much as the call itself, so re-prototype the same function address
        # with `void *` parameters and hand it plain addresses instead.
        self._call = ctypes.CFUNCTYPE(
            impl.restype,
            *(
                ctypes.c_void_p if is_pointer else t
                for is_pointer, t in zip(self._is_pointer, argtypes, strict=True)
            ),
        )(ctypes.cast(impl, ctypes.c_void_p).value)

    def _pointer_of(self, x):
        """The address of ``x``'s buffer, rejecting anything this backend cannot pass."""
        raise NotImplementedError

    def __call__(self, *args) -> None:
        assert len(args) == len(self._is_pointer)

        args = maybe_materialize(*args)
        pointer_of = self._pointer_of

        return self._call(
            *(
                pointer_of(x) if is_pointer else x
                for x, is_pointer in zip(args, self._is_pointer, strict=True)
            )
        )


class NumpyKernel(CTypesKernel):
    def _pointer_of(self, x):
        # Do we have a NumPy-owned array?
        if numpy.is_own_array(x):
            assert x.flags.c_contiguous, "kernel expects contiguous array"
            if x.ndim > 0:
                return x.ctypes.data
            else:
                return x
        # Or, do we have a ctypes type
        elif hasattr(x, "_b_base_"):
            return ctypes.cast(x, ctypes.c_void_p)
        else:
            raise AssertionError(
                f"Only NumPy buffers should be passed to Numpy Kernels, received {x}"
            )
