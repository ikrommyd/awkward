# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import reduce
from operator import mul

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._operators import NDArrayOperatorsMixin
from awkward._typing import TYPE_CHECKING, Any, Callable, DType, Self
from awkward._util import Sentinel

np = NumpyMetadata.instance()

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


UNMATERIALIZED = Sentinel("<UNMATERIALIZED>", None)


def materialize_if_virtual(*args: Any) -> tuple[Any, ...]:
    """
    A little helper function to materialize all virtual arrays in a list of arrays.
    """
    return tuple(
        arg.materialize() if isinstance(arg, VirtualArray) else arg for arg in args
    )


class VirtualArray(NDArrayOperatorsMixin, ArrayLike):
    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DType,
        generator: Callable[[], ArrayLike],
    ) -> None:
        if not isinstance(nplike, (ak._nplikes.numpy.Numpy, ak._nplikes.cupy.Cupy)):
            raise TypeError(
                f"Only numpy and cupy nplikes are supported for {type(self).__name__}. Received {type(nplike)}"
            )
        if any(not isinstance(item, int) for item in shape):
            raise TypeError(
                f"{type(self).__name__} supports only shapes of integer dimensions. Received shape {shape}."
            )

        # array metadata
        self._nplike = nplike
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._array: Sentinel | ArrayLike = UNMATERIALIZED
        self._generator = generator

    def tobytes(self, order="C") -> bytes:
        return self.materialize().tobytes(order)

    def tostring(self, order="C") -> bytes:
        return self.materialize().tostring(order)

    @property
    def real(self):
        return self.materialize().real

    @property
    def imag(self):
        return self.materialize().imag

    def max(self, axis=None, out=None, keepdims=False):
        return self.materialize().max(axis, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return self.materialize().min(axis, out, keepdims)

    def argsort(self, axis=-1, kind=None, order=None, *, stable=None):
        return self.materialize().argsort(axis, kind, order, stable=stable)

    def byteswap(self, inplace=False):
        return self.materialize().byteswap(inplace)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> ShapeItem:
        return reduce(mul, self._shape)

    @property
    def nbytes(self) -> ShapeItem:
        if self.is_materialized:
            return self._array.nbytes  # type: ignore[union-attr]
        return 0

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        return self.materialize().strides  # type: ignore[attr-defined]

    def materialize(self) -> ArrayLike:
        if self._array is UNMATERIALIZED:
            array = self._nplike.asarray(self.generator())
            assert self._shape == array.shape, (
                f"{type(self).__name__} had shape {self._shape} before materialization while the materialized array has shape {array.shape}"
            )
            assert self._dtype == array.dtype, (
                f"{type(self).__name__} had dtype {self._dtype} before materialization while the materialized array has dtype {array.dtype}"
            )
            self._array = array
        return self._array  # type: ignore[return-value]

    @property
    def is_materialized(self) -> bool:
        return self._array is not UNMATERIALIZED

    @property
    def T(self):
        if self.is_materialized:
            return self._array.T

        return type(self)(
            self._nplike,
            self._shape[::-1],
            self._dtype,
            lambda: self.materialize().T,
        )

    def view(self, dtype: DTypeLike) -> Self:
        dtype = np.dtype(dtype)

        if self.is_materialized:
            return self.materialize().view(dtype)  # type: ignore[return-value]

        if len(self._shape) >= 1:
            last, remainder = divmod(
                self._shape[-1] * self._dtype.itemsize, dtype.itemsize
            )
            if remainder != 0:
                raise ValueError(
                    "new size of array with larger dtype must be a "
                    "divisor of the total size in bytes (of the last axis of the array)"
                )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        return type(self)(
            self._nplike,
            shape,
            dtype,
            lambda: self.materialize().view(dtype),
        )

    @property
    def generator(self) -> Callable:
        return self._generator

    @property
    def nplike(self) -> NumpyLike:
        if not isinstance(
            self._nplike, (ak._nplikes.numpy.Numpy, ak._nplikes.cupy.Cupy)
        ):
            raise TypeError(
                f"Only numpy and cupy nplikes are supported for {type(self).__name__}. Received {type(self._nplike)}"
            )
        return self._nplike

    def copy(self) -> VirtualArray:
        new_virtual = type(self)(
            self._nplike,
            self._shape,
            self._dtype,
            lambda: self.materialize().copy(),  # type: ignore[attr-defined]
        )
        new_virtual.materialize()
        return new_virtual

    def tolist(self) -> list:
        return self.materialize().tolist()

    @property
    def ctypes(self):
        if isinstance((self._nplike), ak._nplikes.cupy.Cupy):
            raise AttributeError("Cupy ndarrays do not have a ctypes attribute.")
        return self.materialize().ctypes

    @property
    def data(self):
        return self.materialize().data

    def __array__(self, *args, **kwargs):
        return self.materialize()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.nplike.apply_ufunc(ufunc, method, inputs, kwargs)

    def __repr__(self):
        dtype = repr(self._dtype)
        if self.shape is None:
            shape = ""
        else:
            shape = ", shape=" + repr(self._shape)
        return f"VirtualArray(array={self._array}, {dtype}{shape})"

    def __str__(self):
        if self.ndim == 0:
            return "??"
        else:
            return repr(self)

    def __getitem__(self, index):
        if self.is_materialized:
            return self._array.__getitem__(index)

        if isinstance(index, slice):
            length = self._shape[0]

            if (
                index.start is unknown_length
                or index.stop is unknown_length
                or index.step is unknown_length
            ):
                raise TypeError(
                    f"{type(self).__name__} does not support slicing with unknown_length while slice {index} was provided."
                )
            else:
                start, stop, step = index.indices(length)
                new_length = (stop - start) // step

            return type(self)(
                self._nplike,
                (new_length,),
                self._dtype,
                lambda: self.materialize()[index],
            )
        else:
            return self.materialize().__getitem__(index)

    def __setitem__(self, key, value):
        array = self.materialize()
        array.__setitem__(key, value)

    def __bool__(self) -> bool:
        array = self.materialize()
        return bool(array)

    def __int__(self) -> int:
        array = self.materialize()
        if array.ndim == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be converted to an int.")

    def __index__(self) -> int:
        array = self.materialize()
        if array.ndim == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be used as an index.")

    def __len__(self) -> int:
        return int(self._shape[0])

    def __iter__(self):
        array = self.materialize()
        return iter(array)

    # TODO: The following can be implemented, but they will need materialization.
    # Also older numpy versions don't support them.
    # One needs them to use from_dlpack() on a virtual array.
    def __dlpack_device__(self) -> tuple[int, int]:
        raise RuntimeError("cannot realise an unknown value")

    def __dlpack__(self, stream: Any = None) -> Any:
        raise RuntimeError("cannot realise an unknown value")
