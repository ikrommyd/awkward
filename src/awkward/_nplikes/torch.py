# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpy_like import UfuncLike
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual
from awkward._typing import Final, cast


@register_nplike
class Torch(ArrayModuleNumpyLike):
    is_eager: Final = True
    supports_structured_dtypes: Final = False
    supports_virtual_arrays: Final = True

    def __init__(self):
        import awkward._connect.torch  # noqa: F401

        self._module = ak._connect.torch.import_torch("Awkward Arrays with PyTorch")

    def prepare_ufunc(self, ufunc: UfuncLike) -> UfuncLike:
        from awkward._connect.torch import get_torch_ufunc

        return get_torch_ufunc(ufunc)

    @property
    def ma(self):
        raise ValueError(
            "CUDA arrays cannot have missing values until PyTorch implements "
            "numpy.ma.MaskedArray"
        )

    @property
    def char(self):
        raise ValueError(
            "CUDA arrays cannot do string manipulations until PyTorch implements "
            "numpy.char"
        )

    @property
    def ndarray(self):
        return self._module.Tensor

    def strides(self, x: ArrayLike) -> tuple[int, ...]:
        out: tuple[int, ...] = (x.dtype.itemsize,)
        for item in cast(tuple[int, ...], x.shape[-1:0:-1]):
            out = (item * out[0], *out)
        return out

    @classmethod
    def is_own_array_type(cls, type_: type) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a torch buffer, otherwise `False`.

        """
        module, *_ = type_.__module__.partition(".")
        return module == "torch"

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        if isinstance(x, PlaceholderArray):
            return True
        else:
            (x,) = materialize_if_virtual(x)
            return x.is_c_contiguous()  # type: ignore[attr-defined]

    def ascontiguousarray(self, x: ArrayLike) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            return x
        elif isinstance(x, VirtualArray):
            if x.is_materialized:
                return self.ascontiguousarray(x.materialize())  #  type: ignore[arg-type]
            else:
                return VirtualArray(
                    x._nplike,
                    x._shape,
                    x._dtype,
                    lambda: self.ascontiguousarray(x.materialize()),  # type: ignore[arg-type]
                    lambda: x.shape,
                )
        else:
            return x.contiguous()  # type: ignore[attr-defined]

    def memory_ptr(self, x: ArrayLike) -> int:
        (x,) = materialize_if_virtual(x)
        assert not isinstance(x, PlaceholderArray)
        return x.data_ptr()  # type: ignore[attr-defined]
