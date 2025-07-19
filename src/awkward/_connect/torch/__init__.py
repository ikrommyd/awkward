# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

try:
    import torch

    error_message = None

except ModuleNotFoundError:
    torch = None
    error_message = """to use {0}, you must install pytorch:

    pip install torch

or

    conda install -c conda-forge pytorch
"""

from awkward._connect.torch.reducers import get_torch_reducer  # noqa: F401


def get_torch_ufunc(ufunc):
    return getattr(torch, ufunc.__name__, ufunc)


def import_torch(name="Awkward Arrays with PyTorch"):
    if torch is None:
        raise ModuleNotFoundError(error_message.format(name))

    return torch
