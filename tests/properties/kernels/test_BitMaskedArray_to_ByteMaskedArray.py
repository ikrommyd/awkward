# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property-based test for the ``awkward_BitMaskedArray_to_ByteMaskedArray`` kernel.

Prototype for replacing the frozen samples in ``kernel-test-data.json`` with
Hypothesis-generated inputs. The property is::

    backend_kernel(inputs) == reference_definition(inputs)   for all valid inputs

The same property runs against every available backend (the compiled CPU
kernel). The kernel's pure-Python reference implementation is taken
straight from ``kernel-specification.yml`` (the single source of truth),
so the test never depends on the generated ``awkward-cpp/tests-spec/kernels.py``.
"""

import ctypes
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

yaml = pytest.importorskip("yaml")

KERNEL = "awkward_BitMaskedArray_to_ByteMaskedArray"
SPEC_PATH = Path(__file__).parents[3] / "kernel-specification.yml"


def _load_reference(name: str):
    """Exec the kernel's reference ``definition`` from the spec and return it.

    Each reference implementation mutates its output argument(s) in place and
    needs only ``uint8`` in scope (mirroring the prelude that
    ``dev/generate-tests.py`` injects into the generated ``kernels.py``).
    """
    import numpy

    spec = yaml.safe_load(SPEC_PATH.read_text())
    definition = next(k["definition"] for k in spec["kernels"] if k["name"] == name)
    namespace: dict = {"numpy": numpy, "uint8": numpy.uint8}
    exec(definition, namespace)
    return namespace[name]


reference = _load_reference(KERNEL)


def _run_cpu(
    frombitmask: list[int], bitmasklength: int, validwhen: bool, lsb_order: bool
) -> list[int]:
    """Run the compiled CPU kernel via ctypes; return ``tobytemask`` as 0/1 ints."""
    from awkward_cpp.cpu_kernels import lib

    n_out = bitmasklength * 8
    tobytemask = (ctypes.c_int8 * n_out)()
    c_frombitmask = (ctypes.c_uint8 * bitmasklength)(*frombitmask)
    ret = getattr(lib, KERNEL)(
        tobytemask, c_frombitmask, bitmasklength, validwhen, lsb_order
    )
    assert not ret.str, ret.str  # error message is falsy on success
    return [int(bool(x)) for x in tobytemask]


def _available_backends():
    """The backends whose compiled kernels are tested."""
    return [pytest.param(_run_cpu, id="cpu")]


@pytest.mark.parametrize("run", _available_backends())
@given(
    frombitmask=st.lists(st.integers(min_value=0, max_value=255), max_size=64),
    validwhen=st.booleans(),
    lsb_order=st.booleans(),
)
def test_matches_reference(
    run, frombitmask: list[int], validwhen: bool, lsb_order: bool
) -> None:
    """Each backend kernel must agree with the spec's reference implementation."""
    bitmasklength = len(frombitmask)

    # Reference implementation: pure-Python, mutating a plain list in place.
    expected: list = [0] * (bitmasklength * 8)
    reference(expected, frombitmask, bitmasklength, validwhen, lsb_order)

    got = run(frombitmask, bitmasklength, validwhen, lsb_order)

    # Reference yields numpy.bool_, the kernels write int8 0/1 — normalise both.
    assert got == [int(bool(x)) for x in expected]
