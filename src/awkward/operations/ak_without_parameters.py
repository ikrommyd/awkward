# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def without_parameters(array, highlevel=True, behavior=None):
    """
    Args:
        array: Data convertible into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function returns a new array without any parameters in its
    #ak.Array.layout, on nodes of any level of depth.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.
    """
    with ak._errors.OperationErrorContext(
        "ak.without_parameters",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)

    out = layout.recursively_apply(
        lambda layout, behavior=behavior, **kwargs: None, keep_parameters=False
    )

    return ak._util.wrap(out, behavior, highlevel)