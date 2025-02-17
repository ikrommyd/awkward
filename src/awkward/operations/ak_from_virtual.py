# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward.forms.form import index_to_dtype, regularize_buffer_key

__all__ = ("from_virtual",)

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@high_level_function()
def from_virtual(
    form,
    container,
    buffer_key="{form_key}-{attribute}",
    *,
    use_dtypes_from_form=True,
    allow_noncanonical_form=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to reconstitute from named buffers.
        container (Mapping, such as dict): A mapping of string keys to virtual arrays.
            This `container` is only assumed to have a `__getitem__` method that accepts strings as keys.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            as keyword arguments and returns a string to use as a key for a buffer
            in the `container`.
        use_dtypes_from_form (bool): If True, dtypes of virtual arrays will be overwritten
            by the dtypes that are specified by the form.
        allow_noncanonical_form (bool): If True, non-canonical forms will be
            simplified to produce arrays with canonical layouts; otherwise,
            an exception will be thrown for such forms.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Reconstitutes an Awkward Array from a Form and a collection of 1D virtual arrays.

    In contrast to #ak.from_buffers, this function accepts a Mapping of virtual arrays,
    and requires the lengths of the arrays to be correct in the first place.

    The `container` argument lets you specify your own Mapping from form keys to virtual arrays.
    #ak.from_virtual makes sure that all the dtypes align with what's provided in the form.

    The `buffer_key` should match the form keys.

    When `use_dtypes_from_form` is set to True, the dtypes of the virtual arrays will be
    taken from the form. Otherwise, the dtypes of the given virtual arrays will be used.

    When `allow_noncanonical_form` is set to True, this function readily accepts
    non-simplified forms, i.e. forms which will be simplified by Awkward Array
    into "canonical" representations, e.g. `option[option[...]]` → `option[...]`.
    Such forms can be produced by the low-level ArrayBuilder `snapshot()` method.
    Given that Awkward Arrays must have canonical layouts, it follows that
    invoking this function with `allow_noncanonical_form` may produce arrays
    whose forms differ to the input form.

    In order for a non-simplified form to be considered valid, it should be one
    that the #ak.contents.Content layout classes could produce iff. the
    simplification rules were removed.


    See #ak.from_buffers for examples.
    """
    return _impl(
        form,
        container,
        buffer_key,
        use_dtypes_from_form,
        allow_noncanonical_form,
        highlevel,
        behavior,
        attrs,
    )


def _impl(
    form,
    container,
    buffer_key,
    use_dtypes_from_form,
    allow_noncanonical_form,
    highlevel,
    behavior,
    attrs,
):
    # form preparation
    if isinstance(form, str):
        if ak.types.numpytype.is_primitive(form):
            form = ak.forms.NumpyForm(form)
        else:
            form = ak.forms.from_json(form)
    elif isinstance(form, dict):
        form = ak.forms.from_dict(form)

    if not isinstance(form, ak.forms.Form):
        raise TypeError(
            "'form' argument must be a Form or its Python dict/JSON string representation"
        )

    getkey = regularize_buffer_key(buffer_key)

    out = _reconstitute(
        form, container, getkey, use_dtypes_from_form, allow_noncanonical_form
    )
    return wrap_layout(out, highlevel=highlevel, attrs=attrs, behavior=behavior)


def _reconstitute(form, container, getkey, use_dtypes_from_form, simplify):
    if isinstance(form, ak.forms.EmptyForm):
        return ak.contents.EmptyArray()

    elif isinstance(form, ak.forms.NumpyForm):
        dtype = ak.types.numpytype.primitive_to_dtype(form.primitive)
        data = container[getkey(form, "data")]
        if use_dtypes_from_form:
            data._dtype = dtype
        assert dtype == data.dtype
        return ak.contents.NumpyArray(data, parameters=form._parameters)

    elif isinstance(form, ak.forms.UnmaskedForm):
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        if simplify:
            make = ak.contents.UnmaskedArray.simplified
        else:
            make = ak.contents.UnmaskedArray
        return make(content, parameters=form._parameters)

    elif isinstance(form, ak.forms.BitMaskedForm):
        mask = container[getkey(form, "mask")]
        dtype = index_to_dtype[form.mask]
        if use_dtypes_from_form:
            mask._dtype = dtype
        assert dtype == mask.dtype
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        if simplify:
            make = ak.contents.BitMaskedArray.simplified
        else:
            make = ak.contents.BitMaskedArray
        return make(
            ak.index.Index(mask),
            content,
            form.valid_when,
            form.lsb_order,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ByteMaskedForm):
        mask = container[getkey(form, "mask")]
        dtype = index_to_dtype[form.mask]
        if use_dtypes_from_form:
            mask._dtype = dtype
        assert dtype == mask.dtype
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        if simplify:
            make = ak.contents.ByteMaskedArray.simplified
        else:
            make = ak.contents.ByteMaskedArray
        return make(
            ak.index.Index(mask),
            content,
            form.valid_when,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.IndexedOptionForm):
        index = container[getkey(form, "index")]
        dtype = index_to_dtype[form.index]
        if use_dtypes_from_form:
            index._dtype = dtype
        assert dtype == index.dtype
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        if simplify:
            make = ak.contents.IndexedOptionArray.simplified
        else:
            make = ak.contents.IndexedOptionArray
        return make(
            ak.index.Index(index),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.IndexedForm):
        index = container[getkey(form, "index")]
        dtype = index_to_dtype[form.index]
        if use_dtypes_from_form:
            index._dtype = dtype
        assert dtype == index.dtype
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        if simplify:
            make = ak.contents.IndexedArray.simplified
        else:
            make = ak.contents.IndexedArray
        return make(
            ak.index.Index(index),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ListForm):
        # starts
        starts = container[getkey(form, "starts")]
        starts_dtype = index_to_dtype[form.starts]
        if use_dtypes_from_form:
            index._dtype = starts_dtype
        assert starts_dtype == starts.dtype
        # stops
        stops = container[getkey(form, "stops")]
        stops_dtype = index_to_dtype[form.stops]
        if use_dtypes_from_form:
            index._dtype = stops_dtype
        assert stops_dtype == stops.dtype
        assert len(starts) == len(stops)
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        return ak.contents.ListArray(
            ak.index.Index(starts),
            ak.index.Index(stops),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ListOffsetForm):
        offsets = container[getkey(form, "offsets")]
        dtype = index_to_dtype[form.offsets]
        if use_dtypes_from_form:
            offsets._dtype = dtype
        assert dtype == offsets.dtype
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        return ak.contents.ListOffsetArray(
            ak.index.Index(offsets),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.RegularForm):
        content = _reconstitute(
            form.content,
            container,
            getkey,
            use_dtypes_from_form,
            simplify,
        )
        return ak.contents.RegularArray(
            content,
            form.size,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.RecordForm):
        contents = [
            _reconstitute(
                content,
                container,
                getkey,
                use_dtypes_from_form,
                simplify,
            )
            for content, field in zip(form.contents, form.fields)
        ]
        return ak.contents.RecordArray(
            contents,
            None if form.is_tuple else form.fields,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.UnionForm):
        tags = container[getkey(form, "tags")]
        tags_dtype = index_to_dtype[form.tags]
        if use_dtypes_from_form:
            tags._dtype = tags_dtype
        assert tags_dtype == tags.dtype
        index = container[getkey(form, "index")]
        index_dtype = index_to_dtype[form.index]
        if use_dtypes_from_form:
            index._dtype = index_dtype
        assert index_dtype == index.dtype
        contents = [
            _reconstitute(
                content,
                container,
                getkey,
                use_dtypes_from_form,
                simplify,
            )
            for i, content in enumerate(form.contents)
        ]
        if simplify:
            make = ak.contents.UnionArray.simplified
        else:
            make = ak.contents.UnionArray
        return make(
            ak.index.Index(tags),
            ak.index.Index(index),
            contents,
            parameters=form._parameters,
        )

    else:
        raise AssertionError("unexpected form node type: " + str(type(form)))
