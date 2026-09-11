# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np

import awkward as ak

form = ak.forms.from_dict(
    {
        "class": "RecordArray",
        "fields": ["x"],
        "contents": [
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": "x.list.content",
                },
                "parameters": {},
                "form_key": "x.list",
            }
        ],
        "parameters": {},
    }
)


def test_buffer_keys_on_virtual_arrays():
    buffers = {
        "x.list-offsets": lambda: np.array([0, 2, 3, 5]),
        "x.list.content-data": lambda: np.array([1, 2, 3, 4, 5]),
    }

    va = ak.from_buffers(form, 3, buffers, buffer_key="{form_key}-{attribute}")

    assert va.layout.content("x").offsets.data.buffer_key == "x.list-offsets"
    assert va.layout.content("x").content.data.buffer_key == "x.list.content-data"
