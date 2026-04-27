# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write(tmp_path, name, table):
    pq.write_table(table, tmp_path / name)


def test_missing_top_level_column(tmp_path):
    _write(
        tmp_path,
        "file1.parquet",
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]}),
    )
    _write(
        tmp_path,
        "file2.parquet",
        pa.table({"a": [7, 8, 9], "b": [10, 11, 12]}),
    )

    arr = ak.from_parquet(tmp_path)

    assert ak.fields(arr) == ["a", "b", "c"]
    assert ak.to_list(arr) == [
        {"a": 1, "b": 4, "c": "x"},
        {"a": 2, "b": 5, "c": "y"},
        {"a": 3, "b": 6, "c": "z"},
        {"a": 7, "b": 10, "c": None},
        {"a": 8, "b": 11, "c": None},
        {"a": 9, "b": 12, "c": None},
    ]


def test_disjoint_columns(tmp_path):
    _write(tmp_path, "file1.parquet", pa.table({"a": [1, 2], "b": [3, 4]}))
    _write(tmp_path, "file2.parquet", pa.table({"c": [5, 6], "d": [7, 8]}))

    arr = ak.from_parquet(tmp_path)

    assert set(ak.fields(arr)) == {"a", "b", "c", "d"}
    out = ak.to_list(arr)
    assert len(out) == 4
    # The first two records have a, b set and c, d as None
    assert out[0]["a"] == 1 and out[0]["b"] == 3
    assert out[0]["c"] is None and out[0]["d"] is None
    # The last two records have c, d set and a, b as None
    assert out[2]["c"] == 5 and out[2]["d"] == 7
    assert out[2]["a"] is None and out[2]["b"] is None


def test_three_files_progressive_columns(tmp_path):
    _write(tmp_path, "file1.parquet", pa.table({"a": [1]}))
    _write(tmp_path, "file2.parquet", pa.table({"a": [2], "b": [20]}))
    _write(tmp_path, "file3.parquet", pa.table({"a": [3], "b": [30], "c": [300]}))

    arr = ak.from_parquet(tmp_path)

    assert set(ak.fields(arr)) == {"a", "b", "c"}
    assert ak.to_list(arr) == [
        {"a": 1, "b": None, "c": None},
        {"a": 2, "b": 20, "c": None},
        {"a": 3, "b": 30, "c": 300},
    ]


def test_columns_argument_with_unified_schema(tmp_path):
    _write(
        tmp_path,
        "file1.parquet",
        pa.table({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]}),
    )
    _write(tmp_path, "file2.parquet", pa.table({"a": [7, 8], "b": [10, 11]}))

    # Selecting only common columns should also work
    arr = ak.from_parquet(tmp_path, columns=["a", "b"])
    assert ak.fields(arr) == ["a", "b"]
    assert ak.to_list(arr) == [
        {"a": 1, "b": 3},
        {"a": 2, "b": 4},
        {"a": 7, "b": 10},
        {"a": 8, "b": 11},
    ]


def test_columns_argument_includes_missing_field(tmp_path):
    _write(
        tmp_path,
        "file1.parquet",
        pa.table({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]}),
    )
    _write(tmp_path, "file2.parquet", pa.table({"a": [7, 8], "b": [10, 11]}))

    # Selecting `c` (missing from file2) fills nulls
    arr = ak.from_parquet(tmp_path, columns=["a", "c"])
    assert set(ak.fields(arr)) == {"a", "c"}
    assert ak.to_list(arr) == [
        {"a": 1, "c": "x"},
        {"a": 2, "c": "y"},
        {"a": 7, "c": None},
        {"a": 8, "c": None},
    ]


def test_nested_columns_argument_is_exact_projection(tmp_path):
    ak.to_parquet(
        ak.Array([{"a": 1, "nested": {"x": 10, "only1": "x"}}]),
        tmp_path / "file1.parquet",
    )
    ak.to_parquet(
        ak.Array([{"a": 2, "nested": {"x": 20, "only2": True}}]),
        tmp_path / "file2.parquet",
    )

    arr = ak.from_parquet(tmp_path, columns=["nested.x"])

    assert ak.to_list(arr) == [
        {"nested": {"x": 10}},
        {"nested": {"x": 20}},
    ]
    assert str(arr.type) == "2 * {nested: ?{x: ?int64}}"


def test_awkward_metadata_top_level_unified_schema(tmp_path):
    ak.to_parquet(
        ak.Array([{"a": 1, "only1": 10}]),
        tmp_path / "file1.parquet",
    )
    ak.to_parquet(
        ak.Array([{"a": 2, "only2": 20}]),
        tmp_path / "file2.parquet",
    )

    arr = ak.from_parquet(tmp_path)

    assert ak.to_list(arr) == [
        {"a": 1, "only1": 10, "only2": None},
        {"a": 2, "only1": None, "only2": 20},
    ]


def test_awkward_metadata_missing_scalar_is_none(tmp_path):
    ak.to_parquet(
        ak.Array([{"a": 1, "b": 5}]),
        tmp_path / "file1.parquet",
    )
    ak.to_parquet(
        ak.Array([{"a": 2}]),
        tmp_path / "file2.parquet",
    )

    arr = ak.from_parquet(tmp_path)

    assert ak.to_list(arr) == [
        {"a": 1, "b": 5},
        {"a": 2, "b": None},
    ]


def test_awkward_metadata_missing_list_is_none(tmp_path):
    ak.to_parquet(
        ak.Array([{"a": 1, "b": [1, 2]}]),
        tmp_path / "file1.parquet",
    )
    ak.to_parquet(
        ak.Array([{"a": 2}]),
        tmp_path / "file2.parquet",
    )

    arr = ak.from_parquet(tmp_path)

    assert ak.to_list(arr) == [
        {"a": 1, "b": [1, 2]},
        {"a": 2, "b": None},
    ]


def test_awkward_metadata_nested_unified_schema(tmp_path):
    ak.to_parquet(
        ak.Array([{"a": 1, "nested": {"x": 10, "only1": "x"}}]),
        tmp_path / "file1.parquet",
    )
    ak.to_parquet(
        ak.Array([{"a": 2, "nested": {"x": 20, "only2": True}}]),
        tmp_path / "file2.parquet",
    )

    arr = ak.from_parquet(tmp_path)

    assert ak.to_list(arr) == [
        {"a": 1, "nested": {"x": 10, "only1": "x", "only2": None}},
        {"a": 2, "nested": {"x": 20, "only1": None, "only2": True}},
    ]


def test_row_groups_with_unified_schema(tmp_path):
    _write(
        tmp_path,
        "file1.parquet",
        pa.table({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]}),
    )
    _write(tmp_path, "file2.parquet", pa.table({"a": [7, 8], "b": [10, 11]}))

    # Combined dataset has 2 row groups (one per file). Pick only the second.
    arr = ak.from_parquet(tmp_path, row_groups=[1])
    assert set(ak.fields(arr)) == {"a", "b", "c"}
    assert ak.to_list(arr) == [
        {"a": 7, "b": 10, "c": None},
        {"a": 8, "b": 11, "c": None},
    ]


def test_metadata_from_parquet_with_unified_schema(tmp_path):
    _write(
        tmp_path,
        "file1.parquet",
        pa.table({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]}),
    )
    _write(tmp_path, "file2.parquet", pa.table({"a": [7, 8], "b": [10, 11]}))

    meta = ak.metadata_from_parquet(tmp_path)
    assert meta["num_rows"] == 4
    # Form should be a record with all unified fields
    assert set(meta["form"].fields) == {"a", "b", "c"}


def test_identical_schemas_unchanged(tmp_path):
    """Ensure the existing fast path for matching schemas still works."""
    _write(tmp_path, "file1.parquet", pa.table({"a": [1, 2], "b": [3, 4]}))
    _write(tmp_path, "file2.parquet", pa.table({"a": [5, 6], "b": [7, 8]}))

    arr = ak.from_parquet(tmp_path)
    assert ak.fields(arr) == ["a", "b"]
    assert ak.to_list(arr) == [
        {"a": 1, "b": 3},
        {"a": 2, "b": 4},
        {"a": 5, "b": 7},
        {"a": 6, "b": 8},
    ]
