# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import re
import ast
import glob
import io
import subprocess
import pathlib
import os
import warnings

import sphinx.ext.napoleon

config = sphinx.ext.napoleon.Config(napoleon_use_param=True, napoleon_use_rtype=True)

reference_path = pathlib.Path("reference")
output_path = reference_path / "generated"
output_path.mkdir(exist_ok=True)

latest_commit = (
    subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

toctree = ["ak.behavior.rst"]


def tostr(node):
    if isinstance(node, ast.NameConstant):
        return repr(node.value)

    elif isinstance(node, ast.Num):
        return repr(node.n)

    elif isinstance(node, ast.Str):
        return repr(node.s)

    elif isinstance(node, ast.Name):
        return node.id

    elif isinstance(node, ast.Index):
        return tostr(node.value)

    elif isinstance(node, ast.Attribute):
        return tostr(node.value) + "." + node.attr

    elif isinstance(node, ast.Subscript):
        return "{0}[{1}]".format(tostr(node.value), tostr(node.slice))

    elif isinstance(node, ast.Slice):
        start = "" if node.lower is None else tostr(node.lower)
        stop = "" if node.upper is None else tostr(node.upper)
        step = "" if node.step is None else tostr(node.step)
        if step == "":
            return "{0}:{1}".format(start, stop)
        else:
            return "{0}:{1}:{2}".format(start, stop, step)

    elif isinstance(node, ast.Call):
        return "{0}({1})".format(
            tostr(node.func), ", ".join(tostr(x) for x in node.args)
        )

    elif isinstance(node, ast.List):
        return "[{0}]".format(", ".join(tostr(x) for x in node.elts))

    elif isinstance(node, ast.Set):
        return "{{{0}}}".format(", ".join(tostr(x) for x in node.elts))

    elif isinstance(node, ast.Tuple):
        return "({0})".format(
            ", ".join(tostr(x) for x in node.elts)
            + ("," if len(node.elts) == 1 else "")
        )

    elif isinstance(node, ast.Dict):
        return "{{{0}}}".format(
            ", ".join(
                "{0}: {1}".format(tostr(x), tostr(y))
                for x, y in zip(node.keys, node.values)
            )
        )

    elif isinstance(node, ast.Lambda):
        return "lambda {0}: {1}".format(
            ", ".join(x.arg for x in node.args.args), tostr(node.body)
        )

    elif isinstance(node, ast.UnaryOp):
        return tostr.op[type(node.op)] + tostr(node.operand)

    elif isinstance(node, ast.BinOp):
        return tostr(node.left) + tostr.op[type(node.op)] + tostr(node.right)

    elif isinstance(node, ast.Compare):
        return tostr(node.left) + "".join(
            tostr.op[type(x)] + tostr(y) for x, y in zip(node.ops, node.comparators)
        )

    elif isinstance(node, ast.BoolOp):
        return tostr.op[type(node.op)].join(tostr(x) for x in node.values)

    else:
        raise Exception(ast.dump(node))


tostr.op = {
    ast.And: " and ",
    ast.Or: " or ",
    ast.Add: " + ",
    ast.Sub: " - ",
    ast.Mult: " * ",
    ast.MatMult: " @ ",
    ast.Div: " / ",
    ast.Mod: " % ",
    ast.Pow: "**",
    ast.LShift: " << ",
    ast.RShift: " >> ",
    ast.BitOr: " | ",
    ast.BitXor: " ^ ",
    ast.BitAnd: " & ",
    ast.FloorDiv: " // ",
    ast.Invert: "~",
    ast.Not: "not ",
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Eq: " == ",
    ast.NotEq: " != ",
    ast.Lt: " < ",
    ast.LtE: " <= ",
    ast.Gt: " > ",
    ast.GtE: " >= ",
    ast.Is: " is ",
    ast.IsNot: " is not ",
    ast.In: " in ",
    ast.NotIn: " not in ",
}


def dosig(node):
    if node is None:
        return "self"
    else:
        argnames = [x.arg for x in node.args.args]
        defaults = ["=" + tostr(x) for x in node.args.defaults]
        defaults = [""] * (len(argnames) - len(defaults)) + defaults
        return ", ".join(x + y for x, y in zip(argnames, defaults))


def dodoc(docstring, qualname, names):
    out = docstring.replace("`", "``")
    out = re.sub(r"<<([^>]*)>>", r"`\1`_", out)
    out = re.sub(r"#(ak\.[A-Za-z0-9_\.]*[A-Za-z0-9_])", r":py:obj:`\1`", out)
    for x in names:
        out = out.replace("#" + x, ":py:meth:`{1} <{0}.{1}>`".format(qualname, x))
    out = re.sub(r"\[([^\]]*)\]\(([^\)]*)\)", r"`\1 <\2>`__", out)
    out = str(sphinx.ext.napoleon.GoogleDocstring(out, config))
    out = re.sub(
        r"([^\. \t].*\n[ \t]*)((\n    .*[^ \t].*)(\n    .*[^ \t].*|\n[ \t]*)*)",
        "\\1\n.. code-block:: python\n\n\\2",
        out,
    )
    out = re.sub(r"(\n:param|^:param)", "\n    :param", out)
    out = re.sub(r"(\n:type|^:type)", "\n    :type", out)
    out = re.sub(r"(\n:returns|^:returns)", "\n    :returns", out)
    out = re.sub(r"(\n:raises|^:raises)", "\n    :raises", out)
    return out


def make_anchor(name):
    name = name.lower()
    parts = name.split(".")
    anchor = "-".join(parts)
    return f"\n\n.. _{anchor}:\n\n"


def doclass(link, linelink, shortname, name, astcls):
    if name.startswith("_"):
        return

    qualname = shortname + "." + name

    init, rest, names = None, [], []
    for node in astcls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init = node
            else:
                rest.append(node)
                names.append(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            rest.append(node)
            names.append(node.targets[0].id)

    outfile = io.StringIO()
    outfile.write(qualname + "\n" + "-" * len(qualname) + "\n\n")
    outfile.write(f".. py:module: {qualname}\n\n")
    outfile.write("Defined in {0}{1}.\n\n".format(link, linelink))
    outfile.write(".. py:class:: {0}({1})\n\n".format(qualname, dosig(init)))

    docstring = ast.get_docstring(astcls)
    if docstring is not None:
        outfile.write(dodoc(docstring, qualname, names) + "\n\n")

    for node in rest:
        if isinstance(node, ast.Assign):
            attrtext = "{0}.{1}".format(qualname, node.targets[0].id)
            outfile.write(make_anchor(attrtext))
            outfile.write(".. py:attribute:: " + attrtext + "\n")
            outfile.write("    :value: {0}\n\n".format(tostr(node.value)))
            docstring = None

        elif any(
            isinstance(x, ast.Name) and x.id == "property" for x in node.decorator_list
        ):
            attrtext = "{0}.{1}".format(qualname, node.name)
            outfile.write(make_anchor(attrtext))
            outfile.write(".. py:attribute:: " + attrtext + "\n\n")
            docstring = ast.get_docstring(node)

        elif any(
            isinstance(x, ast.Attribute) and x.attr == "setter"
            for x in node.decorator_list
        ):
            docstring = None

        else:
            methodname = "{0}.{1}".format(qualname, node.name)
            methodtext = "{0}({1})".format(methodname, dosig(node))
            outfile.write(make_anchor(methodname))
            outfile.write(".. py:method:: " + methodtext + "\n\n")
            docstring = ast.get_docstring(node)

        if docstring is not None:
            outfile.write(dodoc(docstring, qualname, names) + "\n\n")

    toctree.append(os.path.join("generated", qualname + ".rst"))
    out = outfile.getvalue()
    entry_path = reference_path / toctree[-1]
    if not (entry_path.exists() and entry_path.read_text() == out):
        print("writing", toctree[-1])
        entry_path.write_text(out)


def dofunction(link, linelink, shortname, name, astfcn):
    if name.startswith("_"):
        return

    qualname = shortname + "." + name

    outfile = io.StringIO()
    outfile.write(qualname + "\n" + "-" * len(qualname) + "\n\n")
    outfile.write(f".. py:module: {qualname}\n\n")
    outfile.write("Defined in {0}{1}.\n\n".format(link, linelink))

    functiontext = "{0}({1})".format(qualname, dosig(astfcn))
    outfile.write(".. py:function:: " + functiontext + "\n\n")

    docstring = ast.get_docstring(astfcn)
    if docstring is not None:
        outfile.write(dodoc(docstring, qualname, []) + "\n\n")

    out = outfile.getvalue()

    toctree.append(os.path.join("generated", qualname + ".rst"))

    entry_path = reference_path / toctree[-1]
    if not (entry_path.exists() and entry_path.read_text() == out):
        print("writing", toctree[-1])
        entry_path.write_text(out)


done_extra = False
for filename in glob.glob("../src/awkward/**/*.py", recursive=True):

    modulename = (
        filename.replace("../src/", "")
        .replace("/__init__.py", "")
        .replace(".py", "")
        .replace("/", ".")
    )

    shortname = (
        modulename.replace("awkward.", "ak.")
        .replace(".highlevel", "")
        .replace(".behaviors.mixins", "")
        .replace(".behaviors.categorical", "")
        .replace(".behaviors.string", "")
    )
    shortname = re.sub(r"\.operations\.ak_\w+", "", shortname)
    shortname = re.sub(r"\.(contents|types|forms)\.\w+", r".\1", shortname)

    if not done_extra and modulename.startswith("awkward._"):
        done_extra = True
        toctree.extend(
            [
                "ak.numba.register.rst",
                "ak.numexpr.evaluate.rst",
                "ak.numexpr.re_evaluate.rst",
                "awkwardforth.rst",
            ]
        )

    if (
        modulename.startswith("awkward._")
        or modulename == "awkward.nplikes"
        or modulename == "awkward.types._awkward_datashape_parser"
    ):
        continue  # don't show awkward._*, including _v2

    link = "`{0} <https://github.com/scikit-hep/awkward-1.0/blob/" "{1}/{2}>`__".format(
        modulename, latest_commit, filename.replace("../", "")
    )

    module = ast.parse(open(filename).read())

    for toplevel in module.body:
        if hasattr(toplevel, "lineno"):
            linelink = (
                " on `line {0} <https://github.com/scikit-hep/awkward-1.0/blob/"
                "{1}/{2}#L{0}>`__".format(
                    toplevel.lineno, latest_commit, filename.replace("../", "")
                )
            )
        else:
            lineline = ""
        if isinstance(toplevel, ast.ClassDef):
            doclass(link, linelink, shortname, toplevel.name, toplevel)
        if isinstance(toplevel, ast.FunctionDef):
            dofunction(link, linelink, shortname, toplevel.name, toplevel)


categories = [
    ["ak.Array", "ak.Record"],
    ["ak.ArrayBuilder"],
    ["ak.behavior"],
    ["ak.fields", "ak.is_valid", "ak.parameters", "ak.type", "ak.validity_error"],
    [
        "ak.from_arrow",
        "ak.from_iter",
        "ak.from_json",
        "ak.from_numpy",
        "ak.from_parquet",
        "ak.from_cupy",
        "ak.from_jax",
        "ak.from_buffers"
    ],
    [
        "ak.to_arrow",
        "ak.to_dataframe",
        "ak.to_json",
        "ak.to_list",
        "ak.to_numpy",
        "ak.to_parquet",
        "ak.to_cupy",
        "ak.to_jax",
        "ak.to_buffers"
    ],
    ["ak.mask"],
    ["ak.count", "ak.num"],
    ["ak.unzip", "ak.zip"],
    ["ak.with_field", "ak.with_name",
     "ak.with_parameter", "ak.without_parameters"],
    ["ak.broadcast_arrays"],
    ["ak.sort", "ak.argsort"],
    ["ak.ones_like", "ak.zeros_like", "ak.full_like"],
    ["ak.concatenate", "ak.where"],
    ["ak.flatten", "ak.unflatten", "ak.ravel"],
    ["ak.fill_none", "ak.is_none", "ak.pad_none"],
    ["ak.firsts", "ak.singletons"],
    ["ak.argcartesian", "ak.argcombinations", "ak.cartesian", "ak.combinations"],
    ["ak.atleast_1d", "ak.size"],
    [
        "ak.all",
        "ak.any",
        "ak.argmax",
        "ak.argmin",
        "ak.count",
        "ak.count_nonzero",
        "ak.max",
        "ak.min",
        "ak.num",
        "ak.prod",
        "ak.sum",
        "ak.nansum",
        "ak.nanprod",
        "ak.nanmin",
        "ak.nanmax",
        "ak.nanargmax",
        "ak.nanargmin",
    ],
    [
        "ak.corr",
        "ak.covar",
        "ak.linear_fit",
        "ak.mean",
        "ak.moment",
        "ak.softmax",
        "ak.std",
        "ak.ptp",
        "ak.var",
        "ak.nanstd",
        "ak.nanvar",
        "ak.nanmean",
    ],
    ["ak.behaviors.string"],
    ["ak.numba.register_and_check"],
    ["ak.to_dataframe"],
    ["ak.numexpr.evaluate", "ak.numexpr.re_evaluate"],
    ["ak.jax.register_and_check"],
    [
        "ak.contents.BitMaskedArray",
        "ak.contents.ByteMaskedArray",
        "ak.contents.Content",
        "ak.contents.Content",
        "ak.contents.EmptyArray",
        "ak.contents.EmptyArray",
        "ak.contents.IndexedArray",
        "ak.contents.IndexedArray",
        "ak.contents.IndexedOptionArray",
        "ak.contents.ListArray",
        "ak.contents.ListOffsetArray",
        "ak.contents.NumpyArray",
        "ak.contents.NumpyArray",
        "ak.contents.RecordArray",
        "ak.contents.RecordArray",
        "ak.contents.RegularArray",
        "ak.contents.UnionArray",
        "ak.contents.UnionArray",
        "ak.contents.UnmaskedArray",
        "ak.record.Record",
    ],
    ["ak.layout.ArrayBuilder"],
    ["ak.index.Index"],
    [
        "ak.types.ArrayType",
        "ak.types.ArrayType",
        "ak.types.ListType",
        "ak.types.OptionType",
        "ak.types.PrimitiveType",
        "ak.types.RecordType",
        "ak.types.RegularType",
        "ak.types.Type",
        "ak.types.Type",
        "ak.types.UnionType",
        "ak.types.UnknownType",
    ],
    [
        "ak.contents.BitMaskedArray",
        "ak.contents.ByteMaskedArray",
        "ak.contents.Content",
        "ak.contents.Content",
        "ak.contents.EmptyArray",
        "ak.contents.IndexedArray",
        "ak.contents.IndexedOptionArray",
        "ak.contents.ListArray",
        "ak.contents.ListArray",
        "ak.contents.ListOffsetArray",
        "ak.contents.ListOffsetArray",
        "ak.contents.NumpyArray",
        "ak.contents.RecordArray",
        "ak.contents.RegularArray",
        "ak.contents.UnionArray",
        "ak.contents.UnmaskedArray",
        "ak.forms.BitMaskedForm",
        "ak.forms.ByteMaskedForm",
        "ak.forms.EmptyForm",
        "ak.forms.Form",
        "ak.forms.Form",
        "ak.forms.Form",
        "ak.forms.IndexedForm",
        "ak.forms.IndexedOptionForm",
        "ak.forms.ListForm",
        "ak.forms.ListOffsetForm",
        "ak.forms.NumpyForm",
        "ak.forms.RecordForm",
        "ak.forms.RegularForm",
        "ak.forms.UnionForm",
        "ak.forms.UnmaskedForm",
        "ak.types.Type",
        "generated/kernels",
    ],
    ["ak.to_layout"],
]


def get_category_index(path: str) -> tuple[int, str]:
    for i, contents in enumerate(categories):
        for c in contents:
            if c in path:
                return i, path
    warnings.warn(f"couldn't find category for {path}", )
    return len(categories), path


toctree_path = reference_path / "toctree.txt"
toctree_contents = toctree_path.read_text()

for p in toctree:
    if p.removesuffix(".rst") not in toctree_contents:
        warnings.warn(f"Couldn't find {p} in toctree contents")