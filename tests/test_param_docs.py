"""Keep the parameter documentation in sync with the code.

The estimators forward **kwargs to BoostParams, and the parameter reference
lives in three places that can drift apart: BoostParams (the source of truth),
the structured _PARAM_GROUPS used to render docstrings, and the README table.
These tests fail if any of them disagree.
"""

import ast
import re
from pathlib import Path

from yabt.boosting import BoostParams
from yabt.sklearn_api import (
    YABTClassifier,
    YABTMultiTaskRegressor,
    YABTRegressor,
    _MULTITASK_PARAMS,
    _PARAM_GROUPS,
)

README = Path(__file__).resolve().parent.parent / "README.md"

# cat_smoothing lives on the estimator (not BoostParams); its default is set in
# _YABTBase.__init__.
ALL_PARAMS = {f.name for f in BoostParams.__dataclass_fields__.values()} | {"cat_smoothing"}
EXTRA_DEFAULTS = {"cat_smoothing": 10.0}


def _flat_groups():
    for group in _PARAM_GROUPS:
        yield from group


def _default_token(sig: str) -> str:
    """'float, default=1e-3' -> '1e-3'; 'str, default=\"auto\"' -> '\"auto\"'."""
    return sig.split("default=")[1]


def test_param_groups_cover_boostparams_exactly():
    names = [name for name, _, _ in _flat_groups()]
    assert len(names) == len(set(names)), "duplicate parameter in _PARAM_GROUPS"
    assert set(names) == ALL_PARAMS


def test_param_group_defaults_match_actual_defaults():
    actual = {f.name: f.default for f in BoostParams.__dataclass_fields__.values()}
    actual.update(EXTRA_DEFAULTS)
    for name, sig, _ in _flat_groups():
        documented = ast.literal_eval(_default_token(sig))
        assert documented == actual[name], (
            f"{name}: docstring default {documented!r} != actual {actual[name]!r}"
        )


def test_multitask_params_are_a_valid_subset():
    assert _MULTITASK_PARAMS <= ALL_PARAMS


def _doc_param_names(cls) -> set[str]:
    return set(re.findall(r"^    ([a-z0-9_]+) :", cls.__doc__, re.M))


def test_full_estimators_document_every_param():
    for cls in (YABTClassifier, YABTRegressor):
        assert _doc_param_names(cls) == ALL_PARAMS


def test_multitask_docstring_lists_only_honored_params():
    assert _doc_param_names(YABTMultiTaskRegressor) == set(_MULTITASK_PARAMS)


def test_readme_table_matches_param_groups():
    text = README.read_text()
    start = text.index("## Parameters")
    table = text[start : text.index("## License", start)]

    rows = set(re.findall(r"^\| `([a-z0-9_]+)` \|", table, re.M))
    assert rows == ALL_PARAMS, "README table param set differs from the code"

    for name, sig, desc in _flat_groups():
        default = _default_token(sig)
        desc_md = " ".join(desc.split("\n")).replace("``", "`")
        expected = f"| `{name}` | `{default}` | {desc_md} |"
        assert expected in table, f"README row out of sync for {name!r}:\n{expected}"
