"""Verify the benchmark-table renderer highlights wins and draws correctly.

The renderer (benchmarks/render_results_table.py) shades each row's best-score
cell(s): green when YABT is the *sole* winner, yellow on a draw (every tied
model), and nothing for a non-YABT outright win. "Best" is judged on the
*displayed* value (".3f") so visually-equal cells count as a tie. These tests
re-derive that spec from the results JSON and assert the generated LaTeX marks
exactly the expected cells, so the figure can't silently drift from the data.
"""

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

render = pytest.importorskip("render_results_table")

DRAW = r"\cellcolor{draw}"
GREEN = r"\cellcolor{yabtwin}"


def _best_columns(rec):
    """Indices of render.MODELS that tie for the best displayed score in a row
    (empty when the row has no scored model)."""
    scores = [rec["models"].get(m, {}).get(render.DEVICE, {}).get("mean", float("nan"))
              for m in render.MODELS]
    real = [i for i, s in enumerate(scores) if s == s]  # drop NaN
    if not real:
        return []
    top = max(format(scores[i], ".3f") for i in real)
    return [i for i in real if format(scores[i], ".3f") == top]


def _body_lines(tex):
    """The per-dataset rows of the table, in order (between the rules)."""
    body = tex[tex.index(r"\midrule") + len(r"\midrule"):tex.index(r"\bottomrule")]
    return [ln for ln in body.splitlines() if ln.strip().endswith(r"\\")]


@pytest.fixture(scope="module")
def rows():
    data = render.load()
    lines = _body_lines(render.build_tex(data))
    assert len(lines) == len(data), "one table row per dataset"
    # Zip by position, not name: a dataset name (e.g. "pol") can recur across
    # suites, so keying by name would collide.
    return [(_best_columns(rec), line) for rec, line in zip(data.values(), lines)]


def test_draw_rows_shade_every_tied_cell_yellow(rows):
    draws = [(best, line) for best, line in rows if len(best) > 1]
    assert draws, "expected some draws in the committed results"
    for best, line in draws:
        assert line.count(DRAW) == len(best), f"draw must color all tied cells:\n{line}"
        assert GREEN not in line, f"a draw is never green:\n{line}"


def test_outright_yabt_wins_are_green_only(rows):
    for best, line in rows:
        if best == [0]:  # YABT is render.MODELS[0]
            assert line.count(GREEN) == 1 and DRAW not in line, f"sole YABT win is green:\n{line}"


def test_non_yabt_outright_wins_are_unshaded(rows):
    for best, line in rows:
        if len(best) == 1 and best != [0]:
            assert DRAW not in line and GREEN not in line, f"non-YABT win is bold-only:\n{line}"


def test_a_row_is_shaded_iff_yabt_wins_or_ties(rows):
    for best, line in rows:
        shaded = DRAW in line or GREEN in line
        should = len(best) > 1 or best == [0]
        assert shaded == should, f"shading disagrees with the best set {best}:\n{line}"


def test_every_shaded_cell_is_also_bold(rows):
    # Each colored cell bolds its value: \cellcolor{..}\textbf{..}.
    for _, line in rows:
        for tag in (DRAW, GREEN):
            i = line.find(tag)
            while i != -1:
                assert line[i + len(tag):].lstrip().startswith(r"\textbf{"), (
                    f"shaded cell must be bold:\n{line}")
                i = line.find(tag, i + 1)
