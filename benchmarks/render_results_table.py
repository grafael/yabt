#!/usr/bin/env python3
"""Render benchmarks/openml_regression.png from openml_benchmark_results.json.

Produces a publication-quality table image of the OpenML numeric-regression
suite: per dataset and per model the test R^2 (higher is better) and wall-clock
fit time t in seconds (lower is better). The best R^2 and the fastest t in each
row are bolded.

The table is typeset with LaTeX (booktabs) and compiled to PDF with Tectonic,
then rasterized to PNG with pdftocairo -- giving a clean research aesthetic
rather than a matplotlib bitmap.

Requires: tectonic and pdftocairo (poppler) on PATH.

Usage:
    python benchmarks/render_results_table.py
"""
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "openml_benchmark_results.json")
PNG_PATH = os.path.join(HERE, "openml_regression.png")

MODELS = ["YABT", "XGBoost", "LightGBM", "CatBoost"]
DEVICE = "cpu"
DPI = 300


def load():
    with open(JSON_PATH) as f:
        return json.load(f)


def tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def num(x, fmt, bold):
    if x != x:  # NaN
        return "--"
    s = format(x, fmt)
    return r"\textbf{" + s + "}" if bold else s


def build_tex(data):
    datasets = list(data.keys())
    n_models = len(MODELS)

    # l for dataset, r r for (n, p), then a pair of r-cols per model.
    colspec = "l r r " + " ".join(["r r"] * n_models)

    # Spanning model-name header row.
    model_heads = " & ".join(
        r"\multicolumn{2}{c}{%s}" % m for m in MODELS
    )
    # cmidrule under each model span; columns 4-5, 6-7, ...
    cmids = []
    start = 4
    for _ in MODELS:
        cmids.append(r"\cmidrule(lr){%d-%d}" % (start, start + 1))
        start += 2
    cmid_line = "".join(cmids)

    sub_heads = " & ".join([r"$R^2$ & $t$"] * n_models)
    n_cols = 3 + 2 * n_models

    body = []
    for key in datasets:
        name = tex_escape(key.split(":")[-1])
        rec = data[key]
        models = rec["models"]
        r2s, times = [], []
        for m in MODELS:
            md = models.get(m, {}).get(DEVICE, {})
            r2s.append(md.get("mean", float("nan")))
            times.append(md.get("time", float("nan")))

        best_r2 = max(range(n_models), key=lambda i: (r2s[i] != r2s[i], r2s[i]))
        best_t = min(range(n_models), key=lambda i: (times[i] != times[i], times[i]))

        cells = [name, str(rec.get("n", "")), str(rec.get("p", ""))]
        for i in range(n_models):
            cells.append(num(r2s[i], ".3f", i == best_r2))
            cells.append(num(times[i], ".2f", i == best_t))
        body.append(" & ".join(cells) + r" \\")

    title1 = r"\textbf{\large OpenML numeric-regression benchmark suite}"
    title2 = (
        r"\small Test $R^2$ (higher is better) and wall-clock fit time "
        r"$t$ in seconds (lower is better), \textsc{cpu}"
    )
    foot = (
        r"\footnotesize\color{rulegray} Best $R^2$ and fastest $t$ per "
        r"dataset in \textbf{bold}.\quad $n$: samples,\; $p$: features."
    )

    tex = r"""\documentclass[border=14pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\definecolor{rulegray}{gray}{0.45}
\begin{document}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{%(colspec)s}
\multicolumn{%(ncols)d}{c}{%(title1)s} \\[2pt]
\multicolumn{%(ncols)d}{c}{%(title2)s} \\[8pt]
\toprule
& & & %(model_heads)s \\
%(cmid_line)s
Dataset & $n$ & $p$ & %(sub_heads)s \\
\midrule
%(body)s
\bottomrule
\addlinespace[4pt]
\multicolumn{%(ncols)d}{c}{%(foot)s} \\
\end{tabular}
\end{document}
""" % {
        "colspec": colspec,
        "ncols": n_cols,
        "title1": title1,
        "title2": title2,
        "model_heads": model_heads,
        "cmid_line": cmid_line,
        "sub_heads": sub_heads,
        "body": "\n".join(body),
        "foot": foot,
    }
    return tex


def main():
    if not shutil.which("tectonic"):
        raise SystemExit("tectonic not found on PATH")
    if not shutil.which("pdftocairo"):
        raise SystemExit("pdftocairo (poppler) not found on PATH")

    data = load()
    tex = build_tex(data)

    with tempfile.TemporaryDirectory() as td:
        tex_path = os.path.join(td, "table.tex")
        pdf_path = os.path.join(td, "table.pdf")
        with open(tex_path, "w") as f:
            f.write(tex)

        subprocess.run(
            ["tectonic", "--outdir", td, tex_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # pdftocairo appends nothing for single-page when -singlefile is set.
        out_prefix = os.path.splitext(PNG_PATH)[0]
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", str(DPI),
             pdf_path, out_prefix],
            check=True,
        )

    n_rows = len(data)
    print(f"wrote {PNG_PATH}  ({n_rows} datasets x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
