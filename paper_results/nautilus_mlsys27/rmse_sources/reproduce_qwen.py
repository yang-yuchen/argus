"""Offline Qwen KDE replay; uses the archived plot function without GPU imports."""
import argparse
import ast
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/argus_rmse_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=HERE / "reconstructed")
    args = ap.parse_args()
    tree = ast.parse((HERE / "scripts/error.py").read_text())
    # Load the original plot function and colors without importing GPU runners.
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "save_rmse_plot"
    ) or (isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "METHOD_COLORS" for t in node.targets
    ))]
    assert len(selected) == 2, "Archived plotting interface changed"
    namespace = dict(plt=plt, np=np, sns=sns, Path=Path)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "archived error.py", "exec"), namespace)
    for operator, suffix in [("global", "global"), ("causal", "causal_seed1"), ("gqa", "gqa")]:
        source = HERE / "argus_artifacts" / f"qwen_rmse_kde_3curves_1000_{suffix}.json"
        payload = json.loads(source.read_text())
        methods = [(label, item["values"]) for label, item in payload["methods"].items()]
        assert payload["num_runs"] == 1000 and all(len(v) == 1000 for _, v in methods)
        assert payload["operator_name"] == f"prefill_{operator}"
        output = args.output_dir / f"qwen_rmse_kde_3curves_1000_{operator}.pdf"
        namespace["save_rmse_plot"](methods, output, payload["model_id"],
                                    payload["operator_name"], payload.get("plot_xlim"))
        print(output)

if __name__ == "__main__":
    main()
