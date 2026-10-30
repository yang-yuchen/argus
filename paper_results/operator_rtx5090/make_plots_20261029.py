"""Re-plot RTX 5090 decode / causal MTP-Q16 from rtx5090_decode_mtpc16_results.md.

The markdown holds the newest (2026-10-29) nsys medians with the updated
baselines (TileLang main GQA decode, GitHub/official Flex). The older
rtx5090-nsys-*.pdf plots use the earlier, slower baselines.

Reuses load_data / normalize_by_baseline / plot_multi_operator_row from the
repo-root plot_from_csv_5090.py. Run from the repo root:

    python paper_results/operator_rtx5090/make_plots_20261029.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import plot_from_csv_5090 as p  # noqa: E402

SRC = HERE / "rtx5090_decode_mtpc16_results.md"
# markdown section -> (operator, columns after the KV-length column)
SECTIONS = {
    "Decode": ("DC_CAUSAL", ["Ours", "TileLang", "FlexAttn"]),
    "MTP Q16": ("MTP_CAUSAL", ["Ours", "FlexAttn"]),
}


def parse_tables(text: str) -> dict[str, list[list[float]]]:
    tables, current = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            continue
        if current in SECTIONS and re.match(r"\|\s*\d", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            tables.setdefault(current, []).append([float(c) for c in cells])
    return tables


def main() -> None:
    tables = parse_tables(SRC.read_text())
    for section, (operator, cols) in SECTIONS.items():
        rows = tables[section]
        csv_path = HERE / f"rtx5090_20261029_{operator.lower()}_plot_input.csv"
        with csv_path.open("w") as f:
            f.write(",".join(["Operator", "SeqLen", *cols]) + "\n")
            for row in rows:
                f.write(",".join([operator, str(int(row[0])), *map(str, row[1:])]) + "\n")
        df = p.normalize_by_baseline(p.load_data(csv_path), "Ours")
        suffix = "-20261029-decode" if operator == "DC_CAUSAL" else "-20261029-causal-mtp"
        p.plot_multi_operator_row(df, [operator], "Ours", HERE, suffix=suffix)
        generated_pdf = HERE / f"rtx5090{suffix}.pdf"
        if operator == "DC_CAUSAL":
            aliases = (
                HERE / "rtx5090_decode_plot_input.csv",
                HERE / "rtx5090_nsys_decode_plot_input.csv",
                HERE / "rtx5090-decode.pdf",
                HERE / "rtx5090-nsys-decode.pdf",
            )
        else:
            aliases = (
                HERE / "rtx5090_causal-mtp_plot_input.csv",
                HERE / "rtx5090_nsys_causal-mtp_plot_input.csv",
                HERE / "rtx5090-causal-mtp.pdf",
                HERE / "rtx5090-nsys-causal-mtp.pdf",
            )
        for alias in aliases:
            shutil.copy2(csv_path if alias.suffix == ".csv" else generated_pdf, alias)
        print(f"wrote {csv_path.name} and {generated_pdf.name}")


if __name__ == "__main__":
    main()
