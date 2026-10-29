"""Regenerate and check all five figures, from any working directory."""

from contextlib import redirect_stdout
import os
from pathlib import Path
import subprocess
import sys

from verify import verify

ROOT = Path(__file__).resolve().parent


def main():
    env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    # Avoid accepting an old PDF if a plotting script stops producing its target.
    for reference in (ROOT / "reference").glob("*.pdf"):
        (ROOT / "figures" / reference.name).unlink(missing_ok=True)
    log_path = output / "reproduction.log"
    print(f"Writing generation log to {log_path}", flush=True)
    with log_path.open("w") as log:
        for script in (
            "plot_from_csv.py",
            "plot_from_csv_5090.py",
            "csv_figures.py",
            "csv_figures_5090.py",
        ):
            print(f"Running {script}", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)], env=env, check=True,
                stdout=log, stderr=subprocess.STDOUT,
            )
        with redirect_stdout(log):
            verify()
    print("PASS: all five figures match their references at 144 dpi (0 changed pixels).")


if __name__ == "__main__":
    main()
