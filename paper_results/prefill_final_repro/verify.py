"""Check input/reference hashes and compare generated PDFs with their references."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent


def verify():
    if not shutil.which("pdftoppm"):
        raise RuntimeError("Install Poppler (apt install poppler-utils) to compare PDFs.")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for item in manifest["inputs"]:
        path = ROOT / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Input checksum changed: {path}")

    results = []
    with tempfile.TemporaryDirectory(prefix="nautilus-verify-") as temp:
        for item in manifest["figures"]:
            name = item["name"]
            reference = ROOT / "reference" / name
            if hashlib.sha256(reference.read_bytes()).hexdigest() != item["reference_sha256"]:
                raise ValueError(f"Reference checksum changed: {reference}")
            images = []
            for kind in ("reference", "figures"):
                pdf = ROOT / kind / name
                if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
                    raise ValueError(f"Missing or invalid PDF: {pdf}")
                prefix = Path(temp) / f"{kind}-{name}"
                subprocess.run(
                    ["pdftoppm", "-r", "144", "-png", "-singlefile", str(pdf), str(prefix)],
                    check=True,
                )
                with Image.open(str(prefix) + ".png") as image:
                    images.append(image.convert("RGB"))
            same_size = images[0].size == images[1].size
            changed = None
            if same_size:
                bands = ImageChops.difference(*images).split()
                mask = ImageChops.lighter(ImageChops.lighter(bands[0], bands[1]), bands[2])
                changed = mask.width * mask.height - mask.histogram()[0]
            passed = same_size and changed == 0
            results.append(
                {"figure": name, "passed": passed, "changed_pixels": changed,
                 "reference_size": images[0].size, "generated_size": images[1].size}
            )
            print(f"{'PASS' if passed else 'FAIL'} {name}: changed pixels = {changed}")

    report = {"dpi": 144, "figures": results, "passed": all(r["passed"] for r in results)}
    (ROOT / "figures" / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError("Generated figures differ from references; see figures/verification.json")


if __name__ == "__main__":
    verify()
