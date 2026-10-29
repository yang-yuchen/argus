"""Regression checks for rejecting changed data and incorrect generated figures."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import verify


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest = json.loads((verify.ROOT / "manifest.json").read_text())
        manifest["inputs"] = manifest["inputs"][:1]
        manifest["figures"] = manifest["figures"][:1]
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        for folder in ("data", "reference", "figures"):
            (self.root / folder).mkdir()
        self.input_path = manifest["inputs"][0]["path"]
        self.figure = manifest["figures"][0]["name"]
        shutil.copyfile(verify.ROOT / self.input_path, self.root / self.input_path)
        shutil.copyfile(
            verify.ROOT / "reference" / self.figure, self.root / "reference" / self.figure
        )

    def test_changed_input_is_rejected(self):
        (self.root / self.input_path).write_text("wrong data\n")
        with patch.object(verify, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "Input checksum changed"):
                verify.verify()

    def test_wrong_figure_is_rejected_and_reported(self):
        shutil.copyfile(
            verify.ROOT / "reference" / "throughput_plot.pdf",
            self.root / "figures" / self.figure,
        )
        with patch.object(verify, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "Generated figures differ"):
                verify.verify()
        report = json.loads((self.root / "figures" / "verification.json").read_text())
        self.assertFalse(report["passed"])
