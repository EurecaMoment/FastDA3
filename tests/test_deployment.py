"""Exercise the real Bash wrappers with an isolated checkout and no GPU model."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("bash"), "Linux Bash required"
)
class BashDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fastda3 path with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.checkout = Path(self.temporary.name)
        shutil.copytree(ROOT / "scripts", self.checkout / "scripts")
        self.python = self.checkout / "fake python"
        self.python.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$FASTDA3_ROOT" "$PWD" "$CUDA_VISIBLE_DEVICES" '
            '"$DA3_MODEL_DIR" "$DA3_API_KEY" "$@"\n'
        )
        self.python.chmod(0o755)
        self.environment = {
            "PATH": os.environ["PATH"],
            "PYTHON": str(self.python),
            "HOME": self.temporary.name,
        }
        (self.checkout / ".env").write_text(
            "DA3_API_KEY=test-file-secret\n"
            'DA3_MODEL_DIR="./weights with spaces"\n'
            "CUDA_VISIBLE_DEVICES=7\n"
            "PYTHON=/incorrect/python\n"
        )

    def run_script(self, *arguments, check=True):
        return subprocess.run(
            ["bash", str(self.checkout / "scripts/serve.sh"), *arguments],
            cwd="/",
            env=self.environment,
            text=True,
            capture_output=True,
            check=check,
        )

    def test_foreign_working_directory_spaces_and_argument_forwarding(self):
        lines = self.run_script(
            "--port", "8123", "--model-dir", "/model with spaces"
        ).stdout.splitlines()
        self.assertEqual(
            lines[:5],
            [
                str(self.checkout),
                str(self.checkout),
                "7",
                "./weights with spaces",
                "test-file-secret",
            ],
        )
        self.assertEqual(
            lines[5:],
            [
                "-u",
                "-m",
                "fastda3.server",
                "--port",
                "8123",
                "--model-dir",
                "/model with spaces",
            ],
        )

    def test_exported_values_override_configuration(self):
        self.environment.update(
            CUDA_VISIBLE_DEVICES="3", DA3_API_KEY="test-caller-secret"
        )
        lines = self.run_script().stdout.splitlines()
        self.assertEqual(lines[2], "3")
        self.assertEqual(lines[4], "test-caller-secret")

    def test_alternate_environment_file_and_missing_file(self):
        alternate = self.checkout / "alternate config.env"
        alternate.write_text("DA3_API_KEY=test-alt-secret\nDA3_MODEL_DIR=./alt\n")
        self.environment["FASTDA3_ENV_FILE"] = str(alternate)
        lines = self.run_script().stdout.splitlines()
        self.assertEqual(lines[2:5], ["0", "./alt", "test-alt-secret"])
        alternate.unlink()
        result = self.run_script(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Configuration file does not exist", result.stderr)


class RelativePathTests(unittest.TestCase):
    def test_paths_are_relative_to_checkout_instead_of_process_directory(self):
        from fastda3.server import arguments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(
                os.environ,
                {
                    "FASTDA3_ROOT": str(root),
                    "DA3_MODEL_DIR": "models/from-environment",
                    "DA3_DATA_DIR": "data with spaces",
                },
            ):
                args = arguments(["--model-dir", "models/from-cli"])
                self.assertEqual(args.model_dir, str(root / "models/from-cli"))
                self.assertEqual(args.files_dir, str(root / "data with spaces/files"))
                self.assertEqual(
                    arguments([]).model_dir, str(root / "models/from-environment")
                )


if __name__ == "__main__":
    unittest.main()
