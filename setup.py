"""Embed the source revision in non-editable builds for run provenance."""

import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithRevision(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                    capture_output=True, text=True, check=False)
            revision = result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            revision = None
        destination = Path(self.build_lib) / "gradient_vanishing" / "_build_info.py"
        destination.write_text(f"SOURCE_REVISION = {revision!r}\n", encoding="utf-8")


setup(cmdclass={"build_py": BuildWithRevision})
