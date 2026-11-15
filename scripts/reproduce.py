#!/usr/bin/env python3
"""
End-to-end deterministic reproducibility runner for Paper 21
(`tau-core-black-hole-cosmology-duality-paper21`):
1. Runs SymPy & numerical audit (`scripts/compute_bh_cosmology_duality_audit.py`).
2. Generates all 7 EN + 7 HU publication-grade figures (`scripts/generate_paper21_figures.py`).
3. Compiles `paper21_submission_source/main.tex` -> `paper21_submission_source/main.pdf`.
4. Compiles `paper21_hungarian_source/main.tex` -> `paper21_hungarian_source/main.pdf`.
5. Packages the clean source-only `arxiv_submission_source.zip`.
6. Executes the automated verification suite (`pytest -q`).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
TINYTEX_BIN = Path("/Users/jolcsak/Library/TinyTeX/bin/universal-darwin")


def get_env_with_latex() -> dict[str, str]:
    env = os.environ.copy()
    if TINYTEX_BIN.exists():
        env["PATH"] = f"{TINYTEX_BIN}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[reproduce] Running: {' '.join(cmd)} (in {cwd})")
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def compile_latex_folder(folder: Path, env: dict[str, str]) -> None:
    pdflatex = shutil.which("pdflatex", path=env.get("PATH"))
    bibtex = shutil.which("bibtex", path=env.get("PATH"))
    if not pdflatex or not bibtex:
        raise RuntimeError("pdflatex or bibtex not found in PATH or TinyTeX directory.")

    run_step([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=folder, env=env)
    run_step([bibtex, "main"], cwd=folder, env=env)
    run_step([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=folder, env=env)
    run_step([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=folder, env=env)


def main() -> None:
    env = get_env_with_latex()

    # 1. Run SymPy duality & observational audit
    run_step([sys.executable, str(REPO_ROOT / "scripts" / "compute_bh_cosmology_duality_audit.py")], cwd=REPO_ROOT, env=env)

    # 2. Generate all 7 EN + 7 HU figures (300 DPI)
    run_step([sys.executable, str(REPO_ROOT / "scripts" / "generate_paper21_figures.py")], cwd=REPO_ROOT, env=env)

    # 3. Compile English submission manuscript
    compile_latex_folder(REPO_ROOT / "paper21_submission_source", env=env)

    # 4. Compile Hungarian companion manuscript
    compile_latex_folder(REPO_ROOT / "paper21_hungarian_source", env=env)

    # 5. Build clean arXiv source zip
    run_step([sys.executable, str(REPO_ROOT / "scripts" / "build_arxiv_source.py")], cwd=REPO_ROOT, env=env)

    # 6. Run pytest verification suite
    run_step([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT, env=env)

    print("PAPER21_BH_COSMOLOGY_DUALITY_REPRODUCTION_COMPLETE")


if __name__ == "__main__":
    main()
