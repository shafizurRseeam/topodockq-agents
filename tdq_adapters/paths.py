"""Fixed locations of the authors' code, the downloaded Zenodo data, and our outputs."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The authors' repository XDaiNYU/TopoDockQ (MIT License), shipped unmodified in authors_code/TopoDockQ at
# commit 5696f82. Read-only: never written to. It contains everything this project uses from the authors:
# the model class and scripts, interface extraction, and the feature generator's Python 3.8 bytecode
# (main.pyc, src/*.pyc), which is byte-identical to the separately published wangru25/TopoDockQ-Feature.
AUTHORS_CODE_DIR = PROJECT_ROOT / "authors_code"
AUTHORS_REPO = AUTHORS_CODE_DIR / "TopoDockQ"

# Downloaded Zenodo record 15469415. Read-only: never written to.
DATA_DIR = PROJECT_ROOT / "data"
ZENODO_DIR = DATA_DIR / "zenodo"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
PROCESSED_FILES = (
    "singlePPD_full_bins_features.csv",
    "singlePPD_DockQ.csv",
    "singlePPD_filtered_DockQ.csv",
)
TRAINED_MODEL = DATA_DIR / "trained_model" / "best_model.pth"

# Structure archives from the same Zenodo record, unpacked unchanged (e.g. PFPD_AF2_models/).
EXTERNAL_DIR = DATA_DIR / "external"

# Authors' interface extraction script (run in place, read-only) and the separate
# environment it needs: ProDy 2.4.1 per the authors' src/prody_environment.yml.
AUTHORS_EXTRACT_INTERFACE = AUTHORS_REPO / "src" / "extract_interface.py"


def venv_python(venv: Path) -> Path:
    """Interpreter inside a virtual environment, on Windows or POSIX."""
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


PRODY_VENV = PROJECT_ROOT / ".venv-prody"
PRODY_PYTHON = venv_python(PRODY_VENV)

# Authors' original feature generator (Python 3.8 bytecode) and the environment it needs:
# Python 3.8.18, gudhi 3.8.0, numpy 1.24.3, pandas 2.0.3, per the authors' environment.yaml.
FEATURE38_VENV = PROJECT_ROOT / ".venv-feature38"
FEATURE38_PYTHON = venv_python(FEATURE38_VENV)

# Our outputs.
RUNS_DIR = PROJECT_ROOT / "runs"
PREPROCESSING_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"

# Directories whose contents must never change. Checked before and after runs.
PROTECTED_DIRS = (
    AUTHORS_CODE_DIR,
    ZENODO_DIR,
    PROCESSED_DATA_DIR,
    TRAINED_MODEL.parent,
    EXTERNAL_DIR,
)

# Published reference value from the authors' README for 04_inference_from_generated_csv.py on 4k38.
PUBLISHED_4K38_PDOCKQ = 0.15463054


def to_project_relative(path: Path | str) -> str:
    """Store paths relative to the project root so records survive renaming the project folder."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def from_project_relative(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
