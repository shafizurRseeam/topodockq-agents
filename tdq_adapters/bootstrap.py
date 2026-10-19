"""One-time setup after ``uv sync``: download the published model and data, build the side environments.

    uv run topodockq-setup                  core: trained model, two side environments, self-test
    uv run topodockq-setup --examples       + the PFPD AlphaFold models (233 MB download) for demos and tests
    uv run topodockq-setup --reproduction   + everything needed to rerun the paper reproduction (Steps 1-4)
    uv run topodockq-setup --plan           show what would be done, change nothing

The authors' code (XDaiNYU/TopoDockQ, MIT License) ships with this repository in
authors_code/TopoDockQ; setup only verifies that every file is identical to the published
commit. The model and data come from Zenodo record 15469415 (CC BY 4.0) and are checked
against Zenodo's MD5; the checkpoint is also checked against the hash recorded in the
preprocessing package. Re-running skips everything already in place; large downloads resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
# tempfile is used by the self-test; zipfile and tarfile by the Zenodo extraction.
from pathlib import Path

from .paths import (AUTHORS_CODE_DIR, DATA_DIR, EXTERNAL_DIR, FEATURE38_PYTHON, FEATURE38_VENV, PREPROCESSING_DIR,
                    PRODY_PYTHON, PRODY_VENV, TRAINED_MODEL, ZENODO_DIR)

RESOURCES = Path(__file__).with_name("resources")
ZENODO_RECORD = "15469415"
ZENODO_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"

# Zenodo files: published MD5 and size; group decides when they are downloaded. Unpacked files are checked
# against the archive's own listing (extraction_problems), not against a single marker file.
ZENODO_FILES = {
    "trained_model.zip": {"md5": "4ab197f0cd54108b5931064b54f54030", "bytes": 66_919_596, "group": "core",
                          "extract": "zip", "into": DATA_DIR},
    "PFPD_AF2_models.tar.gz": {"md5": "31f617e3c2bd0e28e91646d7efde3156", "bytes": 233_173_167, "group": "examples",
                               "extract": "tar", "into": EXTERNAL_DIR},
    "processed_data.zip": {"md5": "7be6f8edd84d8af8242152bb47c45718", "bytes": 661_674_553, "group": "reproduction",
                           "extract": "zip", "into": DATA_DIR},
    "singlePPD_interface_files.tar.gz": {"md5": "1c3e6e13fc2287b637622231b78030cf", "bytes": 3_991_895_076,
                                         "group": "reproduction", "extract": None, "into": None},
}

# Side environments for the authors' tools, with the pins that were verified to work.
SIDE_ENVS = {
    "prody": {"venv": PRODY_VENV, "python_exe": PRODY_PYTHON, "python": "3.9",
              "packages": ["prody==2.4.1", "numpy<1.24", "scipy==1.10.1", "pyparsing==3.0.9", "setuptools<81"],
              "check": "import prody; print(prody.__version__)",
              "why": "runs the authors' extract_interface.py (ProDy 2.4.1, per their prody_environment.yml)"},
    "feature38": {"venv": FEATURE38_VENV, "python_exe": FEATURE38_PYTHON, "python": "3.8.18",
                  "packages": ["gudhi==3.8.0", "numpy<=1.24.3", "pandas<=2.2.0", "scikit-learn==1.3.0"],
                  "check": "import gudhi; print(gudhi.__version__)",
                  "why": "runs the authors' original feature code (Python 3.8 bytecode, per their environment.yaml)"},
}


class SetupError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def _make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")] if root.is_dir() else [root]:
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass


def download(url: str, dest: Path, expected_bytes: int | None = None, retries: int = 3) -> Path:
    """Stream ``url`` to ``dest`` via a .part file, resuming an interrupted download when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "topodockq-agents-setup"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                done = have if mode == "ab" else 0
                total = expected_bytes or (done + int(resp.headers.get("Content-Length") or 0))
                last = time.time()
                with open(part, mode) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2:
                            pct = f" ({100 * done / total:.0f}%)" if total else ""
                            print(f"    {dest.name}: {_human(done)}{pct}", flush=True)
                            last = time.time()
            part.replace(dest)
            return dest
        except OSError as e:
            if attempt == retries:
                raise SetupError(f"download failed after {retries} attempts: {url}: {e}") from e
            print(f"    retrying ({attempt}/{retries}) after: {e}")
            time.sleep(3 * attempt)
    return dest


# ------------------------------------------------------------------ authors' code
def authors_manifest() -> dict:
    return json.loads((RESOURCES / "authors_code_sha256.json").read_text(encoding="utf-8"))


# macOS metadata files in the authors' repository; their own .gitignore excludes them, so git may not keep them.
IGNORED_NAMES = {".DS_Store"}


def check_repo(dest: Path, expected: dict[str, str]) -> list[str]:
    """Differences between a local copy and the expected files (empty list = identical)."""
    if not dest.is_dir():
        return ["missing"]
    expected = {f: h for f, h in expected.items() if Path(f).name not in IGNORED_NAMES}
    have = {p.relative_to(dest).as_posix(): p for p in dest.rglob("*")
            if p.is_file() and p.name not in IGNORED_NAMES}
    problems = [f"missing {f}" for f in sorted(set(expected) - set(have))]
    problems += [f"unexpected {f}" for f in sorted(set(have) - set(expected))]
    problems += [f"changed {f}" for f in sorted(set(expected) & set(have)) if _hash(have[f], "sha256") != expected[f]]
    return problems


def verify_authors_code() -> None:
    """The authors' repository ships with this project: confirm it is complete and byte-identical."""
    for name, info in authors_manifest()["repos"].items():
        dest = AUTHORS_CODE_DIR / name
        problems = check_repo(dest, info["files"])
        if problems:
            raise SetupError(f"authors_code/{name} is incomplete or modified ({problems[:5]}). Restore it with "
                             f"`git checkout -- authors_code/{name}`, or from {info['github']} at commit {info['commit']}.")
        n = sum(1 for f in info["files"] if Path(f).name not in IGNORED_NAMES)
        print(f"  [ok] authors_code/{name}: {n} files identical to {info['github']}@{info['commit'][:7]} "
              f"({info['license']})")


# ------------------------------------------------------------------ safe extraction
_S_IFMT, _S_IFREG, _S_IFDIR = 0o170000, 0o100000, 0o040000


def safe_target(root: Path, name: str) -> Path:
    """Destination of an archive entry, refusing anything that would land outside ``root``.

    Uses path components (Path.is_relative_to), not string prefixes: "/tmp/x-evil" is not inside "/tmp/x".
    """
    root_r = Path(root).resolve()
    if not name or Path(name).is_absolute() or Path(name).drive or ":" in name:
        raise SetupError(f"unsafe archive entry {name!r}: absolute path")
    target = (root_r / name).resolve()
    if target == root_r or not target.is_relative_to(root_r):
        raise SetupError(f"unsafe archive entry {name!r}: resolves outside {root_r}")
    return target


def extract_zip(archive: Path, into: Path) -> int:
    """Extract only regular files and directories, every one inside ``into``. Returns the number of files."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for m in members:                                    # validate everything before writing anything
            mode = (m.external_attr >> 16) & _S_IFMT
            if mode not in (0, _S_IFREG, _S_IFDIR):
                raise SetupError(f"refusing archive entry {m.filename!r}: not a regular file or directory")
            safe_target(into, m.filename)
        files = 0
        for m in members:
            target = safe_target(into, m.filename)
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                _make_writable(target)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            files += 1
    return files


def _strict_tar_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    """tarfile's "data" filter (no absolute paths, nothing outside the target), plus: no links or special files."""
    member = tarfile.data_filter(member, path)
    if not (member.isfile() or member.isdir()):
        raise SetupError(f"refusing archive entry {member.name!r}: not a regular file or directory")
    return member


def extract_tar(archive: Path, into: Path) -> None:
    Path(into).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as t:
        t.extractall(into, filter=_strict_tar_filter)


def archive_ok(archive: Path, spec: dict) -> bool:
    """The archive is present, has Zenodo's size, and matches Zenodo's MD5."""
    return archive.is_file() and archive.stat().st_size == spec["bytes"] and _hash(archive, "md5") == spec["md5"]


def extraction_problems(archive: Path, spec: dict) -> list[str]:
    """Files of the archive that are missing or have a different size on disk (empty list = complete)."""
    if not spec["extract"]:
        return []
    into = Path(spec["into"])
    problems: list[str] = []
    if spec["extract"] == "zip":
        with zipfile.ZipFile(archive) as z:
            entries = [(m.filename, m.file_size) for m in z.infolist() if not m.is_dir()]
    else:
        with tarfile.open(archive, "r:gz") as t:
            entries = [(m.name, m.size) for m in t if m.isfile()]
    for name, size in entries:
        p = into / name
        if not p.is_file():
            problems.append(f"missing {name}")
        elif p.stat().st_size != size:
            problems.append(f"size differs: {name}")
    return problems


# ------------------------------------------------------------------ Zenodo data
def fetch_zenodo(name: str, force: bool = False) -> None:
    spec = ZENODO_FILES[name]
    archive = ZENODO_DIR / name
    if not force and archive_ok(archive, spec):
        if not extraction_problems(archive, spec):
            print(f"  [ok] data/zenodo/{name}: size and MD5 verified" + (", unpacked files complete"
                                                                         if spec["extract"] else ""))
            return
        print(f"  {name}: archive verified, unpacked files incomplete; unpacking again")
    else:
        print(f"  downloading {name} ({_human(spec['bytes'])}) from Zenodo {ZENODO_RECORD} ...")
        if archive.exists():
            _make_writable(archive)
            archive.unlink()
        download(ZENODO_URL.format(record=ZENODO_RECORD, name=name), archive, spec["bytes"])
        if not archive_ok(archive, spec):
            got = _hash(archive, "md5")
            archive.unlink()
            raise SetupError(f"{name}: size or MD5 ({got}) does not match Zenodo's; the file was removed, rerun setup")
    if spec["extract"] == "zip":
        extract_zip(archive, spec["into"])
    elif spec["extract"] == "tar":
        extract_tar(archive, spec["into"])
    left = extraction_problems(archive, spec)
    if left:
        raise SetupError(f"{name}: unpacked files are incomplete after extraction: {left[:5]}")
    print(f"  [ok] {name}: size and MD5 verified" + (f", unpacked into {Path(spec['into']).relative_to(DATA_DIR.parent)}"
                                                     if spec["extract"] else ", kept packed"))


def check_checkpoint() -> None:
    manifest = json.loads((PREPROCESSING_DIR / "published-model-v1" / "manifest.json").read_text(encoding="utf-8"))
    want = manifest["checkpoint"]["sha256"]
    got = _hash(TRAINED_MODEL, "sha256")
    if got != want:
        raise SetupError(f"checkpoint {TRAINED_MODEL} has SHA-256 {got}, the preprocessing package expects {want}")
    print(f"  [ok] checkpoint matches the preprocessing package ({want[:12]}...)")


# ------------------------------------------------------------------ side environments
def _env_works(env: dict) -> str | None:
    exe = Path(env["python_exe"])
    if not exe.is_file():
        return None
    r = subprocess.run([str(exe), "-W", "ignore", "-c", env["check"]], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def build_side_envs(force: bool = False) -> None:
    uv = shutil.which("uv")
    for name, env in SIDE_ENVS.items():
        version = None if force else _env_works(env)
        if version:
            print(f"  [ok] {env['venv'].name} ({name} {version}): {env['why']}")
            continue
        if not uv:
            raise SetupError("uv is not on PATH; install uv (https://docs.astral.sh/uv/) and rerun setup")
        print(f"  creating {env['venv'].name} with Python {env['python']} ...")
        subprocess.run([uv, "venv", "--python", env["python"], str(env["venv"])], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(env["python_exe"]), *env["packages"]], check=True)
        version = _env_works(env)
        if not version:
            raise SetupError(f"{env['venv'].name} was created but `{env['check']}` fails")
        print(f"  [ok] {env['venv'].name} ({name} {version}): {env['why']}")


# ------------------------------------------------------------------ installation self-test
SELF_TEST_PDOCKQ = 0.15463054          # published value for the authors' 4k38 example (TopoDockQ README)


def self_test(with_original: bool) -> None:
    """Regenerate the authors' 4k38 example features and p-DockQ; both must match the published values."""
    import numpy as np

    from . import feature_gen, preprocessing
    from .paths import AUTHORS_REPO

    iface = AUTHORS_REPO / "data" / "interface_files" / "4k38_ranked_44_sp_interface.pdb"
    reference = np.load(AUTHORS_REPO / "feature" / "feature_4k38_ranked_44_sp_interface.npy")
    engines = ["compat"] + (["original"] if with_original else [])
    with tempfile.TemporaryDirectory() as tmp:
        for engine in engines:
            job = feature_gen.generate_features([feature_gen.FeatureSample("4k38_ranked_44", "4k38", 44, str(iface))],
                                                Path(tmp) / engine, engine=engine, workers=1)
            rec = job["records"][0]
            if rec["status"] != "ok":
                raise SetupError(f"self-test: the {engine} feature engine failed: {rec.get('error')}")
            arr = np.load(rec["npy_file"])
            if not np.allclose(arr, reference, rtol=1e-9, atol=1e-9):
                raise SetupError(f"self-test: {engine} features differ from the authors' 4k38 example "
                                 f"(max abs diff {np.abs(arr - reference).max():.3e})")
            p = preprocessing.predict_frame(job["table"], device="cpu")["predictions"][0]
            if abs(p - SELF_TEST_PDOCKQ) > 5e-7:
                raise SetupError(f"self-test: p-DockQ {p:.8f} differs from the published {SELF_TEST_PDOCKQ}")
            print(f"  [ok] {engine} engine: 2,754 features match the authors' example; p-DockQ {p:.8f} "
                  f"(published {SELF_TEST_PDOCKQ})")


# ------------------------------------------------------------------ main
def plan(groups: set[str], skip_envs: bool) -> list[str]:
    steps = ["verify the shipped authors' code in authors_code/TopoDockQ (no download)"]
    for name, spec in ZENODO_FILES.items():
        if spec["group"] in groups:
            steps.append(f"Zenodo {name} ({_human(spec['bytes'])}, group: {spec['group']})")
    if not skip_envs:
        steps += [f"side environment {e['venv'].name}: {e['why']}" for e in SIDE_ENVS.values()]
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="topodockq-setup", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true", help="also download the PFPD AlphaFold models (233 MB)")
    ap.add_argument("--reproduction", action="store_true",
                    help="also download the data for the paper reproduction (about 4.7 GB, includes --examples)")
    ap.add_argument("--skip-envs", action="store_true", help="do not create the two side environments")
    ap.add_argument("--force", action="store_true", help="re-download and replace even if files are present")
    ap.add_argument("--plan", action="store_true", help="only print what would be done")
    args = ap.parse_args(argv)

    groups = {"core"} | ({"examples"} if args.examples or args.reproduction else set()) | (
        {"reproduction"} if args.reproduction else set())
    print("TopoDockQ Agents setup")
    for s in plan(groups, args.skip_envs):
        print(f"  - {s}")
    if args.plan:
        return 0
    try:
        print("\n[1] authors' code (shipped with this repository)")
        verify_authors_code()
        print("\n[2] data and model (Zenodo, CC BY 4.0)")
        for name, spec in ZENODO_FILES.items():
            if spec["group"] in groups:
                fetch_zenodo(name, args.force)
        check_checkpoint()
        if not args.skip_envs:
            print("\n[3] side environments")
            build_side_envs(args.force)
        print("\n[4] self-test: reproduce the authors' published 4k38 example")
        self_test(with_original=not args.skip_envs)
    except (SetupError, subprocess.CalledProcessError) as e:
        print(f"\nSETUP FAILED: {e}")
        return 1
    print("\nSetup complete. Next: create your .env with an Anthropic API key (see README), then run\n"
          "  uv run topodockq --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
