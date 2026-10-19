"""Milestone 3: DataAgent tools - structure validation and interface extraction.

Run from the project root:

    python -m tdq_adapters.step3 [--workers 8] [--limit-per-complex N]

Steps (no LLM involved; the run guard in runner.py adds the protected-directory
check before and after, and writes a structured report even if a step fails):
  1. Authors' example: run their extract_interface.py, unchanged, on
     src/example/1b3l_crystal.pdb and compare with their src/example/1b3l_interface.pdb.
  2. PFPD benchmark (27 complexes x 50 AF2-M models from Zenodo): for every model,
     validate ranked_N_sp.pdb, extract its interface with the authors' script through
     data_prep.prepare_interfaces, and compare with the authors' published
     ranked_N_sp_interface.pdb.
Writes runs/step3-<timestamp>/step3_report.json and manifest.jsonl.
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data_prep, structures
from .paths import AUTHORS_EXTRACT_INTERFACE, AUTHORS_REPO, EXTERNAL_DIR, PRODY_PYTHON, RUNS_DIR
from .provenance import sha256
from .runner import run_guarded

PFPD_DIR = EXTERNAL_DIR / "PFPD_AF2_models"
EXAMPLE_IN = AUTHORS_REPO / "src" / "example" / "1b3l_crystal.pdb"
EXAMPLE_REF = AUTHORS_REPO / "src" / "example" / "1b3l_interface.pdb"
MODEL_RE = re.compile(r"ranked_(\d+)_sp\.pdb$")
PFPD_EXPECTED_COMPLEXES = 27            # paper Table 1: PFPD has 27 complexes
PFPD_EXPECTED_MODEL_IDS = set(range(50))  # 50 AF2-M models per complex, ranked_0 .. ranked_49


def prody_env_versions() -> dict[str, str]:
    code = ("import sys,prody,numpy,scipy,pyparsing,setuptools;"
            "print(sys.version.split()[0],prody.__version__,numpy.__version__,scipy.__version__,"
            "pyparsing.__version__,setuptools.__version__)")
    out = subprocess.run([str(PRODY_PYTHON), "-W", "ignore", "-c", code], capture_output=True, text=True).stdout.split()
    keys = ("python", "prody", "numpy", "scipy", "pyparsing", "setuptools")
    return dict(zip(keys, out)) if len(out) == len(keys) else {"error": "could not query ProDy environment"}


def pfpd_samples(limit_per_complex: int | None) -> list[data_prep.Sample]:
    samples = []
    for cdir in sorted(p for p in PFPD_DIR.iterdir() if p.is_dir()):
        models = sorted((int(m.group(1)), f) for f in cdir.iterdir() if (m := MODEL_RE.search(f.name)))
        for model_id, f in models[:limit_per_complex]:
            pdb_id = cdir.name.lower()
            samples.append(data_prep.Sample(sample_id=f"{pdb_id}_ranked_{model_id}", pdb_id=pdb_id,
                                            model_id=model_id, input_pdb=str(f)))
    return samples


def body(report: dict, run_dir: Path, workers: int, limit_per_complex: int | None) -> dict[str, bool]:
    report["prody_environment"] = prody_env_versions()
    report["rules"] = structures.RULES
    report["authors_script"] = {"path": str(AUTHORS_EXTRACT_INTERFACE), "sha256": sha256(AUTHORS_EXTRACT_INTERFACE)}

    print("[1/2] authors' example: 1b3l crystal -> interface")
    ex = data_prep.prepare_interfaces(
        [data_prep.Sample("1b3l_crystal", "1b3l", 0, str(EXAMPLE_IN))], run_dir / "example", workers=1)
    ex_rec = ex["records"][0]
    report["example"] = {"record": ex_rec,
                         "comparison": structures.compare_interfaces(ex_rec["interface_file"], EXAMPLE_REF)
                         if ex_rec["interface_file"] else None}

    samples = pfpd_samples(limit_per_complex)
    per_complex: dict[str, set[int]] = {}
    for s in samples:
        per_complex.setdefault(s.pdb_id, set()).add(s.model_id)
    complete = (len(per_complex) == PFPD_EXPECTED_COMPLEXES
                and all(ids == PFPD_EXPECTED_MODEL_IDS for ids in per_complex.values())
                and len(samples) == PFPD_EXPECTED_COMPLEXES * len(PFPD_EXPECTED_MODEL_IDS))
    report["scope"] = "full" if limit_per_complex is None else f"partial (limit_per_complex={limit_per_complex})"
    report["pfpd_population"] = {"complexes": len(per_complex), "samples": len(samples),
                                 "expected_complexes": PFPD_EXPECTED_COMPLEXES,
                                 "expected_samples": PFPD_EXPECTED_COMPLEXES * len(PFPD_EXPECTED_MODEL_IDS),
                                 "complete": complete,
                                 "complexes_with_wrong_id_set": sorted(p for p, ids in per_complex.items()
                                                                       if ids != PFPD_EXPECTED_MODEL_IDS)}
    if not samples:
        raise RuntimeError(f"no PFPD models found under {PFPD_DIR}")
    print(f"[2/2] PFPD ({report['scope']}): {len(samples)} models from {len(per_complex)} complexes, "
          f"{workers} workers")
    t0 = time.time()
    job = data_prep.prepare_interfaces(samples, run_dir / "interfaces", workers=workers,
                                       manifest_path=run_dir / "manifest.jsonl")
    report["pfpd_job"] = {**job["summary"], "seconds": round(time.time() - t0, 1)}

    def compare(rec):
        ref = Path(rec["input_pdb"]).with_name(Path(rec["input_pdb"]).stem + "_interface.pdb")
        if not rec["interface_file"]:
            return {"sample_id": rec["sample_id"], "compared": False}
        return {"sample_id": rec["sample_id"], "compared": True,
                **structures.compare_interfaces(rec["interface_file"], ref)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        comps = list(pool.map(compare, job["records"]))
    differing = [c for c in comps if c["compared"] and not c["identical_feature_input"]]
    pc = report["pfpd_comparison"] = {
        "compared": sum(c["compared"] for c in comps),
        "identical_atoms": sum(1 for c in comps if c.get("identical_atoms")),
        "identical_feature_input": sum(1 for c in comps if c.get("identical_feature_input")),
        "identical_bytes": sum(1 for c in comps if c.get("identical_bytes")),
        "byte_differences_explained": "our files (written by the authors' script via ProDy) add a REMARK "
                                      "line and TER/END records and renumber atom serials from 1; the "
                                      "published files keep the source serials. The feature code ignores both.",
        "differing": differing[:50],
    }
    (run_dir / "pfpd_comparison.json").write_text(json.dumps(comps, indent=1), encoding="utf-8")
    print(json.dumps({"example": report["example"]["comparison"], "pfpd_job": report["pfpd_job"],
                      "pfpd_comparison": {k: v for k, v in pc.items() if k != "differing"}}, indent=1))

    exc = report["example"]["comparison"] or {}
    population_check = ({"pfpd_full_population_27x50": complete} if limit_per_complex is None
                        else {"partial_run_explicitly_requested": True})
    return {
        **population_check,
        "example_identical_feature_input": bool(exc.get("identical_atoms") and exc.get("identical_feature_input")),
        "pfpd_all_extracted": report["pfpd_job"]["excluded"] == 0,
        "pfpd_all_identical_feature_input": (pc["compared"] == len(samples)
                                             and pc["identical_feature_input"] == len(samples)
                                             and pc["identical_atoms"] == len(samples)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-per-complex", type=int, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or RUNS_DIR / f"step3-{time.strftime('%Y%m%d-%H%M%S')}"
    code, _ = run_guarded("step3", run_dir,
                          lambda report, rd: body(report, rd, args.workers, args.limit_per_complex))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
