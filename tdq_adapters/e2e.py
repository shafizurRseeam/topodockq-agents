"""End-to-end deterministic smoke test: full PDB -> interface -> features -> p-DockQ -> ranking.

    python -m tdq_adapters.e2e [--complexes 3] [--models 10] [--workers 8]

Uses full AlphaFold models (ranked_N_sp.pdb) of the first ``--complexes`` PFPD
complexes, with no hand-prepared intermediate files:
  A. pipeline.score_structures with the compat feature engine;
  B. pipeline.score_structures with the authors' original feature code;
  C. reference path: the authors' own published interface files for the same
     models -> features (compat) -> preprocessing package -> p-DockQ.
Checks: every structure scored; A, B and C give the same p-DockQ (1e-6) and the
same top-1 candidate per complex; protected directories unchanged.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_prep, feature_gen, pipeline, preprocessing
from .paths import EXTERNAL_DIR, RUNS_DIR
from .runner import run_guarded

PFPD_DIR = EXTERNAL_DIR / "PFPD_AF2_models"
TOL = 1e-6


def body(report, run_dir: Path, n_complexes: int, n_models: int, workers: int):
    complexes = sorted(p for p in PFPD_DIR.iterdir() if p.is_dir())[:n_complexes]
    structures = [data_prep.Sample(f"{c.name.lower()}_ranked_{m}", c.name.lower(), m, str(c / f"ranked_{m}_sp.pdb"))
                  for c in complexes for m in range(n_models)]
    report["inputs"] = {"complexes": [c.name for c in complexes], "models_per_complex": n_models,
                        "structures": len(structures)}

    results = {}
    for engine in feature_gen.ENGINES:
        print(f"[{engine}] full PDB -> p-DockQ for {len(structures)} structures")
        res = pipeline.score_structures(structures, run_dir / f"pipeline_{engine}", engine=engine, workers=workers)
        results[engine] = res
        report[f"pipeline_{engine}"] = res["provenance"]

    print("[reference] authors' published interface files -> features -> p-DockQ")
    ref_dir = run_dir / "reference_interfaces"
    ref_dir.mkdir()
    fs = []
    for s in structures:
        src = Path(s.input_pdb).with_name(f"ranked_{s.model_id}_sp_interface.pdb")
        dst = ref_dir / f"{s.pdb_id}_ranked_{s.model_id}_sp_interface.pdb"
        shutil.copyfile(src, dst)
        fs.append(feature_gen.FeatureSample(s.sample_id, s.pdb_id, s.model_id, str(dst)))
    feats = feature_gen.generate_features(fs, run_dir / "reference_features", engine="compat", workers=workers)
    ref = feats["table"][pipeline.KEY].copy()
    ref["p_dockq"] = preprocessing.predict_frame(feats["table"], device="cpu")["predictions"]
    ref["sample_id"] = [f"{p}_ranked_{m}" for p, m in zip(ref.pdb_id, ref.af_model_id)]
    ref["status"] = "scored"
    ref_rank = pipeline.rank_candidates(ref)

    a, b = results["compat"]["predictions"], results["original"]["predictions"]
    m = a.merge(b, on=pipeline.KEY, suffixes=("_compat", "_original")).merge(
        ref.rename(columns={"p_dockq": "p_dockq_reference"})[pipeline.KEY + ["p_dockq_reference"]], on=pipeline.KEY)
    m.to_csv(run_dir / "pdockq_three_paths.csv", index=False)
    d_co = float(np.abs(m.p_dockq_compat - m.p_dockq_original).max())
    d_cr = float(np.abs(m.p_dockq_compat - m.p_dockq_reference).max())
    top = lambda r: dict(zip(r.pdb_id, r.top1_af_model_id))  # noqa: E731
    report["comparison"] = {"rows": int(len(m)), "max_abs_compat_vs_original": d_co,
                            "max_abs_compat_vs_reference": d_cr,
                            "top1_compat": top(results["compat"]["ranking"]),
                            "top1_original": top(results["original"]["ranking"]), "top1_reference": top(ref_rank)}
    print(results["compat"]["ranking"].to_string(index=False))
    print(report["comparison"])
    return {
        "all_structures_scored_compat": results["compat"]["provenance"]["counts"]["scored"] == len(structures),
        "all_structures_scored_original": results["original"]["provenance"]["counts"]["scored"] == len(structures),
        "pdockq_compat_equals_original": len(m) == len(structures) and d_co <= TOL,
        "pdockq_full_pdb_equals_published_interface_path": len(m) == len(structures) and d_cr <= TOL,
        "same_top1_all_paths": top(results["compat"]["ranking"]) == top(results["original"]["ranking"]) == top(ref_rank),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complexes", type=int, default=3)
    ap.add_argument("--models", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or RUNS_DIR / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    code, _ = run_guarded("e2e", run_dir, lambda rep, rd: body(rep, rd, args.complexes, args.models, args.workers))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
