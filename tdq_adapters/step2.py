"""Milestone 2: reproduce the SinglePPD-Test_70% panel of the paper's Figure 2.

Run from the project root:

    python -m tdq_adapters.step2

Steps (no LLM involved; the run guard in runner.py adds the protected-directory
check before and after, and writes a structured report even if a step fails):
  1. Stage the authors' repository into runs/step2-<timestamp>/workdir.
  2. Authors' 02_tutorial_inference.py, unchanged: p-DockQ for every validation
     and test candidate (baseline predictions; their output has no IDs).
  3. Rebuild the authors' merged table with their own lines to recover the IDs and
     af_confidence for those rows, and verify alignment: their True_DockQ column
     must equal our pdb2sql_DockQ column row for row.
  4. Predict the same rows with the preprocessing package and compare.
  5. Audit candidate coverage of the published tables (read-only).
  6. Top-1 selection metrics (see evaluation.py) for af_confidence and p-DockQ on
     SinglePPD-Test_70% (paper Figure 2), plus the full test set and
     SinglePPD-Test_LowSimilarity (paper reports these only in the Supplement).
  7. Compare with the paper's Figure 2 values.
Writes runs/step2-<timestamp>/step2_report.json, candidate_coverage.json and
per-complex selection CSVs. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import author_scripts, data_audit, evaluation, preprocessing
from .paths import AUTHORS_REPO, RUNS_DIR
from .provenance import sha256
from .runner import fmt, run_guarded
from .staging import stage_author_repo

SIMILARITY_ZIP = AUTHORS_REPO / "data" / "sequence_similairty.zip"   # authors' spelling
SIM70_MEMBER = "five_sim_70.csv"
SIM70_SOURCE = "test"                                                # rows of five_sim_70.csv for SinglePPD-Test
LOWSIM_CSV = AUTHORS_REPO / "data" / "PDB_ID" / "singlePPD_test_LowSimilarity_similarity_30.csv"

# Paper Figure 2, SinglePPD-Test_70% bars (read from the rendered figure; printed to 2 decimals).
# Text p.4: "F1 score changes ranging from ... to an increase of 0.09 for SinglePPD-Test_70%".
PAPER_TEST70 = {
    "af_confidence": {"FPR": 0.40, "precision": 0.67},
    "TopoDockQ": {"FPR": 0.15, "precision": 0.84},
    "F1_change_TopoDockQ_minus_af": 0.09,
}
PAPER_TOL = 0.005        # half a unit in the paper's last printed digit
ALIGN_TOL = 0.0          # both sides are float32; aligned rows must be identical
PACKAGE_TOL = 1e-6       # package vs authors' script on the same device

AUTHORS_OUTPUT_COLUMNS = ("True_DockQ", "Predicted_DockQ(p-DockQ)")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.replace("%", "pct")).strip("_")


def _compare(paper: float, ours: float | None) -> dict:
    if ours is None:
        return {"paper": paper, "ours": None, "abs_diff": None, "match": False}
    return {"paper": paper, "ours": ours, "abs_diff": abs(ours - paper), "match": abs(ours - paper) <= PAPER_TOL}


def body(report: dict, run_dir: Path) -> dict[str, bool]:
    workdir, log_dir = run_dir / "workdir", run_dir / "logs"
    report["definitions"] = evaluation.__doc__
    report["tolerances"] = {"paper": PAPER_TOL, "alignment": ALIGN_TOL, "package_vs_script": PACKAGE_TOL}

    print(f"[1/7] staging authors' repository -> {workdir}")
    report["staging"] = stage_author_repo(workdir)
    report["inputs"] = {str(p): sha256(p) for p in (
        AUTHORS_REPO / "02_tutorial_inference.py", AUTHORS_REPO / "src" / "model.py",
        preprocessing.FEATURES_CSV, preprocessing.LABELS_CSV, SIMILARITY_ZIP, LOWSIM_CSV)}

    print("[2/7] authors' 02 script, unchanged (validation + test predictions)")
    report["run_02"], authors_out = author_scripts.tutorial_inference(workdir, log_dir)

    print("[3/7] rebuilding the authors' merged table and checking row alignment")
    df1, _ = preprocessing.author_merged_table()
    tables = {"val": df1[df1["data_class"] == "valid"].reset_index(drop=True),     # 02:L89
              "test": df1[df1["data_class"] == "test"].reset_index(drop=True)}     # 02:L90
    report["alignment"] = {}
    for split, t in tables.items():
        a = authors_out[split]
        missing_cols = [c for c in AUTHORS_OUTPUT_COLUMNS if c not in a.columns]
        if missing_cols:
            raise RuntimeError(f"{split}: authors' output lacks columns {missing_cols}")
        entry = {"rows_authors": int(len(a)), "rows_ours": int(len(t))}
        if len(a) != len(t):
            entry.update(aligned=False, reason="row counts differ")
        else:
            # The authors' script passes DockQ through a float32 tensor, so compare in float32.
            diff = float(np.abs(a["True_DockQ"].to_numpy(dtype=np.float32)
                                - t["pdb2sql_DockQ"].to_numpy(dtype=np.float32)).max())
            entry.update(max_abs_true_dockq_diff=diff, aligned=diff <= ALIGN_TOL)
        report["alignment"][split] = entry
        if not entry["aligned"]:
            raise RuntimeError(f"{split}: authors' output does not align with the merged table: {entry}")
        t["p_dockq_authors_02"] = a["Predicted_DockQ(p-DockQ)"].to_numpy()

    print("[4/7] predicting the same rows with the preprocessing package")
    report["package_check"] = {}
    for split, t in tables.items():
        res = preprocessing.predict_frame(t, device=report["device"]["device"])
        t["p_dockq_package"] = res["predictions"]
        d = float(np.abs(t["p_dockq_package"] - t["p_dockq_authors_02"]).max())
        report["package_check"][split] = {"device": res["device"], "rows": res["n_rows"],
                                          "max_abs_diff_vs_authors_02": d, "pass": d <= PACKAGE_TOL}

    print("[5/7] auditing candidate coverage of the published tables (read-only)")
    audit = data_audit.audit_published_tables()
    (run_dir / "candidate_coverage.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    report["label_only_train_rows"] = audit["label_only_train_rows"]

    print("[6/7] top-1 selection metrics")
    test = tables["test"]
    test_ids = set(test["pdb_id"])
    sim70 = pd.read_csv(io.BytesIO(zipfile.ZipFile(SIMILARITY_ZIP).read(SIM70_MEMBER)))
    ids70 = set(sim70.loc[sim70["source"] == SIM70_SOURCE, "pdb_id"])
    ids_low = set(pd.read_csv(LOWSIM_CSV)["pdb_id"])
    report["subset_ids"] = {}
    for name, ids in (("SinglePPD-Test_70%", ids70), ("SinglePPD-Test_LowSimilarity", ids_low)):
        absent = sorted(ids - test_ids)
        report["subset_ids"][name] = {"listed": len(ids), "found_in_test": len(ids & test_ids), "absent": absent}
        if absent:
            raise RuntimeError(f"{name}: {len(absent)} listed complexes have no test candidates: {absent[:10]}")
    subsets = {"SinglePPD-Test_70%": test[test["pdb_id"].isin(ids70)],
               "SinglePPD-Test (full)": test,
               "SinglePPD-Test_LowSimilarity": test[test["pdb_id"].isin(ids_low)]}
    scorers = {"af_confidence": "af_confidence", "TopoDockQ": "p_dockq_authors_02",
               "TopoDockQ (package)": "p_dockq_package"}
    report["metrics"], report["candidate_coverage"] = {}, {"published_tables": "candidate_coverage.json"}
    for name, sub in subsets.items():
        ev = evaluation.evaluate(sub, scorers)
        for scorer in scorers:
            pd.DataFrame(ev[scorer].pop("selected")).to_csv(run_dir / f"top1_{_slug(name)}_{_slug(scorer)}.csv",
                                                            index=False)
        report["metrics"][name] = ev
        report["candidate_coverage"][name] = data_audit.coverage(sub)
    feature_cols = [c for c in test.columns if c.startswith(("persistent_", "static_"))]
    test.drop(columns=feature_cols).to_csv(run_dir / "test_predictions_with_ids.csv", index=False)

    print("[7/7] comparing with the paper")
    m = report["metrics"]["SinglePPD-Test_70%"]
    comparison = {f"{s}.{k}": _compare(PAPER_TEST70[s][k], m[s][k])
                  for s in ("af_confidence", "TopoDockQ") for k in ("FPR", "precision")}
    f1_t, f1_a = m["TopoDockQ"]["F1"], m["af_confidence"]["F1"]
    comparison["F1_change"] = _compare(PAPER_TEST70["F1_change_TopoDockQ_minus_af"],
                                       None if f1_t is None or f1_a is None else f1_t - f1_a)
    report["paper_comparison_test70"] = comparison

    for name, ev in report["metrics"].items():
        print(f"\n{name}: {ev['n_complexes']} complexes, {ev['n_candidates']} candidates, "
              f"{ev['complexes_with_fewer_than_50_candidates']} complexes with <50 candidates")
        for scorer in scorers:
            r = ev[scorer]
            print(f"  {scorer:22s} {r['confusion']}  FPR={fmt(r['FPR'])}  precision={fmt(r['precision'])}  "
                  f"recall={fmt(r['recall'])}  F1={fmt(r['F1'])}  ties={r['tied_top_scores']}")
    print("\npaper comparison (Test_70%):")
    for k, v in comparison.items():
        print(f"  {k:26s} paper={v['paper']}  ours={fmt(v['ours'])}  match={v['match']}")

    return {
        "alignment": all(v["aligned"] for v in report["alignment"].values()),
        "package_matches_authors": all(v["pass"] for v in report["package_check"].values()),
        "paper_values_match": all(v["match"] for v in comparison.values()),
        "no_unexplained_label_only_rows": report["label_only_train_rows"]["unexplained_rows"] == 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or RUNS_DIR / f"step2-{time.strftime('%Y%m%d-%H%M%S')}"
    code, _ = run_guarded("step2", run_dir, body)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
