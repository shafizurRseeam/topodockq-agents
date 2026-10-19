"""Read-only audit of candidate coverage in the published SinglePPD tables.

The paper (Methods, AF2-M section) states: "All 50 models were successfully
generated for each peptide-protein complex without failure." The published label
and feature tables nevertheless contain fewer than 50 candidates for some
complexes. This module documents that gap. It never modifies the published files.

Coverage is judged from the set of model IDs per complex, not from row counts, so
duplicate rows, out-of-range IDs and missing IDs are each detected on their own.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import PROCESSED_DATA_DIR
from .paths import to_project_relative
from .preprocessing import FEATURES_CSV, LABELS_CSV

FILTERED_LABELS_CSV = PROCESSED_DATA_DIR / "singlePPD_filtered_DockQ.csv"
EXPECTED_CANDIDATES = 50          # paper: 50 AF2-M models per complex
EXPECTED_MODEL_IDS = frozenset(range(EXPECTED_CANDIDATES))   # tables use af_model_id 0-49
SPLITS = ("train", "valid", "test")
KEY = ["pdb_id", "af_model_id"]


def _valid_model_id(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f)) and f == int(f) and int(f) in EXPECTED_MODEL_IDS


def coverage(df: pd.DataFrame, expected_ids: frozenset[int] = EXPECTED_MODEL_IDS) -> dict[str, Any]:
    """Model-ID coverage per complex for one table (or subset)."""
    dup_mask = df.duplicated(KEY, keep="first")
    valid_mask = df["af_model_id"].map(_valid_model_id)
    duplicates = df[dup_mask].groupby("pdb_id")["af_model_id"].apply(lambda s: sorted(map(int, s))).to_dict()
    out_of_range = df[~valid_mask].groupby("pdb_id")["af_model_id"].apply(lambda s: sorted(map(str, s))).to_dict()

    ids = df[valid_mask & ~dup_mask].groupby("pdb_id")["af_model_id"].apply(lambda s: set(map(int, s)))
    missing_ids = {p: sorted(expected_ids - s) for p, s in ids.items() if expected_ids - s}
    for p in set(df["pdb_id"]) - set(ids.index):          # complexes with no valid IDs at all
        missing_ids[p] = sorted(expected_ids)
    n_complexes = int(df["pdb_id"].nunique())
    freq = Counter(i for v in missing_ids.values() for i in v)
    return {
        "complexes": n_complexes,
        "rows": int(len(df)),
        "unique_valid_candidates": int(sum(len(s) for s in ids)),
        "expected_candidates": int(len(expected_ids) * n_complexes),
        "missing_candidates": int(sum(len(v) for v in missing_ids.values())),
        "complexes_with_incomplete_id_set": len(missing_ids),
        "duplicate_rows": int(dup_mask.sum()),
        "out_of_range_or_invalid_ids": int((~valid_mask).sum()),
        "min_unique_candidates_per_complex": int(ids.map(len).min()) if len(ids) else 0,
        "most_frequently_missing_model_ids": freq.most_common(10),
        "duplicates": duplicates,
        "invalid_ids": out_of_range,
        "missing_model_ids": missing_ids,
    }


def classify_label_only_train_rows(labels: pd.DataFrame, filtered: pd.DataFrame,
                                   feats: pd.DataFrame) -> dict[str, Any]:
    """Explain every training label row that has no feature row, by direct ID comparison.

    intentionally_filtered   key absent from singlePPD_filtered_DockQ.csv (the authors'
                             near-duplicate filter removed it)
    missing_label            key present in the filtered file, but its DockQ label is blank
    unexplained              key present in the filtered file with a label, yet no features
    """
    lab = labels[labels["data_class"] == "train"]
    flt = filtered[filtered["data_class"] == "train"][KEY].assign(_in_filtered=True)
    fea = feats[feats["data_class"] == "train"][KEY].assign(_in_features=True)
    m = lab.merge(fea, on=KEY, how="left").merge(flt, on=KEY, how="left")
    only = m[m["_in_features"].isna()]
    in_filtered = only["_in_filtered"].notna()
    no_label = only["pdb2sql_DockQ"].isna()
    groups = {
        "intentionally_filtered": only[~in_filtered],
        "missing_label": only[in_filtered & no_label],
        "unexplained": only[in_filtered & ~no_label],
    }
    return {
        "train_label_rows": int(len(lab)),
        "label_only_rows": int(len(only)),
        **{f"{k}_rows": int(len(v)) for k, v in groups.items()},
        "missing_label_keys": groups["missing_label"][KEY].to_dict("records"),
        "unexplained_keys": groups["unexplained"][KEY].to_dict("records")[:100],
        "filtered_train_rows_absent_from_full_labels": int(len(
            flt.merge(lab[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'"))),
    }


def audit_published_tables(features_csv: Path = FEATURES_CSV, labels_csv: Path = LABELS_CSV,
                           filtered_csv: Path = FILTERED_LABELS_CSV) -> dict[str, Any]:
    labels = pd.read_csv(labels_csv)
    filtered = pd.read_csv(filtered_csv)
    feats = pd.read_csv(features_csv, usecols=["pdb_id", "af_model_id", "pdb2sql_DockQ", "data_class"])
    out: dict[str, Any] = {
        "labels_csv": to_project_relative(labels_csv), "filtered_csv": to_project_relative(filtered_csv),
        "features_csv": to_project_relative(features_csv),
        "missing_dockq_labels": labels[labels["pdb2sql_DockQ"].isna()][KEY + ["data_class"]].to_dict("records"),
        "label_only_train_rows": classify_label_only_train_rows(labels, filtered, feats),
        "by_split": {},
    }
    summary_keys = ("complexes", "rows", "unique_valid_candidates", "expected_candidates", "missing_candidates",
                    "complexes_with_incomplete_id_set", "duplicate_rows", "out_of_range_or_invalid_ids",
                    "min_unique_candidates_per_complex", "most_frequently_missing_model_ids")
    for split in SPLITS:
        lab_s, feat_s = labels[labels["data_class"] == split], feats[feats["data_class"] == split]
        out["by_split"][split] = {
            "labels": {k: v for k, v in coverage(lab_s).items() if k in summary_keys},
            "features": {k: v for k, v in coverage(feat_s).items() if k in summary_keys},
            "complexes_in_labels_not_in_features": sorted(set(lab_s["pdb_id"]) - set(feat_s["pdb_id"])),
        }
    return out


def audit_interface_archive(tar_path: Path, features_csv: Path = FEATURES_CSV,
                            labels_csv: Path = LABELS_CSV) -> dict[str, Any]:
    """Read-only accounting of empty interface files in singlePPD_interface_files.tar.gz.

    An empty file means the authors' interface extraction found no protein-peptide
    contact within 10 A for that AlphaFold model. Each empty file is matched by
    (pdb_id, af_model_id), parsed from its name, against the label table, the
    feature table and the set of missing candidate IDs (coverage()).
    """
    import re
    import tarfile

    name_re = re.compile(r"^(?P<pdb>[A-Za-z0-9_-]+)_ranked_(?P<model>\d+)_sp_interface\.pdb$")
    sizes: dict[tuple[str, int], int] = {}
    unparsed: list[str] = []
    with tarfile.open(tar_path, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            name = Path(m.name).name
            mt = name_re.match(name)
            if not mt:
                unparsed.append(name)
                continue
            sizes[(mt["pdb"], int(mt["model"]))] = m.size
    labels = pd.read_csv(labels_csv)
    feats = pd.read_csv(features_csv, usecols=KEY + ["data_class"])
    label_keys = {(p, int(m)): c for p, m, c in zip(labels.pdb_id, labels.af_model_id, labels.data_class)}
    feat_keys = set(zip(feats.pdb_id, feats.af_model_id.astype(int)))
    missing = set()
    for split in SPLITS:
        for p, ids_missing in coverage(labels[labels["data_class"] == split])["missing_model_ids"].items():
            missing.update((p, i) for i in ids_missing)
    empty = sorted(k for k, s in sizes.items() if s == 0)
    in_labels = [k for k in empty if k in label_keys]
    in_feats = [k for k in empty if k in feat_keys]
    in_missing = [k for k in empty if k in missing]
    other = [k for k in empty if k not in label_keys and k not in missing]
    rec = lambda keys: [{"pdb_id": p, "af_model_id": m} for p, m in keys]  # noqa: E731
    return {
        "archive": to_project_relative(tar_path),
        "files": len(sizes) + len(unparsed), "unparsed_names": unparsed[:20],
        "empty_files": len(empty),
        "empty_with_label_row": len(in_labels),
        "empty_with_label_row_by_split": dict(Counter(label_keys[k] for k in in_labels)),
        "empty_with_feature_row": len(in_feats),
        "empty_among_missing_candidates": len(in_missing),
        "missing_candidates_total": len(missing),
        "share_of_missing_candidates_explained": round(len(in_missing) / len(missing), 4) if missing else None,
        "empty_neither_labelled_nor_missing": len(other),
        "empty_neither_labelled_nor_missing_complexes": sorted({p for p, _ in other})[:50],
        "missing_candidates_with_nonempty_interface_file": sum(1 for k in missing if sizes.get(k, -1) > 0),
        "missing_candidates_without_interface_file": sum(1 for k in missing if k not in sizes),
        "empty_with_feature_row_keys": rec(in_feats),
        "empty_keys": rec(empty),
    }
