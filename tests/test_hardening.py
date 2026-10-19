"""Regression tests for the Step 3 review fixes: output identity, path containment, profiles, chain roles."""
from pathlib import Path

import pytest

from tdq_adapters import data_prep, feature_gen, ids
from tdq_adapters import structures as st


def _pdb(tmp_path: Path, name="in.pdb", peptide_residues=1, protein_residues=2) -> Path:
    def atom(serial, chain, resnum, el):
        return (f"ATOM  {serial:>5}  CA  ALA {chain}{resnum:>4}    {float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"  1.00  0.00          {el:>2}  \n")
    lines, n = [], 1
    for r in range(1, protein_residues + 1):
        lines.append(atom(n, "A", r, "C")); n += 1
    for r in range(1, peptide_residues + 1):
        lines.append(atom(n, "B", r, "N")); n += 1
    p = tmp_path / name
    p.write_text("".join(lines) + "END\n")
    return p


# ---------------------------------------------------------------- identifiers and paths
@pytest.mark.parametrize("bad", ["../evil", "..", "a/b", "a\\b", "", "x" * 65, "4k38.pdb", None, 7, "C:evil"])
def test_unsafe_pdb_ids_are_rejected(bad):
    with pytest.raises(ids.UnsafeIdentifierError):
        ids.validate_pdb_id(bad)


@pytest.mark.parametrize("bad", [-1, True, "3", 2.0, None])
def test_invalid_model_ids_are_rejected(bad):
    with pytest.raises(ids.UnsafeIdentifierError):
        ids.validate_model_id(bad)


def test_valid_ids_pass():
    assert ids.validate_pdb_id("1b0hB") == "1b0hB" and ids.validate_model_id(0) == 0


def test_contained_path_blocks_escape(tmp_path):
    assert ids.contained_path(tmp_path, "ok.pdb").parent == tmp_path.resolve()
    with pytest.raises(ids.UnsafeIdentifierError):
        ids.contained_path(tmp_path / "out", "../escaped.pdb")


def test_traversal_pdb_id_writes_nothing_outside(tmp_path):
    out = tmp_path / "out"
    s = data_prep.Sample("x", "../escaped", 0, str(_pdb(tmp_path)))
    with pytest.raises(ids.UnsafeIdentifierError):
        data_prep.prepare_interfaces([s], out)
    assert not list(tmp_path.glob("escaped*"))


# ---------------------------------------------------------------- duplicate and stale outputs
def test_duplicate_identities_rejected_before_any_extraction(tmp_path):
    a, b = _pdb(tmp_path, "a.pdb"), _pdb(tmp_path, "b.pdb")
    samples = [data_prep.Sample("s1", "1abc", 3, str(a)), data_prep.Sample("s2", "1abc", 3, str(b))]
    with pytest.raises(ids.DuplicateOutputError):
        data_prep.prepare_interfaces(samples, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.pdb"))


def test_existing_output_is_never_reused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "1abc_ranked_3_sp_interface.pdb").write_text("stale")
    with pytest.raises(ids.StaleOutputError):
        data_prep.prepare_interfaces([data_prep.Sample("s", "1abc", 3, str(_pdb(tmp_path)))], out)
    assert (out / "1abc_ranked_3_sp_interface.pdb").read_text() == "stale"


def test_feature_batch_rejects_duplicates_traversal_and_stale(tmp_path):
    f = tmp_path / "1abc_ranked_3_sp_interface.pdb"
    f.write_text("END\n")
    dup = [feature_gen.FeatureSample("a", "1abc", 3, str(f)), feature_gen.FeatureSample("b", "1abc", 3, str(f))]
    with pytest.raises(ids.DuplicateOutputError):
        feature_gen.generate_features(dup, tmp_path / "job1")
    with pytest.raises(ids.UnsafeIdentifierError):
        feature_gen.generate_features([feature_gen.FeatureSample("../x", "1abc", 3, str(f))], tmp_path / "job2")
    (tmp_path / "job3" / "npy").mkdir(parents=True)
    (tmp_path / "job3" / "npy" / "feature_1abc_ranked_3_sp_interface.npy").write_text("old")
    with pytest.raises(ids.StaleOutputError):
        feature_gen.generate_features([feature_gen.FeatureSample("a", "1abc", 3, str(f))], tmp_path / "job3")
    with pytest.raises(ValueError, match="unknown engine"):
        feature_gen.generate_features([feature_gen.FeatureSample("a", "1abc", 3, str(f))], tmp_path / "job4",
                                      engine="other")


# ---------------------------------------------------------------- profiles
def test_paper_profile_locks_cutoff(tmp_path):
    s = [data_prep.Sample("s", "1abc", 0, str(_pdb(tmp_path)))]
    with pytest.raises(ValueError, match="paper profile"):
        data_prep.check_batch(s, tmp_path / "o1", "paper", 8.0)
    data_prep.check_batch(s, tmp_path / "o2", "experiment", 8.0)     # allowed only when declared
    with pytest.raises(ValueError, match="unknown profile"):
        data_prep.check_batch(s, tmp_path / "o3", "fast", 10.0)


# ---------------------------------------------------------------- chain roles
def test_swapped_chain_roles_warn_but_do_not_exclude(tmp_path):
    r = st.validate_structure(_pdb(tmp_path, peptide_residues=5, protein_residues=2))
    assert r["ok"]
    w = [i for i in r["issues"] if i["code"] == "W_CHAIN_ROLE_SUSPICIOUS"]
    assert w and w[0]["peptide_residues"] == 5 and w[0]["protein_residues"] == 2


def test_normal_chain_roles_do_not_warn(tmp_path):
    r = st.validate_structure(_pdb(tmp_path, peptide_residues=1, protein_residues=3))
    assert "W_CHAIN_ROLE_SUSPICIOUS" not in {i["code"] for i in r["issues"]}


def test_record_documents_chain_assignment_basis(tmp_path):
    bad = _pdb(tmp_path, protein_residues=0)          # no protein chain -> excluded before extraction
    rec = data_prep.prepare_one(data_prep.Sample("x", "1abc", 0, str(bad)), tmp_path / "out")
    assert rec["chain_assignment"]["protein"] == "A" and "do not prove" in rec["chain_assignment"]["basis"]
    assert "input_residues_by_chain" in rec


# ---------------------------------------------------------------- Step 4 review fixes
def _iface(tmp_path, pdb_id="1abc", model_id=0):
    f = tmp_path / f"{pdb_id}_ranked_{model_id}_sp_interface.pdb"
    f.write_text("END\n")
    return str(f)


def test_duplicate_sample_ids_rejected_even_with_distinct_models(tmp_path):
    s = [feature_gen.FeatureSample("sample1", "1abc", 0, _iface(tmp_path, model_id=0)),
         feature_gen.FeatureSample("sample1", "1abc", 1, _iface(tmp_path, model_id=1))]
    with pytest.raises(ids.DuplicateOutputError):
        feature_gen.generate_features(s, tmp_path / "job")
    assert not (tmp_path / "job" / "csv").exists() or not list((tmp_path / "job" / "csv").iterdir())


def test_paper_profile_locks_bins_and_filtration(tmp_path):
    s = [feature_gen.FeatureSample("a", "1abc", 0, _iface(tmp_path))]
    custom = [2.0 + 0.25 * i for i in range(33)]
    custom[5] = 3.3                                  # still 33 values -> still 2754 features
    with pytest.raises(ValueError, match="paper profile"):
        feature_gen.generate_features(s, tmp_path / "job", filtration=custom)


def test_custom_settings_never_get_the_published_schema():
    from tdq_adapters.preprocessing import FEATURE_SCHEMA
    from tdq_adapters.topo_features import PAPER_BINS, PAPER_FILTRATION
    assert feature_gen.feature_schema(PAPER_BINS, PAPER_FILTRATION) == FEATURE_SCHEMA
    custom = list(PAPER_FILTRATION)
    custom[5] = 3.3
    schema = feature_gen.feature_schema(PAPER_BINS, custom)
    assert schema != FEATURE_SCHEMA and schema.startswith("custom-")


def test_experiment_table_cannot_be_scored_by_published_package(tmp_path):
    import numpy as np
    import pandas as pd
    from tdq_adapters import preprocessing as pp
    table = pd.DataFrame(np.zeros((1, 2754)), columns=[f"raw_{i:04d}" for i in range(2754)])
    with pytest.raises(pp.FeatureInputError, match="missing"):
        pp.validate_features(table, [f"persistent_2.0_{i:02d}" for i in range(1, 4)])


def test_existing_feature_repo_staging_is_not_reused(tmp_path):
    (tmp_path / "job" / "feature_repo").mkdir(parents=True)
    with pytest.raises(ids.StaleOutputError):
        feature_gen.stage_feature_repo(tmp_path / "job")
    with pytest.raises(ids.StaleOutputError):
        feature_gen.generate_features([feature_gen.FeatureSample("a", "1abc", 0, _iface(tmp_path))],
                                      tmp_path / "job", engine="original")


def test_feature_comparison_requires_exact_unique_key_sets():
    import pandas as pd
    from tdq_adapters import step4
    a = pd.DataFrame({"pdb_id": ["x", "y"], "af_model_id": [0, 0], "f": [1.0, 2.0]})
    step4.check_keys(a, a.copy(), "ok")
    with pytest.raises(step4.KeyMismatchError, match="duplicate"):
        step4.check_keys(pd.concat([a, a.iloc[[0]]]), a, "dup")
    with pytest.raises(step4.KeyMismatchError, match="key sets differ"):
        step4.check_keys(a, a.iloc[[0]], "missing")
