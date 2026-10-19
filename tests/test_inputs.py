"""Unit tests for tdq_adapters.inputs (input discovery), on small synthetic folders."""
from pathlib import Path

import pytest

from tdq_adapters import ids, inputs


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("END\n")
    return p


def test_complex_dirs_prefer_superposed_and_ignore_other_files(tmp_path):
    for m in range(3):
        _touch(tmp_path / "1ABC" / f"ranked_{m}.pdb")
        _touch(tmp_path / "1ABC" / f"ranked_{m}_sp.pdb")
        _touch(tmp_path / "1ABC" / f"ranked_{m}_sp_interface.pdb")
    _touch(tmp_path / "2XYZ" / "ranked_0.pdb")
    samples, rep = inputs.discover(tmp_path)
    assert rep["layout"] == "complex_dirs" and len(samples) == 4
    assert [s.sample_id for s in samples] == ["1abc_ranked_0", "1abc_ranked_1", "1abc_ranked_2", "2xyz_ranked_0"]
    assert samples[0].input_pdb.endswith("ranked_0_sp.pdb") and samples[3].input_pdb.endswith("ranked_0.pdb")
    assert rep["superposed_sp_files_used"] == 3 and rep["raw_files_used"] == 1


def test_selection_by_complex_and_models(tmp_path):
    for c in ("1ABC", "2XYZ", "3QRS"):
        for m in range(5):
            _touch(tmp_path / c / f"ranked_{m}.pdb")
    s, _ = inputs.discover(tmp_path, complexes=["2xyz"], models=[1, 3])
    assert [x.sample_id for x in s] == ["2xyz_ranked_1", "2xyz_ranked_3"]
    s, _ = inputs.discover(tmp_path, max_complexes=2, max_models_per_complex=2)
    assert [x.sample_id for x in s] == ["1abc_ranked_0", "1abc_ranked_1", "2xyz_ranked_0", "2xyz_ranked_1"]
    with pytest.raises(inputs.InputError, match="not found"):
        inputs.discover(tmp_path, complexes=["9ZZZ"])
    with pytest.raises(inputs.InputError, match="matched no structures"):
        inputs.discover(tmp_path, models=[42])


def test_flat_files_parse_ranked_names_and_sanitize_others(tmp_path):
    _touch(tmp_path / "4k38_ranked_44_sp.pdb")
    _touch(tmp_path / "my complex (v2).pdb")
    s, rep = inputs.discover(tmp_path)
    assert rep["layout"] == "flat_files"
    assert {(x.pdb_id, x.model_id) for x in s} == {("4k38", 44), ("my_complex_v2", 0)}


def test_manifest_relative_paths_chains_and_missing_files_kept(tmp_path):
    _touch(tmp_path / "data" / "a.pdb")
    (tmp_path / "m.csv").write_text("pdb_id,model_id,input_pdb,peptide_chain\n1abc,0,data/a.pdb,P\n1abc,1,data/gone.pdb,\n")
    s, rep = inputs.discover(tmp_path / "m.csv")
    assert rep["layout"] == "manifest_csv" and rep["missing_files"] == 1
    assert Path(s[0].input_pdb) == tmp_path / "data" / "a.pdb" and s[0].peptide_chain == "P"
    assert s[1].peptide_chain == "B"                      # default when the column is empty


def test_manifest_rejects_duplicates_bad_ids_and_missing_columns(tmp_path):
    (tmp_path / "dup.csv").write_text("pdb_id,af_model_id,input_pdb\n1abc,0,a.pdb\n1abc,0,b.pdb\n")
    with pytest.raises(ids.DuplicateOutputError):
        inputs.discover(tmp_path / "dup.csv")
    (tmp_path / "bad.csv").write_text("pdb_id,af_model_id,input_pdb\n../x,0,a.pdb\n")
    with pytest.raises(ids.UnsafeIdentifierError):
        inputs.discover(tmp_path / "bad.csv")
    (tmp_path / "cols.csv").write_text("pdb_id,input_pdb\n1abc,a.pdb\n")
    with pytest.raises(inputs.InputError, match="lacks columns"):
        inputs.discover(tmp_path / "cols.csv")


def test_manifest_roundtrip(tmp_path):
    for m in range(2):
        _touch(tmp_path / "in" / "1ABC" / f"ranked_{m}.pdb")
    s, _ = inputs.discover(tmp_path / "in")
    back = inputs.read_manifest(inputs.write_manifest(s, tmp_path / "structures.csv"))
    assert [(x.sample_id, Path(x.input_pdb)) for x in back] == [(x.sample_id, Path(x.input_pdb)) for x in s]


def test_missing_source_is_an_input_error(tmp_path):
    with pytest.raises(inputs.InputError, match="does not exist"):
        inputs.discover(tmp_path / "nothing_here")


def test_single_complex_folder_is_one_complex(tmp_path):
    for m in range(3):
        _touch(tmp_path / "1AWR" / f"ranked_{m}.pdb")
        _touch(tmp_path / "1AWR" / f"ranked_{m}_sp.pdb")
    _touch(tmp_path / "1AWR" / "ranked_0_sp_interface.pdb")
    s, rep = inputs.discover(tmp_path / "1AWR")
    assert rep["layout"] == "complex_dir" and rep["complexes_selected"] == 1
    assert [x.sample_id for x in s] == ["1awr_ranked_0", "1awr_ranked_1", "1awr_ranked_2"]
    assert all(x.input_pdb.endswith("_sp.pdb") for x in s)
