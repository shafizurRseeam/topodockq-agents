"""Unit tests for the fixed validation rules in tdq_adapters.structures (no ProDy, no downloads)."""
from pathlib import Path

from tdq_adapters import data_prep
from tdq_adapters import structures as st


def atom(serial, name, resname, chain, resnum, x, element, record="ATOM", altloc=" ", icode=" "):
    return (f"{record:<6}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}{resnum:>4}{icode}   "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}  \n")


def good_lines():
    return [atom(1, "N", "ALA", "A", 1, 0.0, "N"), atom(2, "CA", "ALA", "A", 1, 1.0, "C"),
            atom(3, "O", "ALA", "A", 1, 2.0, "O"), atom(4, "N", "GLY", "B", 1, 4.0, "N"),
            atom(5, "CA", "GLY", "B", 1, 5.0, "C")]


def write(tmp_path: Path, lines, name="s.pdb") -> Path:
    p = tmp_path / name
    p.write_text("".join(lines) + "END\n", encoding="utf-8")
    return p


def codes(result):
    return {i["code"] for i in result["issues"]}


def test_fixed_columns_parse_like_the_authors_reader(tmp_path):
    rec = st.parse_records(write(tmp_path, good_lines()))[1]
    assert (rec["chain"], rec["element"], rec["name"], rec["resnum"]) == ("A", "C", "CA", "1")


def test_valid_structure_passes(tmp_path):
    r = st.validate_structure(write(tmp_path, good_lines()))
    assert r["ok"] and r["issues"] == []
    assert r["stats"]["cno_atoms_by_chain"] == {"A": 3, "B": 2}


def test_missing_file(tmp_path):
    r = st.validate_structure(tmp_path / "nope.pdb")
    assert not r["ok"] and codes(r) == {"E_FILE_UNREADABLE"}


def test_no_atom_records(tmp_path):
    assert "E_NO_ATOM_RECORDS" in codes(st.validate_structure(write(tmp_path, ["REMARK x\n"])))


def test_chain_ids_must_be_a_and_b(tmp_path):
    r = st.validate_structure(write(tmp_path, good_lines()), protein_chain="H", peptide_chain="B")
    assert not r["ok"] and "E_CHAIN_IDS" in codes(r)


def test_missing_peptide_chain(tmp_path):
    r = st.validate_structure(write(tmp_path, good_lines()[:3]))
    assert not r["ok"] and "E_CHAIN_MISSING" in codes(r)


def test_chain_without_cno_atoms(tmp_path):
    lines = good_lines()[:3] + [atom(4, "SG", "CYS", "B", 1, 4.0, "S")]
    r = st.validate_structure(write(tmp_path, lines))
    assert not r["ok"] and "E_NO_CNO_ATOMS" in codes(r)


def test_same_rule_is_reported_as_empty_interface_after_extraction(tmp_path):
    lines = good_lines()[:3] + [atom(4, "SG", "CYS", "B", 1, 4.0, "S")]
    assert "E_EMPTY_INTERFACE" in codes(st.validate_structure(write(tmp_path, lines), stage="interface"))


def test_blank_element_column_is_an_error(tmp_path):
    lines = good_lines()
    lines[1] = lines[1][:76] + "  \n"
    r = st.validate_structure(write(tmp_path, lines))
    assert not r["ok"] and "E_MISSING_ELEMENT" in codes(r)


def test_modified_residue_is_a_warning_but_water_and_ions_are_not(tmp_path):
    lines = good_lines() + [atom(6, "C1", "PTR", "B", 2, 6.0, "C", record="HETATM"),
                            atom(7, "O", "HOH", "B", 3, 7.0, "O", record="HETATM"),
                            atom(8, "ZN", "ZN", "A", 9, 8.0, "ZN", record="HETATM")]
    r = st.validate_structure(write(tmp_path, lines))
    assert r["ok"]
    mods = [i for i in r["issues"] if i["code"] == "W_MODIFIED_RESIDUE"]
    assert len(mods) == 1 and mods[0]["count"] == 1 and mods[0]["residues"][0][1] == "PTR"


def test_altloc_insertion_code_and_extra_chain_warnings(tmp_path):
    lines = good_lines() + [atom(6, "CB", "ALA", "A", 1, 1.5, "C", altloc="B"),
                            atom(7, "N", "SER", "A", 52, 3.0, "N", icode="A"),
                            atom(8, "N", "LYS", "C", 1, 9.0, "N")]
    r = st.validate_structure(write(tmp_path, lines))
    assert r["ok"] and {"W_ALTLOC", "W_INSERTION_CODE", "W_EXTRA_CHAINS"} <= codes(r)


def test_compare_interfaces_ignores_serials_and_remarks_but_not_order(tmp_path):
    a = write(tmp_path, good_lines(), "a.pdb")
    renumbered = [l[:6] + f"{int(l[6:11]) + 700:>5}" + l[11:] for l in good_lines()]
    b = write(tmp_path, ["REMARK selection\n"] + renumbered, "b.pdb")
    c = write(tmp_path, list(reversed(good_lines())), "c.pdb")
    ab, ac = st.compare_interfaces(a, b), st.compare_interfaces(a, c)
    assert ab["identical_atoms"] and ab["identical_feature_input"] and not ab["identical_bytes"]
    assert ac["identical_atoms"] and not ac["identical_feature_input"]


def test_interface_filename_matches_authors_feature_code():
    assert data_prep.interface_filename("4k38", 44) == "4k38_ranked_44_sp_interface.pdb"


def test_invalid_input_is_excluded_before_extraction(tmp_path):
    bad = write(tmp_path, good_lines()[:3])
    rec = data_prep.prepare_one(data_prep.Sample("x", "x", 0, str(bad)), tmp_path / "out")
    assert rec["status"] == "excluded" and "extraction" not in rec
    assert "E_CHAIN_MISSING" in {i["code"] for i in rec["issues"]}
