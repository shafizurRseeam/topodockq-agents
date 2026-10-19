"""Read-only PDB inspection, validation rules, and interface comparison.

No scientific computation happens here. The rules only check that a structure can
be consumed by the authors' code as written:

* ``src/extract_interface.py`` selects ``chain X and protein`` atoms with ProDy.
* The feature code (``src/read_pdb.pyc``, disassembled) reads only lines starting
  with ``ATOM``, takes the chain from column 22 and the element from columns 77-78,
  keeps C, N and O, and expects the protein as chain A and the peptide as chain B.

Validation rules are fixed and declared here, before any data is seen. An ERROR
excludes the sample; a WARNING keeps it and is recorded. Nothing is excluded for
being difficult or for its score.

Chain roles: chain IDs alone do not prove which chain is the protein and which the
peptide. The assignment is recorded with residue counts and handled by fixed rules:
  * confirmed by the user (explicit chain columns in a manifest, or a request that
    states the roles): processed normally, no warning;
  * not confirmed and implausible (peptide chain longer than the protein chain):
    WARNING W_CHAIN_ROLE_SUSPICIOUS, never an automatic exclusion. In strict mode the
    ranking of a complex whose top-1 candidate carries it needs confirmation;
  * declared reversed (protein not chain A, peptide not chain B): ERROR E_CHAIN_IDS,
    because the authors' feature code reads the protein from chain A and the peptide
    from chain B; the structure needs a corrected chain assignment.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

FEATURE_ELEMENTS = ("C", "N", "O")
FEATURE_CHAINS = {"protein": "A", "peptide": "B"}
WATER = {"HOH", "WAT", "DOD", "H2O"}
IONS = {"NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD", "HG", "IOD", "BR", "LI",
        "SR", "BA", "CS", "RB"}

RULES: dict[str, tuple[str, str]] = {
    # code: (severity, meaning)
    "E_FILE_UNREADABLE": ("ERROR", "file is missing or cannot be read as text"),
    "E_NO_ATOM_RECORDS": ("ERROR", "no ATOM records"),
    "E_CHAIN_IDS": ("ERROR", "protein must be chain A and peptide chain B: the authors' feature code reads only A and B"),
    "E_CHAIN_MISSING": ("ERROR", "protein or peptide chain has no ATOM records"),
    "E_NO_CNO_ATOMS": ("ERROR", "protein or peptide chain has no C, N or O ATOM atoms, so a feature point cloud would be empty"),
    "E_MISSING_ELEMENT": ("ERROR", "ATOM records in chain A or B with a blank element column; the feature code would drop them silently"),
    "E_EXTRACTION_FAILED": ("ERROR", "the authors' extract_interface.py exited with an error"),
    "E_EMPTY_INTERFACE": ("ERROR", "the extracted interface has no C, N or O ATOM atoms for the protein or the peptide"),
    "W_MODIFIED_RESIDUE": ("WARNING", "HETATM residues (not water or ions) inside chain A or B; the feature code ignores HETATM, so these atoms are left out"),
    "W_ALTLOC": ("WARNING", "atoms with alternate locations; the feature code keeps every alternate, duplicating atoms"),
    "W_INSERTION_CODE": ("WARNING", "residue insertion codes; extract_interface.py selects by residue number only"),
    "W_EXTRA_CHAINS": ("WARNING", "chains other than the protein and peptide are present and will be ignored"),
    "W_CHAIN_ROLE_SUSPICIOUS": ("WARNING", "the peptide chain has more residues than the protein chain and the chain "
                                           "roles were not confirmed; check that protein and peptide are not swapped"),
}


def issue(code: str, **detail: Any) -> dict[str, Any]:
    severity, meaning = RULES[code]
    return {"code": code, "severity": severity, "meaning": meaning, **detail}


def parse_records(path: Path | str) -> list[dict[str, Any]]:
    """ATOM/HETATM records with the fixed-column fields the authors' code relies on."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            out.append({
                "record": rec.strip(), "name": line[12:16].strip(), "altloc": line[16:17].strip(),
                "resname": line[17:20].strip(), "chain": line[21:22].strip(), "resnum": line[22:26].strip(),
                "icode": line[26:27].strip(), "xyz": line[30:54], "element": line[76:78].strip(),
            })
    return out


def inspect_structure(path: Path | str) -> dict[str, Any]:
    recs = parse_records(path)
    atom = [r for r in recs if r["record"] == "ATOM"]
    het = [r for r in recs if r["record"] == "HETATM"]
    return {
        "atom_records_by_chain": dict(Counter(r["chain"] for r in atom)),
        "hetatm_records_by_chain": dict(Counter(r["chain"] for r in het)),
        "cno_atoms_by_chain": dict(Counter(r["chain"] for r in atom if r["element"] in FEATURE_ELEMENTS)),
        "residues_by_chain": {c: len({(r["resnum"], r["icode"]) for r in atom if r["chain"] == c})
                              for c in sorted({r["chain"] for r in atom})},
    }


def validate_structure(path: Path | str, protein_chain: str = "A", peptide_chain: str = "B",
                       stage: str = "input", chains_confirmed: bool = False) -> dict[str, Any]:
    """Apply the fixed rules. ``stage`` is "input" (before extraction) or "interface" (after).

    ``chains_confirmed``: the user confirmed which chain is the protein and which the peptide;
    the chain-role plausibility warning is then not raised.
    """
    path = Path(path)
    issues: list[dict[str, Any]] = []
    try:
        recs = parse_records(path)
    except OSError as exc:
        return {"path": str(path), "stage": stage, "ok": False,
                "issues": [issue("E_FILE_UNREADABLE", error=str(exc))], "stats": {}}

    atom = [r for r in recs if r["record"] == "ATOM"]
    if not atom:
        issues.append(issue("E_NO_ATOM_RECORDS"))
    if (protein_chain, peptide_chain) != (FEATURE_CHAINS["protein"], FEATURE_CHAINS["peptide"]):
        issues.append(issue("E_CHAIN_IDS", protein_chain=protein_chain, peptide_chain=peptide_chain))

    chains = {protein_chain, peptide_chain}
    present = {r["chain"] for r in atom}
    for role, ch in (("protein", protein_chain), ("peptide", peptide_chain)):
        if ch not in present:
            issues.append(issue("E_CHAIN_MISSING", role=role, chain=ch))
        elif not any(r["chain"] == ch and r["element"] in FEATURE_ELEMENTS for r in atom):
            issues.append(issue("E_EMPTY_INTERFACE" if stage == "interface" else "E_NO_CNO_ATOMS",
                                role=role, chain=ch))

    blank = sum(1 for r in atom if r["chain"] in chains and not r["element"])
    if blank:
        issues.append(issue("E_MISSING_ELEMENT", atoms=blank))

    modified = sorted({(r["chain"], r["resname"], r["resnum"]) for r in recs
                       if r["record"] == "HETATM" and r["chain"] in chains
                       and r["resname"] not in WATER and r["resname"] not in IONS})
    if modified:
        issues.append(issue("W_MODIFIED_RESIDUE", residues=[list(m) for m in modified[:20]], count=len(modified)))
    altloc = sum(1 for r in atom if r["chain"] in chains and r["altloc"])
    if altloc:
        issues.append(issue("W_ALTLOC", atoms=altloc))
    icode = len({(r["chain"], r["resnum"], r["icode"]) for r in atom if r["chain"] in chains and r["icode"]})
    if icode:
        issues.append(issue("W_INSERTION_CODE", residues=icode))
    extra = sorted(present - chains)
    if extra:
        issues.append(issue("W_EXTRA_CHAINS", chains=extra))
    if stage == "input" and not chains_confirmed and protein_chain in present and peptide_chain in present:
        res = {c: len({(r["resnum"], r["icode"]) for r in atom if r["chain"] == c}) for c in chains}
        if res[peptide_chain] > res[protein_chain]:
            issues.append(issue("W_CHAIN_ROLE_SUSPICIOUS", protein_residues=res[protein_chain],
                                peptide_residues=res[peptide_chain]))

    return {"path": str(path), "stage": stage, "ok": not any(i["severity"] == "ERROR" for i in issues),
            "issues": issues, "stats": inspect_structure(path) if recs else {}}


def atom_table(path: Path | str) -> dict[tuple, tuple]:
    """(chain, resnum+icode, atom name, altloc) -> (record, xyz text, element)."""
    return {(r["chain"], r["resnum"] + r["icode"], r["name"], r["altloc"]): (r["record"], r["xyz"], r["element"])
            for r in parse_records(path)}


def _feature_view(path: Path | str) -> list[tuple]:
    """What the authors' feature code reads: ATOM lines, in file order, as (chain, element, xyz)."""
    return [(r["chain"], r["element"], r["xyz"]) for r in parse_records(path) if r["record"] == "ATOM"]


def compare_interfaces(ours: Path | str, reference: Path | str) -> dict[str, Any]:
    """Compare two interface files at three levels.

    identical_atoms         same atoms with the same fields, in any order
    identical_feature_input same ATOM chain/element/coordinates in the same order: the
                            authors' feature code receives exactly the same input
    identical_bytes         byte-for-byte equal files (serial numbers, REMARK/TER/END lines)
    """
    a, b = atom_table(ours), atom_table(reference)
    shared = set(a) & set(b)
    mismatched = sum(1 for k in shared if a[k] != b[k])
    return {"atoms_ours": len(a), "atoms_reference": len(b), "only_ours": len(set(a) - set(b)),
            "only_reference": len(set(b) - set(a)), "field_mismatches": mismatched,
            "identical_atoms": len(a) == len(b) == len(shared) and mismatched == 0,
            "identical_feature_input": _feature_view(ours) == _feature_view(reference),
            "identical_bytes": Path(ours).read_bytes() == Path(reference).read_bytes()}
