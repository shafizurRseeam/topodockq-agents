"""Python 3.13 compatibility replacement for the authors' TopoDockQ feature generator.

Derived from the TopoDockQ feature code by Xuhang Dai, Rui Wang and Yingkai Zhang,
distributed under the MIT License (Copyright (c) 2025; full text in
authors_code/TopoDockQ/LICENSE).

The authors ship their feature generator only as Python 3.8 bytecode
(main.pyc, src/read_pdb.pyc, src/get_PHinputs.pyc, src/get_PHcomplex.pyc,
src/get_MLfeature.pyc in authors_code/TopoDockQ). Those files run unchanged in the
separate Python 3.8 environment (``.venv-feature38``), and that run is the
reference. This module exists only so the same computation can run in the main
environment; it must match the reference numerically (checked in step4.py).

It mirrors the authors' code structure and logic as recovered by disassembly,
one function per original method, with the original names in comments. The one
deliberate change: the per-pair Python double loop that fills the distance matrix
(get_PHinputs.generate_pts_dis) is vectorised. It computes the same Euclidean
norms of the same coordinate differences.

Command line, identical to ``python -m main`` in the authors' repository:

    python -m tdq_adapters.topo_features --pdb_id 4k38 --model_id 44 \\
        --bins "[0, 2., ..., 10.]" --filtration "[2., ..., 10.]" \\
        --file_path ./data/interface_files --saving_path ./feature
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gudhi
import numpy as np

# Paper settings (authors' README, "same feature generation process as in the paper").
PAPER_FILTRATION = [float(x) for x in np.round(np.arange(2.0, 10.25, 0.25), 2)]
PAPER_BINS = [0.0] + PAPER_FILTRATION
RIPS_MAX_EDGE = 7          # main.pyc L46, L49: hard-coded 7
ALPHA_CUT = 7              # main.pyc L51-52: hard-coded 7
ATOM_TYPES = ["C", "N", "O"]
CHAINS = ["A", "B"]


def str_to_float_list(input_str: str) -> list[float]:
    """main.pyc str_to_float_list."""
    try:
        return list(map(float, input_str.strip("[]").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("List must be a series of floats separated by commas.")


def interface_path(file_path: str | Path, pdb_id: str, model_id: int) -> Path:
    """main.pyc L32-33."""
    return Path(f"{file_path}/{pdb_id}_ranked_{model_id}_sp_interface.pdb")


# ------------------------------------------------------------------ src/read_pdb.pyc: PDBParser
def parse_pdb(filepath: str | Path, atom_types=ATOM_TYPES, chains=CHAINS) -> dict[str, list[list[float]]]:
    """PDBParser.parse + _extract_and_store_atom_data: ATOM lines only, chain col 22, element cols 77-78."""
    atoms: dict[str, list[list[float]]] = {f"{c}{a}": [] for c in chains for a in atom_types}
    with open(filepath, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                chain_id = line[21].strip()
                element = line[76:78].strip()
                if chain_id in chains and element in atom_types:
                    int(line[6:11].strip())                       # atom_number, parsed (and validated) as in original
                    int(line[22:26].strip())                      # residue_number
                    atoms[f"{chain_id}{element}"].append(
                        [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
    return atoms


# ------------------------------------------------------------------ src/get_PHinputs.pyc: GetPHInputs
def generate_pts_dis(atoms, chain_A_ele: str, chain_B_ele: str) -> tuple[np.ndarray, np.ndarray]:
    """GetPHInputs.generate_pts_dis: bipartite distance matrix (other entries 100) and the joint point set."""
    chain_A_pts = atoms.get(f"A{chain_A_ele}", [])
    chain_B_pts = atoms.get(f"B{chain_B_ele}", [])
    alpha_all_pts = np.asarray(chain_A_pts + chain_B_pts)
    nA, nB = len(chain_A_pts), len(chain_B_pts)
    rips_dis_matrix = np.ones((nA + nB, nA + nB)) * 100
    if nA and nB:
        a, b = np.asarray(chain_A_pts), np.asarray(chain_B_pts)
        dis = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)      # vectorised original double loop
        rips_dis_matrix[:nA, nA:] = dis
        rips_dis_matrix[nA:, :nA] = dis.T
    return rips_dis_matrix, alpha_all_pts


# ------------------------------------------------------------------ src/get_PHcomplex.pyc: GetGudhi
def get_rips(dis_matrix, max_edge_length):
    """GetGudhi.get_rips (default simplex tree dimension, as in the original)."""
    if dis_matrix is None:
        dis_matrix = np.array([[0, 0], [0, 0], [0, 0]])
    rips_complex = gudhi.RipsComplex(distance_matrix=dis_matrix, max_edge_length=max_edge_length)
    return rips_complex.create_simplex_tree().persistence()


def get_alpha(pts_matrix):
    """GetGudhi.get_alpha."""
    if pts_matrix is None:
        pts_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    alpha_complex = gudhi.AlphaComplex(points=pts_matrix)
    return alpha_complex.create_simplex_tree().persistence()


# ------------------------------------------------------------------ src/get_MLfeature.pyc: MLFeature
def bin_id(x, B):
    """MLFeature.bin_id: last bin i with B[i] <= x <= B[i+1] (both ends inclusive)."""
    idx = None
    for i in range(len(B) - 1):
        if x >= B[i] and x <= B[i + 1]:
            idx = i
    return idx


def rips_topo(PH, bins=(0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    """MLFeature.rips_topo: counts of bar deaths per bin; deaths >= bins[-1] (incl. inf) skipped."""
    bin_idx = len(bins) - 1
    death_cut = bins[-1]
    rips_topo_bin_feature = np.zeros(bin_idx)
    if len(PH) == 1:
        rips_topo_bin_feature = np.zeros(bin_idx)
    else:
        for simplex in PH:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d >= death_cut:
                continue
            did = bin_id(d, bins)
            rips_topo_bin_feature[did] += 1
    return rips_topo_bin_feature.flatten()


def rips_betti_stats(PH, cut):
    """MLFeature.rips_betti_stats: [mean, std, max, min, sum, count] of finite H0 deaths < cut."""
    rips_betti_stats_feature = np.zeros(6)
    betti0_death = []
    if len(PH) == 1:
        feature = [0] * 6                                    # original assigns this and never uses it
    else:
        for simplex in PH:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if dim == 0 and d < cut:
                betti0_death.append(d)
        betti0_death = np.asarray(betti0_death)
        if betti0_death.shape[0] > 0:
            rips_betti_stats_feature[0] = np.mean(betti0_death)
            rips_betti_stats_feature[1] = np.std(betti0_death)
            rips_betti_stats_feature[2] = np.max(betti0_death)
            rips_betti_stats_feature[3] = np.min(betti0_death)
            rips_betti_stats_feature[4] = np.sum(betti0_death)
            rips_betti_stats_feature[5] = betti0_death.shape[0]
    return rips_betti_stats_feature.flatten()


def rips_spectra(rips_dis_matrix, filtrations=(2.0, 3.0, 4.0, 5.0, 6.0)):
    """MLFeature.rips_spectra: graph-Laplacian eigenvalue stats per filtration value."""
    rips_speactra_feature = np.zeros((len(filtrations), 7))
    for filt_idx, filt in enumerate(filtrations):
        Laplacian = np.zeros((rips_dis_matrix.shape[0], rips_dis_matrix.shape[1]))
        Laplacian[np.where(rips_dis_matrix < filt)] = -1
        Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))
        eigens = np.sort(np.linalg.eigvalsh(Laplacian))
        eigens_no_zeros = eigens[eigens > 1e-05]
        if len(eigens_no_zeros) > 0:
            rips_speactra_feature[filt_idx][0] = np.sum(eigens_no_zeros)
            rips_speactra_feature[filt_idx][1] = np.min(eigens_no_zeros)
            rips_speactra_feature[filt_idx][2] = np.max(eigens_no_zeros)
            rips_speactra_feature[filt_idx][3] = np.mean(eigens_no_zeros)
            rips_speactra_feature[filt_idx][4] = np.std(eigens_no_zeros)
            rips_speactra_feature[filt_idx][5] = np.var(eigens_no_zeros)
            rips_speactra_feature[filt_idx][6] = np.sum(np.abs(eigens) <= 1e-05)
    return rips_speactra_feature.flatten()


def alpha_topo(PH, cut, degree):
    """MLFeature.alpha_topo: 18 barcode stats of dimension ``degree``; alpha values -> 2*sqrt."""
    alpha_topo_stats_feature = np.zeros(18)
    betti_birth, betti_death = [], []
    if len(PH) == 1:
        alpha_topo_stats_feature = np.zeros(18)
    else:
        for simplex in PH:
            dim = int(simplex[0])
            b = float(2 * np.sqrt(simplex[1][0]))
            d = float(2 * np.sqrt(simplex[1][1]))
            if d > cut and b == 0:
                continue
            if d > cut and b > 0:
                d = cut
            if d - b < 0.1:
                continue
            if dim == degree:
                betti_birth.append(b)
                betti_death.append(d)
        betti_birth = np.asarray(betti_birth)
        betti_death = np.asarray(betti_death)
        if betti_death.shape[0] > 0:
            p = betti_death - betti_birth
            alpha_topo_stats_feature[0] = np.mean(p)
            alpha_topo_stats_feature[1] = np.std(p)
            alpha_topo_stats_feature[2] = np.max(p)
            alpha_topo_stats_feature[3] = np.min(p)
            alpha_topo_stats_feature[4] = np.sum(p)
            alpha_topo_stats_feature[5] = betti_birth[np.argmax(p)]
            alpha_topo_stats_feature[6] = betti_death[np.argmax(p)]
            alpha_topo_stats_feature[7] = np.mean(betti_birth)
            alpha_topo_stats_feature[8] = np.std(betti_birth)
            alpha_topo_stats_feature[9] = np.max(betti_birth)
            alpha_topo_stats_feature[10] = np.min(betti_birth)
            alpha_topo_stats_feature[11] = np.sum(betti_birth)
            alpha_topo_stats_feature[12] = np.mean(betti_death)
            alpha_topo_stats_feature[13] = np.std(betti_death)
            alpha_topo_stats_feature[14] = np.max(betti_death)
            alpha_topo_stats_feature[15] = np.min(betti_death)
            alpha_topo_stats_feature[16] = np.sum(betti_death)
            alpha_topo_stats_feature[17] = betti_death.shape[0]
    return alpha_topo_stats_feature.flatten()


# ------------------------------------------------------------------ main.pyc: main
def compute_features(pdb_filepath: str | Path, bins=PAPER_BINS, filtration=PAPER_FILTRATION,
                     verbose: bool = False) -> np.ndarray:
    """main.pyc L36-57 for one interface file. Returns shape (1, 9 * (len(bins)-1 + 6 + 7*len(filtration) + 36))."""
    atoms = parse_pdb(pdb_filepath)
    all_features = []
    for chain_A_ele in ATOM_TYPES:
        for chain_B_ele in ATOM_TYPES:
            rips_dis_matrix, alpha_all_pts = generate_pts_dis(atoms, chain_A_ele, chain_B_ele)
            rips_PH = get_rips(rips_dis_matrix, RIPS_MAX_EDGE)
            alpha_PH = get_alpha(alpha_all_pts)
            tmp_feature = np.concatenate([
                rips_topo(rips_PH, bins),
                rips_betti_stats(rips_PH, RIPS_MAX_EDGE),
                rips_spectra(rips_dis_matrix, filtration),
                alpha_topo(alpha_PH, ALPHA_CUT, 1),
                alpha_topo(alpha_PH, ALPHA_CUT, 2),
            ])
            if verbose:
                print(f"{Path(pdb_filepath).stem}, chain_A_ele: {chain_A_ele}, chain_B_ele: {chain_B_ele}", "Done")
            all_features.append(tmp_feature.tolist())
    return np.asarray(all_features).reshape(1, -1)


def main(pdb_id, model_id, bins, filtration, file_path, saving_path) -> Path:
    """main.pyc main: same inputs, same output file name and shape."""
    pdb = f"{pdb_id}_ranked_{model_id}_sp_interface"
    features = compute_features(interface_path(file_path, pdb_id, model_id), bins, filtration, verbose=True)
    print(f"Feature shape is {features.shape}")
    out = Path(f"{saving_path}/feature_{pdb}.npy")
    np.save(out, features)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features from PDB files.")
    parser.add_argument("--pdb_id", type=str, help="PDB ID, for example: 2fns")
    parser.add_argument("--model_id", type=int, help="Model ID, for example: 3")
    parser.add_argument("--bins", type=str_to_float_list, help="Filtration bins (for betti0)")
    parser.add_argument("--filtration", type=str_to_float_list, help="Filtration (for spectra)")
    parser.add_argument("--file_path", type=str, help="File path, for example: ./data/interface_files")
    parser.add_argument("--saving_path", type=str, help="Saving path, for example: ./feature")
    args = parser.parse_args()
    main(args.pdb_id, args.model_id, args.bins, args.filtration, args.file_path, args.saving_path)
