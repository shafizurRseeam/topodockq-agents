"""Unit tests for tdq_adapters.topo_features: each checks a rule recovered from the authors' bytecode."""
import math

import numpy as np
import pytest

from tdq_adapters import feature_gen
from tdq_adapters import topo_features as tf

INF = float("inf")


def test_paper_settings_give_2754_features():
    per_pair = (len(tf.PAPER_BINS) - 1) + 6 + 7 * len(tf.PAPER_FILTRATION) + 18 + 18
    assert len(tf.PAPER_FILTRATION) == 33 and per_pair == 306 and 9 * per_pair == 2754


def test_str_to_float_list_accepts_readme_format():
    assert tf.str_to_float_list("[0, 2.  ,  2.25, 10.]") == [0.0, 2.0, 2.25, 10.0]


def test_bin_id_is_inclusive_and_last_match_wins():
    bins = [0, 2.0, 2.25, 2.5]
    assert tf.bin_id(2.0, bins) == 1          # boundary value goes to the later bin
    assert tf.bin_id(2.1, bins) == 1
    assert tf.bin_id(0.0, bins) == 0
    assert tf.bin_id(2.5, bins) == 2


def test_rips_topo_counts_deaths_and_skips_infinite_and_cut():
    ph = [(0, (0.0, INF)), (0, (0.0, 2.1)), (0, (0.0, 2.3)), (0, (0.0, 2.5)), (1, (0.0, INF))]
    out = tf.rips_topo(ph, [0, 2.0, 2.25, 2.5])
    assert out.tolist() == [0.0, 1.0, 1.0]     # 2.5 equals the last bin edge -> skipped (d >= death_cut)


def test_rips_topo_single_bar_gives_zeros():
    assert tf.rips_topo([(0, (0.0, INF))], [0, 2.0, 3.0]).tolist() == [0.0, 0.0]


def test_rips_betti_stats_order_and_cut():
    ph = [(0, (0.0, 3.0)), (0, (0.0, 5.0)), (0, (0.0, INF)), (0, (0.0, 8.0))]
    out = tf.rips_betti_stats(ph, 7)
    assert out.tolist() == [4.0, 1.0, 5.0, 3.0, 8.0, 2.0]   # mean, std, max, min, sum, count


def test_rips_spectra_counts_zero_eigenvalues_and_zero_row_without_edges():
    d = np.full((3, 3), 100.0)
    d[0, 2] = d[2, 0] = 1.5                       # one bipartite edge, one isolated vertex
    out = tf.rips_spectra(d, [1.0, 2.0]).reshape(2, 7)
    assert out[0].tolist() == [0.0] * 7           # no edge below 1.0 -> whole row stays zero
    assert out[1][0] == pytest.approx(2.0) and out[1][6] == 2   # eigenvalues {0, 0, 2}


def test_generate_pts_dis_is_bipartite_with_100_elsewhere():
    atoms = {"AC": [[0, 0, 0], [3, 0, 0]], "BN": [[0, 4, 0]]}
    d, pts = tf.generate_pts_dis(atoms, "C", "N")
    assert d.shape == (3, 3) and d[0, 1] == 100 and d[2, 2] == 100
    assert d[0, 2] == d[2, 0] == 4.0 and d[1, 2] == pytest.approx(5.0)
    assert pts.tolist() == [[0, 0, 0], [3, 0, 0], [0, 4, 0]]


def test_alpha_topo_rules():
    # gudhi alpha values are squared radii: 2*sqrt(v) converts to a diameter-like length.
    def sq(x):
        return (x / 2) ** 2
    ph = [(0, (0.0, INF)),                         # dropped: d > cut and b == 0
          (1, (sq(1.0), sq(9.0))),                 # kept, death clamped to cut 7
          (1, (sq(2.0), sq(2.05))),                # dropped: persistence < 0.1
          (1, (sq(3.0), sq(4.0))),                 # kept
          (2, (sq(1.0), sq(2.0)))]                 # other dimension
    out = tf.alpha_topo(ph, 7, 1)
    p = [6.0, 1.0]
    assert out[0] == pytest.approx(np.mean(p)) and out[4] == pytest.approx(7.0)
    assert out[5] == pytest.approx(1.0) and out[6] == pytest.approx(7.0)      # longest bar birth, death
    assert out[9] == pytest.approx(3.0) and out[14] == pytest.approx(7.0) and out[17] == 2


def test_alpha_topo_single_bar_gives_zeros():
    assert tf.alpha_topo([(0, (0.0, INF))], 7, 1).tolist() == [0.0] * 18


def test_parse_pdb_reads_atom_lines_only(tmp_path):
    def line(rec, chain, el, x):
        return f"{rec:<6}{1:>5}  CA  ALA {chain}{1:>4}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {el:>2}\n"
    p = tmp_path / "x.pdb"
    p.write_text(line("ATOM", "A", "C", 1.0) + line("HETATM", "B", "C", 2.0) + line("ATOM", "B", "S", 3.0)
                 + line("ATOM", "C", "C", 4.0) + line("ATOM", "B", "N", 5.0))
    atoms = tf.parse_pdb(p)
    assert atoms["AC"] == [[1.0, 0.0, 0.0]] and atoms["BN"] == [[5.0, 0.0, 0.0]] and atoms["BC"] == []


def test_feature_gen_rejects_wrongly_named_interface(tmp_path):
    f = tmp_path / "wrong_name.pdb"
    f.write_text("END\n")
    rec = feature_gen.generate_one("compat", feature_gen.FeatureSample("x", "4k38", 44, str(f)), tmp_path)
    assert rec["status"] == "failed" and "must be named" in rec["error"]
