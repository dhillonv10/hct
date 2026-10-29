"""Invariants that, if broken, silently invalidate results.   python -m guidefam.tests.test_basic"""
import bz2
import io
import pickle

import numpy as np
import pandas as pd

from .. import cv, data as D, metrics as M, sirna as SI
from ..safepickle import load_strict


def test_reverse_complement():
    assert D.rc("ACGT") == "ACGT" and D.rc("AAAC") == "GTTT" and D.rc("acgu") == "ACGT"


def test_strict_loader_refuses_code():
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))
    buf = io.BytesIO()
    with bz2.open(buf, "wb") as f:
        pickle.dump(Evil(), f)
    open("/tmp/_evil.bz2", "wb").write(buf.getvalue())
    try:
        load_strict("/tmp/_evil.bz2")
    except pickle.UnpicklingError:
        return
    raise AssertionError("strict loader executed or accepted a hostile pickle")


def test_within_gene_metric_ignores_gene_offsets():
    rng = np.random.default_rng(0)
    g = np.repeat(list("abcd"), 100)
    y = rng.normal(size=400)
    p = y + rng.normal(size=400) * 0.5
    base = M.within_group(pd.DataFrame({"gene": g, "act": y, "p": p}), "gene", "act", "p")
    shifted = M.within_group(pd.DataFrame({"gene": g, "act": y, "p": p + np.repeat([0, 5, -3, 9], 100)}), "gene", "act", "p")
    assert abs(base - shifted) < 1e-9


def test_target_normalisation_uses_only_rows_given():
    d = pd.DataFrame({"gene": ["a"] * 6 + ["b"] * 6, "y": [-3, -2, -1, 0, 0, 0, -9, -6, -3, 0, 0, 0.0]})
    d["source"] = "hek"
    t = cv.activity_target(d)
    assert (t[:3] > t[3:6].max()).all() and (t[6:9] > t[9:12].max()).all()          # more depleted -> more active, within each gene
    t_only_a = cv.activity_target(d[d.gene == "a"])
    assert np.allclose(t[:6], t_only_a)                                               # gene b's labels cannot influence gene a's target


def test_sirna_rules_and_groups():
    assert SI.rc_dna("AUGC") == "GCAT"
    w = pd.Series(["A" * 10 + "CGTACGTACGTACGTACGTACGTACGTA", "G" * 5 + "CGTACGTACGTACGTACGTACGTACGTA" + "T" * 5, "TTTTTTTTTTGGGGGGGGGGCCCCCCCCCCAAAAAAAAAA"])
    g = SI.target_groups(w, k=12)
    assert g[0] == g[1] and g[2] != g[1]


def test_sirna_binary_conventions():
    d = pd.DataFrame({"label": [0.2, 0.8, 0.71, 0.7], "ybin_file": [0, 1, 1, 0]})
    assert SI.binary(d, "inter").tolist() == [0, 1, 1, 0] and SI.binary(d, "intra").tolist() == [0, 1, 1, 0]
    d["label"] = [0.6, 0.6, 0.6, 0.6]
    assert SI.binary(d, "inter").sum() == 0 and SI.binary(d, "intra").sum() == 2


def test_official_tiger_encoding():
    import torch
    from .. import official_tiger as OT
    d = pd.DataFrame({"5p_context": ["ACGTAC", "G"], "target_seq": ["A" * 23, "C" * 23]})
    t, g = OT.encode(d)
    assert t.shape == (2, 26, 4) and g.shape == (2, 26, 4)
    assert t[0, 2].tolist() == [0, 0, 0, 1] or t[0, 2].tolist() == [0, 1, 0, 0]          # 3rd flank nt of row 0 is 'C' (last 3 of ACGTAC = TAC -> idx 2 = C)
    assert torch.equal(t[0, 3:], torch.tensor([[1., 0, 0, 0]] * 23))                      # target A's
    assert torch.equal(g[0, 3:], torch.tensor([[0., 0, 0, 1]] * 23)) and g[0, :3].sum() == 0   # guide = complement (T), 3 empty positions first
    assert t[1, :2].sum() == 0 and t[1, 2].tolist() == [0, 0, 1, 0]                       # short flank: padded with empty positions, 'G' last


def test_paired_within_gene_matches_groupby_version():
    rng = np.random.default_rng(1)
    g = np.repeat([f"g{i}" for i in range(20)], 12)
    act = rng.normal(size=len(g))
    df = pd.DataFrame({"gene": g, "act": act, "a": act + rng.normal(size=len(g)) * 0.5, "b": act + rng.normal(size=len(g)) * 1.5, "block": g})
    d0, lo, hi = M.paired_within_gene(df, "a", "b", n=100, min_n=5)
    ref = M.within_group(df, "gene", "act", "a", min_n=5) - M.within_group(df, "gene", "act", "b", min_n=5)
    assert abs(d0 - ref) < 1e-9 and lo < d0 < hi and d0 > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
