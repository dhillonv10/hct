"""Leakage audit of the training corpus against the LongevityBench DNAm prompts.

1. GSM overlap: no training sample may be a GSM that appears in any prompt.
2. Series overlap: no training series may be one of the 42 prompt series.
3. LB-0190 value matching (its prompts carry no sample or series ids): for every one of its 1,438 profiles (353 CpGs,
   2-decimal betas), find the closest training profile; a (near-)identical profile means the sample is in the corpus."""
import numpy as np

from epimod.lb.store import lb_meta, lb_series, load_lb_store
from epimod.lb.tasks import load_parsed

parsed = load_parsed(); lb_ser = lb_series(parsed)
meta = lb_meta(None)
gsm = set()
for t in parsed.values():
    gsm |= set(t.gsm) | set(t.gsm_b)
print(f"training corpus: {len(meta)} samples, {meta.DatasetID.nunique()} series")
print("GSM overlap with prompts (all samples, before exclusion):", len(gsm & set(meta.index)), "| series overlap:", len(lb_ser & set(meta.DatasetID)))
ex = lb_meta(lb_ser)
print("after excluding prompt series -> GSM overlap:", len(gsm & set(ex.index)), "| series overlap:", len(lb_ser & set(ex.DatasetID)))

st = load_lb_store(exclude_series=lb_ser)
t = parsed["LB-0190"]; ids = list(t.ids[0]); X = np.array(t.beta, np.float32)
cols = st.universe.get_indexer(ids); assert (cols >= 0).all()
T = np.asarray(st.prof[:, cols]).astype(np.float32); miss = T == 255; T = np.where(miss, np.nan, T / 100.0)
fill = np.nanmean(T, 0)
T0 = np.where(np.isnan(T), fill, T)                                          # mean-impute; identical profiles still match exactly
d2 = (X ** 2).sum(1)[:, None] + (T0 ** 2).sum(1)[None, :] - 2 * X @ T0.T
j = d2.argmin(1); best = np.sqrt(np.maximum(d2[np.arange(len(X)), j], 0) / X.shape[1])
print(f"LB-0190 vs {st.n} training profiles: RMS diff to nearest  min {best.min():.4f} | 1% {np.percentile(best, 1):.4f} | median {np.median(best):.4f}")
close = best < 0.006
print("profiles matching a training sample to < 0.006 RMS (~2-decimal rounding):", int(close.sum()))
if close.any():
    hit = st.meta.iloc[j[close]]
    print("series of the matched training samples:", hit.DatasetID.value_counts().head(10).to_dict())
