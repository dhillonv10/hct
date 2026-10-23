"""Where does a trained EpiSet lose on LB-0010? Error by true-age bin and tissue-agnostic summaries; no stores needed."""
import sys
import numpy as np, pandas as pd, torch

from epimod.age import t_to_age
from epimod.lb.episet import EpiSet, SetConfig
from epimod.lb.store import lb_series
from epimod.lb.tasks import Vocab, eval_single, load_parsed
from epimod.paths import subdir

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
parsed = load_parsed()
uni = pd.Index(pd.read_parquet(subdir("lb") / "universe.parquet")["cpg"].values)
ids = pd.Index(ck["vocab_ids"]); u2m = np.full(len(uni), -1, np.int64); pos = uni.get_indexer(ids); u2m[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
model = EpiSet(SetConfig(**ck["cfg"])); model.load_state_dict(ck["state"]); model.eval()
vocab = Vocab(u2m, ids, model.cpg_mean[:-1].numpy(), model.cpg_std[:-1].numpy())
t = parsed["LB-0010"]; b = eval_single(t, vocab, uni)
mu, sig, _ = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])
pred, age = t_to_age(mu), np.array(t.age)
df = pd.DataFrame({"age": age, "pred": pred, "err": pred - age, "abs": np.abs(pred - age), "series": [g if isinstance(g, str) else (g[0] if len(g) else "") for g in t.gse], "n_tok": (b["cpg"] >= 0).sum(1).numpy()})
print(f"MAE {df['abs'].mean():.2f} | bias {df.err.mean():+.2f} | corr {np.corrcoef(age, pred)[0,1]:.3f} | pred sd {pred.std():.1f} vs age sd {age.std():.1f}")
print(df.groupby(pd.cut(df.age, [0, 20, 30, 40, 50, 60, 70, 80, 120])).agg(n=("age", "size"), mae=("abs", "mean"), bias=("err", "mean")).round(2))
g = df.groupby("series").agg(n=("age", "size"), age=("age", "mean"), mae=("abs", "mean"), bias=("err", "mean"), tok=("n_tok", "mean")).round(2)
print(g.sort_values("n", ascending=False).head(15))
print("tokens kept in vocabulary: mean", df.n_tok.mean().round(1), "of 200")

print("train-age prior check: LB-0010 age mean %.1f sd %.1f | pred mean %.1f" % (age.mean(), age.std(), pred.mean()))
print("n distinct series in LB-0010 items:", df.series.nunique())
