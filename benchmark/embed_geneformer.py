"""Frozen Geneformer embeddings of bulk RNA-seq samples, exactly as in the benchmark notebook.

Tokenisation = Geneformer's rank-value encoding: expression -> counts-per-10k -> divide by the gene's median in the pretraining corpus
-> rank descending -> keep the top 2,048 tokens (V2 models add <cls>/<eos>).  Representation (fixed before any evaluation):
mean over gene positions of the SECOND-TO-LAST hidden layer ('l2'; the last layer 'last' is stored as a robustness variant).
Control: identical architecture, RANDOM weights (seeded) -> isolates what pretraining adds.

Input: one .npz per study with `expr` (samples x genes, LINEAR scale, non-negative), `genes` (HGNC symbols), `sid`.
Usage: python embed_geneformer.py <inputs_dir> <gf_dir> <out_dir> [model ...] [--limit N]     models: gf10 gf104 gf316
`gf_dir` must contain Geneformer-V1-10M/, Geneformer-V2-104M/, Geneformer-V2-316M/ and geneformer/ (dictionaries) as in the
HuggingFace repo ctheodoris/Geneformer; use --download to fetch them there.
"""
import sys, os, time, glob, warnings, numpy as np
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rest_pickle import load as plk          # dictionaries are pickles: load through a numpy-scalar-only whitelist, never plain pickle.load
MAXLEN = 2048
CFGS = {'gf10': dict(path='Geneformer-V1-10M', d='geneformer/gene_dictionaries_30m/%s_gc30M.pkl', special=False),
        'gf104': dict(path='Geneformer-V2-104M', d='geneformer/%s_gc104M.pkl', special=True),
        'gf316': dict(path='Geneformer-V2-316M', d='geneformer/%s_gc104M.pkl', special=True)}      # 316M shares the 104M dictionaries
DICTS = {'tok': 'token_dictionary', 'med': 'gene_median_dictionary', 'names': 'gene_name_id_dict'}


def download(gf_dir, models):
    from huggingface_hub import snapshot_download
    pats = ['geneformer/gene_median_dictionary_gc104M.pkl', 'geneformer/token_dictionary_gc104M.pkl', 'geneformer/gene_name_id_dict_gc104M.pkl', 'geneformer/gene_dictionaries_30m/*.pkl']
    for m in models:
        pats += [f"{CFGS[m]['path']}/config.json", f"{CFGS[m]['path']}/model.safetensors"]
    snapshot_download('ctheodoris/Geneformer', local_dir=gf_dir, allow_patterns=pats)


def tokenise(expr, genes, cfg, gf_dir):
    tok, med, names = (plk(os.path.join(gf_dir, cfg['d'] % DICTS[k])) for k in ('tok', 'med', 'names'))
    up = {}
    for n, e in names.items():
        up.setdefault(str(n).upper(), e)
    ens = [up.get(g) for g in genes]
    ok = np.array([e is not None and e in tok and e in med for e in ens]); idx = np.where(ok)[0]
    tid = np.array([int(tok[ens[i]]) for i in idx], dtype=np.int64); mval = np.array([float(med[ens[i]]) for i in idx])
    seqs = []
    for r in range(len(expr)):
        x = expr[r, idx].astype(np.float64); x = x / max(x.sum(), 1e-12) * 1e4          # counts-per-10k over the genes Geneformer knows
        v = x / mval; nz = np.where(x > 0)[0]
        order = nz[np.argsort(-v[nz], kind='stable')][:MAXLEN - (2 if cfg['special'] else 0)]
        ids = [int(t) for t in tid[order]]
        if cfg['special']:
            ids = [int(tok['<cls>'])] + ids + [int(tok['<eos>'])]
        seqs.append(ids)
    return seqs, int(ok.sum())


def embed(seqs, cfg, gf_dir, variant, threads=4):
    import torch
    from transformers import BertModel, BertConfig
    torch.set_num_threads(threads); p = os.path.join(gf_dir, cfg['path'])
    if variant == 'pretrained':
        mdl = BertModel.from_pretrained(p, add_pooling_layer=False)
    else:
        torch.manual_seed(0); mdl = BertModel(BertConfig.from_pretrained(p), add_pooling_layer=False)
    mdl.eval(); l2, last = [], []; t0 = time.time()
    for i, ids in enumerate(seqs):
        inp = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            hs = mdl(input_ids=inp, attention_mask=torch.ones_like(inp), output_hidden_states=True).hidden_states
        lo, hi = (1, len(ids) - 1) if cfg['special'] else (0, len(ids))
        l2.append(hs[-2][0, lo:hi].mean(0).numpy()); last.append(hs[-1][0, lo:hi].mean(0).numpy())
        if i in (0, 3) or (i + 1) % 20 == 0:
            print(f'    {variant}: {i + 1}/{len(seqs)}  {(time.time() - t0) / (i + 1):.1f}s/sample', flush=True)
    return np.array(l2), np.array(last)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1]); args = [a for a in args if a != str(limit)]
    inputs, gf_dir, out_dir = args[:3]; models = args[3:] or ['gf10', 'gf104', 'gf316']
    if '--download' in sys.argv:
        download(gf_dir, models)
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(inputs, '*.npz'))):
        study = os.path.basename(f)[:-4]; z = np.load(f, allow_pickle=False); expr, genes, sid = z['expr'], list(z['genes']), z['sid']
        if limit:
            expr, sid = expr[:limit], sid[:limit]
        for m in models:
            dest = os.path.join(out_dir, f'{study}__{m}.npz')
            if os.path.exists(dest):
                print('skip', dest); continue
            seqs, nk = tokenise(expr, genes, CFGS[m], gf_dir)
            print(f'{study}/{m}: {len(seqs)} samples, {nk} of {len(genes)} genes in the model vocabulary, median sequence length {int(np.median([len(s) for s in seqs]))}', flush=True)
            out = {}
            for variant in ('pretrained', 'random'):
                out[f'{variant}_l2'], out[f'{variant}_last'] = embed(seqs, CFGS[m], gf_dir, variant)
            np.savez_compressed(dest, sid=sid, **out)
