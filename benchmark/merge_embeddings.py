"""Merge Geneformer embeddings into benchmark/data/geneformer_embeddings.npz
   keys: <study>__sid, <study>__<model>__<variant>_<layer>   (model: gf10|gf104|gf316, variant: pretrained|random, layer: l2|last)
   PTCY comes from the first run of the same procedure (gf_embeddings.npz, gf316_embeddings.npz); the other studies from embed_geneformer.py."""
import sys, glob, os, numpy as np
fm_out, old10_104, old316, dest = sys.argv[1:5]
out = {}
a, b = np.load(old10_104, allow_pickle=False), np.load(old316, allow_pickle=False)
assert list(a['sid']) == list(b['sid'])
out['PTCY__sid'] = np.array(a['sid']).astype('U')
for src, models in ((a, ('gf10', 'gf104')), (b, ('gf316',))):
    for m in models:
        for v in ('pretrained', 'random'):
            for l in ('l2', 'last'):
                out[f'PTCY__{m}__{v}_{l}'] = src[f'{m}_{v}_{l}'].astype(np.float32)
for f in sorted(glob.glob(os.path.join(fm_out, '*__*.npz'))):
    study, model = os.path.basename(f)[:-4].split('__'); z = np.load(f, allow_pickle=False)
    out[f'{study}__sid'] = np.array(z['sid']).astype('U')
    for v in ('pretrained', 'random'):
        for l in ('l2', 'last'):
            out[f'{study}__{model}__{v}_{l}'] = z[f'{v}_{l}'].astype(np.float32)
np.savez_compressed(dest, **out); print(len(out), 'arrays ->', dest, f'{os.path.getsize(dest) / 1e6:.1f} MB')
