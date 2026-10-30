#!/usr/bin/env python3
"""mito_nt_compare.py -- zero-shot mtDNA variant scores from the ORIGINAL Nucleotide Transformer (InstaDeep, ESM architecture, ~500M parameters).

Loads with standard `transformers` classes only (no trust_remote_code: the v1 configs have auto_map = None; the v2 models need remote code and are NOT used).
Licence of the weights: CC-BY-NC-SA-4.0 (non-commercial research use).

Score (higher = more damaging): for each variant, window offsets and both strands, the 6-mer token covering the variant is masked and
    score = log P(reference 6-mer) - log P(variant 6-mer)
averaged over offsets and strands (circular 768-nt windows of the human mitogenome, the variant at a different position inside its 6-mer for each offset).

  python mito_nt_compare.py --workdir mito_work --out mito_work/ext_nt500m.csv [--model InstaDeepAI/nucleotide-transformer-500m-human-ref] [--name nt500m]
  python mito_lm.py evaluate --workdir mito_work --sizes small --external mito_work/ext_nt500m.csv --constraint-n 6000
(run `python mito_lm.py export-variants --constraint-n 6000` first; it writes the variant lists this script scores.)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mitolm"))
import mito_lm as M                                                           # noqa: E402

T, TC = 128, 64                                                               # 128 six-mer tokens per window (768 nt); the variant's token is number 64
COMP = str.maketrans("ACGT", "TGCA")


def circ(seq, start, n):
    L = len(seq)
    start %= L
    return (seq + seq)[start:start + n] if start + n <= 2 * L else "".join(seq[(start + i) % L] for i in range(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", default="mito_work")
    ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    ap.add_argument("--name", default="nt500m")
    ap.add_argument("--out", default="")
    ap.add_argument("--offsets", default="0,1,2,3,4,5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N variants (testing)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true", help="CUDA only: half-precision autocast")
    a = ap.parse_args()
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ref = M.Reference(os.path.join(a.workdir, "raw", "human.gb"))
    G = ref.seq.replace("N", "A")                                             # rCRS position 3107 is a placeholder N
    R = G.translate(COMP)[::-1]
    L = len(G)
    exp = os.path.join(a.workdir, "export")
    v = pd.concat([pd.read_csv(os.path.join(exp, f)) for f in ("variants_clinvar.csv", "variants_constraint.csv")]).drop_duplicates().reset_index(drop=True)
    if a.limit:
        v = v.iloc[:a.limit]
    print(f"{len(v)} variants; model {a.model}; device {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev).eval()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)
    vocab = tok.get_vocab()
    cls_id, mask_id = tok.cls_token_id, tok.mask_token_id
    # make sure my fast 6-mer encoding equals the tokenizer's, including special tokens
    probe = circ(G, 5000, 6 * T)
    mine = [cls_id] + [vocab[probe[i:i + 6]] for i in range(0, 6 * T, 6)]
    ref_ids = tok(probe)["input_ids"]
    if ref_ids == mine:
        suffix = []
    elif ref_ids == mine + [tok.eos_token_id]:
        suffix = [tok.eos_token_id]
    else:
        sys.exit(f"tokenizer output differs from the expected [cls] + 6-mers layout:\n  tokenizer: {ref_ids[:6]}...{ref_ids[-3:]} (len {len(ref_ids)})\n  expected:  {mine[:6]}...{mine[-3:]} (len {len(mine)})")
    offsets = [int(x) for x in a.offsets.split(",")]
    jpos = 1 + TC                                                              # token index of the variant's 6-mer (after <cls>)
    scores = np.zeros(len(v))
    count = np.zeros(len(v))
    jobs = []
    for i, (p, r, al) in enumerate(zip(v.pos.values, v.ref.values, v.alt.values)):
        for strand in (0, 1):
            for o in offsets:
                jobs.append((i, strand, o, int(p) - 1, r, al))
    t0 = time.time()
    for b0 in range(0, len(jobs), a.batch):
        chunk = jobs[b0:b0 + a.batch]
        ids = np.zeros((len(chunk), len(mine) + len(suffix)), np.int64)
        ref_t, alt_t = np.zeros(len(chunk), np.int64), np.zeros(len(chunk), np.int64)
        for k, (i, strand, o, p0, r, al) in enumerate(chunk):
            if strand == 0:
                g, p, rb, ab = G, p0, r, al
            else:
                g, p, rb, ab = R, L - 1 - p0, r.translate(COMP), al.translate(COMP)
            inner = TC * 6 + o
            seq = circ(g, p - inner, 6 * T)
            assert seq[inner] == rb, (seq[inner], rb, p0, strand)
            alt_seq = seq[:inner] + ab + seq[inner + 1:]
            row = [cls_id] + [vocab[seq[x:x + 6]] for x in range(0, 6 * T, 6)] + suffix
            ids[k] = row
            tok_idx = inner // 6                                               # == TC for every offset 0..5
            ref_t[k] = vocab[seq[tok_idx * 6:tok_idx * 6 + 6]]
            alt_t[k] = vocab[alt_seq[tok_idx * 6:tok_idx * 6 + 6]]
        x = torch.from_numpy(ids).to(dev)
        x[:, jpos] = mask_id
        with torch.inference_mode():
            if a.fp16 and dev == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, jpos]
            else:
                lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, jpos]
        lp = lg.float().log_softmax(-1).cpu().numpy()
        s = lp[np.arange(len(chunk)), ref_t] - lp[np.arange(len(chunk)), alt_t]
        for k, (i, *_rest) in enumerate(chunk):
            scores[i] += s[k]
            count[i] += 1
        if (b0 // a.batch) % 50 == 0:
            done = b0 + len(chunk)
            print(f"  {done}/{len(jobs)} windows, {done / (time.time() - t0):.0f}/s", flush=True)
    v[a.name] = scores / count
    out = a.out or os.path.join(a.workdir, f"ext_{a.name}.csv")
    v.to_csv(out, index=False)
    print(f"wrote {out}: {len(v)} variants; score mean {v[a.name].mean():.3f} sd {v[a.name].std():.3f}")


if __name__ == "__main__":
    main()
