#!/usr/bin/env python3
"""mito_lm.py -- a vertebrate mitogenome language model: download, build, pretrain, and test it against honest baselines. One file.

WHAT IT BUILDS
  A masked nucleotide transformer (single-base tokens, 512-nt windows, circular wrap, reverse-complement augmentation) pretrained on the
  ~7,900 RefSeq vertebrate mitochondrial genomes. It scores human mtDNA single-nucleotide variants ZERO-SHOT (no labels): the log-likelihood
  ratio log P(alt) - log P(ref) at the masked variant position, averaged over both strands. Three sizes (tiny ~0.9M, small ~5M, base ~14M
  parameters) make it a small family.

WHAT IT TESTS (the part that decides whether the model is worth anything)
  Pathogenic vs benign discrimination on human mtDNA SNVs from ClinVar (labels with >= 1 review star), plus a second set that adds common
  gnomAD-mito polymorphisms as benign proxies. Against:
    * conservation  : alignment-column statistics from the same vertebrate / mammal genomes (simple, but it is the baseline that matters)
    * annotation    : region, codon position, synonymous / missense / stop, BLOSUM62, transition vs transversion (logistic regression, cross-validated)
    * stacked       : annotation + conservation, with and without the language-model score. The headline question is whether the LM adds anything
                      beyond conservation + annotation (paired block-bootstrap).
  POPULATION CONSTRAINT (second task, ~100x more labels): of all ~49,700 possible mtDNA SNVs, which were never seen among the 56,434 gnomAD individuals?
  Scored within strata of (substitution x flanking bases), so mutation-rate differences cannot explain a score; positions with any non-PASS gnomAD record
  are excluded. (A plain "does the variant reach homoplasmy?" label is degenerate: 98.8% of variants with >= 3 carriers are seen homoplasmic.)
  All intervals are block bootstraps over 100-bp blocks (nearby variants are not independent). Pretraining validation uses held-out GENERA.

HONEST CAVEATS BUILT INTO THE OUTPUT
  * ClinVar has only ~118 pathogenic / likely-pathogenic mtDNA SNVs (>= 1 star): AUROC intervals are wide (about +-0.05).
  * Pathogenic variants sit mostly in tRNA and ND genes, benign ones mostly in protein genes: the report gives per-region and region-stratified AUROC.
  * ClinVar classifications partly rest on conservation / in-silico evidence, so "beats conservation" can be partly circular.
  * MITOMAP blocks automated downloads (HTTP 403), so it is not used.

USAGE
  pip install torch numpy pandas scikit-learn scipy biopython
  python mito_lm.py                          # everything: download -> corpus -> pretrain (small, 4 h cap) -> evaluate
  python mito_lm.py --sizes tiny,small,base --hours 3        # the whole family, 3 h cap per size
  python mito_lm.py download | corpus | pretrain | evaluate | constraint  # single stages (resumable); `evaluate` runs ClinVar tests + constraint
  After training, `python mito_lm.py evaluate --sizes tiny,small,base` scores every checkpoint it finds (the first run caches the conservation panel).
  python mito_lm.py --smoke                  # 2-minute plumbing check on tiny data (separate workdir)
  Options: --workdir DIR (default ./mito_work)  --device mps|cuda|cpu (default: auto)  --steps N  --api-key NCBI_KEY  --email you@x.org
  Memory: the effective batch is 64 windows, run as --batch 16 micro-batches x --accum 4. The script stops cleanly (checkpoint saved) if memory use passes
  --mem-gb (default 55% of RAM). If it stops or reports out-of-memory, use --batch 8 --accum 8 (or 4 x 16).
  Downloads: ~130 MB sequences (NCBI Entrez), 200 MB ClinVar VCF (filtered then deletable), 5 MB gnomAD chrM. Total disk ~1 GB.
  Outputs in WORKDIR/results: report.txt, variant_scores.csv, metrics.csv, train_log_<size>.csv; checkpoints in WORKDIR/ckpt.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ENTREZ = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
Q_VERT = "txid7742[Organism:exp] AND refseq[filter] AND mitochondrion[filter]"       # Vertebrata, RefSeq mitochondrial genomes
Q_MAMM = "txid40674[Organism:exp] AND refseq[filter] AND mitochondrion[filter]"       # Mammalia
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
GNOMAD_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1/vcf/genomes/gnomad.genomes.v3.1.sites.chrM.vcf.bgz"
HUMAN_ACC = "NC_012920.1"                                                              # rCRS = GRCh38 chrM = coordinates of ClinVar / gnomAD
UA = {"User-Agent": "mito_lm/1.0 (research script)"}

SIZES = {"tiny": dict(d=128, layers=4, heads=4, ff=512, lr=1e-3, steps=20000),
         "small": dict(d=256, layers=6, heads=8, ff=1024, lr=6e-4, steps=40000),
         "base": dict(d=384, layers=8, heads=6, ff=1536, lr=4e-4, steps=60000)}
WIN = 512
PAD, N_TOK, MASK = 0, 5, 6                      # tokens: 0 pad, 1-4 = A C G T, 5 = N, 6 = [MASK]
VOCAB = 7
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}         # array codes 0-3, 4 = N (tokens = code + 1)
BASES = ("A", "C", "G", "T")
LUT = np.full(256, 4, np.uint8)
for _c, _v in CODE.items():
    LUT[ord(_c)] = _v
    LUT[ord(_c.lower())] = _v


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wp(a, *p):
    return os.path.join(a.workdir, *p)


# --------------------------------------------------------------------------------------------------------------------- downloading
def http_get(url, data=None, tries=5, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            log(f"  retry {k + 1} after error: {e}")
            time.sleep(2 ** (k + 1))


def download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        log(f"  {label}: already present ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return
    tmp = dest + ".part"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r, open(tmp, "wb") as f:
                total, got, nxt = int(r.headers.get("Content-Length") or 0), 0, 50e6
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if got >= nxt:
                        log(f"  {label}: {got / 1e6:.0f} / {total / 1e6:.0f} MB")
                        nxt += 50e6
            if total and got != total:
                raise IOError(f"short read {got} of {total}")
            os.replace(tmp, dest)
            log(f"  {label}: done ({got / 1e6:.1f} MB)")
            return
        except Exception as e:
            log(f"  {label}: attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    sys.exit(f"could not download {url}")


def entrez_params(a, **kw):
    kw.update(tool="mito_lm")
    if a.email:
        kw["email"] = a.email
    if a.api_key:
        kw["api_key"] = a.api_key
    return kw


def esearch(a, query):
    u = ENTREZ + "esearch.fcgi?" + urllib.parse.urlencode(entrez_params(a, db="nuccore", term=query, retmax=20000, retmode="json", idtype="acc"))
    return json.loads(http_get(u))["esearchresult"]["idlist"]


def efetch(a, accs, rettype):
    data = urllib.parse.urlencode(entrez_params(a, db="nuccore", id=",".join(accs), rettype=rettype, retmode="text")).encode()
    time.sleep(0.12 if a.api_key else 0.40)                                  # NCBI limit: 3 requests/s without a key, 10/s with
    return http_get(ENTREZ + "efetch.fcgi", data=data).decode()


def prepare_clinvar(raw, out):
    """MT SNVs from the ClinVar GRCh38 VCF -> TSV with pos, ref, alt, class (P / B / other), review stars, gene."""
    def klass(s):
        s = s.lower()
        if "conflict" in s or "uncertain" in s:
            return "other"
        if "pathogenic" in s and "benign" not in s:
            return "P"
        if "benign" in s and "pathogenic" not in s:
            return "B"
        return "other"

    def stars(r):
        r = r.lower()
        if "practice_guideline" in r or "reviewed_by_expert_panel" in r:
            return 3
        if "multiple_submitters" in r and "no_conflicts" in r:
            return 2
        if "criteria_provided" in r and "single_submitter" in r:
            return 1
        return 0
    rows = []
    with gzip.open(raw, "rt") as f:
        for line in f:
            if not line.startswith("MT\t"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[3] not in BASES or c[4] not in BASES:                            # single-base A/C/G/T alleles only (drops indels, '.', '*')
                continue
            d = dict(kv.split("=", 1) for kv in c[7].split(";") if "=" in kv)
            gene = re.match(r"([^:|]+)", d.get("GENEINFO", ""))
            rows.append((int(c[1]), c[3], c[4], klass(d.get("CLNSIG", "")), stars(d.get("CLNREVSTAT", "")), gene.group(1) if gene else ""))
    with open(out, "w") as f:
        f.write("pos\tref\talt\tclass\tstars\tgene\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    log(f"  ClinVar: {len(rows)} MT SNVs kept")


def prepare_gnomad(raw, out):
    rows = []
    with gzip.open(raw, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[6] != "PASS" or c[3] not in BASES or c[4] not in BASES:
                continue
            d = dict(kv.split("=", 1) for kv in c[7].split(";") if "=" in kv)
            rows.append((int(c[1]), c[3], c[4], d.get("AN", ""), d.get("AC_hom", "0"), d.get("AC_het", "0"), d.get("AF_hom", "0"), d.get("AF_het", "0"), d.get("max_hl", "")))
    with open(out, "w") as f:
        f.write("pos\tref\talt\tAN\tAC_hom\tAC_het\tAF_hom\tAF_het\tmax_hl\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    log(f"  gnomAD-mito: {len(rows)} PASS SNVs kept")


def stage_download(a):
    raw = wp(a, "raw")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(wp(a, "labels"), exist_ok=True)
    log("1/4 sequences: RefSeq vertebrate mitochondrial genomes (NCBI Entrez)")
    accp = os.path.join(raw, "accessions.json")
    if not os.path.exists(accp):
        vert, mamm = esearch(a, Q_VERT), esearch(a, Q_MAMM)
        if a.smoke:                                                               # 40 mammals + 20 others + human
            others = [x for x in vert if x not in set(mamm)]
            vert = sorted(set(mamm[:40] + others[:20] + [HUMAN_ACC]))
        json.dump({"vert": vert, "mammal": sorted(set(mamm) & set(vert))}, open(accp, "w"))
    accs = json.load(open(accp))["vert"]
    log(f"  {len(accs)} accessions")
    for i in range(0, len(accs), 100):
        fp = os.path.join(raw, f"fasta_{i // 100:04d}.fa")
        if os.path.exists(fp):
            continue
        txt = efetch(a, accs[i:i + 100], "fasta")
        if txt.count(">") < len(accs[i:i + 100]) * 0.9:
            log(f"  warning: batch {i // 100} returned {txt.count('>')} of {len(accs[i:i + 100])} records")
        open(fp + ".part", "w").write(txt)
        os.replace(fp + ".part", fp)
        if (i // 100) % 10 == 0:
            log(f"  fetched {min(i + 100, len(accs))} / {len(accs)}")
    log("2/4 human reference annotation (NC_012920.1 GenBank record)")
    gb = os.path.join(raw, "human.gb")
    if not os.path.exists(gb):
        open(gb, "w").write(efetch(a, [HUMAN_ACC], "gb"))
    log("3/4 ClinVar")
    tsv = wp(a, "labels", "clinvar_mt.tsv")
    if not os.path.exists(tsv):
        download(CLINVAR_URL, os.path.join(raw, "clinvar.vcf.gz"), "ClinVar VCF")
        prepare_clinvar(os.path.join(raw, "clinvar.vcf.gz"), tsv)
    log("4/4 gnomAD-mito")
    tsv = wp(a, "labels", "gnomad_mt.tsv")
    if not os.path.exists(tsv):
        download(GNOMAD_URL, os.path.join(raw, "gnomad_chrM.vcf.bgz"), "gnomAD chrM VCF")
        prepare_gnomad(os.path.join(raw, "gnomad_chrM.vcf.bgz"), tsv)


# --------------------------------------------------------------------------------------------------------------------- corpus
def read_fasta(text):
    head, seq = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if head is not None:
                yield head, "".join(seq)
            head, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if head is not None:
        yield head, "".join(seq)


def stage_corpus(a):
    out = wp(a, "corpus.npz")
    if os.path.exists(out):
        log("corpus: already built")
        return
    mamm = set(json.load(open(wp(a, "raw", "accessions.json")))["mammal"])
    recs, seen = [], set()
    files = sorted(f for f in os.listdir(wp(a, "raw")) if f.startswith("fasta_") and f.endswith(".fa"))
    dropped = dict(length=0, ambiguous=0, duplicate=0)
    for fn in files:
        for head, seq in read_fasta(open(wp(a, "raw", fn)).read()):
            acc, desc = head.split(None, 1) if " " in head else (head, "")
            arr = LUT[np.frombuffer(seq.encode(), np.uint8)]
            if not 14000 <= len(arr) <= 20000:
                dropped["length"] += 1
                continue
            if (arr == 4).mean() > 0.01:
                dropped["ambiguous"] += 1
                continue
            h = hashlib.md5(arr.tobytes()).hexdigest()
            if h in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(h)
            genus = desc.split()[0] if desc else acc
            recs.append((acc, genus, acc in mamm, arr))
    # validation = ~3% of GENERA (congeneric mitogenomes are near-identical, so a random genome split would leak)
    val = np.array([int(hashlib.md5(r[1].encode()).hexdigest()[:8], 16) % 100 < 3 for r in recs])
    off = np.cumsum([0] + [len(r[3]) for r in recs])
    np.savez_compressed(out, seq=np.concatenate([r[3] for r in recs]), off=off, val=val, mammal=np.array([r[2] for r in recs]),
                        acc=np.array([r[0] for r in recs]), genus=np.array([r[1] for r in recs]))
    log(f"corpus: {len(recs)} genomes, {off[-1] / 1e6:.1f} Mnt, {len(set(r[1] for r in recs))} genera, {int(val.sum())} validation genomes "
        f"({len(set(r[1] for r, v in zip(recs, val) if v))} held-out genera), {int(sum(r[2] for r in recs))} mammals; dropped {dropped}")


def load_corpus(a):
    z = np.load(wp(a, "corpus.npz"), allow_pickle=False)
    off, seq = z["off"], z["seq"]                                                 # decompress the 130 MB array ONCE; genomes are views of it
    g = [seq[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return g, z["val"], z["mammal"], z["acc"]


# --------------------------------------------------------------------------------------------------------------------- model
def get_torch():
    import torch
    return torch


def phys_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 16.0


def used_mem_gb(dev):
    """GPU / process memory in use by this script, in GB (MPS: the driver's allocation, which is what macOS actually counts)."""
    torch = get_torch()
    try:
        if dev == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        if dev == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if sys.platform == "darwin" else 1e6)


def pick_device(a):
    torch = get_torch()
    if a.device != "auto":
        return a.device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg):
    torch = get_torch()
    nn = torch.nn

    class MitoLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, cfg["d"], padding_idx=PAD)
            self.pos = nn.Embedding(WIN, cfg["d"])
            layer = nn.TransformerEncoderLayer(cfg["d"], cfg["heads"], cfg["ff"], dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg["d"])
            self.head = nn.Linear(cfg["d"], 4)

        def forward(self, x):
            h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
            return self.head(self.norm(self.enc(h)))                              # logits over A, C, G, T
    return MitoLM()


def revcomp_codes(x):
    """x: array of codes 0-3 (4 = N). Reverse complement along the last axis."""
    y = np.where(x < 4, 3 - x, 4)
    return y[..., ::-1]


class Sampler:
    def __init__(self, genomes, mammal_boost, seed):
        self.g = genomes
        w = np.array([len(x) for x in genomes], float)
        self.rng = np.random.default_rng(seed)
        self.w = w
        self.mammal = None
        self.p = w / w.sum()

    def set_mammal(self, mask, boost):
        w = self.w * np.where(mask, boost, 1.0)
        self.p = w / w.sum()

    def batch(self, B):
        gi = self.rng.choice(len(self.g), B, p=self.p)
        out = np.empty((B, WIN), np.uint8)
        for k, i in enumerate(gi):
            g = self.g[i]
            idx = (self.rng.integers(len(g)) + np.arange(WIN)) % len(g)           # circular wrap: mtDNA has no ends
            x = g[idx]
            out[k] = revcomp_codes(x) if self.rng.random() < 0.5 else x
        return out


def mask_batch(codes, rng, frac=0.15):
    """BERT-style: 15% of A/C/G/T positions; 80% -> [MASK], 10% -> random base, 10% unchanged. Returns tokens, labels (-100 = not scored)."""
    tok = codes.astype(np.int64) + 1
    cand = codes < 4
    m = (rng.random(codes.shape) < frac) & cand
    labels = np.where(m, codes.astype(np.int64), -100)
    r = rng.random(codes.shape)
    tok = np.where(m & (r < 0.8), MASK, tok)
    rnd = m & (r >= 0.8) & (r < 0.9)
    tok = np.where(rnd, rng.integers(1, 5, codes.shape), tok)
    return tok, labels


def stage_pretrain(a, size):
    torch = get_torch()
    F = torch.nn.functional
    cfg = SIZES[size]
    dev = pick_device(a)
    genomes, val, mammal, _ = load_corpus(a)
    train_g = [g for g, v in zip(genomes, val) if not v]
    val_g = [g for g, v in zip(genomes, val) if v]
    os.makedirs(wp(a, "ckpt"), exist_ok=True)
    os.makedirs(wp(a, "results"), exist_ok=True)
    ck = wp(a, "ckpt", f"{size}.pt")
    torch.manual_seed(a.seed)
    model = build_model(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    step, total = 0, a.steps or cfg["steps"]
    if a.smoke:
        total = 60
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, total = s["step"], s["total"]
        log(f"[{size}] resumed from step {step} / {total}")
        if step >= total:
            log(f"[{size}] already trained")
            return
    log(f"[{size}] {n_par / 1e6:.2f}M parameters, device {dev}, {len(train_g)} training / {len(val_g)} validation genomes")
    smp = Sampler(train_g, 1.0, a.seed + step)
    smp.set_mammal(mammal[~val], a.mammal_boost)
    # fixed validation windows from held-out genera
    vs = Sampler(val_g, 1.0, 12345) if val_g else None
    vcodes = vs.batch(min(1024, 16 * len(val_g) + 64)) if vs else None
    vtok, vlab = mask_batch(vcodes, np.random.default_rng(7)) if vs else (None, None)
    base = np.bincount(np.concatenate([g[:2000] for g in train_g[:200]]).clip(max=3), minlength=4)
    base = base / base.sum()

    def validate():
        model.eval()
        tl, tc, tn = 0.0, 0, 0
        with torch.inference_mode():
            for i in range(0, len(vtok), 128):
                x = torch.from_numpy(vtok[i:i + 128]).to(dev)
                y = torch.from_numpy(vlab[i:i + 128]).to(dev)
                lg = model(x)
                sel = y >= 0
                tl += float(F.cross_entropy(lg[sel], y[sel], reduction="sum"))
                tc += int((lg[sel].argmax(-1) == y[sel]).sum())
                tn += int(sel.sum())
        model.train()
        return tl / tn, tc / tn
    if vs:
        yv = vlab[vlab >= 0]
        log(f"[{size}] validation reference: unigram baseline accuracy {np.bincount(yv, minlength=4).max() / len(yv):.3f}, loss {-(np.log(base)[yv]).mean():.3f}"
            f" (uniform = {math.log(4):.3f})")
    mem_cap = a.mem_gb or 0.55 * phys_ram_gb()
    logf = wp(a, "results", f"train_log_{size}.csv")
    if not os.path.exists(logf):
        open(logf, "w").write("step,train_loss,val_loss,val_acc,seq_per_s,elapsed_min\n")
    rng = np.random.default_rng(a.seed + 1000 + step)
    t0, tlast, run_loss, nrun = time.time(), time.time(), 0.0, 0
    model.train()
    warm = 300
    while step < total:
        lr = cfg["lr"] * min(1.0, (step + 1) / warm) * (1.0 if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        try:
            for _ in range(a.accum):                                              # micro-batches: same effective batch, a fraction of the memory
                tok, lab = mask_batch(smp.batch(a.batch), rng)
                x, y = torch.from_numpy(tok).to(dev), torch.from_numpy(lab).to(dev)
                loss = F.cross_entropy(model(x).reshape(-1, 4), y.reshape(-1), ignore_index=-100) / a.accum
                loss.backward()
                step_loss += loss.detach().item()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sys.exit(f"\nOut of memory at step {step}. Re-run with a smaller micro-batch, e.g.  --batch 8 --accum 8  (same effective batch). Progress up to the last checkpoint is kept.")
            raise
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run_loss += step_loss
        nrun += 1
        if step in (1, 5) or step % 25 == 0:
            gb = used_mem_gb(dev)
            if step in (1, 5):
                log(f"[{size}] memory in use after step {step}: {gb:.1f} GB (cap {mem_cap:.1f} GB of {phys_ram_gb():.0f} GB RAM)")
            if gb > mem_cap:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, total=total, cfg=cfg, size=size), ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                sys.exit(f"\nMemory use {gb:.1f} GB exceeds the cap of {mem_cap:.1f} GB; checkpoint saved at step {step}. Re-run with --batch 8 --accum 8 (or raise --mem-gb).")
        if step == 50 and a.hours and not a.smoke:                                # fit the schedule to the time budget
            rate = 50 * a.batch * a.accum / (time.time() - t0)
            fit = int(a.hours * 3600 * rate / (a.batch * a.accum))
            if fit < total:
                log(f"[{size}] {rate:.0f} windows/s -> {fit} steps fit in {a.hours} h (requested {total}); using {fit}")
                total = fit
            else:
                log(f"[{size}] {rate:.0f} windows/s -> the full {total} steps take about {total * a.batch * a.accum / rate / 3600:.1f} h")
        if step % a.eval_every == 0 or step == total:
            vl, vac = validate() if vs else (float("nan"), float("nan"))
            sps = nrun * a.batch * a.accum / (time.time() - tlast)
            open(logf, "a").write(f"{step},{run_loss / nrun:.4f},{vl:.4f},{vac:.4f},{sps:.1f},{(time.time() - t0) / 60:.1f}\n")
            log(f"[{size}] step {step}/{total} train {run_loss / nrun:.3f} val {vl:.3f} acc {vac:.3f} lr {lr:.1e} ({sps:.0f} windows/s)")
            run_loss, nrun, tlast = 0.0, 0, time.time()
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, total=total, cfg=cfg, size=size), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        if a.hours and (time.time() - t0) > a.hours * 3600 * 1.05:
            log(f"[{size}] time cap reached at step {step}; saving")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=total, total=total, cfg=cfg, size=size), ck)
            break
    log(f"[{size}] done")


# --------------------------------------------------------------------------------------------------------------------- reference annotation
class Reference:
    """rCRS sequence + per-position region / codon information from the GenBank record."""

    def __init__(self, gb_path):
        from Bio import SeqIO
        from Bio.Align import substitution_matrices
        rec = SeqIO.read(gb_path, "genbank")
        self.seq = str(rec.seq).upper()
        self.L = len(self.seq)
        self.codes = LUT[np.frombuffer(self.seq.encode(), np.uint8)]
        self.region = np.zeros(self.L, np.int8)                                   # 0 noncoding, 1 protein, 2 tRNA, 3 rRNA
        self.cds = []
        self.cds_at = {}
        for f in rec.features:
            if f.type not in ("CDS", "tRNA", "rRNA"):
                continue
            strand = f.location.strand or 1
            order = []
            for part in (f.location.parts if strand == 1 else reversed(f.location.parts)):
                r = range(int(part.start), int(part.end))
                order.extend(r if strand == 1 else reversed(r))
            if f.type == "CDS":
                name = (f.qualifiers.get("gene") or ["?"])[0]
                k = len(self.cds)
                self.cds.append(dict(gene=name, strand=strand, order=order, seq=str(f.location.extract(rec.seq)).upper()))
                for i, p in enumerate(order):
                    self.cds_at.setdefault(p, (k, i))
                    if self.region[p] == 0:
                        self.region[p] = 1
            else:
                for p in order:
                    self.region[p] = 3 if f.type == "rRNA" else 2
        for p in list(self.cds_at):
            if self.region[p] in (2, 3):
                del self.cds_at[p]
        self.blosum = substitution_matrices.load("BLOSUM62")
        from Bio.Data import CodonTable
        self.table = CodonTable.unambiguous_dna_by_id[2]                          # vertebrate mitochondrial code
        counts = np.bincount(self.region, minlength=4)
        log(f"reference: {self.L} nt, {len(self.cds)} CDS, positions: noncoding {counts[0]}, protein {counts[1]}, tRNA {counts[2]}, rRNA {counts[3]}")

    def aa(self, codon):
        if codon in self.table.stop_codons:
            return "*"
        return self.table.forward_table.get(codon, "X")

    def consequence(self, p0, alt):
        """(codon_pos 0-3, class in none/syn/mis/stopgain/stoploss/partial, blosum)."""
        if p0 not in self.cds_at:
            return 0, "none", 0.0
        k, i = self.cds_at[p0]
        c = self.cds[k]
        cp = i % 3
        codon = c["seq"][i - cp:i - cp + 3]
        if len(codon) < 3:
            return cp + 1, "partial", 0.0
        nt = alt if c["strand"] == 1 else {"A": "T", "C": "G", "G": "C", "T": "A"}[alt]
        mut = codon[:cp] + nt + codon[cp + 1:]
        a0, a1 = self.aa(codon), self.aa(mut)
        if a0 == a1:
            return cp + 1, "syn", 0.0
        if a1 == "*":
            return cp + 1, "stopgain", 0.0
        if a0 == "*":
            return cp + 1, "stoploss", 0.0
        try:
            bl = float(self.blosum[a0][a1])
        except Exception:
            bl = 0.0
        return cp + 1, "mis", bl


# --------------------------------------------------------------------------------------------------------------------- baselines
def conservation_counts(a, ref, genomes, accs, mammal, k=10, win=48, min_ident=0.55, min_ident_prop=0.5):
    """Column base counts, for all vertebrates and for mammals only, from genomes aligned to the human reference by ungapped 48-nt windows:
    a window takes the offset voted by >= 2 shared 10-mer seeds (identity >= 55%); windows without seeds borrow the offset of a window within 3 steps
    (+-3 nt indel shifts tried, identity >= 50%). A weak conservation baseline would flatter the language model, so sensitivity matters here.
    Returns (counts_all, counts_mammal), each (L, 4)."""
    L = ref.L
    nwin = (L + win - 1) // win
    hc = ref.codes.astype(np.int64)
    kk = np.zeros(L - k + 1, np.int64)
    ok = np.ones(L - k + 1, bool)
    for j in range(k):
        kk = kk * 4 + np.minimum(hc[j:L - k + 1 + j], 3)
        ok &= hc[j:L - k + 1 + j] < 4
    pos = np.flatnonzero(ok)
    u, first, cnt = np.unique(kk[pos], return_index=True, return_counts=True)
    keep = cnt == 1                                                               # human-unique seeds only
    hs, hp = u[keep], pos[first[keep]]
    counts = np.zeros((L, 4), np.int32)
    counts_m = np.zeros((L, 4), np.int32)

    def seeds(g):
        n = len(g)
        c = np.zeros(n - k + 1, np.int64)
        ok2 = np.ones(n - k + 1, bool)
        for j in range(k):
            c = c * 4 + np.minimum(g[j:n - k + 1 + j].astype(np.int64), 3)
            ok2 &= g[j:n - k + 1 + j] < 4
        idx = np.searchsorted(hs, c)
        idx[idx >= len(hs)] = 0
        hit = ok2 & (hs[idx] == c)
        return hp[idx[hit]], np.flatnonzero(hit)

    used = 0
    for gi in range(len(genomes)):
        if accs[gi] == HUMAN_ACC:
            continue
        g = genomes[gi]
        h, o = seeds(g)
        h2, o2 = seeds(revcomp_codes(g).copy())
        if len(h2) > len(h):
            g, h, o = revcomp_codes(g).copy(), h2, o2
        if len(h) < 20:
            continue
        diag = h - o
        key = (h // win).astype(np.int64) * 200000 + (diag + 100000)
        uk, c2 = np.unique(key, return_counts=True)
        w_id, d_val = uk // 200000, uk % 200000 - 100000
        order = np.lexsort((-c2, w_id))
        first_w = np.r_[True, w_id[order][1:] != w_id[order][:-1]]
        seeded = {int(wi): int(dv) for wi, dv, cc in zip(w_id[order][first_w], d_val[order][first_w], c2[order][first_w]) if cc >= 2}
        if not seeded:
            continue
        for wi in range(nwin):
            if wi in seeded:
                cand, shifts, thr = [seeded[wi]], (0,), min_ident
            else:                                                                 # divergent window: borrow a neighbour's offset (genomes are collinear), allow small indel shifts
                cand = sorted({seeded[nb] for dist in (1, 2, 3) for nb in (wi - dist, wi + dist) if nb in seeded})
                shifts, thr = range(-3, 4), min_ident_prop
                if not cand:
                    continue
            hi = np.arange(wi * win, min((wi + 1) * win, L))
            dvals = np.array([c + s for c in cand for s in shifts])
            oi = hi[None, :] - dvals[:, None]
            b = g[np.clip(oi, 0, len(g) - 1)]
            ok_ = (oi >= 0) & (oi < len(g)) & (b < 4)
            nv = ok_.sum(1)
            ident = np.where(nv >= win // 2, ((b == ref.codes[hi][None, :]) & ok_).sum(1) / np.maximum(nv, 1), 0.0)
            j = int(np.argmax(ident))
            if ident[j] < thr:
                continue
            sel = ok_[j]
            np.add.at(counts, (hi[sel], b[j][sel]), 1)
            if mammal[gi]:
                np.add.at(counts_m, (hi[sel], b[j][sel]), 1)
        used += 1
    log(f"  conservation panel: {used} genomes aligned; median depth {int(np.median(counts.sum(1)))} (all) / {int(np.median(counts_m.sum(1)))} (mammals) per position")
    return counts, counts_m


def cons_scores(counts, pos0, ref_c, alt_c, min_depth=20):
    n = counts[pos0].sum(1).astype(float)
    n_alt = counts[pos0, alt_c]
    n_ref = counts[pos0, ref_c]
    rarity = -np.log((n_alt + 0.5) / (n + 2.0))                                   # high = alt allele unseen in other species
    refc = (n_ref + 0.5) / (n + 2.0)
    bad = n < min_depth
    return np.where(bad, np.nan, rarity), np.where(bad, np.nan, refc), n


def ngram_scores(genomes, ref, pos0, ref_c, alt_c, k=5):
    """CONTROL for the language model: the same zero-shot score from a trivial model, p(base | k bases each side) counted over the training genomes
    (both strands, circular). If the transformer cannot beat this, it has learned local composition and nothing more."""
    n = 4 ** (2 * k + 1)
    tab = np.zeros(n, np.int64)
    chunk = []

    def flush():
        if chunk:
            tab[:] += np.bincount(np.concatenate(chunk), minlength=n)
            chunk.clear()
    for g in genomes:
        for x in (g, revcomp_codes(g)):
            y = np.concatenate([x, x[:2 * k]]).astype(np.int64)
            m = len(y) - 2 * k
            code, good = np.zeros(m, np.int64), np.ones(m, bool)
            for j in range(2 * k + 1):
                code = code * 4 + np.minimum(y[j:m + j], 3)
                good &= y[j:m + j] < 4
            chunk.append(code[good])
        if len(chunk) > 800:
            flush()
    flush()
    offs = np.arange(1, k + 1)
    left = np.zeros(len(pos0), np.int64)
    right = np.zeros(len(pos0), np.int64)
    for o in range(k, 0, -1):
        left = left * 4 + ref.codes[(pos0 - o) % ref.L]
    for o in offs:
        right = right * 4 + ref.codes[(pos0 + o) % ref.L]
    base = left * 4 ** (k + 1) + right
    cnt = np.stack([tab[base + c * 4 ** k] for c in range(4)], 1).astype(float)
    p = (cnt + 0.5) / (cnt.sum(1, keepdims=True) + 2.0)
    r = np.arange(len(pos0))
    return -(np.log(p[r, alt_c]) - np.log(p[r, ref_c]))


def get_conservation(a, ref, genomes, accs, mammal):
    """Conservation counts, cached in WORKDIR/cons_counts.npz (delete it if the corpus or the aligner changes)."""
    p = wp(a, "cons_counts.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["all"], z["mam"]
    cvt, cm = conservation_counts(a, ref, genomes, accs, mammal)
    np.savez_compressed(p, all=cvt, mam=cm)
    return cvt, cm


def gnomad_filtered_positions(a):
    """1-based positions with ANY non-PASS gnomAD record (artifact-prone sites etc.): absence there is not evidence of selection."""
    p = wp(a, "labels", "gnomad_filtered_pos.txt")
    if not os.path.exists(p):
        pos = set()
        with gzip.open(wp(a, "raw", "gnomad_chrM.vcf.bgz"), "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                c = line.split("\t", 8)
                if c[6] != "PASS":
                    pos.add(int(c[1]))
        open(p, "w").write("\n".join(map(str, sorted(pos))) + "\n")
    return {int(x) for x in open(p).read().split()}


def annotation_frame(ref, pos0, ref_c, alt_c):
    import pandas as pd
    region = ref.region[pos0]
    info = [ref.consequence(int(p), BASES[int(al)]) for p, al in zip(pos0, alt_c)]
    A = pd.DataFrame({"reg_prot": region == 1, "reg_trna": region == 2, "reg_rrna": region == 3,
                      "cp1": [c[0] == 1 for c in info], "cp2": [c[0] == 2 for c in info], "cp3": [c[0] == 3 for c in info],
                      "syn": [c[1] == "syn" for c in info], "mis": [c[1] == "mis" for c in info], "stop": [c[1] in ("stopgain", "stoploss") for c in info],
                      "blosum": [c[2] for c in info],
                      "transition": [(int(r), int(al)) in {(0, 2), (2, 0), (1, 3), (3, 1)} for r, al in zip(ref_c, alt_c)]}).astype(float)
    for r0 in range(4):
        for t0 in range(4):
            if r0 != t0:
                A[f"sub_{BASES[r0]}{BASES[t0]}"] = ((ref_c == r0) & (alt_c == t0)).astype(float)
    return A, region


def matched_auc(y, s, strata, min_n=5):
    """AUROC computed WITHIN strata (same substitution + flanking bases) and averaged with weights n1*n0: removes first-order mutation-rate effects.
    y = 1 for the class the score should rank higher."""
    from scipy.stats import rankdata
    o = np.argsort(strata, kind="stable")
    st, yy, ss = strata[o], y[o], s[o]
    cuts = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    num, den = 0.0, 0.0
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        n1 = int(yy[a0:b0].sum())
        n0 = (b0 - a0) - n1
        if n1 >= min_n and n0 >= min_n:
            r = rankdata(ss[a0:b0])
            num += (n1 * n0) * ((r[yy[a0:b0] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            den += n1 * n0
    return num / den if den else float("nan")


def stage_constraint(a, sizes):
    """Population-constraint task: of all ~49,700 possible mtDNA SNVs, which were never seen among 56,434 gnomAD individuals? Compared WITHIN
    substitution + flanking-base strata, so mutation-rate differences cannot explain a score. ~100x more labels than ClinVar pathogenic variants."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rep = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        rep.append(s)
    ref = Reference(wp(a, "raw", "human.gb"))
    gn = pd.read_csv(wp(a, "labels", "gnomad_mt.tsv"), sep="\t")
    tot = gn.AC_hom + gn.AC_het
    observed = {(int(p), al) for p, al, t in zip(gn.pos, gn.alt, tot) if t >= a.min_carriers}
    excl = gnomad_filtered_positions(a)
    L = ref.L
    valid = np.flatnonzero(ref.codes < 4)                                         # rCRS position 3107 is an 'N' placeholder
    alts = np.array([[c for c in range(4) if c != ref.codes[p]] for p in valid])
    pos0 = np.repeat(valid, 3)
    ref_c, alt_c = np.repeat(ref.codes[valid], 3).astype(int), alts.ravel()
    keep = np.array([(p + 1) not in excl for p in pos0])
    pos0, ref_c, alt_c = pos0[keep], ref_c[keep], alt_c[keep]
    y = np.array([(int(p) + 1, BASES[int(al)]) not in observed for p, al in zip(pos0, alt_c)]).astype(int)       # 1 = never observed (selected against)
    say(f"constraint task: {len(y)} possible SNVs at {len(set(pos0))} positions after excluding {len(excl)} artifact-prone positions; "
        f"observed in >= {a.min_carriers} individual(s): {int((1 - y).sum())} ({(1 - y).mean():.1%}), never observed: {int(y.sum())}")
    if len(y) > a.constraint_n:
        sel = np.random.default_rng(0).choice(len(y), a.constraint_n, replace=False)
        pos0, ref_c, alt_c, y = pos0[sel], ref_c[sel], alt_c[sel], y[sel]
        say(f"  scored subsample: {len(y)} variants ({int(y.sum())} never observed)")
    left, right = ref.codes[(pos0 - 1) % L].astype(int), ref.codes[(pos0 + 1) % L].astype(int)
    strata = ((ref_c * 4 + alt_c) * 4 + left) * 4 + right
    A, region = annotation_frame(ref, pos0, ref_c, alt_c)
    blocks = pos0 // 100
    genomes, val, mammal, accs = load_corpus(a)
    cvt, cm = get_conservation(a, ref, genomes, accs, mammal)
    md = 5 if a.smoke else 20
    cons_m, cons_mr, _ = cons_scores(cm, pos0, ref_c, alt_c, md)
    cons_v, _, _ = cons_scores(cvt, pos0, ref_c, alt_c, md)
    ng = ngram_scores([g for g, v in zip(genomes, val) if not v], ref, pos0, ref_c, alt_c)
    lm = {}
    for sz in sizes:
        s = lm_scores(a, ref, sz, pos0, ref_c, alt_c)
        if s is None:
            say(f"(no checkpoint for size {sz}; skipped)")
        else:
            lm[sz] = s
    ok = ~np.isnan(cons_m) & ~np.isnan(cons_v)
    say(f"  variants with conservation depth >= {md}: {int(ok.sum())}")
    pos0, ref_c, alt_c, y, strata, region, blocks = pos0[ok], ref_c[ok], alt_c[ok], y[ok], strata[ok], region[ok], blocks[ok]
    A = A[ok].reset_index(drop=True)
    cons_m, cons_v, cons_mr, ng = cons_m[ok], cons_v[ok], cons_mr[ok], ng[ok]
    lm = {k: v[ok] for k, v in lm.items()}

    def cv_pred(F):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, blocks):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
            m.fit(F.iloc[tr], y[tr])
            p[te] = m.predict_proba(F.iloc[te])[:, 1]
        return p
    F2 = A.copy()
    F2["cons_m"], F2["cons_v"], F2["cons_mr"] = cons_m, cons_v, cons_mr
    scores = {"conservation (mammals)": cons_m, "conservation (vertebrates)": cons_v, "control: context n-gram (zero-shot)": ng}
    for sz, s in lm.items():
        scores[f"language model {sz} (zero-shot)"] = s
    scores["annotation (LR, CV)"] = cv_pred(A)
    scores["annotation + conservation (LR, CV)"] = cv_pred(F2)
    F3 = F2.copy()
    F3["ngram"] = ng
    scores["annotation + conservation + n-gram control (LR, CV)"] = cv_pred(F3)
    for sz, s in lm.items():
        F4 = F2.copy()
        F4["lm"] = s
        scores[f"annotation + conservation + LM {sz} (LR, CV)"] = cv_pred(F4)
    say(f"\n=== Population constraint (never-observed vs observed SNVs), strata = substitution x flanking bases ({len(np.unique(strata))} strata)")
    say(f"{'method':58s} {'matched AUROC [95% CI]':26s} {'protein':>8s} {'tRNA':>6s} {'rRNA':>6s} {'unmatched':>9s}")
    rows = []
    for nm, s in scores.items():
        lo, hi = block_boot(lambda ii: matched_auc(y[ii], s[ii], strata[ii]), blocks, n=a.boot)
        per = [matched_auc(y[region == r], s[region == r], strata[region == r]) for r in (1, 2, 3)]
        m0 = matched_auc(y, s, strata)
        rows.append(dict(method=nm, matched_auroc=m0, lo=lo, hi=hi, protein=per[0], trna=per[1], rrna=per[2], unmatched=auc(y, s), n=len(y)))
        say(f"{nm:58s} {m0:.3f} [{lo:.3f},{hi:.3f}]        {per[0]:8.3f} {per[1]:6.3f} {per[2]:6.3f} {auc(y, s):9.3f}")
    base, ctrl = "annotation + conservation (LR, CV)", "annotation + conservation + n-gram control (LR, CV)"
    d = lambda k1, k2: (lambda ii: matched_auc(y[ii], scores[k1][ii], strata[ii]) - matched_auc(y[ii], scores[k2][ii], strata[ii]))
    full = np.arange(len(y))
    lo, hi = block_boot(d(ctrl, base), blocks, n=a.boot)
    say(f"  control: does the trivial n-gram score add to annotation + conservation?  matched-AUROC difference {d(ctrl, base)(full):+.3f} [{lo:+.3f},{hi:+.3f}]")
    for sz in lm:
        k = f"annotation + conservation + LM {sz} (LR, CV)"
        lo, hi = block_boot(d(k, ctrl), blocks, n=a.boot)
        zs_lm, zs_ng = matched_auc(y, lm[sz], strata), matched_auc(y, ng, strata)
        say(f"  KEY TEST: LM {sz} minus n-gram control, both stacked on annotation + conservation: {d(k, ctrl)(full):+.3f} [{lo:+.3f},{hi:+.3f}]")
        say(f"  VERDICT {sz}: " + ("LM adds information beyond annotation + conservation + a trivial context model." if (lo > 0 and zs_lm > zs_ng) else
            f"NOT demonstrated (needs key-test interval above 0 AND zero-shot matched AUROC above the n-gram control's: here {zs_lm:.3f} vs {zs_ng:.3f})."))
    say("\nCaveats: absence from gnomAD is also driven by sampling (56,434 people, mostly of European ancestry) and by calling filters; strata remove first-order mutation-rate effects only.")
    pd.DataFrame(rows).to_csv(wp(a, "results", "metrics_constraint.csv"), index=False)
    open(wp(a, "results", "report_constraint.txt"), "w").write("\n".join(rep) + "\n")
    log(f"wrote {wp(a, 'results', 'report_constraint.txt')}")


COLS = {"pos": ("pos", "position", "location", "start", "chromstart", "mt_pos", "m_pos"), "ref": ("ref", "reference", "ref_allele", "refallele"),
        "alt": ("alt", "alternate", "alt_allele", "altallele", "mut_allele"), "vaf": ("vaf", "af", "frequency", "heteroplasmy", "hl", "allele_fraction", "maf", "hf"),
        "sample": ("sample", "sample_id", "patient", "patient_id", "tumor_sample_barcode", "case")}


def read_variant_table(path):
    """TSV/CSV with pos/ref/alt (rCRS / GRCh38 chrM, 1-based), optional sample and vaf (heteroplasmy; percent is converted). Also accepts a
    'variant' / 'mutation' column like 'm.8993T>G' or '8993T>G'."""
    import pandas as pd
    d = pd.read_csv(path, sep=None, engine="python")
    low = {c.lower().strip(): c for c in d.columns}
    out = {}
    for k, names in COLS.items():
        for n in names:
            if n in low:
                out[k] = d[low[n]]
                break
    if not {"pos", "ref", "alt"} <= set(out):
        for n in ("variant", "mutation", "hgvs", "name"):
            if n in low:
                m = d[low[n]].astype(str).str.extract(r"(?:m\.|chrM:|MT:)?(\d+)\s*([ACGT])\s*>\s*([ACGT])")
                if m.notna().all(axis=1).mean() > 0.5:
                    out["pos"], out["ref"], out["alt"] = m[0].astype(float), m[1], m[2]
                    break
    if not {"pos", "ref", "alt"} <= set(out):
        sys.exit(f"{path}: need columns pos/ref/alt (or a 'variant' column like m.8993T>G). Found: {list(d.columns)}")
    v = pd.DataFrame({k: s.values for k, s in out.items()})
    v["ref"], v["alt"] = v["ref"].astype(str).str.upper(), v["alt"].astype(str).str.upper()
    v = v[v.pos.notna() & v.ref.isin(BASES) & v.alt.isin(BASES)].copy()
    v["pos"] = v.pos.astype(int)
    if "vaf" in v:
        v["vaf"] = pd.to_numeric(v["vaf"], errors="coerce")
        if v.vaf.max() > 1.0:
            v["vaf"] = v.vaf / 100.0
    return v.reset_index(drop=True)


def stage_score(a, sizes):
    """Score ANY table of mtDNA SNVs (e.g. somatic calls from AML samples, with heteroplasmy). Per variant: region, consequence, conservation, a 'damage'
    score fitted on population-constraint labels, and its percentile among all possible SNVs with the same substitution + flanking bases + region.
    Then (a) selection test: are the observed variants more / less damaging than mutation-matched random SNVs? (b) damage vs heteroplasmy."""
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ref = Reference(wp(a, "raw", "human.gb"))
    L = ref.L
    gn = pd.read_csv(wp(a, "labels", "gnomad_mt.tsv"), sep="\t")
    observed = {(int(p), al) for p, al, t in zip(gn.pos, gn.alt, gn.AC_hom + gn.AC_het) if t >= a.min_carriers}
    excl = gnomad_filtered_positions(a)
    valid = np.flatnonzero(ref.codes < 4)
    alts = np.array([[c for c in range(4) if c != ref.codes[p]] for p in valid])
    b_pos, b_ref, b_alt = np.repeat(valid, 3), np.repeat(ref.codes[valid], 3).astype(int), alts.ravel()
    keep = np.array([(p + 1) not in excl for p in b_pos])
    b_pos, b_ref, b_alt = b_pos[keep], b_ref[keep], b_alt[keep]
    b_y = np.array([(int(p) + 1, BASES[int(al)]) not in observed for p, al in zip(b_pos, b_alt)]).astype(int)
    genomes, val, mammal, accs = load_corpus(a)
    cvt, cm = get_conservation(a, ref, genomes, accs, mammal)
    lmz = {}

    def features(pos0, ref_c, alt_c):
        A, region = annotation_frame(ref, pos0, ref_c, alt_c)
        F = A[[c for c in A.columns if not c.startswith("sub_")]].copy()
        cmv, cmr, depth = cons_scores(cm, pos0, ref_c, alt_c, 5 if a.smoke else 20)
        cvv, _, _ = cons_scores(cvt, pos0, ref_c, alt_c, 5 if a.smoke else 20)
        F["cons_m"], F["cons_v"], F["cons_mr"] = cmv, cvv, cmr
        for sz in sizes:
            s = lm_scores(a, ref, sz, pos0, ref_c, alt_c)
            if s is not None:
                F[f"lm_{sz}"] = s
        return F, region, A
    Fb, region_b, _ = features(b_pos, b_ref, b_alt)
    okb = Fb.notna().all(axis=1).values
    Fb, b_pos, b_ref, b_alt, b_y, region_b = Fb[okb].reset_index(drop=True), b_pos[okb], b_ref[okb], b_alt[okb], b_y[okb], region_b[okb]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    b_score = np.zeros(len(b_y))
    for tr, te in GroupKFold(5).split(Fb, b_y, b_pos // 100):                    # out-of-fold, so the background is not scored by a model that saw its labels
        b_score[te] = mk().fit(Fb.iloc[tr], b_y[tr]).predict_proba(Fb.iloc[te])[:, 1]
    final = mk().fit(Fb, b_y)
    log(f"damage score fitted on {len(b_y)} possible SNVs ({int(b_y.sum())} never observed); features: {list(Fb.columns)}")
    lft, rgt = ref.codes[(b_pos - 1) % L].astype(int), ref.codes[(b_pos + 1) % L].astype(int)
    key = lambda r_, a_, l_, rr_, reg_: (((r_ * 4 + a_) * 4 + l_) * 4 + rr_) * 4 + reg_
    bkey = key(b_ref, b_alt, lft, rgt, region_b)
    groups = {}
    for k_ in np.unique(bkey):
        groups[int(k_)] = np.sort(b_score[bkey == k_])
    ctx_groups = {}
    ckey = key(b_ref, b_alt, lft, rgt, 0) // 4
    for k_ in np.unique(ckey):
        ctx_groups[int(k_)] = np.sort(b_score[ckey == k_])

    def pct(score, k_, ck_):
        g = groups.get(int(k_))
        if g is None or len(g) < 20:
            g = ctx_groups.get(int(ck_))
        if g is None or len(g) < 20:
            return np.nan, np.nan
        return (np.searchsorted(g, score, "left") + np.searchsorted(g, score, "right")) / (2.0 * len(g)), len(g)

    def run(v, name, say):
        v = v[(v.pos >= 1) & (v.pos <= L)].copy()
        v["pos0"] = v.pos - 1
        v = v[[ref.seq[p] == r for p, r in zip(v.pos0, v.ref)]].copy()
        if v.empty:
            say(f"{name}: no usable variants after reference check")
            return v
        r_c = np.array([CODE[x] for x in v.ref])
        a_c = np.array([CODE[x] for x in v.alt])
        F, region, _ = features(v.pos0.values, r_c, a_c)
        F = F.reindex(columns=Fb.columns)
        okv = F.notna().all(axis=1).values
        v, F, region, r_c, a_c = v[okv].copy(), F[okv], region[okv], r_c[okv], a_c[okv]
        v["damage"] = final.predict_proba(F)[:, 1]
        L_, R_ = ref.codes[(v.pos0.values - 1) % L].astype(int), ref.codes[(v.pos0.values + 1) % L].astype(int)
        kk = key(r_c, a_c, L_, R_, region)
        res = [pct(s, k_, k_ // 4) for s, k_ in zip(v.damage.values, kk)]
        v["percentile"], v["bg_n"] = [r[0] for r in res], [r[1] for r in res]
        v["region"] = np.array(["noncoding", "protein", "tRNA", "rRNA"])[region]
        v["cons_mammal"] = F["cons_m"].values
        v = v[v.percentile.notna()].copy()
        n = len(v)
        obs = v.percentile.mean()
        rng = np.random.default_rng(0)
        sizes_g = np.array([len(groups.get(int(k_)) if groups.get(int(k_)) is not None and len(groups[int(k_)]) >= 20 else ctx_groups[int(k_) // 4]) for k_ in kk[:n]])
        null = np.array([((rng.integers(0, sizes_g) + 0.5) / sizes_g).mean() for _ in range(2000)])
        p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (len(null) + 1)
        say(f"{name}: n={n}; mean damage percentile {obs:.3f} (0.5 = same as mutation-matched random SNVs; null sd {null.std():.3f}); permutation p = {p:.4f}; "
            f"fraction in top decile {np.mean(v.percentile >= 0.9):.2f} (null 0.10), bottom decile {np.mean(v.percentile <= 0.1):.2f}")
        for reg in ("protein", "tRNA", "rRNA", "noncoding"):
            s_ = v[v.region == reg]
            if len(s_) >= 5:
                say(f"    {reg:10s} n={len(s_):4d} mean percentile {s_.percentile.mean():.3f}")
        if "vaf" in v and v.vaf.notna().sum() >= 10:
            from scipy.stats import rankdata
            w = v.dropna(subset=["vaf"])

            def partial(x, y_, z):
                """Spearman between x and y_ after removing the linear effect of rank(z) from both ranks."""
                rx, ry, rz = rankdata(x), rankdata(y_), rankdata(z)
                A_ = np.c_[np.ones(len(rz)), rz]
                ex = rx - A_ @ np.linalg.lstsq(A_, rx, rcond=None)[0]
                ey = ry - A_ @ np.linalg.lstsq(A_, ry, rcond=None)[0]
                return float(np.corrcoef(ex, ey)[0, 1])
            has_cov = "covar" in w and w.covar.notna().all()
            units = w["sample"].values if "sample" in w else np.arange(len(w))
            ub = np.unique(units)
            idx = {u: np.flatnonzero(units == u) for u in ub}
            bs, bp = [], []
            for _ in range(500):
                ii = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
                bs.append(spearmanr(w.percentile.values[ii], w.vaf.values[ii])[0])
                if has_cov:
                    bp.append(partial(w.percentile.values[ii], w.vaf.values[ii], w.covar.values[ii]))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            say(f"    damage vs heteroplasmy: Spearman {spearmanr(w.percentile, w.vaf)[0]:+.3f} [{lo:+.3f},{hi:+.3f}] over {len(w)} variants "
                f"(negative = more damaging variants held at lower heteroplasmy, as expected under purifying selection)")
            if has_cov:
                lo, hi = np.nanpercentile(bp, [2.5, 97.5])
                say(f"    ...controlling for the covariate (carrier count / read depth): partial Spearman {partial(w.percentile.values, w.vaf.values, w.covar.values):+.3f} [{lo:+.3f},{hi:+.3f}]")
        return v
    rep = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        rep.append(s)
    if a.self_test:
        cv = pd.read_csv(wp(a, "labels", "clinvar_mt.tsv"), sep="\t")
        cv = cv[cv.stars >= 1]
        say("SELF-TEST on known sets (expected: ClinVar pathogenic high, benign and gnomAD-common low):")
        run(cv[cv["class"] == "P"][["pos", "ref", "alt"]], "ClinVar pathogenic / likely pathogenic", say)
        run(cv[cv["class"] == "B"][["pos", "ref", "alt"]], "ClinVar benign / likely benign", say)
        run(gn[gn.AF_hom >= 1e-3][["pos", "ref", "alt"]], "gnomAD common (AF_hom >= 0.001)", say)
        het = gn[(gn.AC_het >= 3) & (gn.AF_het > 0)].copy()
        het["vaf"] = het.max_hl
        het["covar"] = np.log10(het.AC_hom + het.AC_het)                          # max heteroplasmy rises with the number of carriers, damaging variants are rarer
        run(het[["pos", "ref", "alt", "vaf", "covar"]], "gnomAD heteroplasmic variants (vaf = max heteroplasmy)", say)
    if a.variants:
        res = run(read_variant_table(a.variants), os.path.basename(a.variants), say)
        outp = a.out_table or (os.path.splitext(a.variants)[0] + ".scored.tsv")
        res.to_csv(outp, sep="\t", index=False)
        say(f"per-variant scores written to {outp}")
    os.makedirs(wp(a, "results"), exist_ok=True)
    open(wp(a, "results", "report_score.txt"), "w").write("\n".join(rep) + "\n")


def lm_scores(a, ref, size, pos0, ref_c, alt_c):
    """Zero-shot masked-marginal LLR (alt - ref) at the centre of a 512-nt circular window, forward and reverse strand averaged. Higher = more damaging."""
    torch = get_torch()
    ck = wp(a, "ckpt", f"{size}.pt")
    if not os.path.exists(ck):
        return None
    s = torch.load(ck, map_location="cpu", weights_only=True)
    dev = pick_device(a)
    model = build_model(s["cfg"]).to(dev)
    model.load_state_dict(s["model"])
    model.eval()
    half = WIN // 2
    idx = (pos0[:, None] - half + np.arange(WIN)[None, :]) % ref.L
    fwd = ref.codes[idx]
    rev = revcomp_codes(fwd).copy()
    out = np.zeros(len(pos0))
    for strand, win, ce, r_c, a_c in ((0, fwd, half, ref_c, alt_c), (1, rev, WIN - 1 - half, 3 - ref_c, 3 - alt_c)):
        tok = win.astype(np.int64) + 1
        tok[:, ce] = MASK
        sc = np.zeros(len(pos0))
        with torch.inference_mode():
            for i in range(0, len(tok), 256):
                lg = model(torch.from_numpy(tok[i:i + 256]).to(dev))[:, ce].float().log_softmax(-1).cpu().numpy()
                r_i, a_i = r_c[i:i + 256], a_c[i:i + 256]
                sc[i:i + 256] = lg[np.arange(len(lg)), a_i] - lg[np.arange(len(lg)), r_i]
        out += sc / 2
    return -out                                                                   # damaging = alt much less likely than ref


# --------------------------------------------------------------------------------------------------------------------- evaluation
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def strat_auc(y, s, region):
    """AUROC within each region (protein / tRNA / rRNA) averaged with weights = number of pathogenic variants: removes the 'tRNA vs protein' shortcut."""
    num, den = 0.0, 0.0
    for r in (1, 2, 3):
        m = region == r
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos >= 5 and nneg >= 5:
            num += npos * auc(y[m], s[m])
            den += npos
    return num / den if den else float("nan")


def block_boot(fn, blocks, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx = {b: np.flatnonzero(blocks == b) for b in ub}
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub), replace=True)])
        vals.append(fn(ii))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def stage_evaluate(a, sizes):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    os.makedirs(wp(a, "results"), exist_ok=True)
    rep = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        rep.append(s)
    ref = Reference(wp(a, "raw", "human.gb"))
    cv = pd.read_csv(wp(a, "labels", "clinvar_mt.tsv"), sep="\t")
    gn = pd.read_csv(wp(a, "labels", "gnomad_mt.tsv"), sep="\t")
    for d, nm in ((cv, "ClinVar"), (gn, "gnomAD")):
        bad = (d.pos < 1) | (d.pos > ref.L)
        ok = ~bad & (np.array([ref.seq[p - 1] if 1 <= p <= ref.L else "?" for p in d.pos]) == d.ref.values)
        say(f"{nm}: {len(d)} SNVs, reference allele matches rCRS for {ok.sum()} ({ok.mean():.1%}); non-matching dropped")
        d.drop(d.index[~ok], inplace=True)
    cvl = cv[(cv.stars >= 1) & cv["class"].isin(["P", "B"])].copy()
    cvl["y"] = (cvl["class"] == "P").astype(int)
    cvl["set"] = "clinvar"
    pvar = set(zip(cvl[cvl.y == 1].pos, cvl[cvl.y == 1].alt))
    common = gn[(gn.AF_hom >= 1e-3)].copy()
    common = common[[(p, al) not in pvar for p, al in zip(common.pos, common.alt)]]
    common = common[~common.set_index(["pos", "alt"]).index.isin(cvl.set_index(["pos", "alt"]).index)]
    common["y"], common["set"], common["class"], common["stars"], common["gene"] = 0, "gnomad_common", "B", 0, ""
    pos_set = cvl[cvl.y == 1]
    say(f"labels: ClinVar P/LP (>=1 star) {len(pos_set)} at {pos_set.pos.nunique()} positions; B/LB {int((cvl.y == 0).sum())}; gnomAD common (AF_hom>=1e-3, not ClinVar) {len(common)}")
    allv = pd.concat([cvl[["pos", "ref", "alt", "y", "set", "gene"]], common[["pos", "ref", "alt", "y", "set", "gene"]]], ignore_index=True)
    allv["pos0"] = allv.pos - 1
    allv["ref_c"] = [CODE[x] for x in allv.ref]
    allv["alt_c"] = [CODE[x] for x in allv.alt]
    allv["region"] = ref.region[allv.pos0.values]
    cvg = allv.gene.fillna("")
    cvr = np.where(cvg.str.match(r"^MT-T[A-Z]\d?$"), 2, np.where(cvg.str.startswith("MT-RNR"), 3, np.where(cvg.str.startswith("MT-"), 1, -1)))
    chk = cvr >= 0
    say(f"region cross-check (GenBank annotation vs ClinVar gene name): agree for {(cvr[chk] == allv.region.values[chk]).mean():.1%} of {int(chk.sum())} annotated variants")
    allv["block"] = allv.pos0 // 100

    # annotation features
    cons_info = [ref.consequence(p, al) for p, al in zip(allv.pos0, allv.alt)]
    A = pd.DataFrame({"reg_prot": (allv.region == 1), "reg_trna": (allv.region == 2), "reg_rrna": (allv.region == 3),
                      "cp1": [c[0] == 1 for c in cons_info], "cp2": [c[0] == 2 for c in cons_info], "cp3": [c[0] == 3 for c in cons_info],
                      "syn": [c[1] == "syn" for c in cons_info], "mis": [c[1] == "mis" for c in cons_info],
                      "stop": [c[1] in ("stopgain", "stoploss") for c in cons_info], "blosum": [c[2] for c in cons_info],
                      "transition": [(r, al) in {(0, 2), (2, 0), (1, 3), (3, 1)} for r, al in zip(allv.ref_c, allv.alt_c)]}).astype(float)
    for r0 in BASES:                                                              # the 12 exact substitution types: a barely trained LM score mostly encodes these
        for t0 in BASES:
            if r0 != t0:
                A[f"sub_{r0}{t0}"] = ((allv.ref == r0) & (allv.alt == t0)).values.astype(float)
    # conservation baselines
    genomes, val, mammal, accs = load_corpus(a)
    cvt, cm = get_conservation(a, ref, genomes, accs, mammal)
    md = 5 if a.smoke else 20
    allv["cons_mammal"], allv["cons_mammal_ref"], allv["depth_mammal"] = cons_scores(cm, allv.pos0.values, allv.ref_c.values, allv.alt_c.values, md)
    allv["cons_vert"], allv["cons_vert_ref"], _ = cons_scores(cvt, allv.pos0.values, allv.ref_c.values, allv.alt_c.values, md)
    # language models
    lm_cols = []
    for sz in sizes:
        s = lm_scores(a, ref, sz, allv.pos0.values, allv.ref_c.values, allv.alt_c.values)
        if s is None:
            say(f"(no checkpoint for size {sz}; skipped)")
            continue
        allv[f"lm_{sz}"] = s
        lm_cols.append(f"lm_{sz}")
    allv["ngram"] = ngram_scores([g for g, v in zip(genomes, val) if not v], ref, allv.pos0.values, allv.ref_c.values, allv.alt_c.values)
    allv.to_csv(wp(a, "results", "variant_scores.csv"), index=False)
    rows = []

    def cv_pred(feats, d, y, groups):
        p = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(d, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
            m.fit(feats.iloc[tr], y[tr])
            p[te] = m.predict_proba(feats.iloc[te])[:, 1]
        return p

    for setname, sel in (("ClinVar P/LP vs B/LB", (allv["set"] == "clinvar").values), ("ClinVar P/LP vs B/LB + gnomAD common", np.ones(len(allv), bool))):
        d = allv[sel].reset_index(drop=True)
        Ad = A[sel].reset_index(drop=True)
        need = ["cons_mammal", "cons_vert", "ngram"] + lm_cols
        full = d[need].notna().all(axis=1).values
        d, Ad = d[full].reset_index(drop=True), Ad[full].reset_index(drop=True)
        y = d.y.values
        say(f"\n=== {setname}: n={len(d)} (pathogenic {int(y.sum())}, benign {int((1 - y).sum())}); common subset with every score available")
        blocks, region = d.block.values, d.region.values
        scores = {"conservation (mammals)": d.cons_mammal.values, "conservation (vertebrates)": d.cons_vert.values}
        scores["control: context n-gram (zero-shot)"] = d.ngram.values
        for c in lm_cols:
            scores[f"language model {c[3:]} (zero-shot)"] = d[c].values
        scores["annotation (LR, CV)"] = cv_pred(Ad, d, y, blocks)
        F2 = Ad.copy()
        F2["cons_m"], F2["cons_v"], F2["cons_mr"] = d.cons_mammal.values, d.cons_vert.values, d.cons_mammal_ref.values
        scores["annotation + conservation (LR, CV)"] = cv_pred(F2, d, y, blocks)
        F3 = F2.copy()
        F3["ngram"] = d.ngram.values
        scores["annotation + conservation + n-gram control (LR, CV)"] = cv_pred(F3, d, y, blocks)
        for c in lm_cols:
            F3 = F2.copy()
            F3["lm"] = d[c].values
            scores[f"annotation + conservation + LM {c[3:]} (LR, CV)"] = cv_pred(F3, d, y, blocks)
        say(f"{'method':58s} {'AUROC [95% CI]':24s} {'region-strat':>12s} {'protein':>8s} {'tRNA':>6s} {'rRNA':>6s} {'AP':>6s}")
        for nm, s in scores.items():
            lo, hi = block_boot(lambda ii: auc(y[ii], s[ii]), blocks, n=a.boot)
            pr = [auc(y[region == r], s[region == r]) if (y[region == r].sum() >= 5 and (1 - y[region == r]).sum() >= 5) else float("nan") for r in (1, 2, 3)]
            row = dict(set=setname, method=nm, auroc=auc(y, s), lo=lo, hi=hi, strat=strat_auc(y, s, region), protein=pr[0], trna=pr[1], rrna=pr[2],
                       ap=float(average_precision_score(y, s)), n=len(d), n_pos=int(y.sum()))
            rows.append(row)
            say(f"{nm:58s} {row['auroc']:.3f} [{lo:.3f},{hi:.3f}]   {row['strat']:12.3f} {pr[0]:8.3f} {pr[1]:6.3f} {pr[2]:6.3f} {row['ap']:6.3f}")
        base = "annotation + conservation (LR, CV)"
        for c in lm_cols:
            k = f"annotation + conservation + LM {c[3:]} (LR, CV)"
            diff = lambda ii, k=k: auc(y[ii], scores[k][ii]) - auc(y[ii], scores[base][ii])
            lo, hi = block_boot(diff, blocks, n=a.boot)
            say(f"  does LM {c[3:]} add to annotation + conservation?  AUROC difference {diff(np.arange(len(y))):+.3f} [{lo:+.3f},{hi:+.3f}]")
        ctrl = "annotation + conservation + n-gram control (LR, CV)"
        say(f"  control: does the trivial n-gram score add to annotation + conservation?  AUROC difference "
            f"{auc(y, scores[ctrl]) - auc(y, scores[base]):+.3f} [{block_boot(lambda ii: auc(y[ii], scores[ctrl][ii]) - auc(y[ii], scores[base][ii]), blocks, n=a.boot)[0]:+.3f},"
            f"{block_boot(lambda ii: auc(y[ii], scores[ctrl][ii]) - auc(y[ii], scores[base][ii]), blocks, n=a.boot)[1]:+.3f}]")
        for c in lm_cols:
            k = f"annotation + conservation + LM {c[3:]} (LR, CV)"
            diff = lambda ii, k=k: auc(y[ii], scores[k][ii]) - auc(y[ii], scores[ctrl][ii])
            lo, hi = block_boot(diff, blocks, n=a.boot)
            say(f"  KEY TEST: LM {c[3:]} minus n-gram control, both stacked on annotation + conservation: {diff(np.arange(len(y))):+.3f} [{lo:+.3f},{hi:+.3f}]")
            zs_lm, zs_ng = auc(y, d[c].values), auc(y, d.ngram.values)
            credible = lo > 0 and zs_lm > zs_ng
            say(f"  VERDICT {c[3:]}: " + ("LM adds information beyond annotation + conservation + a trivial context model."
                if credible else f"NOT demonstrated (needs key-test interval above 0 AND zero-shot AUROC above the n-gram control's: here {zs_lm:.3f} vs {zs_ng:.3f})."))
            diff = lambda ii, c=c: auc(y[ii], d[c].values[ii]) - auc(y[ii], d.cons_mammal.values[ii])
            lo, hi = block_boot(diff, blocks, n=a.boot)
            say(f"  zero-shot LM {c[3:]} minus mammal conservation (both zero-shot): {diff(np.arange(len(y))):+.3f} [{lo:+.3f},{hi:+.3f}]")
        say("  (a zero-shot AUROC below 0.5 means the score is anti-informative -- expected for an untrained or barely trained model; the sign is not flipped)")
    pd.DataFrame(rows).to_csv(wp(a, "results", "metrics.csv"), index=False)
    say("\nRead this with the caveats in the header of mito_lm.py: ~100 positives, region imbalance, partly circular ClinVar labels.")
    open(wp(a, "results", "report.txt"), "w").write("\n".join(rep) + "\n")
    log(f"wrote {wp(a, 'results', 'report.txt')}")


# --------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "download", "corpus", "pretrain", "evaluate", "constraint", "score"])
    ap.add_argument("--variants", default="", help="score stage: TSV/CSV of mtDNA SNVs (pos, ref, alt, optional sample, optional vaf/heteroplasmy)")
    ap.add_argument("--out-table", default="", help="score stage: where to write the per-variant table (default: next to the input)")
    ap.add_argument("--self-test", action="store_true", help="score stage: run the known-answer controls (ClinVar pathogenic/benign, gnomAD common)")
    ap.add_argument("--constraint-n", type=int, default=24000, help="variants scored in the population-constraint task (of ~49,700 possible)")
    ap.add_argument("--min-carriers", type=int, default=1, help="a variant counts as 'observed' in gnomAD if carried by at least this many individuals")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--sizes", default="small")
    ap.add_argument("--hours", type=float, default=4.0, help="time cap per size (the step count is fitted to it); 0 = no cap")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="micro-batch (windows per forward pass); the effective batch is --batch x --accum = 64")
    ap.add_argument("--accum", type=int, default=4, help="gradient-accumulation micro-batches per optimiser step")
    ap.add_argument("--mem-gb", type=float, default=0.0, help="stop cleanly (checkpoint saved) if GPU/process memory exceeds this; 0 = 55%% of RAM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--mammal-boost", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--email", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.workdir = a.workdir or ("mito_smoke" if a.smoke else "mito_work")
    sizes = a.sizes.split(",")
    if a.smoke:
        sizes, a.eval_every, a.boot = ["tiny"], 20, 100
    for s in sizes:
        if s not in SIZES:
            sys.exit(f"unknown size {s}; choose from {list(SIZES)}")
    os.makedirs(a.workdir, exist_ok=True)
    if a.stage in ("all", "download"):
        stage_download(a)
    if a.stage in ("all", "corpus"):
        stage_corpus(a)
    if a.stage in ("all", "pretrain"):
        for s in sizes:
            stage_pretrain(a, s)
    if a.stage in ("all", "evaluate"):
        stage_evaluate(a, sizes)
    if a.stage in ("all", "evaluate", "constraint"):
        stage_constraint(a, sizes)
    if a.stage == "score":
        if not (a.variants or a.self_test):
            sys.exit("score stage: give --variants FILE and/or --self-test")
        stage_score(a, sizes)


if __name__ == "__main__":
    main()
