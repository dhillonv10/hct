"""Build the analysis-ready benchmark bundle (benchmark/data/) from the public GEO files used in this project.

This is the script AS RUN to produce the committed bundle.  It reads the project's cached GEO downloads and the loaders of the
pipeline (HCTFormer), z-scoring is NOT done here (the notebook does it, label-free, per study x compartment block).
What is stored: log-scale expression (native units of each study) for the 3,072 pipeline-vocabulary genes UNION the MSigDB Hallmark
members, plus sample metadata.  Genes a platform does not measure are NaN.
Accessions / files:  see DATA_CARD in the notebook.   Usage:  python build_bundle.py <scratch_root> <out_dir>
"""
import sys, os, io, csv, gzip, json, collections, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore')
SCR = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data'
sys.path.insert(0, SCR + '/t_new'); os.environ['HCT_ROOT'] = SCR + '/new_root3'; os.environ['HCT_PROFILE'] = 'smoke'
from load import load as load_hct          # loader shim around the pipeline source (cohort parsers with sha256-pinned GEO downloads)
h = load_hct()
SETS = json.load(open(SCR + '/hallmark/hallmark.json'))
vocab = h.load_json(h.WORK / 'vocab.json'); VG = [g.upper() for g in vocab['genes']]
KEEP = sorted(set(VG) | {g for s in SETS.values() for g in s})
pos = {g: i for i, g in enumerate(KEEP)}
co = h.load_cohorts(vocab)
os.makedirs(OUT, exist_ok=True); os.makedirs(SCR + '/bundle/fm_inputs', exist_ok=True)

arrays, rows, info = {}, [], {}


def to_keep(X, genes):
    """(n, g) expression in native units -> (n, len(KEEP)) float32, NaN where the platform does not measure the gene."""
    out = np.full((X.shape[0], len(KEEP)), np.nan, np.float32); seen = set()
    for j, g in enumerate(genes):
        g = str(g).upper()
        if g in pos and g not in seen:
            out[:, pos[g]] = X[:, j]; seen.add(g)
    return out


def add(study, X, genes, meta, platform, series, ctx, note):
    arrays['X_' + study] = to_keep(np.asarray(X, float), genes)
    m = meta.copy(); m.insert(0, 'study', study); m['sid'] = list(m['sid']); m['ctx'] = ctx; m['platform'] = platform; m['series'] = series
    rows.append(m)
    info[study] = dict(platform=platform, series=series, ctx=ctx, n_samples=len(m), native_genes=int(len(set(map(str, genes)))),
                       genes_in_bundle=int(np.isfinite(arrays['X_' + study]).any(0).sum()), note=note)


# ---------------------------------------------------------------------------- cohorts parsed by the pipeline
for cid in ('GSE73809', 'GSE103569', 'GSE4624', 'PTCY', 'GSE295359'):
    c = co[cid]; m = c.meta.copy(); m['sid'] = m.index.astype(str)
    m['block'] = m.compartment.astype(str); m['y'] = m.gvhd.astype(int); m['patient'] = m.patient.astype(str)
    cols = ['sid', 'patient', 'block', 'y'] + [k for k in ('timepoint', 'day_post_hct', 'batch', 'site', 'organ', 'relapse', 'cgvhd', 'remission', 'n_cells', 'array_order') if k in m.columns]
    plat = {'GSE73809': 'microarray (GPL17586)', 'GSE103569': 'microarray (GPL13497)', 'GSE4624': 'microarray (GPL3639)', 'PTCY': 'bulk RNA-seq (counts)',
            'GSE295359': 'scRNA-seq pseudobulk (3,072-gene release)'}[cid]
    series = {'PTCY': 'GSE182610+GSE182612+GSE182678'}.get(cid, cid)
    add(cid, c.native, [str(g) for g in c.native_genes], m[cols], plat, series, 'blood',
        {'GSE4624': 'PROGNOSTIC: donor T cells -> recipient aGVHD', 'PTCY': 'patient ids not provided by GEO (one id per sample)',
         'GSE295359': '10 patients x (timepoint, compartment); Zenodo release restricted to 3,072 genes'}.get(cid, 'diagnostic: GVHD at sampling'))

# ---------------------------------------------------------------------------- tissue biopsy studies
B = SCR + '/survey/bx/'


def sm(path):
    meta = collections.defaultdict(list)
    for line in gzip.open(path, 'rt', errors='replace'):
        if line.startswith('!Sample_'):
            f = next(csv.reader([line.rstrip('\n')], delimiter='\t')); meta[f[0]].append(f[1:])
    return meta


def chars(meta):
    return {row[0].split(': ')[0]: [r.split(': ', 1)[1] if ': ' in r else r for r in row] for row in meta['!Sample_characteristics_ch1']}


meta = sm(B + 'GSE215068/sm.gz'); ch = chars(meta); lab = dict(zip(meta['!Sample_title'][0], ch['disease state']))
b = pd.read_csv(io.BytesIO(gzip.open(B + 'GSE215068/fpkm.xls.gz').read()), sep='\t', index_col=0)
cols = [t for t in meta['!Sample_title'][0] if t in b.columns]; d = b[cols].copy(); d.index = b['Gene_name'].astype(str).str.upper().values
fp1 = d.groupby(level=0).sum().fillna(0); y1 = np.array([int(lab[c] == 'GVHD') for c in cols])
meta = sm(B + 'GSE168116/sm.gz'); ch = chars(meta); gsm = meta['!Sample_geo_accession'][0]; files = {f.split('_')[0]: f for f in os.listdir(B + 'GSE168116/x')}
cc = {}
for gm in gsm:
    t = pd.read_csv(B + 'GSE168116/x/' + files[gm], sep='\t', index_col=0).iloc[:, 0]; t.index = t.index.astype(str).str.upper(); cc[gm] = t.groupby(level=0).sum().fillna(0)
tp2 = pd.DataFrame(cc); y2 = np.array([int(s == 'GVHD') for s in ch['gvhd status']])
a = pd.read_csv(B + 'GSE263557/fpkm.csv.gz', index_col=0); a = a[~a.index.duplicated()].fillna(0.0); a.index = a.index.str.upper(); a = a[~a.index.duplicated()]
# QC (found during the benchmark audit): the supplementary matrix is log2(FPKM + 1e-5); four libraries have EVERY gene at the floor value (-16.61) = failed/empty libraries.
failed = [c for c in a.columns if a[c].std() < 1e-9]
qc_removed = [dict(study='SKIN263557', column=c, reason='every gene equals log2(1e-5): empty library') for c in failed]
a = a.drop(columns=failed)
a = np.log2(2.0 ** a + 1.0)         # back to log2(FPKM + 1), the same scale as the colon studies (removes the -16.6 zero floor)
sl = np.array([('NR' if 'No_Response' in c else 'R' if c.endswith('...Response') else 'C') for c in a.columns])


def tissue_meta(prefix, n, y, extra=None):
    m = pd.DataFrame({'sid': [f'{prefix}_{i}' for i in range(n)], 'patient': [f'{prefix}:{i}' for i in range(n)], 'block': 'biopsy', 'y': y})
    if extra is not None:
        for k, v in extra.items(): m[k] = v
    return m


add('COLON215068', np.log2(fp1.T.to_numpy(float) + 1), fp1.index, tissue_meta('COLON215068', len(y1), y1), 'bulk RNA-seq (FFPE, FPKM)', 'GSE215068', 'tissue',
    'diagnostic: colon biopsy GVHD vs no GVHD')
add('COLON168116', np.log2(tp2.T.to_numpy(float) + 1), tp2.index, tissue_meta('COLON168116', len(y2), y2), 'bulk RNA-seq (TPM)', 'GSE168116', 'tissue',
    'diagnostic: paediatric rectosigmoid biopsy GVHD vs no GVHD')
y3 = (sl != 'C').astype(int); nr = np.where(sl == 'NR', 1.0, np.where(sl == 'R', 0.0, np.nan))
import re
lib_order = [int(re.search(r'_S(\d+)', c).group(1)) if re.search(r'_S(\d+)', c) else -1 for c in a.columns]       # library-prep order (technical confounder)
add('SKIN263557', a.T.to_numpy(float), a.index, tissue_meta('SKIN263557', len(y3), y3, {'y_nr': nr, 'libprep_order': lib_order}), 'bulk RNA-seq (log2 FPKM supplementary matrix)', 'GSE263557', 'tissue',
    'diagnostic: skin GVHD vs control; PROGNOSTIC y_nr = day-28 steroid non-response (45 GVHD biopsies)')

# ---------------------------------------------------------------------------- plasma cfRNA (longitudinal)
CF = SCR + '/survey/cfrna/'
cm = pd.read_csv(CF + 'cfrna_meta.csv', index_col=0); cm = cm[cm['sample group'] != 'Healthy Donor'].copy()
cnt = pd.read_csv(CF + 'counts.csv.gz', index_col=0)[cm.index]
e2s = json.load(open(SCR + '/new_root3/work/ens2sym.json'))
sym = pd.Series(cnt.index.map(lambda e: e2s.get(str(e).split('.')[0])), index=cnt.index); ok = sym.notna().to_numpy()
bysym = cnt[ok].groupby(sym[ok].to_numpy().astype(str)).sum()
cpm = bysym.div(cnt.sum(0), axis=1) * 1e6
cpm = cpm[(cpm > 1).mean(1) >= 0.5]            # label-free expression filter (>=50% of samples with CPM>1)
lcpm = np.log1p(cpm)
m = pd.DataFrame({'sid': cm.index.astype(str), 'patient': cm['patient id'].astype(str).to_numpy(), 'block': 'plasma', 'y': cm['complications'].str.contains('agvhd').astype(int).to_numpy(),
                  'timepoint': cm['event sym'].astype(str).to_numpy(), 'batch': np.where(cm.index.str.startswith('gvhd_'), 'gvhd_', 'plain'),
                  'libsize_log10': np.log10(cnt.sum(0).to_numpy(float)), 'cgvhd': cm['complications'].str.contains('cgvhd').astype(int).to_numpy(),
                  'relapse': cm['complications'].str.contains('relapse').astype(int).to_numpy(), 'bkv': cm['complications'].str.contains('bkv').astype(int).to_numpy()})
add('CFRNA268164', lcpm.T.to_numpy(float), lcpm.index, m, 'plasma cfRNA-seq (log1p CPM)', 'GSE268164', 'plasma',
    'LONGITUDINAL: patient-level ever-aGVHD label repeated at each timepoint; onset date not in GEO')

# ---------------------------------------------------------------------------- write
np.savez_compressed(OUT + '/hsct_bundle.npz', genes=np.array(KEEP), **arrays, **{'sid_' + k[2:]: rows[i].sid.astype(str).to_numpy().astype('U') for i, k in enumerate(arrays)})
S_ = pd.concat(rows, ignore_index=True); S_.to_csv(OUT + '/hsct_samples.csv', index=False)
pd.DataFrame(qc_removed).to_csv(OUT + '/qc_removed.csv', index=False)
json.dump(SETS, open(OUT + '/hallmark50.json', 'w')); json.dump(VG, open(OUT + '/vocab_genes.json', 'w')); json.dump(info, open(OUT + '/bundle_info.json', 'w'), indent=1)
print(S_.groupby('study').agg(n=('sid', 'size'), pos=('y', 'sum'), patients=('patient', 'nunique')).to_string())
print({k: (v['native_genes'], v['genes_in_bundle']) for k, v in info.items()}, 'KEEP genes', len(KEEP))

# ---------------------------------------------------------------------------- full-transcriptome inputs for the Geneformer embedder (not committed)
def dump_fm(name, lin, genes, sid):
    np.savez_compressed(SCR + f'/bundle/fm_inputs/{name}.npz', expr=np.asarray(lin, np.float32), genes=np.array([str(g).upper() for g in genes]), sid=np.array(sid, dtype=str))


cP = co['PTCY']; kp = cP.meta.compartment.isin(['CD8', 'CD4_Tconv', 'Treg']).to_numpy()
dump_fm('PTCY', np.expm1(np.nan_to_num(cP.native[kp].astype(np.float64), nan=0.0)), cP.native_genes, cP.meta.index[kp].astype(str))
dump_fm('COLON215068', np.maximum(2 ** (np.log2(fp1.T.to_numpy(float) + 1)) - 1, 0), fp1.index, [f'COLON215068_{i}' for i in range(len(y1))])
dump_fm('COLON168116', np.maximum(tp2.T.to_numpy(float), 0), tp2.index, [f'COLON168116_{i}' for i in range(len(y2))])
dump_fm('SKIN263557', np.maximum(2 ** a.T.to_numpy(float) - 1, 0), a.index, [f'SKIN263557_{i}' for i in range(len(y3))])      # a is log2(FPKM+1) here
