# T1 — lncRNA essentiality: BLOCKED on data access (2026-11-24)

Goal: test whether lncRNA essentiality is predictable from sequence, before building the DNA/locus arm.
Status: **blocked**. The labels are not publicly downloadable. Four routes checked, none works.

## What I checked

| source | what it is | outcome |
|---|---|---|
| CRiNCL, Liu et al., Science 2017, DOI 10.1126/science.aah7111 | **the right dataset**: 16,401 lncRNA loci x 7 cell lines, CRISPRi growth phenotypes. Per-locus gamma, p-value and hit calls are in Supplementary **Table S4** | science.org returns **HTTP 403** on both the article and the supplement page. Table not reachable |
| PMC5394926 (the same paper, author manuscript) | public | only supplement is `NIHMS845370-supplement-supplement_1.pdf`, 12.4 MB. Holds Methods, Figs S1-S12 and the *captions* for Tables S1-S11. A 16k-row table is not realistically in it |
| GEO **GSE85011** (the study's deposit) | public FTP, three files | `chip_counts.txt.gz` (H3K9me3), `rnaseq_tpm.txt.gz`, `crincl_lncrna_transcripts.gtf.zip.gz`. The GTF is useful for mapping loci later, but **no growth phenotypes**. Raw sgRNA counts are in SRA SRP080367 only |
| Cas13 transcriptome-scale lncRNA screen | the alternative I preferred, because Cas13 knocks down the *transcript* and so avoids the neighbouring-gene confound | PMC12436244 is an **editorial**, not the screen. The primary study it describes (Liang WW et al., Cell 2024, DOI 10.1016/j.cell.2024.10.021) is **RETRACTED** (PubMed 41344327) |
| STAR Protocols CRISPRi lncRNA dropout screens (PMC10550846) | protocol paper | method only, no results table |
| search for a redistributed copy | — | none found. Smaller CRiNCL-library screens exist and are open (e.g. a monocyte screen, 38 hits; a pluripotency screen) but 38 hits is not a training set |

## Correction to something I told the user

I cited "778 essential lncRNAs, 46 across all five cell lines" as evidence this arm had data. That number comes
from a paper that has since been **retracted**. It should not be used.

## What would unblock this

One file: **Supplementary Table S4 of Science 2017;355:eaah7111**, from a browser with institutional access to
science.org. It is an xlsx of per-locus gamma / p-value / hit call across the 7 lines. With it, T1 runs as written.

Fallbacks if that is not available, in order of preference:
1. Ask the authors (Weissman / Lim) for Tables S3-S4 — routinely granted, but slower than this sprint.
2. Re-derive phenotypes from SRA SRP080367 raw sgRNA counts with the published library. Days of work and its own
   analysis risk; out of scope for a verification sprint.
3. Drop the DNA/locus arm. The family then has no novel member and becomes a benchmark-plus-compact-models paper.

## Decision taken

T1 is parked, not failed: its kill criterion was never reached, because the test could not be run. The sprint
moves to T2, which needs no new data. T1 resumes the moment the table is in hand.
