
### all_v2.json:rinalmo-micro  minus  all_v2.json:kmer   (dev_all)
| task | metric | A | B | delta [95% CI] | p(A<=B) |
|---|---|---|---|---|---|
| loc_reg | spearman | 0.316 | 0.283 | +0.034 [-0.007,+0.078] | 0.055 |
| loc_clf | auroc | 0.794 | 0.754 | +0.040 [+0.018,+0.063] | 0.000 |
| tau_reg | spearman | 0.293 | 0.305 | -0.012 [-0.062,+0.033] | 0.728 |
| testis_clf | auroc | 0.617 | 0.576 | +0.042 [+0.001,+0.084] | 0.023 |
| brain_clf | auroc | 0.526 | 0.591 | -0.065 [-0.133,-0.002] | 0.976 |
| coding_clf | auroc | 0.921 | 0.816 | +0.105 [+0.074,+0.136] | 0.000 |
| aml_vs_normal_reg | spearman | 0.258 | 0.238 | +0.020 [-0.022,+0.063] | 0.203 |
| aml_vs_normal_resid_reg | spearman | 0.256 | 0.238 | +0.018 [-0.025,+0.063] | 0.200 |
| aml_vs_normal_resid_clf | auroc | 0.624 | 0.633 | -0.009 [-0.051,+0.035] | 0.651 |
| aml_npm1_reg | spearman | -0.015 | -0.021 | +0.006 [-0.090,+0.101] | 0.431 |

### all_v2.json:full_3000  minus  all_v2.json:kmer   (dev_all)
| task | metric | A | B | delta [95% CI] | p(A<=B) |
|---|---|---|---|---|---|
| loc_reg | spearman | 0.204 | 0.283 | -0.079 [-0.145,-0.009] | 0.987 |
| loc_clf | auroc | 0.777 | 0.754 | +0.023 [-0.004,+0.051] | 0.039 |
| tau_reg | spearman | 0.312 | 0.305 | +0.007 [-0.036,+0.049] | 0.368 |
| testis_clf | auroc | 0.572 | 0.576 | -0.004 [-0.054,+0.043] | 0.564 |
| brain_clf | auroc | 0.608 | 0.591 | +0.017 [-0.036,+0.066] | 0.265 |
| coding_clf | auroc | 0.789 | 0.816 | -0.027 [-0.056,+0.000] | 0.974 |
| aml_vs_normal_reg | spearman | 0.277 | 0.238 | +0.038 [-0.018,+0.091] | 0.088 |
| aml_vs_normal_resid_reg | spearman | 0.235 | 0.238 | -0.002 [-0.053,+0.046] | 0.534 |
| aml_vs_normal_resid_clf | auroc | 0.595 | 0.633 | -0.038 [-0.079,+0.001] | 0.971 |
| aml_npm1_reg | spearman | -0.098 | -0.021 | -0.077 [-0.179,+0.021] | 0.937 |

### all_v2.json:rinalmo-micro  minus  all_v2.json:full_3000   (dev_all)
| task | metric | A | B | delta [95% CI] | p(A<=B) |
|---|---|---|---|---|---|
| loc_reg | spearman | 0.316 | 0.204 | +0.112 [+0.055,+0.170] | 0.000 |
| loc_clf | auroc | 0.794 | 0.777 | +0.017 [-0.007,+0.044] | 0.091 |
| tau_reg | spearman | 0.293 | 0.312 | -0.019 [-0.065,+0.027] | 0.813 |
| testis_clf | auroc | 0.617 | 0.572 | +0.046 [-0.002,+0.093] | 0.028 |
| brain_clf | auroc | 0.526 | 0.608 | -0.082 [-0.150,-0.016] | 0.992 |
| coding_clf | auroc | 0.921 | 0.789 | +0.132 [+0.101,+0.161] | 0.000 |
| aml_vs_normal_reg | spearman | 0.258 | 0.277 | -0.018 [-0.068,+0.033] | 0.757 |
| aml_vs_normal_resid_reg | spearman | 0.256 | 0.235 | +0.021 [-0.024,+0.071] | 0.218 |
| aml_vs_normal_resid_clf | auroc | 0.624 | 0.595 | +0.029 [-0.016,+0.074] | 0.099 |
| aml_npm1_reg | spearman | -0.015 | -0.098 | +0.083 [-0.008,+0.174] | 0.033 |

### RBP tasks: paired over RBPs, expression+length-stratified AUROC (A - B)
| task | A | B | mean delta [95% CI over RBPs] | RBPs A>B |
|---|---|---|---|---|
| rbp_K562 | rinalmo-micro | kmer | +0.055 [+0.033,+0.078] | 34/43 |
| rbp_K562 | full_3000 | kmer | +0.023 [-0.001,+0.046] | 27/43 |
| rbp_K562 | rinalmo-micro | full_3000 | +0.032 [+0.011,+0.054] | 25/43 |
| rbp_K562 | kmer | lengc | +0.074 [+0.042,+0.105] | 33/43 |
| rbp_HepG2 | rinalmo-micro | kmer | +0.038 [+0.021,+0.055] | 26/35 |
| rbp_HepG2 | full_3000 | kmer | +0.020 [+0.003,+0.038] | 24/35 |
| rbp_HepG2 | rinalmo-micro | full_3000 | +0.018 [+0.003,+0.035] | 23/35 |
| rbp_HepG2 | kmer | lengc | +0.074 [+0.051,+0.099] | 33/35 |
