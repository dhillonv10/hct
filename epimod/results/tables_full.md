## Held-out benchmark: all models

| model | MedAE | MedAE_18_90 | MedE | AA2_hits | AA2_tests | AA1_hits | AA1_tests | score |
|---|---|---|---|---|---|---|---|---|
| PhenoAgeV2 | 7.53 | 7.50 | -2.55 | 19 | 41 | 9 | 24 | 28.00 |
| PhenoAgeV1 | 8.00 | 7.99 | -4.19 | 9 | 41 | 7 | 24 | 16.00 |
| HorvathV1 | 5.34 | 5.33 | -0.04 | 3 | 41 | 12 | 24 | 15.00 |
| no_linear_route | 4.42 | 4.46 | 2.77 | 10 | 41 | 13 | 24 | 14.84 |
| Ridge(same panel) | 4.12 | 4.15 | -0.81 | 5 | 41 | 9 | 24 | 14.00 |
| HorvathV2 | 4.15 | 4.22 | 1.15 | 5 | 41 | 12 | 24 | 13.67 |
| no_aug | 4.06 | 4.10 | 1.28 | 5 | 41 | 12 | 24 | 13.22 |
| no_id | 4.27 | 4.32 | 2.45 | 6 | 41 | 16 | 24 | 12.82 |
| point_estimate | 4.06 | 4.10 | 1.85 | 5 | 41 | 14 | 24 | 12.61 |
| no_ssl | 3.98 | 4.02 | 2.04 | 5 | 41 | 15 | 24 | 12.30 |
| full | 3.93 | 3.96 | 2.03 | 5 | 41 | 15 | 24 | 12.25 |
| no_context | 3.84 | 3.88 | 1.86 | 6 | 41 | 12 | 24 | 12.17 |
| full_seed1 | 3.93 | 3.98 | 2.21 | 5 | 41 | 16 | 24 | 12.01 |
| Lin | 7.46 | 7.51 | 2.19 | 5 | 41 | 9 | 24 | 11.36 |
| ElasticNet(same panel) | 3.68 | 3.72 | 1.51 | 3 | 41 | 14 | 24 | 11.26 |
| no_module_mixing | 4.10 | 4.16 | 2.52 | 5 | 41 | 14 | 24 | 10.39 |
| Zhang19_EN | 10.60 | 10.46 | 9.73 | 2 | 41 | 19 | 24 | 3.56 |
| Hannum | 8.40 | 8.38 | 7.63 | 1 | 41 | 20 | 24 | 2.83 |

## Sensitivity: without sibling series of training studies (GSE72774, GSE72776)

| model | MedAE | MedAE_18_90 | MedE | AA2_hits | AA2_tests | AA1_hits | AA1_tests | score |
|---|---|---|---|---|---|---|---|---|
| PhenoAgeV2 | 7.92 | 7.91 | -2.14 | 16 | 39 | 9 | 24 | 25.00 |
| PhenoAgeV1 | 7.91 | 7.90 | -3.76 | 9 | 39 | 7 | 24 | 16.00 |
| Ridge(same panel) | 3.95 | 4.00 | -0.16 | 5 | 39 | 9 | 24 | 14.00 |
| HorvathV1 | 5.17 | 5.15 | 0.68 | 3 | 39 | 12 | 24 | 13.43 |
| no_linear_route | 4.69 | 4.75 | 3.15 | 9 | 39 | 13 | 24 | 13.27 |
| no_context | 4.04 | 4.07 | 2.20 | 7 | 39 | 12 | 24 | 12.47 |
| HorvathV2 | 4.21 | 4.25 | 1.67 | 5 | 39 | 12 | 24 | 12.25 |
| no_ssl | 4.21 | 4.26 | 2.36 | 5 | 39 | 15 | 24 | 11.60 |
| no_id | 4.53 | 4.60 | 2.97 | 6 | 39 | 16 | 24 | 11.51 |
| full | 4.14 | 4.20 | 2.36 | 5 | 39 | 15 | 24 | 11.45 |
| point_estimate | 4.23 | 4.31 | 2.29 | 5 | 39 | 14 | 24 | 11.44 |
| no_aug | 4.09 | 4.15 | 1.91 | 5 | 39 | 12 | 24 | 11.39 |
| full_seed1 | 4.15 | 4.20 | 2.64 | 5 | 39 | 16 | 24 | 10.81 |
| ElasticNet(same panel) | 3.86 | 3.91 | 1.80 | 3 | 39 | 14 | 24 | 10.48 |
| Lin | 7.60 | 7.64 | 3.05 | 5 | 39 | 9 | 24 | 10.39 |
| no_module_mixing | 4.33 | 4.37 | 2.87 | 5 | 39 | 14 | 24 | 9.73 |
| Zhang19_EN | 11.47 | 11.26 | 10.70 | 2 | 39 | 19 | 24 | 3.27 |
| Hannum | 9.05 | 8.99 | 8.38 | 1 | 39 | 20 | 24 | 2.48 |

## Ablations (identical budget; `full_seed1` is the same model with another init seed = noise floor)

| model | MedAE | MedAE_18_90 | MedE | AA2_hits | AA1_hits | score | dMedAE_vs_full | dScore_vs_full |
|---|---|---|---|---|---|---|---|---|
| no_linear_route | 4.42 | 4.46 | 2.77 | 10 | 13 | 14.84 | 0.49 | 2.59 |
| no_aug | 4.06 | 4.10 | 1.28 | 5 | 12 | 13.22 | 0.13 | 0.97 |
| no_id | 4.27 | 4.32 | 2.45 | 6 | 16 | 12.82 | 0.34 | 0.57 |
| point_estimate | 4.06 | 4.10 | 1.85 | 5 | 14 | 12.61 | 0.13 | 0.36 |
| no_ssl | 3.98 | 4.02 | 2.04 | 5 | 15 | 12.30 | 0.05 | 0.05 |
| full | 3.93 | 3.96 | 2.03 | 5 | 15 | 12.25 | 0.00 | 0.00 |
| no_context | 3.84 | 3.88 | 1.86 | 6 | 12 | 12.17 | -0.09 | -0.09 |
| full_seed1 | 3.93 | 3.98 | 2.21 | 5 | 16 | 12.01 | 0.00 | -0.25 |
| no_module_mixing | 4.10 | 4.16 | 2.52 | 5 | 14 | 10.39 | 0.17 | -1.87 |

## Healthy-control MedAE by subgroup (years)

| by | level | n | full | ElasticNet(same panel) | Ridge(same panel) | HorvathV1 | HorvathV2 | PhenoAgeV2 |
|---|---|---|---|---|---|---|---|---|
| age_band | 18-40 | 592 | 3.06 | 2.84 | 2.91 | 4.97 | 4.07 | 5.74 |
| age_band | 40-60 | 1107 | 4.86 | 3.91 | 3.17 | 4.67 | 4.57 | 7.23 |
| age_band | 60-90 | 1088 | 3.79 | 4.20 | 6.52 | 6.44 | 3.96 | 9.04 |
| age_band | <18 | 41 | 1.70 | 1.53 | 2.47 | 5.58 | 2.12 | 9.78 |
| age_band | >90 | 2 | 1.70 | 6.03 | 7.02 | 8.87 | 6.67 | 12.30 |
| platform | 27K | 103 | 12.13 | 11.48 | 12.33 | 3.89 | 10.90 | 9.92 |
| platform | 450K | 2560 | 3.87 | 3.61 | 4.04 | 5.27 | 4.05 | 7.42 |
| platform | EPIC | 167 | 3.05 | 2.85 | 3.93 | 7.62 | 4.14 | 7.66 |
| tissue | Blood | 2699 | 3.69 | 3.49 | 4.04 | 5.12 | 4.24 | 7.15 |
| tissue | Saliva | 131 | 16.31 | 12.39 | 8.67 | 10.03 | 2.61 | 32.26 |

## Predictive-interval calibration on benchmark healthy controls (nominal 90%)

|  | cover90_hc | mean_sigma_years_hc | cover90_hc[Platform=27K] | cover90_hc[Platform=450K] | cover90_hc[Platform=EPIC] | cover90_hc[Tissue=Blood] | cover90_hc[Tissue=Saliva] | cover90_hc[blood,450K/EPIC] |
|---|---|---|---|---|---|---|---|---|
| full | 0.756 | 5.083 | 0.311 | 0.768 | 0.844 | 0.784 | 0.191 | 0.802 |
| full_seed1 | 0.766 | 5.182 | 0.369 | 0.777 | 0.844 | 0.795 | 0.168 | 0.812 |
| no_context | 0.752 | 4.850 | 0.291 | 0.766 | 0.820 | 0.779 | 0.183 | 0.799 |
| no_id | 0.746 | 5.178 | 0.340 | 0.757 | 0.826 | 0.774 | 0.160 | 0.792 |
| no_module_mixing | 0.746 | 5.067 | 0.291 | 0.758 | 0.838 | 0.777 | 0.115 | 0.796 |
| point_estimate | 0.435 | 2.082 | 0.146 | 0.448 | 0.401 | 0.454 | 0.038 | 0.466 |
| no_aug | 0.794 | 5.090 | 0.291 | 0.814 | 0.802 | 0.815 | 0.351 | 0.836 |
| no_ssl | 0.779 | 5.206 | 0.311 | 0.794 | 0.838 | 0.810 | 0.153 | 0.829 |
| no_linear_route | 0.707 | 4.857 | 0.417 | 0.712 | 0.814 | 0.734 | 0.153 | 0.747 |

## Training runs

| run | best_val_medae | epochs | wall_min |
|---|---|---|---|
| celltype_bal | 2.91 | 15 | 27.40 |
| full | 2.72 | 15 | 24.50 |
| full_seed1 | 2.82 | 15 | 25.40 |
| no_aug | 2.69 | 15 | 15.20 |
| no_context | 2.74 | 15 | 19.40 |
| no_id | 2.70 | 15 | 17.60 |
| no_linear_route | 2.77 | 15 | 15.90 |
| no_module_mixing | 2.67 | 15 | 16.90 |
| no_ssl | 2.76 | 15 | 18.60 |
| point_estimate | 2.89 | 15 | 18.80 |
| transfer_A | 3.10 | 15 | 9.60 |

## Acceleration head (cross-fitted)

| model | MedAE | MedAE_18_90 | MedE | AA2_hits | AA2_tests | AA1_hits | AA1_tests | score |
|---|---|---|---|---|---|---|---|---|
| PhenoAgeV2 | 7.53 | 7.50 | -2.55 | 19 | 41 | 9 | 24 | 28.00 |
| PhenoAgeV1 | 8.00 | 7.99 | -4.19 | 9 | 41 | 7 | 24 | 16.00 |
| HorvathV1 | 5.34 | 5.33 | -0.04 | 3 | 41 | 12 | 24 | 15.00 |
| EpiMod + accel head (study-held-out) | 3.79 | 3.84 | 1.77 | 7 | 41 | 15 | 24 | 14.99 |
| HorvathV2 | 4.15 | 4.22 | 1.15 | 5 | 41 | 12 | 24 | 13.67 |
| EpiMod + accel head (class-held-out) | 4.06 | 4.10 | 2.14 | 6 | 41 | 14 | 24 | 12.63 |
| EpiMod (age only) | 3.93 | 3.96 | 2.03 | 5 | 41 | 15 | 24 | 12.25 |
