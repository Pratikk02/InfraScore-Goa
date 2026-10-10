# InfraScore Goa

**Measuring, explaining and closing town-level infrastructure gaps in Goa with machine learning**

Advanced Machine Learning for Business Transformation (AMLBT), Goa Institute of Management, PGDM Big Data Analytics

Team: Mallika Gupta (B2026086), Pratik Kelkar (B2026090), Kshitij Mahajan (B2026081). Faculty: Dr. Suman Sanyal.

## What the project does

Goa ranks among India's best-served states at household level, but its 69 towns differ sharply in access to hospitals, colleges, banks, fire services and civic facilities. InfraScore answers four planning questions:

1. **How well served is each town?** A 0-100 composite index built from 26 indicators in 6 pillars (health, education, utilities, connectivity, finance, social), tested across 3,600 methodological scenarios.
2. **What kind of gap does a town have?** K-Means typologies, selected against Ward and Gaussian-mixture clustering by bootstrap stability.
3. **Is a town under-served relative to comparable towns?** A peer-adjusted gap model (Lasso, selected from six regressors and a mean baseline by nested cross-validation) with conformal-style 90% intervals.
4. **Where should the next investments go?** A facility presence classifier proposes candidate sites; a greedy optimiser allocates interventions (equal-cost and cost-aware versions, tested under 200 cost scenarios) and is compared with rule-of-thumb strategies.
5. **Do the recommendations hold up?** The plan built from 2011 data is checked against post-2011 government action and news reports.

## Data (real, public)

| Dataset | Level | Source |
|---|---|---|
| Census of India 2011, District Census Handbook, Goa: Town Directory / Town Amenities | 69 towns | Machine-readable copy by IndiaWikiFiles (MIT licence): https://github.com/IndiaWikiFiles/Goa |
| Census of India 2011, district household amenity tables | 640 districts | https://github.com/nishusharma1608/India-Census-2011-Analysis |

Both are included in `data/raw/`; the notebook downloads them again if the folder is empty.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook InfraScore_Goa_Final.ipynb      # then Kernel > Restart & Run All
```

Runtime is about 4 minutes on a single CPU core (less on a multi-core laptop). All random processes use `SEED = 42`, so results are identical on every run. The notebook is saved with its outputs, so results and graphs are visible even before re-running. `infrascore_goa_final.py` is the same code as a plain script (`python infrascore_goa_final.py`).

## Repository structure

```
InfraScore-Goa/
├── InfraScore_Goa_Final.ipynb     # complete analysis, 12 sections, saved with outputs
├── infrascore_goa_final.py        # same code exported as a script
├── requirements.txt
├── README.md
├── data/raw/
│   ├── census2011_goa_towns/      # 69 Town Directory records + source licence
│   └── india_districts_census_2011.csv
└── outputs/
    ├── infrascore_rankings.csv    # scores, pillars, tiers, rank bands, cluster per town
    ├── gap_analysis.csv           # expected score, gap, 90% interval, flag per town
    ├── gap_model_comparison.csv   # 7 models, nested CV
    ├── gap_model_ablation.csv
    ├── gap_model_drivers.csv
    ├── clustering_comparison.csv
    ├── cluster_profiles.csv
    ├── uncertainty_analysis.csv   # 18 configurations x 200 weight draws
    ├── facility_model_metrics.csv
    ├── facility_candidate_sites.csv
    ├── policy_simulation.csv
    ├── greedy_plan_K10.csv
    ├── policy_simulation_costed.csv  # budget in cost units, 4 strategies
    ├── policy_cost_scenarios.csv     # 200 random cost scenarios
    ├── post2011_evidence.csv         # sourced later evidence for the plan's towns
    ├── validation_literacy.csv
    ├── district_benchmark.csv
    ├── features.csv
    └── figures/                   # 10 figures, each as 300-dpi PNG and vector PDF
```

## Headline results

| Component | Result |
|---|---|
| InfraScore | Mean 40.4 (SD 10.8). Top: Panaji 74.2, Margao 67.0, Mapusa 60.7. Statutory towns 51.1 vs census towns 37.7 |
| Robustness | 3,600 scenarios; median Spearman rho with baseline 0.87 (worst 0.59). Panaji and Margao stay top-10 in 95% of scenarios |
| Typologies | K-Means k = 4, bootstrap ARI 0.59 (Ward 0.37, GMM 0.41) |
| Gap model | Lasso nested-CV MAE 6.80 vs 8.53 for the mean baseline (20% lower), R² 0.24 |
| Under-served | Cuncolim (-19.3), Quepem (-16.9), Sancoale (-15.2) fall below their 90% expected range |
| Facility model | Logistic regression mean ROC-AUC 0.69 (private bank 0.85); 61 candidate sites |
| Policy (K = 10) | Greedy plan: population-weighted gain 0.76 points, 4.3x random and 1.9x "lowest score first" |
| Cost-aware plan | Budget of 20 cost units: 0.64 points vs 0.50 (cost-blind greedy), 0.21 (lowest first), 0.16 (random); best in 99-100% of 200 cost scenarios |
| Post-2011 check | Exact evidence for 4 of the plan's 6 towns (e.g. Cuncolim fire station), related evidence for the other 2; 0 of 3 eligible random control towns (Fisher p = 0.12) |
| Validation | InfraScore vs literacy: rho 0.22 (p = 0.10); controlling for town size rho 0.31 (p = 0.02) |

## Data quality log

| Issue | Handling |
|---|---|
| Panaji civic status blank in source | Set to Municipal Corporation (status code "M Corp. + OG") |
| Panaji literate population 767 for 70,991 residents | Entry error; treated as missing |
| Literacy missing for all 13 Municipal Councils | Left missing; validation uses the remaining 55 towns |
| Medical college distance missing for 2 towns | Imputed with state median |
| 69 towns sum to 902,212 vs official urban population 906,814 | The 70th town, Orgao (census town, Ponda taluka), is absent from the source copy |
| District file's "Households" column overstates totals | Recomputed from household-size buckets; India total then matches the official 24.67 crore |

## Limitations

Census 2011 is the latest complete town-level amenity census, so later investments (e.g. Mopa airport, new colleges) are not reflected. Facility counts measure presence, not capacity or quality. Census distances are reported estimates, not computed travel times. Facility costs in the simulation are illustrative (the strategy ranking is robust to them), and a new facility updates only the receiving town because the source has no coordinates. The post-2011 check relies on news and government records and uses a small random control sample.
