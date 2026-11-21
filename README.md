# InfraScore Goa

[![Reproduce results](https://github.com/Pratikk02/InfraScore-Goa/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Pratikk02/InfraScore-Goa/actions/workflows/reproduce.yml)

**Measuring, explaining and closing town-level infrastructure gaps in Goa with machine learning**

AMLBT project, Goa Institute of Management, PGDM - Big Data Analytics. Team 23B: Mallika Gupta (B2026086), Pratik Kelkar (B2026090),
Kshitij Satish Mahajan (B2026081). Faculty: Dr. Suman Sanyal.

## Reproduce the paper (23B-V2)

```bash
git clone https://github.com/Pratikk02/InfraScore-Goa
cd InfraScore-Goa
pip install -r requirements-lock.txt          # exact versions used for the paper
jupyter nbconvert --to notebook --execute InfraScore_Goa_Final.ipynb
python check_results.py                        # compares 20 headline numbers with the paper
```

The full run takes about 3-4 minutes on one CPU core; a fixed seed (42) makes every run identical. The notebook is also saved
with its outputs, and `infrascore_goa_final.py` is the same code as a script. The GitHub Actions workflow
(`.github/workflows/reproduce.yml`) repeats these steps on a clean machine at every commit and uploads the regenerated outputs.

## What the project does

| Section | Question | Method |
|---|---|---|
| 1-4 | How well served is each town? | 26 indicators, 6 pillars, distance-decay access, 0-100 index |
| 5 | How robust is the ranking? | 3,600 scenarios (decay, normalisation, aggregation, weights) |
| 6 | What kind of gap does a town have? | K-Means vs Ward vs GMM, bootstrap stability |
| 7 | Is a town under-served relative to peers? | 7 regressors, nested CV, Lasso selected; conformal-style intervals |
| 8 | Where could a facility go? | Facility presence classifier, 61 candidate sites |
| 9-10 | Allocation and corroboration | Greedy vs heuristics; post-2011 records vs random controls |
| 11a | Coverage | 69 of 70 towns; Orgao record absent; leave-one-town-out test |
| 11b | Map and contemporary proxies | GeoNames coordinates; census-estimated circuity; current hospitals and fire stations |
| 11c | Do the intervals cover? | Held-out coverage (random and leave-one-taluka-out), Mondrian bands, Moran's I |
| 11d | Spillovers and real costs | Closed-form spillover allocation; sourced capital + operating costs; 200 cost scenarios; 400 alternative weightings |

## Data (public)

| Folder / file | Content | Source |
|---|---|---|
| `data/raw/census2011_goa_towns/` | 69 Town Directory records | Census of India 2011 DCHB, machine-readable copy by IndiaWikiFiles (MIT) |
| `data/raw/india_districts_census_2011.csv` | 640-district household tables | github.com/nishusharma1608/India-Census-2011-Analysis |
| `data/raw/spatial/` | Town coordinates, Goa outline, contemporary hospitals and fire stations | GeoNames (CC BY 4.0), datta07/INDIAN-SHAPEFILES, Goa DHS and DFES records; see `SOURCES.md` |
| `data/raw/evidence/` | Search log for the post-2011 corroboration (plan and control towns) | Queries, coding rule; see `README.md` there |
| `data/raw/costs/` | Capital and operating costs by facility | Goa project records, Prinja et al. (2016), Goa Budget 2025-26; see `SOURCES.md` |

## Analytical coverage

Census 2011 lists 70 towns in Goa (14 statutory, 56 census towns). The source copy contains 69: all 14 statutory towns and 55 census
towns, i.e. 902,212 of 906,814 urban residents (99.49%). The absent record is the census town of **Orgao** (Ponda taluka, 4,602
residents). Orgao is not scored; it is shown on the map without a value. Removing any single town shifts no other town's rank by more
than five places.

## Headline results (V2)

| Component | Result |
|---|---|
| Index | Mean 40.4 (SD 10.8); Panaji 74.2, Margao 67.0, Mapusa 60.7 |
| Robustness | 3,600 scenarios, median Spearman 0.87 |
| Gap model | Lasso nested-CV MAE 6.80 vs 8.53 baseline (20% lower), R^2 0.24 |
| Interval coverage (90%) | First-review rule 89.6% overall but 77.1% for statutory towns; conformal 94.1%; Mondrian 91.4% statutory |
| Flags | Conformal band: Cuncolim significantly under-served; Quepem and Sancoale review candidates |
| Contemporary proxies | Partially updated index vs 2011: Spearman 0.987; median km to public inpatient care 4.8 |
| Experiment A (10 equal interventions) | Greedy 0.76 vs random 0.19 (4.1x); with spillovers 1.10 vs 0.34 (3.3x) |
| Experiment B (Rs 250 crore lifecycle) | Cost-aware 0.86 vs cost-blind 0.72; best in 96-100% of 200 cost scenarios |
| Weakest spillovers off | Hall and library spillovers switched off: cost-aware 0.83 vs cost-blind 0.69, still best in 96% of cost scenarios |
| Weight robustness | Plans rebuilt under 400 alternative pillar weightings: Sancoale and Cuncolim chosen in 99-100%; Sancoale fire station in 96-100% |
| Post-2011 records | Exact evidence for 4 of 6 plan towns vs 2 of 12 random controls (Fisher p = 0.057); exact or related 6 of 6 vs 3 of 12 (p = 0.005) |

## Data quality log

| Issue | Handling |
|---|---|
| Panaji civic status blank | Set to Municipal Corporation |
| Panaji literate population 767 of 70,991 | Treated as missing |
| Literacy missing for Municipal Councils | Validation uses the remaining 55 towns |
| Medical-college distance missing for 2 towns | State median |
| Orgao record absent | Documented; unscored; leave-one-town-out test |
| District file household totals overstated | Recomputed from household-size buckets |
| District polygons place Ponda taluka in South Goa | Polygons used only as an unlabelled outline; census districts used in analysis |

## Limitations

The index uses 2011 data; the contemporary update covers hospital and fire access only. Distances are census estimates or
circuity-adjusted straight lines, not road-network travel times. Operating costs for colleges, stadiums and libraries are
budget averages, and the community-hall cost is an assumption. Statutory-town intervals rest on 14 observations. Post-2011 records are cited, not independently verified.
