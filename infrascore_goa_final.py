#!/usr/bin/env python
# coding: utf-8

# # InfraScore Goa: Measuring, Explaining and Closing Town-Level Infrastructure Gaps with Machine Learning
# 
# **Advanced Machine Learning for Business Transformation (AMLBT) | Goa Institute of Management, PGDM Big Data Analytics**
# 
# **Team 23B:** Mallika Gupta (B2026086), Pratik Kelkar (B2026090), Kshitij Satish Mahajan (B2026081)  
# **Faculty:** Dr. Suman Sanyal
# 
# ---
# **Business problem.** Goa's districts rank in the top decile of India for household amenities, but its 69 towns differ sharply in access to hospitals, colleges, banks, fire services and civic facilities. State planners, the 14 urban local bodies, and private providers (banks, hospital chains) need to know *which towns are under-served, in what way, compared with which peers, and where the next unit of investment does the most good.*
# 
# **What this notebook does**
# 
# | Section | Method | Output |
# |---|---|---|
# | 1-2 | Parse Census 2011 Town Directory; data-quality checks | Clean table, 69 towns |
# | 3 | 26 indicators, 6 pillars, distance-decay access | Indicator matrix |
# | 4 | Composite InfraScore (0-100) | Town ranking and tiers |
# | 5 | Global uncertainty analysis (3,600 scenarios) | Rank confidence bands |
# | 6 | Town typologies: K-Means vs Ward vs GMM, bootstrap stability | 4 town types |
# | 7 | Peer-adjusted gap model: 7 regressors, nested CV, ablation, conformal intervals | Under-served towns with confidence |
# | 8 | Facility presence model: classification of 6 facility types | Candidate sites |
# | 9 | Policy simulation: equal-cost and cost-aware allocation, 200 cost scenarios | Investment plan |
# | 10 | External validation, post-2011 corroboration, state context | Validity checks |
# | 11 | Second review: coverage audit, town map, contemporary access, interval-coverage test, spatial spillovers, sourced costs | V2 robustness checks |
# 
# **Run:** `Kernel > Restart & Run All`. Runtime is about 3-4 minutes on a single CPU core. All random processes use `SEED = 42`.

# ## 0. Setup

# In[1]:


import io, re, sys, json, time, zipfile, platform, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import requests
import sklearn
from scipy.stats import spearmanr
from IPython.display import display, Markdown

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, mean_absolute_error, roc_auc_score,
                             average_precision_score, f1_score, brier_score_loss)
from sklearn.model_selection import (KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_validate, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:          # the notebook still runs without xgboost; that model is skipped
    HAS_XGB = False

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30, "display.width", 220, "display.precision", 3)

SEED = 42
ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
TOWN_DIR = RAW / "census2011_goa_towns"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
for p in (TOWN_DIR, FIG):
    p.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": "#444444", "ytick.color": "#444444",
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
COL1, COL2 = 3.5, 7.2                     # IEEE single- and double-column widths (inches)
NORTH, SOUTH, GREY = "#1f6f8b", "#e07a5f", "#b8bec6"
CLUSTER_COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#c8553d"]
STRAT_COLORS = {"Greedy optimiser": "#2a9d8f", "Lowest score first": "#e9a23b",
                "Random (mean)": "#9aa1a9", "Largest towns first": "#1f6f8b"}
PRETTY = {"is_statutory_town": "Statutory town", "rainfall_mm": "Annual rainfall", "log_density": "Density (log)",
          "log_population": "Population (log)", "sex_ratio": "Sex ratio", "avg_household_size": "Household size",
          "st_share": "ST share", "sc_share": "SC share", "is_south_goa": "South Goa district", "log_area": "Area (log)"}
T0 = time.time()

print(f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | "
      f"matplotlib {matplotlib.__version__} | xgboost {'available' if HAS_XGB else 'not installed (skipped)'}")


def savefig(fig, name):
    """Save each figure as 300-dpi PNG (notebook/README) and vector PDF (paper)."""
    stem = Path(name).stem
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=300)
    # PDF version for the paper: single-panel figures drop the in-chart title (the caption carries it)
    titles = [(ax, ax.get_title(loc="left")) for ax in fig.axes if ax.get_title(loc="left")]
    if len(fig.axes) == 1 or (len(fig.axes) == 2 and fig.axes[1].get_label() == "<colorbar>"):
        for ax, _ in titles:
            ax.set_title("", loc="left")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    for ax, tt in titles:
        ax.set_title(tt, loc="left")
    plt.show()


def label_points(ax, xs, ys, labels, fontsize=6.5, sides=None, step=1.2, max_iter=400):
    """Annotate points and push overlapping labels apart vertically (simple repel, no extra package).
    sides: optional list of 'r' / 'l' to place each label right or left of its point."""
    fig = ax.figure
    sides = sides or ["r"] * len(labels)
    starts = [(5, 3) if sd == "r" else (-5, 3) for sd in sides]
    texts = [ax.annotate(l, (x, y), xytext=st, textcoords="offset points", fontsize=fontsize, color="#222222",
                         ha="left" if sd == "r" else "right", va="center", zorder=6,
                         arrowprops=dict(arrowstyle="-", color="#8a8a8a", lw=0.5, shrinkA=0, shrinkB=3))
             for x, y, l, st, sd in zip(xs, ys, labels, starts, sides)]
    offs = [list(st) for st in starts]
    for t in texts:
        t.arrow_patch.set_visible(False)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for _ in range(max_iter):
        boxes = [matplotlib.text.Text.get_window_extent(t, rend).expanded(1.04, 1.3) for t in texts]  # text only, not arrow
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    up = boxes[i].y0 >= boxes[j].y0
                    offs[i][1] += step if up else -step
                    offs[j][1] -= step if up else -step
                    moved = True
        for t, o in zip(texts, offs):
            o[1] = float(np.clip(o[1], -45, 45))
            t.set_position(o)
        if not moved:
            break
        fig.canvas.draw()
    for t, o, st in zip(texts, offs, starts):   # connector only where a label moved away from its point
        t.arrow_patch.set_visible(abs(o[1] - st[1]) > 3)
    return texts


# ## 1. Data acquisition
# 
# **Primary source:** Census of India 2011, District Census Handbook (DCHB), Goa: *Town Directory / Town Amenities*. We use the machine-readable copy (one file per town) published by the IndiaWikiFiles project under the MIT licence. Files are included in `data/raw/`; if the folder is empty they are downloaded from GitHub.
# 
# **Secondary source:** Census 2011 district household tables (640 districts), used for state context in Section 10.

# In[2]:


TOWN_ZIP = "https://codeload.github.com/IndiaWikiFiles/Goa/zip/refs/heads/master"
DIST_URL = ("https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/"
            "master/india-districts-census-2011.csv")
DIST_CSV = RAW / "india_districts_census_2011.csv"

if not any(TOWN_DIR.glob("*.wiki")):
    print("Downloading Census 2011 town files ...")
    with zipfile.ZipFile(io.BytesIO(requests.get(TOWN_ZIP, timeout=120).content)) as zf:
        for n in zf.namelist():
            if "/Goa-Towns-English/" in n and n.endswith(".wiki"):
                (TOWN_DIR / Path(n).name).write_bytes(zf.read(n))
if not DIST_CSV.exists():
    DIST_CSV.write_bytes(requests.get(DIST_URL, timeout=120).content)

files = sorted(TOWN_DIR.glob("*.wiki"))
print(f"{len(files)} town files found")
print("\nSample record (first 25 lines of Mapusa):\n")
print("\n".join((TOWN_DIR / [f.name for f in files if f.name.startswith("Mapusa")][0]).read_text().splitlines()[:25]))


# ## 2. Parsing and data-quality checks
# 
# Each record mixes an infobox with narrative sentences. The parser extracts:
# * identity and demography (population, households, SC/ST, literacy, area);
# * utilities (water capacity, drainage, fire service);
# * every sentence of the form *"There is/are N facility in the town"* (count) and *"The nearest facility is at ... (N km)"* (distance);
# * connectivity (km to district HQ, state HQ, railway station).

# In[3]:


def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    try:
        return cast(m.group(1).replace(",", "")) if m else None
    except ValueError:
        return None


def _key(name, plural):
    name = name.replace("[[", "").replace("]]", "").strip()
    if plural and name.endswith("s"):          # source pluralises by adding one "s" ("facilitys")
        name = name[:-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def parse_town(path):
    t = path.read_text(encoding="utf-8")
    r = {"census_code": _num(r"Location Code=(\d+)", t, int)}
    full = re.search(r"\|native_name=(.+)", t).group(1).strip()
    r["town"] = re.sub(r"\s*\(.*\)$", "", full).strip()
    st = re.search(r"\((.+)\)$", full)
    r["status_code"] = st.group(1) if st else None
    r["taluka"] = re.sub(r"\s*\d+$", "", re.search(r"\|subdivision_name3=(.+)", t).group(1).strip()).title()
    r["district"] = re.search(r"\|subdivision_name2=(.+)", t).group(1).strip().title()
    r["area_sqkm"] = _num(r"\|Area=([\d.]+)", t)
    r["population"] = _num(r"\|population_total=(\d+)", t, int)
    r["density"] = _num(r"\|population_density=(\d+)", t, int)
    r["sex_ratio"] = _num(r"\|female_per_1000_male=(\d+)", t, int)
    r["households"] = _num(r"harbouring (\d+) households", t, int)
    r["sc_pop"] = _num(r"Scheduled Caste population is (\d+)", t, int)
    r["st_pop"] = _num(r"Scheduled Tribes population is (\d+)", t, int)
    r["civic_status"] = (re.search(r"civic status of the town is '(.+?)'", t) or [None, None])[1]
    r["dist_district_hq_km"] = _num(r"district headquarter .*? at a distance of ([\d.]+) km", t)
    r["dist_state_hq_km"] = _num(r"state headquarter .*? at a distance of ([\d.]+) km", t)
    r["dist_railway_km"] = _num(r"Railway station is at .+? at a distance of ([\d.]+) km", t)
    r["rainfall_mm"] = _num(r"Rainfall \(mm\.\): ([\d.]+)", t)
    r["literate_pop"] = _num(r"Total literate population: (\d+)", t, int)
    r["water_capacity_kl"] = _num(r"capacity of the water supply system is ([\d.]+) kilo-litres", t)
    r["drainage"] = ("open_and_closed" if "open and closed drainage" in t else
                     "closed" if "closed drainage" in t else "open" if "open drainage" in t else "none")
    r["fire_in_town"] = int("There is fire fighting service in the town" in t)
    r["dist_fire_km"] = 0.0 if r["fire_in_town"] else _num(r"nearest fire fighting service is at .*?\(([\d.]+) km\)", t)
    body = t.split("== Health Services ==", 1)[-1]
    for m in re.finditer(r"There (is|are) (\d+) (.+?) in the town", body):
        k = "n_" + _key(m.group(3), m.group(1) == "are")
        r[k] = r.get(k, 0) + int(m.group(2))
    for m in re.finditer(r"The nearest (.+?) is at (?:[^\n]*?\()?([\d.]+) km", body):
        r["d_" + _key(m.group(1), False)] = float(m.group(2))
    return r


raw = pd.DataFrame([parse_town(f) for f in files])
ncols = [c for c in raw if c.startswith("n_")]
raw[ncols] = raw[ncols].fillna(0).astype(int)
for d in [c for c in raw if c.startswith("d_")]:            # facility inside town -> distance 0
    if "n_" + d[2:] in raw:
        raw.loc[raw[d].isna() & (raw["n_" + d[2:]] > 0), d] = 0.0

# ---- data-quality corrections (each one is logged)
dq_log = []
mask = raw["status_code"].str.startswith("M Corp", na=False)
raw.loc[mask, "civic_status"] = "Municipal Corporation"
dq_log.append(("Panaji civic status blank in source", "Set to Municipal Corporation (status code 'M Corp. + OG')"))
raw["literacy_rate"] = raw["literate_pop"] / raw["population"]
bad = raw["literacy_rate"] < 0.30
dq_log.append((f"Implausible literacy for {', '.join(raw.loc[bad, 'town'])} "
               f"({int(raw.loc[bad, 'literate_pop'].iloc[0])} literates)", "Treated as missing"))
raw.loc[bad, "literacy_rate"] = np.nan
dq_log.append((f"Literacy missing for {raw['literacy_rate'].isna().sum()} towns (all Municipal Councils + Panaji)",
               "Left missing; validation uses remaining towns"))

total_pop = raw["population"].sum()
dq_log.append((f"69 towns sum to {total_pop:,} vs official urban population 906,814 "
               f"(gap {906814 - total_pop:,})",
               "One census town (Orgao, Ponda taluka) is absent from the source copy; documented as a limitation"))

print(f"Parsed {len(raw)} towns x {raw.shape[1]} fields")
print(raw["civic_status"].value_counts().to_string(), "\n")
display(pd.DataFrame(dq_log, columns=["Issue", "Handling"]))
raw[["town", "civic_status", "taluka", "district", "population", "households", "water_capacity_kl",
     "drainage", "dist_fire_km", "dist_railway_km"]].sort_values("population", ascending=False).head(8)


# **Automated sanity checks.** These assert properties the data must satisfy; the notebook stops if any fails.

# In[4]:


mapusa = raw.set_index("town").loc["Mapusa"]
checks = {
    "69 unique census codes": raw["census_code"].is_unique and len(raw) == 69,
    "Mapusa population = 39,989 (source file)": mapusa["population"] == 39989,
    "Mapusa nationalised banks = 25 (source file)": mapusa["n_nationalised_bank"] == 25,
    "Every town has population and households": raw[["population", "households"]].notna().all().all(),
    "14 statutory + 55 census towns": (raw["civic_status"] != "Census Town").sum() == 14,
    "No negative distances": bool((raw[[c for c in raw if c.startswith("d_")]].fillna(0) >= 0).all().all()),
}
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
assert all(checks.values()), "Data checks failed"


# ## 3. Feature engineering: 26 indicators in 6 pillars
# 
# * **Density** indicators: facilities per 10,000 residents (supply relative to demand).
# * **Access** indicators: km to the nearest facility, 0 if inside the town (Hansen-type accessibility).
# * **Context** variables (used only by the ML models, never inside the score): population, density, household size, SC/ST share, sex ratio, statutory status, district, rainfall, area.

# In[5]:


def s_(df, cols):
    cols = [c for c in cols if c in df]
    return df[cols].sum(axis=1) if cols else pd.Series(0, index=df.index)


def nearest(df, stems):
    d = [f"d_{s}" for s in stems if f"d_{s}" in df]
    n = [f"n_{s}" for s in stems if f"n_{s}" in df]
    out = df[d].min(axis=1) if d else pd.Series(np.nan, index=df.index)
    return out.where(df[n].sum(axis=1) == 0, 0.0) if n else out


P10K = 10_000 / raw["population"]
F = raw[["census_code", "town", "civic_status", "taluka", "district", "population"]].copy()
# Health
F["hospitals_per_10k"] = s_(raw, ["n_allopathic_hospital_facility", "n_non_government_in_and_out_patient_facility",
                                  "n_non_government_charitable_hospital_or_nursing_home"]) * P10K
F["primary_care_per_10k"] = s_(raw, ["n_dispensary_health_centre", "n_family_welfare_centre",
                                     "n_maternity_and_child_welfare_centre", "n_maternity_home", "n_t_b_clinic",
                                     "n_non_government_out_patient_facility"]) * P10K
F["pharmacies_per_10k"] = s_(raw, ["n_non_government_medicine_shop"]) * P10K
F["acc_hospital_km"] = nearest(raw, ["allopathic_hospital_facility"])
F["acc_maternity_km"] = nearest(raw, ["maternity_home", "maternity_and_child_welfare_centre"])
F["acc_medical_college_km"] = nearest(raw, ["government_medical_college"])
# Education
F["primary_schools_per_10k"] = s_(raw, ["n_government_primary_school", "n_private_primary_school"]) * P10K
F["middle_schools_per_10k"] = s_(raw, ["n_government_middle_school", "n_private_middle_school"]) * P10K
F["secondary_schools_per_10k"] = s_(raw, ["n_government_secondary_school", "n_private_secondary_school",
                                          "n_government_senior_secondary_school",
                                          "n_private_senior_secondary_school"]) * P10K
F["acc_degree_college_km"] = nearest(raw, sorted({c[2:] for c in raw if "degree_college" in c}))
F["acc_engineering_km"] = nearest(raw, ["government_engineering_college", "private_engineering_college"])
F["acc_polytechnic_km"] = nearest(raw, ["government_polytechnic", "private_polytechnic"])
F["acc_vocational_km"] = nearest(raw, ["government_vocational_training_school", "private_vocational_training_school"])
# Utilities
F["water_lpcd"] = raw["water_capacity_kl"] * 1000 / raw["population"]
F["drainage_score"] = raw["drainage"].map({"closed": 1.0, "open_and_closed": 0.5, "open": 0.0}).fillna(0)
F["acc_fire_km"] = raw["dist_fire_km"]
# Connectivity
F["acc_district_hq_km"] = raw["dist_district_hq_km"]
F["acc_railway_km"] = raw["dist_railway_km"]
F["acc_state_hq_km"] = raw["dist_state_hq_km"]
# Finance
F["banks_per_10k"] = s_(raw, ["n_nationalised_bank", "n_private_commercial_bank", "n_co_operative_bank"]) * P10K
F["credit_soc_per_10k"] = s_(raw, ["n_agricultural_credit_society", "n_non_agricultural_credit_society"]) * P10K
# Social and civic
F["acc_library_km"] = nearest(raw, ["government_public_library", "private_public_library"])
F["acc_auditorium_km"] = nearest(raw, ["government_auditorium_community_hall", "private_auditorium_community_hall"])
F["acc_stadium_km"] = nearest(raw, ["government_stadium", "private_stadium"])
F["acc_cinema_km"] = nearest(raw, ["private_cinema_theatre"])
welfare = [["government_old_age_home", "private_old_age_home"], ["government_orphanage_home", "private_orphanage_home"],
           ["government_working_women_s_hostel", "private_working_women_s_hostel"],
           ["government_special_school_for_disabled", "private_special_school_for_disabled"]]
F["acc_welfare_km"] = pd.concat([nearest(raw, w) for w in welfare], axis=1).mean(axis=1)

INDICATORS = {  # name: (pillar, kind)
    "hospitals_per_10k": ("health", "density"), "primary_care_per_10k": ("health", "density"),
    "pharmacies_per_10k": ("health", "density"), "acc_hospital_km": ("health", "access"),
    "acc_maternity_km": ("health", "access"), "acc_medical_college_km": ("health", "access"),
    "primary_schools_per_10k": ("education", "density"), "middle_schools_per_10k": ("education", "density"),
    "secondary_schools_per_10k": ("education", "density"), "acc_degree_college_km": ("education", "access"),
    "acc_engineering_km": ("education", "access"), "acc_polytechnic_km": ("education", "access"),
    "acc_vocational_km": ("education", "access"),
    "water_lpcd": ("utilities", "density"), "drainage_score": ("utilities", "ratio"), "acc_fire_km": ("utilities", "access"),
    "acc_district_hq_km": ("connectivity", "access"), "acc_railway_km": ("connectivity", "access"),
    "acc_state_hq_km": ("connectivity", "access"),
    "banks_per_10k": ("finance", "density"), "credit_soc_per_10k": ("finance", "density"),
    "acc_library_km": ("social", "access"), "acc_auditorium_km": ("social", "access"),
    "acc_stadium_km": ("social", "access"), "acc_cinema_km": ("social", "access"), "acc_welfare_km": ("social", "access"),
}
PILLARS = ["health", "education", "utilities", "connectivity", "finance", "social"]
BASE_W = {"health": 0.25, "education": 0.20, "utilities": 0.20, "connectivity": 0.15, "finance": 0.10, "social": 0.10}

for c in [c for c in INDICATORS if F[c].isna().any()]:
    print(f"Imputed {F[c].isna().sum()} missing value(s) in {c} with the state median")
    F[c] = F[c].fillna(F[c].median())

# ---- context variables (ML inputs only)
F["log_population"] = np.log(raw["population"])
F["log_density"] = np.log(raw["density"].clip(lower=1))
F["avg_household_size"] = raw["population"] / raw["households"]
F["sc_share"] = raw["sc_pop"] / raw["population"]
F["st_share"] = raw["st_pop"] / raw["population"]
F["sex_ratio"] = raw["sex_ratio"]
F["is_statutory_town"] = (raw["civic_status"] != "Census Town").astype(int)
F["is_south_goa"] = (raw["district"] == "South Goa").astype(int)
F["rainfall_mm"] = raw["rainfall_mm"]
F["log_area"] = np.log(raw["area_sqkm"])
F["literacy_rate"] = raw["literacy_rate"]

print(f"\n{len(INDICATORS)} indicators built for {len(F)} towns")
F[list(INDICATORS)].describe().T[["mean", "std", "min", "50%", "max"]].round(2)


# ## 4. InfraScore: composite index
# 
# * Density indicators: winsorised at the 5th/95th percentile, then min-max scaled to 0-100.
# * Access indicators: negative-exponential distance decay $s = 100\,e^{-d/\lambda}$ with $\lambda = 10$ km (0 km = 100, 10 km = 37, 20 km = 14).
# * Pillar score = mean of its indicators. InfraScore = $\sum_j w_j P_j$ with planning weights (health 0.25, education 0.20, utilities 0.20, connectivity 0.15, finance 0.10, social 0.10).
# 
# The scaler bounds are stored so that the policy simulation (Section 9) can re-score a town after an intervention on the same scale.

# In[6]:


def fit_bounds(feats, norm="minmax"):
    b = {}
    for c, (_, kind) in INDICATORS.items():
        if kind == "density":
            s = feats[c]
            b[c] = (s.quantile(0.05), s.quantile(0.95), s.mean(), s.std(), np.sort(s.values))
    return b


def score_indicators(feats, bounds, lam=10.0, norm="minmax"):
    out = pd.DataFrame(index=feats.index)
    for c, (_, kind) in INDICATORS.items():
        s = feats[c]
        if kind == "access":
            out[c] = 100 * np.exp(-s / lam)
        elif kind == "ratio":
            out[c] = 100 * s
        else:
            lo, hi, mu, sd, ref = bounds[c]
            if norm == "minmax":
                out[c] = 100 * (s.clip(lo, hi) - lo) / (hi - lo)
            elif norm == "zscore":
                out[c] = 100 * ((s - mu) / sd).clip(-2.5, 2.5).add(2.5) / 5
            elif norm == "rank":
                out[c] = 100 * np.searchsorted(ref, s.values, side="right") / len(ref)
    return out


def pillar_table(ind):
    return pd.DataFrame({p: ind[[c for c, (pp, _) in INDICATORS.items() if pp == p]].mean(axis=1) for p in PILLARS})


def aggregate(pil, w, how="arithmetic"):
    w = pd.Series(w)[PILLARS]
    w = w / w.sum()
    if how == "geometric":
        return np.exp((np.log(pil[PILLARS].clip(lower=1)) * w).sum(axis=1))
    return (pil[PILLARS] * w).sum(axis=1)


BOUNDS = fit_bounds(F)
IND = score_indicators(F, BOUNDS)
PIL = pillar_table(IND)
R = F[["census_code", "town", "civic_status", "taluka", "district", "population"]].copy()
R[[f"{p}" for p in PILLARS]] = PIL.values
R["infrascore"] = aggregate(PIL, BASE_W)
R["rank"] = R["infrascore"].rank(ascending=False, method="min").astype(int)
q = R["infrascore"].quantile([0.2, 0.4, 0.6, 0.8]).values
R["tier"] = pd.cut(R["infrascore"], [-np.inf, *q, np.inf], labels=["E Critical", "D Lagging", "C Moderate", "B Good", "A Leading"])
R = R.sort_values("rank").reset_index(drop=True)

print(f"InfraScore: mean {R.infrascore.mean():.1f}, SD {R.infrascore.std():.1f}, "
      f"range {R.infrascore.min():.1f}-{R.infrascore.max():.1f}")
print(R.groupby("civic_status")["infrascore"].agg(["count", "mean"]).round(1).to_string())
print(R.groupby("district")["infrascore"].agg(["count", "mean"]).round(1).to_string())
show = ["rank", "town", "district", "civic_status", "infrascore", "tier"] + PILLARS
display(Markdown("**Top 10**")); display(R[show].head(10).round(1))
display(Markdown("**Bottom 10**")); display(R[show].tail(10).round(1))


# In[7]:


r_ = R.sort_values("infrascore")
fig, ax = plt.subplots(figsize=(COL1 + 0.6, 9.2))
ax.barh(r_["town"], r_["infrascore"], height=0.72, color=np.where(r_["district"] == "North Goa", NORTH, SOUTH))
for y, (s, c) in enumerate(zip(r_["infrascore"], r_["civic_status"])):
    ax.text(s + 0.6, y, f"{s:.0f}", va="center", fontsize=5.8, color="#333333",
            fontweight="bold" if c != "Census Town" else "normal")
ax.set_ylim(-0.7, len(r_) - 0.3); ax.set_xlim(0, 82)
ax.tick_params(axis="y", labelsize=6, length=0); ax.grid(axis="y", visible=False)
for lbl, c in zip(ax.get_yticklabels(), r_["civic_status"]):
    if c != "Census Town":
        lbl.set_fontweight("bold")
ax.set_xlabel("InfraScore (0-100)")
ax.set_title("InfraScore of Goa's 69 towns (bold = statutory town)")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=NORTH), plt.Rectangle((0, 0), 1, 1, color=SOUTH)],
          ["North Goa", "South Goa"], loc="lower right")
savefig(fig, "fig01_infrascore_ranking.png")

sel = pd.concat([R.head(12), R.tail(12)]).set_index("town")[PILLARS + ["infrascore"]]
fig, ax = plt.subplots(figsize=(COL1 + 0.4, 5.0))
im = ax.imshow(sel.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(sel.shape[1]), ["Health", "Educ.", "Util.", "Conn.", "Finance", "Social", "InfraScore"],
              rotation=35, ha="right")
ax.set_yticks(range(len(sel)), sel.index, fontsize=6.5)
ax.grid(False); ax.tick_params(length=0)
for i in range(sel.shape[0]):
    for j in range(sel.shape[1]):
        v = sel.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.8, color="white" if v < 20 or v > 85 else "#222222")
ax.axhline(11.5, color="black", lw=1.2)
ax.axvline(5.5, color="white", lw=2)
ax.set_title("Pillar scores: top 12 (above line) vs bottom 12 towns")
cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
savefig(fig, "fig02_pillar_heatmap.png")


# ## 5. Global uncertainty analysis
# 
# Every composite index embeds choices. Following the OECD-JRC handbook and Saisana et al. (2005), we vary all of them jointly:
# * distance-decay $\lambda \in \{5, 10, 15\}$ km;
# * normalisation of density indicators: winsorised min-max, z-score, percentile rank;
# * aggregation: arithmetic vs geometric mean of pillars;
# * pillar weights: 200 random draws from a flat Dirichlet distribution per configuration.
# 
# That gives 3 x 3 x 2 x 200 = **3,600 scenarios**. For each town we report the median rank and the 5th-95th percentile band.

# In[8]:


rng = np.random.default_rng(SEED)
W_DRAWS = rng.dirichlet(np.ones(6), size=200)
base_order = R.set_index("census_code").loc[F["census_code"], "infrascore"].values
rank_mat, rho_rows = [], []
for lam in (5, 10, 15):
    for norm in ("minmax", "zscore", "rank"):
        pil = pillar_table(score_indicators(F, BOUNDS, lam=lam, norm=norm))
        for how in ("arithmetic", "geometric"):
            if how == "arithmetic":
                S = pil[PILLARS].values @ W_DRAWS.T
            else:
                S = np.exp(np.log(pil[PILLARS].clip(lower=1).values) @ W_DRAWS.T)
            ranks = pd.DataFrame(S).rank(ascending=False).values
            rank_mat.append(ranks)
            rho = [spearmanr(S[:, k], base_order).statistic for k in range(S.shape[1])]
            rho_rows.append({"lambda_km": lam, "normalisation": norm, "aggregation": how,
                             "median_rho": np.median(rho), "min_rho": np.min(rho)})
RM = np.hstack(rank_mat)                                          # towns x 3600
UA = pd.DataFrame({"census_code": F["census_code"], "rank_median": np.median(RM, 1),
                   "rank_p05": np.percentile(RM, 5, 1), "rank_p95": np.percentile(RM, 95, 1)})
R = R.drop(columns=[c for c in R if c.startswith("rank_")], errors="ignore").merge(UA, on="census_code")
UA_TAB = pd.DataFrame(rho_rows)
print(f"{RM.shape[1]:,} scenarios. Spearman rho with baseline ranking: "
      f"median {UA_TAB.median_rho.median():.3f}, worst single scenario {UA_TAB.min_rho.min():.3f}")
print(f"Median width of the 90% rank band: {(R.rank_p95 - R.rank_p05).median():.0f} places")
top_stable = R[(R.rank_p95 <= 10)]["town"].tolist()
bottom_stable = R[(R.rank_p05 >= 55)]["town"].tolist()
print(f"Always top-10 (95% of scenarios): {top_stable}")
print(f"Always bottom-15 (95% of scenarios): {bottom_stable}")
display(UA_TAB.groupby(["normalisation", "aggregation"])[["median_rho", "min_rho"]].mean().round(3))

r_ = R.sort_values("rank")
fig, ax = plt.subplots(figsize=(COL1 + 0.6, 9.2))
y = np.arange(len(r_))[::-1]
ax.hlines(y, r_["rank_p05"], r_["rank_p95"], color=GREY, lw=2.4, label="5th-95th percentile rank")
ax.scatter(r_["rank"], y, color="#222222", s=9, zorder=3, label="Baseline rank")
ax.set_yticks(y, r_["town"], fontsize=6); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
ax.set_ylim(-0.7, len(r_) - 0.3); ax.set_xlim(0, 70)
ax.set_xlabel("Rank (1 = best) across 3,600 scenarios")
ax.set_title("Rank uncertainty across 3,600 scenarios")
ax.legend(loc="upper right")
savefig(fig, "fig03_uncertainty_rank_bands.png")


# ## 6. Town typologies (unsupervised learning)
# 
# A single score says *how much* infrastructure a town has; typologies say *what kind of gap* it has. We compare three algorithms on standardised pillar scores for k = 2..6 using silhouette (higher better), Davies-Bouldin (lower better) and Calinski-Harabasz (higher better), then test **stability**: re-fit on 100 bootstrap samples and measure agreement with the full-data solution (Adjusted Rand Index).

# In[9]:


Xc = StandardScaler().fit_transform(R[PILLARS])


def fit_labels(method, k, X, seed=SEED):
    if method == "K-Means":
        m = KMeans(k, n_init=30, random_state=seed).fit(X); return m.labels_, m
    if method == "Ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(X), None
    m = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed).fit(X)
    return m.predict(X), m


rows = []
for method in ("K-Means", "Ward", "GMM"):
    for k in range(2, 7):
        lab, _ = fit_labels(method, k, Xc)
        rows.append({"method": method, "k": k, "silhouette": silhouette_score(Xc, lab),
                     "davies_bouldin": davies_bouldin_score(Xc, lab), "calinski_harabasz": calinski_harabasz_score(Xc, lab)})
CL_TAB = pd.DataFrame(rows)


def bootstrap_ari(method, k, X, B=100):
    full, _ = fit_labels(method, k, X)
    rng = np.random.default_rng(SEED); aris = []
    for b in range(B):
        idx = rng.choice(len(X), len(X), replace=True)
        lab_b, _ = fit_labels(method, k, X[idx], seed=b)
        cents = np.vstack([X[idx][lab_b == c].mean(0) for c in np.unique(lab_b)])
        pred = np.argmin(((X[:, None, :] - cents[None]) ** 2).sum(-1), axis=1)   # nearest-centroid assignment
        aris.append(adjusted_rand_score(full, pred))
    return np.mean(aris), np.std(aris)


K_FINAL = 4
stab = {m: bootstrap_ari(m, K_FINAL, Xc) for m in ("K-Means", "Ward", "GMM")}
CL_TAB["bootstrap_ARI_k4"] = CL_TAB.apply(lambda r: stab[r.method][0] if r.k == K_FINAL else np.nan, axis=1)
display(CL_TAB.round(3).pivot(index="k", columns="method", values="silhouette").rename_axis("silhouette by k"))
display(CL_TAB[CL_TAB.k == K_FINAL].round(3))
best_method = CL_TAB[CL_TAB.k == K_FINAL].sort_values("bootstrap_ARI_k4", ascending=False).iloc[0]["method"]
print(f"Selected: {best_method}, k = {K_FINAL} (highest bootstrap stability among methods at k = 4)")


# In[10]:


labels, _ = fit_labels(best_method, K_FINAL, Xc)
R["cluster_id"] = labels
prof = R.groupby("cluster_id")[PILLARS + ["infrascore"]].mean().sort_values("infrascore", ascending=False)
order = list(prof.index)
names = {order[0]: "C1 Urban hubs", order[-1]: "C4 Lagging"}
mid = order[1:-1]
fin_minus_conn = {c: prof.loc[c, "finance"] - prof.loc[c, "connectivity"] for c in mid}
mid_sorted = sorted(mid, key=lambda c: -fin_minus_conn[c])
names[mid_sorted[0]] = "C2 Taluka centres"
names[mid_sorted[1]] = "C3 Peri-urban"
R["cluster"] = R["cluster_id"].map(names)
CL_PROF = R.groupby("cluster").agg(n=("town", "count"), **{p: (p, "mean") for p in PILLARS + ["infrascore"]}).round(1)
display(CL_PROF)
for c, g in R.groupby("cluster"):
    print(f"{c} ({len(g)}): {', '.join(g.sort_values('population', ascending=False)['town'].head(10))}{' ...' if len(g) > 10 else ''}")

pcs = PCA(2, random_state=SEED).fit(Xc)
Z = pcs.transform(Xc)
fig, ax = plt.subplots(figsize=(COL1, 3.3))
pal = dict(zip(sorted(R.cluster.unique()), CLUSTER_COLORS))
for c, g in R.groupby("cluster"):
    ax.scatter(Z[g.index, 0], Z[g.index, 1], s=8 + g.population / 1400, color=pal[c], alpha=0.85,
               edgecolor="white", linewidth=0.5, label=f"{c} (n={len(g)})", zorder=3)
lab = list(R.nlargest(6, "population").index) + list(R[R.town.isin(["Cuncolim", "Quepem", "Sancoale", "Ponda"])].index)
lab = list(dict.fromkeys(lab))
label_points(ax, Z[lab, 0], Z[lab, 1], R.town[lab].tolist(), fontsize=6)
ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} of variance): overall level")
ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
ax.set_title(f"Town typologies ({best_method}, k = 4)")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, columnspacing=1.0,
                markerscale=0.8)
for h in leg.legend_handles:
    h.set_sizes([25])
savefig(fig, "fig04_typologies.png")


# ## 7. Peer-adjusted gap model (supervised regression)
# 
# **Question:** given a town's size, density, social composition, civic status and geography, what InfraScore would we *expect*? The residual (actual minus expected) separates genuine under-provision from differences explained by size or status.
# 
# **Design**
# * Inputs: 10 context variables. **No index ingredient is an input** (no leakage).
# * 7 models: mean baseline, Ridge, Lasso, SVR (RBF), k-NN, Random Forest, XGBoost.
# * **Nested cross-validation**: hyperparameters tuned by an inner 5-fold grid search inside each outer fold; outer loop = 5-fold x 3 repeats. Reported scores are therefore unbiased by tuning.
# * **Ablation**: add feature groups one at a time.
# * **Conformal-style intervals**: out-of-fold absolute residuals give a 90% band; a town is *significantly* under-served only if its actual score falls below the band.

# In[11]:


GROUPS = {
    "Demand": ["log_population", "log_density", "avg_household_size"],
    "Social": ["sc_share", "st_share", "sex_ratio"],
    "Administrative": ["is_statutory_town", "is_south_goa"],
    "Geography": ["rainfall_mm", "log_area"],
}
CTX = sum(GROUPS.values(), [])
D = F.merge(R[["census_code", "infrascore"]], on="census_code")
X, yv = D[CTX], D["infrascore"]
inner = KFold(5, shuffle=True, random_state=SEED)
outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)

MODELS = {
    "Mean baseline": (DummyRegressor(), None),
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-2, 3, 12)}),
    "Lasso": (make_pipeline(StandardScaler(), Lasso(max_iter=50000)), {"lasso__alpha": np.logspace(-2, 1, 12)}),
    "SVR (RBF)": (make_pipeline(StandardScaler(), SVR()), {"svr__C": [3, 10, 30, 100], "svr__epsilon": [1, 3]}),
    "k-NN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 5, 8, 12]}),
    "Random Forest": (RandomForestRegressor(n_estimators=100, max_features=0.5, random_state=SEED),
                      {"min_samples_leaf": [3, 6]}),
}
if HAS_XGB:
    MODELS["XGBoost"] = (XGBRegressor(n_estimators=150, learning_rate=0.05, subsample=0.8, random_state=SEED,
                                      n_jobs=1, verbosity=0), {"max_depth": [1, 2, 3]})


def tuned(est, grid):
    return GridSearchCV(est, grid, cv=inner, scoring="neg_mean_absolute_error") if grid else est


rows = []
for name, (est, grid) in MODELS.items():
    t = time.time()
    cv = cross_validate(tuned(est, grid), X, yv, cv=outer, scoring=("neg_mean_absolute_error", "r2"))
    rows.append({"model": name, "MAE": -cv["test_neg_mean_absolute_error"].mean(),
                 "MAE_sd": cv["test_neg_mean_absolute_error"].std(), "R2": cv["test_r2"].mean(),
                 "seconds": time.time() - t})
GAP_TAB = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
BEST = GAP_TAB[GAP_TAB.model != "Mean baseline"].iloc[0]["model"]
base_mae = GAP_TAB.set_index("model").loc["Mean baseline", "MAE"]
best_mae = GAP_TAB.set_index("model").loc[BEST, "MAE"]
print(f"Selected model: {BEST} (nested-CV MAE {best_mae:.2f} vs baseline {base_mae:.2f}; "
      f"{1 - best_mae / base_mae:.0%} lower error)")
display(GAP_TAB.round(3))


# In[12]:


# ---- ablation with the selected model family
rows, cols = [], []
for gname, gcols in GROUPS.items():
    cols = cols + gcols
    est, grid = MODELS[BEST]
    cv = cross_validate(tuned(est, grid), D[cols], yv, cv=outer, scoring=("neg_mean_absolute_error", "r2"))
    rows.append({"features": " + ".join(list(GROUPS)[:list(GROUPS).index(gname) + 1]), "n_features": len(cols),
                 "MAE": -cv["test_neg_mean_absolute_error"].mean(), "R2": cv["test_r2"].mean()})
ABL_TAB = pd.DataFrame(rows)
display(ABL_TAB.round(3))

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
g = GAP_TAB.sort_values("MAE", ascending=False)
cols_ = ["#2a9d8f" if m == BEST else ("#d9d9d9" if m == "Mean baseline" else GREY) for m in g.model]
axes[0].barh(g.model, g.MAE, xerr=g.MAE_sd, color=cols_, height=0.65,
             error_kw=dict(ecolor="#555555", lw=0.8, capsize=2))
for yy, (m, sd) in enumerate(zip(g.MAE, g.MAE_sd)):
    axes[0].text(m + sd + 0.15, yy, f"{m:.2f}", va="center", fontsize=6.5)
axes[0].set_xlim(0, 11.5); axes[0].grid(axis="y", visible=False); axes[0].tick_params(axis="y", length=0)
axes[0].set_xlabel("Nested-CV MAE, index points (lower is better)")
axes[0].set_title("(a) Gap model comparison")
abl_lbl = [f"{n}\n({k})" for n, k in zip(["Demand", "+Social", "+Admin.", "+Geography"], ABL_TAB.n_features)]
bars = axes[1].bar(abl_lbl, ABL_TAB.R2, color=["#9fc5d4", "#9fc5d4", "#5b9bb3", NORTH], width=0.55)
axes[1].tick_params(axis="x", labelsize=6.5)
for b, v in zip(bars, ABL_TAB.R2):
    axes[1].text(b.get_x() + b.get_width() / 2, v + (0.008 if v >= 0 else -0.008), f"{v:.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=6.5)
axes[1].axhline(0, color="#666666", lw=0.7)
axes[1].set_ylim(min(ABL_TAB.R2.min(), 0) - 0.03, max(ABL_TAB.R2) + 0.05); axes[1].grid(axis="x", visible=False)
axes[1].set_ylabel("Nested-CV R$^2$")
axes[1].set_title(f"(b) Ablation, {BEST} (no. of features)")
fig.tight_layout(w_pad=2.5)
savefig(fig, "fig05_model_comparison_ablation.png")


# In[13]:


# ---- out-of-fold expected scores (averaged over 5 shuffles) and conformal intervals
est, grid = MODELS[BEST]
oof = np.mean([cross_val_predict(tuned(est, grid), X, yv, cv=KFold(5, shuffle=True, random_state=s)) for s in range(5)], axis=0)
GAP = D[["census_code", "town", "district", "civic_status", "population", "infrascore"]].copy()
GAP["expected"] = oof
GAP["gap"] = GAP["infrascore"] - GAP["expected"]
q90 = np.quantile(np.abs(GAP["gap"]), 0.90)
GAP["lower_90"], GAP["upper_90"] = GAP["expected"] - q90, GAP["expected"] + q90
GAP["flag"] = np.select([GAP.infrascore < GAP.lower_90, GAP.infrascore > GAP.upper_90],
                        ["Significantly under-served", "Significantly over-performing"], "Within expected range")
GAP = GAP.sort_values("gap").reset_index(drop=True)
print(f"90% interval half-width: +/- {q90:.1f} points")
print(GAP.flag.value_counts().to_string())
display(GAP[GAP.flag != "Within expected range"][["town", "civic_status", "infrascore", "expected", "gap", "flag"]].round(1))
display(Markdown("**Ten largest negative gaps (watch list)**"))
display(GAP.head(10)[["town", "civic_status", "population", "infrascore", "expected", "gap", "flag"]].round(1))


# In[14]:


# ---- explanation: fit on all data; for linear models contributions are exact (coef x standardised value)
final = tuned(est, grid).fit(X, yv)
model = final.best_estimator_ if hasattr(final, "best_estimator_") else final
if BEST in ("Ridge", "Lasso"):
    lin = model[-1]
    Zs = model[0].transform(X)
    COEF = pd.DataFrame({"feature": CTX, "coef_per_SD": lin.coef_}).sort_values("coef_per_SD", key=abs, ascending=False)
    CONTRIB = pd.DataFrame(Zs * lin.coef_, columns=CTX, index=D["town"])
else:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X, yv, n_repeats=30, random_state=SEED)
    COEF = pd.DataFrame({"feature": CTX, "coef_per_SD": pi.importances_mean}).sort_values("coef_per_SD", ascending=False)
    CONTRIB = None
display(COEF.round(2))

nz = COEF[COEF.coef_per_SD.abs() > 1e-6].sort_values("coef_per_SD")
zeroed = [PRETTY.get(f, f) for f in COEF[COEF.coef_per_SD.abs() <= 1e-6].feature]
fig, ax = plt.subplots(figsize=(COL1, 2.0))
ax.barh([PRETTY.get(f, f) for f in nz.feature], nz.coef_per_SD, height=0.6,
        color=np.where(nz.coef_per_SD >= 0, "#2a9d8f", "#c8553d"))
for yy, v in enumerate(nz.coef_per_SD):
    ax.text(v + (0.1 if v >= 0 else -0.1), yy, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5)
ax.axvline(0, color="#666666", lw=0.7); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_xlim(nz.coef_per_SD.min() - 1, nz.coef_per_SD.max() + 1)
ax.set_xlabel("Effect on expected InfraScore (points per 1 SD)")
ax.set_title(f"Drivers of the expected score ({BEST})")
if zeroed:
    ax.text(0, -0.32, "Shrunk to zero: " + ", ".join(zeroed), transform=ax.transAxes, fontsize=6, color="#555555")
savefig(fig, "fig06_gap_drivers.png")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
FLAG_COL = {"Significantly under-served": "#c8553d", "Significantly over-performing": "#2a9d8f",
            "Within expected range": GREY}
fig, ax = plt.subplots(figsize=(COL1, 3.2))
lo = np.floor(min(GAP.expected.min(), GAP.infrascore.min()) / 5) * 5 - 2
hi = np.ceil(max(GAP.expected.max(), GAP.infrascore.max()) / 5) * 5 + 2
xx = np.linspace(lo, hi, 10)
ax.fill_between(xx, xx - q90, xx + q90, color="#eef1f4", zorder=1)
ax.plot(xx, xx, "--", color="#999999", lw=0.8, zorder=2)
ax.scatter(GAP.expected, GAP.infrascore, c=GAP.flag.map(FLAG_COL), s=16, edgecolor="white", linewidth=0.4, zorder=3)
lab = GAP[(GAP.flag != "Within expected range") | (GAP.gap.rank() <= 5)]
label_points(ax, lab.expected.values, lab.infrascore.values, lab.town.tolist(), fontsize=6,
             sides=["l" if f == "Significantly over-performing" else "r" for f in lab.flag])
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Expected InfraScore (out-of-fold)"); ax.set_ylabel("Actual InfraScore")
ax.set_title("Peer-adjusted gaps (actual vs expected)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FLAG_COL["Significantly under-served"], label="Under-served"),
                   Line2D([], [], marker="o", ls="", color=FLAG_COL["Significantly over-performing"], label="Over-performing"),
                   Line2D([], [], marker="o", ls="", color=GREY, label="Within range"),
                   Patch(color="#eef1f4", label=f"90% range (±{q90:.1f})")],
          loc="upper left", handletextpad=0.2, borderaxespad=0.3, markerscale=0.7, fontsize=6.2, labelspacing=0.3)
savefig(fig, "fig07_gap_actual_vs_expected.png")


# ## 8. Facility presence model (supervised classification)
# 
# For six facility types we ask: *does this town have one, and should it?* A classifier learns, from the same 10 context variables, which towns typically host each facility. Towns that **lack** the facility but receive a **high out-of-fold probability** look like towns that do have it: they are candidate sites. This is the shortlist the policy simulation draws from.
# 
# Metrics suit imbalanced classes: ROC-AUC, PR-AUC (compared with prevalence, the no-skill level), F1 and Brier score. Evaluation: repeated stratified 5-fold CV (3 repeats), class-balanced weights.

# In[15]:


TARGETS = {
    "Private bank": raw["n_private_commercial_bank"] > 0,
    "Co-operative bank": raw["n_co_operative_bank"] > 0,
    "Public library": F["acc_library_km"] == 0,
    "Auditorium / community hall": F["acc_auditorium_km"] == 0,
    "Degree college": F["acc_degree_college_km"] == 0,
    "Stadium": F["acc_stadium_km"] == 0,
}
CLF = {
    "Logistic (L2)": GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                                  {"logisticregression__C": [0.1, 0.3, 1, 3]}, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  scoring="roc_auc"),
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
}
rows, PROBS = [], {}
for tname, yt in TARGETS.items():
    yt = yt.astype(int).values
    for mname, clf in CLF.items():
        probs = np.mean([cross_val_predict(clf, X, yt, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                           method="predict_proba")[:, 1] for s in range(3)], axis=0)
        rows.append({"facility": tname, "model": mname, "positives": int(yt.sum()), "prevalence": yt.mean(),
                     "ROC_AUC": roc_auc_score(yt, probs), "PR_AUC": average_precision_score(yt, probs),
                     "F1": f1_score(yt, probs >= 0.5), "Brier": brier_score_loss(yt, probs)})
        PROBS[(tname, mname)] = probs
FAC_TAB = pd.DataFrame(rows)
display(FAC_TAB.round(3))
best_clf = FAC_TAB.groupby("model")["ROC_AUC"].mean().idxmax()
print(f"Mean ROC-AUC by model: {FAC_TAB.groupby('model')['ROC_AUC'].mean().round(3).to_dict()} -> using {best_clf}")

cand = []
for tname, yt in TARGETS.items():
    p = PROBS[(tname, best_clf)]
    for i in np.where((~yt.values) & (p >= 0.5))[0]:
        cand.append({"facility": tname, "town": F.town[i], "census_code": F.census_code[i],
                     "probability": p[i], "population": F.population[i]})
CAND = pd.DataFrame(cand).sort_values(["facility", "probability"], ascending=[True, False])
print(f"\n{len(CAND)} candidate sites (facility absent, out-of-fold probability >= 0.5)")
display(CAND.groupby("facility").apply(lambda g: ", ".join(g.town.head(6))).rename("top candidates").to_frame())

fb = FAC_TAB[FAC_TAB.model == best_clf].set_index("facility").sort_values("ROC_AUC")
fig, ax = plt.subplots(figsize=(COL1, 2.6))
yi = np.arange(len(fb))
ax.barh(yi + 0.19, fb.ROC_AUC, 0.36, color=NORTH, label="ROC-AUC")
ax.barh(yi - 0.19, fb.PR_AUC, 0.36, color=SOUTH, label="PR-AUC")
ax.scatter(fb.prevalence, yi - 0.19, marker="|", s=90, color="#222222", zorder=3, label="PR-AUC no-skill (prevalence)")
for k, (a, p) in enumerate(zip(fb.ROC_AUC, fb.PR_AUC)):
    ax.text(a + 0.01, k + 0.19, f"{a:.2f}", va="center", fontsize=6)
    ax.text(p + 0.01, k - 0.19, f"{p:.2f}", va="center", fontsize=6)
ax.axvline(0.5, color="#999999", ls="--", lw=0.7)
ax.set_yticks(yi, [f.replace("Auditorium / community hall", "Auditorium / hall") for f in fb.index])
ax.set_xlim(0, 1.05); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Score (repeated stratified 5-fold CV)")
ax.set_title(f"Facility presence model ({best_clf})")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, handletextpad=0.3, columnspacing=0.8)
savefig(fig, "fig08_facility_model.png")


# ## 9. Policy simulation: where should the next investments go?
# 
# **Decision problem.** A planner can fund *K* interventions (each one facility in one town). Which set maximises benefit?
# 
# * **Action menu:** add a primary health centre, a hospital, a fire station, a bank branch (feasible only where the bank models give probability >= 0.5 or a bank already exists), or one of the facilities from Section 8 at its candidate sites.
# * **Re-scoring:** the town's indicators are updated and re-scored with the *frozen* baseline scaler, so scores stay comparable.
# * **Objective (equity-weighted):** benefit = population x score gain x $(100-\text{IS})/(100-\overline{\text{IS}})$, giving more weight to gains in weaker towns.
# * **Strategies compared:** (1) greedy model-guided optimiser, (2) largest towns first, (3) lowest-score towns first, (4) random feasible actions (mean of 200 runs).
# 
# Interventions are counted in equal units; no cost data are assumed.

# In[16]:


IS0 = aggregate(PIL, BASE_W).values
POP = F["population"].values
MEAN_IS = IS0.mean()
cand_set = set(zip(CAND.facility, CAND.census_code))
bank_ok = (raw["n_private_commercial_bank"] + raw["n_co_operative_bank"] + raw["n_nationalised_bank"] > 0).values | \
          np.isin(F.census_code, CAND[CAND.facility.str.contains("bank")].census_code)
ACTIONS = ["Primary health centre", "Hospital", "Fire station", "Bank branch", "Public library",
           "Auditorium / community hall", "Degree college", "Stadium"]


def apply(row, action):
    r = dict(row)
    if action == "Primary health centre": r["primary_care_per_10k"] += 10_000 / r["population"]
    elif action == "Hospital": r["hospitals_per_10k"] += 10_000 / r["population"]; r["acc_hospital_km"] = 0.0
    elif action == "Fire station": r["acc_fire_km"] = 0.0
    elif action == "Bank branch": r["banks_per_10k"] += 10_000 / r["population"]
    elif action == "Public library": r["acc_library_km"] = 0.0
    elif action == "Auditorium / community hall": r["acc_auditorium_km"] = 0.0
    elif action == "Degree college": r["acc_degree_college_km"] = 0.0
    elif action == "Stadium": r["acc_stadium_km"] = 0.0
    return r


def feasible(i, action, state):
    if action in ("Public library", "Auditorium / community hall", "Degree college", "Stadium"):
        return (action, F.census_code[i]) in cand_set and state[i][{"Public library": "acc_library_km",
                "Auditorium / community hall": "acc_auditorium_km", "Degree college": "acc_degree_college_km",
                "Stadium": "acc_stadium_km"}[action]] > 0
    if action == "Hospital": return state[i]["acc_hospital_km"] > 0
    if action == "Fire station": return state[i]["acc_fire_km"] > 0
    if action == "Bank branch": return bank_ok[i]
    return True


IND_ITEMS = list(INDICATORS.items())
W_NORM = {p: BASE_W[p] / sum(BASE_W.values()) for p in PILLARS}


def rescore(row, lam=10.0):
    """Fast scalar version of score_indicators -> pillar_table -> aggregate (baseline settings, frozen bounds)."""
    acc = {p: [] for p in PILLARS}
    for c, (pil, kind) in IND_ITEMS:
        v = row[c]
        if kind == "access":
            sc = 100 * np.exp(-v / lam)
        elif kind == "ratio":
            sc = 100 * v
        else:
            lo, hi = BOUNDS[c][0], BOUNDS[c][1]
            sc = 100 * (min(max(v, lo), hi) - lo) / (hi - lo)
        acc[pil].append(sc)
    return sum(W_NORM[p] * np.mean(acc[p]) for p in PILLARS)


assert abs(rescore(F.iloc[0]) - IS0[0]) < 1e-9, "fast re-scorer must match the index"


def best_action_gain(i, state, cur):
    best = (None, 0.0, None)
    for a in ACTIONS:
        if feasible(i, a, state):
            new_row = apply(state[i], a)
            g = rescore(new_row) - cur[i]
            if g > best[1]:
                best = (a, g, new_row)
    return best


def equity_benefit(i, gain, cur):
    return POP[i] * gain * (100 - cur[i]) / (100 - MEAN_IS)


def run(strategy, K, seed=0):
    state, cur, plan = F.to_dict("records"), IS0.copy(), []
    rng = np.random.default_rng(seed)
    for _ in range(K):
        if strategy == "Greedy optimiser":
            opts = [(equity_benefit(i, *best_action_gain(i, state, cur)[1:2], cur), i) for i in range(len(F))]
            _, i = max(opts)
        elif strategy == "Largest towns first":
            i = next(j for j in np.argsort(-POP) if best_action_gain(j, state, cur)[0])
        elif strategy == "Lowest score first":
            i = next(j for j in np.argsort(cur) if best_action_gain(j, state, cur)[0])
        else:   # random feasible action
            while True:
                i, a = rng.integers(len(F)), ACTIONS[rng.integers(len(ACTIONS))]
                if feasible(i, a, state):
                    new_row = apply(state[i], a); break
        if strategy != "Random":
            a, g, new_row = best_action_gain(i, state, cur)
        state[i] = new_row
        new = rescore(new_row)
        plan.append({"step": len(plan) + 1, "town": F.town[i], "action": a, "gain": new - cur[i]})
        cur[i] = new
    return cur, pd.DataFrame(plan)


def metrics(cur):
    e_cut = np.quantile(IS0, 0.2)
    gini = lambda v: np.abs(v[:, None] - v[None]).mean() / (2 * v.mean())
    return {"pop_weighted_gain": np.average(cur - IS0, weights=POP), "mean_gain": (cur - IS0).mean(),
            "towns_left_tier_E": int(((IS0 <= e_cut) & (cur > e_cut)).sum()), "gini_after": gini(cur),
            "gain_in_bottom_quintile": (cur - IS0)[IS0 <= e_cut].sum()}


t = time.time()
rows, PLANS = [], {}
for K in (5, 10, 20):
    for strat in ("Greedy optimiser", "Largest towns first", "Lowest score first"):
        cur, plan = run(strat, K)
        rows.append({"budget_K": K, "strategy": strat, **metrics(cur)}); PLANS[(strat, K)] = plan
    rnd = [metrics(run("Random", K, seed=s)[0]) for s in range(30)]
    rows.append({"budget_K": K, "strategy": "Random (mean)", **pd.DataFrame(rnd).mean().to_dict()})
SIM = pd.DataFrame(rows)
print(f"Simulation time {time.time() - t:.0f}s | baseline Gini {metrics(IS0)['gini_after']:.4f}")
display(SIM.round(3))
display(Markdown("**Greedy plan for K = 10 interventions**"))
display(PLANS[("Greedy optimiser", 10)].round(2))

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))
order_ = ["Greedy optimiser", "Lowest score first", "Random (mean)", "Largest towns first"]
for strat in order_:
    s = SIM[SIM.strategy == strat]
    kw = dict(color=STRAT_COLORS[strat], marker="o", ms=3.5, lw=1.6 if strat == "Greedy optimiser" else 1.2)
    axes[0].plot(s.budget_K, s.pop_weighted_gain, label=strat, **kw)
    axes[1].plot(s.budget_K, s.gain_in_bottom_quintile, **kw)
    axes[0].text(20.6, s.pop_weighted_gain.iloc[-1], f"{s.pop_weighted_gain.iloc[-1]:.2f}", va="center",
                 fontsize=6.3, color=STRAT_COLORS[strat])
    axes[1].text(20.6, s.gain_in_bottom_quintile.iloc[-1], f"{s.gain_in_bottom_quintile.iloc[-1]:.0f}", va="center",
                 fontsize=6.3, color=STRAT_COLORS[strat])
for a in axes:
    a.set_xticks([5, 10, 20]); a.set_xlim(4, 22.5); a.set_xlabel("Interventions funded (K)")
axes[0].set_ylabel("Points"); axes[1].set_ylabel("Points (sum over towns)")
axes[0].set_title("(a) Population-weighted score gain")
axes[1].set_title("(b) Gain in bottom-quintile towns")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=3)
savefig(fig, "fig09_policy_simulation.png")


# ### 9b. Cost-aware allocation and robustness to cost assumptions
# 
# Facilities do not cost the same. We repeat the simulation with a **budget in cost units** (1 unit = one primary health centre) and a cost-aware greedy rule that maximises *benefit per unit cost*.
# 
# Exact unit costs are not published for Goa, so we use **illustrative relative costs**, anchored on one reported figure (a 120-bed sub-district hospital at Chicalim built for about Rs 30 crore, Herald Goa), and then test whether the conclusion survives **200 random cost scenarios** in which every cost is multiplied by an independent log-normal factor (about ±50%).

# In[17]:


BASE_COST = {"Primary health centre": 1.0, "Hospital": 6.0, "Fire station": 2.0, "Bank branch": 0.5,
             "Public library": 0.5, "Auditorium / community hall": 1.0, "Degree college": 5.0, "Stadium": 3.0}
BUDGET = 20.0


def town_options(i, state, cur):
    """All feasible actions for town i with their score gain and new row."""
    opts = []
    for a in ACTIONS:
        if feasible(i, a, state):
            new_row = apply(state[i], a)
            g = rescore(new_row) - cur[i]
            if g > 1e-9:
                opts.append((a, g, new_row))
    return opts


def run_budget(strategy, cost, budget=BUDGET, seed=0):
    state, cur, spent, plan = F.to_dict("records"), IS0.copy(), 0.0, []
    cache = {i: town_options(i, state, cur) for i in range(len(F))}
    rng = np.random.default_rng(seed)
    while True:
        left = budget - spent
        cands = [(i, a, g, row) for i, ops in cache.items() for a, g, row in ops if cost[a] <= left + 1e-9]
        if not cands:
            break
        if strategy == "Cost-aware greedy":
            i, a, g, row = max(cands, key=lambda c: equity_benefit(c[0], c[2], cur) / cost[c[1]])
        elif strategy == "Greedy (ignores cost)":
            i, a, g, row = max(cands, key=lambda c: equity_benefit(c[0], c[2], cur))
        elif strategy == "Lowest score first":
            j = min({c[0] for c in cands}, key=lambda t: cur[t])
            i, a, g, row = max([c for c in cands if c[0] == j], key=lambda c: c[2])
        else:
            i, a, g, row = cands[rng.integers(len(cands))]
        state[i] = row
        cur[i] = rescore(row)
        spent += cost[a]
        plan.append({"town": F.town[i], "action": a, "cost": cost[a], "gain": g})
        cache[i] = town_options(i, state, cur)
    return cur, pd.DataFrame(plan)


STRATS_B = ["Cost-aware greedy", "Greedy (ignores cost)", "Lowest score first", "Random"]
t = time.time()
base_rows, BUDGET_PLANS = [], {}
for s_ in STRATS_B:
    reps = 10 if s_ == "Random" else 1
    res = [run_budget(s_, BASE_COST, seed=k) for k in range(reps)]
    m = pd.DataFrame([metrics(c) for c, _ in res]).mean()
    base_rows.append({"strategy": s_, "interventions": np.mean([len(p) for _, p in res]), **m.to_dict()})
    BUDGET_PLANS[s_] = res[0][1]
COST_TAB = pd.DataFrame(base_rows)

rng = np.random.default_rng(SEED)
mc = []
for d in range(200):
    cost = {a: c * float(np.exp(rng.normal(0, 0.4))) for a, c in BASE_COST.items()}
    row = {}
    for s_ in STRATS_B[:3]:
        row[s_] = metrics(run_budget(s_, cost)[0])["pop_weighted_gain"]
    row["Random"] = metrics(run_budget("Random", cost, seed=d)[0])["pop_weighted_gain"]
    mc.append(row)
MC = pd.DataFrame(mc)
win = {s_: (MC["Cost-aware greedy"] > MC[s_]).mean() for s_ in STRATS_B[1:]}
COST_TAB["mean_gain_200_cost_draws"] = COST_TAB.strategy.map(MC.mean())
print(f"Budget = {BUDGET:g} cost units; simulation time {time.time() - t:.0f}s")
display(COST_TAB.round(3))
print("Share of 200 cost scenarios in which cost-aware greedy beats:",
      {k: f"{v:.0%}" for k, v in win.items()})
display(Markdown("**Cost-aware greedy plan at baseline costs**"))
display(BUDGET_PLANS["Cost-aware greedy"].round(2))


# ## 10. External validation and state context
# 
# **Validity test.** If InfraScore measures something real, better-served towns should show better human-development outcomes. Literacy is *not* used in the score. We report the raw Spearman correlation and a partial correlation controlling for town size (log population), so the result is not just "bigger towns are more literate".

# In[18]:


v = F[["census_code", "literacy_rate", "log_population"]].merge(R[["census_code", "infrascore"] + PILLARS], on="census_code").dropna()


def partial_spearman(a, b, ctrl):
    ra, rb, rc = (pd.Series(x).rank().values for x in (a, b, ctrl))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return spearmanr(res(ra), res(rb))


val_rows = []
for col in ["infrascore"] + PILLARS:
    r1 = spearmanr(v[col], v.literacy_rate); r2 = partial_spearman(v[col], v.literacy_rate, v.log_population)
    val_rows.append({"measure": col, "spearman_rho": r1.statistic, "p": r1.pvalue,
                     "partial_rho_ctrl_size": r2.statistic, "p_partial": r2.pvalue})
VAL = pd.DataFrame(val_rows)
print(f"n = {len(v)} towns with valid literacy data")
display(VAL.round(3))


# In[19]:


d = pd.read_csv(DIST_CSV)
size_cols = ["Household_size_1_to_2_persons", "Household_size_3_to_5_persons_Households",
             "Household_size_6_8_persons_Households", "Household_size_9_persons_and_above_Households"]
d = d.assign(hh=d[size_cols].sum(axis=1))      # the file's 'Households' column overstates totals; see README
MEAS = {"Electric lighting": "Housholds_with_Electric_Lighting",
        "Latrine in premises": "Having_latrine_facility_within_the_premises_Total_Households",
        "Tap water": "Main_source_of_drinking_water_Tapwater_Households",
        "LPG/PNG": "LPG_or_PNG_Households", "Internet": "Households_with_Internet", "Computer": "Households_with_Computer"}
pct = pd.DataFrame({k: d[c] / d.hh * 100 for k, c in MEAS.items()})
pct["state"], pct["district"] = d["State name"].str.title(), d["District name"]
goa = pct[pct.state == "Goa"].set_index("district")[list(MEAS)]
BENCH = goa.T
BENCH["India median district"] = pct[list(MEAS)].median()
for dist in goa.index:
    BENCH[f"{dist} percentile"] = [(pct[m] < goa.loc[dist, m]).mean() * 100 for m in MEAS]
print(f"India total households (check vs official 24.67 crore): {d.hh.sum() / 1e7:.2f} crore")
display(BENCH.round(1))

fig, ax = plt.subplots(figsize=(COL1 + 1.2, 2.4))
xi = np.arange(len(MEAS))
for k, (col, c) in enumerate(zip(["North Goa", "South Goa", "India median district"], [NORTH, SOUTH, GREY])):
    ax.bar(xi + (k - 1) * 0.27, BENCH[col], 0.27, color=c, label=col)
ax.set_xticks(xi, [m.replace(" in premises", "\nin premises").replace("Electric ", "Electric\n") for m in MEAS], fontsize=6.8)
ax.grid(axis="x", visible=False); ax.set_ylim(0, 105)
ax.set_ylabel("% of households (2011)")
ax.set_title("Goa's districts vs the median Indian district")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
savefig(fig, "fig10_district_benchmark.png")


# ### 10b. Real-world corroboration after 2011
# 
# The model only sees 2011 data. A strong test is whether the towns and facilities it prioritises were later the subject of government action or documented public demand. We searched news and government sources for each town in the greedy K = 10 plan (Section 9), and applied the same search protocol to **randomly drawn control towns** that are not in the plan (four in V1, enlarged to thirteen in V2, of which twelve lacked a hospital or fire station in 2011). Samples remain small, and news coverage is uneven across towns, so the test is limited corroboration rather than proof.

# In[20]:


# match = "exact" when the evidence concerns the same facility type the plan assigns,
#         "related" when it concerns a lesser facility of the same kind (e.g. emergency care or a dispensary for a hospital)
EVIDENCE = pd.DataFrame([
    ("Cuncolim", "Fire station", "exact", "MHA fire hazard and risk analysis recommended an urban fire station; DFES later commissioned one",
     "dfes.goa.gov.in/?p=16254"),
    ("Cuncolim", "Hospital", "related", "Local MLA (2021): a 24x7 emergency care centre was a long-pending demand",
     "thegoan.net/goa-news/.../74394.html"),
    ("Quepem", "Hospital", "exact", "Residents demanded a 30-bed hospital; a completed health centre stood unopened for years",
     "heraldgoa.in/goa/when-will-quepem-get-a-full-fledged-health-centre/73538"),
    ("Curchorem-Cacora", "Hospital", "exact", "Community health centre upgraded to 82 beds, announced as Goa's second sub-district hospital",
     "heraldgoa.in/goa/curchorem-to-have-goas-second-sub-district-hospital/272916"),
    ("Usgao", "Fire station", "exact", "DFES 25-year retrospective: a fire station at Usgaon is being proposed",
     "dfes.goa.gov.in (25 years of Goa Fire & Emergency Services, 2025)"),
    ("Sancoale", "Hospital", "related", "Government takeover of the local dispensary was a long-pending demand; sarpanch asked for beds and night doctors",
     "heraldgoa.in/goa/row-over-sancoale-dispensary-takeover/313360"),
    ("Sanquelim", "Hospital", "related", "GSIDC project for reconstruction / upgradation of the Primary Health Centre at Sanquelim",
     "gsidcltd.com/project/page/78"),
], columns=["town", "facility", "match", "evidence (post-2011)", "source"])

plan10 = PLANS[("Greedy optimiser", 10)][["step", "town", "action"]]
CHK = plan10.merge(EVIDENCE[["town", "facility", "match", "evidence (post-2011)"]], left_on=["town", "action"],
                   right_on=["town", "facility"], how="left").drop(columns="facility")
CHK["corroborated"] = CHK["evidence (post-2011)"].notna()
CHK["match"] = CHK["match"].fillna("none")
searched = set(EVIDENCE.town)
CHK.loc[CHK.town.isin(searched) & ~CHK.corroborated, "evidence (post-2011)"] = "none found in our search"
CHK.loc[~CHK.town.isin(searched), "evidence (post-2011)"] = "not searched"
display(CHK)
towns_plan = plan10.town.unique()
exact_towns = set(CHK[CHK.match == "exact"].town)
any_towns = set(CHK[CHK.corroborated].town)
print(f"Plan assignments with exact evidence: {(CHK.match == 'exact').sum()} of {len(CHK)}; "
      f"with exact or related evidence: {CHK.corroborated.sum()} of {len(CHK)}")
print(f"Towns with exact evidence: {len(exact_towns)} of {len(towns_plan)}; with exact or related evidence: "
      f"{len(any_towns)} of {len(towns_plan)} (not searched: {', '.join(sorted(set(towns_plan) - searched)) or 'none'})")
EVIDENCE.to_csv(OUT / "post2011_evidence.csv", index=False)

# ---- control comparison. Stage 1: four towns drawn at random (seed 42) from towns NOT in the plan.
# Stage 2 (added in V2 to give the test some power): the remaining non-plan towns are shuffled (seed 43) and the
# first nine that lacked a hospital or a fire station in 2011 are added. Each control is searched with the same
# protocol as the plan towns ("<town> Goa hospital / health centre", "<town> Goa fire station"; news and
# government sources) for the facility types it lacked in 2011, and coded with the same exact / related rule.
pool = sorted(set(R.town) - set(towns_plan))
controls = list(np.random.default_rng(SEED).choice(pool, 4, replace=False))
lack_all = F.set_index("town")[["acc_hospital_km", "acc_fire_km"]] > 0
rest = [t for t in pool if t not in controls]
extra = [t for t in np.random.default_rng(SEED + 1).permutation(rest) if lack_all.loc[t].any()][:9]
controls = controls + extra
lack = lack_all.loc[controls]
CONTROL = pd.DataFrame({
    "town": controls,
    "stage": [1] * 4 + [2] * len(extra),
    "facility types lacking in 2011": [", ".join(n for n, flag in zip(["Hospital", "Fire station"], row) if flag) or "none (ineligible)"
                                        for row in lack.values],
})
# (match, evidence, source); towns not listed: nothing found for the facility types they lacked
CONTROL_FOUND = {
    "Ponda": ("not applicable", "had both facility types in 2011", ""),
    "Siolim": ("exact", "Hospital: health minister announced upgrade of the PHC to a community health centre with casualty (Jan 2024)",
               "thegoan.net/goa-news/siolim-phc-to-be-upgraded-to-chc/108514.html"),
    "Sanguem": ("exact", "Hospital: new sub-district hospital with dialysis unit to open (Feb 2026); fire: residents' demand for a fire station",
                "medicaldialogues.in/.../164502; heraldgoa.in/goa/.../173855"),
    "Benaulim": ("related", "Hospital: proposed primary health centre opposed at a special gram sabha",
                 "heraldgoa.in/goa/commotion-over-proposed-health-centre-at-benaulim/151356"),
    "Anjuna": ("none", "Fire: editorial notes reliance on Mapusa station (45 min) but no proposal or demand for a station",
               "thegoan.net (editorial, Apr 2026)"),
}
CONTROL["match"] = [CONTROL_FOUND.get(t, ("none",))[0] for t in CONTROL.town]
CONTROL["post-2011 evidence (same search protocol)"] = [CONTROL_FOUND.get(t, (None, "none found"))[1] for t in CONTROL.town]
CONTROL["source"] = [CONTROL_FOUND.get(t, (None, None, ""))[2] for t in CONTROL.town]
display(Markdown("**Randomly drawn control towns (not in the plan)**")); display(CONTROL)
eligible = CONTROL[~CONTROL["facility types lacking in 2011"].str.startswith("none")]
ctrl_hits = (eligible["match"] == "exact").sum()
ctrl_any = eligible["match"].isin(["exact", "related"]).sum()
plan_hits = len(exact_towns)
from scipy.stats import fisher_exact
FISHER = fisher_exact([[plan_hits, len(towns_plan) - plan_hits], [ctrl_hits, len(eligible) - ctrl_hits]], alternative="greater")
FISHER_ANY = fisher_exact([[len(any_towns), len(towns_plan) - len(any_towns)], [ctrl_any, len(eligible) - ctrl_any]], alternative="greater")
print(f"Exact evidence: plan towns {plan_hits}/{len(towns_plan)} vs eligible control towns {ctrl_hits}/{len(eligible)} "
      f"(one-sided Fisher exact p = {FISHER.pvalue:.3f})")
print(f"Exact or related: plan towns {len(any_towns)}/{len(towns_plan)} vs controls {ctrl_any}/{len(eligible)} "
      f"(p = {FISHER_ANY.pvalue:.3f})")
CONTROL.to_csv(OUT / "post2011_controls.csv", index=False)


# ## 11. Second-review extensions
# 
# The first review asked for: (1) a precise account of analytical coverage, a town-level map and contemporary spatial proxies; (2) a clear separation of the ten-intervention experiment from the cost-budget experiment, with post-2011 records treated as limited corroboration; (3) realistic capital and operating costs, a test of interval coverage, and spatial spillovers. Sections 11a-11d implement these checks. Every number quoted in the V2 report is printed by the summary cell at the end.
# 
# ### 11a. Analytical coverage and the missing Orgao record

# In[21]:


OFFICIAL = {"Towns": 70, "Statutory towns": 14, "Census towns": 56, "Urban population": 906_814}  # Census 2011 (DPSE, Goa at a Glance)
COORDS = pd.read_csv(RAW / "spatial" / "town_coordinates.csv")
orgao = COORDS[COORDS.town.str.startswith("Orgao")].iloc[0]
analysed = {"Towns": len(F), "Statutory towns": int((F.civic_status != "Census Town").sum()),
            "Census towns": int((F.civic_status == "Census Town").sum()), "Urban population": int(F.population.sum())}
COVERAGE = pd.DataFrame({"official (Census 2011)": OFFICIAL, "analysed": analysed})
COVERAGE["share analysed"] = (COVERAGE["analysed"] / COVERAGE["official (Census 2011)"]).round(4)
MISSING_POP = OFFICIAL["Urban population"] - analysed["Urban population"]
print(f"Absent record: Orgao (census town, Ponda taluka). GeoNames population {int(orgao.geonames_pop):,}; "
      f"official urban population minus analysed towns = {MISSING_POP:,}")
display(COVERAGE)


def scores_without(drop_idx):
    """Re-estimate scaling bounds and scores with one town removed."""
    Fk = F.drop(index=drop_idx)
    s = aggregate(pillar_table(score_indicators(Fk, fit_bounds(Fk))), BASE_W)
    return pd.Series(s.values, index=Fk.index)


base_rank = pd.Series(IS0, index=F.index).rank(ascending=False)
jk = []
for i in F.index:
    s = scores_without(i).rank(ascending=False)
    ref = base_rank.drop(index=i).rank()
    jk.append({"dropped": F.town[i], "population": F.population[i],
               "max_rank_shift": float((s - ref).abs().max()), "spearman": spearmanr(s, ref).statistic})
JACK = pd.DataFrame(jk)
SMALL = JACK[JACK.population < 10_000]
print(f"Leave-one-town-out (69 runs): worst rank shift of any remaining town = {JACK.max_rank_shift.max():.0f} "
      f"(median {JACK.max_rank_shift.median():.0f}); dropping a town under 10,000 residents: worst shift "
      f"{SMALL.max_rank_shift.max():.0f}, minimum Spearman {SMALL.spearman.min():.3f}")


# ### 11b. Town map and contemporary spatial proxies
# 
# Town coordinates come from GeoNames (matched by name; 53 of 69 matches also agree exactly on the 2011 population). Road distance is approximated as straight-line distance times a circuity factor **estimated from the census itself**: the ratio of each town's reported road distance to its district headquarters to the straight-line distance.
# 
# Contemporary proxies (post-2011, public records): the current public inpatient network (Directorate of Health Services bed list tabled in the Legislative Assembly, plus Goa Medical College) and the 16 current fire stations (DFES). The two access indicators that these records cover are recomputed and the index is partially updated on the frozen 2011 scale.

# In[22]:


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


TC = F[["census_code", "town"]].merge(COORDS[["census_code", "lat", "lon"]], on="census_code", how="left")
assert TC.lat.notna().all(), "every analysed town needs coordinates"
LAT, LON = TC.lat.values, TC.lon.values
DMAT = haversine_km(LAT[:, None], LON[:, None], LAT[None, :], LON[None, :])

ctown = COORDS.set_index("town")
hq_lat = np.where(F.district == "North Goa", ctown.loc["Panaji", "lat"], ctown.loc["Margao", "lat"])
hq_lon = np.where(F.district == "North Goa", ctown.loc["Panaji", "lon"], ctown.loc["Margao", "lon"])
straight_hq = haversine_km(LAT, LON, hq_lat, hq_lon)
okc = (raw.dist_district_hq_km.values > 3) & (straight_hq > 2)
ratio = raw.dist_district_hq_km.values[okc] / straight_hq[okc]
CIRC = float(np.median(ratio))
print(f"Circuity factor (census road km / straight-line km to district HQ, n={okc.sum()}): median {CIRC:.2f}, "
      f"IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}")

CF = pd.read_csv(RAW / "spatial" / "contemporary_facilities.csv")


def access_to(sub):
    d = haversine_km(LAT[:, None], LON[:, None], sub.lat.values[None, :], sub.lon.values[None, :]) * CIRC
    for j, t in enumerate(sub.census_town.values):
        if isinstance(t, str):
            d[(TC.town == t).values, j] = 0.0
    return d.min(axis=1)


HOSP_NOW = CF[(CF.facility_type == "health") & (CF.functional_beds.fillna(999) >= 30)]   # GMC, DH, SDH, CHC
INPT_NOW = CF[CF.facility_type == "health"]                                            # + bedded PHCs
FIRE_NOW = CF[CF.facility_type == "fire"]
NOW = pd.DataFrame({"census_code": F.census_code, "town": F.town, "district": F.district,
                    "hosp_2011": F.acc_hospital_km.values, "hosp_now": access_to(HOSP_NOW),
                    "inpatient_now": access_to(INPT_NOW), "fire_2011": F.acc_fire_km.values, "fire_now": access_to(FIRE_NOW)})
print(f"Public hospitals >=30 beds now: {len(HOSP_NOW)}; bedded PHCs: {len(INPT_NOW) - len(HOSP_NOW)}; fire stations: {len(FIRE_NOW)}")
print(f"Median km to hospital: 2011 census {NOW.hosp_2011.median():.1f} | now (public, >=30 beds) {NOW.hosp_now.median():.1f} | "
      f"now (any public inpatient) {NOW.inpatient_now.median():.1f}")
print(f"Median km to fire station: 2011 {NOW.fire_2011.median():.1f} | now {NOW.fire_now.median():.1f}; "
      f"towns with a fire station within 5 km: 2011 {(NOW.fire_2011 <= 5).sum()} | now {(NOW.fire_now <= 5).sum()}")

F_NOW = F.copy()
F_NOW["acc_hospital_km"] = NOW.hosp_now.values
F_NOW["acc_fire_km"] = NOW.fire_now.values
IS_NOW = aggregate(pillar_table(score_indicators(F_NOW, BOUNDS)), BASE_W).values
NOW["is_2011"], NOW["is_partial_update"] = IS0, IS_NOW
NOW["rank_2011"] = pd.Series(IS0).rank(ascending=False).values
NOW["rank_update"] = pd.Series(IS_NOW).rank(ascending=False).values
RHO_UPDATE = spearmanr(IS0, IS_NOW).statistic
_r = spearmanr(IS0, NOW.inpatient_now); RHO_HOSP_NOW, P_HOSP_NOW = _r.statistic, _r.pvalue
print(f"Partially updated index vs 2011 index: Spearman {RHO_UPDATE:.3f}; 2011 InfraScore vs current km to any public "
      f"inpatient facility: Spearman {RHO_HOSP_NOW:.2f} (p = {P_HOSP_NOW:.3f})")

# re-run the peer-gap classification on the partially updated index (same model, same features, same protocol)
est, grid = MODELS[BEST]
y_now = pd.Series(IS_NOW)
oof_now = np.mean([cross_val_predict(tuned(est, grid), X, y_now, cv=KFold(5, shuffle=True, random_state=s)) for s in range(5)], axis=0)
gap_now = IS_NOW - oof_now
q90_now = np.quantile(np.abs(gap_now), 0.90)
NOW["gap_update"] = gap_now
NOW["flag_update"] = np.where(gap_now < -q90_now, "under-served", np.where(gap_now > q90_now, "above", "within"))
UNDER_2011 = GAP[GAP.flag == "Significantly under-served"].town.tolist()
UNDER_NOW = NOW[NOW.flag_update == "under-served"].town.tolist()
print(f"Under-served on 2011 data: {UNDER_2011}; on the partially updated index: {UNDER_NOW} (band +/-{q90_now:.1f})")
display(NOW[NOW.town.isin(set(UNDER_2011) | set(UNDER_NOW) | {"Usgao", "Sanquelim"})]
        [["town", "hosp_2011", "hosp_now", "inpatient_now", "fire_2011", "fire_now", "rank_2011", "rank_update", "flag_update"]].round(1))


# In[23]:


GOA = json.load(open(RAW / "spatial" / "goa_districts_datta07.geojson"))
LAT0 = np.cos(np.radians(15.4))


def draw_goa(ax):
    for ft in GOA["features"]:
        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            ring = np.array(poly[0])
            ax.fill(ring[:, 0], ring[:, 1], color="#f3f4f6", ec="#9aa1a9", lw=0.5, zorder=0)
    ax.set_aspect(1 / LAT0)
    ax.set_xlim(73.66, 74.36); ax.set_ylim(14.88, 15.82)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


gap_by_code = GAP.set_index("census_code")
FLAG_UNDER = TC.census_code.map(gap_by_code.flag).eq("Significantly under-served").values
fig, axes = plt.subplots(1, 3, figsize=(COL2, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
ax = axes[0]
draw_goa(ax)
sc = ax.scatter(LON, LAT, c=IS0, cmap="RdYlGn", vmin=20, vmax=75, s=6 + F.population / 2500, ec="#333333", lw=0.3, zorder=3)
ax.scatter(LON[FLAG_UNDER], LAT[FLAG_UNDER], s=70, facecolors="none", ec="#b2182b", lw=1.2, zorder=4)
ax.scatter([orgao.lon], [orgao.lat], marker="x", s=18, color="#555555", lw=0.8, zorder=4)


def put(ax, x, y, text, dx, dy, color="#222222", size=6.2):
    """Label a point at a fixed offset (points) with a thin leader line and a white halo, so labels never sit on markers."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=size, color=color,
                ha="right" if dx < 0 else "left", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=2))


MAP_OFFSETS = {"Mapusa": (-16, 6), "Panaji": (-18, -2), "Sancoale": (-18, -12), "Margao": (-22, -8),
               "Quepem": (16, 8), "Cuncolim": (16, -10)}
for name, (dx, dy) in MAP_OFFSETS.items():
    k = F.index[F.town == name][0]
    put(ax, LON[k], LAT[k], name, dx, dy, color="#b2182b" if FLAG_UNDER[k] else "#222222")
put(ax, orgao.lon, orgao.lat, "Orgao (no record)", 14, 16, color="#555555", size=5.8)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6)
cb.set_label("InfraScore (2011)", fontsize=6); cb.outline.set_visible(False)
ax.set_title("(a) 2011 InfraScore (ring = flagged)", fontsize=7)

ax = axes[1]
draw_goa(ax)
sc2 = ax.scatter(LON, LAT, c=NOW.inpatient_now, cmap="YlOrRd", vmin=0, vmax=15, s=10, ec="#333333", lw=0.3, zorder=3)
h = HOSP_NOW
ax.scatter(h.lon, h.lat, marker="P", s=26, color="#1f3b73", ec="white", lw=0.4, zorder=5, label="Hospital / CHC, 30+ beds")
p = INPT_NOW[INPT_NOW.functional_beds.fillna(999) < 30]
ax.scatter(p.lon, p.lat, marker="+", s=16, color="#1f3b73", lw=0.7, zorder=5, label="Bedded PHC")
ax.scatter(FIRE_NOW.lon, FIRE_NOW.lat, marker="^", s=14, color="#c8553d", ec="white", lw=0.3, zorder=5, label="Fire station")
cb2 = fig.colorbar(sc2, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02); cb2.ax.tick_params(labelsize=6)
cb2.set_label("km to public inpatient care (now)", fontsize=6); cb2.outline.set_visible(False)
ax.legend(loc="lower left", fontsize=5.2, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.3, markerscale=0.8,
          frameon=True, framealpha=0.95, facecolor="white", edgecolor="#d0d4d9", borderpad=0.35)
ax.set_title("(b) Contemporary public facilities", fontsize=7)

ax = axes[2]
FLAG_COL2 = {"Significantly under-served": "#c8553d", "Significantly over-performing": "#2a9d8f", "Within expected range": GREY}
lo_, hi_ = 18, 77
xx = np.linspace(lo_, hi_, 10)
ax.fill_between(xx, xx - q90, xx + q90, color="#eef1f4", zorder=1)
ax.plot(xx, xx, "--", color="#999999", lw=0.8, zorder=2)
ax.scatter(GAP.expected, GAP.infrascore, c=GAP.flag.map(FLAG_COL2), s=12, ec="white", lw=0.3, zorder=3)
labg = GAP[GAP.flag != "Within expected range"]
GAP_OFFSETS = {"Panaji": (-12, 0), "Aquem": (-14, 6), "Chicalim": (-16, 4), "Verna": (-14, -2),
               "Quepem": (14, 12), "Cuncolim": (16, -2), "Sancoale": (14, -12)}
for _, r_ in labg.iterrows():
    dx, dy = GAP_OFFSETS.get(r_.town, (12, 0) if r_.flag == "Significantly under-served" else (-12, 0))
    put(ax, r_.expected, r_.infrascore, r_.town, dx, dy, color=FLAG_COL2[r_.flag])
ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Expected (out-of-fold)", fontsize=6.5); ax.set_ylabel("Actual InfraScore", fontsize=6.5)
ax.tick_params(labelsize=6)
ax.set_title(f"(c) Peer gaps, 90% band $\\pm${q90:.1f}", fontsize=7)
fig.tight_layout(w_pad=0.6)
savefig(fig, "fig11_map_contemporary_gaps.png")


# ### 11c. Does the 90% interval cover 90%? Spatial dependence of the gaps
# 
# The V1 band used the 90th percentile of all 69 out-of-fold residuals. Here the procedure is tested honestly: in each outer fold the calibration quantile is computed **only from training towns** (inner out-of-fold residuals), and coverage is measured on held-out towns. Two schemes: random 5-fold (10 repeats) and leave-one-taluka-out (a spatial block test). Moran's I tests whether the gaps are spatially clustered, which would violate the exchangeability that conformal intervals assume.

# In[24]:


from sklearn.model_selection import LeaveOneGroupOut


def conf_q(res, a):
    n = len(res)
    return np.quantile(res, min(1.0, np.ceil((n + 1) * a) / n), method="higher")


def coverage_runs(splitters, groups=None, alphas=(0.8, 0.9, 0.95)):
    rows = []
    est, grid = MODELS[BEST]
    for r, sp in enumerate(splitters):
        for tr, te in sp.split(X, yv, groups):
            inner_pred = cross_val_predict(tuned(est, grid), X.iloc[tr], yv.iloc[tr], cv=KFold(5, shuffle=True, random_state=r))
            res = np.abs(yv.iloc[tr].values - inner_pred)
            pred = tuned(est, grid).fit(X.iloc[tr], yv.iloc[tr]).predict(X.iloc[te])
            err = np.abs(yv.iloc[te].values - pred)
            grp_tr = D.is_statutory_town.iloc[tr].values
            for a in alphas:
                q_plain = np.quantile(res, a)                                    # V1 rule: plain empirical quantile
                q = conf_q(res, a)                                               # finite-sample (split-conformal) quantile
                q_grp = {g_: conf_q(res[grp_tr == g_], a) for g_ in (0, 1)}      # Mondrian (group-conditional)
                for k, idx in enumerate(te):
                    g_ = int(D.is_statutory_town.iloc[idx])
                    rows.append({"scheme_run": r, "alpha": a, "town": D.town.iloc[idx], "statutory": g_,
                                 "covered_plain": bool(err[k] <= q_plain), "half_width_plain": q_plain,
                                 "covered": bool(err[k] <= q), "half_width": q,
                                 "covered_mondrian": bool(err[k] <= q_grp[g_]), "half_width_mondrian": q_grp[g_]})
    return pd.DataFrame(rows)


t = time.time()
COV_RAND = coverage_runs([KFold(5, shuffle=True, random_state=100 + r) for r in range(10)])
COV_TAL = coverage_runs([LeaveOneGroupOut()], groups=F.taluka.values)
print(f"coverage tests: {time.time() - t:.0f}s")


def cov_summary(df, label):
    out = []
    for rule, col, hw in [("plain (V1)", "covered_plain", "half_width_plain"), ("conformal", "covered", "half_width"),
                          ("Mondrian", "covered_mondrian", "half_width_mondrian")]:
        g = df.groupby(["alpha", "scheme_run"])[col].mean().groupby("alpha")
        out.append(pd.DataFrame({"scheme": label, "rule": rule, "nominal": g.mean().index, "empirical": g.mean().values,
                                 "min_run": g.min().values, "max_run": g.max().values,
                                 "half_width": df.groupby("alpha")[hw].mean().values,
                                 "cov_statutory": df[df.statutory == 1].groupby("alpha")[col].mean().values,
                                 "cov_census_town": df[df.statutory == 0].groupby("alpha")[col].mean().values}))
    return pd.concat(out, ignore_index=True)


COVERAGE_TEST = pd.concat([cov_summary(COV_RAND, "random 5-fold x10"), cov_summary(COV_TAL, "leave-one-taluka-out")], ignore_index=True)
display(COVERAGE_TEST.round(3))
CT90 = COVERAGE_TEST[COVERAGE_TEST.nominal == 0.9].set_index(["scheme", "rule"])

# Full-sample bands: plain (V1), finite-sample conformal, and group-conditional (Mondrian). Are the flags robust?
GAP["statutory"] = (GAP.civic_status != "Census Town").astype(int)
Q_CORR = conf_q(np.abs(GAP.gap.values), 0.90)
UNDER_CORR = GAP[GAP.gap < -Q_CORR].town.tolist()
print(f"Finite-sample conformal 90% half-width {Q_CORR:.1f} (V1 plain {q90:.1f}); under-served: {UNDER_CORR}")
Q_MOND = {g_: conf_q(np.abs(GAP.gap[GAP.statutory == g_].values), 0.90) for g_ in (0, 1)}
GAP["q_mondrian"] = GAP.statutory.map(Q_MOND)
GAP["flag_mondrian"] = np.select([GAP.gap < -GAP.q_mondrian, GAP.gap > GAP.q_mondrian], ["under-served", "above"], "within")
UNDER_MOND = GAP[GAP.flag_mondrian == "under-served"].town.tolist()
print(f"Mondrian 90% half-widths: statutory {Q_MOND[1]:.1f}, census towns {Q_MOND[0]:.1f}; under-served under Mondrian bands: {UNDER_MOND}")
display(GAP[GAP.town.isin(["Cuncolim", "Quepem", "Sancoale"])][["town", "civic_status", "gap", "q_mondrian", "flag_mondrian"]].round(1))


def morans_i(z, dmax=15.0, perms=9999, seed=SEED):
    z = np.asarray(z, float); z = z - z.mean()
    W = np.where((DMAT > 0) & (DMAT <= dmax), 1.0 / np.maximum(DMAT, 1e-9), 0.0)
    n, S0 = len(z), W.sum()
    stat = lambda v: n / S0 * (v @ W @ v) / (v @ v)
    I = stat(z)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(z)) for _ in range(perms)])
    return I, (np.sum(null >= I) + 1) / (perms + 1), -1 / (n - 1)


gap_aligned = F.census_code.map(gap_by_code.gap).values
MORAN = pd.DataFrame([("InfraScore", *morans_i(IS0)), ("Peer gap (residual)", *morans_i(gap_aligned))],
                     columns=["variable", "morans_I", "p_perm", "expected_I"])
display(MORAN.round(3))


# ### 11d. Spillover-aware allocation with sourced capital and operating costs
# 
# **Spillovers.** A new hospital, fire station, library, hall, college or stadium in town *i* also shortens the access distance of every other town *j* to `min(current_j, circuity x straight-line(i, j))`. Per-capita indicators change only in the receiving town. Because InfraScore is linear in the indicator scores, the gain of every candidate action for every town is computed in closed form.
# 
# **Costs.** Capital costs are recent Goa project figures; operating costs come from a published costing study (health, uprated with the CPI), the Goa budget (fire stations: DFES revenue budget per station; colleges: the government-college budget head per college), a Sports Authority of Goa statement (stadium) and the Art & Culture performance budget (village libraries); only the community-hall figure is an assumption. Lifecycle cost = capital + 10 years of operating cost discounted at 8%. Bank branches are excluded because they are private investments. A 200-draw Monte Carlo perturbs every capital and operating cost independently (log-normal, sigma = 0.4).

# In[25]:


COSTS = pd.read_csv(RAW / "costs" / "facility_costs.csv").set_index("facility")
DISC, YEARS = 0.08, 10
ANNUITY = (1 - (1 + DISC) ** -YEARS) / DISC
COSTS["lifecycle_cr"] = COSTS.capex_cr + COSTS.opex_cr_per_yr * ANNUITY
COSTS["capex_units_vs_PHC"] = COSTS.capex_cr / COSTS.loc["Primary health centre", "capex_cr"]
COSTS["illustrative_units_v1"] = COSTS.index.map(BASE_COST)
display(COSTS[["capex_cr", "opex_cr_per_yr", "lifecycle_cr", "capex_units_vs_PHC", "illustrative_units_v1"]].round(2))
DFES_CAPEX_2025 = 37.35
print(f"Annuity factor {ANNUITY:.3f}. One fire station = {COSTS.loc['Fire station', 'capex_cr'] / DFES_CAPEX_2025:.0%} of the "
      f"DFES 2025-26 capital budget (Rs {DFES_CAPEX_2025} cr); its 10-year operating cost (PV) is "
      f"{COSTS.loc['Fire station', 'opex_cr_per_yr'] * ANNUITY / COSTS.loc['Fire station', 'capex_cr']:.1f}x its capital cost")

IND_NAMES = list(INDICATORS)
IDX = {c: k for k, c in enumerate(IND_NAMES)}
N_IN_PILLAR = {p: sum(1 for c in IND_NAMES if INDICATORS[c][0] == p) for p in PILLARS}
WVEC = np.array([W_NORM[INDICATORS[c][0]] / N_IN_PILLAR[INDICATORS[c][0]] for c in IND_NAMES])
LO = {c: BOUNDS[c][0] for c in IND_NAMES if INDICATORS[c][1] == "density"}
HI = {c: BOUNDS[c][1] for c in IND_NAMES if INDICATORS[c][1] == "density"}


def ind_score(c, v):
    kind = INDICATORS[c][1]
    if kind == "access":
        return 100 * np.exp(-v / 10.0)
    if kind == "ratio":
        return 100 * v
    return 100 * (np.clip(v, LO[c], HI[c]) - LO[c]) / (HI[c] - LO[c])


def is_from_state(S):
    return sum(WVEC[IDX[c]] * ind_score(c, S[:, IDX[c]]) for c in IND_NAMES)


S0_STATE = F[IND_NAMES].values.astype(float)
assert np.allclose(is_from_state(S0_STATE), IS0), "closed-form scorer must reproduce the index"
ACCESS_OF = {"Hospital": "acc_hospital_km", "Fire station": "acc_fire_km", "Public library": "acc_library_km",
             "Auditorium / community hall": "acc_auditorium_km", "Degree college": "acc_degree_college_km", "Stadium": "acc_stadium_km"}
DENSITY_OF = {"Primary health centre": "primary_care_per_10k", "Hospital": "hospitals_per_10k", "Bank branch": "banks_per_10k"}
CAND_ARR = {a: np.isin(F.census_code.values, CAND[CAND.facility == a].census_code.values) for a in CAND.facility.unique()}
REACH = CIRC * DMAT
np.fill_diagonal(REACH, 0.0)
POPV = F.population.values.astype(float)


def feasible_mask(a, S):
    if a in ("Public library", "Auditorium / community hall", "Degree college", "Stadium"):
        return CAND_ARR[a] & (S[:, IDX[ACCESS_OF[a]]] > 0)
    if a in ("Hospital", "Fire station"):
        return S[:, IDX[ACCESS_OF[a]]] > 0
    if a == "Bank branch":
        return bank_ok.copy()
    return np.ones(len(F), bool)


NO_SPILL = set()   # facility types whose spillovers are switched off (sensitivity analysis)


def action_deltas(a, S, spill):
    """Matrix (site i x town j) of score gains if action a is placed at site i."""
    spill = spill and a not in NO_SPILL
    n = len(F)
    dmat = np.zeros((n, n))
    if a in ACCESS_OF:
        c = ACCESS_OF[a]; k = IDX[c]; cur = S[:, k]
        new = np.minimum(cur[None, :], REACH) if spill else np.where(np.eye(n, dtype=bool), 0.0, cur[None, :])
        dmat += WVEC[k] * (ind_score(c, new) - ind_score(c, cur)[None, :])
    if a in DENSITY_OF:
        c = DENSITY_OF[a]; k = IDX[c]; cur = S[:, k]
        dmat[np.arange(n), np.arange(n)] += WVEC[k] * (ind_score(c, cur + 1e4 / POPV) - ind_score(c, cur))
    return dmat


def apply_action(a, i, S, spill):
    spill = spill and a not in NO_SPILL
    S = S.copy()
    if a in ACCESS_OF:
        k = IDX[ACCESS_OF[a]]
        S[:, k] = np.minimum(S[:, k], REACH[i]) if spill else S[:, k]
        S[i, k] = 0.0
    if a in DENSITY_OF:
        k = IDX[DENSITY_OF[a]]
        S[i, k] += 1e4 / POPV[i]
    return S


def allocate(strategy, actions, cost=None, budget=None, K=None, spill=True, seed=0, start=None):
    """Greedy / heuristic allocation. Either K (count budget) or cost+budget (Rs crore).
    start: optional initial indicator state (towns x indicators); default = 2011 census state."""
    S = (S0_STATE if start is None else start).copy()
    cur, spent, plan = is_from_state(S), 0.0, []
    rng = np.random.default_rng(seed)
    while True:
        if K is not None and len(plan) >= K:
            break
        wts = POPV * (100 - cur) / (100 - MEAN_IS)
        best, cands = None, []
        for a in actions:
            if cost is not None and cost[a] > budget - spent + 1e-9:
                continue
            m = feasible_mask(a, S)
            if not m.any():
                continue
            dm = action_deltas(a, S, spill)
            ben = dm @ wts
            own = np.diag(dm)
            for i in np.where(m & (ben > 1e-9))[0]:
                cands.append((a, i, ben[i], own[i]))
        if not cands:
            break
        if strategy == "greedy_cost":
            a, i, b, _ = max(cands, key=lambda z: z[2] / cost[z[0]])
        elif strategy == "greedy":
            a, i, b, _ = max(cands, key=lambda z: z[2])
        elif strategy == "lowest_first":
            j = min({z[1] for z in cands}, key=lambda t: cur[t])
            a, i, b, _ = max([z for z in cands if z[1] == j], key=lambda z: z[2])
        elif strategy == "largest_first":
            j = max({z[1] for z in cands}, key=lambda t: POPV[t])
            a, i, b, _ = max([z for z in cands if z[1] == j], key=lambda z: z[2])
        else:
            a, i, b, _ = cands[rng.integers(len(cands))]
        S = apply_action(a, i, S, spill)
        new = is_from_state(S)
        plan.append({"step": len(plan) + 1, "town": F.town[i], "action": a, "cost_cr": (cost[a] if cost is not None else np.nan),
                     "own_gain": new[i] - cur[i], "towns_improved": int((new - cur > 1e-9).sum())})
        spent += cost[a] if cost is not None else 0
        cur = new
    return cur, pd.DataFrame(plan)


# sanity check: without spillover, the closed-form allocator reproduces the V1 ten-step plan exactly
_, chk = allocate("greedy", ACTIONS, K=10, spill=False)
v1 = PLANS[("Greedy optimiser", 10)]
assert (chk.town.values == v1.town.values).all() and (chk.action.values == v1.action.values).all()
print("V1 ten-step plan reproduced by the closed-form allocator.")


# In[26]:


# (i) Ten-intervention experiment (count budget), with and without spillovers
SPILL_ROWS, SPILL_PLANS = [], {}
for spill in (False, True):
    for strat, lab_ in [("greedy", "Greedy optimiser"), ("lowest_first", "Lowest score first"), ("largest_first", "Largest towns first")]:
        cur, plan = allocate(strat, ACTIONS, K=10, spill=spill)
        SPILL_ROWS.append({"spillover": spill, "strategy": lab_, **metrics(cur)}); SPILL_PLANS[(lab_, spill)] = plan
    rnd = pd.DataFrame([metrics(allocate("random", ACTIONS, K=10, spill=spill, seed=s)[0]) for s in range(200)]).mean()
    SPILL_ROWS.append({"spillover": spill, "strategy": "Random (mean of 200)", **rnd.to_dict()})
K10 = pd.DataFrame(SPILL_ROWS)
display(K10.round(3))
p_off, p_on = SPILL_PLANS[("Greedy optimiser", False)], SPILL_PLANS[("Greedy optimiser", True)]
SHARED_TOWNS = sorted(set(p_off.town) & set(p_on.town))
print(f"Greedy K=10 with spillovers: towns {sorted(set(p_on.town))}; shared with no-spillover plan: {SHARED_TOWNS}")
display(p_on)
K10i = K10.set_index(["spillover", "strategy"]).pop_weighted_gain
RATIO_NOSPILL = K10i[(False, "Greedy optimiser")] / K10i[(False, "Random (mean of 200)")]
RATIO_SPILL = K10i[(True, "Greedy optimiser")] / K10i[(True, "Random (mean of 200)")]
print(f"Greedy / random: {RATIO_NOSPILL:.2f} without spillover, {RATIO_SPILL:.2f} with spillover")
# Why does Quepem leave the spillover-aware plan? Its hospital distance after the Curchorem-Cacora hospital:
iq, ic = F.index[F.town == "Quepem"][0], F.index[F.town == "Curchorem-Cacora"][0]
print(f"Quepem km to hospital: 2011 {F.acc_hospital_km[iq]:.0f}; after a hospital in Curchorem-Cacora (spillover) "
      f"{min(F.acc_hospital_km[iq], REACH[ic, iq]):.1f}")


# In[27]:


# (ii) Realistic lifecycle-cost budget (public facilities only), spillovers on
PUBLIC_ACTIONS = [a for a in ACTIONS if a != "Bank branch"]
LC = COSTS.lifecycle_cr.to_dict()
BUDGETS = (100, 250, 500)
BUD_ROWS, BUD_PLANS = [], {}
for B in BUDGETS:
    for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)"), ("lowest_first", "Lowest score first")]:
        cur, plan = allocate(strat, PUBLIC_ACTIONS, cost=LC, budget=B, spill=True)
        BUD_ROWS.append({"budget_cr": B, "strategy": lab_, "n_facilities": len(plan), "spent_cr": plan.cost_cr.sum(), **metrics(cur)})
        BUD_PLANS[(lab_, B)] = plan
    rr = [allocate("random", PUBLIC_ACTIONS, cost=LC, budget=B, spill=True, seed=s) for s in range(30)]
    BUD_ROWS.append({"budget_cr": B, "strategy": "Random (mean of 30)", "n_facilities": np.mean([len(p) for _, p in rr]),
                     "spent_cr": np.mean([p.cost_cr.sum() for _, p in rr]), **pd.DataFrame([metrics(c) for c, _ in rr]).mean().to_dict()})
BUDGET_TAB = pd.DataFrame(BUD_ROWS)
display(BUDGET_TAB.round(3))
PLAN250 = BUD_PLANS[("Cost-aware greedy", 250)]
COMPOSITION = PLAN250.groupby("action").agg(n=("town", "size"), spend_cr=("cost_cr", "sum")).sort_values("spend_cr", ascending=False)
display(COMPOSITION.round(1))

# Essential-services variant: the same Rs 250 crore restricted to health and fire facilities
ESSENTIAL = ["Primary health centre", "Hospital", "Fire station"]
ess_rows = {}
for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)"), ("lowest_first", "Lowest score first")]:
    cur, plan = allocate(strat, ESSENTIAL, cost=LC, budget=250, spill=True)
    ess_rows[lab_] = {"n_facilities": len(plan), "spent_cr": plan.cost_cr.sum(), **metrics(cur)}; BUD_PLANS[(lab_ + " (essential)", 250)] = plan
rr = [allocate("random", ESSENTIAL, cost=LC, budget=250, spill=True, seed=s) for s in range(30)]
ess_rows["Random (mean of 30)"] = {"n_facilities": np.mean([len(p) for _, p in rr]), "spent_cr": np.mean([p.cost_cr.sum() for _, p in rr]),
                                   **pd.DataFrame([metrics(c) for c, _ in rr]).mean().to_dict()}
ESSENTIAL_TAB = pd.DataFrame(ess_rows).T
display(ESSENTIAL_TAB.round(3))
PLAN250_ESS = BUD_PLANS[("Cost-aware greedy (essential)", 250)]
display(Markdown("**Essential-services plan (health and fire only), Rs 250 crore**")); display(PLAN250_ESS.round(2))

# The same essential-services budget, starting from the contemporary network (Section 11b) rather than 2011
S_NOW_STATE = F_NOW[IND_NAMES].values.astype(float)
cur_now_end, PLAN_NOW_ESS = allocate("greedy_cost", ESSENTIAL, cost=LC, budget=250, spill=True, start=S_NOW_STATE)
GAIN_NOW_ESS = np.average(cur_now_end - IS_NOW, weights=POPV)
display(Markdown("**Essential-services plan from the contemporary network, Rs 250 crore**")); display(PLAN_NOW_ESS.round(2))
print(f"Population-weighted gain from the contemporary start: {GAIN_NOW_ESS:.2f} points")
display(Markdown("**Cost-aware plan, Rs 250 crore lifecycle budget, with spillovers**")); display(PLAN250.round(2))

rng = np.random.default_rng(SEED)
mc_rows = []
t = time.time()
for d in range(200):
    f_cap = {a: float(np.exp(rng.normal(0, 0.4))) for a in PUBLIC_ACTIONS}
    f_op = {a: float(np.exp(rng.normal(0, 0.4))) for a in PUBLIC_ACTIONS}
    lc = {a: COSTS.loc[a, "capex_cr"] * f_cap[a] + COSTS.loc[a, "opex_cr_per_yr"] * f_op[a] * ANNUITY for a in PUBLIC_ACTIONS}
    row = {}
    for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)"), ("lowest_first", "Lowest score first"), ("random", "Random")]:
        row[lab_] = metrics(allocate(strat, PUBLIC_ACTIONS, cost=lc, budget=250, spill=True, seed=d)[0])["pop_weighted_gain"]
    mc_rows.append(row)
MC_REAL = pd.DataFrame(mc_rows)
WIN_REAL = {k: float((MC_REAL["Cost-aware greedy"] > MC_REAL[k]).mean()) for k in MC_REAL.columns if k != "Cost-aware greedy"}
print(f"Monte Carlo ({time.time() - t:.0f}s): cost-aware greedy beats", {k: f"{v:.0%}" for k, v in WIN_REAL.items()},
      "| mean gains", MC_REAL.mean().round(3).to_dict())


# #### Does the cost result depend on the weakest spillovers?
# 
# The census supports the spillover rule well for hospitals and fire stations but poorly for libraries and halls (Spearman 0.42 and 0.27), and halls and libraries make up most of the cost-aware plan. We therefore rerun Experiment B with hall and library spillovers switched off (each benefits only its own town), including the 200-draw cost Monte Carlo, and, separately, with the one assumed operating cost (community hall) raised tenfold.

# In[28]:


NO_SPILL = {"Auditorium / community hall", "Public library"}
try:
    ns = {}
    for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)"), ("lowest_first", "Lowest score first")]:
        cur, plan = allocate(strat, PUBLIC_ACTIONS, cost=LC, budget=250, spill=True)
        ns[lab_] = {"n_facilities": len(plan), "halls_libraries": int(plan.action.isin(NO_SPILL).sum()), **metrics(cur)}
        if strat == "greedy_cost":
            PLAN250_NOSPILL = plan
    rr = [allocate("random", PUBLIC_ACTIONS, cost=LC, budget=250, spill=True, seed=s) for s in range(30)]
    ns["Random (mean of 30)"] = {"n_facilities": np.mean([len(p) for _, p in rr]), **pd.DataFrame([metrics(c) for c, _ in rr]).mean().to_dict()}
    rng_ns = np.random.default_rng(SEED)
    mc_ns = []
    for d in range(200):
        f_cap = {a: float(np.exp(rng_ns.normal(0, 0.4))) for a in PUBLIC_ACTIONS}
        f_op = {a: float(np.exp(rng_ns.normal(0, 0.4))) for a in PUBLIC_ACTIONS}
        lc = {a: COSTS.loc[a, "capex_cr"] * f_cap[a] + COSTS.loc[a, "opex_cr_per_yr"] * f_op[a] * ANNUITY for a in PUBLIC_ACTIONS}
        mc_ns.append({lab_: metrics(allocate(strat, PUBLIC_ACTIONS, cost=lc, budget=250, spill=True, seed=d)[0])["pop_weighted_gain"]
                      for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)"),
                                          ("lowest_first", "Lowest score first"), ("random", "Random")]})
finally:
    NO_SPILL = set()
NOSPILL_TAB = pd.DataFrame(ns).T
MC_NS = pd.DataFrame(mc_ns)
WIN_NS = {k: float((MC_NS["Cost-aware greedy"] > MC_NS[k]).mean()) for k in MC_NS.columns if k != "Cost-aware greedy"}
display(NOSPILL_TAB.round(3)); display(PLAN250_NOSPILL.round(2))
print("Hall and library spillovers off: cost-aware beats", {k: f"{v:.0%}" for k, v in WIN_NS.items()})

lc_hall = dict(LC); lc_hall["Auditorium / community hall"] = COSTS.loc["Auditorium / community hall", "capex_cr"] + 10 * COSTS.loc["Auditorium / community hall", "opex_cr_per_yr"] * ANNUITY
HALL10 = {lab_: metrics(allocate(strat, PUBLIC_ACTIONS, cost=lc_hall, budget=250, spill=True)[0])["pop_weighted_gain"]
          for strat, lab_ in [("greedy_cost", "Cost-aware greedy"), ("greedy", "Greedy (ignores cost)")]}
print(f"Hall operating cost x10 (lifecycle Rs {lc_hall['Auditorium / community hall']:.1f} cr): cost-aware {HALL10['Cost-aware greedy']:.2f} vs cost-blind {HALL10['Greedy (ignores cost)']:.2f}")
NOSPILL_TAB.round(4).to_csv(OUT / "allocation_no_hall_library_spillover.csv")
pd.DataFrame([{"test": "hall opex x10", **HALL10}, {"test": "no hall/library spillover, MC win share", **WIN_NS}]).round(4).to_csv(OUT / "cost_sensitivity_extra.csv", index=False)


# In[29]:


COVERAGE.to_csv(OUT / "coverage_audit.csv")
JACK.round(4).to_csv(OUT / "leave_one_town_out.csv", index=False)
NOW.round(3).to_csv(OUT / "contemporary_access.csv", index=False)
COVERAGE_TEST.round(4).to_csv(OUT / "interval_coverage_test.csv", index=False)
MORAN.round(4).to_csv(OUT / "morans_i.csv", index=False)
COSTS.round(4).to_csv(OUT / "facility_costs_lifecycle.csv")
K10.round(4).to_csv(OUT / "allocation_k10_spillover.csv", index=False)
BUDGET_TAB.round(4).to_csv(OUT / "allocation_realistic_budget.csv", index=False)
PLAN250.round(3).to_csv(OUT / "plan_rs250cr_cost_aware.csv", index=False)
PLAN250_ESS.round(3).to_csv(OUT / "plan_rs250cr_essential_services.csv", index=False)
PLAN_NOW_ESS.round(3).to_csv(OUT / "plan_rs250cr_essential_from_contemporary.csv", index=False)
ESSENTIAL_TAB.round(4).to_csv(OUT / "allocation_essential_services.csv")
GAP[["town", "civic_status", "gap", "q_mondrian", "flag_mondrian"]].round(3).to_csv(OUT / "gap_mondrian_flags.csv", index=False)
MC_REAL.round(4).to_csv(OUT / "realistic_cost_scenarios.csv", index=False)


# ### 11e. Validating the spillover distance model, interpretable benefits, and sensitivity
# 
# **Validation.** The spillover rule assumes that a town without a facility reaches the nearest town that has one at about `1.40 x straight-line` km. The census independently reports each town's distance to its nearest facility, so the rule can be tested directly. Because the nearest facility may sit in a village (outside the 69-town set), the model distance should behave as an upper bound.
# 
# **Interpretable benefits.** Index points are translated into residents brought within 5 km of a hospital or fire station and into the population-weighted reduction in travel distance.
# 
# **Sensitivity.** The spillover and cost results are recomputed with the circuity factor at its interquartile bounds and with alternative discount rates and horizons.

# In[30]:


VAL_ROWS = []
for c in ["acc_hospital_km", "acc_fire_km", "acc_degree_college_km", "acc_library_km", "acc_auditorium_km", "acc_stadium_km", "acc_cinema_km"]:
    has = F[c].values == 0
    if has.sum() < 3:
        continue
    obs = F[c].values[~has]
    pred = (CIRC * DMAT[~has][:, has]).min(axis=1)
    VAL_ROWS.append({"indicator": c, "towns_with": int(has.sum()), "towns_without": int((~has).sum()),
                     "spearman": spearmanr(pred, obs).statistic, "median_pred_km": np.median(pred), "median_census_km": np.median(obs),
                     "share_pred_ge_census": float(np.mean(pred >= obs - 0.5)), "median_abs_err_km": float(np.median(np.abs(pred - obs)))})
SPILL_VAL = pd.DataFrame(VAL_ROWS)
display(SPILL_VAL.round(3))
print(f"Spillover distance model vs census nearest-facility distance: median Spearman {SPILL_VAL.spearman.median():.2f} "
      f"(range {SPILL_VAL.spearman.min():.2f}-{SPILL_VAL.spearman.max():.2f}); model >= census in "
      f"{np.average(SPILL_VAL.share_pred_ge_census, weights=SPILL_VAL.towns_without):.0%} of town-facility pairs")


def replay(plan, spill=True, start=None):
    S = (S0_STATE if start is None else start).copy()
    for _, row in plan.iterrows():
        S = apply_action(row.action, int(F.index[F.town == row.town][0]), S, spill)
    return S


def access_summary(S_before, S_after):
    out = {}
    for lab, c in [("hospital", "acc_hospital_km"), ("fire", "acc_fire_km")]:
        k = IDX[c]
        out[f"{lab}_residents_newly_within_5km"] = int(POPV[(S_after[:, k] <= 5) & (S_before[:, k] > 5)].sum())
        out[f"{lab}_popwtd_km_before"] = np.average(S_before[:, k], weights=POPV)
        out[f"{lab}_popwtd_km_after"] = np.average(S_after[:, k], weights=POPV)
    return out


PLAN_SET = {
    "A: greedy, no spillover": (SPILL_PLANS[("Greedy optimiser", False)], False),
    "A: greedy, spillover": (SPILL_PLANS[("Greedy optimiser", True)], True),
    "A: lowest-first, spillover": (SPILL_PLANS[("Lowest score first", True)], True),
    "B: cost-aware Rs 250 cr": (PLAN250, True),
    "B: essential Rs 250 cr": (PLAN250_ESS, True),
}
BENEFIT = pd.DataFrame({k: access_summary(S0_STATE, replay(p, sp)) for k, (p, sp) in PLAN_SET.items()}).T
display(BENEFIT.round(2))

# --- sensitivity: circuity at its IQR bounds; discount rate and horizon for lifecycle costs
REACH_BASE = REACH.copy()
SENS_ROWS = []
base_A = set(SPILL_PLANS[("Greedy optimiser", True)].town)
base_B = set(PLAN250_ESS.town + "|" + PLAN250_ESS.action)
jac = lambda a, b: len(a & b) / len(a | b)
for circ in (1.26, CIRC, 1.60):
    REACH = circ * DMAT; np.fill_diagonal(REACH, 0.0)
    curA, pA = allocate("greedy", ACTIONS, K=10, spill=True)
    rndA = np.mean([metrics(allocate("random", ACTIONS, K=10, spill=True, seed=s)[0])["pop_weighted_gain"] for s in range(50)])
    curB, pB = allocate("greedy_cost", PUBLIC_ACTIONS, cost=LC, budget=250, spill=True)
    curE, pE = allocate("greedy_cost", ESSENTIAL, cost=LC, budget=250, spill=True)
    SENS_ROWS.append({"setting": f"circuity {circ:.2f}", "A_gain": metrics(curA)["pop_weighted_gain"], "A_ratio_vs_random": metrics(curA)["pop_weighted_gain"] / rndA,
                      "A_jaccard_towns": jac(set(pA.town), base_A), "B_gain": metrics(curB)["pop_weighted_gain"],
                      "essential_jaccard": jac(set(pE.town + "|" + pE.action), base_B)})
REACH = REACH_BASE
for r_, yrs in [(0.05, 10), (0.10, 10), (0.08, 15)]:
    ann = (1 - (1 + r_) ** -yrs) / r_
    lc = (COSTS.capex_cr + COSTS.opex_cr_per_yr * ann).to_dict()
    curB, pB = allocate("greedy_cost", PUBLIC_ACTIONS, cost=lc, budget=250, spill=True)
    curE, pE = allocate("greedy_cost", ESSENTIAL, cost=lc, budget=250, spill=True)
    curBb, _ = allocate("greedy", PUBLIC_ACTIONS, cost=lc, budget=250, spill=True)
    SENS_ROWS.append({"setting": f"discount {r_:.0%}, {yrs} yr", "A_gain": np.nan, "A_ratio_vs_random": np.nan, "A_jaccard_towns": np.nan,
                      "B_gain": metrics(curB)["pop_weighted_gain"], "B_cost_blind_gain": metrics(curBb)["pop_weighted_gain"],
                      "essential_jaccard": jac(set(pE.town + "|" + pE.action), base_B)})
SENSITIVITY = pd.DataFrame(SENS_ROWS)
display(SENSITIVITY.round(3))
CORE = {"Sancoale|Fire station", "Curchorem-Cacora|Hospital"}
SPILL_VAL.round(4).to_csv(OUT / "spillover_distance_validation.csv", index=False)
BENEFIT.round(3).to_csv(OUT / "plan_benefits_interpretable.csv")
SENSITIVITY.round(4).to_csv(OUT / "sensitivity_circuity_discount.csv", index=False)


# #### Do the plans depend on the pillar weights?
# 
# The allocator maximises InfraScore, whose pillar weights are a judgement. Section 6 showed that the town *ranking* is stable under alternative weights; here we test the *plans*. We redraw the six pillar weights 400 times (200 flat Dirichlet(1) draws, which allow extreme weightings, and 200 draws centred on the baseline weights, Dirichlet(20 x baseline)), rebuild the index under each, and rerun the Experiment A plan (greedy, K = 10, spillover) and the Experiment B essential-services plan (cost-aware, Rs 250 crore). For each draw we record which towns and town-facility assignments are chosen.

# In[31]:


WVEC_BASE, MEAN_IS_BASE = WVEC.copy(), MEAN_IS
w_base = np.array([W_NORM[p] for p in PILLARS])
rng_w = np.random.default_rng(SEED + 7)
W_ALT = {"flat Dirichlet(1)": rng_w.dirichlet(np.ones(len(PILLARS)), 200),
         "centred Dirichlet(20 x base)": rng_w.dirichlet(20 * w_base, 200)}
pillar_of = np.array([INDICATORS[c][0] for c in IND_NAMES])
n_of = np.array([N_IN_PILLAR[p] for p in pillar_of])
base_A_towns = set(SPILL_PLANS[("Greedy optimiser", True)].town)
base_E_assign = set(PLAN250_ESS.town + "|" + PLAN250_ESS.action)
WROWS, FREQ_A, FREQ_E = [], {}, {}
try:
    for fam, draws in W_ALT.items():
        for w in draws:
            wmap = dict(zip(PILLARS, w))
            WVEC = np.array([wmap[p] for p in pillar_of]) / n_of
            MEAN_IS = is_from_state(S0_STATE).mean()
            _, pA = allocate("greedy", ACTIONS, K=10, spill=True)
            _, pE = allocate("greedy_cost", ESSENTIAL, cost=LC, budget=250, spill=True)
            tA, aE = set(pA.town), set(pE.town + "|" + pE.action)
            for t in tA: FREQ_A[(fam, t)] = FREQ_A.get((fam, t), 0) + 1
            for t in aE: FREQ_E[(fam, t)] = FREQ_E.get((fam, t), 0) + 1
            WROWS.append({"family": fam, "A_jaccard_towns": jac(tA, base_A_towns), "A_base_towns_kept": len(tA & base_A_towns) / len(base_A_towns),
                          "E_jaccard": jac(aE, base_E_assign), "E_core_both": CORE <= aE})
finally:
    WVEC, MEAN_IS = WVEC_BASE, MEAN_IS_BASE      # restore the baseline index
assert np.allclose(is_from_state(S0_STATE), IS0)
WROB = pd.DataFrame(WROWS)
WROB_SUM = WROB.groupby("family").agg(A_median_jaccard=("A_jaccard_towns", "median"), A_mean_share_base_towns=("A_base_towns_kept", "mean"),
                                      E_median_jaccard=("E_jaccard", "median"), E_share_core_kept=("E_core_both", "mean"))
display(WROB_SUM.round(3))
n_draw = {f: len(d) for f, d in W_ALT.items()}
freq = lambda D, fam: pd.Series({k[1]: v / n_draw[fam] for k, v in D.items() if k[0] == fam}).sort_values(ascending=False)
WFREQ_A = pd.DataFrame({fam: freq(FREQ_A, fam) for fam in W_ALT}).fillna(0).sort_values("flat Dirichlet(1)", ascending=False)
WFREQ_E = pd.DataFrame({fam: freq(FREQ_E, fam) for fam in W_ALT}).fillna(0).sort_values("flat Dirichlet(1)", ascending=False)
display(Markdown("**Share of weight draws in which each town is in the Experiment A plan**")); display(WFREQ_A.head(12).round(2))
display(Markdown("**Share of weight draws in which each essential-service assignment is chosen**")); display(WFREQ_E.head(10).round(2))
for fam in W_ALT:
    print(f"{fam}: base Experiment A towns chosen in >=50% of draws: "
          f"{', '.join(t for t in sorted(base_A_towns) if WFREQ_A.loc[t, fam] >= 0.5)}; "
          f"Sancoale fire {WFREQ_E.loc['Sancoale|Fire station', fam]:.0%}, Curchorem-Cacora hospital {WFREQ_E.loc['Curchorem-Cacora|Hospital', fam]:.0%}")
WROB_SUM.round(4).to_csv(OUT / "weight_robustness_plans.csv")
pd.concat({"A_towns": WFREQ_A, "E_assignments": WFREQ_E}).round(4).to_csv(OUT / "weight_robustness_selection_freq.csv")

# Figure 12: selection frequency under alternative weightings
topA = WFREQ_A.head(6).iloc[::-1]
topE = WFREQ_E.head(6).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.6), gridspec_kw={"wspace": 0.75})
for ax, tab, title in [(axes[0], topA, "(a) Experiment A: towns in the ten-step plan"),
                       (axes[1], topE, "(b) Health and fire plan: assignments")]:
    y = np.arange(len(tab))
    ax.barh(y + 0.19, tab["flat Dirichlet(1)"], height=0.36, color=NORTH, label="flat weights")
    ax.barh(y - 0.19, tab["centred Dirichlet(20 x base)"], height=0.36, color=SOUTH, label="centred weights")
    labels = [t.replace("|", ": ").replace("Fire station", "fire").replace("Hospital", "hospital").replace("Primary health centre", "PHC") for t in tab.index]
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_title(title, fontsize=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlabel("share of 200 weight draws", fontsize=7); axes[1].set_xlabel("share of 200 weight draws", fontsize=7)
axes[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(-0.45, -0.32), ncol=2)
savefig(fig, "fig12_weight_robustness.png")
plt.show()


# ## 12. Export results

# In[32]:


R.round(2).to_csv(OUT / "infrascore_rankings.csv", index=False)
GAP.round(2).to_csv(OUT / "gap_analysis.csv", index=False)
GAP_TAB.round(4).to_csv(OUT / "gap_model_comparison.csv", index=False)
ABL_TAB.round(4).to_csv(OUT / "gap_model_ablation.csv", index=False)
COEF.round(4).to_csv(OUT / "gap_model_drivers.csv", index=False)
CL_TAB.round(4).to_csv(OUT / "clustering_comparison.csv", index=False)
CL_PROF.to_csv(OUT / "cluster_profiles.csv")
UA_TAB.round(4).to_csv(OUT / "uncertainty_analysis.csv", index=False)
FAC_TAB.round(4).to_csv(OUT / "facility_model_metrics.csv", index=False)
CAND.round(3).to_csv(OUT / "facility_candidate_sites.csv", index=False)
SIM.round(4).to_csv(OUT / "policy_simulation.csv", index=False)
COST_TAB.round(4).to_csv(OUT / "policy_simulation_costed.csv", index=False)
MC.round(4).to_csv(OUT / "policy_cost_scenarios.csv", index=False)
PLANS[("Greedy optimiser", 10)].round(3).to_csv(OUT / "greedy_plan_K10.csv", index=False)
VAL.round(4).to_csv(OUT / "validation_literacy.csv", index=False)
BENCH.round(2).to_csv(OUT / "district_benchmark.csv")
F.round(4).to_csv(OUT / "features.csv", index=False)
print(f"Saved {len(list(OUT.glob('*.csv')))} tables and {len(list(FIG.glob('*.png')))} figures to {OUT}")
print(f"Total runtime: {time.time() - T0:.0f} s")


# ## 13. Summary of findings
# 
# The cell below prints the headline numbers used in the report, so every figure quoted in the paper can be traced to this run.

# In[33]:


g10 = PLANS[("Greedy optimiser", 10)]
s10 = SIM[SIM.budget_K == 10].set_index("strategy")
print(f"""
INFRASCORE   mean {R.infrascore.mean():.1f} (SD {R.infrascore.std():.1f}); top: {', '.join(R.town.head(3))}; bottom: {', '.join(R.town.tail(3))}
             statutory {R[R.civic_status != 'Census Town'].infrascore.mean():.1f} vs census towns {R[R.civic_status == 'Census Town'].infrascore.mean():.1f}
ROBUSTNESS   3,600 scenarios, median rho {UA_TAB.median_rho.median():.3f}, worst {UA_TAB.min_rho.min():.3f}; median 90% band {(R.rank_p95 - R.rank_p05).median():.0f} ranks
TYPOLOGIES   {best_method} k=4, silhouette {CL_TAB[(CL_TAB.method == best_method) & (CL_TAB.k == 4)].silhouette.iloc[0]:.2f}, bootstrap ARI {stab[best_method][0]:.2f}
GAP MODEL    {BEST}: MAE {best_mae:.2f} vs baseline {base_mae:.2f}, R2 {GAP_TAB.set_index('model').loc[BEST, 'R2']:.2f}; 90% band +/-{q90:.1f}
             significantly under-served: {', '.join(GAP[GAP.flag == 'Significantly under-served'].town) or 'none'}
             largest negative gaps: {', '.join(GAP.town.head(5))}
FACILITIES   {best_clf}: mean ROC-AUC {FAC_TAB[FAC_TAB.model == best_clf].ROC_AUC.mean():.2f}; {len(CAND)} candidate sites
POLICY K=10  pop-weighted gain: greedy {s10.loc['Greedy optimiser', 'pop_weighted_gain']:.2f} | largest-first {s10.loc['Largest towns first', 'pop_weighted_gain']:.2f} | lowest-first {s10.loc['Lowest score first', 'pop_weighted_gain']:.2f} | random {s10.loc['Random (mean)', 'pop_weighted_gain']:.2f}
COSTED       budget {BUDGET:g} units: cost-aware greedy {COST_TAB.set_index('strategy').loc['Cost-aware greedy', 'pop_weighted_gain']:.2f} | ignores-cost {COST_TAB.set_index('strategy').loc['Greedy (ignores cost)', 'pop_weighted_gain']:.2f} | lowest-first {COST_TAB.set_index('strategy').loc['Lowest score first', 'pop_weighted_gain']:.2f} | random {COST_TAB.set_index('strategy').loc['Random', 'pop_weighted_gain']:.2f}
             wins in 200 cost scenarios vs: {', '.join(f'{k} {v:.0%}' for k, v in win.items())}
EVIDENCE     plan assignments after 2011: exact {(CHK.match == 'exact').sum()} of {len(CHK)}, exact or related {CHK.corroborated.sum()} of {len(CHK)}; towns exact {len(exact_towns)} of {len(towns_plan)}, exact or related {len(any_towns)} of {len(towns_plan)}; random controls exact {ctrl_hits} of {len(eligible)} (Fisher p = {FISHER.pvalue:.3f}), exact or related {ctrl_any} of {len(eligible)} (p = {FISHER_ANY.pvalue:.3f})
COVERAGE     {analysed['Towns']} of {OFFICIAL['Towns']} towns ({analysed['Statutory towns']}/14 statutory, {analysed['Census towns']}/56 census towns); {analysed['Urban population']:,} of {OFFICIAL['Urban population']:,} urban residents ({analysed['Urban population'] / OFFICIAL['Urban population']:.2%}); Orgao {int(orgao.geonames_pop):,}
             leave-one-town-out: worst rank shift {JACK.max_rank_shift.max():.0f}, median {JACK.max_rank_shift.median():.0f}; small towns min Spearman {SMALL.spearman.min():.3f}
CONTEMPORARY circuity {CIRC:.2f}; median km hospital 2011 {NOW.hosp_2011.median():.1f} -> now >=30 beds {NOW.hosp_now.median():.1f}, any inpatient {NOW.inpatient_now.median():.1f}; fire {NOW.fire_2011.median():.1f} -> {NOW.fire_now.median():.1f}; fire<=5km towns {(NOW.fire_2011 <= 5).sum()} -> {(NOW.fire_now <= 5).sum()}
             partial update vs 2011 rho {RHO_UPDATE:.3f}; 2011 IS vs current inpatient km rho {RHO_HOSP_NOW:.2f} (p={P_HOSP_NOW:.3f}); under-served on update: {', '.join(UNDER_NOW) or 'none'}
INTERVALS    90% held-out coverage (random 5-fold x10 | taluka-out), overall / statutory / census / mean half-width:
             plain V1  {CT90.loc[('random 5-fold x10', 'plain (V1)'), 'empirical']:.3f}/{CT90.loc[('random 5-fold x10', 'plain (V1)'), 'cov_statutory']:.3f}/{CT90.loc[('random 5-fold x10', 'plain (V1)'), 'cov_census_town']:.3f}/{CT90.loc[('random 5-fold x10', 'plain (V1)'), 'half_width']:.1f} | {CT90.loc[('leave-one-taluka-out', 'plain (V1)'), 'empirical']:.3f}/{CT90.loc[('leave-one-taluka-out', 'plain (V1)'), 'cov_statutory']:.3f}/{CT90.loc[('leave-one-taluka-out', 'plain (V1)'), 'cov_census_town']:.3f}/{CT90.loc[('leave-one-taluka-out', 'plain (V1)'), 'half_width']:.1f}
             conformal {CT90.loc[('random 5-fold x10', 'conformal'), 'empirical']:.3f}/{CT90.loc[('random 5-fold x10', 'conformal'), 'cov_statutory']:.3f}/{CT90.loc[('random 5-fold x10', 'conformal'), 'cov_census_town']:.3f}/{CT90.loc[('random 5-fold x10', 'conformal'), 'half_width']:.1f} | {CT90.loc[('leave-one-taluka-out', 'conformal'), 'empirical']:.3f}/{CT90.loc[('leave-one-taluka-out', 'conformal'), 'cov_statutory']:.3f}/{CT90.loc[('leave-one-taluka-out', 'conformal'), 'cov_census_town']:.3f}/{CT90.loc[('leave-one-taluka-out', 'conformal'), 'half_width']:.1f}
             Mondrian  {CT90.loc[('random 5-fold x10', 'Mondrian'), 'empirical']:.3f}/{CT90.loc[('random 5-fold x10', 'Mondrian'), 'cov_statutory']:.3f}/{CT90.loc[('random 5-fold x10', 'Mondrian'), 'cov_census_town']:.3f}/{CT90.loc[('random 5-fold x10', 'Mondrian'), 'half_width']:.1f} | {CT90.loc[('leave-one-taluka-out', 'Mondrian'), 'empirical']:.3f}/{CT90.loc[('leave-one-taluka-out', 'Mondrian'), 'cov_statutory']:.3f}/{CT90.loc[('leave-one-taluka-out', 'Mondrian'), 'cov_census_town']:.3f}/{CT90.loc[('leave-one-taluka-out', 'Mondrian'), 'half_width']:.1f}
             run range (random, plain) {CT90.loc[('random 5-fold x10', 'plain (V1)'), 'min_run']:.2f}-{CT90.loc[('random 5-fold x10', 'plain (V1)'), 'max_run']:.2f}
             full-sample half-widths: plain {q90:.1f} -> {', '.join(GAP[GAP.flag == 'Significantly under-served'].town)}; conformal {Q_CORR:.1f} -> {', '.join(UNDER_CORR) or 'none'}; Mondrian statutory {Q_MOND[1]:.1f} / census {Q_MOND[0]:.1f} -> {', '.join(UNDER_MOND) or 'none'}
             Moran's I: InfraScore {MORAN.iloc[0].morans_I:.3f} (p={MORAN.iloc[0].p_perm:.3f}); gap {MORAN.iloc[1].morans_I:.3f} (p={MORAN.iloc[1].p_perm:.3f})
SPILLOVER    K=10 greedy pop-weighted gain {K10i[(False, 'Greedy optimiser')]:.3f} (random {K10i[(False, 'Random (mean of 200)')]:.3f}, ratio {RATIO_NOSPILL:.2f}) -> {K10i[(True, 'Greedy optimiser')]:.3f} with spillover (random {K10i[(True, 'Random (mean of 200)')]:.3f}, ratio {RATIO_SPILL:.2f}); shared towns {', '.join(SHARED_TOWNS)}
REAL COSTS   Rs 250 cr lifecycle: cost-aware {BUDGET_TAB[(BUDGET_TAB.budget_cr == 250) & (BUDGET_TAB.strategy == 'Cost-aware greedy')].pop_weighted_gain.iloc[0]:.2f} | cost-blind {BUDGET_TAB[(BUDGET_TAB.budget_cr == 250) & (BUDGET_TAB.strategy == 'Greedy (ignores cost)')].pop_weighted_gain.iloc[0]:.2f} | lowest-first {BUDGET_TAB[(BUDGET_TAB.budget_cr == 250) & (BUDGET_TAB.strategy == 'Lowest score first')].pop_weighted_gain.iloc[0]:.2f} | random {BUDGET_TAB[(BUDGET_TAB.budget_cr == 250) & (BUDGET_TAB.strategy.str.startswith('Random'))].pop_weighted_gain.iloc[0]:.2f}; MC wins {', '.join(f'{k} {v:.0%}' for k, v in WIN_REAL.items())}
             essential services: cost-aware {ESSENTIAL_TAB.loc['Cost-aware greedy', 'pop_weighted_gain']:.2f} | cost-blind {ESSENTIAL_TAB.loc['Greedy (ignores cost)', 'pop_weighted_gain']:.2f}; plan {', '.join(PLAN250_ESS.town + ' ' + PLAN250_ESS.action)}; from today: {', '.join(PLAN_NOW_ESS.town + ' ' + PLAN_NOW_ESS.action)}
SPILL VALID  model vs census nearest-facility km: median Spearman {SPILL_VAL.spearman.median():.2f} (range {SPILL_VAL.spearman.min():.2f}-{SPILL_VAL.spearman.max():.2f}); upper bound in {np.average(SPILL_VAL.share_pred_ge_census, weights=SPILL_VAL.towns_without):.0%} of pairs
BENEFITS     residents newly within 5 km (hospital / fire): {'; '.join(f"{k}: {int(v.hospital_residents_newly_within_5km):,} / {int(v.fire_residents_newly_within_5km):,}" for k, v in BENEFIT.iterrows())}
SENSITIVITY  {'; '.join(f"{r.setting}: A {r.A_gain:.2f} (x{r.A_ratio_vs_random:.1f}, J={r.A_jaccard_towns:.2f}) B {r.B_gain:.2f} essJ={r.essential_jaccard:.2f}" if pd.notna(r.A_gain) else f"{r.setting}: B {r.B_gain:.2f} vs blind {r.B_cost_blind_gain:.2f} essJ={r.essential_jaccard:.2f}" for r in SENSITIVITY.itertuples())}
WEIGHTS      plans under 400 alternative pillar weightings (flat | centred): A median Jaccard {WROB_SUM.loc['flat Dirichlet(1)', 'A_median_jaccard']:.2f} | {WROB_SUM.loc['centred Dirichlet(20 x base)', 'A_median_jaccard']:.2f}; Sancoale in A {WFREQ_A.loc['Sancoale', 'flat Dirichlet(1)']:.0%} | {WFREQ_A.loc['Sancoale', 'centred Dirichlet(20 x base)']:.0%}, Cuncolim {WFREQ_A.loc['Cuncolim', 'flat Dirichlet(1)']:.0%} | {WFREQ_A.loc['Cuncolim', 'centred Dirichlet(20 x base)']:.0%}; essential: Sancoale fire {WFREQ_E.loc['Sancoale|Fire station', 'flat Dirichlet(1)']:.0%} | {WFREQ_E.loc['Sancoale|Fire station', 'centred Dirichlet(20 x base)']:.0%}, Cuncolim fire {WFREQ_E.loc['Cuncolim|Fire station', 'flat Dirichlet(1)']:.0%} | {WFREQ_E.loc['Cuncolim|Fire station', 'centred Dirichlet(20 x base)']:.0%}, Curchorem-Cacora hospital {WFREQ_E.loc['Curchorem-Cacora|Hospital', 'flat Dirichlet(1)']:.0%} | {WFREQ_E.loc['Curchorem-Cacora|Hospital', 'centred Dirichlet(20 x base)']:.0%}
NO SPILL     hall/library spillovers off, Rs 250 cr: cost-aware {NOSPILL_TAB.loc['Cost-aware greedy', 'pop_weighted_gain']:.2f} ({NOSPILL_TAB.loc['Cost-aware greedy', 'n_facilities']:.0f} facilities, {NOSPILL_TAB.loc['Cost-aware greedy', 'halls_libraries']:.0f} halls/libraries) | cost-blind {NOSPILL_TAB.loc['Greedy (ignores cost)', 'pop_weighted_gain']:.2f} | lowest-first {NOSPILL_TAB.loc['Lowest score first', 'pop_weighted_gain']:.2f} | random {NOSPILL_TAB.loc['Random (mean of 30)', 'pop_weighted_gain']:.2f}; MC wins {', '.join(f'{k} {v:.0%}' for k, v in WIN_NS.items())}; hall opex x10: {HALL10['Cost-aware greedy']:.2f} vs {HALL10['Greedy (ignores cost)']:.2f}
VALIDATION   literacy rho {VAL.iloc[0].spearman_rho:.2f} (p={VAL.iloc[0].p:.3f}); partial (size-controlled) {VAL.iloc[0].partial_rho_ctrl_size:.2f} (p={VAL.iloc[0].p_partial:.3f})
""")

