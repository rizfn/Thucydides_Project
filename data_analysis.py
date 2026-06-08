"""
Empirical data analysis: confront the minimal Thucydides model with COW,
MID, NMC and Brecke data dropped in DataSets/.

Outputs into figs/:
  fig5_warsize_data_vs_model.pdf : Richardson CCDF, COW + Brecke + simulation
  fig6_cinc_dyads.pdf            : CINC trajectories for rising-vs-incumbent
                                   power dyads
  fig7_mid_escalation.pdf        : MID hostility-level distribution and the
                                   fraction that escalate to war (proxy for
                                   the sigma->war transition in the model)
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "DataSets")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


# ---------- helpers ---------------------------------------------------------

def _read_cr_csv(path):
    """Read a CSV that uses CR (Mac) line endings."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", "replace")
    from io import StringIO
    return pd.read_csv(StringIO(text))


def ccdf(values):
    v = np.sort(np.asarray(values))
    n = len(v)
    return v, 1.0 - np.arange(n) / n


# ---------- loaders ---------------------------------------------------------

def load_cow_war():
    df = _read_cr_csv(os.path.join(DATA, "Inter-StateWarData_v4.0.csv"))
    # Aggregate battle-deaths to war level: sum across all participants
    df["BatDeath"] = pd.to_numeric(df["BatDeath"], errors="coerce")
    df.loc[df["BatDeath"] < 0, "BatDeath"] = np.nan  # COW uses -8/-9 for missing
    war = (df.groupby(["WarNum", "WarName"], as_index=False)
             .agg(total_deaths=("BatDeath", "sum"),
                  startyear=("StartYear1", "min"),
                  endyear=("EndYear1", "max"),
                  n_participants=("ccode", "nunique")))
    war = war[war["total_deaths"] > 0].reset_index(drop=True)
    return war


def load_brecke():
    path = os.path.join(DATA, "Conflict-Catalog-18-vars.xlsx")
    df = pd.read_excel(path)
    return df


def load_mid():
    df = pd.read_csv(os.path.join(DATA, "MID-5-Data-and-Supporting-Materials", "MIDA 5.0.csv"))
    df.loc[df["fatality"] < 0, "fatality"] = np.nan
    return df


def load_nmc():
    df = pd.read_csv(os.path.join(DATA, "NMC_Documentation-6", "NMC-60-abridged.csv"))
    # Convert flagged missing
    for c in ["milex", "milper", "irst", "pec", "tpop", "upop", "cinc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan
    return df


def load_states():
    return pd.read_csv(os.path.join(DATA, "States2024", "statelist2024.csv"))


# ---------- model output ----------------------------------------------------

def model_war_sizes(seed=2, gap=2.0):
    """Cluster simulated war events into 'conflicts' and return sizes."""
    pkl = os.path.join(HERE, f"baseline_v3_L32_T1500_s{seed}.pkl")
    if not os.path.exists(pkl):
        # Fall back: try outputs cache
        pkl = os.path.join(os.path.dirname(HERE), "outputs", f"baseline_v3_L32_T1500_s{seed}.pkl")
    with open(pkl, "rb") as f:
        sim = pickle.load(f)
    by_pair = defaultdict(list)
    for ev in sim.war_log:
        t, w, l = ev[0], ev[1], ev[2]
        by_pair[frozenset((w, l))].append(t)
    sizes = []
    for ts in by_pair.values():
        ts.sort()
        chunk = [ts[0]]
        for x in ts[1:]:
            if x - chunk[-1] < gap:
                chunk.append(x)
            else:
                sizes.append(len(chunk)); chunk = [x]
        sizes.append(len(chunk))
    return np.array(sizes)


# ---------- figure 5: war-size CCDF -----------------------------------------

def figure_warsize_data_vs_model():
    cow = load_cow_war()
    brecke = load_brecke()
    # Brecke has columns like 'TotalFatalities' or similar; locate it.
    fatcol = None
    for c in brecke.columns:
        if "fat" in c.lower() or "death" in c.lower() or "tot" in c.lower():
            fatcol = c
            break
    print("Brecke columns:", list(brecke.columns)[:15])
    if fatcol is None:
        # fall back to any numeric column with reasonable range
        for c in brecke.columns:
            if pd.api.types.is_numeric_dtype(brecke[c]):
                if brecke[c].max() > 100:
                    fatcol = c; break
    print("Brecke fatalcol:", fatcol)
    brecke_fat = pd.to_numeric(brecke[fatcol], errors="coerce") if fatcol else None
    if brecke_fat is not None:
        brecke_fat = brecke_fat.dropna()
        brecke_fat = brecke_fat[brecke_fat > 0]

    sim_sizes = model_war_sizes(seed=2)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    # COW
    v, c = ccdf(cow["total_deaths"].values)
    ax.plot(v, c, '-', lw=1.4, color='#2c4d8a', label=f"COW interstate wars (n={len(v)})")

    # Brecke (if usable)
    if brecke_fat is not None and len(brecke_fat) > 50:
        v, c = ccdf(brecke_fat.values)
        ax.plot(v, c, '-', lw=1.0, color='#7d4caf',
                label=f"Brecke conflict catalog (n={len(v)})")

    # Model — rescale x by a constant so two CCDFs occupy comparable ranges.
    # The model's "size" is events-per-conflict, dimensionally different from
    # battle deaths. Show it on a normalised x = s / s_median for comparison.
    if len(sim_sizes) > 50:
        # scale so the median of the model matches the median of COW
        sf = np.median(cow["total_deaths"].values) / max(1, np.median(sim_sizes))
        v, c = ccdf(sim_sizes * sf)
        ax.plot(v, c, ':', lw=1.4, color='#d62728',
                label=f"Model (rescaled, n={len(v)})")

    # power-law guide lines (slope -1 and -1.5 on CCDF = PDF exponent 2 / 2.5)
    xs = np.array([1e2, 1e7])
    for slope, ls, lab in [(-0.6, '--', r'$\alpha=1.6$'),
                            (-1.0, ':', r'$\alpha=2.0$')]:
        ys = 0.5 * (xs / xs[0]) ** slope
        ax.plot(xs, ys, ls, color='gray', lw=0.7, alpha=0.6,
                label=lab + ' guide')

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("war size  $s$  (battle deaths or rescaled events)")
    ax.set_ylabel(r"CCDF  $P(S \geq s)$")
    ax.set_title("Empirical war-size distributions vs. minimal model")
    ax.legend(fontsize=8, loc='lower left')
    ax.set_xlim(1e1, 1e8); ax.set_ylim(2e-4, 1.2)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig5_warsize_data_vs_model.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- figure 6: CINC dyads --------------------------------------------

def figure_cinc_dyads():
    nmc = load_nmc()
    states = load_states()
    # Curated rising-vs-incumbent dyads from Allison's catalogue
    # ccode reference: USA=2, UK=200, Germany=255, France=220,
    #                  Russia/USSR=365, Japan=740, China(PRC)=710
    dyads = [
        ("UK vs Germany (pre-1914)",     200, 255, 1816, 1918, "war"),
        ("UK vs USA (1850-1945)",        200,   2, 1850, 1945, "no war"),
        ("USSR vs USA (1945-1991)",      365,   2, 1945, 1991, "no war"),
        ("USA vs Japan (pre-1941)",        2, 740, 1900, 1945, "war"),
        ("USA vs China (1990-2016)",       2, 710, 1990, 2016, "no war yet"),
        ("France vs UK (1816-1900)",     220, 200, 1816, 1900, "no war"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharey=False)
    axes = axes.ravel()
    for ax, (label, c1, c2, y0, y1, status) in zip(axes, dyads):
        d1 = nmc[(nmc["ccode"] == c1) & (nmc["year"] >= y0) & (nmc["year"] <= y1)]
        d2 = nmc[(nmc["ccode"] == c2) & (nmc["year"] >= y0) & (nmc["year"] <= y1)]
        if d1.empty or d2.empty:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=9); continue
        s1 = d1.set_index("year")["cinc"]; s2 = d2.set_index("year")["cinc"]
        ax.plot(s1.index, s1.values, '-', lw=1.4, color='#2c4d8a',
                label=_state_name(states, c1))
        ax.plot(s2.index, s2.values, '-', lw=1.4, color='#d62728',
                label=_state_name(states, c2))
        ax.set_title(f"{label}  [{status}]", fontsize=9)
        ax.set_xlabel("year", fontsize=8); ax.set_ylabel("CINC", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.suptitle("CINC trajectories for selected great-power dyads",
                 fontsize=10, y=0.99)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig6_cinc_dyads.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _state_name(states, ccode):
    row = states[states["ccode"] == ccode]
    if len(row) == 0: return f"ccode={ccode}"
    return row.iloc[0]["statenme"]


# ---------- figure 7: MID escalation ----------------------------------------

def figure_mid_escalation():
    """Plot MID hostility-level histogram + escalation fraction.

    hostlev ranges from 1 (no militarised action) to 5 (interstate war).
    The model's analogue: sigma_ij level vs. probability of war event.
    """
    mid = load_mid()
    # binned by hostility level
    hl = mid["hostlev"].astype(int).values
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    counts, _ = np.histogram(hl, bins=bins)
    levels = np.arange(1, 6)
    labels = ["1: none", "2: threat", "3: display", "4: use of force", "5: war"]
    axes[0].bar(levels, counts, color="#2c4d8a")
    axes[0].set_xticks(levels); axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel("count of MIDs"); axes[0].set_title("MID hostility distribution (1816-2014)")
    axes[0].grid(True, alpha=0.2, axis='y')

    # Fraction of MIDs at each level that "escalated" (had any fatality)
    df = mid.copy()
    df["fatal"] = (df["fatality"] >= 2).astype(int)  # fatality 1 = none, 2+ = some
    grp = df.groupby("hostlev")["fatal"].mean()
    axes[1].bar(grp.index, grp.values, color="#d62728")
    axes[1].set_xticks(levels); axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel("fraction with fatalities")
    axes[1].set_title("Escalation fraction by hostility level")
    axes[1].grid(True, alpha=0.2, axis='y')

    fig.suptitle("MID v5 (Maoz et al.): tension level → war analogue",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig7_mid_escalation.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- summary ---------------------------------------------------------

def print_summary():
    cow = load_cow_war()
    mid = load_mid()
    nmc = load_nmc()
    states = load_states()
    print("=" * 62)
    print("DATASET SUMMARY")
    print("=" * 62)
    print(f"COW Inter-State Wars     : {len(cow):4d} wars, "
          f"years {int(cow.startyear.min())}–{int(cow.endyear.max())}, "
          f"battle deaths total = {cow.total_deaths.sum():,.0f}")
    print(f"  median war size = {cow.total_deaths.median():,.0f}, "
          f"max = {cow.total_deaths.max():,.0f}")
    print(f"COW MID v5.0             : {len(mid):4d} disputes, "
          f"years {mid.styear.min()}–{mid.endyear.max()}")
    print(f"COW NMC v6.0             : {len(nmc):,d} country-years, "
          f"{nmc.ccode.nunique():3d} states, "
          f"years {int(nmc.year.min())}–{int(nmc.year.max())}")
    print(f"States system            : {len(states):3d} states")


# ---------- main ------------------------------------------------------------

if __name__ == "__main__":
    print_summary()
    print()
    figure_warsize_data_vs_model()
    figure_cinc_dyads()
    figure_mid_escalation()
