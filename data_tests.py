"""
Empirical tests of the minimal-model assumptions against COW data.

Tests:
  T1  P(i wins) vs CINC ratio    -- direct test of the model rule
                                      P(i wins) = S_i/(S_i+S_j)
                                      (Lanchester-style war outcome)
  T2  Dyadic CINC differential rate -- "Thucydides hazard": does the
                                      rate of relative change predict
                                      MID/war onset?
  T3  Khaldun ceiling             -- after peak-CINC, do states decline?
  T4  Boundary effect (B_ij)      -- do wars cluster among geographic
                                      neighbours? Use COW contig
                                      (if present) or heuristic.

Output: figs/fig8_model_tests.pdf
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from data_analysis import _read_cr_csv, load_nmc, load_mid, load_cow_war, load_states

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


# ----------------------------------------------------------------------------
# T1: P(win) vs CINC ratio
# ----------------------------------------------------------------------------

def t1_outcome_vs_cinc():
    """SIDE-level test of the model rule P(side_i wins) = C_i/(C_i+C_j),
    where C_i = sum of CINC over side i. This is the correct analogue of
    the model's strength-summation: in R3 the winner is drawn weighted by
    aggregate strength, not by individual.
    """
    df = _read_cr_csv(os.path.join(HERE, "DataSets", "Inter-StateWarData_v4.0.csv"))
    nmc = load_nmc()
    df = df[df["Outcome"].isin([1, 2])].copy()
    df["year"] = df["StartYear1"].astype(int)
    df = df.merge(nmc[["ccode", "year", "cinc"]],
                  on=["ccode", "year"], how="left")
    df = df.dropna(subset=["cinc"])

    pairs = []
    for wnum, sub in df.groupby("WarNum"):
        # Aggregate CINC by Side
        side_C = sub.groupby("Side")["cinc"].sum()
        side_O = sub.groupby("Side")["Outcome"].first()
        if len(side_C) != 2:
            continue
        sides = list(side_C.index)
        s1, s2 = sides[0], sides[1]
        C1, C2 = side_C[s1], side_C[s2]
        O1, O2 = side_O[s1], side_O[s2]
        if C1 + C2 == 0:
            continue
        # Skip wars where outcomes are not exactly {1, 2}
        if {O1, O2} != {1, 2}:
            continue
        f1 = C1 / (C1 + C2)
        win1 = 1 if O1 == 1 else 0
        pairs.append((f1, win1))
        pairs.append((1 - f1, 1 - win1))  # symmetric: side 2 view
    arr = np.array(pairs)
    f = arr[:, 0]; y = arr[:, 1].astype(int)
    print(f"T1: n={len(arr)} side-level observations from COW wars")
    return f, y


# ----------------------------------------------------------------------------
# T2: Thucydides hazard via dyadic CINC dynamics
# ----------------------------------------------------------------------------

def t2_dyadic_hazard(major_only=True, window=10):
    """Look at all dyad-years among major powers. Tension proxy = the
    absolute change in log(CINC_i / CINC_j) over a `window`-year window.
    Then bin years by tension and compute the fraction of years where the
    dyad had a MID or war begin.

    The model predicts war rate ~ w * B * sigma_ij. Empirically: war hazard
    should rise monotonically with tension proxy.
    """
    nmc = load_nmc()
    mid = load_mid()
    war = _read_cr_csv(os.path.join(HERE, "DataSets", "Inter-StateWarData_v4.0.csv"))

    # MID dyad-events. MIDA gives dispute-level info; for dyads we need MIDB.
    midb_path = os.path.join(HERE, "DataSets",
                             "MID-5-Data-and-Supporting-Materials", "MIDB 5.0.csv")
    midb = pd.read_csv(midb_path)
    midb.columns = [c.lower() for c in midb.columns]
    # Each row is a participant in a dispute. Pair them up by dispnum to form
    # ordered dyads. Use side A vs side B.
    if "sidea" in midb.columns:
        a = midb[midb["sidea"] == 1][["dispnum", "ccode", "styear"]].rename(columns={"ccode": "ccode_a"})
        b = midb[midb["sidea"] == 0][["dispnum", "ccode", "styear"]].rename(columns={"ccode": "ccode_b"})
        dyads = a.merge(b, on=["dispnum", "styear"])
    else:
        # Fallback: use first vs others
        dyads = pd.DataFrame()
    dyad_events = set()
    for _, r in dyads.iterrows():
        dyad_events.add((int(r["ccode_a"]), int(r["ccode_b"]), int(r["styear"])))
        dyad_events.add((int(r["ccode_b"]), int(r["ccode_a"]), int(r["styear"])))
    print(f"T2: {len(dyad_events)} MID dyad-year events")

    # Major powers list
    maj = pd.read_csv(os.path.join(HERE, "DataSets", "MajorPowers2024", "majors2024.csv"))
    maj_codes = set(maj["ccode"].tolist())

    # Build dyad CINC time-series for major powers
    nmc_m = nmc[nmc["ccode"].isin(maj_codes)].copy() if major_only else nmc.copy()
    if not major_only:
        # restrict to states present long enough
        counts = nmc_m.groupby("ccode")["year"].count()
        nmc_m = nmc_m[nmc_m["ccode"].isin(counts[counts > 20].index)]
    codes = sorted(nmc_m["ccode"].unique())

    records = []
    for i in codes:
        for j in codes:
            if i >= j:
                continue
            ci = nmc_m[nmc_m["ccode"] == i].set_index("year")["cinc"]
            cj = nmc_m[nmc_m["ccode"] == j].set_index("year")["cinc"]
            common = ci.index.intersection(cj.index)
            if len(common) < window + 1:
                continue
            ci = ci.loc[common]; cj = cj.loc[common]
            ratio = np.log(ci.values / np.maximum(cj.values, 1e-6))
            # tension proxy = absolute change in log-ratio over `window` years
            tens = np.abs(ratio[window:] - ratio[:-window])
            years = common[window:]
            for yr, t in zip(years, tens):
                event = ((i, j, int(yr)) in dyad_events) or ((j, i, int(yr)) in dyad_events)
                records.append((float(t), int(event)))
    arr = np.array(records)
    print(f"T2: {len(arr)} dyad-year observations")
    return arr


# ----------------------------------------------------------------------------
# T3: Khaldun ceiling -- post-peak CINC trajectories
# ----------------------------------------------------------------------------

def t3_khaldun_post_peak(top_n=8, post_years=40):
    """For each state's CINC trajectory, find the year of maximum CINC.
    Align traces around that peak and report mean trajectory. The model
    predicts that high-CINC empires subsequently decline (because the
    Khaldun ceiling C_i = A_i/(1+A_i/A*) prevents indefinite scaling).
    """
    nmc = load_nmc()
    # Pick states that ever reached CINC > 0.05 (great powers)
    max_by = nmc.groupby("ccode")["cinc"].max().sort_values(ascending=False)
    great = max_by[max_by > 0.05].head(top_n).index.tolist()
    states = load_states()

    traces = []
    labels = []
    for code in great:
        s = nmc[nmc["ccode"] == code].set_index("year")["cinc"].dropna()
        if len(s) < post_years:
            continue
        peak_yr = s.idxmax()
        # take 20 years before to post_years after
        rel = s.copy(); rel.index = rel.index - peak_yr
        traces.append(rel)
        nm = states[states["ccode"] == code]
        name = nm.iloc[0]["statenme"] if len(nm) else f"ccode {code}"
        labels.append((name, peak_yr, float(s.max())))
    return traces, labels


# ----------------------------------------------------------------------------
# T4: Boundary / geographic effect
# ----------------------------------------------------------------------------

def t4_boundary_effect():
    """Most COW participants are in adjacent or close-by regions.
    Cheap proxy: COW lists 'WhereFought' region codes; same WhereFought
    means geographically clustered. The model predicts war rate ~ B_ij,
    so wars between same-region states should dominate.
    """
    df = _read_cr_csv(os.path.join(HERE, "DataSets", "Inter-StateWarData_v4.0.csv"))
    # Region key from WhereFought
    df["WhereFought"] = pd.to_numeric(df["WhereFought"], errors="coerce")
    wf_counts = df["WhereFought"].dropna().astype(int).value_counts()
    region_names = {1: "W. Hemisphere", 2: "Europe", 3: "Africa", 4: "Sub-Sah. Africa",
                    5: "Middle East", 6: "MENA", 7: "Asia", 11: "Eur+ME", 12: "Eur+Asia",
                    13: "WHem+Eur", 14: "WW2-like", 15: "WW1-like", 16: "Pacific",
                    17: "WW2 global", 18: "Africa+ME", 19: "WW2 Pacific"}
    return wf_counts, region_names


# ----------------------------------------------------------------------------
# Composite figure
# ----------------------------------------------------------------------------

def make_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # ---- (a) outcome vs CINC ratio ----
    f, y = t1_outcome_vs_cinc()
    bins = np.linspace(0, 1, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    pw = np.zeros(len(centers)); pw[:] = np.nan
    counts = np.zeros(len(centers))
    for k in range(len(centers)):
        m = (f >= bins[k]) & (f < bins[k + 1])
        if m.sum() >= 3:
            pw[k] = y[m].mean(); counts[k] = m.sum()
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label="model: $P(i\\,\\mathrm{wins}) = f$")
    sizes = 30 + 200 * counts / max(counts.max(), 1)
    ax.scatter(centers, pw, s=sizes, c='#d62728', edgecolor='black',
               linewidth=0.5, zorder=3, label="COW data (bin size ∝ n)")
    ax.set_xlabel("CINC fraction  $f = c_i/(c_i+c_j)$")
    ax.set_ylabel("empirical P(i wins)")
    ax.set_title("(a) War outcome vs. capability ratio")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)

    # ---- (b) hazard vs CINC tension ----
    arr = t2_dyadic_hazard()
    t = arr[:, 0]; ev = arr[:, 1]
    # Log-spaced bins
    qs = np.quantile(t[t > 0], np.linspace(0, 1, 9))
    hz = []; ts = []
    for k in range(len(qs) - 1):
        m = (t >= qs[k]) & (t < qs[k + 1])
        if m.sum() > 30:
            hz.append(ev[m].mean()); ts.append(0.5 * (qs[k] + qs[k + 1]))
    ax = axes[0, 1]
    ax.plot(ts, hz, 'o-', color='#2c4d8a', ms=5, lw=1.2)
    ax.set_xlabel("tension proxy  $|\\Delta_{10y}\\log(c_i/c_j)|$")
    ax.set_ylabel("P(MID dyad-event in year)")
    ax.set_title("(b) Thucydides hazard: MID rate vs. CINC change")
    ax.grid(True, alpha=0.25)
    # also fit a logistic for context
    from numpy.polynomial import polynomial as P
    if len(ts) > 3:
        ax.text(0.02, 0.92, "monotone rise expected by model",
                transform=ax.transAxes, fontsize=8, color='#444')

    # ---- (c) Khaldun: post-peak CINC ----
    traces, labels = t3_khaldun_post_peak(top_n=8, post_years=50)
    ax = axes[1, 0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(traces)))
    aligned = []
    for tr, lbl, col in zip(traces, labels, cmap):
        # normalise by peak value
        tr_n = tr / tr.max()
        ax.plot(tr_n.index, tr_n.values, '-', color=col, alpha=0.85,
                lw=1.0, label=f"{lbl[0]} (peak {lbl[1]})")
        aligned.append(tr_n)
    # mean trajectory in [-20, +50]
    grid = np.arange(-20, 51)
    rows = []
    for tr_n in aligned:
        rows.append(tr_n.reindex(grid).values)
    M = np.vstack(rows)
    mean = np.nanmean(M, axis=0)
    ax.plot(grid, mean, 'k-', lw=2.5, label="mean across great powers")
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel("years since CINC peak")
    ax.set_ylabel("CINC / CINC$_{\\max}$")
    ax.set_title("(c) Khaldun: post-peak decline of great powers")
    ax.set_xlim(-20, 50); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)

    # ---- (d) regional clustering ----
    wf_counts, region_names = t4_boundary_effect()
    top = wf_counts.head(10)
    names = [region_names.get(int(c), str(c)) for c in top.index]
    ax = axes[1, 1]
    ax.barh(range(len(top)), top.values[::-1], color='#7d4caf')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("number of COW wars (1816-2003)")
    ax.set_title("(d) Wars by 'WhereFought' region — geographic clustering")
    ax.grid(True, alpha=0.25, axis='x')

    fig.suptitle("Empirical tests of model assumptions (COW data)", fontsize=11, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig8_model_tests.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return labels  # for downstream printing


def t1_summary_stats():
    """Print a quantitative comparison: average empirical P(win | f) vs model."""
    f, y = t1_outcome_vs_cinc()
    bins = np.linspace(0, 1, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    print("\nT1 P(i wins) by CINC fraction bin:")
    print(f"  bin centre     n   P(win) empirical   model (=f)")
    for k in range(len(centers)):
        m = (f >= bins[k]) & (f < bins[k + 1])
        if m.sum() >= 3:
            print(f"   {centers[k]:.2f}      {m.sum():4d}    {y[m].mean():.3f}"
                  f"             {centers[k]:.3f}")
    # average difference
    diff = []
    for k in range(len(centers)):
        m = (f >= bins[k]) & (f < bins[k + 1])
        if m.sum() >= 3:
            diff.append((y[m].mean() - centers[k]))
    print(f"  Mean |empirical - model| across bins: {np.mean(np.abs(diff)):.3f}")
    print(f"  Pearson r(f, win) = {np.corrcoef(f, y)[0,1]:.3f}")


if __name__ == "__main__":
    print("Running model-vs-data tests ...")
    labels = make_figure()
    t1_summary_stats()
    print("\nGreat powers identified (T3):")
    for name, peak, cmax in labels:
        print(f"  {name:30s}  peak={peak}  CINC={cmax:.3f}")
