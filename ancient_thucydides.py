"""
A hand-coded replication of the Thucydides hazard analysis on
ancient state systems.

DATA SOURCE / CAVEAT
====================
The COW NMC series does not exist for antiquity. The estimates below
are coded from secondary literature:

Warring States China (475-221 BC):
  Lewis, "Sanctioned Violence in Early China" (1990)
  Hsu, "Ancient China in Transition" (Stanford 1965)
  Bodde, "China's First Unifier" (Brill 1938)
  Loewe & Shaughnessy (eds), "Cambridge History of Ancient China" (1999)
  Zhao Dingxin, "The Confucian-Legalist State" (Oxford 2015)

Wars of the Diadochi (323-281 BC):
  Anson, "Alexander's Heirs: The Age of the Successors" (Wiley 2014)
  Bosworth, "The Legacy of Alexander" (Oxford 2002)
  Cambridge Ancient History vol. VII.1

The strength estimates are a composite of territorial extent, army
size and economic capacity at decadal snapshots. Each strength score
should be read as plus/minus 30%; the estimates are illustrative,
not precise. The same Thucydides test pipeline is applied as in the
COW analysis, but the small sample makes the statistical inference
descriptive rather than decisive. A proper version would use the
Cioffi-Revilla & Lai (1995, JCR 39:467) digitized catalogue of 1162
ancient Chinese wars (which I do not have access to in this session).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)
NOTE = os.path.join(HERE, "note"); os.makedirs(NOTE, exist_ok=True)


# ============================================================================
# Warring States polity strength estimates
# (composite strength index; absolute scale arbitrary, ratios meaningful)
# Decadal snapshots 400-220 BC
# ============================================================================

YEARS_WS = list(range(-400, -210, 10))   # 400 BC ... 220 BC

WS_STRENGTH = {
    "Qin":  [4, 4, 5, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 30, 45, 70],
    "Chu":  [12,12,11,11,10, 10, 9, 9, 9, 8, 8, 7, 7, 7, 6, 6, 5, 0, 0],
    "Qi":   [8, 8, 8, 7, 7, 7, 8, 8, 9, 6, 5, 6, 7, 7, 7, 6, 5, 4, 0],
    "Wei":  [7, 7, 8, 8, 8, 7, 6, 6, 5, 5, 5, 4, 4, 3, 3, 3, 0, 0, 0],
    "Zhao": [5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 6, 5, 0, 0, 0, 0],
    "Han":  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 0, 0, 0, 0],
    "Yan":  [3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 4, 4, 4, 4, 4, 3, 3, 0, 0],
}

# Major Warring States wars: (start_year, side_A_states, side_B_states, initiator_state)
# initiator chosen as the dominant attacker per the historical record.
WS_WARS = [
    (-354,  ["Wei"], ["Zhao", "Qi"], "Wei"),       # Wei attacks Zhao -> Qi intervenes (Guiling)
    (-341,  ["Wei"], ["Han", "Qi"],  "Wei"),       # Battle of Maling
    (-318,  ["Wei","Han","Zhao","Yan","Chu"], ["Qin"], "Wei"),  # Five-state attack on Qin
    (-316,  ["Qin"], ["Shu","Ba"],   "Qin"),       # Conquest of Shu/Ba (no Warring State target)
    (-312,  ["Qin"], ["Chu"],        "Qin"),       # Battle of Danyang
    (-307,  ["Zhao"], ["Han","Wei"], "Zhao"),      # Wuling reforms / cavalry
    (-301,  ["Qi","Han","Wei"], ["Chu"], "Qi"),    # Garrison campaign
    (-293,  ["Qin"], ["Wei","Han"],  "Qin"),       # Battle of Yique (Bai Qi, 240k dead)
    (-285,  ["Yan","Qin","Zhao","Wei","Han"], ["Qi"], "Yan"),  # Yue Yi's coalition
    (-279,  ["Qi"], ["Yan"],         "Qi"),        # Qi restoration under Tian Dan
    (-278,  ["Qin"], ["Chu"],        "Qin"),       # Capture of Ying (Chu capital)
    (-273,  ["Qin"], ["Wei","Zhao"], "Qin"),       # Battle of Huayang
    (-260,  ["Qin"], ["Zhao"],       "Qin"),       # Changping (400k Zhao dead)
    (-249,  ["Qin"], ["Han"],        "Qin"),       # Annexation of Eastern Zhou
    (-244,  ["Qin"], ["Han","Wei"],  "Qin"),       # Border raids
    (-230,  ["Qin"], ["Han"],        "Qin"),       # Han conquered (END)
    (-228,  ["Qin"], ["Zhao"],       "Qin"),       # Zhao conquered
    (-225,  ["Qin"], ["Wei"],        "Qin"),       # Wei conquered
    (-223,  ["Qin"], ["Chu"],        "Qin"),       # Chu conquered (largest)
    (-222,  ["Qin"], ["Yan"],        "Qin"),       # Yan conquered
    (-221,  ["Qin"], ["Qi"],         "Qin"),       # Qi conquered (last)
]


# ============================================================================
# Hellenistic Diadochi strength estimates (323-275 BC)
# composite index for major successor states; year scale = -year (BCE)
# ============================================================================

YEARS_DI = list(range(-320, -270, 5))

DI_STRENGTH = {
    "Antigonid":   [8, 12, 14, 12, 10, 5, 4, 3, 3, 2],  # Antigonus I peak 314-301, Ipsus crash
    "Ptolemaic":   [6, 7, 8, 9, 10, 11, 11, 12, 12, 12],  # steady growth in Egypt
    "Seleucid":    [3, 4, 6, 8, 10, 14, 16, 18, 20, 22],  # rapid growth post-Ipsus
    "Lysimachid":  [5, 6, 7, 8, 8, 9, 9, 9, 6, 0],       # destroyed at Corupedium 281
    "Cassandrid":  [5, 6, 7, 7, 6, 5, 4, 3, 2, 0],       # Macedon, eventually absorbed
}

DI_WARS = [
    (-321, ["Perdiccas"], ["Ptolemaic"], "Perdiccas"),  # First War of Successors (we'll skip Perdiccas)
    (-319, ["Antigonid"], ["Polyperchon"], "Antigonid"),
    (-315, ["Antigonid"], ["Ptolemaic","Lysimachid","Cassandrid","Seleucid"], "Antigonid"),  # 3rd Diadoch war
    (-312, ["Ptolemaic"], ["Antigonid"], "Ptolemaic"),  # Battle of Gaza
    (-307, ["Antigonid"], ["Cassandrid"], "Antigonid"),
    (-306, ["Antigonid"], ["Ptolemaic"], "Antigonid"),   # Battle of Salamis (naval)
    (-301, ["Cassandrid","Lysimachid","Seleucid"], ["Antigonid"], "Lysimachid"),  # Ipsus - Antigonus killed
    (-288, ["Lysimachid","Ptolemaic"], ["Antigonid"], "Lysimachid"),  # Demetrius defeated
    (-281, ["Seleucid"], ["Lysimachid"], "Seleucid"),    # Corupedium - Lysimachus killed
]


# ============================================================================
# Generic helpers
# ============================================================================

def make_strength_df(strength_dict, years):
    rows = []
    for state, vals in strength_dict.items():
        for y, v in zip(years, vals):
            if v > 0:
                rows.append({"state": state, "year": y, "S": v})
    return pd.DataFrame(rows)


def interp_strength(df, state, year):
    sub = df[df["state"] == state].sort_values("year")
    if sub.empty: return np.nan
    xs = sub["year"].values; ys = sub["S"].values
    if year < xs[0] or year > xs[-1]: return np.nan
    return float(np.interp(year, xs, ys))


def build_dyad_panel(strength_dict, wars, years, window=20):
    """Generic dyad-year panel for an ancient state system.
    window is the lag (in years) for tau."""
    df = make_strength_df(strength_dict, years)
    states = list(strength_dict.keys())

    # build year grid (annual)
    yr_min = min(years); yr_max = max(years)
    year_grid = np.arange(yr_min, yr_max + 1)

    rows = []
    # determine war-set per dyad-year (any war between any state in side A and side B)
    war_set = set()
    init_map = {}
    for (yr, A, B, init) in wars:
        for i in A:
            for j in B:
                if i in strength_dict and j in strength_dict:
                    a, b = sorted([i, j])
                    war_set.add((a, b, yr))
                    init_map[(a, b, yr)] = init

    for i in states:
        for j in states:
            if i >= j: continue
            for t in year_grid:
                Si = interp_strength(df, i, t)
                Sj = interp_strength(df, j, t)
                Si_lag = interp_strength(df, i, t - window)
                Sj_lag = interp_strength(df, j, t - window)
                if any(np.isnan([Si, Sj, Si_lag, Sj_lag])): continue
                if min(Si, Sj, Si_lag, Sj_lag) <= 0: continue
                lr_now = np.log(Si / Sj)
                lr_lag = np.log(Si_lag / Sj_lag)
                tau = abs(lr_now - lr_lag)
                joint = Si + Sj
                war = (i, j, int(t)) in war_set
                rows.append({
                    "i": i, "j": j, "year": int(t),
                    "S_i": Si, "S_j": Sj,
                    "log_ratio": lr_now, "tau": tau,
                    "joint": joint,
                    "war": int(war),
                    "init": init_map.get((i, j, int(t)), ""),
                })
    return pd.DataFrame(rows)


# ============================================================================
# Apply the five tests
# ============================================================================

def stratified(df, n_bins=4):
    qs = np.quantile(df["tau"][df.tau > 0], np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    out = []
    for k in range(len(qs) - 1):
        m = (df["tau"] >= qs[k]) & (df["tau"] < qs[k + 1])
        if m.sum() < 30: continue
        out.append({"tau_lo": qs[k], "tau_hi": qs[k+1], "n": int(m.sum()),
                    "P_war": float(df.loc[m, "war"].mean())})
    return pd.DataFrame(out)


def logit_fit(X, y, lam=1e-2):
    n, p = X.shape; beta = np.zeros(p)
    for _ in range(200):
        z = np.clip(X @ beta, -25, 25); ph = 1/(1+np.exp(-z))
        grad = X.T @ (y - ph) - lam * beta
        W = ph * (1 - ph)
        H = -(X.T * W) @ X - lam * np.eye(p)
        try: step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError: break
        bn = beta - step
        if np.max(np.abs(bn - beta)) < 1e-7: beta = bn; break
        beta = bn
    z = X @ beta; ph = 1/(1+np.exp(-z)); W = ph*(1-ph)
    cov = np.linalg.inv((X.T * W) @ X + lam * np.eye(p))
    return beta, np.sqrt(np.diag(cov))


def regress(df):
    d = df.copy()
    d["log_joint"] = np.log(d["joint"] + 1e-6)
    d["t_scaled"] = (d["year"] - d["year"].mean()) / 50.0
    d["intercept"] = 1.0
    cols = ["intercept", "tau", "log_joint", "t_scaled"]
    X = d[cols].values; y = d["war"].values
    beta, se = logit_fit(X, y)
    return list(zip(cols, beta, se))


def selection_bias(df):
    base = df["war"].mean()
    p80 = np.quantile(df["tau"][df.tau > 0], 0.80)
    p95 = np.quantile(df["tau"][df.tau > 0], 0.95)
    high80 = df["war"][df["tau"] >= p80].mean()
    high95 = df["war"][df["tau"] >= p95].mean()
    return {"base": float(base), "p80": float(p80), "p95": float(p95),
            "h80": float(high80), "h95": float(high95)}


def initiator(df):
    wars = df[df["war"] == 1].copy()
    inc = 0; rsg = 0; tot = 0
    for _, r in wars.iterrows():
        if not r["init"]: continue
        tot += 1
        if r["S_i"] > r["S_j"]: incumbent = r["i"]; rising = r["j"]
        else: incumbent = r["j"]; rising = r["i"]
        if r["init"] == incumbent: inc += 1
        elif r["init"] == rising: rsg += 1
    return {"total": tot, "incumbent": inc, "rising": rsg}


def parity_timing(df):
    war_rows = df[df["war"] == 1].copy()
    timings = []
    for (i, j), grp in df.groupby(["i", "j"]):
        wars_here = war_rows[(war_rows.i == i) & (war_rows.j == j)]
        if wars_here.empty: continue
        lr = grp["log_ratio"].values; yrs = grp["year"].values
        for k in range(1, len(lr)):
            if lr[k-1] * lr[k] < 0:
                cross = yrs[k]
                for _, w in wars_here.iterrows():
                    dt = int(w.year) - int(cross)
                    if abs(dt) <= 50: timings.append(dt)
                break
    return np.array(timings) if timings else np.array([])


# ============================================================================
# Build figure
# ============================================================================

def make_figure_and_report():
    # Warring States panel
    df_ws = build_dyad_panel(WS_STRENGTH, WS_WARS, YEARS_WS, window=20)
    df_di = build_dyad_panel(DI_STRENGTH, DI_WARS, YEARS_DI, window=10)

    print("==================== Warring States China ====================")
    print(f"  panel: {len(df_ws)} dyad-years, wars = {df_ws['war'].sum()}, "
          f"unique dyads = {df_ws.groupby(['i','j']).ngroups}")
    print(); print("T1 Stratified hazard:")
    t1_ws = stratified(df_ws, n_bins=4)
    print(t1_ws.to_string(float_format=lambda x: f"{x:.4f}"))
    print(); print("T2 Logistic regression of war on tau (controlled):")
    t2_ws = regress(df_ws)
    for n, b, s in t2_ws:
        z = b/s; print(f"   {n:<11s} {b:+.3f} ± {s:.3f}  z={z:+.2f}")
    print(); print("T3 Selection-bias-corrected hazards:")
    t3_ws = selection_bias(df_ws)
    print(f"   baseline P(war)               = {t3_ws['base']:.4f}")
    print(f"   P(war | tau >= 80th pct)      = {t3_ws['h80']:.4f}  RR={t3_ws['h80']/t3_ws['base']:.2f}")
    print(f"   P(war | tau >= 95th pct)      = {t3_ws['h95']:.4f}  RR={t3_ws['h95']/t3_ws['base']:.2f}")
    print(); print("T4 Initiator analysis:")
    t4_ws = initiator(df_ws)
    print(f"   n coded={t4_ws['total']}, incumbent={t4_ws['incumbent']}, rising={t4_ws['rising']}")
    print(); print("T5 Timing relative to parity:")
    t5_ws = parity_timing(df_ws)
    if len(t5_ws):
        print(f"   n={len(t5_ws)}, mean Δt = {t5_ws.mean():.1f}y, |Δt|<20y: {(np.abs(t5_ws)<=20).sum()}/{len(t5_ws)}")

    print()
    print("==================== Wars of the Diadochi ====================")
    print(f"  panel: {len(df_di)} dyad-years, wars = {df_di['war'].sum()}, "
          f"unique dyads = {df_di.groupby(['i','j']).ngroups}")
    print(); print("T1 Stratified hazard:")
    t1_di = stratified(df_di, n_bins=3)
    print(t1_di.to_string(float_format=lambda x: f"{x:.4f}"))
    print(); print("T2 Logistic:")
    t2_di = regress(df_di)
    for n, b, s in t2_di:
        z = b/s; print(f"   {n:<11s} {b:+.3f} ± {s:.3f}  z={z:+.2f}")
    print(); print("T3 Selection-bias:")
    t3_di = selection_bias(df_di)
    print(f"   baseline = {t3_di['base']:.4f}")
    print(f"   high-80  = {t3_di['h80']:.4f}  RR={t3_di['h80']/t3_di['base']:.2f}")
    print(f"   high-95  = {t3_di['h95']:.4f}  RR={t3_di['h95']/t3_di['base']:.2f}")
    print(); print("T4 Initiator:")
    t4_di = initiator(df_di)
    print(f"   n={t4_di['total']}, incumbent={t4_di['incumbent']}, rising={t4_di['rising']}")

    # Figure: three-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # (a) WS: stratified
    ax = axes[0, 0]
    tau_c = 0.5*(t1_ws.tau_lo + t1_ws.tau_hi)
    ax.plot(tau_c, t1_ws.P_war, 'o-', color='#d62728', ms=6, lw=1.2,
            label="Warring States China")
    tau_c2 = 0.5*(t1_di.tau_lo + t1_di.tau_hi)
    ax.plot(tau_c2, t1_di.P_war, 's-', color='#2c4d8a', ms=6, lw=1.2,
            label="Hellenistic Diadochi")
    ax.set_xlabel(r"tension proxy  $\tau = |\Delta_{20y}\log(S_i/S_j)|$")
    ax.set_ylabel("P(war) per dyad-year")
    ax.set_title("(a) Hazard vs. tension proxy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (b) Strength time series for Warring States
    ax = axes[0, 1]
    for state, vals in WS_STRENGTH.items():
        ax.plot(YEARS_WS, vals, lw=1.4, label=state)
    ax.set_xlabel("year (BCE, negative)"); ax.set_ylabel("strength index")
    ax.set_title("(b) Warring States strength trajectories")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
    # mark Qin conquests
    for yr, lbl in [(-230,"Han"), (-228,"Zhao"), (-223,"Chu"), (-221,"Qi")]:
        ax.axvline(yr, color='gray', lw=0.4, ls=':')

    # (c) Selection-bias comparison
    ax = axes[1, 0]
    labels = ["all", r"$\tau$>80%", r"$\tau$>95%"]
    x = np.arange(3); w = 0.35
    ws_vals = [t3_ws['base'], t3_ws['h80'], t3_ws['h95']]
    di_vals = [t3_di['base'], t3_di['h80'], t3_di['h95']]
    ax.bar(x - w/2, ws_vals, w, color='#d62728', label='Warring States')
    ax.bar(x + w/2, di_vals, w, color='#2c4d8a', label='Diadochi')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("P(war) per dyad-year")
    ax.set_title("(c) Hazard at high tension vs baseline")
    for k in [1, 2]:
        ax.text(x[k] - w/2, ws_vals[k]+0.001, f"{ws_vals[k]/ws_vals[0]:.1f}×",
                ha='center', fontsize=8, color='#d62728')
        ax.text(x[k] + w/2, di_vals[k]+0.001, f"{di_vals[k]/di_vals[0]:.1f}×",
                ha='center', fontsize=8, color='#2c4d8a')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis='y')

    # (d) Initiator
    ax = axes[1, 1]
    cats = ["WS\nall", "WS\nhigh-τ", "Di\nall", "Di\nhigh-τ"]
    # Recompute high-τ subset initiators
    def hi_init(df):
        p80 = np.quantile(df["tau"][df.tau > 0], 0.70)
        sub = df[df["tau"] >= p80]
        return initiator(sub)
    t4_ws_hi = hi_init(df_ws)
    t4_di_hi = hi_init(df_di)
    incs = [t4_ws['incumbent'], t4_ws_hi['incumbent'],
            t4_di['incumbent'], t4_di_hi['incumbent']]
    risgs = [t4_ws['rising'], t4_ws_hi['rising'],
             t4_di['rising'], t4_di_hi['rising']]
    tots = [t4_ws['total'], t4_ws_hi['total'],
            t4_di['total'], t4_di_hi['total']]
    pct_inc = [100*a/max(1, t) for a, t in zip(incs, tots)]
    pct_rsg = [100*a/max(1, t) for a, t in zip(risgs, tots)]
    x = np.arange(4); w = 0.35
    ax.bar(x - w/2, pct_inc, w, color='#2c4d8a', label='incumbent')
    ax.bar(x + w/2, pct_rsg, w, color='#d62728', label='rising')
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("% of wars initiated")
    ax.set_title("(d) Initiator role")
    ax.axhline(50, color='k', ls=':', lw=0.5)
    for k, t in enumerate(tots):
        ax.text(k, 100, f"n={t}", ha='center', fontsize=7, color='#555')
    ax.set_ylim(0, 110); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle("Thucydides mechanism applied to ancient state systems (hand-coded)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_ancient_thucydides.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # JSON dump for the note
    import json
    summary = {
        "WS": {
            "n_dyad_years": int(len(df_ws)),
            "n_wars": int(df_ws.war.sum()),
            "T1": t1_ws.to_dict(orient="list"),
            "T2": [(n, b, s) for n, b, s in t2_ws],
            "T3": t3_ws,
            "T4_all": t4_ws,
            "T4_high": t4_ws_hi,
            "T5_n": int(len(t5_ws)),
            "T5_mean": float(np.mean(t5_ws)) if len(t5_ws) else None,
            "T5_within_20": int((np.abs(t5_ws) <= 20).sum()) if len(t5_ws) else 0,
        },
        "Di": {
            "n_dyad_years": int(len(df_di)),
            "n_wars": int(df_di.war.sum()),
            "T1": t1_di.to_dict(orient="list"),
            "T2": [(n, b, s) for n, b, s in t2_di],
            "T3": t3_di,
            "T4_all": t4_di,
            "T4_high": t4_di_hi,
        },
    }
    with open(os.path.join(NOTE, "ancient_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {NOTE}/ancient_summary.json")
    return summary


if __name__ == "__main__":
    make_figure_and_report()
