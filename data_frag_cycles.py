"""
Empirical tests of (a) fragmentation and (b) cyclicity, against the
predictions of the minimal model.

Predictions:
  (a) The fragmentation hazard scales with state size A and with the
      strength deficit (1 - S/A).  Empirical signature: large states
      should die sooner than small states, all else equal.  And the
      decade-rate of state births/deaths should be highly bursty.
  (b) The model produces relaxation-oscillator cycles in A_max(t):
      a state grows -> Khaldun ceiling caps strength density -> low
      S/A makes it brittle -> collapses -> new state grows.
      Empirical signature: low-frequency periodic component in war
      activity / great-power capability concentration.

Outputs: figs/fig_fragmentation.pdf, figs/fig_cycles.pdf
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_analysis import _read_cr_csv, load_nmc, load_states

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


# ----------------------------------------------------------------------------
# Fragmentation evidence
# ----------------------------------------------------------------------------

KNOWN_FRAG_DEATHS = {
    # Modern fragmentations (state split into smaller pieces)
    "USSR": 1991, "YUG": 2006, "CZE": 1992, "YUG_91": 1991,
    "PAK": 1971,                 # Pakistan -> Bangladesh
    "AUH": 1918,                 # Austria-Hungary
    "OTT": 1920,                 # Ottoman Empire
}


def collect_state_lifetimes():
    """Collect (state, lifetime, peak_CINC, death_type) for each state."""
    nmc = load_nmc()
    states = load_states()
    # Aggregate: for each ccode pick the FIRST start and LAST end if multiple
    info = (states.groupby(["ccode", "stateabb"], as_index=False)
                  .agg(styear=("styear", "min"),
                       endyear=("endyear", "max"),
                       statenme=("statenme", "first")))
    # Compute peak CINC per ccode
    peak = nmc.groupby("ccode")["cinc"].max().rename("peak_cinc")
    info = info.join(peak, on="ccode")
    info["lifetime"] = info.endyear - info.styear
    info["died"] = info.endyear < 2024
    return info


def figure_fragmentation():
    info = collect_state_lifetimes()
    cow_states = info.dropna(subset=["peak_cinc"])
    dead = info[info.died]

    # Decade-level "state death" rate (proxy for fragmentation flux)
    info_d = info.copy()
    info_d["death_decade"] = (info_d.endyear // 10) * 10
    death_counts = info_d[info_d.died].groupby("death_decade").size()
    birth_counts = info_d.assign(birth_decade=(info_d.styear // 10) * 10) \
                         .groupby("birth_decade").size()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    # --- (a) State birth/death rate over time -------------------------------
    ax = axes[0, 0]
    decs = sorted(set(death_counts.index) | set(birth_counts.index))
    ax.bar(decs, [birth_counts.get(d, 0) for d in decs], width=8, color="#2c4d8a",
           alpha=0.7, label="state births")
    ax.bar(decs, [-death_counts.get(d, 0) for d in decs], width=8, color="#d62728",
           alpha=0.7, label="state deaths")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("decade"); ax.set_ylabel("count per decade")
    ax.set_title("(a) State births and deaths in the COW state list (1816-2024)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    # Annotate major fragmentation waves
    for yr, lbl in [(1810, "indep.\nL.Am."), (1920, "post-WWI"),
                    (1960, "decolon."), (1990, "USSR/\nYugoslavia")]:
        ax.annotate(lbl, xy=(yr, birth_counts.get((yr // 10) * 10, 0) + 1),
                    ha='center', fontsize=7, color="#2c4d8a")

    # --- (b) Lifetime distribution ----------------------------------------
    ax = axes[0, 1]
    finite = info[info.died].lifetime.dropna()
    ax.hist(finite, bins=np.linspace(0, 200, 21), color="#7d4caf", alpha=0.8)
    ax.set_xlabel("state lifetime (years)")
    ax.set_ylabel("count of dead states")
    ax.set_title(f"(b) Lifetime distribution of dead states (n={len(finite)})")
    ax.axvline(finite.median(), color='k', ls='--', lw=0.8,
               label=f"median = {finite.median():.0f} yr")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- (c) Lifetime vs peak CINC (model prediction: anti-correlation
    #         once you fix S/A.  Use peak CINC as a proxy for peak A).
    ax = axes[1, 0]
    cow_dead = cow_states[cow_states.died].copy()
    ax.scatter(cow_dead.peak_cinc, cow_dead.lifetime,
               s=22, alpha=0.7, color="#d62728", edgecolor='k', lw=0.4,
               label="dead states")
    cow_alive = cow_states[~cow_states.died]
    ax.scatter(cow_alive.peak_cinc, cow_alive.lifetime,
               s=22, alpha=0.4, color="#2c4d8a", edgecolor='k', lw=0.4,
               label="extant (right-censored)")
    # log scale on x
    ax.set_xscale("log")
    ax.set_xlabel("peak CINC")
    ax.set_ylabel("lifetime (years)")
    ax.set_title("(c) State lifetime vs. peak capability")
    # annotate notable cases
    for nm, code in [("USSR", 365), ("Yugoslavia", 345), ("Austria-H.", 300),
                      ("Ottoman E.", 640), ("UK", 200), ("USA", 2),
                      ("Germany", 255)]:
        row = cow_states[cow_states.ccode == code]
        if len(row):
            r = row.iloc[0]
            if not np.isnan(r.peak_cinc):
                ax.annotate(nm, xy=(r.peak_cinc, r.lifetime),
                            xytext=(5, 5), textcoords='offset points', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- (d) Pre-death CINC trajectory (Khaldun signature) -----------------
    ax = axes[1, 1]
    nmc = load_nmc()
    # For each dead state, find CINC values 20 yr before death and compute mean trajectory
    pre_death = []
    death_year = {}
    for _, r in dead.iterrows():
        ts = nmc[(nmc.ccode == r.ccode) & (nmc.year <= r.endyear)]
        if len(ts) < 8: continue
        ts = ts.sort_values("year")
        rel_yr = ts.year.values - r.endyear
        cinc_n = ts.cinc.values / max(1e-9, ts.cinc.max())
        # keep 20yr before
        mask = rel_yr >= -25
        if mask.sum() > 5:
            pre_death.append((rel_yr[mask], cinc_n[mask], r.statenme))
    # plot a sample of 6 plus mean
    sample = pre_death[:8]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample)))
    grid = np.arange(-25, 1)
    rows = []
    for (yr, ci, nm), col in zip(sample, cmap):
        ax.plot(yr, ci, '-', alpha=0.6, lw=0.9, color=col, label=nm[:18])
        s = pd.Series(ci, index=yr).reindex(grid)
        rows.append(s.values)
    if rows:
        M = np.vstack(rows)
        mean = np.nanmean(M, axis=0)
        ax.plot(grid, mean, "k-", lw=2.4, label="mean")
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("years before state death")
    ax.set_ylabel("CINC / CINC$_{\\max}$")
    ax.set_title("(d) CINC trajectory leading to state death (Khaldun signature)")
    ax.legend(fontsize=6, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.2)

    fig.suptitle("Empirical fragmentation evidence in COW data", fontsize=11, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_fragmentation.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------
# Cyclicity evidence
# ----------------------------------------------------------------------------

def figure_cycles():
    war = _read_cr_csv(os.path.join(HERE, "DataSets", "Inter-StateWarData_v4.0.csv"))
    nmc = load_nmc()
    war_starts = war.drop_duplicates("WarNum")[["WarNum", "StartYear1"]].rename(
        columns={"StartYear1": "year"})

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    # --- (a) War onset rate over time (smoothed) ---------------------------
    yrs = np.arange(1816, 2010)
    cnt = np.array([(war_starts.year == y).sum() for y in yrs])
    window = 11
    smoothed = np.convolve(cnt, np.ones(window) / window, mode="same")
    ax = axes[0, 0]
    ax.bar(yrs, cnt, width=1.0, color="#d62728", alpha=0.35,
           label="annual war onsets")
    ax.plot(yrs, smoothed, color="#2c4d8a", lw=1.4,
            label=f"{window}-yr rolling mean")
    ax.set_xlabel("year"); ax.set_ylabel("new wars / year")
    ax.set_title("(a) COW war-onset rate 1816-2007")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- (b) Power spectrum of war onsets ---------------------------------
    ax = axes[0, 1]
    series = cnt.astype(float) - cnt.mean()
    freqs = np.fft.rfftfreq(len(series), d=1.0)
    psd = np.abs(np.fft.rfft(series)) ** 2
    # period = 1/freq
    periods = 1.0 / np.maximum(freqs[1:], 1e-6)
    ax.semilogx(periods, psd[1:], color="#2c4d8a")
    ax.axvline(50, color="grey", ls="--", lw=0.6,
               label="50 yr (Goldstein/Modelski)")
    ax.set_xlabel("period (years)"); ax.set_ylabel("power spectral density")
    ax.set_title("(b) Power spectrum of COW annual war onsets")
    ax.set_xlim(2, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')

    # --- (c) Capability concentration H_2(t) -----------------------------
    # H_2 = sum of top-2 squared CINC fractions among major powers
    ax = axes[1, 0]
    years = sorted(nmc.year.unique())
    top2 = []
    for y in years:
        cy = nmc[nmc.year == y].cinc.dropna()
        if len(cy) < 2: continue
        s = cy.sort_values(ascending=False).values
        if s.sum() == 0: continue
        f = s / s.sum()
        H2 = float(f[0] ** 2 + f[1] ** 2)
        top2.append((y, H2, f[0]))
    arr = np.array(top2)
    ax.plot(arr[:, 0], arr[:, 2], color="#2c4d8a", lw=1.2,
            label="leading-power share")
    ax.plot(arr[:, 0], arr[:, 1], color="#d62728", lw=1.2,
            label="top-2 Herfindahl $H_2$")
    ax.set_xlabel("year"); ax.set_ylabel("share / index")
    ax.set_title("(c) Concentration of capability (NMC v6.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- (d) Autocorrelation of A_max(t) in the MODEL -----------------------
    ax = axes[1, 1]
    sim = pickle.load(open("baseline_v4_L32_T1200_s2.pkl", "rb"))
    Amax = np.asarray(sim.largest_area, dtype=float)
    t = np.asarray(sim.times, dtype=float)
    # de-trend by detrending mean
    x = Amax - Amax.mean()
    # autocorr
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:] / (x.var() * np.arange(n, 0, -1))
    lag_t = t - t[0]
    ax.plot(lag_t[:n // 2], ac[:n // 2], color="#2ca02c", lw=1.2)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xlabel("lag (model time units)"); ax.set_ylabel("autocorrelation of $A_{\\max}(t)$")
    ax.set_title("(d) Model $A_{\\max}(t)$ autocorrelation (cycle signature)")
    ax.grid(True, alpha=0.2)
    # find first zero/minimum to estimate cycle period
    ac_seg = ac[:n // 2]
    first_zero = np.argmax(ac_seg < 0)
    if first_zero > 0:
        period_est = 4 * lag_t[first_zero]    # quarter wavelength -> full period
        ax.axvline(period_est, color='k', ls='--', lw=0.6,
                   label=f"cycle period ≈ {period_est:.0f} t.u.")
        ax.legend(fontsize=8)

    fig.suptitle("Cycles in the historical record and the model", fontsize=11, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_cycles.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    figure_fragmentation()
    figure_cycles()
