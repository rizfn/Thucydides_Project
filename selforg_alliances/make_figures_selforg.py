"""
Figures + summary statistics for the self-organising Thucydides model.

    python3 make_figures_selforg.py

Writes selforg_overview.{pdf,png} into figs/ and prints the key statistics
used in the note.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thucydides_selforg import Params, SelfOrgThucydides

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 6.5,
    "lines.linewidth": 0.9, "axes.grid": True, "grid.alpha": 0.25,
})

W = {  # war-tuple field indices
    "rnd": 0, "nA": 1, "nB": 2, "SA": 3, "SB": 4, "winA": 5,
    "npart": 6, "big": 7, "sev": 8, "parity": 9, "frac": 10, "cross": 11,
}


def run(seed=0, rounds=4000):
    p = Params(seed=seed, rounds=rounds)
    sim = SelfOrgThucydides(p)
    h = sim.run(sample_every=10)
    return p, sim, h


def war_array(h):
    return np.array(h.wars, dtype=float) if h.wars else np.zeros((0, 12))


# --------------------------------------------------------------------------- #
def main():
    p, sim, h = run(seed=0)
    t = np.array(h.t)
    S = np.array(h.S)                       # (T, N)
    wars = war_array(h)
    # mark only the most severe (genuinely systemic) wars to avoid a gray wash
    if len(wars):
        order = np.argsort(wars[:, W["sev"]])[::-1]
        big_t = wars[order[:15], W["rnd"]]
    else:
        big_t = []

    # ---- statistics --------------------------------------------------------
    acc0 = sim_acc0  # filled below via closure-free approach
    print("=" * 66)
    print(f"rounds={p.rounds}  states N={p.N}  wars={len(wars)}  "
          f"big(systemic)={int(wars[:, W['big']].sum()) if len(wars) else 0}")
    if len(wars):
        big = wars[:, W["big"]] == 1
        sev = wars[:, W["sev"]]
        par = wars[:, W["parity"]]
        cr = wars[:, W["cross"]] == 1
        print(f"belligerent parity   big={par[big].mean():.2f}  "
              f"small={par[~big].mean():.2f}")
        print(f"crossover fraction   big={cr[big].mean():.2f}  "
              f"small={cr[~big].mean():.2f}")
        print(f"severity             big={sev[big].mean():.1f}  "
              f"small={sev[~big].mean():.1f}")
    print(f"neff (final)={h.neff[-1]:.2f}  balance(final)={h.balance[-1]:.2f}  "
          f"Gini(final)={h.gini[-1]:.2f}")
    print(f"mean predictive accuracy {h.mean_acc[0]:.3f} -> {h.mean_acc[-1]:.3f}")
    print("=" * 66)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 7.2))

    # (a) strength trajectories, top-12 by peak strength
    peak = S.max(0)
    top = np.argsort(peak)[-12:]
    for k in top:
        ax[0, 0].plot(t, S[:, k], lw=0.7)
    for bt in big_t:
        ax[0, 0].axvline(bt, color="k", lw=0.4, alpha=0.25)
    ax[0, 0].set(xlabel="round", ylabel="strength $S_i$",
                 title="(a) rise & fall of powers (| = systemic war)")

    # (b) polarization dynamics
    ax[0, 1].plot(t, h.neff, color="C0", label=r"$N_{\rm eff}$ blocs")
    ax[0, 1].plot(t, h.balance, color="C3", label="bloc balance")
    for bt in big_t:
        ax[0, 1].axvline(bt, color="k", lw=0.4, alpha=0.25)
    ax[0, 1].axhline(2, color="C0", ls=":", lw=0.6)
    ax[0, 1].set(xlabel="round", title="(b) alliance polarization")
    ax[0, 1].legend(loc="upper right")

    # (c) war-severity CCDF (heavy tail?)
    if len(wars):
        sev = np.sort(wars[:, W["sev"]])
        ccdf = 1.0 - np.arange(sev.size) / sev.size
        ax[1, 0].loglog(sev, ccdf, "o", ms=2.5)
    ax[1, 0].set(xlabel="war severity (strength destroyed)",
                 ylabel="P(>severity)", title="(c) war-size distribution")

    # (d) Thucydides trigger: parity vs severity, colour = crossover
    if len(wars):
        cr = wars[:, W["cross"]] == 1
        ax[1, 1].scatter(wars[~cr, W["parity"]], wars[~cr, W["sev"]],
                         s=8, c="C7", label="no crossover", alpha=0.6)
        ax[1, 1].scatter(wars[cr, W["parity"]], wars[cr, W["sev"]],
                         s=10, c="C3", label="proj. crossover", alpha=0.8)
    ax[1, 1].set(xlabel="belligerent parity $\\min S/\\max S$",
                 ylabel="severity", title="(d) Thucydides trigger")
    ax[1, 1].legend(loc="upper left")

    # (e) selection on predictability
    ax[2, 0].hist(acc_init, bins=20, range=(0, 1), alpha=0.5,
                  label=f"initial (mean {acc_init.mean():.2f})", color="C7")
    ax[2, 0].hist(sim.acc, bins=20, range=(0, 1), alpha=0.6,
                  label=f"final (mean {sim.acc.mean():.2f})", color="C2")
    ax[2, 0].set(xlabel="predictive accuracy $a_i$", ylabel="count",
                 title="(e) selection for predictability")
    ax[2, 0].legend()

    # (f) inequality + total power
    ax[2, 1].plot(t, h.gini, color="C4", label="Gini($S$)")
    axb = ax[2, 1].twinx()
    axb.plot(t, S.sum(1), color="C1", lw=0.7, label="total power")
    axb.set_ylabel("total power $\\sum S_i$", color="C1")
    ax[2, 1].set(xlabel="round", ylabel="Gini($S$)", title="(f) inequality")
    ax[2, 1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("figs/selforg_overview.pdf")
    fig.savefig("figs/selforg_overview.png", dpi=150)
    print("wrote figs/selforg_overview.pdf and .png")


# capture the initial accuracy distribution (before selection) ---------------
_p0 = Params(seed=0, rounds=4000)
_s0 = SelfOrgThucydides(_p0)
acc_init = _s0.acc.copy()
sim_acc0 = acc_init

if __name__ == "__main__":
    main()
