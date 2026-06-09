"""
Figures for v2 (Options A & B).

  Option A (selforg_v2_optionA.pdf): the multipolar<->bipolar transition and the
  precursor result -- systemic wars break out of bipolar (low N_eff) configs.
  Option B (selforg_v2_optionB.pdf): deliberation depth as a control parameter --
  how collective foresight reshapes war statistics (needs data_v2_sweep.npz).

  python3 make_figures_v2.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thucydides_selforg_v2 import Params, SelfOrgThucydidesV2

plt.rcParams.update({"figure.dpi": 120, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 0.9})
Wc = dict(rnd=0, big=7, sev=8, parity=9, frac=10, cross=11, xmid=12)


def ring_scatter(ax, x, S, bloc, title):
    th = 2 * np.pi * x
    # colour by bloc; singletons grey
    labels, counts = np.unique(bloc, return_counts=True)
    big = set(labels[counts >= 2])
    cmap = plt.cm.tab20
    cidx = {L: i for i, L in enumerate(sorted(big))}
    for i in range(len(x)):
        col = cmap(cidx[bloc[i]] % 20) if bloc[i] in big else "0.75"
        r = 1.0
        ax.scatter(th[i], r, s=8 + 50 * S[i] / (S.max() + 1e-9), color=col,
                   edgecolors="k", linewidths=0.2, zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 1.25); ax.set_yticks([]); ax.set_xticks([])


def option_A():
    p = Params(rounds=4000, seed=2)
    sim = SelfOrgThucydidesV2(p)
    h = sim.run(sample_every=5)
    t = np.array(h.t); neff = np.array(h.neff)
    w = np.array(h.wars, dtype=float); npre = np.array(h.neff_pre)
    # drop the initial transient (all-singleton start) from war statistics
    keep = w[:, Wc["rnd"]] > 200
    w, npre = w[keep], npre[keep]
    big = w[:, Wc["big"]] == 1
    # mark only the 20 most severe wars to keep panel (a) legible
    order = np.argsort(w[:, Wc["sev"]])[::-1]
    big_t = w[order[:20], Wc["rnd"]]

    fig = plt.figure(figsize=(8.0, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.42, wspace=0.32,
                          left=0.08, right=0.97, top=0.9, bottom=0.1)

    # (a) N_eff timeseries with systemic wars marked
    axa = fig.add_subplot(gs[0, :2])
    axa.plot(t, neff, color="C0", lw=0.8)
    axa.axhline(2, color="C3", ls=":", lw=0.8)
    for bt in big_t:
        axa.axvline(bt, color="k", lw=0.4, alpha=0.25)
    axa.set(xlabel="round", ylabel=r"$N_{\rm eff}$ blocs",
            title="(a) polarization over time (| = systemic war)")
    axa.set_ylim(1, min(8, neff.max() + 1))

    # (b) precursor: N_eff just before big vs small wars
    axb = fig.add_subplot(gs[0, 2])
    data = [npre[big], npre[~big]]
    parts = axb.violinplot(data, showmeans=True, widths=0.8)
    axb.set_xticks([1, 2]); axb.set_xticklabels(["systemic", "small"])
    axb.axhline(2, color="C3", ls=":", lw=0.8)
    axb.set_ylim(1, min(7, max(npre.max(), 4)))
    axb.set(ylabel=r"$N_{\rm eff}$ before war", title="(b) bipolar precursor")
    axb.text(0.5, 0.02,
             f"means {npre[big].mean():.2f} vs {npre[~big].mean():.2f}",
             transform=axb.transAxes, ha="center", fontsize=6.5)

    # (c,d) ring snapshots: a genuine multipolar phase (several regional blocs of
    # size>=3, after the transient) and a bipolar moment just before a systemic war
    def n_regional(k):
        b = h.bloc[k]; labs, cnts = np.unique(b, return_counts=True)
        return int((cnts >= 3).sum())
    mid = [k for k in range(len(t)) if t[k] > 200]
    k_multi = max(mid, key=n_regional)
    # pick a low-N_eff sample that sits just before a systemic war
    pre_idx = []
    for bt in big_t:
        k = np.searchsorted(t, bt) - 1
        if 0 <= k < len(neff):
            pre_idx.append(k)
    k_bip = pre_idx[int(np.argmin(neff[pre_idx]))] if pre_idx else int(np.argmin(neff))
    axc = fig.add_subplot(gs[1, 0], projection="polar")
    ring_scatter(axc, h.xpos[k_multi], h.S[k_multi], h.bloc[k_multi],
                 f"(c) multipolar  $N_{{\\rm eff}}$={neff[k_multi]:.1f}\n(round {t[k_multi]})")
    axd = fig.add_subplot(gs[1, 1], projection="polar")
    ring_scatter(axd, h.xpos[k_bip], h.S[k_bip], h.bloc[k_bip],
                 f"(d) bipolar pre-war  $N_{{\\rm eff}}$={neff[k_bip]:.1f}\n(round {t[k_bip]})")

    # (e) war severity vs pre-war N_eff
    axe = fig.add_subplot(gs[1, 2])
    axe.scatter(npre[~big], w[~big, Wc["sev"]], s=6, c="C7", alpha=0.5, label="small")
    axe.scatter(npre[big], w[big, Wc["sev"]], s=12, c="C3", alpha=0.8, label="systemic")
    axe.set(xlabel=r"$N_{\rm eff}$ before war", ylabel="severity",
            title="(e) severity vs polarization")
    axe.set_yscale("log"); axe.legend(fontsize=6)

    fig.suptitle("Option A: multipolar$\\to$bipolar transition precedes systemic war",
                 fontsize=10, y=0.97)
    fig.savefig("figs/selforg_v2_optionA.pdf")
    fig.savefig("figs/selforg_v2_optionA.png", dpi=150)
    print(f"optionA: Neff_pre systemic={npre[big].mean():.2f} small={npre[~big].mean():.2f}; "
          f"wrote figs/selforg_v2_optionA.{{pdf,png}}")


def option_B():
    try:
        d = np.load("data_v2_sweep.npz")
    except FileNotFoundError:
        print("optionB: run experiments_v2.py first (data_v2_sweep.npz missing)")
        return
    n = d["nvals"]
    fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.7))
    specs = [(ax[0], "rate", "systemic wars / 1000 rounds", "C3",
             "(a) more foresight $\\to$ more war"),
             (ax[1], "tau", r"mean inter-systemic-war time $\tau$", "C0",
             "(b) ... shorter peace"),
             (ax[2], "sev", "mean war severity", "C4",
             "(c) ... larger wars")]
    for a, key, lab, col, title in specs:
        m, s = d[key][:, 0], d[key][:, 1]
        a.errorbar(n, m, yerr=s, fmt="o-", ms=4, capsize=2, color=col)
        a.set_xscale("log", base=2)
        a.set(xlabel="deliberation depth $n$", ylabel=lab, title=title)
    fig.suptitle("Option B: with war-driven absorption, collective foresight is "
                 "DESTABILISING (better predators)", y=1.03, fontsize=9.5)
    fig.tight_layout()
    fig.savefig("figs/selforg_v2_optionB.pdf", bbox_inches="tight")
    fig.savefig("figs/selforg_v2_optionB.png", dpi=150, bbox_inches="tight")
    print("optionB: wrote figs/selforg_v2_optionB.{pdf,png}")


if __name__ == "__main__":
    option_A()
    option_B()
