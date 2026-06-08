"""
Generate paper figures for the Thucydides-trap minimal model.

Produces (in /outputs/figs):
  fig1_snapshots.pdf  : Map evolution at 4 times.
  fig2_timeseries.pdf : Causal chain growth -> tension -> war (population observables).
  fig3_warsize.pdf    : War size distribution (size = total cells lost in a clustered conflict).
  fig4_phase.pdf      : Phase diagram in (w, c).
"""
import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict

from thucydides_model import run_baseline, Sim

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(OUTDIR, "figs")
os.makedirs(FIGDIR, exist_ok=True)


def state_color_map(owner_array, rng=None):
    """Map state ids to a colormap. Vacant=0 unused here (every cell owned)."""
    rng = rng or np.random.default_rng(7)
    ids = np.unique(owner_array)
    n = len(ids)
    # use a perceptually decent qualitative colormap, but recycle if too many states
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    extra = plt.cm.Set3(np.linspace(0, 1, 12))
    cycle = np.vstack([base, extra])
    cols = cycle[(np.arange(n)) % len(cycle)]
    rng.shuffle(cols)
    cmap = {int(s): tuple(cols[k]) for k, s in enumerate(ids)}
    return cmap


def render_map(ax, owner_array, title=""):
    cmap = state_color_map(owner_array)
    L = owner_array.shape[0]
    rgba = np.zeros((L, L, 4))
    for s, c in cmap.items():
        mask = owner_array == s
        rgba[mask] = c
    ax.imshow(rgba, interpolation='nearest', origin='lower')
    # draw boundaries between different owners
    o = owner_array
    bnd_v = (o != np.roll(o, -1, axis=1))
    bnd_h = (o != np.roll(o, -1, axis=0))
    ys, xs = np.where(bnd_v)
    for x, y in zip(xs, ys):
        ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color='black', lw=0.4)
    ys, xs = np.where(bnd_h)
    for x, y in zip(xs, ys):
        ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color='black', lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


# --------- run baseline once -------------------------------------------------

def get_baseline(L=32, T=1500.0, seed=0, force=False):
    cache = os.path.join(OUTDIR, f"baseline_v4_L{L}_T{int(T)}_s{seed}.pkl")
    if (not force) and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print(f"Running baseline L={L} T={T} seed={seed} ...", flush=True)
    t0 = time.time()
    sim = run_baseline(L=L, T=T, seed=seed,
                       snap_times=[0.001, T*0.15, T*0.4, T*0.7, T])
    print(f"  elapsed {time.time()-t0:.1f}s, N={len(sim.A)}, wars={len(sim.war_log)}, frags={len(sim.frag_log)}")
    with open(cache, "wb") as f:
        pickle.dump(sim, f)
    return sim


# --------- figure 1 ----------------------------------------------------------

def figure_snapshots(sim, fname):
    snaps = sim.snapshots
    times = sim.snap_times
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    for ax, owner, t in zip(axes, snaps, times):
        render_map(ax, owner, title=f"t = {t:.0f}")
    fig.suptitle("Map evolution: Voronoi initial → consolidation by conquest & fragmentation",
                 fontsize=10, y=0.98)
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# --------- figure 2 ----------------------------------------------------------

def _binned_rate(events_t, t_grid):
    """Count events per bin in t_grid edges -> return rate per unit time."""
    cnt, _ = np.histogram(events_t, bins=t_grid)
    dt = np.diff(t_grid)
    return cnt / dt


def figure_timeseries(sim, fname):
    t = np.asarray(sim.times)
    n = np.asarray(sim.n_states)
    Stot = np.asarray(sim.total_strength)
    Amax = np.asarray(sim.largest_area)
    n_states = np.asarray(sim.n_states)
    # War rate per dt
    if len(t) > 1:
        bins = np.linspace(t[0], t[-1], 200)
        bc = 0.5 * (bins[:-1] + bins[1:])
        war_t = np.array([w[0] for w in sim.war_log])
        war_rate = _binned_rate(war_t, bins)
    else:
        bc = []; war_rate = []
    # growth rate proxy: dStot/dt smoothed
    if len(t) > 5:
        S_b = np.interp(bc, t, Stot)
        N_b = np.interp(bc, t, n_states)
        Amax_b = np.interp(bc, t, Amax)
        dStot = np.gradient(S_b, bc)
    else:
        S_b = N_b = Amax_b = dStot = []

    fig, axes = plt.subplots(4, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(bc, dStot, color="#1f77b4")
    axes[0].set_ylabel(r"growth $\dot S_{\rm tot}$")
    axes[0].axhline(0, color='gray', lw=0.5)

    axes[1].plot(bc, S_b, color="#ff7f0e")
    axes[1].set_ylabel(r"total strength $\Sigma S_i$")

    axes[2].plot(bc, war_rate, color="#d62728")
    axes[2].set_ylabel(r"war rate $\dot W$")

    axes[3].plot(t, Amax, color="#2ca02c", label="largest state $A_{\\max}$")
    axes[3].set_ylabel("largest area $A_{\\max}$")
    axes[3].set_xlabel("time")
    fig.suptitle("Mutual growth $\\to$ high $\\min(S_i,S_j)$ $\\to$ war $\\to$ territorial reorganization", fontsize=10)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# --------- figure 3 ----------------------------------------------------------

def cluster_wars(war_log, gap=2.0):
    """Cluster a stream of (t, winner, loser, ...) events into 'conflicts'.

    A conflict groups consecutive events between the same unordered pair {i,j}
    with inter-event gap < `gap`. War 'size' is the number of events
    (i.e. cells transferred plus battles).
    """
    by_pair = defaultdict(list)
    for ev in war_log:
        t, w, l = ev[0], ev[1], ev[2]
        by_pair[frozenset((w, l))].append(t)
    sizes = []
    for key, ts in by_pair.items():
        ts.sort()
        # split into clusters
        chunk = [ts[0]]
        for x in ts[1:]:
            if x - chunk[-1] < gap:
                chunk.append(x)
            else:
                sizes.append(len(chunk))
                chunk = [x]
        sizes.append(len(chunk))
    return np.array(sizes)


def figure_warsize(sims, fname, gap=2.0):
    fig, ax = plt.subplots(figsize=(5, 4))
    for sim, label, color in sims:
        sizes = cluster_wars(sim.war_log, gap=gap)
        if len(sizes) == 0:
            continue
        # log-binned histogram (CCDF)
        srt = np.sort(sizes)
        ccdf = 1.0 - np.arange(len(srt)) / len(srt)
        ax.plot(srt, ccdf, '.-', ms=3, lw=0.7, color=color, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("war size  $s$  (events per conflict)")
    ax.set_ylabel("CCDF  $P(S \\geq s)$")
    ax.set_title("War size distribution (clustered border conflicts)")
    # power law guide
    xx = np.array([2, 200])
    ax.plot(xx, 0.5 * xx ** -1.0, 'k--', lw=0.7, label=r"slope $-1$ (guide)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# --------- figure 4 ----------------------------------------------------------

def figure_phase(fname, ws, cs, L=28, T=400.0, seed=0):
    """Sweep a 2D grid in (w, c). For each, run short simulation and record:
       - <Amax> (mean largest state size in late half) -> indicates 'great-power' formation
    """
    cache = os.path.join(OUTDIR, f"phase_v4_L{L}_T{int(T)}.npy")
    if os.path.exists(cache):
        Amax = np.load(cache)
    else:
        Amax = np.zeros((len(ws), len(cs)))
        for i, w in enumerate(ws):
            for j, c in enumerate(cs):
                t0 = time.time()
                sim = run_baseline(L=L, T=T, w=w, c=c, seed=seed, n_init=30)
                second_half = np.asarray(sim.largest_area)[len(sim.times) // 2:]
                Amax[i, j] = second_half.mean() if len(second_half) else 0
                print(f"  w={w:.1e} c={c:.1e}  <Amax>={Amax[i,j]:.0f}  ({time.time()-t0:.1f}s)")
        np.save(cache, Amax)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Amax / (L * L), origin='lower', aspect='auto',
                   extent=[np.log10(cs[0]), np.log10(cs[-1]),
                           np.log10(ws[0]), np.log10(ws[-1])],
                   cmap='viridis')
    ax.set_xlabel(r"$\log_{10}\,c$  (fragility prefactor)")
    ax.set_ylabel(r"$\log_{10}\,w$  (war rate prefactor)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\langle A_{\rm max}\rangle / L^2$")
    ax.set_title("Phase diagram: largest-state fraction")
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# --------- main --------------------------------------------------------------

def main():
    sim = get_baseline(L=32, T=1500.0, seed=2)

    figure_snapshots(sim, os.path.join(FIGDIR, "fig1_snapshots.pdf"))
    figure_timeseries(sim, os.path.join(FIGDIR, "fig2_timeseries.pdf"))

    # Multi-seed war size distribution (one extra run for variability)
    sim2 = get_baseline(L=32, T=1500.0, seed=3)
    sim3 = get_baseline(L=32, T=1500.0, seed=5)
    figure_warsize([(sim, "seed 1", "#1f77b4"),
                    (sim2, "seed 2", "#ff7f0e"),
                    (sim3, "seed 3", "#2ca02c")],
                   os.path.join(FIGDIR, "fig3_warsize.pdf"))

    ws = np.array([3e-4, 1e-3, 3e-3, 1e-2])
    cs = np.array([1e-5, 5e-5, 2e-4, 1e-3])
    figure_phase(os.path.join(FIGDIR, "fig4_phase.pdf"), ws, cs, L=22, T=250.0)


if __name__ == "__main__":
    main()
