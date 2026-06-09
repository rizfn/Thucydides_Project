"""
Option B: deliberation depth as a control parameter.
Sweep a frozen, homogeneous depth n across the whole population and measure how
collective foresight reshapes the war statistics:
  - systemic-war rate and mean inter-systemic-war time tau
  - mean war severity (and of systemic wars)
  - polarization N_eff, inequality Gini
Also a self-organising (evolving-depth) run for comparison.
Saves results to data_v2_sweep.npz and prints a table.
"""
import numpy as np
from thucydides_selforg_v2 import Params, SelfOrgThucydidesV2

ROUNDS = 2500
SEEDS = [0, 1, 2]
NVALS = [1, 2, 4, 8, 16, 32, 64]


def stats(h):
    w = np.array(h.wars, dtype=float)
    if len(w) == 0:
        return dict(rate=0, tau=np.nan, sev=np.nan, sev_big=np.nan,
                    nbig=0, neff=np.mean(h.neff), gini=h.gini[-1])
    big = w[:, 7] == 1
    big_rounds = np.sort(w[big, 0])
    tau = np.mean(np.diff(big_rounds)) if big_rounds.size > 1 else np.nan
    return dict(rate=1000.0 * big.sum() / ROUNDS,           # systemic wars / 1000 rounds
                tau=tau,
                sev=w[:, 8].mean(),
                sev_big=w[big, 8].mean() if big.sum() else np.nan,
                nbig=int(big.sum()),
                neff=np.mean(h.neff), gini=h.gini[-1])


def run(n, seed):
    p = Params(rounds=ROUNDS, seed=seed, evolve_depth=False, ndepth_fixed=n)
    return stats(SelfOrgThucydidesV2(p).run())


def agg(rows, k):
    v = np.array([r[k] for r in rows], float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if v.size else (np.nan, np.nan)


print(f"frozen depth sweep ({len(SEEDS)} seeds, {ROUNDS} rounds)\n")
print(f"{'n':>4} | {'sys.rate':>9} {'tau':>7} {'sev_all':>8} {'sev_big':>8} "
      f"{'Neff':>5} {'Gini':>5}")
table = {}
for n in NVALS:
    rows = [run(n, s) for s in SEEDS]
    table[n] = {k: agg(rows, k) for k in
                ["rate", "tau", "sev", "sev_big", "neff", "gini"]}
    t = table[n]
    print(f"{n:>4} | {t['rate'][0]:>9.2f} {t['tau'][0]:>7.0f} {t['sev'][0]:>8.1f} "
          f"{t['sev_big'][0]:>8.1f} {t['neff'][0]:>5.2f} {t['gini'][0]:>5.2f}")

np.savez("data_v2_sweep.npz",
         nvals=np.array(NVALS),
         rate=np.array([table[n]["rate"] for n in NVALS]),
         tau=np.array([table[n]["tau"] for n in NVALS]),
         sev=np.array([table[n]["sev"] for n in NVALS]),
         sev_big=np.array([table[n]["sev_big"] for n in NVALS]),
         neff=np.array([table[n]["neff"] for n in NVALS]),
         gini=np.array([table[n]["gini"] for n in NVALS]))
print("\nsaved data_v2_sweep.npz")
