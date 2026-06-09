import numpy as np
from thucydides_selforg_v2 import Params, SelfOrgThucydidesV2

def init_depth(s, val):
    s.ndepth[:] = val
    return s

# (i) depth interior optimum: start low vs high
for init in [2, 40]:
    p = Params(rounds=4000, seed=3)
    s = SelfOrgThucydidesV2(p); s.ndepth[:] = init
    h = s.run()
    print(f"depth init {init:>2} -> {s.ndepth.mean():5.1f}  (median {np.median(s.ndepth):.0f})")

# (ii) Option-A precursor + full diagnostics, 3 seeds
print("\nseed | gini neffμ neffmx wars big | neff_pre big/small | cross big/small | depth")
for seed in [0,1,2]:
    p = Params(rounds=4000, seed=seed)
    s = SelfOrgThucydidesV2(p); d0=s.ndepth.mean(); h=s.run()
    w = np.array(h.wars, dtype=float); npre = np.array(h.neff_pre)
    big = w[:,7]==1
    nb_big = npre[big].mean() if big.sum() else float('nan')
    nb_sm  = npre[~big].mean() if (~big).sum() else float('nan')
    print(f"  {seed}  | {h.gini[-1]:.2f} {np.mean(h.neff):5.2f} {max(h.neff):5.0f} "
          f"{len(w):4d} {int(big.sum()):3d} | {nb_big:5.2f} / {nb_sm:5.2f} | "
          f"{w[big,11].mean():.2f} / {w[~big,11].mean():.2f} | {d0:.0f}->{s.ndepth.mean():.0f}")
