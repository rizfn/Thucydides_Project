"""Robustness aggregate + two mechanism controls. Prints numbers used in the note."""
import numpy as np
from dataclasses import replace
from thucydides_selforg import Params, SelfOrgThucydides

W = dict(big=7, sev=8, parity=9, frac=10, cross=11)


def one(p):
    s = SelfOrgThucydides(p); a0 = s.acc.mean(); h = s.run()
    w = np.array(h.wars); big = w[:, W["big"]] == 1
    return dict(acc0=a0, acc1=s.acc.mean(), neff=h.neff[-1], bal=h.balance[-1],
                gini=h.gini[-1], nwar=len(w), nbig=int(big.sum()),
                cb=w[big, W["cross"]].mean(), cs=w[~big, W["cross"]].mean(),
                pred=s.n_predation)


def agg(rows, k):
    v = np.array([r[k] for r in rows]); return v.mean(), v.std()


seeds = range(8)
base = [one(Params(seed=s, rounds=4000)) for s in seeds]
print("=== BASELINE (8 seeds, mean +/- sd) ===")
for k, lab in [("acc0", "init accuracy"), ("acc1", "final accuracy"),
               ("neff", "N_eff blocs"), ("bal", "bloc balance"),
               ("gini", "Gini(S)"), ("nwar", "n wars"), ("nbig", "n systemic"),
               ("cb", "crossover|big"), ("cs", "crossover|small"),
               ("pred", "predations")]:
    m, sd = agg(base, k); print(f"  {lab:18s} {m:8.3f} +/- {sd:.3f}")

# control 1: kappa=0 -> no rising-challenger trigger (imminence frozen at 1/2)
c1 = [one(Params(seed=s, rounds=4000, kappa=0.0)) for s in seeds]
print("\n=== CONTROL kappa=0 (Thucydides trigger OFF) ===")
m, _ = agg(c1, "cb"); m2, _ = agg(c1, "cs")
print(f"  crossover|big={m:.3f}  crossover|small={m2:.3f}  "
      f"(baseline gap {agg(base,'cb')[0]-agg(base,'cs')[0]:+.3f} -> "
      f"control gap {m-m2:+.3f})")

# control 2: predation independent of accuracy -> selection should vanish
import thucydides_selforg as M
src = M.SelfOrgThucydides._predation
def neutral_predation(self, rnd):
    p = self.p; rl = p.substeps * p.dt
    labels = np.unique(self.bloc)
    bstr = {L: self.S[self.bloc == L].sum() for L in labels}
    sL = max(bstr, key=bstr.get); ile = np.where(self.bloc == sL)[0]
    lead = ile[np.argmax(self.S[ile])]
    for i in range(p.N):
        if self.bloc[i] != i: continue
        threat = max((v for L, v in bstr.items() if L != i), default=0.0)
        if threat < p.prey_ratio * self.S[i]: continue
        if self.rng.random() < 1.0 - np.exp(-p.prey_rate * rl):   # no (1-acc) factor
            self.S[lead] += p.spoils * self.S[i]; self.S[i] = 0.0; self.n_predation += 1
M.SelfOrgThucydides._predation = neutral_predation
c2 = [one(Params(seed=s, rounds=4000)) for s in seeds]
M.SelfOrgThucydides._predation = src
print("\n=== CONTROL accuracy-neutral predation (selection channel OFF) ===")
print(f"  accuracy {agg(c2,'acc0')[0]:.3f} -> {agg(c2,'acc1')[0]:.3f}  "
      f"(baseline {agg(base,'acc0')[0]:.3f} -> {agg(base,'acc1')[0]:.3f})")
