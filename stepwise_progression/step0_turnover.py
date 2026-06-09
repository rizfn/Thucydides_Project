"""
Step 0 analysis: why the leader always falls (rise-and-fall, no permanent winner).

(a) the base (additive) model: the instantaneous leader (black) is repeatedly
    overtaken -- no permanent winner.
(b) the analytic reason, measured: in the saturated regime (R~0) the drift of S
    is ~0 at every size (a martingale), while the demographic noise grows as
    sqrt(S).  So a large leader is a driftless random walk with the LARGEST
    absolute volatility -> it must make large downward excursions and be
    overtaken.  This is a squared-Bessel BESQ(0) process, which hits 0 a.s.

Writes figs/step0_turnover.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step0_growth import simulate


# ---- (a) one trajectory, leader highlighted -------------------------------
t, R, S, g = simulate(tmax=200, seed=1)
N = S.shape[0]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
for i in range(N):
    ax[0].plot(t, S[i], lw=0.7, alpha=0.55)
lead = np.argmax(S, 0)
lead_S = S[lead, np.arange(len(t))]
ax[0].plot(t, lead_S, color="k", lw=1.8, label="current leader")
chg = np.where(np.diff(lead) != 0)[0]
ax[0].plot(t[chg], lead_S[chg], "v", color="r", ms=3, label=f"{chg.size} lead changes")
ax[0].set(xlabel="time", ylabel="strength $S_i$",
          title="(a) the leader (black) keeps being overtaken")
ax[0].legend(fontsize=7, loc="upper left"); ax[0].grid(alpha=0.25)

# ---- (b) conditional drift & volatility of S, saturated regime ------------
# pool (S, dS-over-one-record-step) from many seeds, after saturation (t>70)
Sv, dSv = [], []
for s in range(40):
    t, R, S, g = simulate(tmax=200, seed=s)
    sat = t[:-1] > 70
    Sv.append(S[:, :-1][:, sat].ravel())
    dSv.append(np.diff(S, axis=1)[:, sat].ravel())
Sv = np.concatenate(Sv); dSv = np.concatenate(dSv)
bins = np.linspace(0, np.percentile(Sv, 99), 16)
idx = np.digitize(Sv, bins)
mid = 0.5 * (bins[:-1] + bins[1:])
mean_dS = np.array([dSv[idx == k + 1].mean() if (idx == k + 1).any() else np.nan
                    for k in range(len(mid))])
std_dS = np.array([dSv[idx == k + 1].std() if (idx == k + 1).any() else np.nan
                   for k in range(len(mid))])

axb = ax[1]
axb.axhline(0, color="0.6", lw=0.8, ls=":")
axb.plot(mid, mean_dS, "o-", color="C3", ms=4, label=r"drift  $\langle\Delta S\rangle\approx0$")
axb.plot(mid, std_dS, "s-", color="C0", ms=4, label=r"volatility  $\mathrm{std}(\Delta S)$")
axb.plot(mid, std_dS[1] / np.sqrt(mid[1]) * np.sqrt(mid), "--", color="C0",
         lw=1, alpha=0.7, label=r"$\propto\sqrt{S}$ (Bessel)")
axb.set(xlabel="strength $S$", ylabel=r"increment over $\Delta t=0.1$",
        title="(b) saturated regime: zero drift, $\\sqrt{S}$ noise")
axb.legend(fontsize=7); axb.grid(alpha=0.25)

fig.suptitle("Step 0: the leader is a driftless $\\sqrt{S}$ martingale "
             "$\\Rightarrow$ it must fall", y=1.02)
fig.tight_layout()
fig.savefig("figs/step0_turnover.png", dpi=150, bbox_inches="tight")
fig.savefig("figs/step0_turnover.pdf", bbox_inches="tight")
print(f"leader overtaken {chg.size} times in one run; "
      f"drift~0 (max |mean dS| over bins = {np.nanmax(np.abs(mean_dS)):.3f}), "
      f"vol grows ~sqrt(S)")
print("wrote figs/step0_turnover.{png,pdf}")
