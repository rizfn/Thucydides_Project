"""
Stepwise progression -- STEP 1: generalised growth exponent alpha
=================================================================

Growth drift is now  dS_i = gamma_i * R * S_i^alpha * dt  (+ demographic noise),

    R       = max(1 - sum(S)/K, 0)
    dS_i    = gamma_i R S_i^alpha dt + sig_s sqrt(S_i) dW_i
    dgamma_i= lam_g (mu_g - gamma_i) dt + sig_g sqrt(dt) xi      (OU, mean mu_g>0)
    S_i     = max(S_i, 0)

alpha interpolates between the two Step-0 limits:
    alpha = 0  -> additive, size-independent growth  (Step 0: churn, BESQ(0))
    alpha = 1  -> multiplicative growth              (lock-in)

The drift-to-noise ratio at size S is  gamma R S^alpha / (sig sqrt(S)) ~ S^{alpha-1/2},
so alpha = 1/2 is the scale-invariant (critical) exponent: below it large states
are noise-dominated (unstable, churn), above it drift-dominated (stable, lock-in).
gamma uses a positive-mean OU here so growth is sustained and alpha is isolated.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate(alpha=1.0, N=10, tmax=200.0, ts=0.1, dt=0.01, K=None,
             lam_g=1.0, mu_g=1.0, sig_g=0.1, sig_s=1.0, seed=0):
    rng = np.random.default_rng(seed)
    if K is None:
        K = N * 10.0
    S = np.maximum(rng.normal(1.0, 1.0, N), 0.0)
    gamma = mu_g + rng.normal(0.0, 0.1, N)
    nrec = int(tmax / ts) + 1
    t_s = np.zeros(nrec); R_s = np.zeros(nrec); S_s = np.zeros((N, nrec))
    t, c = 0.0, 0
    while t < tmax:
        R = max(1.0 - S.sum() / K, 0.0)
        drift = gamma * R * np.power(S, alpha)
        gamma += lam_g * (mu_g - gamma) * dt + sig_g * np.sqrt(dt) * rng.normal(size=N)
        S += dt * drift + sig_s * np.sqrt(S * dt) * rng.normal(size=N)
        S = np.maximum(S, 0.0); t += dt
        if t > ts * c and c < nrec:
            t_s[c] = t; R_s[c] = R; S_s[:, c] = S; c += 1
    return t_s[:c], R_s[:c], S_s[:, :c]


def _gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return (n + 1 - 2 * np.cumsum(x).sum() / x.sum()) / n


def metrics(S, t):
    """Leadership turnover, inequality, and mean leader lifetime."""
    lead = np.argmax(S, 0)
    changes = int((np.diff(lead) != 0).sum())
    # run-lengths of a constant leader -> lifetimes (in time units)
    brk = np.where(np.diff(lead) != 0)[0]
    seg = np.diff(np.concatenate(([0], brk + 1, [len(lead)])))
    life = seg.mean() * (t[1] - t[0])
    Sf = S[:, -1]
    half = len(lead) // 2
    hold = float(np.mean(lead[half:] == lead[half]))
    return dict(changes=changes, life=life, gini=_gini(Sf),
                top=Sf.max() / max(Sf.sum(), 1e-9), hold=hold)


def sweep(alphas, seeds=range(12), **kw):
    out = {}
    for a in alphas:
        rows = []
        for s in seeds:
            t, R, S = simulate(alpha=a, seed=s, **kw)
            rows.append(metrics(S, t))
        out[a] = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    return out


def main():
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    res = sweep(alphas, seeds=range(20))
    print(f"{'alpha':>6} {'lead_changes':>12} {'leader_life':>11} "
          f"{'hold_2ndhalf':>12} {'Gini':>6} {'top_share':>9}")
    for a in alphas:
        r = res[a]
        print(f"{a:>6.2f} {r['changes']:>12.0f} {r['life']:>11.1f} "
              f"{r['hold']:>12.2f} {r['gini']:>6.2f} {r['top']:>9.2f}")

    # ---- figure ----
    fig = plt.figure(figsize=(9.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)
    # (top) example trajectories at alpha = 0 (churn), 1 (marginal), 2 (lock-in)
    for j, (a, lab) in enumerate([(0.0, "churn"), (1.0, "marginal"), (2.0, "lock-in")]):
        ax = fig.add_subplot(gs[0, j])
        t, R, S = simulate(alpha=a, seed=1)
        for i in range(S.shape[0]):
            ax.plot(t, S[i], lw=0.7, alpha=0.6)
        lead = np.argmax(S, 0)
        ax.plot(t, S[lead, np.arange(len(t))], color="k", lw=1.5)
        ax.set(title=fr"$\alpha={a}$ ({lab})", xlabel="time")
        if j == 0:
            ax.set_ylabel("strength $S_i$")
        ax.grid(alpha=0.25)
    # (bottom) sweep metrics vs alpha; critical alpha=1 (relative lock-in)
    A = np.array(alphas)
    def crit(ax):
        ax.axvline(1.0, color="C3", ls="--", lw=1.2)
        ax.axvline(0.5, color="0.6", ls=":", lw=0.9)
    axa = fig.add_subplot(gs[1, 0])
    axa.plot(A, [res[a]['changes'] for a in alphas], "o-", color="C0"); crit(axa)
    axa.set(xlabel=r"$\alpha$", ylabel="leadership changes", title="(a) turnover")
    axb = fig.add_subplot(gs[1, 1])
    axb.plot(A, [res[a]['life'] for a in alphas], "o-", color="C2"); crit(axb)
    axb.set(xlabel=r"$\alpha$", ylabel="mean leader lifetime",
            title=r"(b) hegemon longevity ($\alpha\!=\!1$ dashed)")
    axc = fig.add_subplot(gs[1, 2])
    axc.plot(A, [res[a]['gini'] for a in alphas], "o-", color="C4", label="Gini")
    axc.plot(A, [res[a]['top'] for a in alphas], "s-", color="C1", label="top share")
    crit(axc)
    axc.set(xlabel=r"$\alpha$", title="(c) inequality"); axc.legend(fontsize=7)
    fig.suptitle(r"Step 1: growth exponent $\alpha$ tunes recurrent churn $\to$ "
                 r"hegemonic lock-in (critical $\alpha=1$)", y=0.98)
    fig.savefig("figs/step1_alpha.png", dpi=150, bbox_inches="tight")
    fig.savefig("figs/step1_alpha.pdf", bbox_inches="tight")
    print("wrote figs/step1_alpha.{png,pdf}")


if __name__ == "__main__":
    main()
