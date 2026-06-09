"""
Stepwise progression -- STEP 0: pure growth, no wars / alliances / predictability
=================================================================================

Faithful reproduction of the base model:

    R       = max(1 - sum(S)/K, 0)          shared-resource availability
    dgamma  = lam_g (mu_g - gamma) dt        (Ornstein-Uhlenbeck rate)
    dS      = gamma * R                       additive, resource-limited growth
    gamma  += dt*dgamma + sig_g sqrt(dt) xi
    S      += dt*dS    + sig_s sqrt(S dt) xi  (demographic sqrt-S noise)
    S       = max(S, 0)

N=10 states.  The only coupling is the global resource R: states grow (dS>0)
only while total strength is below the ceiling K.  From this alone, structure
emerges -- typically one (or a few) states take over.

NOTE ON THE gamma UPDATE.  The original code computes `dgamma = lam_g(mu_g-gamma)dt`
and then `gamma += dt*dgamma + ...`, so the mean-reversion drift is multiplied by
dt TWICE (-> dt^2).  With dt=0.01 the reversion is ~1e4x too weak, so gamma is
effectively a driftless random walk away from its start (mean 0), NOT pulled to
mu_g=1.  This is kept here verbatim (`as_written=True`) because it is what
produces the "one takes over" behaviour -- states whose gamma random-walks
positive accumulate strength, those that go negative decay to 0.  Set
`as_written=False` to use the textbook OU update (reversion to mu_g) and compare.
(`delta` is defined but unused in the original; kept for reference.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate(N=10, tmax=200.0, ts=0.1, dt=0.01, K=None, delta=0.1,
             lam_g=1.0, mu_g=1.0, sig_g=0.1, sig_s=1.0,
             as_written=True, seed=0):
    rng = np.random.default_rng(seed)
    if K is None:
        K = N * 10.0
    S = np.maximum(rng.normal(1.0, 1.0, size=N), 0.0)
    gamma = rng.normal(0.0, 0.1, size=N)

    nrec = int(tmax / ts) + 1
    t_s = np.zeros(nrec); R_s = np.zeros(nrec)
    S_s = np.zeros((N, nrec)); g_s = np.zeros((N, nrec))
    t, click = 0.0, 0
    while t < tmax:
        R = max(1.0 - S.sum() / K, 0.0)
        dS = gamma * R
        if as_written:
            dgamma = lam_g * (mu_g - gamma) * dt          # verbatim (extra *dt below)
            gamma += dt * dgamma + sig_g * np.sqrt(dt) * rng.normal(size=N)
        else:
            gamma += lam_g * (mu_g - gamma) * dt + sig_g * np.sqrt(dt) * rng.normal(size=N)
        S += dt * dS + sig_s * np.sqrt(S * dt) * rng.normal(size=N)
        S = np.maximum(S, 0.0)
        t += dt
        if t > ts * click and click < nrec:
            t_s[click] = t; R_s[click] = R
            S_s[:, click] = S; g_s[:, click] = gamma
            click += 1
    sl = slice(0, click)
    return t_s[sl], R_s[sl], S_s[:, sl], g_s[:, sl]


def _gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return (n + 1 - 2 * np.cumsum(x).sum() / x.sum()) / n


def main():
    t, R, S, gamma = simulate(seed=0)
    N = S.shape[0]
    Sf = S[:, -1]
    print(f"final total strength = {Sf.sum():.1f}  (K = {N*10})")
    print(f"largest share        = {Sf.max()/Sf.sum():.2f}")
    print(f"Gini(final S)        = {_gini(Sf):.2f}")
    print(f"# states with S>1    = {(Sf > 1).sum()} / {N}")

    fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.2), sharex=True)
    ax[0].plot(t, R, color="C2"); ax[0].set_ylabel("resource $R$")
    ax[0].set_title("Step 0: pure resource-limited growth ($N=10$, no wars)")
    for i in range(N):
        ax[1].plot(t, S[i], lw=0.9)
    ax[1].set_ylabel("strength $S_i$")
    for i in range(N):
        ax[2].plot(t, gamma[i], lw=0.9)
    ax[2].axhline(0, color="k", lw=0.5, ls=":")
    ax[2].set_ylabel(r"growth rate $\gamma_i$"); ax[2].set_xlabel("time")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("figs/step0_growth.png", dpi=150)
    fig.savefig("figs/step0_growth.pdf")
    print("wrote figs/step0_growth.{png,pdf}")


if __name__ == "__main__":
    main()
