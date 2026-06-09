# Stepwise progression

Build the Thucydides model one mechanism at a time, starting from pure growth,
adding **one** ingredient per step and watching what new structure appears.
Each step is a self-contained `stepN_*.py` that prints a few diagnostics and
writes a figure to `figs/`.

```bash
python3 step0_growth.py     # base model: resource-limited growth, N=10, no interactions
python3 step0_turnover.py   # analysis: why the leader always falls (rise-and-fall)
```

---

## Step 0 — pure growth (no wars / alliances / predictability)

The base model (your code, reproduced faithfully in `step0_growth.py`):

```
R       = max(1 - sum(S)/K, 0)            # shared-resource availability
dS_i    = gamma_i * R                     # additive, resource-limited growth
dgamma_i= lam_g (mu_g - gamma_i) dt       # OU rate (see note)
S_i    += dt*dS_i     + sig_s sqrt(S_i dt) xi    # demographic sqrt-S noise
gamma_i+= dt*dgamma_i + sig_g sqrt(dt) xi
S_i     = max(S_i, 0)
```
`N=10`, `K=100`, `lam_g=mu_g=1`, `sig_g=0.1`, `sig_s=1`. The **only** coupling
between states is the global resource `R`: everyone grows only while total
strength is below the ceiling `K`.

### What emerges
**Condensation — one (or a few) states take over.** Defaults, averaged over 8 seeds:

| quantity | value |
|---|---|
| largest state's share of total strength | **0.75** |
| Gini(final S) | 0.84 |
| survivors (S > 1) out of 10 | **1.9** |
| total strength | ≈ K (saturated) |

`R` depletes to ~0 by `t≈70`; thereafter total strength sits at the ceiling and
the dynamics are noise-dominated.

### Why (mechanism)
Once `R≈0` the drift `gamma·R` nearly vanishes, so each `S_i` is essentially a
**driftless `sqrt(S)` random walk** (Feller/CIR with no immigration), for which
`0` is absorbing: states wander and get absorbed at `0` one by one, while the
resource ceiling holds the total near `K`. This is a generic **fixation /
condensation** of conserved-total multiplicative-noise systems — a noise-driven
winner-take-all, *not* a consequence of any interaction. It is the natural
"someone takes over" baseline the richer model must then complicate.

### ...but the winner always falls (rise-and-fall)
The condensate is **dynamic**: the leader is overtaken ~55–65 times per run and no
state stays on top. Analytically, in the saturated regime ($R\approx0$) the
leader's strength is a *driftless $\sqrt S$ martingale* — a squared-Bessel
BESQ(0) process — which hits 0 almost surely, so dominance is intrinsically
temporary. Full derivations (Itô reduction to Bessel, the optional-stopping bound
$\Pr(\text{reach }a\text{ before }0)=S_0/a$, the additive-vs-multiplicative
drift/noise argument): see **[`ANALYSIS_step0.tex`](ANALYSIS_step0.tex)** (LaTeX,
canonical; `.md` is a plain-text summary) and `step0_turnover.py`.

### Note on the `gamma` update
The original `dgamma = lam_g(mu_g-gamma)dt` followed by `gamma += dt*dgamma`
multiplies the reversion drift by `dt` **twice** (`dt²`), so `gamma` barely
reverts to `mu_g=1` and instead random-walks from its start (mean 0). Kept
verbatim as the default (`as_written=True`) because it matches the observed
behaviour; pass `as_written=False` for the textbook OU update. It barely changes
the outcome (survivors 1.9 → 3.0), confirming the condensation is driven by the
`sqrt(S)` noise, not the rate dynamics. (`delta=0.1` is defined but unused in the
original — it becomes Step 1.)

---

---

## Step 1 — growth exponent alpha

Generalise the growth drift to `dS_i = gamma_i * R * S_i^alpha * dt` (+ sqrt-S
noise). `step1_alpha.py` sweeps alpha; full derivations in
**[`ANALYSIS_step1.tex`](ANALYSIS_step1.tex)**.

`alpha` tunes the **persistence** of dominance (inequality stays high throughout):

| alpha | regime | mean leader lifetime |
|---|---|---|
| 0 (additive, Step 0) | recurrent churn | ~4 |
| 1 (multiplicative) | marginal | ~11 |
| 2 (super-linear) | hegemonic lock-in | ~99 |

**Why — the relative dynamics.** The log-ratio of two states drifts as
`gamma R (S_i^{alpha-1} - S_j^{alpha-1})`, whose sign flips at **alpha = 1**:
- `alpha<1` (diminishing returns): leads *erode*, laggards catch up -> churn;
- `alpha=1` (constant returns): log-ratio is a martingale -> marginal;
- `alpha>1` (increasing returns): leads *amplify* -> a durable hegemon.

A second, distinct threshold `alpha=1/2` (drift vs noise of a *single* state)
governs absolute stability, not who leads. So **alpha tunes rise-and-fall from
fast, shallow, recurrent (alpha<1) to rare, deep, long hegemonic cycles
(alpha>1)**.

---

## Proposed roadmap (one ingredient per step)

Each step adds exactly one mechanism and asks *what new structure appears?*

- **Step 1 — growth exponent `alpha`.** *(done — see above)* `dS=gamma R S^alpha`;
  critical `alpha=1` separates churn from lock-in.
- **Step 1b — decay `delta`.** Turn on the unused `-delta·S` term. A genuine
  birth/death balance: does it sustain a *diversity* of powers (a steady-state
  Gamma/CIR distribution with turnover) instead of fixation? Sets the
  rise-and-fall timescale `~1/delta`.
- **Step 2 — minimal takeover (proto-war).** The simplest interaction: at a low
  rate a stronger state absorbs a weaker neighbour (strength transfer, no
  alliances, no decisions). Does this *accelerate* condensation or, with decay,
  give rise-and-fall cycles of dominant powers?
- **Step 3 — predictability.** States forecast the others' growth and choose
  *when* to attack (e.g. pre-empt a riser) rather than attacking at random.
  *Does foresight change who wins / the war timing?*
- **Step 4 — alliances.** Balancing coalitions against the strongest; war-driven
  absorption of the loser. *Does a bloc structure emerge?*

We freeze each step's parameters before moving on, so every later effect is
attributable to the ingredient just added.
