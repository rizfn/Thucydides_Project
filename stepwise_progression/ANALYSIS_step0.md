# Step 0 analysis — why the leader always falls

**Question.** In the base model the dominant state isn't a permanent winner — it
rises and is eventually eliminated, every run. Is that real, and can it be
explained analytically?

**Short answer.** Yes, it is real and robust, and it follows from one fact: in
the saturated regime the leading state's strength is a *driftless martingale with
√S noise* — a squared-Bessel process of dimension 0 — which is recurrent and hits
0 almost surely. No state can stay on top.

---

## 1. It is real (measurement)

Over a $t=200$ run the instantaneous leader is overtaken **~55–65 times**;
**6–10** different states hold the lead at some point; the mid-run leader is still
leading at the end only **~42%** of the time. (`step0_turnover.py`, panel a.)

## 2. The three ingredients

1. **Fixed pie.** $R=\max(1-\sum S/K,0)$ pins the total near $K$ once the resource
   is used up (by $t\approx70$). Leadership is then zero-sum: one state can only
   grow by others shrinking.
2. **Vanishing drift on top.** The growth drift is $\dot S_i=\gamma_i R$. Once
   saturated, $R\approx0$, so the drift is $\approx0$ **for every state,
   regardless of size**. Panel (b) confirms $\langle\Delta S\rangle\approx0$
   across all $S$.
3. **Demographic noise $\sigma\sqrt{S}$.** The absolute fluctuation grows as
   $\sqrt{S}$ (panel b, blue $\propto\sqrt S$). The leader, being largest, has the
   *largest* absolute wobble.

## 3. The analytic core: a Bessel martingale

In the saturated regime the leader obeys, to leading order,
$$
dS \;\approx\; \sigma\sqrt{S}\,dW \qquad(\sigma=\texttt{sig\_s}),
$$
a nonnegative **martingale** (zero drift). Substituting $Y=2\sqrt{S}$ and applying
Itô ($Y'=S^{-1/2},\,Y''=-\tfrac12 S^{-3/2},\,(dS)^2=\sigma^2 S\,dt$),
$$
dY \;=\; \sigma\,dW \;-\; \frac{\sigma^2}{2Y}\,dt .
$$
This is a **Bessel process of dimension $\delta=0$** (compare
$dY=\frac{\delta-1}{2Y}dt+dW$, giving $\delta-1=-1\Rightarrow\delta=0$).
Equivalently $S=Y^2/4$ is a **squared-Bessel BESQ(0)**. Standard facts for
$\delta<2$: the origin is reached almost surely; for $\delta\le0$ it is absorbing.
Hence:

> A driftless $\sqrt S$ process **hits 0 in finite time with probability 1**.

Even before full absorption, a nonnegative martingale must (by martingale
convergence + recurrence of the driftless walk) make arbitrarily large downward
excursions. So the leader cannot hold its position: it wanders down, the total
falls below $K$, $R$ rises, and the freed resource is captured by some other
state, which becomes the new leader. **Perpetual rise-and-fall.**

## 4. Why the *additive* growth makes the turnover thorough

When a fallen leader is briefly rescued (a dip makes $R>0$), the rescuing drift is
$\gamma_i R$ — the **same for every state**, independent of size. A former leader
gets no preferential recovery. Contrast multiplicative growth $\dot S=\gamma S R$:
there the drift scales with $S$, so the largest state is preferentially rescued
and can *lock in* (relative noise $\sim\sqrt S/S=S^{-1/2}\to0$ for large $S$).
That is the structural knob: **additive (size-independent) growth $\Rightarrow$
churn; multiplicative growth $\Rightarrow$ lock-in.** (Both still churn while
$R\approx0$, since then neither has drift; the difference shows up in the growth
phase and in the rescue dynamics.)

## 5. Caveat: a numerical artifact at long times

The positivity clamp `S = max(S,0)` injects a little strength every time the
$\sqrt S$ Euler step would send $S$ negative, so the total $\sum S$ is **not**
conserved and slowly inflates above $K$ on long runs ($t\gg200$). The clean
BESQ(0) absorption picture holds in the saturated window we actually observe; for
long-time quantitative work use a positivity-preserving CIR scheme (full
truncation / Alfonsi) or enforce the Feller condition, otherwise the absorption at
0 is partly masked by the injected mass.

## 6. Takeaways for the next steps

- The base model already has a non-trivial emergent phenomenon — **dynamic
  condensation with turnover** — explained in closed form. Good foundation.
- If a *durable* hegemon is ever wanted, it needs either multiplicative growth, a
  size advantage in conflict, or lower demographic noise; with the current
  ingredients dominance is intrinsically temporary.
- **Step 1 (decay $\delta$)** will add a real death rate. Open question to test:
  does $-\delta S$ change the BESQ dimension (it adds a linear mean-reversion,
  $\delta_{\rm Bessel}$ shifts) and set a finite, tunable lifetime for the leader
  — i.e. a characteristic rise-and-fall period?
