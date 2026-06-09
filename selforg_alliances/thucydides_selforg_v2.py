"""
Self-organising Thucydides trap -- v2
=====================================

Changes from v1 (in response to design review):

  (1) DYNAMIC growth rate.  gamma_i(t) is now a slow Ornstein-Uhlenbeck process
      with a COMMON mean -- no permanent advantage, full rank mobility.  A minor
      state can enter a growth epoch and rise on its own; "rise and fall" is
      endogenous, not only war-driven.
          dgamma_i = theta_g (mu_g - gamma_i) dt + sigma_g dW^g   (clipped >0)

  (2) NO per-state capacity c_i.  Growth is purely resource-limited,
          dS_i = [gamma_i(t) S_i R - delta S_i] dt + sigma sqrt(S_i) dW,
          R = max(0, 1 - sum_j S_j / K),
      coexistence maintained by (i) rotating gamma and (ii) density-dependent
      mortality (wars/predation hit the strongest).

  (3) Round length Delta = n_sub*dt is the decision clock; tau_ema is the slope
      filter memory.  Distinct objects (see note).

  (4) PREDICTABILITY = DELIBERATION DEPTH.  A state decides whether to start /
      join a war by running n_i Monte-Carlo rollouts of the outcome and acting on
      expected utility.  More rollouts -> lower-variance estimate -> better
      decisions.  n_i is a heritable trait under selection, with a strength tax
      proportional to n_i (an attention budget) so there is an interior optimum.

  (A) RING.  Each state has a position x_i on a 1-D ring; threat and alliance
      affinity decay with arc distance (range ell), wars need spatial adjacency.
      This admits a multipolar (regional-bloc) phase; a sufficiently dominant
      power overrides locality and forces global bipolarisation.

Ito convention, Euler-Maruyama.  Self-contained.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
@dataclass
class Params:
    N: int = 100
    K: float = 1500.0
    c0: float = 120.0           # COMMON saturation scale (one number, same for all
    #                            states) -- prevents winner-take-all without the
    #                            frozen per-state hierarchy of v1's c_i.
    delta: float = 0.035
    sigma: float = 0.16
    # dynamic growth rate (OU, common mean)
    mu_g: float = 0.11
    theta_g: float = 0.012      # reversion rate (1/timescale ~ 80 rounds)
    sigma_g: float = 0.035      # OU volatility
    g_min: float = 0.02
    g_max: float = 0.22

    # time
    rounds: int = 4000
    substeps: int = 5
    dt: float = 0.2             # round length Delta = substeps*dt = 1.0

    # forecasting (trend filter used for the crossover trigger)
    horizon: float = 40.0
    ema_tau: float = 8.0

    # ring geometry
    ell: float = 0.22           # threat/affinity arc-distance range
    r_war: float = 0.27         # max arc distance for two blocs to fight

    # alliances
    realign_rate: float = 0.15
    threat_thresh: float = 1.3
    bandwagon: float = 0.10

    # deliberation (predictability)
    ndepth_init_hi: int = 24    # initial n_i ~ U[1, ndepth_init_hi]
    evolve_depth: bool = True   # if False, depth is frozen (control-parameter sweep)
    ndepth_fixed: float = 0.0   # if >0, all states start at this depth (homogeneous)
    delib_cost: float = 0.0008  # strength tax per unit depth per round
    world_unc: float = 0.40     # per-rollout uncertainty about others' strengths
    #                             (deeper rollouts average it out -> depth pays)
    mut_depth: float = 2.0      # mutation sd on n_i at respawn

    # war
    war_attempt: float = 0.12   # prob a balanced adjacent rivalry is *considered* / round
    kappa: float = 12.0
    q: float = 1.6
    loss_lose: float = 0.45
    loss_win: float = 0.12
    spoils_war: float = 0.5     # fraction of destroyed loser strength taken by winner
    beta_absorb: float = 0.1    # extra (autonomy) cost of being absorbed after losing
    v_sec: float = 4.0          # value of regional security (per unit own strength)
    p_lose_floor: float = 0.1   # residual security of an absorbed subject state
    elim_frac: float = 0.04

    # predation
    prey_rate: float = 0.05
    prey_ratio: float = 3.0
    spoils: float = 0.30

    mut_g: float = 0.0          # gamma respawns to mean; no extra mutation needed
    seed: int = 0


@dataclass
class History:
    t: list = field(default_factory=list)
    S: list = field(default_factory=list)
    bloc: list = field(default_factory=list)
    xpos: list = field(default_factory=list)
    n_blocs: list = field(default_factory=list)
    neff: list = field(default_factory=list)
    f1: list = field(default_factory=list)
    f2: list = field(default_factory=list)
    balance: list = field(default_factory=list)
    gini: list = field(default_factory=list)
    mean_depth: list = field(default_factory=list)
    # war tuple: (round, nA, nB, SA, SB, winA, npart, big, severity,
    #             parity, comb_frac, crossover, xmid)
    wars: list = field(default_factory=list)
    neff_pre: list = field(default_factory=list)   # N_eff just before each war
    n_predation: int = 0


# --------------------------------------------------------------------------- #
def _gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(np.cumsum(x)) / np.cumsum(x)[-1]) / n


def _bloc_fractions(S, bloc):
    labels = np.unique(bloc); Stot = S.sum()
    if Stot <= 0:
        return len(labels), 0.0, 0.0, 1.0
    shares = np.sort(np.array([S[bloc == L].sum() for L in labels]) / Stot)
    f1 = shares[-1]; f2 = shares[-2] if shares.size > 1 else 0.0
    neff = float(shares.sum() ** 2 / np.sum(shares ** 2))
    return len(labels), float(f1), float(f2), neff


# --------------------------------------------------------------------------- #
class SelfOrgThucydidesV2:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        N = p.N
        self.gamma = np.clip(self.rng.normal(p.mu_g, p.sigma_g, N), p.g_min, p.g_max)
        self.x = self.rng.random(N)                       # ring position in [0,1)
        if p.ndepth_fixed > 0:
            self.ndepth = np.full(N, float(p.ndepth_fixed))
        else:
            self.ndepth = self.rng.integers(1, p.ndepth_init_hi + 1, N).astype(float)
        self.S = self.rng.uniform(2.0, 6.0, N)
        self.S_prev = self.S.copy()
        self.slope = np.zeros(N)
        self.bloc = np.arange(N)
        self.hist = History()

    # ---- ring distance -----------------------------------------------------
    def _dist(self, i):
        d = np.abs(self.x - self.x[i])
        return np.minimum(d, 1.0 - d)                     # arc distance in [0,0.5]

    # ---- growth + dynamic gamma -------------------------------------------
    def _grow(self):
        p = self.p
        rl = p.substeps * p.dt
        # OU update of gamma once per round (slow)
        self.gamma += (p.theta_g * (p.mu_g - self.gamma) * rl
                       + p.sigma_g * np.sqrt(rl) * self.rng.normal(0, 1, p.N))
        np.clip(self.gamma, p.g_min, p.g_max, out=self.gamma)
        for _ in range(p.substeps):
            Stot = self.S.sum()
            R = max(0.0, 1.0 - Stot / p.K)
            drift = self.gamma * self.S * (1.0 - self.S / p.c0) * R - p.delta * self.S
            diff = p.sigma * np.sqrt(np.maximum(self.S, 0.0))
            dW = self.rng.normal(0.0, np.sqrt(p.dt), p.N)
            self.S = self.S + drift * p.dt + diff * dW
            np.maximum(self.S, 0.0, out=self.S)
        # deliberation tax (attention budget)
        self.S = np.maximum(self.S - p.delib_cost * self.ndepth * rl, 0.0)

    # ---- alliance (balancing) with ring locality --------------------------
    def _realign(self):
        p = self.p
        Sproj = self.S + p.horizon * self.slope
        movers = np.where(self.rng.random(p.N) < p.realign_rate)[0]
        self.rng.shuffle(movers)
        for i in movers:
            prox = np.exp(-self._dist(i) / p.ell)          # proximity weights
            labels = np.unique(self.bloc)
            my = self.bloc[i]
            # proximity-weighted perceived bloc strength (threat felt by i)
            bs = {L: float((Sproj * prox)[self.bloc == L].sum()) for L in labels}
            si = Sproj[i] * prox[i]                         # ~ Sproj[i]
            others = {L: v for L, v in bs.items() if L != my}
            if not others:
                continue
            threat_L = max(others, key=others.get); threat_v = others[threat_L]
            if threat_v < p.threat_thresh * si:            # R1 salience
                if self.bloc[i] != i and self.rng.random() < 0.6:
                    self.bloc[i] = i
                continue
            if bs[my] > 1.8 * threat_v and self.bloc[i] != i:   # R2 dissolution
                if self.rng.random() < 0.6:
                    self.bloc[i] = i
                    continue
            best_L, best = my, -np.inf                      # R3 balancing
            for L, v in bs.items():
                if L == threat_L:
                    continue
                side = v + (si if L != my else 0.0)
                score = -abs(side - threat_v) + 0.25 * side
                if score > best:
                    best, best_L = score, L
            if threat_v > 2.5 * si and self.rng.random() < p.bandwagon:  # R4
                best_L = threat_L
            self.bloc[i] = best_L

    # ---- rollout EV of a war (vectorised over n samples) ------------------
    def _war_ev(self, S_self, S_own_other, S_opp, n, *, as_initiator=False,
                baseline_share=None):
        """Monte-Carlo expected utility of JOIN vs ABSTAIN for a decider of
        strength S_self whose side (excluding itself) totals S_own_other against
        an opponent S_opp.  Each of n rollouts draws a noisy world, samples the
        Lanchester outcome, and scores the consequences -- including ABSORPTION:
        the losing side is swallowed into the winner's alliance.  Deeper n -> the
        means below are sharper -> better decisions.

        Returns (ev_join, ev_abstain).  For an INITIATOR, abstaining means no war
        (status quo, utility 0).  For an ALLY of an already-started war, abstaining
        still lets the war happen -- and if its side then loses, its coalition is
        absorbed by the enemy, who grows and menaces it (the cost of losing allies).
        """
        p = self.p
        n = int(max(1, n))
        u = p.world_unc
        s = S_self
        own_o = np.maximum(S_own_other * (1 + self.rng.normal(0, u, n)), 0.0)
        opp = np.maximum(S_opp * (1 + self.rng.normal(0, u, n)), 0.0)
        V = p.v_sec * s                  # security value scale (per unit strength)
        Plo = p.p_lose_floor             # residual security of an absorbed subject
        # win probabilities with / without self (Lanchester, q exponent)
        Sj = own_o + s
        pj = Sj ** p.q / (Sj ** p.q + opp ** p.q + 1e-9)
        pa = own_o ** p.q / (own_o ** p.q + opp ** p.q + 1e-9)
        # JOIN: win -> coalition absorbs the enemy, near-total regional security;
        #       lose -> absorbed (security Plo) + blood-price + autonomy cost.
        util_j = (V * (pj * (1 - Plo) + Plo)
                  - pj * p.loss_win * s
                  - (1 - pj) * (p.loss_lose + p.beta_absorb) * s)
        if as_initiator:
            # abstain = no war: keep the PROJECTED share of regional power (a
            # rising challenger lowers it -> preventive-war motive), no blood.
            share = baseline_share if baseline_share is not None \
                else float(np.mean(Sj / (Sj + opp + 1e-9)))
            util_a = np.full(n, V * share)
        else:
            # abstain = the war still happens without me: if my side wins I stay
            # safe; if it LOSES, my coalition is absorbed and I am left exposed.
            util_a = V * (pa * (1 - Plo) + Plo)
        return float(util_j.mean()), float(util_a.mean())

    # ---- war ---------------------------------------------------------------
    def _maybe_war(self, rnd):
        p = self.p
        Sproj = self.S + p.horizon * self.slope
        labels = np.unique(self.bloc)
        if labels.size < 2:
            return
        bstr = {L: self.S[self.bloc == L].sum() for L in labels}
        bprj = {L: Sproj[self.bloc == L].sum() for L in labels}
        cand = sorted(labels, key=lambda L: bstr[L], reverse=True)[:8]
        rl = p.substeps * p.dt
        busy = set()
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                A_L, B_L = cand[a], cand[b]
                if A_L in busy or B_L in busy:
                    continue
                SA, SB = bstr[A_L], bstr[B_L]
                if SA <= 0 or SB <= 0:
                    continue
                iA = np.where(self.bloc == A_L)[0]; iB = np.where(self.bloc == B_L)[0]
                # spatial adjacency: closest members within r_war
                dx = np.abs(self.x[iA][:, None] - self.x[iB][None, :])
                dmin = np.minimum(dx, 1 - dx).min()
                if dmin > p.r_war:
                    continue
                parity = min(SA, SB) / max(SA, SB)
                if parity < 0.25:
                    continue
                # opportunity to consider this rivalry this round
                if self.rng.random() > p.war_attempt:
                    continue
                i = iA[np.argmax(self.S[iA])]; j = iB[np.argmax(self.S[iB])]
                Si, Sj = self.S[i], self.S[j]
                # aggressor deliberates: attack only if the rollout EV of war
                # beats the PROJECTED no-war status quo (preventive-war logic).
                if Si >= Sj:
                    agg, agg_L, opp_L = i, A_L, B_L
                else:
                    agg, agg_L, opp_L = j, B_L, A_L
                base = bprj[agg_L] / (bprj[agg_L] + bprj[opp_L] + 1e-9)
                ev_j, ev_a = self._war_ev(self.S[agg], bstr[agg_L] - self.S[agg],
                                          bstr[opp_L], self.ndepth[agg],
                                          as_initiator=True, baseline_share=base)
                if ev_j <= ev_a:
                    continue
                busy.add(A_L); busy.add(B_L)
                self._fight(rnd, A_L, B_L, Sproj)

    def _fight(self, rnd, A_L, B_L, Sproj):
        p = self.p
        iA = np.where(self.bloc == A_L)[0]; iB = np.where(self.bloc == B_L)[0]
        i = iA[np.argmax(self.S[iA])]; j = iB[np.argmax(self.S[iB])]
        Stot_before = self.S.sum()
        _, pf1, pf2, pneff = _bloc_fractions(self.S, self.bloc)

        sideA, sideB = {int(i)}, {int(j)}
        for (lead, home, side, opp_lead) in ((i, A_L, sideA, j), (j, B_L, sideB, i)):
            allies = [k for k in np.where(self.bloc == home)[0] if k != lead]
            for k in allies:
                S_own_other = self.S[lead] + sum(self.S[m] for m in side if m != lead)
                ev_j, ev_a = self._war_ev(self.S[k], S_own_other, self.S[opp_lead],
                                          self.ndepth[k])
                if ev_j > ev_a:
                    side.add(int(k))

        A = np.array(sorted(sideA)); B = np.array(sorted(sideB))
        SA = self.S[A].sum(); SB = self.S[B].sum()
        parity = min(SA, SB) / max(SA, SB); comb_frac = (SA + SB) / Stot_before
        Pi = self.S[i] + p.horizon * self.slope[i]
        Pj = self.S[j] + p.horizon * self.slope[j]
        crossover = bool(np.sign(self.S[i] - self.S[j]) != np.sign(Pi - Pj))
        xmid = float(self.x[i])
        pA = SA ** p.q / (SA ** p.q + SB ** p.q + 1e-9)
        A_wins = self.rng.random() < pA
        win, lose = (A, B) if A_wins else (B, A)
        loser_destroyed = p.loss_lose * self.S[lose].sum()
        self.S[lose] *= (1.0 - p.loss_lose)
        self.S[win] *= (1.0 - p.loss_win)
        # winner shares spoils (proportional to contribution)
        sw = self.S[win].sum()
        if sw > 0:
            self.S[win] += p.spoils_war * loser_destroyed * self.S[win] / sw
        # ABSORPTION: the defeated belligerents are swallowed into the winner's
        # alliance (the loser lead and its fighting allies switch bloc).  This is
        # what makes wars grow alliances -- and what allies fight to prevent.
        win_label = A_L if A_wins else B_L
        self.bloc[lose] = win_label
        severity = loser_destroyed
        n_part = A.size + B.size
        big = severity >= 0.10 * Stot_before
        self.hist.wars.append((rnd, int(A.size), int(B.size), float(SA), float(SB),
                               bool(A_wins), int(n_part), bool(big), float(severity),
                               float(parity), float(comb_frac), crossover, xmid))
        self.hist.neff_pre.append(float(pneff))

    # ---- predation (proximity-gated) --------------------------------------
    def _predation(self, rnd):
        p = self.p; rl = p.substeps * p.dt
        labels = np.unique(self.bloc)
        if labels.size < 2:
            return
        bstr = {L: self.S[self.bloc == L].sum() for L in labels}
        for i in range(p.N):
            if self.bloc[i] != i:
                continue
            prox = np.exp(-self._dist(i) / p.ell)
            # strongest *nearby* bloc other than self
            tv, tL = 0.0, None
            for L in labels:
                if L == i:
                    continue
                v = float((self.S * prox)[self.bloc == L].sum())
                if v > tv:
                    tv, tL = v, L
            if tL is None or tv < p.prey_ratio * self.S[i]:
                continue
            haz = p.prey_rate * (1.0 - 0.85 * (self.ndepth[i] / p.ndepth_init_hi))
            if self.rng.random() < 1.0 - np.exp(-max(haz, 0.0) * rl):
                ile = np.where(self.bloc == tL)[0]
                lead = ile[np.argmax(self.S[ile])]
                self.S[lead] += p.spoils * self.S[i]
                self.S[i] = 0.0
                self.hist.n_predation += 1

    # ---- selection / respawn ----------------------------------------------
    def _select(self):
        p = self.p
        meanS = self.S.mean()
        dead = np.where(self.S < p.elim_frac * meanS)[0]
        if dead.size == 0:
            return
        alive = np.where(self.S >= p.elim_frac * meanS)[0]
        if alive.size == 0:
            return
        wts = self.S[alive] / self.S[alive].sum()
        for d in dead:
            par = alive[self.rng.choice(alive.size, p=wts)]
            if p.evolve_depth:
                self.ndepth[d] = np.clip(
                    self.ndepth[par] + self.rng.normal(0, p.mut_depth), 1, 200)
            else:
                self.ndepth[d] = self.ndepth[par]
            self.gamma[d] = p.mu_g                          # fresh growth rate
            self.x[d] = self.rng.random()                   # new position
            self.S[d] = 0.5 * p.elim_frac * meanS
            self.slope[d] = 0.0
            self.bloc[d] = d

    # ---- one round ---------------------------------------------------------
    def step(self, rnd):
        p = self.p
        self.S_prev = self.S.copy()
        self._grow()
        realised = (self.S - self.S_prev) / (p.substeps * p.dt)
        a = 1.0 / p.ema_tau
        self.slope = (1 - a) * self.slope + a * realised
        self._realign()
        self._maybe_war(rnd)
        self._predation(rnd)
        self._select()

    def run(self, sample_every=10):
        p = self.p
        for rnd in range(p.rounds):
            self.step(rnd)
            if rnd % sample_every == 0:
                nb, f1, f2, neff = _bloc_fractions(self.S, self.bloc)
                h = self.hist
                h.t.append(rnd); h.S.append(self.S.copy())
                h.bloc.append(self.bloc.copy()); h.xpos.append(self.x.copy())
                h.n_blocs.append(nb); h.neff.append(neff)
                h.f1.append(f1); h.f2.append(f2)
                h.balance.append(1 - abs(f1 - f2) / (f1 + f2 + 1e-12))
                h.gini.append(_gini(self.S)); h.mean_depth.append(self.ndepth.mean())
        return self.hist


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = Params()
    sim = SelfOrgThucydidesV2(p)
    d0 = sim.ndepth.mean()
    h = sim.run()
    w = np.array(h.wars, dtype=float)
    big = w[:, 7] == 1 if len(w) else np.array([])
    print(f"rounds={p.rounds} wars={len(w)} big={int(big.sum()) if len(w) else 0} "
          f"predations={h.n_predation}")
    print(f"deliberation depth: {d0:.1f} -> {sim.ndepth.mean():.1f}")
    print(f"final n_blocs={h.n_blocs[-1]} Neff={h.neff[-1]:.2f} "
          f"balance={h.balance[-1]:.2f} Gini={h.gini[-1]:.2f}")
    print(f"Neff range over run: {min(h.neff):.2f} .. {max(h.neff):.2f} "
          f"(mean {np.mean(h.neff):.2f})")
    if len(w) and big.sum():
        print(f"crossover frac big/small: {w[big,11].mean():.2f} / "
              f"{w[~big,11].mean():.2f}")
