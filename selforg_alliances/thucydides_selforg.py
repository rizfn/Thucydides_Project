"""
Thucydides trap from self-organising principles
================================================

A mean-field (graph, not lattice) agent model of the Thucydides trap built on
three minimal ingredients requested in the project brief:

  (1) Continuous, resource-constrained strength dynamics
          dS_i = [ gamma_i S_i (1 - Stot/K) - delta S_i ] dt + sigma sqrt(S_i) dW_i
      A stochastic competitive-logistic (CIR-type noise) process: heterogeneous
      intrinsic growth gamma_i, global resource ceiling K (no infinite growth),
      decay delta. Differential growth -> power transitions, the engine of the
      Thucydides mechanism.

  (2) Self-organised PREDICTABILITY. Every state forecasts the *slope* dS_j/dt of
      every other state (exponential-moving-average estimate) and projects
      strength a horizon h ahead. Each state has a predictive accuracy a_i in
      (0,1]; its perception of others is noisier the lower a_i. Decisions (war /
      ally / wait) use these forecasts. Accuracy is heritable + mutated; states
      eliminated in war are respawned as offspring of survivors -> SELECTION for
      predictability ("states that lack it get destroyed").

  (3) Balancing ALLIANCES. A state joins the largest bloc that is *not* its top
      projected threat (classical balancing, Waltz). This greedily drives the
      system toward a bipolar (two-bloc) structure; once the balancing bloc
      overtakes the leader it becomes the new threat -> realignment churn. We
      track whether a near-bipolar configuration precedes the large wars.

  War: the focal rivalry is the pair of strongest blocs' lead states (i,j). War
  fires with a Thucydides hazard that peaks at parity *and* imminent projected
  crossover. Once initiated between i and j, each ally in their blocs decides
  INDEPENDENTLY, from its own (accuracy-limited) forecast, whether to join.
  Outcome ~ Lanchester: P(A wins) = SA^q/(SA^q+SB^q). Losers pay heavily and may
  be eliminated; winners pay a smaller toll.

Ito convention, Euler-Maruyama. Self-contained: `python3 thucydides_selforg.py`.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    N: int = 100              # number of states
    K: float = 1500.0         # global resource ceiling (sum of strengths)
    delta: float = 0.035      # strength decay (keeps the system out of frozen saturation)
    gamma_lo: float = 0.06    # intrinsic growth range
    gamma_hi: float = 0.16
    sigma: float = 0.16       # demographic (sqrt-S) noise amplitude
    cap_med: float = 12.0     # median per-state capacity c_i ("territory")
    cap_sig: float = 0.6      # log-normal width of capacities -> great vs minor powers

    # time
    rounds: int = 3000        # number of decision rounds
    substeps: int = 5         # SDE Euler substeps per round
    dt: float = 0.2           # substep size  (round length = substeps*dt)

    # forecasting / predictability
    horizon: float = 25.0     # forecast horizon h (in time units)
    ema_tau: float = 8.0      # EMA time-constant for slope estimation (rounds)
    perc_noise: float = 0.35  # perception-noise scale at accuracy a=0

    # alliances
    realign_rate: float = 0.15   # prob a given state reconsiders its bloc / round
    threat_thresh: float = 1.3   # a rival is a *salient* threat only if this much stronger
    bandwagon: float = 0.10      # tendency to join an overwhelming threat instead

    # war
    w: float = 0.05           # war hazard prefactor
    kappa: float = 12.0       # steepness of the rising-challenger (Thucydides) trigger
    q: float = 1.6            # Lanchester exponent in win probability
    join_gain: float = 6.0    # steepness of allies' join decision (logistic)
    loss_lose: float = 0.45   # strength fraction destroyed on the losing side
    loss_win: float = 0.12    # strength fraction destroyed on the winning side
    elim_frac: float = 0.04   # eliminated if S < elim_frac * mean(S)

    # predation: weak UNALIGNED states next to a strong power get conquered.
    # States that fail to foresee the threat and ally in time are the ones eaten
    # -> this is the selection pressure on predictability.
    prey_rate: float = 0.05   # base conquest hazard / round for an exposed singleton
    prey_ratio: float = 3.0   # "exposed" if a bloc is this much stronger than the state
    spoils: float = 0.30      # fraction of victim strength the conqueror absorbs

    # selection / respawn
    mut_gamma: float = 0.010
    mut_acc: float = 0.05

    seed: int = 0


# --------------------------------------------------------------------------- #
#  Diagnostics container
# --------------------------------------------------------------------------- #
@dataclass
class History:
    S: list = field(default_factory=list)          # strength snapshots (sampled)
    t: list = field(default_factory=list)
    n_blocs: list = field(default_factory=list)
    neff: list = field(default_factory=list)        # effective # of blocs (participation ratio)
    f1: list = field(default_factory=list)         # largest bloc fraction (of Stot)
    f2: list = field(default_factory=list)         # second-largest bloc fraction
    bipolar: list = field(default_factory=list)    # f1+f2 (bloc-share bipolarity)
    balance: list = field(default_factory=list)    # 1-|f1-f2|/(f1+f2)
    gini: list = field(default_factory=list)
    mean_acc: list = field(default_factory=list)
    # war log tuple:
    #  (round, sizeA, sizeB, SA, SB, winner_is_A, n_participants, big,
    #   severity, parity, comb_frac, crossover)
    wars: list = field(default_factory=list)
    bipolar_at_war: list = field(default_factory=list)  # f1+f2 at each war
    neff_at_war: list = field(default_factory=list)      # effective #blocs at each war
    balance_at_war: list = field(default_factory=list)   # bloc balance at each war


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def _bloc_fractions(S, bloc):
    """Return (n_blocs, f1, f2) where f1>=f2 are the two largest bloc shares of Stot."""
    labels = np.unique(bloc)
    Stot = S.sum()
    if Stot <= 0:
        return len(labels), 0.0, 0.0
    shares = np.array([S[bloc == L].sum() for L in labels]) / Stot
    shares.sort()
    f1 = shares[-1]
    f2 = shares[-2] if shares.size > 1 else 0.0
    neff = float(shares.sum() ** 2 / np.sum(shares ** 2))   # 1=unipolar, 2=bipolar, ...
    return len(labels), float(f1), float(f2), neff


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class SelfOrgThucydides:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        N = p.N
        # heterogeneous intrinsic growth
        self.gamma = self.rng.uniform(p.gamma_lo, p.gamma_hi, N)
        # heterogeneous per-state capacity (territory) -> coexisting power hierarchy
        self.cap = p.cap_med * np.exp(self.rng.normal(0, p.cap_sig, N))
        # predictive accuracy in (0,1]; start broad so selection has variance
        self.acc = self.rng.uniform(0.2, 0.95, N)
        # strengths: start small & comparable
        self.S = self.rng.uniform(2.0, 6.0, N)
        self.S_prev = self.S.copy()
        self.slope = np.zeros(N)          # EMA slope estimate dS/dt
        self.bloc = np.arange(N)          # alliance label; start all singletons
        self.n_predation = 0
        self.hist = History()

    # ---- forecasting -------------------------------------------------------
    def _project(self):
        """True projected strength a horizon ahead (h * slope)."""
        return self.S + self.p.horizon * self.slope

    def _perceived(self, i, Sproj):
        """State i's accuracy-limited perception of every state's projected strength."""
        scale = self.p.perc_noise * (1.0 - self.acc[i])
        eps = self.rng.normal(0.0, scale, self.p.N)
        return np.maximum(Sproj * (1.0 + eps), 0.0)

    def _bloc_strength(self, S):
        """Map: bloc label -> summed strength, broadcast back to each state."""
        out = np.zeros(self.p.N)
        for L in np.unique(self.bloc):
            m = self.bloc == L
            out[m] = S[m].sum()
        return out

    # ---- SDE growth --------------------------------------------------------
    def _grow(self):
        p = self.p
        for _ in range(p.substeps):
            Stot = self.S.sum()
            R = max(0.0, 1.0 - Stot / p.K)      # shared-resource depletion factor
            drift = (self.gamma * self.S * (1.0 - self.S / self.cap) * R
                     - p.delta * self.S)
            diff = p.sigma * np.sqrt(np.maximum(self.S, 0.0))
            dW = self.rng.normal(0.0, np.sqrt(p.dt), p.N)
            self.S = self.S + drift * p.dt + diff * dW
            np.maximum(self.S, 0.0, out=self.S)

    # ---- alliance (balancing) update --------------------------------------
    def _realign(self):
        p = self.p
        Sproj = self._project()
        # who reconsiders this round
        movers = np.where(self.rng.random(p.N) < p.realign_rate)[0]
        self.rng.shuffle(movers)
        for i in movers:
            perc = self._perceived(i, Sproj)
            labels = np.unique(self.bloc)
            # projected strength of each bloc as perceived by i
            bs = {L: perc[self.bloc == L].sum() for L in labels}
            my = self.bloc[i]
            # top threat = strongest bloc that is not mine
            others = {L: v for L, v in bs.items() if L != my}
            if not others:
                continue
            threat_L = max(others, key=others.get)
            threat_v = others[threat_L]
            # SALIENCE: only balance against a rival that clearly outweighs me.
            # If no salient threat (diffuse power), an alliance has no rationale
            # -> stand alone. Many states sharing the SAME salient threat (a risen
            # hegemon) converge on one balancing coalition -> bipolarity emerges;
            # when power is diffuse the system stays multipolar.
            if threat_v < p.threat_thresh * perc[i]:
                if self.bloc[i] != i and self.rng.random() < 0.6:
                    self.bloc[i] = i           # no threat worth an alliance -> defect
                continue
            # DISSOLUTION: if my own bloc already dwarfs the top threat, the
            # alliance has no rationale (and dilutes my spoils) -> defect.
            if bs[my] > 1.8 * threat_v and self.bloc[i] != i:
                if self.rng.random() < 0.6:
                    self.bloc[i] = i
                    continue
            # candidate blocs to join = all except the threat bloc (balancing),
            # plus staying put; pick the one that best balances the threat.
            best_L, best_score = my, -np.inf
            for L, v in bs.items():
                if L == threat_L:
                    continue
                # joining L gives my side strength ~ v (+ my own if not already in L)
                side = v + (perc[i] if L != my else 0.0)
                # security: want side >= threat; reward proximity-from-above,
                # penalise being a tiny singleton.
                score = -abs(side - threat_v) + 0.25 * side
                if score > best_score:
                    best_score, best_L = score, L
            # occasional bandwagoning onto an overwhelming threat
            if (threat_v > 2.5 * perc[i]
                    and self.rng.random() < p.bandwagon):
                best_L = threat_L
            self.bloc[i] = best_L

    # ---- war ---------------------------------------------------------------
    def _maybe_war(self, rnd):
        """Any sufficiently-balanced pair of blocs may fight; hazard follows the
        Thucydides rule (parity * imminent projected crossover). Several wars per
        round are possible (small minor-power clashes up to systemic bipolar war),
        producing a war-size distribution. Each bloc fights at most once/round."""
        p = self.p
        Sproj = self._project()
        labels = np.unique(self.bloc)
        if labels.size < 2:
            return
        bstr = {L: self.S[self.bloc == L].sum() for L in labels}
        bprj = {L: Sproj[self.bloc == L].sum() for L in labels}
        # restrict to the strongest M blocs as candidate belligerents
        cand = sorted(labels, key=lambda L: bstr[L], reverse=True)[:8]
        round_len = p.substeps * p.dt
        busy = set()
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                A_L, B_L = cand[a], cand[b]
                if A_L in busy or B_L in busy:
                    continue
                SA, SB = bstr[A_L], bstr[B_L]
                if SA <= 0 or SB <= 0:
                    continue
                parity = min(SA, SB) / max(SA, SB)
                if parity < 0.25:                 # too lopsided -> no power-transition war
                    continue
                # THUCYDIDES TRIGGER: lead state of each bloc; is the weaker side's
                # projected growth closing the gap on the stronger (preventive-war
                # incentive)?  imminence in (0,1), >1/2 when the challenger rises.
                iA = np.where(self.bloc == A_L)[0]; iB = np.where(self.bloc == B_L)[0]
                i = iA[np.argmax(self.S[iA])]; j = iB[np.argmax(self.S[iB])]
                Si, Sj = self.S[i], self.S[j]
                closing = p.horizon * (self.slope[j] - self.slope[i]) / (Si + Sj + 1e-9)
                if Si < Sj:                       # ensure "closing>0" = challenger gains
                    closing = -closing
                imminence = 1.0 / (1.0 + np.exp(-p.kappa * closing))
                hazard = p.w * parity * imminence
                if self.rng.random() > 1.0 - np.exp(-hazard * round_len):
                    continue
                busy.add(A_L); busy.add(B_L)
                self._fight(rnd, A_L, B_L, Sproj)

    def _fight(self, rnd, A_L, B_L, Sproj):
        p = self.p
        iA = np.where(self.bloc == A_L)[0]
        iB = np.where(self.bloc == B_L)[0]
        i = iA[np.argmax(self.S[iA])]
        j = iB[np.argmax(self.S[iB])]
        Stot_before = self.S.sum()
        # pre-war alliance structure (the configuration the war is born into)
        _, pf1, pf2, pneff = _bloc_fractions(self.S, self.bloc)
        pbal = 1 - abs(pf1 - pf2) / (pf1 + pf2 + 1e-12)

        # ---- coalition formation: allies decide individually -------------
        sideA, sideB = {int(i)}, {int(j)}
        for (lead, home_L, side, opp_lead) in ((i, A_L, sideA, j),
                                               (j, B_L, sideB, i)):
            allies = [k for k in np.where(self.bloc == home_L)[0] if k != lead]
            for k in allies:
                perc = self._perceived(k, Sproj)
                # k joins if its forecast says its side can win and the opponent
                # bloc is a real threat to k.
                own_side = perc[lead] + perc[k]
                opp_side = perc[opp_lead]
                pwin = own_side ** p.q / (own_side ** p.q + opp_side ** p.q + 1e-9)
                threat = opp_side / (perc[k] + 1e-9)
                u = p.join_gain * (pwin - 0.5) + 0.4 * np.tanh(threat - 1.0)
                if self.rng.random() < 1.0 / (1.0 + np.exp(-u)):
                    side.add(int(k))

        A = np.array(sorted(sideA)); B = np.array(sorted(sideB))
        SA = self.S[A].sum(); SB = self.S[B].sum()
        # belligerent diagnostics (the Thucydides trigger variables)
        parity = min(SA, SB) / max(SA, SB)
        comb_frac = (SA + SB) / Stot_before
        # crossover = the weaker lead state is projected to overtake the stronger
        Si, Sj = self.S[i], self.S[j]
        Pi, Pj = Si + p.horizon * self.slope[i], Sj + p.horizon * self.slope[j]
        crossover = bool(np.sign(Si - Sj) != np.sign(Pi - Pj))
        # ---- resolve (Lanchester) ----------------------------------------
        pA = SA ** p.q / (SA ** p.q + SB ** p.q + 1e-9)
        A_wins = self.rng.random() < pA
        win, lose = (A, B) if A_wins else (B, A)
        before = self.S[A].sum() + self.S[B].sum()
        self.S[win] *= (1.0 - p.loss_win)
        self.S[lose] *= (1.0 - p.loss_lose)
        severity = before - (self.S[A].sum() + self.S[B].sum())  # strength destroyed

        n_part = A.size + B.size
        big = severity >= 0.10 * Stot_before          # systemic war: >=10% of world power lost
        self.hist.wars.append((rnd, int(A.size), int(B.size),
                               float(SA), float(SB), bool(A_wins),
                               int(n_part), bool(big), float(severity),
                               float(parity), float(comb_frac), crossover))
        self.hist.bipolar_at_war.append(float(pf1 + pf2))
        self.hist.neff_at_war.append(float(pneff))
        self.hist.balance_at_war.append(float(pbal))

    # ---- predation on the unprepared (selection on foresight) -------------
    def _predation(self, rnd):
        """A weak, UNALIGNED state sitting next to a much stronger bloc is liable
        to be conquered. A state that predicted the threat would have allied (and
        thus not be a vulnerable singleton); low-accuracy states misperceive the
        threat, stay exposed, and get eaten -> selection for predictability."""
        p = self.p
        round_len = p.substeps * p.dt
        labels = np.unique(self.bloc)
        bstr = {L: self.S[self.bloc == L].sum() for L in labels}
        strongest_L = max(bstr, key=bstr.get)
        ile = np.where(self.bloc == strongest_L)[0]
        lead = ile[np.argmax(self.S[ile])]
        for i in range(p.N):
            if self.bloc[i] != i:           # only unaligned singletons are exposed
                continue
            # strongest threatening bloc that is not the singleton itself
            threat = max(v for L, v in bstr.items() if L != i) if len(labels) > 1 else 0.0
            if threat < p.prey_ratio * self.S[i]:
                continue
            # foresight is protective: a high-accuracy state anticipates the
            # threat and is hard to catch unprepared; a blind one is surprised.
            haz = p.prey_rate * (1.0 - 0.85 * self.acc[i])
            if self.rng.random() < 1.0 - np.exp(-haz * round_len):
                self.S[lead] += p.spoils * self.S[i]   # conqueror takes the spoils
                self.S[i] = 0.0                        # victim destroyed -> respawned
                self.n_predation += 1

    # ---- elimination + respawn (selection on predictability) --------------
    def _select(self):
        p = self.p
        meanS = self.S.mean()
        dead = np.where(self.S < p.elim_frac * meanS)[0]
        if dead.size == 0:
            return
        # parents drawn from survivors, weighted by strength (success breeds)
        alive = np.where(self.S >= p.elim_frac * meanS)[0]
        if alive.size == 0:
            return
        wts = self.S[alive] / self.S[alive].sum()
        for d in dead:
            par = alive[self.rng.choice(alive.size, p=wts)]
            self.gamma[d] = np.clip(
                self.gamma[par] + self.rng.normal(0, p.mut_gamma),
                p.gamma_lo, p.gamma_hi)
            self.acc[d] = np.clip(
                self.acc[par] + self.rng.normal(0, p.mut_acc), 0.02, 1.0)
            self.cap[d] = self.cap[par] * np.exp(self.rng.normal(0, 0.15))
            self.S[d] = 0.5 * p.elim_frac * meanS   # respawn weak
            self.slope[d] = 0.0
            self.bloc[d] = d                          # break off as singleton

    # ---- one round ---------------------------------------------------------
    def step(self, rnd):
        p = self.p
        self.S_prev = self.S.copy()
        self._grow()
        # update slope estimate (EMA of realised slope)
        realised = (self.S - self.S_prev) / (p.substeps * p.dt)
        a = 1.0 / p.ema_tau
        self.slope = (1 - a) * self.slope + a * realised
        self._realign()
        self._maybe_war(rnd)
        self._predation(rnd)
        self._select()

    # ---- run ---------------------------------------------------------------
    def run(self, sample_every=10):
        p = self.p
        for rnd in range(p.rounds):
            self.step(rnd)
            if rnd % sample_every == 0:
                nb, f1, f2, neff = _bloc_fractions(self.S, self.bloc)
                h = self.hist
                h.t.append(rnd)
                h.S.append(self.S.copy())
                h.n_blocs.append(nb)
                h.neff.append(neff)
                h.f1.append(f1); h.f2.append(f2)
                h.bipolar.append(f1 + f2)
                h.balance.append(1 - abs(f1 - f2) / (f1 + f2 + 1e-12))
                h.gini.append(_gini(self.S))
                h.mean_acc.append(self.acc.mean())
        return self.hist


# --------------------------------------------------------------------------- #
#  CLI demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = Params()
    sim = SelfOrgThucydides(p)
    h = sim.run()
    nwar = len(h.wars)
    nbig = sum(w[7] for w in h.wars)
    print(f"rounds={p.rounds}  wars={nwar}  big_wars={nbig}")
    print(f"final n_blocs={h.n_blocs[-1]}  bipolarity f1+f2={h.bipolar[-1]:.2f}"
          f"  balance={h.balance[-1]:.2f}")
    print(f"mean predictive accuracy: {h.mean_acc[0]:.3f} -> {h.mean_acc[-1]:.3f}")
    print(f"final Gini(S)={h.gini[-1]:.3f}")
    if nbig:
        big_mask = np.array([w[7] for w in h.wars])
        neff = np.array(h.neff_at_war); bal = np.array(h.balance_at_war)
        print(f"effective #blocs  at BIG wars = {neff[big_mask].mean():.2f}"
              f"  | small wars = {neff[~big_mask].mean():.2f}")
        print(f"bloc balance      at BIG wars = {bal[big_mask].mean():.2f}"
              f"  | small wars = {bal[~big_mask].mean():.2f}")
