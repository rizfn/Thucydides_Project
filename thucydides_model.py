"""
Minimal Gillespie-on-lattice model of the Thucydides trap. v2 (June 2026).

State i has territory T_i on an L x L lattice with periodic BCs,
area A_i = |T_i|, integer strength S_i. For each pair of neighbouring
states (i,j) the model uses the shared boundary length B_ij.

Reactions (rate per unit time):
  R1  Growth        : S_i -> S_i + 1            rate = max(0, C_i - S_i)
                      capacity C_i = A_i / (1 + A_i / A_star)  [Khaldun]
  R2  War i<->j     : conquest of one site       rate = w * B_ij * min(S_i, S_j)
                      [the "available conflict strength" of a dyad is set by
                       the weaker side; mutual growth -> high war rate]
                      winner ~ S_i / (S_i + S_j);
                      loser concedes ONE boundary site, chosen with
                      probability proportional to the number of edges that
                      site shares with the winner (bond-percolation rule:
                      front-line cells fall first, territory stays compact);
                      both sides pay xi * min(S_i, S_j) strength.
  R3  Fragmentation : split state                rate = c * A_i * (1 - S_i/A_i)+
                      Procedure: pick two random seeds s1, s2 in T_i; every
                      cell of T_i is assigned to whichever seed is closer
                      under periodic L2 distance (2-seed Voronoi inside T_i).
                      Both new fragments start with S = 0 (collapse of
                      cohesion). Setting c=0 yields zero fragmentations.

Free parameters: w, c, A_star  (with r = 1, xi = 0.5).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def _periodic_neighbors(x: int, y: int, L: int):
    return ((x + 1) % L, y), ((x - 1) % L, y), (x, (y + 1) % L), (x, (y - 1) % L)


def _periodic_d2(x1, y1, x2, y2, L):
    dx = abs(x1 - x2); dx = min(dx, L - dx)
    dy = abs(y1 - y2); dy = min(dy, L - dy)
    return dx * dx + dy * dy


@dataclass
class Sim:
    L: int = 64
    w: float = 0.002
    c: float = 5e-5
    A_star: float = 200.0       # Khaldun capacity ceiling
    xi: float = 0.5             # war damage fraction
    seed: int = 0
    n_init: int = 40
    record_every: float = 2.0

    owner: np.ndarray = field(default=None)
    S: dict = field(default_factory=dict)
    A: dict = field(default_factory=dict)
    boundary: dict = field(default_factory=dict)   # frozenset({i,j}) -> int B_ij
    nbrs: dict = field(default_factory=dict)
    next_id: int = 1
    t: float = 0.0
    rng: np.random.Generator = field(default=None)

    times: list = field(default_factory=list)
    n_states: list = field(default_factory=list)
    largest_area: list = field(default_factory=list)
    total_strength: list = field(default_factory=list)
    war_count: list = field(default_factory=list)
    war_log: list = field(default_factory=list)
    frag_log: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    snap_times: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._init_voronoi()
        self._rebuild_boundary()

    def capacity(self, A):
        return A / (1.0 + A / self.A_star)

    # ---- init ----------------------------------------------------------------
    def _init_voronoi(self):
        L = self.L
        coords = self.rng.integers(0, L, size=(self.n_init, 2))
        seeds, seen = [], set()
        for c in coords:
            t = tuple(c)
            if t not in seen:
                seen.add(t); seeds.append(t)
        owner = np.zeros((L, L), dtype=np.int32)
        XX, YY = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        best = np.full((L, L), np.inf)
        for k, (sx, sy) in enumerate(seeds, start=1):
            dx = np.minimum(np.abs(XX - sx), L - np.abs(XX - sx))
            dy = np.minimum(np.abs(YY - sy), L - np.abs(YY - sy))
            d2 = dx * dx + dy * dy
            mask = d2 < best
            owner[mask] = k
            best[mask] = d2[mask]
        self.owner = owner
        unique, counts = np.unique(owner, return_counts=True)
        for uid, ct in zip(unique, counts):
            self.A[int(uid)] = int(ct)
            self.S[int(uid)] = 0   # cold start
        self.next_id = int(unique.max()) + 1

    def _rebuild_boundary(self):
        L = self.L
        self.boundary.clear()
        self.nbrs = {sid: set() for sid in self.A}
        # Each edge counted once (positive direction only).
        for x in range(L):
            for y in range(L):
                a = int(self.owner[x, y])
                for nx, ny in [((x + 1) % L, y), (x, (y + 1) % L)]:
                    b = int(self.owner[nx, ny])
                    if b != a:
                        key = frozenset((a, b))
                        self.boundary[key] = self.boundary.get(key, 0) + 1
                        self.nbrs[a].add(b)
                        self.nbrs[b].add(a)

    # ---- rate computation ----------------------------------------------------
    def _compute_rates(self):
        kinds = []; keys = []; rates = []
        # R1 growth
        for i, A in self.A.items():
            S = self.S[i]
            r1 = max(0.0, self.capacity(A) - S)
            if r1 > 0:
                kinds.append(0); keys.append(i); rates.append(r1)
            # R3 fragmentation
            if self.c > 0 and A >= 2:
                slack = max(0.0, 1.0 - S / max(1, A))
                rate = self.c * A * slack
                if rate > 0:
                    kinds.append(3); keys.append(i); rates.append(rate)
        # R2 war   rate = w * B_ij * min(S_i, S_j)
        for key, B in self.boundary.items():
            if B <= 0:
                continue
            i, j = tuple(key)
            Si = self.S.get(i, 0); Sj = self.S.get(j, 0)
            m = min(Si, Sj)
            if m > 0:
                kinds.append(2); keys.append(key); rates.append(self.w * B * m)
        return kinds, keys, np.asarray(rates, dtype=np.float64)

    # ---- events --------------------------------------------------------------
    def _do_grow(self, i):
        self.S[i] += 1

    def _do_war(self, key):
        i, j = tuple(key)
        Si, Sj = self.S.get(i, 0), self.S.get(j, 0)
        Stot = Si + Sj
        if Stot == 0:
            return
        p_i = Si / Stot
        winner, loser = (i, j) if self.rng.random() < p_i else (j, i)
        loss = self.xi * min(Si, Sj)
        self.S[i] = max(0, int(round(Si - loss)))
        self.S[j] = max(0, int(round(Sj - loss)))
        site = self._weighted_boundary_site(winner, loser)
        if site is not None:
            self._transfer_site(site, loser, winner)
        self.war_log.append((self.t, winner, loser,
                             int(self.A.get(winner, 0)), int(self.A.get(loser, 0))))

    def _weighted_boundary_site(self, winner, loser):
        """Pick a loser-owned site adjacent to winner; weight by number of
        edges (1..4) that the site shares with winner. This is the
        bond-percolation rule: a site with three winner-edges is 3x more
        likely than one with a single edge.
        """
        L = self.L; ow = self.owner
        cands = []; weights = []
        # Scan only loser-owned sites
        ys, xs = np.where(ow == loser)
        for x, y in zip(ys, xs):
            n_shared = 0
            for nx, ny in _periodic_neighbors(int(x), int(y), L):
                if ow[nx, ny] == winner:
                    n_shared += 1
            if n_shared > 0:
                cands.append((int(x), int(y)))
                weights.append(n_shared)
        if not cands:
            return None
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        idx = int(self.rng.choice(len(cands), p=weights))
        return cands[idx]

    def _transfer_site(self, site, from_id, to_id):
        x, y = site
        L = self.L
        for nx, ny in _periodic_neighbors(x, y, L):
            nb = int(self.owner[nx, ny])
            if from_id != nb:
                key_old = frozenset((from_id, nb))
                self.boundary[key_old] = max(0, self.boundary.get(key_old, 0) - 1)
                if self.boundary[key_old] == 0:
                    self.boundary.pop(key_old, None)
                    if from_id in self.nbrs: self.nbrs[from_id].discard(nb)
                    if nb in self.nbrs: self.nbrs[nb].discard(from_id)
            if to_id != nb:
                key_new = frozenset((to_id, nb))
                self.boundary[key_new] = self.boundary.get(key_new, 0) + 1
                self.nbrs.setdefault(to_id, set()).add(nb)
                self.nbrs.setdefault(nb, set()).add(to_id)
        self.owner[x, y] = to_id
        self.A[from_id] -= 1
        self.A[to_id] = self.A.get(to_id, 0) + 1
        if self.A[from_id] <= 0:
            self._remove_state(from_id)

    def _remove_state(self, sid):
        self.A.pop(sid, None); self.S.pop(sid, None)
        nbrs = self.nbrs.pop(sid, set())
        for nb in nbrs:
            self.nbrs.get(nb, set()).discard(sid)
            self.boundary.pop(frozenset((sid, nb)), None)

    def _do_frag(self, i):
        """Fragment state i by a 2-seed Voronoi inside its territory.

        Procedure:
          1. Enumerate the sites of T_i.
          2. Pick two distinct random seeds s1, s2 in T_i.
          3. Every site is reassigned to whichever seed is closer under
             periodic L2 distance. One half keeps id i, the other gets a new id.
          4. Both halves have S_i = S_new = 0 (collapse of cohesion).
        """
        ow = self.owner
        L = self.L
        sites = np.argwhere(ow == i)
        if len(sites) < 4:
            return
        # pick two distinct seeds
        idx = self.rng.choice(len(sites), size=2, replace=False)
        s1 = tuple(sites[idx[0]])
        s2 = tuple(sites[idx[1]])
        # assign each site to closer seed
        d1 = np.array([_periodic_d2(int(x), int(y), s1[0], s1[1], L) for x, y in sites])
        d2 = np.array([_periodic_d2(int(x), int(y), s2[0], s2[1], L) for x, y in sites])
        to_new = d2 < d1
        if to_new.sum() == 0 or to_new.sum() == len(sites):
            return
        new_id = self.next_id; self.next_id += 1
        self.A[new_id] = 0; self.S[new_id] = 0
        self.nbrs[new_id] = set()
        for (x, y), n in zip(sites, to_new):
            if n:
                self._transfer_site((int(x), int(y)), i, new_id)
        # both halves: cohesion collapse
        self.S[i] = 0
        self.S[new_id] = 0
        self.frag_log.append((self.t, i, new_id))

    # ---- main loop -----------------------------------------------------------
    def run(self, t_max: float, snapshot_times=None, verbose: bool = False):
        if snapshot_times is None:
            snapshot_times = []
        snap_iter = iter(sorted(snapshot_times))
        next_snap = next(snap_iter, None)
        next_record = self.t
        self._record()
        while self.t < t_max:
            kinds, keys, rates = self._compute_rates()
            total = float(rates.sum()) if rates.size else 0.0
            if total <= 0:
                break
            dt = self.rng.exponential(1.0 / total)
            self.t += dt
            cdf = np.cumsum(rates)
            r = self.rng.random() * total
            idx = int(np.searchsorted(cdf, r))
            if idx >= len(kinds): idx = len(kinds) - 1
            kind = kinds[idx]; key = keys[idx]
            if kind == 0: self._do_grow(key)
            elif kind == 2: self._do_war(key)
            elif kind == 3: self._do_frag(key)

            if self.t >= next_record:
                self._record(); next_record += self.record_every
            if next_snap is not None and self.t >= next_snap:
                self.snapshots.append(self.owner.copy())
                self.snap_times.append(self.t)
                next_snap = next(snap_iter, None)
            if verbose and len(self.times) % 100 == 0 and self.times:
                print(f"t={self.t:6.1f}  N={self.n_states[-1]:3d}  "
                      f"Amax={self.largest_area[-1]:4d}  "
                      f"Stot={self.total_strength[-1]:5d}  "
                      f"wars={len(self.war_log)}  frags={len(self.frag_log)}")

    def _record(self):
        self.times.append(self.t)
        self.n_states.append(len(self.A))
        self.largest_area.append(max(self.A.values()) if self.A else 0)
        self.total_strength.append(sum(self.S.values()))
        self.war_count.append(len(self.war_log))


def run_baseline(L=48, T=2000.0, w=2e-3, c=5e-5, A_star=200.0,
                 n_init=40, seed=0, snap_times=None, verbose=False):
    sim = Sim(L=L, w=w, c=c, A_star=A_star,
              n_init=n_init, seed=seed, record_every=2.0)
    sim.run(T, snapshot_times=snap_times, verbose=verbose)
    return sim


if __name__ == "__main__":
    sim = run_baseline(L=40, T=300.0, verbose=True)
    print(f"Final: N={len(sim.A)}, Amax={max(sim.A.values()) if sim.A else 0}, "
          f"wars={len(sim.war_log)}, frags={len(sim.frag_log)}")
