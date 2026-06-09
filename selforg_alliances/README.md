# selforg_alliances — a self-organising Thucydides trap

A new, self-contained subproject (independent of the parent lattice Gillespie
model). Mean-field / graph model of the Thucydides trap from minimal principles:
resource-bounded stochastic growth, **endogenous predictability under
selection**, and **balancing alliances**. See `NOTE_selforg.md` for the writeup.

```
thucydides_selforg.py     v1 core model: SDE growth + forecasting + alliances + war
make_schematic.py         v1 model schematic (4 panels)
make_figures_selforg.py   v1 6-panel overview figure
experiments_selforg.py    v1 8-seed robustness aggregate + 2 mechanism controls

thucydides_selforg_v2.py  v2 core: dynamic gamma(t), no c_i, ring locality,
                          rollout-based deliberation (n_i), predation
make_figures_v2.py        Option A (multipolar<->bipolar precursor) + Option B figs
experiments_v2.py         Option B sweep: frozen deliberation depth n -> war stats
diag_v2.py                depth-optimum + Option-A precursor diagnostics

NOTE_selforg.tex / .md    v1 note (the lattice-free baseline model)
NOTE_v2.tex               v2 note (STANDALONE: full rules, worked example, A & B)
```

v2 in one line: dynamic growth (any state can rise), prediction by Monte-Carlo
rollout, balancing alliances on a 1-D ring, and **war-driven absorption** -- the
loser joins the winner's alliance, which is why allies fight (not to lose
partners to the enemy bloc). Results: **systemic wars break out of more bipolar
configs** (Option A); and because winning means conquest, **more collective
foresight makes states better predators -> more frequent, larger wars, shorter
peace** (Option B -- the opposite of the absorption-free variant). See
`NOTE_v2.tex` for the detailed rules and a fully worked war example.

```bash
# v2 (current)
python3 thucydides_selforg_v2.py     # CLI demo
python3 make_figures_v2.py           # Option A + Option B figures
python3 experiments_v2.py            # Option B sweep (~6 min) -> data_v2_sweep.npz

# v1 (reference)
python3 thucydides_selforg.py        # CLI demo
python3 make_figures_selforg.py      # 6-panel overview
python3 experiments_selforg.py       # robustness + controls
```

Itô convention, Euler–Maruyama. NumPy only. Knobs live in the `Params`
dataclass in `thucydides_selforg.py`.
