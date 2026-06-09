# CLAUDE.md

This file is auto-loaded by Claude Code when running in this repo. It documents
the project so Claude can be productive immediately.

## What this is

A minimal two-parameter Gillespie process on a 2D periodic lattice that
reproduces the qualitative phenomenology of the Thucydides trap: power-transition
wars, rise-and-fall cycles of dominant powers, heavy-tailed war-size
distributions, multipolar-to-hegemonic transitions. The project includes:

- The core simulator in Python.
- A JavaScript port embedded inside a standalone interactive applet (`applet.html`).
- A PRL-style manuscript and an empirical-evidence note in LaTeX.
- Empirical analyses confronting the model with COW data and with hand-coded
  Warring States China + Hellenistic Diadochi catalogues.

## Model summary

State *i* on a periodic L×L lattice occupies a territory T_i with area A_i and
integer strength S_i. Three reaction channels:

- **R1 Growth:** `S_i -> S_i + 1` at rate `max(0, C_i - S_i)` with Khaldun
  capacity `C_i = A_i / (1 + A_i/A*)`.
- **R2 War (i↔j):** rate `w * B_ij * min(S_i, S_j)`. Winner ~ `S_i/(S_i+S_j)`;
  loser concedes one boundary cell weighted by shared edges with the winner
  (bond-percolation rule); both sides pay `½ min(S_i, S_j)` strength.
- **R3 Fragmentation:** rate `c * A_i * (1 - S_i/A_i)+`. Two random seeds inside
  T_i; cells reassigned by closer-seed Voronoi; both halves start with S=0.

Free parameters with r=1, ξ=½, A*=L²/5: **w (war prefactor)** and **c (fragility
prefactor)**. Default: w=2e-3, c=5e-5.

## File layout

```
.
├── README.md                          public-facing overview
├── CLAUDE.md                          this file (Claude Code context)
├── LICENSE                            MIT
├── requirements.txt                   Python deps for empirical analyses
├── .gitignore                         excludes DataSets/ and pickle caches
├── index.html                         GitHub Pages landing page
├── applet.html                        standalone interactive simulator
│                                      (contains a JS port of the Sim class — keep
│                                       in sync with thucydides_model.py)
├── thucydides_model.py                core simulator. The Sim dataclass + Gillespie
│                                      direct-method loop.  Run-cache pickles go to
│                                      baseline_v4_*.pkl alongside the source.
├── make_figures.py                    paper figures 1-4. Caches sim runs to pickles.
├── data_analysis.py                   COW/MID/NMC/Brecke loaders + figs 5-7
├── data_frag_cycles.py                fragmentation + cycles figures (fig_frag, fig_cycles)
├── data_tests.py                      model assumption tests on COW (fig8)
├── thucydides_evidence.py             formal 5-test pipeline producing the note's Fig.1
├── ancient_thucydides.py              Warring States + Diadochi hand-coded replication
├── verify.py                          simulator self-tests
├── paper/                             PRL manuscript (thucydides_prl.tex + .pdf)
├── note/                              evidence note (thucydides_evidence_note.tex + .pdf)
├── figs/                              all generated figures (.pdf)
└── DataSets/                          (NOT TRACKED — see README for download links)
```

## Key commands

```bash
# Self-tests for the simulator (area conservation, boundary cache, causality)
python3 verify.py

# Quick demo run (prints final N, Amax, wars, frags)
python3 thucydides_model.py

# Paper figures 1-4 (takes a few minutes; uses cached pickles if present)
python3 make_figures.py

# Empirical figures (require DataSets/ — see README for downloads)
python3 data_analysis.py
python3 data_frag_cycles.py
python3 data_tests.py
python3 thucydides_evidence.py
python3 ancient_thucydides.py             # self-contained, no DataSets/ needed

# Rebuild the PDF manuscript
cd paper && pdflatex thucydides_prl.tex && pdflatex thucydides_prl.tex
cd note  && pdflatex thucydides_evidence_note.tex && pdflatex thucydides_evidence_note.tex
```

## Conventions and gotchas

- **JS-Python parity.** The `Sim` class in `applet.html` mirrors the Python
  `Sim` in `thucydides_model.py`. When the Python model changes, the JS port must
  be updated to match — otherwise the applet and the paper figures will disagree.
  A regression test script `_test_applet_v*.js` extracts the JS Sim class via
  Node and runs a short Gillespie loop; rebuild after any model change.

- **Pickle caches.** `make_figures.py` writes `baseline_v4_L<L>_T<T>_s<seed>.pkl`
  files. They are regenerable but slow (~20 s each). They are gitignored.
  Bumping the simulator version should also bump the cache prefix (currently `v4`).

- **Edge counting in `_rebuild_boundary`.** Each lattice edge is counted *once*
  (only the +x and +y neighbours of each cell are inspected). The incremental
  updates in `_transfer_site` match this convention. A previous bug double-counted
  edges — kept here as a flag so future refactors don't reintroduce it.

- **Datasets are not redistributable.** `DataSets/` is gitignored. The README
  documents which CSVs to download from where (COW, MID, NMC, Brecke).

- **LaTeX figures point to `../figs/...`** from `paper/` and `note/` — keep the
  relative path stable.

- **Baseline parameters are physically meaningful** (see paper); changing the
  defaults in `thucydides_model.py` will silently change every paper figure.
  If you want to explore a different regime, pass parameters explicitly to
  `run_baseline()` instead.

## Common Claude Code workflows

- *"regenerate the paper"* → run `make_figures.py`, then `pdflatex` twice in `paper/`.
- *"add an empirical test"* → new function inside `thucydides_evidence.py`, add a
  panel to the figure, regenerate the note.
- *"port a Python change to the applet"* → edit the matching method in the
  `Sim` class inside `applet.html`, run `_test_applet_v*.js` under Node to confirm.

## What is intentionally *out* of scope

- Alliances. The paper flags this as the natural extension; not implemented.
- Continuous strength variables. The current Sim uses integer S; a Langevin
  version is mentioned in the paper's outlook but not in the repo.
- Real-time data feeds. The empirical scripts assume static downloaded CSVs.
