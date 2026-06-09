# Thucydides trap — a minimal stochastic model

A two-parameter Gillespie process on a 2D lattice that produces the
qualitative phenomenology of the Thucydides trap (mutual growth → war,
rise-and-fall cycles of dominant powers, heavy-tailed war-size
distribution) from three local reaction channels and a Khaldūn-type
sustainability ceiling.

This repository contains the model, the figures, the manuscript, an
interactive HTML applet, and an empirical-evidence note that tests the
Thucydides claim against the Correlates of War (COW) data and against
hand-coded Warring States China and Hellenistic Diadochi datasets.

**[→ Try the interactive applet](https://YOUR-GITHUB-USER.github.io/Thucydides_Project/applet.html)**
(after enabling GitHub Pages — see below).

---

## Newer explorations (continuous / agent-based directions)

Two self-contained subprojects extend the lattice model toward continuous,
self-organising dynamics. Each ships its own README and LaTeX notes (with PDFs).

- **[`stepwise_progression/`](stepwise_progression/)** — the model built up one
  mechanism at a time, starting from pure resource-limited growth. Includes the
  analytic result that the leader is a driftless Bessel martingale (it always
  falls) and how the growth exponent α tunes recurrent churn → hegemonic lock-in.
  Start here: `ANALYSIS_step0.tex`, `ANALYSIS_step1.tex`.
- **[`selforg_alliances/`](selforg_alliances/)** — a mean-field/agent model with
  endogenous predictability and balancing alliances (war-driven absorption).
  See `NOTE_selforg.tex` (v1) and `NOTE_v2.tex` (current).

---

## Repository layout

```
.
├── README.md                       this file
├── LICENSE                         MIT
├── applet.html                     standalone interactive simulator (run in any browser)
├── thucydides_model.py             core Python simulator (Sim class + Gillespie loop)
├── make_figures.py                 generates paper figures 1-4
├── data_analysis.py                COW/MID/NMC loaders + figs 5-7
├── data_frag_cycles.py             fragmentation + cycles analyses
├── data_tests.py                   model-vs-data tests (outcome, hazard, Khaldun, B_ij)
├── thucydides_evidence.py          formal Thucydides test (logistic + selection-bias)
├── ancient_thucydides.py           Warring States + Diadochi replication
├── verify.py                       simulator self-tests (conservation, boundary, causality)
├── paper/                          PRL-style manuscript (.tex + .pdf)
├── note/                           empirical-evidence note (.tex + .pdf)
├── figs/                           all generated figures (.pdf)
└── DataSets/                       (NOT TRACKED — download separately, see below)
```

---

## Quick start

```bash
git clone https://github.com/YOUR-GITHUB-USER/Thucydides_Project.git
cd Thucydides_Project
pip install -r requirements.txt

# run a baseline simulation
python3 thucydides_model.py

# regenerate paper figures (takes a few minutes)
python3 make_figures.py

# run self-tests
python3 verify.py

# open the interactive applet
open applet.html        # macOS
xdg-open applet.html    # Linux
```

The applet is a single self-contained HTML file with no build step. It
runs entirely client-side (Gillespie events in JavaScript at ~300k/s).

## Continuing in Claude Code

This project ships with a `CLAUDE.md` that documents the file layout,
conventions, and key commands. After cloning, just run:

```bash
cd Thucydides_Project
claude
```

Claude Code will auto-load `CLAUDE.md` and have full context. Useful prompts:

- `regenerate the paper figures and rebuild the PDF`
- `port my latest change in thucydides_model.py into the JS applet and run the Node parity test`
- `add a new empirical test to thucydides_evidence.py for the alliance dyad subset`
- `bump the cache prefix to v5 because I changed the war damage rule`

The `CLAUDE.md` also flags the Python↔JavaScript parity invariant (the `Sim`
class in `applet.html` mirrors `thucydides_model.py`), the gitignored
`DataSets/`, and the edge-counting convention in `_rebuild_boundary` — so
Claude won't reintroduce known footguns.

---

## Reproducing the empirical analysis

The COW / MID / NMC datasets are licensed and large; they are **not**
checked in. To reproduce the empirical figures (5–8 + the fragmentation
and cycle figures + the evidence note) you need to download four files
into `DataSets/`:

| File | Source |
|---|---|
| `Inter-StateWarData_v4.0.csv` | [COW Inter-State War v4.0](https://correlatesofwar.org/data-sets/cow-war/) |
| `MID-5-Data-and-Supporting-Materials/MIDA 5.0.csv` and `MIDB 5.0.csv` | [COW MID v5.0](https://correlatesofwar.org/data-sets/mids/) |
| `NMC_Documentation-6/NMC-60-abridged.zip` (and unpack the CSV) | [COW NMC v6.0](https://correlatesofwar.org/data-sets/national-material-capabilities/) |
| `States2024/statelist2024.csv` and `MajorPowers2024/majors2024.csv` | [COW State System Membership v2024](https://correlatesofwar.org/data-sets/state-system-membership/) |
| `Conflict-Catalog-18-vars.xlsx` (optional) | [Brecke Conflict Catalog](http://brecke.inta.gatech.edu/research/conflict/) |

Then:

```bash
python3 data_analysis.py          # COW + MID baseline figures
python3 data_frag_cycles.py       # state-death and war-onset cycles
python3 data_tests.py             # outcome ~ CINC, MID escalation, Khaldun, B_ij
python3 thucydides_evidence.py    # formal 5-test pipeline (note Fig. 1)
python3 ancient_thucydides.py     # Warring States + Diadochi replication (no extra data needed)
```

The ancient script is fully self-contained — the Warring States and
Diadochi state-strength estimates and war catalogues are hand-coded
from secondary literature inside `ancient_thucydides.py`.

---

## Hosting the applet on GitHub Pages

To make the applet accessible at a URL you can share:

1. Push the repository to GitHub (instructions below).
2. In the GitHub repo settings → **Pages** → set source to **Deploy from
   a branch**, branch `main`, folder `/ (root)`. Save.
3. Wait ~30 seconds. The applet will be available at:
   `https://YOUR-GITHUB-USER.github.io/Thucydides_Project/applet.html`

Anyone with the URL can run the simulator, tune the parameters live, and
copy a share-link that encodes the current configuration.

---

## Model summary

State *i* occupies a territory *T_i* ⊂ L×L periodic lattice with area
*A_i*, integer strength *S_i*. For each neighbouring pair (i,j) the
shared boundary length is *B_ij*. Three reactions:

- **R1 Growth:** S_i → S_i+1 at rate `max(0, C_i − S_i)`
  with Khaldūn capacity `C_i = A_i / (1 + A_i/A*)`.
- **R2 War:** dyadic rate `w · B_ij · min(S_i, S_j)`. Winner drawn with
  probability `S_i/(S_i+S_j)`; loser concedes one boundary site weighted
  by shared edges (bond-percolation rule); both sides pay
  `½·min(S_i,S_j)` strength.
- **R3 Fragmentation:** rate `c · A_i · (1 − S_i/A_i)+`. Two random seeds
  inside T_i; cells reassigned by closer-seed (Voronoi); both halves
  start with S = 0 (cohesion collapse). Setting c = 0 disables R3.

Free parameters with r = 1, ξ = 1/2, A* = L²/5: **w (war prefactor)**
and **c (fragility prefactor)**.

---

## Citation

If you use this code or paper, please cite

> M. Heltberg, *A minimal stochastic model for the Thucydides trap:
> growth, tension, and war on a lattice of states* (preprint, 2026).

---

## License

MIT — see [LICENSE](LICENSE).
