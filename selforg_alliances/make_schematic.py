"""
Schematic of the self-organising Thucydides model (4 conceptual panels):
  (A) resource-bounded growth + slope estimation + projection horizon
  (B) predictability: accuracy-limited perception of projected strength
  (C) alliance formation by balancing (salience / dissolution / bandwagon)
  (D) the Thucydides war trigger: projected crossover -> preventive war

Pure matplotlib (patches + annotations), no simulation data needed.
Writes figs/selforg_schematic.{pdf,png}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5,
                     "mathtext.fontset": "cm"})

C_RUL = "#c0392b"     # ruler / threat (red)
C_CHA = "#2e86c1"     # challenger / balancing camp (blue)
C_GRY = "#7f8c8d"
C_GRN = "#27ae60"
C_BG = "#fbfbfb"

fig = plt.figure(figsize=(8.0, 7.0))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24,
                      left=0.07, right=0.97, top=0.92, bottom=0.06)


def box(ax, x, y, text, fc="#eef3f8", ec=C_GRY, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), 0, 0, boxstyle="round,pad=0.3",
                                fc=fc, ec=ec, lw=0.8, mutation_scale=1))
    ax.text(x, y, text, fontsize=fs, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=0.8))


# ===================================================================== A
axA = fig.add_subplot(gs[0, 0])
t = np.linspace(0, 10, 400)
cap = 9.0
S = cap * (1 - np.exp(-0.45 * t)) + 0.25 * np.sin(2.1 * t) * np.exp(-0.15 * t)
axA.plot(t, S, color=C_CHA, lw=1.6, zorder=3)
axA.axhline(cap, ls="--", color=C_GRY, lw=1.0)
axA.text(0.2, cap + 0.25, r"capacity $c_i$ (resource ceiling $R$)",
         color=C_GRY, fontsize=7)
# current time, EMA slope tangent, projection over horizon h
t0 = 6.0
i0 = np.argmin(abs(t - t0))
S0 = S[i0]
m = (S[i0] - S[np.argmin(abs(t - (t0 - 1.5)))]) / 1.5   # local slope ~ EMA
h = 3.0
axA.axvspan(t0, t0 + h, color="#f4ecd8", zorder=0)
axA.plot([t0, t0 + h], [S0, S0 + m * h], color=C_RUL, lw=1.6, zorder=4)
axA.plot(t0, S0, "o", color="k", ms=4, zorder=5)
axA.plot(t0 + h, S0 + m * h, "s", color=C_RUL, ms=5, zorder=5)
axA.annotate(r"$\hat S_i(t{+}h)=S_i+h\,\widehat{\dot S}_i$",
             (t0 + h, S0 + m * h), (t0 + h - 0.2, S0 + m * h + 1.3),
             fontsize=7.2, color=C_RUL, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_RUL, lw=0.9))
axA.annotate("", (t0 + h, 1.0), (t0, 1.0),
             arrowprops=dict(arrowstyle="<->", color="#b8860b", lw=1.0))
axA.text(t0 + h / 2, 0.5, "horizon $h$", ha="center", color="#b8860b", fontsize=7)
axA.text(t0 - 1.9, S0 + 0.1, r"slope $\widehat{\dot S}_i$ (EMA)",
         color=C_RUL, fontsize=7)
axA.set(xlim=(0, 10), ylim=(0, cap + 2), xlabel="time", ylabel=r"strength $S_i$",
        title="(A)  growth, slope estimate, projection")
axA.set_xticks([]); axA.set_yticks([])

# ===================================================================== B
axB = fig.add_subplot(gs[0, 1])
axB.set_title("(B)  predictability: accuracy-limited perception")
# axis of projected strength with three rivals
xax = np.linspace(0, 10, 500)
rivals = {"weak rival": 3.2, "focal rival $j$": 5.0, "strong rival": 7.4}
for name, xv in rivals.items():
    axB.axvline(xv, ymin=0.08, ymax=0.30, color="k", lw=1.0)
    axB.text(xv, 0.02, name, rotation=0, ha="center", va="bottom", fontsize=6.6)
axB.text(0.1, 0.33, r"true projected strength $\hat S_j$  (number line)",
         fontsize=7, color="k")


def bell(mu, sig, scale, base):
    y = np.exp(-(xax - mu) ** 2 / (2 * sig ** 2))
    return base + scale * y


# high-accuracy observer (narrow) vs low-accuracy (wide), centred on focal rival
mu = rivals["focal rival $j$"]
axB.plot(xax, bell(mu, 0.35, 0.30, 0.45), color=C_GRN, lw=1.5)
axB.fill_between(xax, 0.45, bell(mu, 0.35, 0.30, 0.45), color=C_GRN, alpha=0.18)
axB.plot(xax, bell(mu, 1.45, 0.30, 0.45), color=C_RUL, lw=1.5)
axB.fill_between(xax, 0.45, bell(mu, 1.45, 0.30, 0.45), color=C_RUL, alpha=0.12)
axB.text(mu + 0.1, 0.80, "high $a_i$\n(sharp)", color=C_GRN, fontsize=6.8, ha="center")
axB.text(8.7, 0.62, "low $a_i$\n(blurred:\nmay misrank)", color=C_RUL,
         fontsize=6.8, ha="center")
# show that low-acc tail crosses the strong rival -> ranking error
axB.annotate("", (7.4, 0.52), (6.2, 0.52),
             arrowprops=dict(arrowstyle="->", color=C_RUL, lw=0.8))
axB.text(0.1, 0.93,
         r"$\tilde S^{(i)}_j=\hat S_j\,(1+\varepsilon),\ \ "
         r"\varepsilon\sim\mathcal{N}(0,\eta_i^2),\ \ \eta_i=\sigma_p(1-a_i)$",
         fontsize=7.3)
axB.set(xlim=(0, 10), ylim=(0, 1.02)); axB.axis("off")

# ===================================================================== C
axC = fig.add_subplot(gs[1, 0])
axC.set_title("(C)  alliance formation by balancing")
axC.set(xlim=(0, 10), ylim=(0, 10)); axC.axis("off")
# threat / risen power (big red) top
threat_xy = (5.0, 8.4)
axC.add_patch(Circle(threat_xy, 1.15, fc=C_RUL, ec="k", lw=1.0, alpha=0.9, zorder=3))
axC.text(*threat_xy, "risen\npower", color="w", ha="center", va="center",
         fontsize=7.5, zorder=4, fontweight="bold")
# bandwagoner (small, joins threat)
bw = (7.7, 7.2)
axC.add_patch(Circle(bw, 0.42, fc=C_RUL, ec="k", lw=0.8, alpha=0.55, zorder=3))
axC.annotate("", bw, threat_xy, arrowprops=dict(arrowstyle="-", color=C_RUL,
             lw=1.0, ls=":", shrinkA=14, shrinkB=8))
axC.text(8.9, 7.0, "bandwagon\n(rare, $p_{\\rm bw}$)", color=C_RUL, fontsize=6.4,
         ha="center")
# balancing camp (blue) bottom: several small states merging
balancers = [(2.0, 3.2), (3.4, 1.7), (5.2, 2.0), (6.7, 3.3), (4.3, 4.0)]
hub = (4.4, 2.9)
for k, b in enumerate(balancers):
    r = 0.34 + 0.06 * (k % 3)
    axC.add_patch(Circle(b, r, fc=C_CHA, ec="k", lw=0.7, alpha=0.85, zorder=3))
    if b != hub:
        axC.annotate("", hub, b, arrowprops=dict(arrowstyle="-", color=C_CHA,
                     lw=1.2, shrinkA=10, shrinkB=10))
# the two camps shaded
axC.add_patch(Rectangle((0.6, 0.7), 7.0, 3.9, fc=C_CHA, ec=C_CHA, lw=1.0,
                        alpha=0.07, zorder=0))
axC.text(1.0, 4.35, "balancing camp", color=C_CHA, fontsize=7, fontweight="bold")
# arrow from camp pointing up at threat (balances it)
axC.annotate("", (5.0, 7.1), (4.6, 4.6),
             arrowprops=dict(arrowstyle="-|>", color=C_CHA, lw=1.8))
axC.text(2.4, 6.0,
         "join the largest\nNON-threat bloc,\nsize $\\to$ match threat",
         color=C_CHA, fontsize=6.6, ha="center")
# rules text boxes
axC.text(0.2, 9.6,
         r"salience: ally only if  $\tilde\Sigma_{\rm threat}>\theta\,\tilde S_i$",
         fontsize=6.9, color="k")
axC.text(0.2, 9.0,
         r"dissolution: if own bloc $>1.8\,\tilde\Sigma_{\rm threat}\Rightarrow$ defect",
         fontsize=6.9, color="k")

# ===================================================================== D
axD = fig.add_subplot(gs[1, 1])
axD.set_title("(D)  Thucydides trigger: projected crossover")
tt = np.linspace(0, 10, 300)
S_rul = 7.0 - 0.05 * tt + 0.2 * np.sin(1.5 * tt) * np.exp(-0.1 * tt)
S_cha = 2.8 + 0.62 * tt
axD.plot(tt, S_rul, color=C_RUL, lw=1.8, label="ruler $A$ (now stronger)")
axD.plot(tt, S_cha, color=C_CHA, lw=1.8, label="challenger $B$ (rising)")
now = 4.0
hh = 4.2
axD.axvspan(now, now + hh, color="#f4ecd8", zorder=0)
axD.axvline(now, color="k", lw=0.8, ls=":")
axD.text(now, 0.4, "now", fontsize=6.8, ha="center")
axD.text(now + hh / 2, 0.4, "horizon $h$", fontsize=6.8, ha="center", color="#b8860b")
# projected lines (dashed continuation of local slope)
mr = (S_rul[np.argmin(abs(tt - now))] - S_rul[np.argmin(abs(tt - (now - 1)))])
mc = (S_cha[np.argmin(abs(tt - now))] - S_cha[np.argmin(abs(tt - (now - 1)))])
Sr0 = S_rul[np.argmin(abs(tt - now))]; Sc0 = S_cha[np.argmin(abs(tt - now))]
tf = np.linspace(now, now + hh, 50)
axD.plot(tf, Sr0 + mr * (tf - now), color=C_RUL, ls="--", lw=1.1)
axD.plot(tf, Sc0 + mc * (tf - now), color=C_CHA, ls="--", lw=1.1)
tx = now + (Sr0 - Sc0) / (mc - mr)
Sx = Sc0 + mc * (tx - now)
axD.plot(tx, Sx, "*", color="k", ms=12, zorder=5)
axD.annotate("projected\ncrossover", (tx, Sx), (tx - 2.4, Sx + 1.6),
             fontsize=6.8, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.9))
axD.annotate("preventive war\nfires here", (now + 0.5, Sr0 - 0.3),
             (now + 0.6, 1.6), color=C_RUL, fontsize=6.8, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_RUL, lw=1.0))
axD.text(0.2, 9.0,
         r"$\lambda=w\cdot\mathrm{parity}\cdot[1+e^{-\kappa c}]^{-1}$",
         fontsize=7.4)
axD.text(0.2, 8.2,
         r"$c=h(\widehat{\dot S}_B-\widehat{\dot S}_A)/(S_A{+}S_B)$",
         fontsize=7.0, color=C_GRY)
axD.legend(loc="lower right", fontsize=6.3, framealpha=0.9)
axD.set(xlim=(0, 10.2), ylim=(0, 9.8), xlabel="time",
        ylabel="coalition strength")
axD.set_xticks([]); axD.set_yticks([])

fig.suptitle("Self-organising Thucydides trap — model schematic",
             fontsize=11, y=0.975)
fig.savefig("figs/selforg_schematic.pdf")
fig.savefig("figs/selforg_schematic.png", dpi=160)
print("wrote figs/selforg_schematic.pdf and .png")
