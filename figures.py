"""
Figures for the write-up. Reads the saved .npz files, no GPU needed.

pip install matplotlib
python figures.py

Writes fig1_positions.png and fig2_refusal.png.
"""

import numpy as np
import matplotlib.pyplot as plt

FILES = [("R1-Distill-Qwen-1.5B", "results.npz"),
         ("Qwen3-1.7B", "results_qwen3.npz")]


def positions(d):
    s, h = d["safe"], d["harmful"]
    sr, hr = s.mean(0), h.mean(0)
    ax = hr - sr
    p = lambda X: ((X - sr) @ ax) / float(ax @ ax)
    return p(s), p(h), p(d["jailbreak"])


# ---- Figure 1: where the three groups sit, both models ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
rng = np.random.default_rng(0)

for ax, (name, path) in zip(axes, FILES):
    d = np.load(path)
    ps, ph, pj = positions(d)
    for i, (label, vals, colour) in enumerate([
            ("benign", ps, "#1D9E75"),
            ("harmful", ph, "#D85A30"),
            ("jailbreak", pj, "#BA7517")]):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=6, alpha=0.35,
                   color=colour, edgecolors="none")
        ax.plot([vals.mean()], [i], marker="|", markersize=28, mew=2.5,
                color=colour)
    ax.axvline(0, color="#999", lw=0.6, ls=":")
    ax.axvline(1, color="#999", lw=0.6, ls=":")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["benign", "harmful", "jailbreak"])
    ax.set_xlabel("position on benign(0) to harmful(1) axis")
    ax.set_title(name, fontsize=11)
    ax.set_xlim(-1.2, 2.2)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("fig1_positions.png", dpi=200, bbox_inches="tight")
print("wrote fig1_positions.png")


# ---- Figure 2: refused vs complied, Qwen3 ------------------------------
d = np.load("results_qwen3.npz")
ps, ph, pj = positions(d)
jb_ref = d["refused_jailbreak"].astype(bool)
h_ref = d["refused_harmful"].astype(bool)

fig, ax = plt.subplots(figsize=(6.5, 4))
groups = [("jailbreak\ncomplied", pj[~jb_ref]), ("jailbreak\nrefused", pj[jb_ref]),
          ("harmful\ncomplied", ph[~h_ref]), ("harmful\nrefused", ph[h_ref])]

for i, (label, vals) in enumerate(groups):
    colour = "#BA7517" if "jailbreak" in label else "#D85A30"
    jitter = rng.normal(0, 0.07, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.4,
               color=colour, edgecolors="none")
    ax.plot([i], [vals.mean()], marker="_", markersize=26, mew=2.5, color="black")
    ax.text(i, vals.mean() + 0.16, f"{vals.mean():.3f}", ha="center", fontsize=9)
    ax.text(i, -0.55, f"n={len(vals)}", ha="center", fontsize=8, color="#666")

ax.set_xticks(range(4))
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("position on benign(0) to harmful(1) axis")
ax.set_title("Qwen3-1.7B: refused prompts sit higher\n"
             "jailbreaks p=0.0067, harmful p=0.0161 (permutation)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("fig2_refusal.png", dpi=200, bbox_inches="tight")
print("wrote fig2_refusal.png")