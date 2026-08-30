"""
How well do the safe and harmful regions actually separate?

Reads results.npz. No GPU, no model, runs in a second.

Why this is needed: in main.py the regions are built from ALL safe and ALL
harmful vectors, then those same vectors are placed on the axis. They land at
0.0 and 1.0 by arithmetic, not because the model separates them. This splits
the data - build the regions on part of it, place the rest - so the numbers
mean something.

python separation.py
"""

import numpy as np

d = np.load("results.npz")
safe, harmful, jailbreak = d["safe"], d["harmful"], d["jailbreak"]
print(f"loaded: {len(safe)} safe, {len(harmful)} harmful, {len(jailbreak)} jailbreak\n")

N_FOLDS = 5
rng = np.random.default_rng(0)


def place(X, safe_region, harmful_region):
    axis = harmful_region - safe_region
    return ((X - safe_region) @ axis) / float(axis @ axis)


# Split both groups into folds, build regions on the rest, place the held-out.
s_fold = rng.permutation(len(safe)) % N_FOLDS
h_fold = rng.permutation(len(harmful)) % N_FOLDS

held_safe, held_harmful, jb_folds = [], [], []
for f in range(N_FOLDS):
    sr = safe[s_fold != f].mean(0)
    hr = harmful[h_fold != f].mean(0)
    held_safe.append(place(safe[s_fold == f], sr, hr))
    held_harmful.append(place(harmful[h_fold == f], sr, hr))
    jb_folds.append(place(jailbreak, sr, hr))

held_safe = np.concatenate(held_safe)
held_harmful = np.concatenate(held_harmful)
jb = np.mean(np.stack(jb_folds), axis=0)

print("held-out positions on the safe(0) -> harmful(1) axis")
print(f"  safe      {held_safe.mean():+.3f}  (sd {held_safe.std():.3f})")
print(f"  harmful   {held_harmful.mean():+.3f}  (sd {held_harmful.std():.3f})")
print(f"  jailbreak {jb.mean():+.3f}  (sd {jb.std():.3f})")

# How often does a random harmful example score above a random safe one?
# 0.5 is chance, 1.0 is perfect. This is the number the whole result rests on.
wins = (held_harmful[:, None] > held_safe[None, :]).mean()
print(f"\n  separation (AUROC): {wins:.3f}")

# Gap between the groups measured in units of their own spread. Above ~2 means
# the two clouds barely overlap.
pooled = np.sqrt((held_safe.var() + held_harmful.var()) / 2)
print(f"  gap in units of spread: {(held_harmful.mean() - held_safe.mean()) / pooled:.2f}")

if wins < 0.8:
    print("\n  WARNING: weak separation. The axis is not cleanly telling safe")
    print("  from harmful, so the jailbreak number sits on shaky ground.")

# Is the jailbreak result bigger than the noise in the safe group?
print(f"\n  jailbreak mean is {(jb.mean() - held_safe.mean()) / held_safe.std():.1f}"
      f" safe-standard-deviations above the safe region")