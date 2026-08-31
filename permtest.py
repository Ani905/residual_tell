"""
Do failed jailbreaks sit higher on the harm axis than successful ones?

Qwen3 showed refused at 0.967 vs complied at 0.803. That gap would mean the
internal position predicts whether an attack lands. But n=26 vs n=170 is
lopsided enough that chance could produce it, so test it properly.

The test: shuffle the worked/refused labels many times and see how often a gap
that large shows up by luck. If it's rare, the effect is real.

python permtest.py
"""

import numpy as np

FILES = {"R1-Distill-Qwen-1.5B": "results.npz",
         "Qwen3-1.7B": "results_qwen3.npz"}
N_PERM = 20000


def positions(d):
    """Rebuild the safe->harmful axis and place every group on it."""
    safe, harmful = d["safe"], d["harmful"]
    sr, hr = safe.mean(0), harmful.mean(0)
    axis = hr - sr
    place = lambda X: ((X - sr) @ axis) / float(axis @ axis)
    return place(safe), place(harmful), place(d["jailbreak"])


def perm_test(scores, refused, n=N_PERM, seed=0):
    """Difference between the refused group and the complied group.

    Two-sided: we care whether the groups differ, not only whether refused is
    higher. R1 pointed the other way, so a one-sided test would be dishonest.
    """
    refused = refused.astype(bool)
    if refused.sum() < 2 or (~refused).sum() < 2:
        return None
    obs = scores[refused].mean() - scores[~refused].mean()
    rng = np.random.default_rng(seed)
    k = refused.sum()
    count = 0
    for _ in range(n):
        idx = rng.permutation(len(scores))
        shuffled = scores[idx[:k]].mean() - scores[idx[k:]].mean()
        if abs(shuffled) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n + 1), int(k), int((~refused).sum())


for name, path in FILES.items():
    print("=" * 60)
    print(name)
    print("=" * 60)
    try:
        d = np.load(path)
    except FileNotFoundError:
        print(f"  {path} not found, skipping\n")
        continue

    ps, ph, pj = positions(d)

    for label, scores, flags in [
            ("jailbreaks", pj, d["refused_jailbreak"]),
            ("plain harmful", ph, d["refused_harmful"])]:
        r = perm_test(scores, flags)
        if r is None:
            print(f"  {label}: too few in one group to test")
            continue
        obs, p, n_ref, n_com = r
        print(f"  {label}:")
        print(f"    refused  {scores[flags.astype(bool)].mean():+.3f}  (n={n_ref})")
        print(f"    complied {scores[~flags.astype(bool)].mean():+.3f}  (n={n_com})")
        print(f"    difference {obs:+.3f}, permutation p = {p:.4f}"
              f"  {'<-- significant' if p < 0.05 else '<-- not significant'}")
    print()