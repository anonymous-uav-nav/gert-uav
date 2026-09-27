# -*- coding: utf-8 -*-
# exp8_convergence.py -- E8: Q-learning hyperparameter robustness.
# Grid alpha x eps in {0.1,0.3,0.5} x {0.05,0.1,0.2}; same tabular
# Q-learning as E5 (20,000 episodes). Metric: rolling (1000-episode)
# greedy-vs-oracle match rate. Output: fig6_convergence.png + rates.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _exp1lib as L

plt.rcParams.update({'font.size': 10,
                     'figure.dpi': 600, 'savefig.bbox': 'tight'})

LO, HI = 3.14, 6.14
BOOST = 1.25
WINDS = (1.0, 1.5, 2.0, 3.0)
ACTS = ('none', 'l1', 'l3', 'l4', 'l5', 'l6')

R = {}
for kappa in WINDS:
    p0 = {k: (v / kappa if k.startswith('l') else v) for k, v in L.BASE.items()}
    for act in ACTS:
        pm = dict(p0)
        if act != 'none':
            pm[act] = pm[act] * BOOST
        R[(kappa, act)] = L.analyze(L.build_WE(pm, 'fp'), LO, HI)['I']

oracle = {s: max(ACTS, key=lambda x: R[(s, x)]) for s in WINDS}
print("oracle:", oracle)


def learn(eps, alpha, n_ep=20000, seed=7):
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in WINDS for a in ACTS}
    greedy = [None] * n_ep
    for ep in range(n_ep):
        s = WINDS[int(rng.integers(0, 4))]
        if rng.random() < eps:
            a = ACTS[int(rng.integers(0, len(ACTS)))]
        else:
            a = max(ACTS, key=lambda x: Q[(s, x)])
        r = R[(s, a)]
        Q[(s, a)] += alpha * (r - Q[(s, a)])
        greedy[ep] = (s, max(ACTS, key=lambda x: Q[(s, x)]))
    return Q, greedy


def rolling_match(greedy, oracle, w=1000):
    m = np.array([1.0 if g == oracle[s] else 0.0 for s, g in greedy])
    return np.convolve(m, np.ones(w) / w, mode='valid')


print("E8 grid (alpha, eps): final match rate / episodes to 100% match")
fig, ax = plt.subplots(figsize=(6, 4))
for alpha in (0.1, 0.3, 0.5):
    for eps in (0.05, 0.1, 0.2):
        Q, greedy = learn(eps, alpha)
        mrate = rolling_match(greedy, oracle)
        full = np.argmax(mrate >= 1.0) if (mrate >= 1.0).any() else -1
        print("  alpha=%.1f eps=%.2f : final=%.3f  full-match@ep=%s"
              % (alpha, eps, mrate[-1], full if full >= 0 else "never"))
        ax.plot(mrate, label="a=%.1f e=%.2f" % (alpha, eps))
ax.set_xlabel("episode")
ax.set_ylabel("greedy = oracle (rolling 1000)")
ax.set_ylim(0.0, 1.05)
ax.legend(ncol=3, loc="lower right")
fig.savefig("fig6_convergence.png")
print("fig6 done")
