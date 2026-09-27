# -*- coding: utf-8 -*-
# exp2_qlearning.py -- E5: wind-adaptive parameter correction (Q-learning,
# GERT-in-the-loop). State = wind level kappa (wind divides ALL rates).
# Action = boost a single branch rate x1.25, or stay idle.
# Reward = analytic window probability I(3.14, 6.14), ms-level evaluation.

import numpy as np
import _exp1lib as L

LO, HI = 3.14, 6.14
BOOST = 1.25
WINDS = (1.0, 1.5, 2.0, 3.0)
ACTS = ('none', 'l1', 'l3', 'l4', 'l5', 'l6')


def wind_params(kappa):
    return {k: (v / kappa if k.startswith('l') else v)
            for k, v in L.BASE.items()}


def apply_act(params, act):
    p = dict(params)
    if act != 'none':
        p[act] = p[act] * BOOST
    return p


# ---- R table: 4 winds x 6 actions, precomputed analytic rewards ----
R = {}
for kappa in WINDS:
    p0 = wind_params(kappa)
    for act in ACTS:
        pm = apply_act(p0, act)
        R[(kappa, act)] = L.analyze(L.build_WE(pm, 'fp'), LO, HI)['I']
print("R table (window I):")
for kappa in WINDS:
    print("  kappa=%.1f: " % kappa
          + "  ".join("%s=%.6f" % (a, R[(kappa, a)]) for a in ACTS))


# ---- Q-learning over the cached R table (episodic, 1 decision/episode) ----
def learn_policy(n_ep=20000, alpha=0.5, eps=0.1, seed=7):
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in WINDS for a in ACTS}
    for ep in range(n_ep):
        s = WINDS[int(rng.integers(0, len(WINDS)))]
        if rng.random() < eps:
            a = ACTS[int(rng.integers(0, len(ACTS)))]
        else:
            a = max(ACTS, key=lambda x: Q[(s, x)])
        r = R[(s, a)]
        Q[(s, a)] += alpha * (r - Q[(s, a)])
    return Q


Q = learn_policy()
learned = {s: max(ACTS, key=lambda x: Q[(s, x)]) for s in WINDS}
oracle = {s: max(ACTS, key=lambda x: R[(s, x)]) for s in WINDS}

print("learned policy vs oracle:")
for s in WINDS:
    ok = "OK " if learned[s] == oracle[s] else "MISMATCH"
    print("  kappa=%.1f: learned=%s  oracle=%s  %s (Q=%.6f R=%.6f)"
          % (s, learned[s], oracle[s], ok, Q[(s, learned[s])], R[(s, oracle[s])]))

v_learned = sum(R[(s, learned[s])] for s in WINDS)
v_oracle = sum(R[(s, oracle[s])] for s in WINDS)
v_none = sum(R[(s, 'none')] for s in WINDS)
v_fixed = max(sum(R[(s, a)] for s in WINDS) for a in ACTS)
print("policy value over 4 winds:")
print("  no-adaptation      : %.6f" % v_none)
print("  best fixed action  : %.6f" % v_fixed)
print("  learned Q policy   : %.6f" % v_learned)
print("  per-state oracle   : %.6f" % v_oracle)
gain_all = v_oracle - v_none
if gain_all > 1e-12:
    frac = (v_learned - v_none) / gain_all
    print("  captured adaptation gain: %.2f%%" % (100.0 * frac))
