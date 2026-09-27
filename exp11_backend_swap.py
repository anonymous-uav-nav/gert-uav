# -*- coding: utf-8 -*-
# exp11_backend_swap.py -- E11 (core comparison): the SAME tabular
# Q-learning agent, only the reward backend differs:
#   gert : analytic window probability (build W_E + residues), ~60 ms;
#   mc   : fresh 20,000-episode Monte Carlo per query, ~0.65 s,
#          sigma ~ 2e-3 in I.
# Measured: per-query latency, total wall-clock for 2,000 episodes,
# learned policy vs oracle, and the accuracy-parity cost of MC.
# Note: E5's published run pre-caches the 24-entry analytic R-table
# (1.4 s total) and then learns for free; this experiment deliberately
# disables caching on BOTH backends so per-query cost is the honest
# apples-to-apples comparison.

import time
import numpy as np

import _exp1lib as L

LO, HI = 3.14, 6.14
BOOST = 1.25
WINDS = (1.0, 1.5, 2.0, 3.0)
ACTS = ('none', 'l1', 'l3', 'l4', 'l5', 'l6')
N_EP = 2000


def wind_params(kappa):
    return {k: (v / kappa if k.startswith('l') else v)
            for k, v in L.BASE.items()}


def apply_act(params, act):
    p = dict(params)
    if act != 'none':
        p[act] = p[act] * BOOST
    return p


def backend_gert(kappa, act):
    pm = apply_act(wind_params(kappa), act)
    return L.analyze(L.build_WE(pm, 'fp'), LO, HI)['I']


def backend_mc(kappa, act, n_ep=20000):
    pm = apply_act(wind_params(kappa), act)
    return L.mc_simulate(pm, n_ep, LO, HI)[1]


# oracle + margin analysis (analytic ground truth)
R = {(s, a): backend_gert(s, a) for s in WINDS for a in ACTS}
oracle = {s: max(ACTS, key=lambda x: R[(s, x)]) for s in WINDS}
print("oracle:", oracle)
for s in WINDS:
    ranked = sorted(ACTS, key=lambda x: -R[(s, x)])
    margin = R[(s, ranked[0])] - R[(s, ranked[1])]
    print("  kappa=%.1f: best=%s margin over runner-up = %.2e"
          % (s, ranked[0], margin))


def learn(backend, alpha=0.5, eps=0.1, seed=7):
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in WINDS for a in ACTS}
    t_eval = 0.0
    for _ep in range(N_EP):
        s = WINDS[int(rng.integers(0, 4))]
        if rng.random() < eps:
            a = ACTS[int(rng.integers(0, len(ACTS)))]
        else:
            a = max(ACTS, key=lambda x: Q[(s, x)])
        t0 = time.perf_counter()
        r = backend(s, a)
        t_eval += time.perf_counter() - t0
        Q[(s, a)] += alpha * (r - Q[(s, a)])
    learned = {s: max(ACTS, key=lambda x: Q[(s, x)]) for s in WINDS}
    return learned, t_eval


print("\nE11: backend swap, %d episodes, no caching" % N_EP)
results = {}
for name, be in (('gert', backend_gert), ('mc', backend_mc)):
    t0 = time.perf_counter()
    learned, t_eval = learn(be)
    wall = time.perf_counter() - t0
    match = sum(learned[s] == oracle[s] for s in WINDS)
    value = sum(R[(s, learned[s])] for s in WINDS)
    results[name] = dict(learned=learned, wall=wall, t_eval=t_eval,
                         match=match, value=value)
    print("  %-4s backend: wall=%7.1f s   eval/query=%6.1f ms   "
          "oracle match=%d/4   policy value=%.6f"
          % (name, wall, t_eval / N_EP * 1e3, match, value))
    print("      learned:", learned)

g, m = results['gert'], results['mc']
print("\n  wall-clock ratio mc/gert = %.1fx" % (m['wall'] / g['wall']))
print("  single-query ratio (E2 anchor 0.65 s / 58 ms) = 11x")
# accuracy parity: MC sigma at 20k episodes vs the thinnest action margin
margin = min(R[(s, sorted(ACTS, key=lambda x: -R[(s, x)])[0])]
             - R[(s, sorted(ACTS, key=lambda x: -R[(s, x)])[1])]
             for s in WINDS)
I_typ = np.mean([R[(s, a)] for s in WINDS for a in ACTS])
n_mc = 20000
sig_mc = np.sqrt(I_typ * (1 - I_typ) / n_mc)
n_parity = int(I_typ * (1 - I_typ) / (margin / 2) ** 2)
print("  thinnest margin = %.2e ; MC sigma(20k) = %.2e -> margin is "
      "%.1f sigma BELOW noise" % (margin, sig_mc, margin / sig_mc))
print("  MC episodes needed for 2-sigma separation: ~%d (%.0fx more, "
      "~%.0f s per query)" % (n_parity, n_parity / n_mc,
                              n_parity / n_mc * 0.65))
