# -*- coding: utf-8 -*-
# exp7_robustness.py -- E7: robustness of the E1-E3 conclusions across
# three parameter groups:
#   baseline   : the well-posed reference parameter set (L.BASE)
#   strong-wind: all branch rates halved (lambda -> lambda/2, kappa=2.0)
#   high-loss  : p4, p6 each reduced by 0.1; node-2/node-4 out-sums then
#                restored to baseline (0.9) by scaling the remaining
#                branch probabilities, so abort mass is unchanged.
# For each group: symbolic Mason equivalence; analytic vs Monte Carlo
# (5 seeds x 20k episodes, mean +/- std) for P(done), E[T|done], I;
# +0.1 sensitivity ranking.

import numpy as np
import sympy as sp

import _exp1lib as L

GROUPS = {}
GROUPS['baseline'] = dict(L.BASE)
GROUPS['strong-wind'] = {k: (v * 0.5 if k.startswith('l') else v)
                         for k, v in L.BASE.items()}
hl = dict(L.BASE)
hl['p4'] -= 0.1                       # 0.35 -> 0.25
hl['p6'] -= 0.1                       # 0.20 -> 0.10
# restore node-2 and node-4 out-sums to baseline 0.9 (same abort mass):
for node in ('2', '4'):
    out_p = [pn for (i, j), (pn, _) in L.EDGES.items()
             if i == node]
    cur = sum(hl[pn] for pn in out_p)
    target = sum(L.BASE[pn] for pn in out_p)
    for pn in out_p:
        hl[pn] *= target / cur
GROUPS['high-loss'] = hl

SEEDS = (1, 2, 3, 4, 5)
N_EP = 20000


def run_group(name, params, window=None):
    we = L.build_WE(params, 'fp')
    mason = L.mason_WE(
        forward_paths=[(L.W_edge(params['p1'], params['l1'])
                        * L.W_edge(params['p4'], params['l4'])
                        * L.W_edge(params['p5'], params['l5']),
                        {1, 2, 4})],
        loops=[(L.W_edge(params['p1'], params['l1'])
                * L.W_edge(params['p2'], params['l2']), {1, 2}),
               (L.W_edge(params['p3'], params['l3'])
                * L.W_edge(params['p3'], params['l3']), {2, 3}),
               (L.W_edge(params['p1'], params['l1'])
                * L.W_edge(params['p4'], params['l4'])
                * L.W_edge(params['p6'], params['l6']), {1, 2, 3, 4})])
    mason_ok = (sp.simplify(we - mason) == 0)

    a_full = L.analyze(we, 0.0, 30.0)
    et = a_full['ET'] / a_full['W0']
    if window is None:
        lo, hi = max(0.0, et - 1.5), et + 1.5
    else:
        lo, hi = window
    a = L.analyze(we, lo, hi)
    mc = np.array([L.mc_simulate(params, N_EP, lo, hi, seed=s)
                   for s in SEEDS])
    mc_mean = mc.mean(axis=0)
    mc_std = mc.std(axis=0, ddof=1)
    se = mc_std / np.sqrt(len(SEEDS))

    sens = {}
    for key in sorted(params):
        if not key.startswith(('p', 'l')):
            continue
        q = dict(params)
        q[key] = params[key] + 0.1
        sens[key] = L.analyze(L.build_WE(q, 'fp'), lo, hi)['I'] - a['I']
    print("\n=== group: %s ===" % name)
    print("  well-posed out-sums:", L.wellposed(params))
    print("  Mason symbolic equivalence: %s" % ("PASS" if mason_ok else "FAIL"))
    print("  window = [%.2f, %.2f]" % (lo, hi))
    print("  analytic : P(done)=%.6f  E[T|done]=%.4f  I=%.6f"
          % (a['W0'], et, a['I']))
    print("  MC mean  : P(done)=%.6f  E[T|done]=%.4f  I=%.6f"
          % (mc_mean[0], mc_mean[2], I_mc := mc_mean[1]))
    print("  MC std   : %.6f / %.4f / %.6f (n=5 seeds)"
          % (mc_std[0], mc_std[2], mc_std[1]))
    zI = abs(a['I'] - I_mc) / se[1] if se[1] > 0 else 0.0
    print("  |analytic-MC| (I): %.2f SE units" % zI)
    rank = sorted(sens.items(), key=lambda kv: -kv[1])
    print("  sensitivity top-6: "
          + ", ".join("%s=%+.4f" % (k, v) for k, v in rank[:6]))
    return dict(name=name, lo=lo, hi=hi, a=a, et=et, mc_mean=mc_mean,
                mc_std=mc_std, zI=zI, rank=rank)


results = [run_group(n, p, window=(3.14, 6.14) if n == 'baseline' else None)
           for n, p in GROUPS.items()]

print("\nE7 summary (analytic vs MC):")
for r in results:
    print("  %-12s I=%.6f  MC I=%.6f+-%.6f  z=%.2f  top1=%s"
          % (r['name'], r['a']['I'], r['mc_mean'][1], r['mc_std'][1],
             r['zI'], r['rank'][0][0]))
