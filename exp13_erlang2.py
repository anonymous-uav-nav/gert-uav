# -*- coding: utf-8 -*-
"""exp13_erlang2.py -- E13: distribution robustness (Erlang-2 segments).
Two-phase phase-type pipeline (one sub-state per directed edge).
"""
import numpy as np
from scipy.linalg import expm

BASE_EDGES = {
    ('1', '2'): ('p1', 'l1'), ('2', '1'): ('p2', 'l2'),
    ('2', '3'): ('p3', 'l3'), ('3', '2'): ('p3', 'l3'),
    ('2', '4'): ('p4', 'l4'), ('4', '5'): ('p5', 'l5'),
    ('4', '1'): ('p6', 'l6'),
}
BASE = dict(p1=0.9, p2=0.15, p3=0.4, p4=0.35, p5=0.7, p6=0.2,
            l1=1.0, l2=0.5, l3=1.1, l4=1.0, l5=0.9, l6=0.4)
WIN = (3.14, 6.14)

def edges_for(params, two_phase):
    E = {}
    for (i, j), (pn, ln) in BASE_EDGES.items():
        if two_phase:
            m = 'm' + i + j
            E[(i, m)] = (params[pn], 2.0 * params[ln])
            E[(m, j)] = (1.0, 2.0 * params[ln])
        else:
            E[(i, j)] = (params[pn], params[ln])
    return E

def ph_eval(E, src, sink, lo, hi):
    es = sorted(E.keys())
    idx = {e: k for k, e in enumerate(es)}
    m = len(es)
    Q = np.zeros((m, m)); qc = np.zeros(m)
    for e in es:
        i, j = e; k = idx[e]; lam = E[e][1]
        Q[k, k] = -lam
        if j == sink:
            qc[k] = lam
        else:
            for e2 in es:
                if e2[0] == j:
                    Q[k, idx[e2]] += lam * E[e2][0]
    alpha = np.array([E[e][0] if e[0] == src else 0.0 for e in es])
    B = expm(lo * Q) - expm(hi * Q)
    I = float(np.real(alpha @ np.linalg.solve(-Q, B) @ qc))
    W0 = float(np.real(alpha @ np.linalg.solve(-Q, qc)))
    ETd = float(np.real(alpha @ np.linalg.solve(-Q, np.linalg.solve(-Q, qc)) / W0))
    return dict(I=I, W0=W0, ETd=ETd)
def evaluate(params, two_phase):
    return ph_eval(edges_for(params, two_phase), '1', '5', WIN[0], WIN[1])

def rel_ranking(two_phase, label):
    a0 = evaluate(BASE, two_phase)
    rows = []
    for key in ['l1', 'l2', 'l3', 'l4', 'l5', 'l6',
                'p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
        mod = dict(BASE)
        mod[key] = BASE[key] * 1.1
        d = evaluate(mod, two_phase)['I'] - a0['I']
        rows.append((key, d))
    rows.sort(key=lambda r: -abs(r[1]))
    print("  %s +10%% top-6:" % label)
    for k, dI in rows[:6]:
        print("    %-3s dI=%+.9f" % (k, dI))
    return [k for k, _ in rows]

print("E13 Erlang-2 vs exponential (window [3.14, 6.14])")
aE = evaluate(BASE, False)
aG = evaluate(BASE, True)
print("  exponential : W0=%.6f  I=%.6f  E[T|done]=%.4f" % (aE['W0'], aE['I'], aE['ETd']))
print("  Erlang-2    : W0=%.6f  I=%.6f  E[T|done]=%.4f" % (aG['W0'], aG['I'], aG['ETd']))
print("  dI = %+.6f  (W0 identical: completion prob is distribution-free)" % (aG['I'] - aE['I']))
rE = rel_ranking(False, 'exponential')
rG = rel_ranking(True, 'Erlang-2   ')
print("  top-6 sets equal: %s" % (set(rE[:6]) == set(rG[:6])))
print("  top-1 identical (p4 dominant): %s" % (rE[0] == rG[0]))
