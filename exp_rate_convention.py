# -*- coding: utf-8 -*-
# exp_rate_convention.py -- minor-comment counterexample: rates belong to
# branches, not head nodes. Minimal excursion network V={1,2,3}, d=3:
#   1->2 (p1, lr) recon out, 2->1 (p2, lr) recon back, 1->3 (p3, lc) delivery.
# Physical truth: the two recon legs share ONE rate lr (same leg, out and
# back). Rate-by-branch encodes this; rate-by-head-node silently replaces the
# return-leg rate lr by the head node's rate (node 1 -> lc), so the analytic
# Mason expansion no longer describes the physical process. Monte Carlo
# arbitration shows the head-node convention misses the window probability.

import numpy as np
import sympy as sp

import _exp1lib as L

NODES = ['1', '2', '3']
SINK = '3'
E = {('1', '2'): ('p1', 'lr'), ('2', '1'): ('p2', 'lr'),
     ('1', '3'): ('p3', 'lc')}
P = dict(p1=0.5, p2=0.3, p3=0.4, lr=1.5, lc=0.3)

s = sp.symbols('s')


def WE(params, conv):
    """Equivalent W-function. conv='branch': edge rate = its own leg rate.
    conv='head': edge (i,j) rate = head node j's rate h_j."""
    gm = {n: sp.Symbol('g_' + n) for n in NODES}
    gm[SINK] = sp.Integer(1)
    unk = [gm[n] for n in NODES if n != SINK]
    eqs = []
    for n in NODES:
        if n == SINK:
            continue
        expr = sp.Integer(0)
        for (i, j), (pn, ln) in E.items():
            if i == n:
                r = params[ln] if conv == 'branch' else params['h_' + j]
                expr += (sp.Rational(str(params[pn]))
                         * sp.Rational(str(r)) / (sp.Rational(str(r)) - s)
                         * gm[j])
        eqs.append(sp.Eq(gm[n], expr))
    sol = sp.solve(eqs, unk, dict=True)[0]
    return sp.cancel(sp.together(sol[gm['1']]))


def mc_ground_truth(LO, HI, n_ep=400000, seed=0):
    """Physical process: BOTH recon legs Exp(lr); completion leg Exp(lc)."""
    succ = {'1': [('2', P['p1'], P['lr']), ('3', P['p3'], P['lc'])],
            '2': [('1', P['p2'], P['lr'])]}
    rng = np.random.default_rng(seed)
    win = 0
    for _ in range(n_ep):
        node, t = '1', 0.0
        for _h in range(500):
            outs = succ[node]
            ps = np.array([p for _, p, _ in outs])
            u = rng.random()
            if u >= ps.sum():
                break
            j, _, lam = outs[int(np.searchsorted(np.cumsum(ps), u))]
            t += rng.exponential(1.0 / lam)
            node = j
            if node == SINK:
                break
        if node == SINK and LO <= t <= HI:
            win += 1
    return win / n_ep


Pb = dict(P, h_1=P['lr'], h_2=P['lr'], h_3=P['lc'])  # (h_* unused in branch mode)
Ph = dict(P, h_1=P['lc'], h_2=P['lr'], h_3=P['lc'])  # head rates: node1->lc, node2->lr, node3->lc

ab = L.analyze(WE(Pb, 'branch'), 0.0, 10.0)
et = ab['ET'] / ab['W0']                      # E[T | completion]
LO, HI = max(0.0, et - 1.5), et + 1.5         # window convention of the paper
ab = L.analyze(WE(Pb, 'branch'), LO, HI)
ah = L.analyze(WE(Ph, 'head'), LO, HI)
imc = mc_ground_truth(LO, HI)

print("rate-by-branch : W0=%.6f  E[T;done]=%.4f  I=%.6f"
      % (ab['W0'], ab['ET'], ab['I']))
print("rate-by-head   : W0=%.6f  E[T;done]=%.4f  I=%.6f"
      % (ah['W0'], ah['ET'], ah['I']))
print("MC ground truth: I=%.6f (400k episodes)  window=[%.2f,%.2f]"
      % (imc, LO, HI))
print("gap branch - MC = %+.6f ;  head - MC = %+.6f"
      % (ab['I'] - imc, ah['I'] - imc))
