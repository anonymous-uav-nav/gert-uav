# -*- coding: utf-8 -*-
"""exp1_generalized_solver.py — 新论文核心实验 1
E1 通用 GERT 等效 W 函数（eq2/fp 双语义）+ 回归验证
E2 良构参数集：解析 vs MC 交叉验证
E3 灵敏度
E4 航线 x 风级网格
"""
import time
from itertools import combinations

import numpy as np
import sympy as sp

s = sp.symbols('s')

# 速率属于分支而非节点：W23/W32 共用 lambda3（表 1 约定）
NODES = ['1', '2', '3', '4', '5']
SOURCE, SINK = '1', '5'
EDGES = {
    ('1', '2'): ('p1', 'l1'), ('2', '1'): ('p2', 'l2'),
    ('2', '3'): ('p3', 'l3'), ('3', '2'): ('p3', 'l3'),
    ('2', '4'): ('p4', 'l4'), ('4', '5'): ('p5', 'l5'),
    ('4', '1'): ('p6', 'l6'),
}


def W_edge(p, lam):
    P, L = sp.Rational(str(p)), sp.Rational(str(lam))
    return P * L / (L - s)


def build_WE(params, mode='fp'):
    if mode == 'eq2':
        W12 = W_edge(params['p1'], params['l1'])
        W21 = W_edge(params['p2'], params['l2'])
        W23 = W_edge(params['p3'], params['l3'])
        W32 = W_edge(params['p3'], params['l3'])
        W24 = W_edge(params['p4'], params['l4'])
        W45 = W_edge(params['p5'], params['l5'])
        W41 = W_edge(params['p6'], params['l6'])
        F = W12 * W23 * W32 * W24 * W45
        return sp.cancel(sp.together(F / (1 - (W12 * W23 * W32 * W24 * W41 + W12 * W21))))
    gm = {n: sp.Symbol('g_' + n) for n in NODES}
    gm[SINK] = sp.Integer(1)
    unk = [gm[n] for n in NODES if n != SINK]
    eqs = []
    for n in NODES:
        if n == SINK:
            continue
        expr = sp.Integer(0)
        for (i, j), (pname, lname) in EDGES.items():
            if i == n:
                expr += W_edge(params[pname], params[lname]) * gm[j]
        eqs.append(sp.Eq(gm[n], expr))
    sol = sp.solve(eqs, unk, dict=True)[0]
    return sp.cancel(sp.together(sol[gm[SOURCE]]))


def mason_WE(forward_paths, loops):
    def delta(sub):
        D = sp.Integer(1)
        for g, _ in sub:
            D -= g
        for r in range(2, len(sub) + 1):
            for comb in combinations(range(len(sub)), r):
                if all(not (sub[a][1] & sub[b][1]) for a, b in combinations(comb, 2)):
                    prod = sp.Integer(1)
                    for i in comb:
                        prod *= sub[i][0]
                    D += (-1) ** r * prod
        return D
    total = sp.Integer(0)
    for F, fnodes in forward_paths:
        sub = [L for L in loops if not (L[1] & fnodes)]
        total += F * delta(sub)
    return sp.cancel(sp.together(total / delta(loops)))


def analyze(WE, lo, hi, n=40001):
    num, den = sp.fraction(sp.cancel(sp.together(WE)))
    ncoef = sp.Poly(num, s).all_coeffs()
    dcoef = sp.Poly(den, s).all_coeffs()
    # exact rational rescale by the largest coefficient so that the float
    # conversion cannot overflow on wide-dynamic-range random networks;
    # num/den share the same scale, so roots and residues are unchanged
    scale = max(abs(sp.Rational(c)) for c in ncoef + dcoef)
    nc = np.trim_zeros(np.array([float(sp.Rational(c) / scale)
                                 for c in ncoef]), 'f')
    dc = np.trim_zeros(np.array([float(sp.Rational(c) / scale)
                                 for c in dcoef]), 'f')
    roots = np.roots(dc)
    Dp = np.polyder(dc)
    res = np.array([np.polyval(nc, z) / np.polyval(Dp, z) for z in roots])
    W0 = float(sp.N(WE.subs(s, 0)))
    ET = float(sp.N(sp.diff(WE, s).subs(s, 0)))
    xs = np.linspace(lo, hi, n)
    dens = np.sum(-res[None, :] * np.exp(-roots[None, :] * xs[:, None]), axis=1)
    I = float(np.trapz(dens, xs))
    return dict(roots=roots, res=res, W0=W0, ET=ET, I=I, xs=xs, dens=dens)


def mc_simulate(params, n_ep, lo, hi, seed=0, max_hops=500):
    succ = {}
    for (i, j), (pname, lname) in EDGES.items():
        succ.setdefault(i, []).append((j, params[pname], params[lname]))
    rng = np.random.default_rng(seed)
    n_done = n_win = 0
    tsum = 0.0
    for _ in range(n_ep):
        node, t, done = SOURCE, 0.0, False
        for _h in range(max_hops):
            outs = succ.get(node)
            if not outs:
                break
            ps = np.array([p for _, p, _ in outs])
            tot = ps.sum()
            u = rng.random()
            if u >= tot:
                break
            cum = np.cumsum(ps)
            j, _, lam = outs[int(np.searchsorted(cum, u))]
            t += rng.exponential(1.0 / lam)
            node = j
            if node == SINK:
                done = True
                break
        if done:
            n_done += 1
            tsum += t
            if lo <= t <= hi:
                n_win += 1
    return (n_done / n_ep, n_win / n_ep,
            (tsum / n_done if n_done else float('nan')))


def wellposed(params):
    out = {}
    for (i, j), (pname, _lname) in EDGES.items():
        out[i] = out.get(i, 0.0) + params[pname]
    return all(v <= 1 + 1e-12 for v in out.values()), out


PAPER_P = dict(p1=0.9, p2=0.3, p3=0.9, p4=0.9, p5=0.6, p6=0.3,
               l1=1.0, l2=0.1, l3=1.1, l4=1.0, l5=1.1, l6=0.1)
BASE = dict(p1=0.9, p2=0.15, p3=0.4, p4=0.35, p5=0.7, p6=0.2,
            l1=1.0, l2=0.5, l3=1.1, l4=1.0, l5=0.9, l6=0.4)



# ---- generalized API: any topology (used by exp3 case study) ----
def build_WE_any(EDGES, NODES, SOURCE, SINK, params):
    gm = {n: sp.Symbol('g_' + n) for n in NODES}
    gm[SINK] = sp.Integer(1)
    unk = [gm[n] for n in NODES if n != SINK]
    eqs = []
    for n in NODES:
        if n == SINK:
            continue
        expr = sp.Integer(0)
        for (i, j), (pname, lname) in EDGES.items():
            if i == n:
                expr += W_edge(params[pname], params[lname]) * gm[j]
        eqs.append(sp.Eq(gm[n], expr))
    sol = sp.solve(eqs, unk, dict=True)[0]
    return sp.cancel(sp.together(sol[gm[SOURCE]]))


def wellposed_any(EDGES, params):
    out = {}
    for (i, j), (pname, _lname) in EDGES.items():
        out[i] = out.get(i, 0.0) + params[pname]
    return all(v <= 1 + 1e-12 for v in out.values()), out
