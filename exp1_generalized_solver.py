# -*- coding: utf-8 -*-
"""exp1_generalized_solver.py — 新论文核心实验 1
E1 通用 GERT 等效 W 函数（fp 语义）+ 独立 Mason 公式符号等价验证
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
    nc = np.array([float(c) for c in sp.Poly(num, s).all_coeffs()])
    dc = np.array([float(c) for c in sp.Poly(den, s).all_coeffs()])
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


BASE = dict(p1=0.9, p2=0.15, p3=0.4, p4=0.35, p5=0.7, p6=0.2,
            l1=1.0, l2=0.5, l3=1.1, l4=1.0, l5=0.9, l6=0.4)

# ================= E1 =================
print("=" * 70)
print("E1  solver validation: independent symbolic Mason check (fp semantics)")
WE_fp = build_WE(BASE, 'fp')
W12 = W_edge(BASE['p1'], BASE['l1'])
W21 = W_edge(BASE['p2'], BASE['l2'])
W23 = W_edge(BASE['p3'], BASE['l3'])
W32 = W23
W24 = W_edge(BASE['p4'], BASE['l4'])
W45 = W_edge(BASE['p5'], BASE['l5'])
W41 = W_edge(BASE['p6'], BASE['l6'])
WE_fpm = mason_WE([(W12 * W24 * W45, frozenset('1245'))],
                  [(W12 * W21, frozenset('12')),
                   (W23 * W32, frozenset('23')),
                   (W12 * W24 * W41, frozenset('124'))])
print("  fp == independent Mason reduction:",
      sp.cancel(sp.together(WE_fp - WE_fpm)) == 0)
a_fp = analyze(WE_fp, 3.14, 6.14)
print("  fp I(3.14,6.14)=%.9f  W0=%.6f  E[T|done]=%.4f"
      % (a_fp['I'], a_fp['W0'], a_fp['ET']/a_fp['W0']))

# ================= E2 =================
print("=" * 70)
okw, sums = wellposed(BASE)
print("E2  well-posed baseline, out-sums %s  ok=%s" % (sums, okw))
WE_b = build_WE(BASE, 'fp')
a_b = analyze(WE_b, 0.0, 60.0, n=200001)
ETc = a_b['ET'] / a_b['W0']
lo_w, hi_w = round(ETc - 1.5, 2), round(ETc + 1.5, 2)
a_win = analyze(WE_b, lo_w, hi_w)
print("  E[T]=%.4f  W0=%.6f  E[T|done]=%.4f  window=[%s, %s]"
      % (a_b['ET'], a_b['W0'], ETc, lo_w, hi_w))
print("  roots min Re = %+.6f  (min Re > 0 -> proper density)"
      % np.min(a_b['roots'].real))
t0 = time.perf_counter()
a_eval = analyze(build_WE(BASE, 'fp'), lo_w, hi_w)
t_ana = time.perf_counter() - t0
N_EP = 20000
t0 = time.perf_counter()
p_done, p_win, tmean = mc_simulate(BASE, N_EP, lo_w, hi_w, seed=42)
t_mc = time.perf_counter() - t0
print("  analytic : P(done)=%.6f  P(win)=%.6f  E[T|done]=%.4f  [%.0f ms]"
      % (a_eval['W0'], a_eval['I'], a_eval['ET']/a_eval['W0'], t_ana*1000))
print("  MC(%d) : P(done)=%.6f  P(win)=%.6f  E[T|done]=%.4f" % (N_EP, p_done, p_win, tmean))
print("  |diff| : dP(done)=%.2e  dP(win)=%.2e  dE[T]=%.2e" % (abs(p_done-a_eval['W0']), abs(p_win-a_eval['I']), abs(tmean-a_eval['ET']/a_eval['W0'])))
print("  speed: analytic %.0fx faster than %d-ep MC" % (t_mc/t_ana, N_EP))

# ================= E3 =================
print("=" * 70)
print("E3  sensitivity, +0.1 perturbation (first-passage, well-posed base)")

rows = []
for key in ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
    mod = dict(BASE)
    mod[key] = BASE[key] + 0.1
    if not wellposed(mod)[0]:
        print("  %s+0.1 skipped (well-posedness)" % key)
        continue
    a_m = analyze(build_WE(mod, 'fp'), lo_w, hi_w)
    rows.append((key, a_m['I'], a_m['I'] - a_win['I']))
for key, I, dI in sorted(rows, key=lambda r: -abs(r[2])):
    print("  %s+0.1: I=%.9f  delta=%+.9f" % (key, I, dI))

# ================= E4 =================
print("=" * 70)
print("E4  route x wind grid (wind scales ALL rates: lam -> lam/kappa)")
ROUTES = {'A': (1.5, 0.30), 'B': (1.2, 0.40), 'C': (0.9, 0.50)}
grid = {}
for kappa in (1.0, 1.5, 2.0):
    for r, (l3r, p3r) in ROUTES.items():
        p = {k: (v / kappa if k.startswith('l') else v) for k, v in BASE.items()}
        p['l3'] = l3r / kappa
        p['p3'] = p3r
        grid[(kappa, r)] = analyze(build_WE(p, 'fp'), lo_w, hi_w)['I']
best = {}
for kappa in (1.0, 1.5, 2.0):
    b = max(('A', 'B', 'C'), key=lambda r: grid[(kappa, r)])
    best[kappa] = b
    print("  kappa=%.1f: " % kappa
          + "  ".join("%s=%.6f" % (r, grid[(kappa, r)]) for r in ('A', 'B', 'C'))
          + "  -> best=%s" % b)
t0 = time.perf_counter()
for kappa in (1.0, 1.5, 2.0):
    for r in ('A', 'B', 'C'):
        p = {k: (v / kappa if k.startswith('l') else v) for k, v in BASE.items()}
        p['l3'] = ROUTES[r][0] / kappa
        p['p3'] = ROUTES[r][1]
        analyze(build_WE(p, 'fp'), lo_w, hi_w)
print("  9-cell grid analytic time: %.2f s" % (time.perf_counter() - t0))
argmaxes = set(best.values())
if len(argmaxes) > 1:
    print("  >> optimal route VARIES with wind -> adaptation necessary")
else:
    print("  >> optimal route STABLE under this grid (report honestly;")
    print("     adaptation still motivated by E3 sensitivity + cost asymmetry)")
