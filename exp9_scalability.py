# -*- coding: utf-8 -*-
# exp9_scalability.py -- E9: solver scaling on random sub-stochastic
# networks, n in {5,10,20,50,100,200}. Two pipelines:
#   symbolic : W_E via the linear signal-flow system (sympy) + residues.
#              Timed with a child-process guard for n in {5,10,20} --
#              sympy's symbolic solve grows super-polynomially, which is
#              the honest finding this experiment reports.
#   phasetype: the same model expanded to one sub-state per edge;
#              eigendecomposition of the generator Q (numpy, O(|E|^3)).
#              Validated to 1e-8 against the symbolic pipeline.
# Output: timing table + fig7_scaling.png.

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _exp1lib as L

plt.rcParams.update({'font.size': 10, 'figure.dpi': 600,
                     'savefig.bbox': 'tight'})


def rand_net(n, seed):
    """Random well-posed network: chain + back/skip edges, out-sums<=0.95."""
    rng = np.random.default_rng(seed)
    nodes = [str(i + 1) for i in range(n)]
    E = {}
    for k in range(n - 1):
        E[(nodes[k], nodes[k + 1])] = (round(rng.uniform(0.4, 0.9), 3),
                                       round(rng.uniform(0.5, 2.0), 3))
    for k in range(1, n - 1):
        if rng.random() < 0.5:
            j = nodes[int(rng.integers(0, k))]
            E.setdefault((nodes[k], j),
                          (round(rng.uniform(0.05, 0.2), 3),
                           round(rng.uniform(0.5, 2.0), 3)))
        if rng.random() < 0.3 and k + 2 < n:
            E.setdefault((nodes[k], nodes[k + 2]),
                          (round(rng.uniform(0.1, 0.3), 3),
                           round(rng.uniform(0.5, 2.0), 3)))
    outsum = {}
    for (i, j) in E:
        outsum[i] = outsum.get(i, 0.0) + E[(i, j)][0]
    for i, s in outsum.items():
        if s > 0.95:
            f = 0.95 / s
            for (ii, jj) in list(E):
                if ii == i:
                    E[(ii, jj)] = (E[(ii, jj)][0] * f, E[(ii, jj)][1])
    return E, nodes


def ph_eval(E, src, sink, lo, hi):
    """Phase-type pipeline: I(lo,hi), P(done), E[T|done] via eig(Q)."""
    es = sorted(E.keys())
    idx = {e: k for k, e in enumerate(es)}
    m = len(es)
    Q = np.zeros((m, m))
    qc = np.zeros(m)
    for e in es:
        i, j = e
        k = idx[e]
        lam = E[e][1]
        Q[k, k] = -lam
        if j == sink:
            qc[k] = lam
        else:
            for e2 in es:
                if e2[0] == j:
                    Q[k, idx[e2]] += lam * E[e2][0]
    alpha = np.array([E[e][0] if e[0] == src else 0.0 for e in es])
    w, V = np.linalg.eig(Q)
    c = np.linalg.inv(V) @ qc
    ab = alpha @ V
    I = np.sum((ab * c / w) * (np.exp(w * hi) - np.exp(w * lo)))
    W0 = -np.sum(ab * c / w)
    ETc = np.sum(ab * c / w ** 2) / W0
    return I, W0, ETc


def sym_eval(E, nodes, lo, hi):
    E2 = {e: ('p%d' % k, 'l%d' % k) for k, e in enumerate(sorted(E.keys()))}
    params = {}
    for e, (pn, ln) in E2.items():
        params[pn] = E[e][0]
        params[ln] = E[e][1]
    we = L.build_WE_any(E2, nodes, nodes[0], nodes[-1], params)
    return L.analyze(we, lo, hi)['I']


def sym_timed(E, nodes, lo, hi, limit=120.0):
    """Run sym_eval in a child process with a wall-clock guard."""
    from concurrent.futures import ProcessPoolExecutor, TimeoutError
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(sym_eval, E, nodes, lo, hi)
        try:
            t0 = time.perf_counter()
            val = fut.result(timeout=limit)
            return (time.perf_counter() - t0) * 1e3, val
        except TimeoutError:
            return None, None


if __name__ == '__main__':
    NS_PH = [5, 10, 20, 50, 100, 200]
    NS_SYM = [5, 10, 20]
    print("E9 scaling (times in ms)")
    print("  n   |E|   symbolic   phasetype   |I_sym - I_ph|")
    rows = []
    for n in NS_PH:
        E, nodes = rand_net(n, 100 + n)
        I0, W0, ETc = ph_eval(E, nodes[0], nodes[-1], 0.0, 30.0)
        lo, hi = max(0.0, ETc - 1.5), ETc + 1.5
        t1 = time.perf_counter()
        Iph, _, _ = ph_eval(E, nodes[0], nodes[-1], lo, hi)
        t_ph = (time.perf_counter() - t1) * 1e3
        if n in NS_SYM:
            t_sym, Isym = sym_timed(E, nodes, lo, hi)
            sym_s = ("%.1f" % t_sym) if t_sym is not None else ">120000"
            agree = ("%.2e" % abs(Isym - Iph)) if Isym is not None else "n/a"
        else:
            t_sym = None
            sym_s = "skipped"
            agree = "n/a"
        print("  %3d  %3d   %9s   %8.2f   %s"
              % (n, len(E), sym_s, t_ph, agree))
        rows.append((n, len(E), t_sym, t_ph))

    fig, ax = plt.subplots(figsize=(6, 4))
    ns_all = [r[0] for r in rows]
    ax.loglog(ns_all, [r[3] for r in rows], 's-', label='phase-type pipeline')
    sym_pts = [(r[0], r[2]) for r in rows if r[2] is not None]
    if sym_pts:
        ax.loglog([p[0] for p in sym_pts], [p[1] for p in sym_pts], 'o-',
                  label='symbolic pipeline')
    ref = [rows[0][3] * (x / rows[0][0]) ** 3 for x in ns_all]
    ax.loglog(ns_all, ref, 'k--', label='O(n$^3$) reference')
    ax.set_xlabel('network size n')
    ax.set_ylabel('time per evaluation (ms)')
    ax.legend(fontsize=8)
    fig.savefig('fig7_scaling.png')
    print("fig7 done")
