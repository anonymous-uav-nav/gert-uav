# -*- coding: utf-8 -*-
# bench_eval.py -- decompose the per-evaluation cost to reconcile the
# E2 58 ms anchor with the E10 142 ms/action measurement.
import time
import numpy as np
import _exp1lib as L

LO, HI = 3.14, 6.14
BOOST = 1.25


def wind_params(kappa):
    return {k: (v / kappa if k.startswith('l') else v)
            for k, v in L.BASE.items()}


def timed(fn, n=10):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


for kappa in (1.0, 2.0):
    p0 = wind_params(kappa)
    pm = dict(p0); pm['l5'] = pm['l5'] * BOOST

    # cold full eval (includes sympy import warm-up done above)
    t0 = time.perf_counter()
    WE = L.build_WE(pm, 'fp')
    t_build_cold = (time.perf_counter() - t0) * 1e3

    # repeated analyze on the prebuilt WE (residue pipeline only)
    ta = timed(lambda: L.analyze(WE, LO, HI), n=20)
    # repeated full eval: build + analyze
    tf = timed(lambda: L.analyze(L.build_WE(pm, 'fp'), LO, HI), n=10)

    print('kappa=%.1f  build_WE cold=%.1f ms' % (kappa, t_build_cold))
    print('  analyze only : median=%.1f ms  min=%.1f' % (np.median(ta), min(ta)))
    print('  full eval     : median=%.1f ms  min=%.1f  max=%.1f'
          % (np.median(tf), min(tf), max(tf)))

    tb = timed(lambda: L.build_WE(pm, 'fp'), n=10)
    print('  build warm    : median=%.1f ms' % np.median(tb))
