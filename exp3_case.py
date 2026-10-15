# -*- coding: utf-8 -*-
# exp3_case.py -- E6 case study: delivery vs inspection scenarios.
# Same solver, two different topologies (zero solver modification).
# Delivery  = 5-node Semenov topology (BASE params).
# Inspection = 6-node topology with an alternate waypoint (node 6):
#              2->6 detour edge and 6->4 rejoin edge added.

import time
import _exp1lib as L

DELIVERY_EDGES = {
    ('1', '2'): ('p1', 'l1'), ('2', '1'): ('p2', 'l2'),
    ('2', '3'): ('p3', 'l3'), ('3', '2'): ('p3', 'l3'),
    ('2', '4'): ('p4', 'l4'), ('4', '5'): ('p5', 'l5'),
    ('4', '1'): ('p6', 'l6'),
}
DELIVERY_PARAMS = dict(L.BASE)

INSPECT_EDGES = dict(DELIVERY_EDGES)
INSPECT_EDGES[('2', '6')] = ('p7', 'l7')
INSPECT_EDGES[('6', '4')] = ('p8', 'l8')
INSPECT_PARAMS = dict(DELIVERY_PARAMS)
INSPECT_PARAMS.update(p3=0.30, p7=0.15, p8=0.90,
                      l7=0.8, l8=1.2)


def eval_case(name, edges, params):
    ok, sums = L.wellposed_any(edges, params)
    n_v = len({n for e in edges for n in e})
    n_e = len(edges)
    t0 = time.perf_counter()
    WE = L.build_WE_any(edges, ['1', '2', '3', '4', '5', '6'],
                        '1', '5', params)
    t_build = time.perf_counter() - t0
    return WE, ok, sums, n_v, n_e, t_build


CASES = [
    ('delivery', DELIVERY_EDGES, DELIVERY_PARAMS),
    ('inspection', INSPECT_EDGES, INSPECT_PARAMS),
]
for name, edges, params in CASES:
    ok, sums = L.wellposed_any(edges, params)
    WE, _, _, nv, ne, t_build = eval_case(name, edges, params)
    a = L.analyze(WE, 0.0, 200.0, n=200001)
    ETc = a['ET'] / a['W0']
    lo, hi = round(ETc - 1.5, 2), round(ETc + 1.5, 2)
    aw = L.analyze(WE, lo, hi)
    print("case: %s  |V|=%d |E|=%d  well-posed=%s  out-sums=%s"
          % (name, nv, ne, ok, {k: round(v, 2) for k, v in sums.items()}))
    print("  W_E build: %.0f ms" % (t_build * 1000))
    print("  P(done)=%.6f  E[T|done]=%.4f  window=[%s,%s]  I=%.6f"
          % (a['W0'], ETc, lo, hi, aw['I']))
    print("  roots min Re = %+.4f" % float(min(z.real for z in a['roots'])))
