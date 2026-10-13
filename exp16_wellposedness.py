# -*- coding: utf-8 -*-
# exp16_wellposedness.py -- E16: transience verification (supervisor A1)
# + fixed-window 5-node vs 6-node comparison (supervisor A3).
#
# A1: separate the LOCAL well-posedness condition (out-sums <= 1, abort
#     reading) from the GLOBAL transience condition (rho(P) < 1, almost
#     sure termination). Verify rho(P) < 1 numerically by an eigenvalue
#     routine for both topologies, and report rho(P), P(completion),
#     P(abort).
# A3: evaluate the 6-node inspection network under the SAME operational
#     window [3.14, 6.14] as the 5-node delivery network (the E6 table
#     re-centered each network on its own conditional mean); the
#     re-centred reading stays as the supplementary view.

import numpy as np
import _exp1lib as L
from exp3_case import DELIVERY_EDGES, DELIVERY_PARAMS, \
    INSPECT_EDGES, INSPECT_PARAMS


def rho_transients(edges, params, all_nodes, sink):
    """Spectral radius of the branch-probability matrix P on the
    transient (non-sink) nodes, by numpy eigvals."""
    trans = [n for n in all_nodes if n != sink]
    idx = {n: k for k, n in enumerate(trans)}
    P = np.zeros((len(trans), len(trans)))
    for (i, j), (pn, _) in edges.items():
        if j != sink:
            P[idx[i], idx[j]] += params[pn]
    ev = np.linalg.eigvals(P)
    return float(max(abs(ev))), P


def report(name, edges, params, nodes, window):
    rho, P = rho_transients(edges, params, nodes, '5')
    WE = L.build_WE_any(edges, nodes, '1', '5', params)
    a_full = L.analyze(WE, 0.0, 200.0, n=200001)
    a = L.analyze(WE, window[0], window[1])
    W0 = a_full['W0']
    print("%s: P is %d x %d, rho(P) = %.6f  (< 1: %s)"
          % (name, P.shape[0], P.shape[1], rho, rho < 1.0))
    print("  P(completion) = W_E(0) = %.6f   P(abort) = 1 - W_E(0) = %.6f"
          % (W0, 1.0 - W0))
    print("  window %s  I = %.6f" % (window, a['I']))
    return W0


# --- A1: transience verification -------------------------------------
print("=== E16(a): transience verification (eigenvalue routine) ===")
report('delivery (5-node)', DELIVERY_EDGES, DELIVERY_PARAMS,
       ['1', '2', '3', '4', '5'], (3.14, 6.14))
report('inspection (6-node)', INSPECT_EDGES, INSPECT_PARAMS,
       ['1', '2', '3', '4', '5', '6'], (3.35, 6.35))

# Local condition does NOT imply transience: a 2-node cycle with unit
# out-sums (p = 1 on both edges) satisfies sum_j p_ij <= 1 at every node
# (with zero abort mass) but rho(P) = 1 -- the mission may never
# terminate. Numerical check:
print("\n--- counterexample: local ok, global fails ---")
CE_EDGES = {('1', '2'): ('q1', 'r1'), ('2', '1'): ('q2', 'r2'),
            ('2', '3'): ('q3', 'r3')}
CE_PARAMS = dict(q1=1.0, r1=1.0, q2=1.0, r2=1.0, q3=0.5, r3=1.0)
rho_ce, P_ce = rho_transients(CE_EDGES, CE_PARAMS,
                              ['1', '2', '3'], '3')
print("2-node unit cycle: out-sums = 1 (local condition holds), "
      "rho(P) = %.12f (>= 1: %s)" % (rho_ce, rho_ce >= 0.9999999))

# --- A3: fixed-window comparison --------------------------------------
print("\n=== E16(b): fixed-window 5-node vs 6-node comparison ===")
WE5 = L.build_WE_any(DELIVERY_EDGES, ['1', '2', '3', '4', '5'],
                     '1', '5', DELIVERY_PARAMS)
WE6 = L.build_WE_any(INSPECT_EDGES, ['1', '2', '3', '4', '5', '6'],
                     '1', '5', INSPECT_PARAMS)
FIXED = (3.14, 6.14)
I5 = L.analyze(WE5, *FIXED)['I']
I6 = L.analyze(WE6, *FIXED)['I']
a6_full = L.analyze(WE6, 0.0, 200.0, n=200001)
print("fixed window [3.14, 6.14]:  5-node I = %.6f   6-node I = %.6f"
      % (I5, I6))
print("  delta I = %+.6f  (%.1f%% relative)"
      % (I6 - I5, 100.0 * (I6 - I5) / I5))
ETc6 = a6_full['ET'] / a6_full['W0']
print("6-node conditional mean %.4f (re-centred window [3.35, 6.35], "
      "I = 0.153748 in E6) kept as supplementary reading" % ETc6)

# --- A3-R: probability-matched control --------------------------------
# The 6-node inspection network has a larger total outgoing probability
# mass (P(completion) 0.444 vs 0.343), so part of its fixed-window
# advantage could come from the extra mass rather than from the backup
# leg itself.  Control: rescale the branch probabilities of one network
# by k so that its completion probability matches the other's, then
# repeat the fixed-window comparison.  (Supervisor round 3.)
print("\n=== E16(c): probability-matched control ===")
SIX = ['1', '2', '3', '4', '5', '6']
FIVE = ['1', '2', '3', '4', '5']
W05 = L.analyze(WE5, 0.0, 200.0, n=200001)['W0']
W06 = L.analyze(WE6, 0.0, 200.0, n=200001)['W0']
print("targets: 5-node W0 = %.6f   6-node W0 = %.6f" % (W05, W06))


def scaled(params, k):
    sp = dict(params)
    for key in sp:
        if key.startswith('p'):
            sp[key] = sp[key] * k
    return sp


def W0_of(edges, nodes, params):
    WE = L.build_WE_any(edges, nodes, '1', '5', params)
    return L.analyze(WE, 0.0, 200.0, n=200001)['W0']


def match_k(edges, nodes, params, target, kmax):
    """Bisect k in (0, kmax] so that W0(edges, params*k) = target.
    Returns (k, W0); W0 is increasing in k."""
    lo, hi = 1e-6, kmax
    if W0_of(edges, nodes, scaled(params, hi)) < target:
        return None, W0_of(edges, nodes, scaled(params, hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if W0_of(edges, nodes, scaled(params, mid)) < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k, W0_of(edges, nodes, scaled(params, k))


# (i) 6-node scaled DOWN to the 5-node completion probability
k6, w6m = match_k(INSPECT_EDGES, SIX, INSPECT_PARAMS, W05, 1.0)
WE6m = L.build_WE_any(INSPECT_EDGES, SIX, '1', '5', scaled(INSPECT_PARAMS, k6))
I6m = L.analyze(WE6m, *FIXED)['I']
r6m, _ = rho_transients(INSPECT_EDGES, scaled(INSPECT_PARAMS, k6), SIX, '5')
print("(i) 6-node scaled by k = %.6f  ->  W0 = %.6f (= 5-node), "
      "rho(P) = %.6f" % (k6, w6m, r6m))
print("    fixed window I:  matched 6-node %.6f  vs  5-node %.6f  "
      "(delta %+.6f, %+.1f%% relative)"
      % (I6m, I5, I6m - I5, 100.0 * (I6m - I5) / I5))

# (ii) 5-node scaled UP toward the 6-node completion probability; the
# local condition (out-sums <= 1) caps k at 1/0.9 ~ 1.1111
KMAX5 = 1.0 / max(0.9, 0.4)
k5, w5m = match_k(DELIVERY_EDGES, FIVE, DELIVERY_PARAMS, W06, KMAX5)
if k5 is None:
    print("(ii) 5-node cannot reach W0 = %.6f: even at the local-condition "
          "cap k = %.4f the completion probability is only %.6f "
          "(local condition would be violated beyond it)"
          % (W06, KMAX5, w5m))
else:
    WE5m = L.build_WE_any(DELIVERY_EDGES, FIVE, '1', '5',
                          scaled(DELIVERY_PARAMS, k5))
    I5m = L.analyze(WE5m, *FIXED)['I']
    print("(ii) 5-node scaled by k = %.6f -> W0 = %.6f (= 6-node)"
          % (k5, w5m))
    print("    fixed window I:  scaled 5-node %.6f  vs  6-node %.6f"
          % (I5m, I6))
