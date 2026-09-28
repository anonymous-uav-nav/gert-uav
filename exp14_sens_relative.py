# -*- coding: utf-8 -*-
"""exp14_sens_relative.py -- E14: sensitivity normalization + window semantics.
(a) +10% relative vs +0.1 absolute perturbation ranking (12 branch params)
(b) fixed window [3.14,6.14] vs re-centered ET|done +/- 1.5
(c) width sensitivity +/-1.0 / +/-1.5 / +/-2.0 (center = E[T|done])
"""
from _exp1lib import BASE, build_WE, analyze

WIN = (3.14, 6.14)
KEYS = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']


def perturbed(k, scheme):
    mod = dict(BASE)
    mod[k] = BASE[k] * 1.1 if scheme == 'rel' else BASE[k] + 0.1
    return mod


BASE_WE = build_WE(BASE, 'fp')

print("E14 sensitivity normalization + window semantics (fp semantics)")
b = analyze(BASE_WE, 3.14, 6.14)
I0, W00, ETs = b['I'], b['W0'], b['ET'] / b['W0']
print("  BASE: W0=%.6f I=%.6f E[T|done]=%.4f" % (W00, I0, ETs))

# ---- (a) +10% relative vs +0.1 absolute (fixed paper window) ----
print("(a) +10% relative vs +0.1 absolute, fixed window [3.14, 6.14]")
relI, absI, REL, ETd = {}, {}, {}, {}
for k in KEYS:
    W_r = build_WE(perturbed(k, 'rel'), 'fp')
    a_r = analyze(W_r, 3.14, 6.14)
    W_a = build_WE(perturbed(k, 'abs'), 'fp')
    a_a = analyze(W_a, 3.14, 6.14)
    relI[k] = a_r['I'] - I0
    absI[k] = a_a['I'] - I0
    REL[k] = W_r
    ETd[k] = a_r['ET'] / a_r['W0']

ord_r = sorted(KEYS, key=lambda k: -abs(relI[k]))
ord_a = sorted(KEYS, key=lambda k: -abs(absI[k]))
rho = 1.0 - sum((ord_r.index(k) - ord_a.index(k)) ** 2 for k in KEYS) / 286.0
print("  %-4s %12s %12s" % ('par', 'dI(+10%)', 'dI(+0.1)'))
for k in KEYS:
    print("  %-4s %+11.6f %+11.6f" % (k, relI[k], absI[k]))
print("  top-6 sets equal: %s" % (set(ord_r[:6]) == set(ord_a[:6])))
print("  top-1: rel=%s abs=%s  Spearman rho=%.3f" % (ord_r[0], ord_a[0], rho))

# ---- (b) fixed vs re-centered window ----
print("(b) fixed window vs re-centered (center follows perturbed E[T|done])")
I0_rec = analyze(BASE_WE, ETs - 1.5, ETs + 1.5)['I']
recI = {}
neg_f = neg_r = 0
for k in KEYS:
    d = analyze(REL[k], ETd[k] - 1.5, ETd[k] + 1.5)['I'] - I0_rec
    recI[k] = d
    neg_f += relI[k] < 0
    neg_r += d < 0
ord_rec = sorted(KEYS, key=lambda k: -abs(recI[k]))
print("  negative dI: fixed=%d  re-centered=%d" % (neg_f, neg_r))
print("  top-6 sets equal: %s" % (set(ord_r[:6]) == set(ord_rec[:6])))
for k in ord_r[:6]:
    print("    %-3s dI_fixed=%+.6f  dI_rec=%+.6f" % (k, relI[k], recI[k]))

# ---- (c) width sensitivity (center = E[T|done]) ----
print("(c) width sensitivity, center=%.4f" % ETs)
tops = []
for w in (1.0, 1.5, 2.0):
    I0w = analyze(BASE_WE, ETs - w, ETs + w)['I']
    rows = sorted(KEYS, key=lambda k: -abs(
        analyze(REL[k], ETs - w, ETs + w)['I'] - I0w))
    tops.append(set(rows[:3]))
    print("  w=%.1f  I0=%.6f  top-3: %s" % (w, I0w, rows[:3]))
print("  top-3 invariant across widths: %s" % (tops[0] == tops[1] == tops[2]))
