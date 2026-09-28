# -*- coding: utf-8 -*-
"""exp15_closedform.py -- E15: closed-form window integral cross-check.
The residue density is a sum of exponentials, so the window integral has
an elementary closed form I = sum_k c_k (e^{-z_k lo} - e^{-z_k hi}) / z_k.
Compares it against the grid quadrature used throughout the paper.
"""
import numpy as np
from _exp1lib import build_WE, analyze, BASE

CASES = [
    ("BASE [3.14,6.14]", BASE, 3.14, 6.14),
    ("BASE [3.64,5.64]", BASE, 3.64, 5.64),
    ("BASE [2.64,6.64]", BASE, 2.64, 6.64),
    ("BASE p4*1.1", dict(BASE, p4=0.385), 3.14, 6.14),
    ("BASE l3*1.1", dict(BASE, l3=1.21), 3.14, 6.14),
    ("BASE p2*1.1", dict(BASE, p2=0.165), 3.14, 6.14),
]

print("E15 closed-form window integral vs grid quadrature (40001-pt trapz)")
worst = 0.0
for name, params, lo, hi in CASES:
    a = analyze(build_WE(params, 'fp'), lo, hi)
    c = -a['res']
    I_cf = float(np.real(np.sum(
        c * (np.exp(-a['roots'] * lo) - np.exp(-a['roots'] * hi)) / a['roots'])))
    d = abs(I_cf - a['I'])
    worst = max(worst, d)
    print("  %-18s trapz=%.12f  closed=%.12f  |diff|=%.2e"
          % (name, a['I'], I_cf, d))
print("  max |diff| = %.2e" % worst)
