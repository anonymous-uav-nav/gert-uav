# -*- coding: utf-8 -*-
# exp12_rrt.py -- E12: simplified RRT* under the same wind field and the
# same operational window. Setting: the recon excursion geometry is a
# planning variable -- the recon waypoint z lies in an allowed recon
# region R (x in [0.6,1.4], y in [0.3,1.5]); the mission route is
# S(0,0) -> W2(1,0) -> z -> W2 -> W4(2,0) -> G(3,0). Wind kappa divides
# cruise speed, so each segment's exponential time has rate
# lambda = v / (kappa * length). RRT* (over the 2D z-configuration
# space) minimizes route length; the analytic GERT evaluator computes
# the resulting window probability and, by enumeration over a z-grid,
# the window-optimal z. Reported: RRT* vs analytic z, window
# probabilities, runtimes, and what RRT* structurally cannot do.

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _exp1lib as L

plt.rcParams.update({'font.size': 10, 'figure.dpi': 600,
                     'savefig.bbox': 'tight'})

LO, HI = 3.14, 6.14
V = 1.0
S, W2, W4, G = (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def route_lambda3(z, kappa):
    """Rate of the recon excursion branches (2->3->2): both legs share
    the recon leg length, rate-by-branch convention."""
    return V / (kappa * dist(W2, z))


def window_I(z, kappa):
    p = dict(L.BASE)
    p['l3'] = route_lambda3(z, kappa)
    return L.analyze(L.build_WE(p, 'fp'), LO, HI)['I']


def simple_rrt_star(kappa, n=2000, step=0.25, gamma=1.0, seed=1):
    """RRT* over the z-plane (recon waypoint position): cost = route
    travel time under wind = kappa * total length / v."""
    rng = np.random.default_rng(seed)

    def cost(z):
        d = (dist(S, W2) + dist(W2, z) + dist(z, W2) + dist(W2, W4)
             + dist(W4, G))
        return kappa * d / V

    pts = [((0.95, 0.35), None)]                     # (point, parent)
    cst = [cost((0.95, 0.35))]
    for _ in range(n):
        q = (rng.uniform(0.6, 1.4), rng.uniform(0.3, 1.5))
        d = np.array([dist(q, p[0]) for p in pts])
        i_near = int(np.argmin(d))
        z = pts[i_near][0]
        qn = (z[0] + step * (q[0] - z[0]) / max(d[i_near], 1e-9),
              z[1] + step * (q[1] - z[1]) / max(d[i_near], 1e-9))
        qn = (min(max(qn[0], 0.6), 1.4), min(max(qn[1], 0.3), 1.5))
        c_new = cost(qn)
        r = gamma * np.sqrt(np.log(len(pts) + 1))
        near = [i for i in range(len(pts))
                if dist(qn, pts[i][0]) <= r and cst[i] > c_new]
        parent = i_near
        best = c_new
        for i in near:
            c_via = cst[i] + kappa * dist(pts[i][0], qn) / V
            if c_via < best:
                best, parent = c_via, i
        pts.append((qn, parent))
        cst.append(best)
    best_i = int(np.argmin(cst))
    z_best = pts[best_i][0]
    # walk back a few rewires for a cleaner node
    return z_best, min(cst)


print("E12: simplified RRT* vs analytic window-optimal recon placement")
for kappa in (1.0, 2.0):
    t0 = time.perf_counter()
    z_rrt, t_cost = simple_rrt_star(kappa)
    t_rrt = (time.perf_counter() - t0) * 1e3

    # analytic window-optimal z by grid enumeration (GERT-in-the-loop)
    t0 = time.perf_counter()
    zg = [(x, y) for x in np.arange(0.6, 1.41, 0.04)
          for y in np.arange(0.3, 1.51, 0.04)]
    Is = [window_I(z, kappa) for z in zg]
    i_opt = int(np.argmax(Is))
    t_enum = (time.perf_counter() - t0) * 1e3
    z_opt = zg[i_opt]

    I_rrt = window_I(z_rrt, kappa)
    print("\n  kappa=%.1f" % kappa)
    print("    RRT* (2000 samples)  : z=(%.2f,%.2f)  route cost=%.3f  "
          "I=%.6f  runtime=%.0f ms"
          % (z_rrt[0], z_rrt[1], t_cost, I_rrt, t_rrt))
    print("    window-optimal (grid): z=(%.2f,%.2f)  I=%.6f  "
          "runtime=%.0f ms (%d evals)"
          % (z_opt[0], z_opt[1], Is[i_opt], t_enum, len(zg)))
    print("    gap I_opt - I_RRT* = %+.6f ; RRT* cost obj = length "
          "only (no probability in its state space)" % (Is[i_opt] - I_rrt))
    print("    minimal-length recon z is the nearest legal point to W2; "
          "shortest-leg monotonicity => RRT* finds length-optimal z fast")
