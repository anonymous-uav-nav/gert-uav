# -*- coding: utf-8 -*-
# figs.py -- paper figures (PNG, 300 dpi).
# Fig 2: mission-time density, delivery vs inspection (E2/E6)
# Fig 3: E3 sensitivity bar chart
# Fig 4: E4 routes x wind lines
# Fig 5: E5 R-table heatmap (4 winds x 6 actions)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import _exp1lib as L
import exp2_qlearning as E2  # reuses WINDS/ACTS/R table

plt.rcParams.update({'font.size': 11,
                     'figure.dpi': 300,
                     'savefig.bbox': 'tight'})
# ---- Fig 2: density, delivery vs inspection ----
DEL_EDGES = {
    ('1', '2'): ('p1', 'l1'),
    ('2', '1'): ('p2', 'l2'),
    ('2', '3'): ('p3', 'l3'),
    ('3', '2'): ('p3', 'l3'),
    ('2', '4'): ('p4', 'l4'),
    ('4', '5'): ('p5', 'l5'),
    ('4', '1'): ('p6', 'l6'),
}
INS_EDGES = dict(DEL_EDGES)
INS_EDGES[('2', '6')] = ('p7', 'l7')
INS_EDGES[('6', '4')] = ('p8', 'l8')
INS_PARAMS = dict(L.BASE); INS_PARAMS.update(p3=0.30, p7=0.15, p8=0.90,
                                             l7=0.8, l8=1.2)

WE_del = L.build_WE_any(DEL_EDGES, L.NODES, '1', '5', L.BASE)
WE_ins = L.build_WE_any(INS_EDGES, L.NODES + ['6'],
                        '1', '5', INS_PARAMS)
a_del = L.analyze(WE_del, 0.0, 15.0)
a_ins = L.analyze(WE_ins, 0.0, 15.0)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(a_del['xs'], a_del['dens'].real, label='delivery (5-node)')
ax.plot(a_ins['xs'], a_ins['dens'].real,
        label='inspection (6-node, backup leg)')
ax.set_xlabel('mission time x')
ax.set_ylabel('density')
ax.legend()
fig.savefig('fig2_density.png')
plt.close(fig)
print('fig2 done')

# ---- Fig 3: E3 sensitivity bar chart (verified values) ----
deltas = [('p4', 0.0370), ('p5', 0.0174), ('p3', 0.0170),
          ('p1', 0.0166), ('p2', 0.0130), ('p6', 0.0030),
          ('l5', -0.0023), ('l4', -0.0019), ('l1', -0.0015),
          ('l2', 0.0014), ('l6', 0.0008), ('l3', 0.0007)]
fig, ax = plt.subplots(figsize=(6, 3.4))
names = [k for k, _ in deltas]
vals = [v for _, v in deltas]
colors = ['tab:blue' if v > 0 else 'tab:orange' for v in vals]
ax.bar(names, vals, color=colors)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('parameter (+0.1 perturbation)')
ax.set_ylabel('delta I')
fig.savefig('fig3_sensitivity.png')
plt.close(fig)
print('fig3 done')

# ---- Fig 4: E4 routes x wind ----
fig, ax = plt.subplots(figsize=(6, 3.4))
mk = {'A': 'o-', 'B': 's--', 'C': '^:'}
for r in ('A', 'B', 'C'):
    ks = sorted({k for (k, rr) in E2.R if rr == r})
    vs = [E2.R[(k, 'none')] for k in ks]
    ax.plot(ks, vs, mk[r], label='route ' + r)
ax.set_xlabel('wind level kappa')
ax.set_ylabel('window probability I')
ax.legend()
fig.savefig('fig4_routes_wind.png')
plt.close(fig)
print('fig4 done')

# ---- Fig 5: E5 R-table heatmap ----
fig, ax = plt.subplots(figsize=(7, 3.2))
M = np.array([[E2.R[(k, a)] for a in E2.ACTS] for k in E2.WINDS])
im = ax.imshow(M, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(E2.ACTS)), E2.ACTS)
ax.set_yticks(range(len(E2.WINDS)),
              ['kappa=%.1f' % k for k in E2.WINDS])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, '%.4f' % M[i, j],
                ha='center', va='center',
                color='w', fontsize=7)
fig.colorbar(im, ax=ax, label='window probability I')
fig.savefig('fig5_rtable.png')
plt.close(fig)
print('fig5 done')
