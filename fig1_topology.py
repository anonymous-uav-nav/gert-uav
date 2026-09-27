# -*- coding: utf-8 -*-
# fig1_topology.py -- Fig 1: reference network topologies.
# Left: 5-node delivery. Right: 6-node inspection (alternate waypoint 6).

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 600,
                     'savefig.bbox': 'tight'})

# node label -> (x, y)
P5 = {'1': (0.0, 0.0), '2': (1.0, 0.0), '3': (1.0, 1.0),
      '4': (2.0, 0.0), '5': (3.0, 0.0)}
P6 = {k: (x + 4.4, y) for k, (x, y) in P5.items()}
P6['6'] = (5.4, 1.0)

E5 = [('1', '2', 'p1,l1'), ('2', '1', 'p2,l2'),
      ('2', '3', 'p3,l3'), ('3', '2', ''),
      ('2', '4', 'p4,l4'), ('4', '5', 'p5,l5'),
      ('4', '1', 'p6,l6')]
E6 = E5 + [('2', '6', 'p7,l7'), ('6', '4', 'p8,l8')]

RAD = {('1', '2'): 0.18, ('2', '1'): 0.18,
       ('2', '3'): 0.18, ('3', '2'): 0.18,
       ('2', '6'): 0.22, ('6', '4'): 0.22}


def draw(ax, pos, edges, title):
    for (i, j, lab) in edges:
        r = RAD.get((i, j), 0.0) or (0.0 if RAD.get((j, i)) else 0.06)
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.1,
                                    connectionstyle='arc3,rad=%s' % r))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if lab:
            dy = 0.13 if r else -0.14
            ax.text(mx, my + dy, lab, ha='center', va='center',
                    fontsize=8)
    for n, (x, y) in pos.items():
        c = plt.Circle((x, y), 0.17, fc='white', ec='k', zorder=3)
        ax.add_patch(c)
        ax.text(x, y, n, ha='center', va='center',
                fontsize=10, zorder=4)
    ax.set_title(title)
    ax.set_xlim(-0.6, 8.2)
    ax.set_ylim(-0.9, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')


fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
draw(axes[0], P5, E5, 'delivery (5 nodes)')
draw(axes[1], P6, E6, 'inspection (6 nodes)')
fig.savefig('fig1_topology.png')
print('fig1 done')
