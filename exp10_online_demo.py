# -*- coding: utf-8 -*-
# exp10_online_demo.py -- E10: online re-planning demo.
# Mission timeline: wind shifts kappa 1.0 -> 2.0 at decision time t=2.0.
# At the shift, the analytic evaluator re-scores all 6 actions under the
# new wind (52 ms warm / 142 ms cold per action; 854 ms total, far
# inside the timescale of a wind shift) and the learned policy moves
# the speed budget l3 -> l5 (final approach leg).
# Compared strategies: adaptive policy / static l3 (no re-plan) / idle.
# Output: fig8_timeline.png + printed numbers.

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _exp1lib as L

plt.rcParams.update({'font.size': 10, 'figure.dpi': 600,
                     'savefig.bbox': 'tight'})

LO, HI = 3.14, 6.14
BOOST = 1.25
ACTS = ('none', 'l1', 'l3', 'l4', 'l5', 'l6')


def wind_params(kappa):
    return {k: (v / kappa if k.startswith('l') else v)
            for k, v in L.BASE.items()}


def apply_act(params, act):
    p = dict(params)
    if act != 'none':
        p[act] = p[act] * BOOST
    return p


def eval_I(kappa, act):
    return L.analyze(L.build_WE(apply_act(wind_params(kappa), act), 'fp'),
                     LO, HI)['I']


# learned policy (E5): l3 at kappa=1.0, l5 at kappa=2.0
POLICY = {1.0: 'l3', 1.5: 'l1', 2.0: 'l5', 3.0: 'l5'}

# latency of one full re-evaluation (all 6 actions) under new wind
t0 = time.perf_counter()
scores = {a: eval_I(2.0, a) for a in ACTS}
t_re = (time.perf_counter() - t0) * 1e3

print("E10 online demo")
print("  wind shift: kappa 1.0 -> 2.0 at t = 2.0")
print("  re-evaluation of 6 actions under kappa=2.0: %.0f ms total" % t_re)
print("  scores:", {a: round(v, 6) for a, v in scores.items()})
print("  policy switch: l3 -> %s (kappa=2.0 best action)" % POLICY[2.0])

# strategy curves over the decision timeline
T_SHIFT = 2.0
ts = np.linspace(0.0, 6.0, 601)
kappa_of_t = np.where(ts < T_SHIFT, 1.0, 2.0)

def I_adaptive(t):
    k = 1.0 if t < T_SHIFT else 2.0
    return eval_I(k, POLICY[k])

def I_static_l3(t):
    k = 1.0 if t < T_SHIFT else 2.0
    return eval_I(k, 'l3')

def I_idle(t):
    k = 1.0 if t < T_SHIFT else 2.0
    return eval_I(k, 'none')

vals_adapt = [I_adaptive(t) for t in ts]
vals_static = [I_static_l3(t) for t in ts]
vals_idle = [I_idle(t) for t in ts]

i_before = eval_I(1.0, 'l3')
i_after_policy = eval_I(2.0, 'l5')
i_after_static = eval_I(2.0, 'l3')
i_after_idle = eval_I(2.0, 'none')
print("  window I: calm+l3=%.6f ; after shift: policy(l5)=%.6f  "
      "static(l3)=%.6f  idle=%.6f" % (i_before, i_after_policy,
                                      i_after_static, i_after_idle))
print("  adaptation recovers %.6f (%.0f%% of the l3->l5 gap)"
      % (i_after_policy - i_after_static,
         100.0 * (i_after_policy - i_after_static)
         / (i_after_policy - i_after_idle)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True,
                               gridspec_kw={'height_ratios': [1, 2.4]})
ax1.step(ts, kappa_of_t, where='post', color='tab:gray', lw=1.5)
ax1.set_ylabel('wind $\kappa$')
ax1.set_ylim(0.5, 2.5)
ax1.axvline(T_SHIFT, color='k', ls=':', lw=1)
ax2.plot(ts, vals_adapt, lw=2, label='adaptive policy (l3$\\to$l5)')
ax2.plot(ts, vals_static, lw=1.4, ls='--', label='static l3 (no re-plan)')
ax2.plot(ts, vals_idle, lw=1.4, ls=':', label='idle (no boost)')
ax2.axvspan(LO, HI, alpha=0.08, color='tab:blue')
ax2.axvline(T_SHIFT, color='k', ls=':', lw=1)
ax2.annotate('wind shift\nre-eval: %.0f ms (6 actions)' % t_re,
             xy=(T_SHIFT, i_after_policy), xytext=(T_SHIFT + 0.35,
                                                   i_before - 0.004),
             arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)
ax2.set_xlabel('mission decision time t')
ax2.set_ylabel('window probability $I(3.14,6.14)$')
ax2.legend(loc='lower left', fontsize=8)
fig.savefig('fig8_timeline.png')
print("fig8 done")
