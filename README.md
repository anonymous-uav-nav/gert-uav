# gert-uav

Reference implementation for: **Generalized GERT Network
Modeling with Sensitivity-Guided Adaptive Planning for UAV Mission
Reliability**.

A GERT (Graphical Evaluation and Review Technique) model of UAV
missions whose flight segments may be retried, aborted, or restarted.
The equivalent W-function W_E(s) — the first-passage transform of
mission completion time — is obtained by solving a linear
signal-flow system (provably equal to the full Mason gain formula),
and the window probability I(t1, t2) = P(t1 <= T <= t2) is evaluated
analytically in ~60 ms, replacing Monte Carlo inner loops inside
learning and optimization.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (numpy, sympy, matplotlib)

All experiments are single-threaded and run on a laptop CPU.

## Repository layout

| file | what it does |
|---|---|
| `_exp1lib.py` | core library: W_E construction (linear system + Mason), residue analysis, window probability, Monte Carlo ground truth, well-posedness check |
| `exp1_generalized_solver.py` | E1-E4: solver validation (independent symbolic Mason cross-check), analytic-vs-MC cross-validation, E3 sensitivity ranking, E4 routes x wind grid |
| `exp2_qlearning.py` | E5: wind-adaptive branch-rate correction via tabular Q-learning (GERT-in-the-loop) |
| `exp3_case.py` | E6: 6-node inspection topology (topology-agnostic solver) |
| `exp7_robustness.py` | E7: E1-E3 conclusions across three parameter groups (baseline / strong wind / high loss) |
| `exp8_convergence.py` | E8: Q-learning convergence over the alpha x epsilon grid (9 agents) |
| `exp9_scalability.py` | E9: solver scaling on random networks, n up to 200 (symbolic vs phase-type pipelines) |
| `exp10_online_demo.py` | E10: online re-planning timeline (wind shift; re-scores all 6 actions analytically, 52 ms warm / 142 ms cold per action) |
| `bench_eval.py` | per-evaluation cost decomposition (warm vs cold symbolic construction) |
| `exp11_backend_swap.py` | E11: same Q-learning, GERT-in-the-loop vs MC-in-the-loop backend swap |
| `exp12_rrt.py` | E12: simplified RRT* under the same wind and window objective |
| `exp13_erlang2.py` | E13: distribution robustness — Erlang-2 segments via the phase-type pipeline (expm-based window integral, immune to pole coalescence) |
| `exp14_sens_relative.py` | E14: sensitivity under +10% relative vs +0.1 absolute perturbation, window re-centering, and width sweep |
| `exp15_closedform.py` | E15: closed-form window integral vs grid quadrature cross-check (max diff 1.4e-12) |
| `exp_rate_convention.py` | numerical counterexample for the rate-by-branch vs rate-by-head-node convention (Remark 2) |
| `fig1_topology.py` | reference topology figure (600 dpi) |
| `figs.py` | remaining paper figures |

## Reproduction

Run any script from this directory:

```bash
python exp1_generalized_solver.py    # E1-E4 numbers
python exp2_qlearning.py             # E5 R-table + learned policy
python exp3_case.py                  # E6 case study
python exp7_robustness.py            # E7 robustness table
python exp8_convergence.py           # E8 convergence grid -> fig6
python exp9_scalability.py           # E9 scaling table -> fig7
python exp10_online_demo.py         # E10 timeline -> fig8
python bench_eval.py                # warm/cold per-evaluation cost
python exp11_backend_swap.py        # E11 backend comparison (~30 min)
python exp12_rrt.py                  # E12 RRT* comparison
python exp13_erlang2.py              # E13 Erlang-2 robustness
python exp14_sens_relative.py        # E14 normalization + window semantics
python exp15_closedform.py           # E15 closed-form cross-check
python exp_rate_convention.py        # Remark 2 counterexample
```

Pre-generated figures are in `figures/` (fig1-fig8). Scripts write
figures to the working directory when re-run.

## Paper table/figure map

| paper item | produced by | output |
|---|---|---|
| Table: E3 sensitivity ranking | `exp1_generalized_solver.py` | stdout |
| Table: E4 routes x wind | `exp1_generalized_solver.py` | stdout |
| Table: E5 R-table | `exp2_qlearning.py` | stdout |
| Table: E6 case study | `exp3_case.py` | stdout |
| Table: E7 robustness (mean +/- std) | `exp7_robustness.py` | stdout |
| E8 convergence curves (Fig. 6) | `exp8_convergence.py` | `fig6_convergence.png` |
| E9 scaling (Fig. 7) | `exp9_scalability.py` | `fig7_scaling.png` |
| E10 online timeline (Fig. 8) | `exp10_online_demo.py` | `fig8_timeline.png` |
| E11 backend swap | `exp11_backend_swap.py` | stdout |
| E12 RRT* comparison | `exp12_rrt.py` | stdout |
| E13 Erlang-2 robustness | `exp13_erlang2.py` | stdout |
| E14 normalization + window semantics | `exp14_sens_relative.py` | stdout |
| E15 closed-form cross-check | `exp15_closedform.py` | stdout |
| Remark 2 counterexample | `exp_rate_convention.py` | stdout |
| Fig. 1 topologies | `fig1_topology.py` | `fig1_topology.png` |
| Figs. 2-5 | `figs.py` | `fig2`-`fig5` |

## Model summary

Nodes are mission waypoints; branch (i, j) carries a probability
p_ij and an exponential segment time with rate lambda_ij
(*rates belong to branches, not head nodes*). Well-posedness:
sum_j p_ij <= 1 at every node, residual mass = explicit mission
abort. The first-passage transform solves the linear system
g_i = sum_j W_ij g_j (g_sink = 1), and I(t1, t2) follows from the
residue expansion of W_E. See the paper for proofs and details.

## License

MIT — see [LICENSE](LICENSE).
