# DSRL Harness & Steering Report — square-mh

**Date:** 2026-08-07. Branch `copredict`. Task: `square_mh_state` (A-arm) and
`square_mh_state_flow_intent` (I-arm), the debugging sandbox per
`docs/intent_dsrl_plan_v3.md` §5 step 7 — **not** a scientific result for the paper.

**Bottom line:** after fixing three harness bugs, DSRL steering shows **no benefit**
on square-mh in any tested configuration. The A-arm's catastrophic failure is fully
explained: the SAC actor drifts to ~21× the prior norm, and the frozen flow policy
collapses to 0.000 SR at even 3× prior norm. The remaining question is whether a
correctly-constrained actor can help at all — not whether the plumbing works. It does.

---

## 1. Headline results (NFE=9, corrected harness)

Mean over 100 evals/run, 20 episodes each. `gain = steered − prior`, where `prior`
is z ~ N(0,I) rolled out through the **identical** harness at the same eval step.

| Run | actor | z-dim | steered | prior | mean gain (95% CI) | verdict |
|---|---|---|---|---|---|---|
| n9tanh_a | TanhGaussian | 100 | 0.029 | 0.585 | **−0.555** [−0.581, −0.530] | significantly worse |
| n9ms_a | mean-shift | 100 | 0.030 | 0.585 | **−0.554** [−0.586, −0.522] | significantly worse |
| n9tanh_i | TanhGaussian | 7 | 0.479 | 0.466 | +0.013 [−0.016, +0.043] | **no effect** |
| n9ms_i | mean-shift | 7 | 0.414 | 0.476 | −0.062 [−0.096, −0.028] | slightly worse |

CIs treat evals as independent; they are autocorrelated across training, so true
intervals are wider. This does not affect the −0.55 conclusions.

---

## 2. Bugs found and fixed

### 2.1 Non-transactional optimizer updates → permanent target-critic corruption

`mip/dsrl/dsrl_agent.py` scrubbed critic/actor **gradients** with `nan_to_num_`, but:

- the entropy-temperature (`log_alpha`) gradient had no guard, so one non-finite
  `log_prob` permanently poisoned `alpha`; and
- a non-finite loss could still corrupt parameters via Adam's running moments, which
  then propagated into the Polyak-averaged target critic — and
  `lerp_` with a NaN source is NaN forever.

Symptom: runs trained normally for ~64k steps, then every loss went NaN permanently.

**Fix:** `_safe_optimizer_step()` — check loss finiteness *before* `backward()`, check
every gradient *before* `step()`, clip, then assert parameters are finite (raise loudly
rather than continue silently). Critic/actor/alpha update independently, and
`soft_update_targets()` runs **only** after a successful critic step.

### 2.2 `num_steps=1` — DSRL steered a crippled policy (the big one)

`examples/configs/optimization/default.yaml` sets `num_steps: 1`, and `train_dsrl.py`
never overrode it, so the frozen policy's ODE ran with a **single Euler step**. The
checkpoint's own recorded eval history:

```
mean_success_9 = 0.64   ← the "success64" in the filename
mean_success_3 = 0.54
mean_success_1 = 0.44   ← what DSRL actually used
```

Measured directly on the flow-intent checkpoint through the DSRL eval loop:

| NFE | native `sample()` | injected z~N(0,I) |
|---|---|---|
| 1 | 0.033 | 0.083 |
| 3 | 0.467 | 0.350 |
| 9 | 0.500 | 0.467 |

Every DSRL run prior to this fix was steering a policy worth ~0.03–0.08, not ~0.50.
All headroom claims computed against "the 0.62/0.64 base policy" were invalid.

**Fix:** `dsrl.flow_num_steps` (default **9**) threaded through `_run_flow` into both
the training rollout and eval. Set this to the NFE the checkpoint was evaluated at.

### 2.3 No like-for-like baseline

DSRL SR was being compared against a base SR measured by a *different* code path at a
*different* NFE. That is how 2.2 stayed invisible.

**Fix:** every eval now also rolls out z ~ N(0,I) through the identical harness and logs
`eval/prior_mean_success` and `eval/steering_gain`. A harness regression can no longer
masquerade as a steering result.

---

## 3. Why the A-arm fails (resolved)

Probe on the frozen A-arm checkpoint + trained `n9ms_a` actor
(`scratchpad/probe_aarm.py`):

```
1. act_0 sensitivity
   stochastic, z1 vs z2 : mean|diff| = 0.057   → injection WORKS
   stochastic, z1 vs z1 : mean|diff| = 0.000   → deterministic given z
   zero-mode,  z1 vs z2 : mean|diff| = 0.000   → act_0 ignored in zero mode

2. actor ||z|| vs prior
   E||z_prior|| = sqrt(100) = 10.00
   ||z_actor||  = 212.36   max|z| = 58   KL/dim = 225.0

3. rollout SR
   zero   (BC native, sample_mode="zero")   0.567
   prior  (N(0,I))                          0.617
   scaled (3 x N(0,I))                      0.000   <-- control
   actor  (||z|| ~ 212)                     0.000
```

**Causal chain:** injection works → `kl_beta=0.1` is far too weak to constrain a
100-dim latent → μ drifts to 21× the prior norm (KL/dim 225 vs a penalty weight of 0.1)
→ the frozen policy emits garbage → SR 0.000.

The `scaled` control is the important one: **3× the prior norm alone collapses the
policy to exactly 0.000.** The actor never needed to find adversarial latents — any
norm drift is fatal. This is why changing the actor parameterization (tanh →
mean-shift) did not help: both drift, and drift is what kills it.

The I-arm survives because 7 dimensions keep μ small (mean-head weight norm 2.09 vs
23.86 for the A-arm).

### Note: `sample_mode` semantics

`mip/samplers.py:48` — with `sample_mode="zero"` (the config default, and what BC
training/eval used) the ODE starts at zeros and `act_0` is **ignored entirely**; the
policy is deterministic. `train_dsrl.py` passes `sample_mode="stochastic"`, which is
required for steering to have any effect. The two modes score 0.567 (zero) and 0.617
(prior) — comparable, so this is a documentation hazard rather than a bug, but worth
knowing: the A-arm base policy was trained as a zero-start flow policy.

---

## 4. Methodological finding: "best success" is a max-of-N artifact

`train_dsrl.py` saves `best.pt` at the maximum eval SR, and runs reported
"Best success: 0.80". Compare each run's steered max against the **prior baseline's**
max in the same run:

| run | best steered | prior max | prior mean ± sd |
|---|---|---|---|
| n9ms_a | 0.80 | **0.80** | 0.585 ± 0.106 |
| n9ms_i | 0.75 | **0.75** | 0.476 ± 0.107 |
| n9tanh_i | 0.80 | 0.70 | 0.466 ± 0.104 |

The unsteered baseline achieves the same max. With 20-episode evals (sd ≈ 0.11), the
max of ~100 draws sits ~2 sd above the mean whether or not anything was learned —
`n9ms_a`'s 0.80 occurred at **step 20,000**, during warmup, where the concurrent prior
scored 0.70.

**Max-over-evals cannot distinguish steering from no steering.** Report mean gain with
CIs (or a final-window mean), per plan v3 §3, never best-checkpoint SR. An earlier
ablation table in this workstream reported 0.75/0.64/0.50/0.35 as headline numbers;
those are max-of-N statistics on runs whose typical behavior was far worse.

---

## 5. Hypotheses raised and falsified

Recorded because the falsifications were informative:

| Hypothesis | Verdict |
|---|---|
| I-arm `sample_with_intent_noise` diverges from native sampling | **False** — agree within noise at NFE 3 and 9 |
| Slice-then-unnormalize vs unnormalize-then-slice differs | **False** — normalizer params are per-action-dim `(10,)`, so equivalent |
| tanh ±2 bound is why the A-arm fails (dimensional argument) | **False** — unbounded mean-shift actor fails identically |
| Actor parameterization is what stands between DSRL and a positive result | **False** — neither parameterization gives a positive result |
| A-arm injection path is broken (`act_0` overridden) | **False** — `act_0` demonstrably controls the output |

The surviving explanation is off-prior drift under insufficient KL regularization
(§3), which is quantified rather than inferred.

---

## 6. Recommended next steps

1. **Adaptive KL constraint (C4).** Learn β to hold KL/dim at a target (0.01–0.1)
   instead of a fixed `kl_beta`. A fixed weight cannot work across z-dims of 7 and 100 —
   the observed KL/dim of 225 against a weight of 0.1 shows the Q-term simply dominates.
2. **Measure the tolerance curve first.** Sweep `‖z‖` scale ∈ {1, 1.5, 2, 3} × prior and
   record SR. §3 shows collapse somewhere below 3×; the usable steering budget is
   whatever lies inside that. If it is very tight, latent steering may be
   fundamentally limited for this base policy, which is a result worth knowing before
   spending more compute.
3. **Paired evaluation with CRN.** `prior` and `steered` are currently separate
   20-episode batches, so the gain carries ~±0.15 noise. Sharing initial states would
   cut that substantially (plan v3 §3 already requires this).
4. **Do not port these numbers to RoboCasa.** square-mh is the sandbox; what
   transfers is the fixed code and the measurement discipline, not the SRs.

---

## 7. Artifacts

- Runs: `dsrl_runs/nfe9/{n9tanh_a,n9tanh_i,n9ms_a,n9ms_i}/` (W&B `square-mh-dsrl-nfe9`)
- Superseded pre-fix runs: `dsrl_runs/ablation/` (NFE=1 — **do not use**)
- Logs: `dsrl_runs/logs/*.log`
- Probes: `scratchpad/probe_iarm.py`, `probe_nfe.py`, `probe_aarm.py`
- Code changed: `mip/dsrl/dsrl_agent.py`, `mip/residual_sac/networks.py`,
  `examples/train/train_dsrl.py`
