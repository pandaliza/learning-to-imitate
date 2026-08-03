# Intent as a Steering Interface — Execution Plan v2.2

**Date:** 2026-08-03. Supersedes the sequencing in `intent_dsrl_robocasa_plan.md` (v1, kept as the
reference for motivation/formulation). Branch: `copredict`.

**v2.1 → v2.2 (second review round — targeted edits only, scope otherwise frozen):** W3 rebased to
a moderate-difficulty adaptation setting with all bases in ~30–70% SR (primary mechanism:
square-mh-state; fallback: reduced-demo square-ph; shifted-reset eval as secondary analysis);
headroom-closed metric added; C0/C1/C3 launch first, C2 trains in parallel and gates only the
semantic-vs-future-grounded conclusion; paired-bootstrap statistics over shared probe states,
per-task before pooling; J-mask `channel_probe` vs T-mask `causal` labeling made a hard code/docs
convention; W3 motivation softened from "causality demonstrated" to "behavioral influence
demonstrated". Venue confirmed: **ICLR 2027**. Training launches (W0.2/W0.3/W0.4) are **on user
hold as of 2026-08-03** — the execution order below is frozen, but those steps await explicit
release.

**v2 → v2.1 (after external review):** W3 control redesigned as a structured-vs-unstructured
latent ladder with per-control latent-usage checks; POSTBC moved from the G1-fail to the G2-fail
branch; gate thresholds relabeled provisional with CI/trial-count requirements; G1c gains
task-level dispersion measures; Claim 5 split into 5a (base transfer) / 5b (steering transfer);
A1-checkpoint results reframed as "useful intent channel," not "causal upstream plan"; T-mask
retrain promoted to launch-now-if-compute-allows; timeline compressed for a September deadline
with writing starting immediately.

---

## 0. Thesis and claims

> **Can co-predicted future intent provide a structured, low-dimensional steering interface for
> adapting generative robot policies?**

| # | Claim | Where tested |
|---|---|---|
| 1 | Decoupled co-prediction separates plan generation from execution | RoboCasa Phase A + diagnostics (strict version needs T-mask arm) |
| 2 | Intent-space DSRL is more sample-efficient than action-noise DSRL — **and than a dim-matched unstructured latent** | Robomimic de-risk (control ladder) → RoboCasa Phase C |
| 3 | Intent steering preferentially fixes reach/wrong-subgoal failures | RoboCasa Phase C (needs validated taxonomy) |
| 4 | The intent-steering advantage grows with compositionality/horizon | RoboCasa atomic vs composite |
| 5a | Co-prediction preserves (or improves) base-policy transfer | RoboCasa unseen cells in Phase A — early kill-risk check (LIBERO precedent: A1 zero-shot 0% vs A0 25%) |
| 5b | A learned intent-steering policy transfers better than an action-noise controller | Phase C: steering trained on seen layouts, evaluated on held-out layouts/tasks vs A-DSRL |

**Claim ladder (what survives which outcome):** if semantic EEF intent ≈ learned future-code
latent but both beat dim-matched noise, the claim degrades gracefully to "temporally-extended,
future-grounded latents beat raw noise" — weaker but publishable. The controls in W3 are designed
to make this distinction measurable rather than all-or-nothing.

**Honest prior from the evidence to date:** every positive result in the program is
*diversity/support-shaped* (VDR, clustering, branching, ghost bundles), and every *success-shaped*
comparison at scale has been neutral-to-negative (M10: s1 ≤ s2 always, decoupled ≤ tied, A0 most
robust, A1 spatial zero-shot 0% vs A0 25%; earlier: flow-intent < baseline on can-mh-image,
tool-hang, multi-task LIBERO). DSRL-intent tests whether diversity-shaped support converts into
success — structured so a negative answer is cheap and a positive answer is credible.

---

## 1. Current state (verified 2026-08-03)

**RoboCasa M11 trainings** (`m11_checkpoints/` on group volume):
- A1 tied: **complete**, 30k steps (ckpts 15k/20k/25k/30k), finished 2026-08-02.
- A0 pure BC: **dead** — job 9644812 died at ~step 300 (node flake, no traceback). Dir empty.
- A2/A3 decoupled: **cancelled** (9644814/9646096). Dir empty.
- No RoboCasa evals have run. Phase A currently has one arm and zero numbers.

**Codebases:**
- `external/dsrl` — clean never-run PyTorch reference (SAC / DSRL-NA over noise space via SB3 fork;
  DPPO-shaped base-policy interface `cond={"state","noise_action"}`; submodules empty).
- `external/OGPO` — JAX flow-matching RL framework; ships DSRL, EXPO, OGPO-{PPO,FPO,AWR} agents on
  Robomimic/PushT/Adroit; trajectory-level advantages; pi0.5 *encoder* only.
- `external/postbc` — minimal PostBC (ensemble variance + target perturbation) on MIP Robomimic.
- M11 stack (dataset/trainer/eval with s1/s2/s3) — built, smoke-tested, launch-validated.
- Early "Intent RL" (lift-mh, 99%, from Sreyas' codebase) was IA-DSRL-shaped (actor supplied noise
  **and** intent jointly), on a saturated task, without I-only or dim-matched controls — the new
  plan's controls are precisely what it lacked. **Open: locate that code.**

**Causal status of existing checkpoints:** all trained arms (LIBERO + RoboCasa) use the joint mask
J — intent tokens attend to action tokens, so z_A leaks into intent even under intent-first
inference. Interventions on these checkpoints therefore answer:

> *Does the existing joint model expose a behaviorally useful intent channel?*

— **not** "is intent a causally upstream plan." The strict factorization p(I|o)·p(A|o,I) requires
the two-stream mask T (M10 ablation C1, never trained). A1 is the fast diagnostic; T-mask is the
clean method arm. Interventions on J-mask models must hold z_A fixed via common random numbers.

**Noise-sample vs noise-level distinction (works in our favor):** tied vs decoupled refers to noise
*levels* τ, not noise *samples*. z_I and z_A are independent draws in every arm. The full
diagnostic suite and even I-DSRL are runnable on the **already-trained A1 checkpoint** under s2 —
no waiting on retrains to start Gate 1/2 measurements.

---

## 2. Decision gates

Thresholds below are **provisional operational thresholds** — internal decision rules, not
paper-grade pre-registration. Report every gate metric with a 95% CI; no go/no-go on a marginal
pass. **Statistics floor:** at 20 trials the unpaired binomial CI half-width is ±22pp at p≈0.5 —
unresolvable for 10–15pp effects. Gate-critical cells use **≥50 trials**; Best-of-N uses **≥50
probe states**. **Paired design (preferred):** interventions and BoN start from identical sim
states with common random numbers, so report **paired bootstrap CIs over probe states** for SR
differences, subgoal-completion differences, destination/target changes, z_I-vs-z_A dispersion,
and BoN gains — paired estimates resolve smaller effects at the same budget than the unpaired
floor above. **Report each task separately before pooling** (an atomic/composite sign flip *motivates and
supports* the compositionality hypothesis (Claim 4) — reported as a finding, but not treated as
established from a single task pair; it must repeat across tasks or difficulty-matched pairs).

Measured on ≥2 RoboCasa tasks with base SR in [15%, 85%] (task choice from Phase A).

| Gate | Test | Provisional threshold | On fail |
|---|---|---|---|
| **G1a** Intent is used (offline) | GT-intent injection vs model-intent on held-out demo states | action MSE ↓ ≥ 20% with GT intent | intent pathway inert → **architecture/representation fix** (T-mask, intent dropout, w_I sweep, self-conditioning) or stop. *Not* POSTBC — support expansion cannot fix a decoder that ignores intent. |
| **G1b** Intent is used (online) | GT/shuffled intent rollouts | GT ≥ default; shuffled ≤ default − 10pp, or visible redirection toward shuffled goal | same as G1a |
| **G1c** Intent > action noise for high-level change | fix z_A vary z_I vs fix z_I vary z_A (common random numbers, N=16 each) | z_I-induced : z_A-induced dispersion ratio ≥ 2 on **at least one task-level measure**: approached object/fixture identity, subgoal order, destination region, success-conditioned cluster separation. Endpoint dispersion alone is insufficient (a noisy intent field also produces it). | same as G1a |
| **G2** Useful support exists | Best-of-N over z_I from identical sim states (N=8, ≥50 probe states pooled) | BoN(8) ≥ default SR + 15pp | support missing → **POSTBC-intent** (coherent (Ĩ, Ã) pairs, v1 §2.7) |
| **G3** Intent is easier to steer | control ladder C0–C3 (§W3) | C3 beats C0 **and** C1 on ≥1 of: steps-to-target-SR, AUC, no_reach reduction, retention. C3-vs-C2 decides the semantic-vs-future-grounded strength of the claim. | steering claim dies; salvage = analysis paper on when co-predicted intent becomes causal |
| **G4** Full optimization needed | only after G3 | I-DSRL plateaus below oracle BoN | hierarchical OGPO |

Gate reviews are human decisions (user + me), not automated.

---

## 3. Workstreams

### W0 — Ops (immediate, ~30 min of work + 48 h wall-clock)

| ID | Task | Detail |
|---|---|---|
| W0.1 | Archive A1 ckpts | `cp -r` 15k/20k/25k/30k → `m11_a1_tied_archive/`. Trainer prunes to last 3; M9's best weights were lost this way. |
| W0.2 | Relaunch A0 | Same sbatch, `ARM=a0`. Consider `--exclude` on the flaky node. **Non-negotiable: every downstream arm compares against it.** |
| W0.3 | Launch T-mask decoupled arm | One-line `att_masks` change (M10 spec §2.4 variant T) + decoupled noise; new config name **with its own assets dir** (norm-stats rule). The only arm that can carry the strict hierarchical claim. |
| W0.4 | Relaunch A2/A3 (J-mask decoupled) | Third priority. If the compute envelope allows only two concurrent trainings, defer this one: M10 already showed J-decoupled ≤ tied, and A1 covers joint-model diagnostics. Cost of deferring: lose RoboCasa comparability with M10's A2/A3 cell. |
| W0.5 | Post-mortem 9644812 | 5-min sacct/node check; only to decide exclusion list. |

### W1 — Phase A evidence (starts now on A1; extends as ckpts land)

**W1.1 — RoboCasa eval grid** (sub-agent: eval-runner)
- A1 @ 30k (and 20k for a curve point), schedules s2 (native), s1, s3.
- Task cells: 4 atomic_seen + 5 composite_seen + **2 unseen from day one** (ArrangeTea, PanTransfer)
  — these measure **Claim 5a** (base-policy transfer), an early kill-risk check given the LIBERO
  precedent. They do not test steering transfer (5b, Phase C).
- 20 trials/cell for the survey grid; **gate-critical cells re-run at ≥50 trials** after task
  selection. Log per-task SR, intent ADE/FDE vs realized EEF, episode lengths.
- **Throughput measurement:** rollout steps/sec and wall-clock per episode → transitions/day/GPU.
  This decides Phase C's compute plan.
- Deliverable: `docs/robocasa_m11/phaseA_results.md` + `logs/eval_m11_*.json`.
- When A0/T-mask/A2a3 finish: same grid.

**W1.2 — Task selection** (output of W1.1): 2 tasks with base SR in [15%, 85%], one atomic one
composite, for all diagnostics and Phase C. Do not pre-commit to KettleBoiling.

### W2 — Causal-intent diagnostic suite (sub-agent: diagnostics; build now, run on A1 immediately)

The gates' measurement instrument. All additive, behind flags; existing eval paths untouched
(repo convention). Results on A1/J-mask arms are framed per §1: useful-channel, not causal-plan.

**Hard labeling convention (code, logs, tables):** every diagnostic output carries a
`mask_regime` tag — `channel_probe` for J-mask arms (A1, A2a3), `causal` for the T-mask arm.
Result tables inherit the tag; no script or table may label `channel_probe` results as causal
intent. On J-mask arms, `sample_intent` records the fixed action-token noise/state present during
intent generation.

**W2.1 — Base-policy API split** (v1 Milestone 1) in `pi0_pytorch.py` behind the copred flag:
```python
intent, cache = model.sample_intent(obs, noise_I=...)          # reuses prefix KV
actions = model.sample_action(obs, intent=..., noise_A=..., cache=...)
```
plus injectable/seedable `noise_I`, `noise_A` in `sample_actions_copred` for all schedules.
(On J-mask checkpoints `sample_intent` necessarily runs with some action-token state present —
document which fixed z_A it uses; on the T-mask arm it is exact.)

**W2.2 — Offline GT-injection probe (G1a)** — `examples/openpi/diag_intent_causal.py`:
held-out demo states → action-prediction MSE under {GT intent, model intent, shuffled intent,
zero intent}. Cheap (no sim); first gate number within a day of code completion.

**W2.3 — Rollout interventions (G1b/G1c)** — flags on `eval_robocasa_intent.py`:
`--inject-gt-intent`, `--shuffle-intent {episode,task}`, `--fix-za SEED`, `--fix-zi SEED`,
`--n-variations N`. Common random numbers across variation sets. Log full EEF + base trajectories
**plus task-level outcomes**: nearest-approached object/fixture per rollout segment, subgoal-order
trace, destination region — the G1c measures.

**W2.4 — Best-of-N oracle (G2)**: sim-state save/restore (`env.sim.get_state()/set_state()` —
**verify in RoboCasa envs first**; the one technical unknown), N=8 intent draws per probe state,
≥50 probe states pooled across the 2 selected tasks, success + outcome dispersion per state.

**W2.5 — Failure-taxonomy validation (pre-req for Claim 3)**: RoboCasa progress detectors
(reach/grasp/subgoal per v1 §3.6) validated against ~20 hand-checked rollout videos per task.
Motivated by the deck's "(WRONG LOGIC NEED TO CHECK)" flag and the qpos-indexing incident:
"all failures are no_reach" is exactly what a broken reach detector produces.

**W2.6 — Base-motion predictability (v1 §3.5 caveat)**: dataset-side regression, future base
displacement from EEF intent. Decides whether the intent representation needs
`[Δbase; EEF_rel]` terms before Phase C. Small analysis script, no training.

- Deliverable: `docs/robocasa_m11/gate_report.md` with G1a/G1b/G1c/G2 numbers (+ CIs) on A1,
  then the T-mask arm.

### W3 — Small-scale de-risk on Robomimic square (sub-agent: rl-derisk; parallel to W1/W2)

The cheapest decisive test of Claim 2, on a base-policy family where intent has already shown
strong **behavioral influence, structured diversity, and rollout-level variation** (flow-intent
square-ph-state: 83% SR vs baseline 71%, VDR 0.71 vs 0.34) — not yet strict causal factorization;
that is what the controls here establish. Rollouts are ~1000× cheaper than RoboCasa.

**Difficulty/headroom design.** The principal comparison must run with **all base policies in
~30–70% SR** — at 83% the headroom is compressed and unequal initializations (83 vs 71) confound
the curves. Mechanism, in order of preference:
1. **Primary: `square-mh-state`** — standard Robomimic multi-human dataset, lands BC in the
   40–60% range naturally (no artificial subsetting to defend); one cheap state-based BC retrain
   per base arm (hours each).
2. **Fallback: reduced-demonstration `square-ph`** (e.g. 20–30% of demos) if mh bases fall
   outside the window.
3. **Secondary analysis: shifted-reset evaluation** on the full-data policies — doubles as the
   Claim-3 recovery probe.
Checkpoint-selection SR-matching: supplementary table only (early checkpoints differ
qualitatively, not just in SR).

**W3 step 0 — square-mh validation gate.** The strong intent evidence (83% SR, VDR 0.71) is from
square-**ph**; square-mh has no flow-intent numbers yet. The mh base trainings (required anyway)
double as the check — before any RL, verify: (a) plain-flow and flow-intent both land in ~30–70%
SR; (b) flow-intent retains structured diversity on mh (VDR/ghost check); (c) intent
interventions change behavior (mini-G1 on the new base); (d) record the base-SR gap for the
SR-matched analysis. Any failure → fallback mechanism 2. Note: mh's richer demonstrations are
support shared by every arm through the dataset — whether the intent latent makes that support
*selectable* is the claim under test; the confound to guard is only renewed base-SR divergence,
via (d).

**Metrics per arm:** ΔSR, AUC of the online curve, steps-to-target-SR, and **headroom closed**
`(SR_final − SR_base)/(1 − SR_base)`. Do not rely solely on improvement over own initialization —
include the difficulty-matched (and where feasible SR-matched) analysis.

**Control ladder** (chunk-level MDP, SAC, identical budgets, 3 seeds). Design rule: **every
control arm must pass its own G1-style latent-usage check before its steering result counts** —
otherwise the comparison is a strawman (an unsupervised latent that collapsed loses for the wrong
reason).

**Sequencing:** launch **C0, C1, C3 immediately**; train C2 in parallel. C2 gates only the final
semantic-vs-future-grounded conclusion — it must never delay the first I-DSRL-vs-DSRL curves.

| Rung | Base (frozen) | RL action space | Dim | Isolates |
|---|---|---|---|---|
| — | flow-intent BC / plain flow BC, no RL | — | — | initialization baselines |
| **C0** | plain flow BC (71%) | raw action noise (standard/A-DSRL) | H×d_A | practice-relevant baseline |
| **C1** | plain flow BC | dim-matched latent, fixed projection → action-noise space | d_I | dimensionality alone (note: projection geometry is itself a caveat — report, don't over-claim) |
| **C2** | two-stage BC, latent = **learned autoencoder code of the future trajectory** (same dim, same pathway, reconstruction-supervised — cannot collapse) | z_C2 | d_I | future-grounding without semantics |
| **C3** | flow-intent BC (83%) | intent noise z_I | d_I | semantic/geometric structure on top of C2 |
| IA | flow-intent BC | z_I (decoder deterministic → no z_A in Config A) | d_I | — |

Notes: (i) a *linear* reparametrization of future EEF is a vacuous control (MLP absorbs it) — C2
must be a learned nonlinear code; (ii) C2 requires one extra BC training (same MIP two-stage
architecture, AE-code targets instead of EEF); (iii) Config-A's deterministic MLP decoder makes C3
maximally clean here — all stochasticity flows through intent.

**C2-vs-C3 required reporting table** (match or report explicitly): latent dimension; latent
prior; encoder/decoder capacity; injection pathway; BC success; latent norm and variance; action
sensitivity to latent perturbations; latent-usage test result.

Report: SR vs env steps, AUC, headroom closed, improvement over own base, retention on
unperturbed resets, distance-from-prior of learned z, per-arm latent-usage check results — per
the difficulty-matched design above.

**Codebase decision:** default **(a)** `external/dsrl` + SB3-dsrl fork + a `MIPBasePolicyWrapper`
implementing the `cond={"state","noise_action"}` interface — PyTorch throughout, reuses existing
MIP checkpoints (needs submodule init; our own base ckpts, no Google Drive). **(c)** Sreyas'
codebase as accelerant if located. (b) OGPO/JAX rejected for now — retraining the base in JAX
solely for framework continuity is unjustified.

**RoboCasa forward-compatibility:** if C3 > C2 matters in W3, the RoboCasa version of C2 (co-pred
training with scrambled/AE intent targets — same trainer, different targets) goes on the training
queue in Phase C. Booked now so it isn't discovered in September.

### W4 — RoboCasa I-DSRL (Phase C; **gated on G1+G2 pass and W3 signal**)

- Native chunk-level SAC-over-z_I trainer built on the `eval_robocasa_intent.py` env loop (obs
  plumbing/normalization already solved) — not a port of pi0.5 into the dppo-shaped repo.
- Prefix-KV reuse per chunk; replay buffer of (obs-embedding, z, chunk-return); n-step chunk
  returns; DSRL-NA dual-critic structure from `external/dsrl` as algorithmic reference.
- Tasks: the 2 selected in W1.2. Arms per v1 §2.5 (A0 base for C0/C1; co-pred base for A/I/IA;
  preferred co-pred base = T-mask arm if its Phase A numbers are sane, else A1 with the
  useful-channel framing). Claim 5b cells: steering trained on seen layouts → held-out layouts.
- Budget (episodes/seed, seeds, GPUs) set by the W1.1 throughput number — not before. Headline
  comparisons get ≥2 seeds or explicit trial-count error bars.

### W5 — Conditional extensions (Phase D)

- **POSTBC-intent** (G2 fail only): coherent (Ĩ, Ã) posterior pairs per v1 §2.7 —
  `external/postbc` ensemble machinery extends naturally; trajectory-level bootstrap (episode
  boundaries available in both datasets).
- **Architecture/representation fixes** (G1 fail): T-mask (already training per W0.3), intent
  dropout, w_I sweep, self-conditioning, `[Δbase; EEF_rel]` representation per W2.6.
- **Hierarchical OGPO** (G3 pass + G4 only): plan-level vs execution-level advantages. Months-scale
  extension (OGPO advantages are trajectory-level today). Firmly conditional.

---

## 4. Sub-agent assignments

| Agent | Workstream | First deliverable | Depends on |
|---|---|---|---|
| ops | W0 | jobs queued + archive done | user go-ahead + compute envelope |
| eval-runner | W1 | Phase A table on A1 + throughput | W0.1 (archive first) |
| diagnostics | W2 | API split + offline G1a number | none (A1 ckpt exists) |
| rl-derisk | W3 | standard-DSRL repro (C0) running on square-ph | codebase decision |
| writer | paper skeleton | outline + figure list + related-work stubs | none — **starts now** |
| (later) rl-robocasa | W4 | — | gates |

Parallelism: ops/eval-runner/diagnostics/rl-derisk/writer are independent and launch together.
Integration rule (repo convention): additive modules, forked entry points, one sbatch per arm,
env-var knobs; never modify a live training's script.

## 5. Execution order and timeline (venue **confirmed: ICLR 2027**, late-Sept)

**Frozen execution order** (per review round 2 — no further re-planning unless a gate fires):
1. Archive A1. *(approved)*
2. Relaunch A0. **(training — on user hold)**
3. Launch T-mask decoupled arm. **(training — on user hold)**
4. Run offline G1a on A1.
5. Measure RoboCasa throughput + Phase-A SR grid on A1.
6. Launch W3 C0 + C3 under the moderate-difficulty setting.
7. Train C2 in parallel.
8. Paper skeleton + figure list immediately.

| Date | Milestones |
|---|---|
| Aug 3–5 | W0 launched; W2.1 API split; offline G1a number on A1; paper skeleton exists |
| Aug 6–10 | A1 online G1/G2 (gate-critical cells at ≥50 trials); Phase A survey grid on A1; C0 repro running |
| Aug 11–15 | A0/T-mask ckpts land → their grids + G1 on T-mask; first W3 curves (C0/C1/C3); C2 BC training |
| **Aug 17–20** | **Hard paper go/no-go**: which claims are alive, which testbed carries the paper (fallback shape: Robomimic ladder + diagnostics + partial RoboCasa) |
| Aug 20 – Sep 10 | W4 RoboCasa I-DSRL if go; else W5 path; writing continuous throughout |
| Sep 10–deadline | Freeze experiments, figures, statistical repeats, write |

The fallback paper (small-scale steering ladder + causal diagnostics + RoboCasa Phase A/B) is the
**primary plan**; RoboCasa I-DSRL is the upside. Slack for failed runs is nearly zero — hence
writer starts now and every eval doubles as a paper artifact.

## 6. Risks

1. **G1 fails on RoboCasa too** (prior: substantial, given M10). Mitigation: W3 keeps the question
   alive on a base where intent works; salvage framing pre-agreed in G3 row; T-mask arm already
   training as the architecture fix.
2. **Composite-task base SR ≈ 0** → no steerable tasks. Mitigation: unseen+atomic cells in W1.1;
   fallback testbed = libero-long (planning-flavored, pipeline exists).
3. **Sim-state restore unsupported in RoboCasa** → BoN blocked. Checked first in W2.4; fallback =
   fixed-seed reset distribution (weaker but serviceable).
4. **RL throughput too low** for online SAC on pi0.5. W1.1 measures before commitment; fallback =
   replay-heavy DSRL-NA or fewer, longer runs.
5. **C2 control pathologies**: AE code too easy/too hard to steer for reasons unrelated to
   semantics (code geometry). Mitigation: latent-usage checks + report C2's BC quality alongside.
6. **Single-seed ambiguity** (the M10 lesson). W3: 3 seeds. RoboCasa headline: ≥2 seeds or explicit
   error bars.
7. Babel node flakes / 48 h wall-clock (bit A0 once): idempotent resubmits, archive on every save,
   exclude known-bad nodes.

## 7. Open items needing user input

*(Resolved 2026-08-03: venue = ICLR 2027; Sreyas' codebase = not pursued; W3 harness default =
`external/dsrl` + SB3 fork with MIP wrapper.)*

1. **Release of the training hold** (steps 2–3 of the execution order: A0 relaunch, T-mask arm) —
   all trainings held per user decision 2026-08-03.
2. **GPU tier for non-training work**: eval grids and W3 runs are single-GPU jobs — confirm these
   are released while trainings are held.
