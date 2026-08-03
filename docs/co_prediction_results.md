# M10 Co-Prediction — Experiment Results

Status as of **2026-07-24**. Spec: `docs/co_prediction.md` (v0.2). Branch `copredict`.
Prior-arm context: `docs/steer_intent_experiments.md` (M8/M9).

**Headline:** the core hypothesis is **not supported** on LIBERO-goal. Intent-first inference (s1)
never beats joint (s2) on the same weights; decoupled noise trains *slower* than tied; and the
pure-BC control reaches **72.0% at just 6k steps** — above every intent arm's 12k number. Tied
co-prediction does reach **78.5% @ 28k ≈ M9's 79.0% @ 12k**, so intent tokens in the deploy graph
cost nothing at the endpoint — but nothing here shows them *buying* anything.

---

## 1. Setup

- **Model:** pi0.5 PyTorch, from `pi05_base` (LoRA on PaliGemma LLM + full action-expert finetune).
  Suffix = 8 intent waypoint tokens + 10 action tokens; per-token adaRMS carries independent
  (tau_I, tau_A); fresh params are only `intent_in_proj` / `intent_out_proj` (zero-init out).
- **Intent targets:** future EEF waypoints, `I_k = eef(t + 2k)`, k=1..8 (lookahead stride Δ=2 →
  reach t+16, past the H=10 chunk), eef-MinMax-normalized, from the LIBERO hdf5 (M9 dataset path).
- **Training:** two arms, identical except noise coupling — `pi05_m10_copred` (decoupled +
  stratified: 25% tau_I=0 / 10% tau_I=1) and `pi05_m10_tied` (tau_I = tau_A). Effective batch 32,
  lr 5e-5, w_I = 1.0, 30k steps nominal.
- **Eval:** LIBERO-goal, 10 tasks × 20 trials (10-trial cells marked), fp32,
  `eval_libero_intent.py --config-name pi05_base_copred --schedule {s1,s2,s3}`.
- **Controls:** A0 = M9-penult reference (79.0 @ 12k, from prior writeup). A0b = pure BC via
  `train_pi05_m9.py --intent-flow-weight 0.0` (zero aux gradient, identical pipeline), 8 GPUs,
  effective batch 32.

Schedules: **s1** intent-first (4 intent steps → clamp → 10 action steps), **s2** joint (shared
tau, 10 steps), **s3** action-first control (intent held at pure noise).

---

## 2. Results — 20 trials/task (the quotable numbers)

### Main grid

| arm | ckpt | schedule | mean SR |
|---|---|---|---|
| **A0 pure BC** | **2k** | standard | **0.715** |
| A0 pure BC | 4k | standard | 0.640 |
| **A0 pure BC** | **6k** | standard | **0.720** |
| A0 pure BC | 12k+ (tail) | standard | *training* |
| A1 tied | 12k | s2 | 0.640 |
| **A1 tied** | **28k** (final) | s2 | **0.785** |
| A2 decoupled | 12k | s2 | 0.555 |
| A2 decoupled | 26k (final) | s2 | 0.720 |
| A3 decoupled | 12k | **s1 (method)** | 0.500 |
| A3 decoupled | 24k | s1 | 0.640 |
| A3 decoupled | 26k (final) | s1 | 0.695 |
| A4 decoupled | 12k | s3 (control) | 0.585 |
| *A0 M9 aux-head (ref)* | *12k* | *standard* | *0.790* |

### Per-task, key cells (20 trials)

| task | A0b @6k | A1 tied @28k | A2 dec @26k | A3 dec @26k | M9 @12k (ref) |
|---|---|---|---|---|---|
| open the middle drawer | 0.85 | 1.00 | 0.75 | 0.80 | 0.80 |
| put the bowl on the stove | 0.95 | 0.95 | 1.00 | 0.85 | 1.00 |
| wine bottle on cabinet | 0.85 | 1.00 | 0.85 | 0.95 | 0.85 |
| top drawer + bowl inside (2-stage) | 0.30 | 0.20 | 0.15 | 0.20 | 0.30 |
| bowl on top of cabinet | 0.95 | 1.00 | 1.00 | 0.95 | 1.00 |
| push plate to stove | 0.80 | 0.70 | 0.80 | 0.70 | 0.65 |
| cream cheese in bowl | 0.30 | 0.75 | 0.30 | 0.45 | 0.90 |
| turn on the stove | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bowl on the plate | 0.70 | 0.65 | 0.70 | 0.50 | 0.85 |
| wine bottle on rack | 0.50 | 0.60 | 0.65 | 0.55 | 0.55 |
| **mean** | **0.720** | **0.785** | **0.720** | **0.695** | **0.790** |

### Supplementary 10-trial curve (quick reads, higher variance)

| cell | 2k | 4k | 8k | 12k |
|---|---|---|---|---|
| dec × s1 | 0.41 | 0.41 | 0.56 | 0.48 |
| dec × s2 | — | — | 0.58 | 0.59 |
| dec × s3 | — | — | — | 0.53 |
| tied × s2 | — | 0.65 | 0.62 | 0.59 |

---

## 3. Findings

1. **The falsification test fired.** The hypothesis was that decoupled noise + intent-first
   inference turns intent into a causal variable and beats tied co-training (A3 > A1). Observed:
   A3 < A1 at every matched point (50.0 vs 64.0 @ 12k), and even the action-first control A4
   (58.5) beat A3 at 12k. The asymmetric schedule is a small consistent tax, not the active
   ingredient.

2. **s1 ≤ s2 on the same weights, always.** 12k: 50.0 vs 55.5. 26k: 69.5 vs 72.0. Conditioning
   actions on a fully-denoised model-generated plan never outperformed just denoising jointly.

3. **Decoupled training ≤ tied training.** Same schedule (s2): 55.5 vs 64.0 @ 12k; 72.0 @ 26k vs
   78.5 @ 28k. Independent noise draws (plus stratified corners) slow convergence without an
   endpoint payoff. Note the tied run *is* the B-of-the-doc "co-training alone" arm — and it's the
   best co-pred cell.

4. **Pure BC is essentially converged by 2k steps.** A0 = 71.5 (2k) / 64.0 (4k) / 72.0 (6k) — the
   three points are one value ±single-seed noise; BC reaches its ~70% level almost immediately off
   the strong pi05_base init. Every intent arm, by contrast, is in the 40s–50s at these steps and
   does not reach ~72 until 26k. **Intent supervision costs ~10× the steps to reach the same place
   BC starts at.** This holds across the whole program (aux-head M8/M9 and co-pred M10).

5. **The one open question is now sharp: does BC's tail rise or plateau?** Two live hypotheses,
   and the A0 tail (training past 6k) decides between them:
   - **BC plateaus ~72** → intent genuinely buys a higher ceiling: A1 tied 28k (78.5) would be a
     real +6pp over BC, just paid for with ~5× the steps. Intent = slower-but-better.
     *(Note this reverses the naive "intent hurts" read.)*
   - **BC climbs to ~78–79** → intent is pure overhead: same ceiling, far slower. Intent = strictly
     worse per GPU-hour.
   Everything downstream keys off this single curve. Highest-priority run.

6. **Where s1 showed life:** the 2-stage drawer task at 4k (0.50 vs 0.00 tied) — the only
   long-horizon task in the suite, and the only place plan-first inference ever posted the best
   score. Consistent with the standing M9 claim that intent should matter where planning/memory
   binds, which LIBERO-goal mostly doesn't test. Points to libero-long / robocasa, not more
   LIBERO-goal iterations.

7. **Per-task variance is the failure mode of s1 at low step counts.** At 4k, s1 was bimodal
   (0.80 on some tasks, 0.00 on others where tied aced) — plan-following amplifies both good and
   bad plans; with an immature intent field the bad-plan tail dominates. By 26k the bimodality
   largely washes out.

---

## 4. Incidents & caveats (read before extending)

- **The first M10 launch (19 h × 8 GPUs) was invalid — silent normalization no-op.** openpi's
  `_load_norm_stats` returns `None` (INFO "skipping") when `assets/<config>/<asset_id>/` is
  missing, and `Normalize(None)` is a no-op → trained on raw states/actions, evaled on normalized
  → ~5% SR on every schedule. Fixed: `assets/pi05_base_copred -> pi05_base_nointent` symlink +
  hard guard in `train_pi05_m10.py` (raises if norm_stats is None). Rule: any NEW config name
  needs its assets dir before training.
- **Final checkpoints are at `steps - save_every`, not `steps`.** The save check runs before the
  loop-exit check (inherited from the M9 trainer), so nominal-30k runs end at 28k. The decoupled
  run additionally hit the 48 h wall-clock at ~27.4k → final ckpt 26k (so tied/dec endpoints are
  28k vs 26k — slightly unmatched).
- **A0b crash:** NCCL collective timeout on babel-q5-28 at step ~6.7k (infra flake); resumed from
  6k. Optimizer state is not checkpointed → Adam moments reset at the resume point (same caveat
  as all preempt-resumed runs in this project).
- **Single seed everywhere; 20 trials/point.** The A1-vs-A2 endpoint gap (78.5 vs 72.0) also has
  a 2k-step endpoint mismatch. 10-trial cells (section 2 curve) are ±10pp noisy — the early
  "plateau at ~59" read from them was wrong; the arms were still climbing.
- **Checkpoint pruning ate the first-launch 4k/8k artifacts** before evals could run. All quoted
  late checkpoints are archived under `*_archive/` on the group volume.
- Do not compare any of these numbers to JAX-pipeline results (M0 = 96.5).

---

## 5. Artifacts

| what | where |
|---|---|
| trainings | `$B/ldahiya_checkpoints/pi05_m10_copred` (dec, final 26k), `pi05_m10_tied` (final 28k), `pi05_a0b_purebc` (in progress) |
| archived ckpts | `pi05_m10_{copred,tied}_archive/{8000,12000}` (+ tied 28k, dec 24k/26k live-dir finals) |
| eval JSONs (per-task included) | `logs/eval_m10_*.json` — `*_20t` = 20 trials; `A0b_*` = pure-BC control |
| train logs | `logs/m10_copred_9383009.out` (dec), `logs/m10_copred_9383010.out` (tied), `logs/a0b_purebc_*.out` |
| launchers | `slurm-scripts/train/pi05/train_pi05_m10_copred.sbatch` (knobs: TIED/MASK/INTENT_W/DELTA), `slurm-scripts/eval/eval_m10_copred.sbatch` (CKPT/SCHEDULE/TRIALS/OUT) |
| B | `/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi` |

Code (branch `copredict`, main repo + `external/openpi` submodule): intent tokens + per-token
adaRMS + schedules in `pi0_pytorch.py` / `modeling_gemma.py`; `pi05_base_copred` train config;
`IntentFlowDataset` lookahead stride; `train_pi05_m10.py`; `--schedule` in
`eval_libero_intent.py`; sanity diag `examples/openpi/diag_copred.py`.

---

## 6. Open items

1. **A0b full curve** (training in progress) — decides endpoint-neutral vs small-late-gain for
   ALL intent arms. Highest priority; everything in section 3.5 hinges on it.
2. **A0b 4k/2k evals** — in flight; completes the convergence-speed comparison.
3. If the intent program continues: **libero-long / robocasa**, where the planning argument
   actually applies (finding 6). Not more LIBERO-goal cells.
4. Untested from the spec (moot unless a harder testbed revives the method): C1 two-stream mask,
   B1 pooled-adaLN conditioning, w_I sweep, self-conditioning fix for clean-intent shift,
   replanning/uncertainty hooks.
5. Trainer hygiene for any next run: final-save-on-exit fix (off-by-one), commit the branch
   (implementation is still uncommitted working tree as of this writing).
