# Intent as a Steering Interface — Execution Plan v3 (RoboCasa-first B0/B1 + DSRL)

**Date:** 2026-08-04. Supersedes v2.2 (`intent_dsrl_plan_v2.md`, kept as history). Branch `copredict`.

**What changed:** the pi0.5 finetuning line is **closed** (user decision 2026-08-04, rationale §1).
The old A0/A1/A2/A3 arms and s1/s2/s3 schedule ablations are removed. The program is now two
from-scratch generative BC policies (B0/B1) over a frozen public VL encoder, trained on RoboCasa,
followed by DSRL steering arms. Venue target unchanged: ICLR 2027; fallback-paper thinking still
applies but the fallback is now "B0/B1 + steering on fewer tasks," not Robomimic (Robomimic =
optional code-debugging sandbox only, not a scientific gate).

---

## 1. Scientific question and why the pivot

> **Can RL steer a frozen generative robot policy more efficiently through a structured
> future-intent latent than through raw action-generation noise?**

**Why pi0.5 finetuning is closed:** intent machinery grafted post-hoc onto a VLA pretrained on an
inaccessible robot-data mix must compete with entrenched pathways; the model routes around the
graft. Evidence: M1–M4 (intent ≈ inert), M10 (pure BC converges by 2k steps, intent arms 10×
slower to the same ceiling; s1 ≤ s2 always; A0 most robust; A1 LIBERO-spatial zero-shot 0% vs A0
25%). We cannot influence the recipe at that scale; therefore we own the recipe at a scale we can.
The G1a offline probe on the pi0.5 A1 checkpoint (queued at time of writing) documents this
closure quantitatively. pi0.5 artifacts (A1 checkpoint + archive, W0 relaunch trainings if
retained) are evidence for the "grafting fails" narrative, not active development.

---

## 2. Base policies

### B0 — action-only generative BC

```
p(A | o, l)          z_A ~ N(0, I),   A = g_0(o, l, z_A)
```

- o = multi-camera observations + proprioception; l = task instruction; A = action chunk.
- B0 **must** be a diffusion/flow-style generative policy (deterministic BC provides no
  action-noise space for DSRL).
- Trained purely with the standard flow-matching action loss.

### B1 — intent-conditioned generative BC

```
p(I | o, l) · p(A | o, l, I)
z_I ~ N(0, I),   I = g_I(o, l, z_I)
z_A ~ N(0, I),   A = g_A(o, l, I, z_A)
```

- I = supervised future-EEF trajectory over a fixed lookahead horizon (h=8 waypoints, stride Δ=2,
  d_I=7 EEF-relative pos+quat — the existing RobocasaCopredDataset extraction, unchanged).
- **Strict two-stream / T-mask from day one:** intent tokens attend to observation/language only
  (never action tokens); action tokens attend to observation/language + intent. Intent is causally
  upstream by construction — no `channel_probe` caveats in this program.
- Trained purely with BC: supervised intent + action flow-matching losses, decoupled noise levels
  (independent τ_I, τ_A draws + stratified corners as in the M10/M11 recipe).
- Note: B1 keeps a stochastic action decoder (z_A exists), so an IA-DSRL extension later needs no
  retraining.

### Controlled constants (B0 ≡ B1 wherever possible)

- RoboCasa training data and task split (target/ seen 9 tasks, 4,543 eps);
- frozen PaliGemma vision-language observation backbone (hidden features, not generated text);
- proprioception encoder;
- policy width, depth, optimization budget;
- action representation (12D) and action horizon;
- action-decoder capacity.

PaliGemma is the shared observation encoder, **not a research variable**. Feature extraction /
token-resampling details are fixed by one minimal implementation test (§5 step 2), not ablated.

---

## 3. DSRL steering arms (after BC; base policies fully frozen)

| Arm | Base | RL controls | Notes |
|---|---|---|---|
| **A-DSRL** | B0 | z_A ~ π_A(z_A\|o,l) | standard action-noise steering baseline |
| **I-DSRL** | B1 | z_I ~ π_I(z_I\|o,l) **only** | z_A stays ~ N(0,I) during training; paired eval uses common-random-number / fixed-seed z_A so differences attribute to intent selection |
| **Dim-matched** | B0 | d_I-dim latent → fixed map into B0's action-noise space | tests "any low-dim space is easier"; known caveat: the projection imposes an arbitrary geometry — report as such |

### Minimum comparison (the paper table)

1. B0 without RL.
2. B1 without RL.
3. B0 + A-DSRL.
4. B0 + dimension-matched latent DSRL.
5. B1 + I-DSRL.

Report per arm: base SR, final SR, learning-curve AUC, steps-to-target-SR, and fraction of
available headroom closed `(SR_final − SR_base)/(1 − SR_base)`. Statistics discipline carries over
from v2.2: paired designs with CRN wherever rollouts share initial states; paired bootstrap CIs;
per-task before pooled; provisional thresholds never decide anything at the margin.

---

## 4. Pre-DSRL gates (on B1; run before any RL)

Do **not** start DSRL merely because B1 finished training. Verify:

1. **GT-intent injection** improves or meaningfully redirects action generation;
2. **shuffled/zero intent** damages or redirects behavior;
3. **varying z_I with z_A held fixed** changes task-level outcomes (approached object/fixture,
   subgoal order, destination region — not just endpoint dispersion);
4. **Best-of-N over z_I** from identical sim states exposes useful successful support.

Fail routing (unchanged from v2.2): intent ignored → architecture/representation fix (it cannot be
a pretrained-prior problem now — B1 is ours from init, so a failure here is informative about the
representation/loss, e.g. w_I, stratification, intent horizon); intent used but no useful support
→ support expansion (POSTBC-style) before DSRL.

Because B1 is T-masked from scratch, all diagnostic outputs are tagged `mask_regime="causal"`.

---

## 5. Execution order

1. **Env validation (BLOCKING EVERYTHING).** The RoboCasa eval loop scored the pi0.5 A1 arm 0/900
   uniformly — under suspicion of harness bugs (norm-stats version, obs construction,
   control_mode semantics, action (un)normalization). The env-side wiring is policy-agnostic, so
   B0/B1 inherit any breakage. Mandatory first move: **GT-demo-action replay through the eval
   env** — if replay fails, fix env wiring; if it succeeds, the remaining bug is in the pi0.5
   policy path (then irrelevant to B0/B1, but the obs/normalization audit still applies to the new
   eval path). Deliverable: `docs/robocasa_m11/eval_zero_debug_report.md`.
2. **Feature-cache pipeline + minimal extraction test.** Frozen public PaliGemma over all 1.87M
   frames × 2 cams, once. Candidate extractions: (a) per-camera mean-pooled last-layer features
   (+ instruction embedding) — cache ~15 GB; (b) small spatial token grid (e.g. 4×4 pooled tokens
   per cam) — cache ~150–250 GB (fp16; quantize if needed). Pick by a short B0-prototype run
   (which trains in ~minutes-hours on cached features), then freeze the choice. Cache lives under
   `/data/group_data/maxlab/common_datasets/pandaliza/` (never home).
3. **Train B0 + B1** (1 GPU each on cached features; both in an evening — see §7).
4. **Task selection:** eval B0/B1 on the 9 seen tasks with the validated harness; pick 1 atomic +
   1 composite with both bases at moderate, non-saturated SR. (Provisional candidates:
   PickPlaceCounterToCabinet, KettleBoiling — subject to measured SR.)
5. **Pre-DSRL gates** (§4) on B1 for the selected tasks.
6. **DSRL arms** (§3): SAC over z (chunk-level MDP), replay of (obs-features, z, chunk-return);
   dual critics per the DSRL-NA reference in `external/dsrl`; KL/entropy pull toward N(0,I).
   In-repo prior art to evaluate first: `examples/train/train_dsrl.py`, `train_residual_sac.py`
   (the early post-training implementations live in this repo).
7. Robomimic square-mh (checkpoints saved, paused): optional SAC/replay code-debug sandbox only.

## 6. Architecture spec (both arms)

| Component | Spec |
|---|---|
| VL encoder | frozen public PaliGemma (3B, 224px); hidden features; precomputed + cached |
| Proprio encoder | MLP on 16D state (obs_steps per dataset config) |
| Generative stack | flow-matching DiT, 6–8 blocks, width 512–768 (~30–80M trainable), per-token adaLN-zero τ conditioning |
| Suffix | B0: `[A_1..A_H]`; B1: `[I_1..I_8 | A_1..A_H]`, T-mask |
| Intent | h=8, Δ=2, d_I=7 (EEF-rel pos+quat), from RobocasaCopredDataset unchanged |
| Action | 12D RoboCasa action, horizon H=10, norm stats from assets/pi05_robocasa_copred (re-validated in §5 step 1) |
| Noise | B1: independent τ_I, τ_A + stratified corners (M10 recipe §3.1) |

## 7. Compute estimate

| Item | Cost |
|---|---|
| Feature precompute (one-time) | ~7–14 GPU-h total (1–2 L40S) |
| B0 training | ~2–6 h × 1 L40S (100–300k steps, bs 256 on cached features) |
| B1 training | same |
| Eval per task-cell (20 trials) | ~1–2 h × 1 L40S (sim-bound, unchanged) |
| Pre-DSRL gates | ~1 GPU-day total |
| DSRL per arm per task | sim-bound; measured throughput from eval logs decides budget — estimate after §5 step 4 |

Iteration time is the point: a full B0/B1 retrain-and-eval cycle is < 1 day on 2 GPUs.

## 8. Status of prior artifacts (2026-08-04)

- **pi0.5 A1 tied** — trained, archived (`m11_a1_tied_robocasa/`). Evidence artifact for §1.
- **pi0.5 A0 relaunch + T-mask trainings** — running at time of writing; **obsolete under this
  plan**; user decision pending on cancellation.
- **G1a probe (queued)** — pi0.5 closure evidence; recommend letting it run (1 GPU, cheap).
- **Phase-A eval JSONs (0/900)** — invalid pending §5 step 1 diagnosis.
- **square-mh B C checkpoints** (baseline 0.38/0.62-best SR, deterministic-collapse; flow-intent
  0.62/0.64, spread retained) — paused; debugging sandbox only.
- **Old plan v2.2** — superseded; its statistics/labeling discipline and POSTBC/architecture fail
  routing carry forward.
