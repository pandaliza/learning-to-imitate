# M10 — Intent–Action Co-Prediction on pi0.5 (LIBERO-goal)
## Implementation Spec v0.2

**Stack.** LIBERO-goal on the pi0.5 **PyTorch** pipeline — same environment, data, and eval harness as
M8/M9 (`docs/steer_intent_experiments.md`). This arm is the successor to M9: intent stops being a
train-time-only auxiliary and becomes a first-class denoised variable in the action expert.

**Goal.** Replace the auxiliary intent head with a co-prediction objective: intent (h=8 future EEF
poses) and actions are denoised jointly in the action expert with *independent noise levels*
(diffusion-forcing style), enabling an intent-first inference schedule where actions are generated
conditioned on a fully denoised intent.

**Core hypothesis.** Decoupled noise levels turn intent from a regularizer into a causal variable. The
asymmetric schedule (clean intent → noisy actions during both training and inference) is what produces
the gain; tied-noise co-prediction should show little or no improvement over the auxiliary head
(Ablation A1 is the falsification test).

**Deploy-graph change (unlike M8/M9).** Every prior arm dropped the head at deploy, so eval was always
baseline pi0.5. M10's intent tokens exist at inference — the deploy graph changes. This is why the
pure-BC control (still missing, §6) matters even more here.

---

## 1. Notation & conventions

Adopt the conventions already in `pi0_pytorch.py` (they match below):

- Data `x₀` (intent block `I ∈ R^{8×d_I}` or action block `A ∈ R^{H×d_A}`), noise `ε ~ N(0, I)`.
- Noised sample: `x_τ = τ·ε + (1−τ)·x₀`, with `τ ∈ [0, 1]`, `τ=1` pure noise.
- Target velocity: `u = ε − x₀`. Network predicts `v_θ(x_τ, τ, ctx)`.
- Inference: Euler integration from `τ=1 → 0`, `x ← x + v_θ·dτ` with `dτ < 0`.

Dimensions (current LIBERO pi0.5 config):

- `d_A`: action dim as configured now (openpi pads to 32; 7 supervised dims for LIBERO). Unchanged.
- `d_I = 6`: absolute EEF pose per waypoint, **same parameterization as M9's targets** (the `eef` field
  the dataset already loads and normalizes; M9's 48-D concat = 8 × 6).
- `H = 10`: action horizon (all pi0.5 LIBERO configs use `action_horizon=10`; note MIP's chunk is 16
  but the pi0.5 expert predicts 10 — the train scripts slice `act[:, :10]`).
- `h = 8`: intent waypoints.

**Intent supervision.** Exactly the M9 path — targets extracted from the LIBERO hdf5 demos via
`steer_intent/intent_flow_dataset.py` (`wsm_intent_target`, `eef`-normalizer normalization, no
teacher/rollout machinery, no `w` cache) — with one change: a **lookahead stride Δ**.
`I_k = eef(t + k·Δ)` for `k = 1..8`, with `Δ = ceil(1.5·H/8) = 2`, so the intent horizon reaches
`t+16`, *past* the H=10 action chunk. M9's current targets (`k = 1..8` contiguous, inside the chunk)
are exactly the redundant regime this spec warns about (§9): intent ≈ subsampled cumsum(actions).
The lookahead beyond the chunk is where intent carries non-redundant information (where the policy is
*going*, not just what it does next). Implementation: a stride parameter on the future-slice in
`IntentFlowDataset.sample_to_data` plus a wider `pad_after` window.

---

## 2. Architecture

### 2.1 Backbone (unchanged)

PaliGemma VL backbone, fine-tuned exactly as in the M8/M9 recipe. Produces prefix tokens that the
action expert attends to via the shared-attention path with prefix KV cache — computed once per
observation, reused across all denoising steps (already how `sample_actions` works).

### 2.2 Action expert: extend the existing Gemma expert — do NOT build a new DiT

v0.1 specced a fresh 6-block DiT replacing a 3-layer MLP head. That maps wrong onto this codebase: the
3-layer MLP is M9's *aux* head; the action expert is already a full pretrained Gemma-300M transformer
(`gemma_expert`, width 1024) with adaRMS τ-conditioning. Building a fresh DiT would throw away the
pretrained expert and the pi0.5 checkpoint init. Instead, extend the existing expert's suffix:

```
suffix = [ I_1 … I_8 | A_1 … A_H ]        # 8 + 10 = 18 tokens (was 10)
```

- Intent tokens: new `intent_in_proj: Linear(d_I, width)` and `intent_out_proj: Linear(width, d_I)`,
  mirroring `action_in_proj`/`action_out_proj`. The separate projections type the tokens; no explicit
  type embedding needed at first (add one only if sanity check (2) fails).
- Position: the existing `position_ids = cumsum(pad_masks)` covers the longer suffix as-is.
- Init: full pi0.5 base checkpoint (same conversion as all prior arms); the only fresh params are the
  two intent projections. `intent_out_proj` zero-init so the intent branch starts silent.

### 2.3 Noise conditioning: pi0.5's adaRMS *is* adaLN-zero — make it per-token

The `RMSNorm` in `transformers_replace/.../modeling_gemma.py` already computes
`scale, shift, gate = Linear_zeroinit(cond)` and applies `x·(1+scale)+shift` with a gated residual —
adaLN-zero by another name, already trained. What's missing for decoupled noise: `adarms_cond` is one
vector per batch `(B, width)`, broadcast over the sequence. Required change:

```
c_I = time_mlp(sinusoidal(τ_I))          # existing time_mlp_in/out, shared for both blocks
c_A = time_mlp(sinusoidal(τ_A))
adarms_cond = [c_I × 8 tokens | c_A × H tokens]   # (B, 18, width)
```

Plumbing: in `RMSNorm.forward`, only `unsqueeze(1)` the modulation when `cond` is 2-D — a 3-D
`(B, S, width)` cond then broadcasts per-token natively. Backward compatible: every existing caller
passes 2-D and is untouched.

Notes vs v0.1:
- No gated cross-attention branch — context enters through the existing prefix-KV shared attention,
  which is the pretrained pathway. Don't add machinery the checkpoint doesn't have.
- No zero-init-gate warmup risk (§9 of v0.1): the expert and its adaRMS dense layers come pretrained.
- The slot-intent `intent_proj` (sums a pooled intent into `adarms_cond`) already exists at
  `pi0_pytorch.py::embed_suffix` — it is precisely ablation B1's conditioning pathway, nearly free.

### 2.4 Attention masking: two variants, both one-line `att_masks` changes

`make_att_2d_masks` builds block-causal masks from a 1/0 pattern (1 opens a new block). Current suffix
pattern is `[1] + [0]*(H−1)` — one bidirectional block.

**Joint (default, Variant J).** `[1] + [0]*(8+H−1)` — one block, full bidirectional self-attention
across intent+action tokens. Hierarchy imposed only by the inference schedule.

**Two-stream (Variant T).** `[1] + [0]*7 + [1] + [0]*(H−1)` — the cumsum machinery then gives exactly:
intent attends {prefix, intent}; actions attend {prefix, intent, actions}; intent never sees actions.
Architectural hierarchy; intent denoising provably independent of action-token state.

Ship J first, run T as ablation C1.

---

## 3. Flow matching loss & noise scheduling

### 3.1 Per-block noise sampling

Per training example, sample independently with the existing `sample_time` (Beta(1.5,1)·0.999+0.001):

```
τ_I ~ p(τ),   τ_A ~ p(τ)        # two independent draws
```

Independence is the load-bearing choice: it exposes the model to all four quadrants {clean, noisy}²,
including the inference-time regime (τ_I≈0, τ_A>0) that tied sampling never visits.

Stratification (recommended): with prob 0.25 force `τ_I = 0` (clean intent conditioning, the exact
inference regime); with prob 0.1 force `τ_I = 1` (intent uninformative, prevents over-reliance);
otherwise independent draws.

### 3.2 Loss

```
L = E [ w_I · ‖v_θ^I − u^I‖² / (8·d_I)  +  w_A · ‖v_θ^A − u^A‖² / (H·d_A) ]
```

- `v_θ^I` read from `suffix_out[:, :8]` via `intent_out_proj`; `v_θ^A` from the last H tokens via
  `action_out_proj` as now.
- Per-dimension normalization so the intent term isn't drowned by the larger action block.
- `w_A = 1.0`, `w_I ∈ {0.5, 1.0}` (sweep once, not a priority). Note the M8/M9 lesson: aux weight was
  a real lever (λ=1.0 plateaued 12 points under λ=0.25) — if w_I=1.0 underperforms, sweep down first.
- When `τ_I = 0` is forced (stratified case), mask the intent loss term — nothing to denoise; the
  intent tokens act purely as conditioning.

### 3.3 Full diffusion-forcing extension (v2, not in first build)

Per-*token* noise within each block (each waypoint its own τ) enables causal rollout and fine-grained
replanning à la Diffusion Forcing. Deferred: the block-level (2-noise) version captures the hypothesis
being tested. (The per-token adaRMS plumbing from §2.3 already supports it when the time comes.)

---

## 4. Inference schedules

All schedules cache prefix KV once (existing `sample_actions` structure); each denoising step is one
expert forward over the 18-token suffix.

### S1 — Intent-first (the method)

```
init:  I ← ε_I,  A ← ε_A
phase 1 (K_I steps):  Euler-update intent tokens only, τ_I: 1 → 0.
                      Action tokens held at pure noise; their conditioning stays τ_A = 1.
clamp: I fixed at its τ_I = 0 value; conditioning for intent tokens set to τ_I = 0.
phase 2 (K_A steps):  Euler-update action tokens, τ_A: 1 → 0, attending to clean intent.
```

Start with `K_I = 4, K_A = 10` (intent is low-dim and smooth; few steps suffice). Total NFE = 14 vs
the current 10 — negligible.

### S2 — Joint (baseline schedule)

Single shared τ: 1 → 0 over K = 10 steps, both blocks updated together. This is what a tied-noise
model would do; running it on the decoupled model isolates schedule effect from training effect.

### S3 — Action-first (control)

Mirror of S1. If S3 ≈ S1, the ordering isn't doing anything and gains come from co-training alone.

### Replanning (cheap MPC-flavored re-use)

At chunk boundaries, instead of full re-generation: re-noise the previous intent to τ_I ≈ 0.4, run 2–3
intent steps under the new observation, then phase 2 as usual. Temporally consistent plans across
chunks. Also the natural hook for uncertainty probes: variance of intent under repeated partial
re-noising is a plan-level uncertainty signal (secondary, §7).

---

## 5. Training configuration

Everything not listed is the M9 recipe verbatim (same sbatch skeleton, same data pipeline).

| Item | Setting |
|---|---|
| Backbone | PaliGemma, fine-tuned as in M8/M9 (unchanged) |
| Expert | Existing Gemma-300M expert, full fine-tune (no fresh DiT); suffix 10 → 18 tokens |
| Init | Full pi0.5 base checkpoint (same convert as prior arms); fresh params = 2 intent projections, `intent_out_proj` zero-init |
| Optimizer | M9 settings unchanged |
| Batch / steps | M9 recipe; eval at 4k/8k/12k like prior arms (12k was both arms' best) |
| Data | LIBERO-goal hdf5 via `IntentFlowDataset` + lookahead stride Δ=3 (§1) |
| Normalization | Existing normalizers: `eef` for intent, action stats as now (`--norm-stats-from-config`) |
| Monitoring | Per-block loss curves; intent-loss binned by τ_I; remember the M8/M9 lesson — **converged aux losses were not predictive of SR**, only rollouts were |

Sanity checks before any ablation: (1) with intent tokens masked out (pad_mask=0), the expert
reproduces current-baseline BC loss (token plumbing didn't break anything — this also exercises the
per-token adaRMS change in isolation); (2) with τ_I forced to 0 and ground-truth intent injected,
action loss drops substantially below baseline (the model *uses* clean intent — if this fails, nothing
downstream will work).

---

## 6. Ablation grid

| ID | Training noise | Inference | Arch | Question answered |
|---|---|---|---|---|
| A0 | — (M9-penult aux head) | standard pi0.5 | current | baseline: **79.0%** @ 12k, λ=0.25 (steer_intent_experiments §3) |
| A0b | — (pure BC, no intent anything) | standard pi0.5 | current | **the still-missing control from M8/M9 §9.1** — run it; it anchors every arm ever trained |
| A1 | tied (τ_I = τ_A) | S2 joint | J | does co-training alone help? (hypothesis: barely) |
| A2 | decoupled | S2 joint | J | training-time decoupling without schedule |
| A3 | decoupled | **S1 intent-first** | J | full method |
| A4 | decoupled | S3 action-first | J | is the ordering causal? |
| B1 | decoupled | S1 | J, intent → pooled embedding summed into adaRMS (the existing `intent_proj` pathway) instead of tokens | token-attention vs adaLN conditioning |
| C1 | decoupled | S1 | T (two-stream mask) | does architectural hierarchy beat scheduled hierarchy? |

Note A1–A4 are **one training run** for A2/A3/A4 (same decoupled checkpoint, three inference
schedules) plus one tied-noise run for A1. The grid is 3 trainings (tied, decoupled, B1) + C1 if
warranted — cheaper than it looks.

Priority order: A0b → A1 → A3 (the three-point story), then A2/A4 to attribute the gain, then B1/C1
if the method works. B1 is expected to lose (pooling destroys waypoint structure) — it exists to close
the "token attention vs adaLN" question with a number rather than an argument.

---

## 7. Evaluation

LIBERO-goal, 10 tasks × 20 trials = 200 rollouts per point, same harness as M8/M9. The eval script
needs a schedule flag since the deploy graph now contains intent tokens:

```bash
python examples/openpi/eval_libero_intent.py \
  --config-name pi05_base_nointent --task-suite libero_goal \
  --num-trials-per-task 20 --norm-stats-from-config --fp32 \
  --schedule {s1,s2,s3} \
  --checkpoint-dir $B/ldahiya_checkpoints/<run>/<step> --out logs/eval_<tag>.json
```

**Task metrics.** Success rate (the M8/M9 comparison table format — per-task breakdown, since prior
arms showed non-uniform shifts). Compare against A0 = 79.0% and A0b once it exists. PyTorch-pipeline
numbers only; never against JAX M0 = 96.5%.

**Intent quality.** ADE/FDE of predicted intent vs (a) realized policy EEF trajectory over the next
16 steps, (b) demo trajectory from the nearest matching state. Divergence between the two indicates
plan-execution mismatch. Free to log during rollouts — the eval loop already has sim EEF state.

**Mode commitment.** For fixed observation, sample N=16 generations; cluster action chunks (e.g. on
first-step direction). Metrics: intra-cluster variance (should drop vs A0/A1 — less mode averaging)
and entropy over clusters (should stay > 0 — commitment, not collapse). Direct test of the
multimodality argument; reuse the pattern from the flow-collapse diagnostics
(`examples/openpi/diag_flow_collapse.py`).

**Uncertainty hooks (secondary).** Intent re-noising variance vs eventual episode failure — AUROC.
Free experiment once S1 + replanning exist.

---

## 8. Codebase plan

Follow the repo's arm conventions: additive modules, fork the train script (live runs train out of
the originals and would pick up edits on resume), one sbatch per arm with env-var knobs.

| path | change |
|---|---|
| `external/openpi/.../models_pytorch/pi0_pytorch.py` | intent tokens in `embed_suffix` (+ the two projections, mask patterns J/T), per-block τ in `forward`, schedules S1/S2/S3 in `sample_actions` — all behind a config flag (e.g. `copred_h > 0`) so every existing path is untouched |
| `external/openpi/.../transformers_replace/.../modeling_gemma.py` | per-token adaRMS: skip the `unsqueeze` when `cond` is 3-D (backward compatible) |
| `steer_intent/intent_flow_dataset.py` | lookahead stride Δ=2 on the future slice (param, default 1 = current M9 behavior) |
| `steer_intent/copred.py` | noise sampler (independent + stratified §3.1), per-block loss, schedule helpers |
| `examples/openpi/train_pi05_m10.py` | fork of `train_pi05_m9.py` per fork convention |
| `examples/openpi/eval_libero_intent.py` | `--schedule {s1,s2,s3}` flag (standard path untouched → A0/A0b evals unchanged) |
| `slurm-scripts/train/pi05/train_pi05_m10_copred.sbatch` | launcher; knobs: `NOISE={tied,decoupled}`, `MASK={j,t}`, `INTENT_W`, `DELTA` |

**Milestones.**
1. Token plumbing + tied noise: A1 trains, sanity check (1) passes. (~week 1)
2. Decoupled sampler + stratification; sanity check (2) passes. (~week 1–2)
3. Inference schedules S1–S3 + `--schedule` eval flag; A0b pure-BC control launched in parallel (it
   needs nothing from this arm). (~week 2–3)
4. Ablation grid A-row; write-up. (~week 3–4)

---

## 9. Risks & open questions

**Intent–action redundancy.** With intent horizon inside the chunk, intent ≈ subsampled
cumsum(actions) and the model can satisfy both losses with one representation — no decoupling benefit.
Mitigated by the lookahead Δ (§1); verify by checking A3 > A1 specifically, and extend Δ if not.
(M8/M9's within-chunk targets never faced this because the head was dropped at deploy.)

**Intent tokens ignored.** The pretrained expert may route around the new tokens (the analogue of
v0.1's gate-collapse risk). Sanity check (2) is the early detector; the τ_I=1 stratification cell
prevents the opposite failure (over-reliance on intent).

**Clean-intent distribution shift.** At inference phase 2, intent is *model-generated* clean, not
ground-truth clean; training only shows ground-truth-derived noised intents. Testable directly:
phase-2 actions degrade with model intent but not with GT intent injected. If it bites, add
self-conditioning: generate intent, use it as the τ_I=0 conditioning during training with prob 0.5.

**Pipeline ceiling & statistics.** All PyTorch arms cap ~70–80% (vs JAX 96.5) — compare only within
the pipeline. Single seed, 200 trials/point: the M9-vs-M8 6pp gap was not error-barred; the A3-vs-A1
gap won't be either unless it's large. And **archive the best checkpoints immediately** — the trainer
keeps last 3, which is how M9's 79% weights were lost.

**Compute.** Suffix grows 10 → 18 tokens on a 300M expert — negligible vs the backbone. bs=32 +
expandable_segments as per the slot-intent OOM note if memory gets tight.
