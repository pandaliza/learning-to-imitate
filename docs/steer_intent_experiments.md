# Intent-supervision arms on pi0.5 (M8 / M9) — LIBERO-goal

Status as of **2026-07-19**. Covers the workspace-conditioned intent head (**M8**) and the stripped
head that replaces it (**M9**), plus the attach-point variant.

**Headline:** removing the entire workspace-model apparatus *improved* success rate. At matched steps
and matched aux weight, **M9 (no `w`) = 79.0%** vs **M8 (full machinery) = 73.0%**.

---

## 1. The question

All arms add a **train-time-only** auxiliary head that predicts future end-effector pose from the
policy's own representations. The head is dropped at deploy, so the inference graph is always baseline
pi0.5 — any SR difference comes purely from how the aux gradient reshaped the trunk during training.

The arms differ in *what supervises the trunk*:

| arm | aux signal | needs |
|---|---|---|
| **M8** | intent regression + `w_t` CFG conditioning + `w_{t+1}` JEPA target | stage-1 encoder, MolmoPoint labels, per-demo `w` cache |
| **M9** | intent only | nothing but the hdf5 |

M9 is a **removal**, not a replacement — nothing fills `w`'s seat. So any SR delta is attributable to
`w` alone (modulo the decoder/target change noted in §5).

The motivation for M9 came from max_simcho: workspace models exist to solve a *memory* problem, and
LIBERO has no memory problem, so the only thing doing work should be the auxiliary **supervision**.

---

## 2. Architecture

Both arms tap the **action expert's penultimate hidden** (pre-`action_out_proj`, `[B, 16, 1024]`),
mean-pool over the action-token axis, and run a 3-layer MLP.

### M8 — `steer_intent/intent_head.py::WSMIntentHead`
```
pooled -> fc1(+ CFG cond from w_t) -> h1 -> fc2 -> h2 -> fc3 -> intent
                                              |
                                              +-> JEPAPredictor -> cosine-align to w_{t+1}
L = L_BC + λ₁·MSE(intent, future-EEF mean) + λ₂·(1 − cos) + λ₃·SIGReg     (λ₃ = 0 in all runs)
```
- `w_t` conditions (input), `w_{t+1}` aligns (target) — different tensors, so no *exact* copy shortcut.
- `w` comes from a frozen stage-1 causal encoder, precomputed per demo to `.npy`.

### M9 — `steer_intent/intent_flow_head.py::IntentFlowHead`
```
pooled -> cond_proj -> (rectified-flow velocity field) -> intent
L = L_BC + λ₁·L_intent
```
Decoders: `flow` (rectified FM: sample τ~U(0,1), z~N(0,I), x_τ = τ·x₁+(1−τ)·z, regress x₁−z),
`l1` (conditional median), `mse` (= M8's objective).
Targets: `concat` (h=8 EEF trajectory, 48-D) or `mean` (6-D, = M8's target).

### M9-attach — `--intent-flow-attach {penult,vl}`
`vl` reads the **live PaliGemma prefix** image-token mean (2048-D) instead, via `return_vl_mean=True`
on the same DDP forward. Its gradient shapes only the VLM (LoRA adapters + SigLIP vision tower); the
action expert never sees it. See §6 — **this variant fails.**

---

## 3. Results — success rate

LIBERO-goal, 10 tasks × 20 trials = 200 rollouts. Deploy graph is baseline pi0.5 in every row
(`eval_libero_intent.py --config-name pi05_base_nointent --norm-stats-from-config --fp32`, **no** `--intent`).

| arm | 2k | 4k | 6k | 8k | 12k | 20k | 22k |
|---|---|---|---|---|---|---|---|
| M8 λ=1.0 | 49.0 | 54.5 | 48.5 | 61.0 | — | 61.0 | — |
| M8 λ=0.25 | — | — | — | — | **73.0** | 68.5 | 70.0 |
| **M9 penult / flow / concat, λ=0.25** | — | 60.5 | — | — | **79.0** | — | — |
| M9 vl / flow / concat, λ=0.25 | — | — | ~0 | — | — | — | — |

Figures: `figures/sr_m8.png`, `figures/sr_m9.png`.
Loss curves: published artifacts (M8, M9) — regenerate from `logs/*.out` via the parsers in
`.venv-label-artifacts/`.

### The matched comparison (12k steps, λ=0.25, both)

| task | M8 (with `w`) | M9 (no `w`) |
|---|---|---|
| open the middle drawer | 0.80 | 0.80 |
| put the bowl on the stove | 0.95 | **1.00** |
| put the wine bottle on top of the cabinet | 0.60 | **0.85** |
| open the top drawer and put the bowl inside | **0.50** | 0.30 |
| put the bowl on top of the cabinet | 1.00 | 1.00 |
| push the plate to the front of the stove | **0.85** | 0.65 |
| put the cream cheese in the bowl | 0.35 | **0.90** |
| turn on the stove | 0.85 | **1.00** |
| put the bowl on the plate | **0.90** | 0.85 |
| put the wine bottle on the rack | 0.50 | **0.55** |
| **mean** | **73.0** | **79.0** |

Not a uniform shift: M9 wins big on cream-cheese (+0.55) and wine-on-cabinet (+0.25), loses on the
2-stage drawer task (−0.20) and push-plate (−0.20).

---

## 4. Findings

1. **The workspace machinery is not load-bearing here.** M9 ≥ M8 at matched steps while needing no
   stage-1 encoder, no MolmoPoint labelling, no `w` precompute. This also makes the arm portable —
   robocasa / libero-long need only a config change, whereas M8 needs a retrained encoder per env.

2. **Aux weight is a real lever.** λ=1.0 plateaus at 61%; λ=0.25 reaches 70–73%. Too much aux pressure
   degrades the policy.

3. **Converged aux losses ≠ downstream success.** Both M8 settings drive `intent` → ~1e-3 and
   `jepa` → ~0.03, yet differ by 12 points of SR. Loss curves alone were not predictive; only SR was.

4. **The JEPA term was weaker than it looked.** Adjacent-frame `cos(w_t, w_{t+1}) = 0.935`, so simply
   forwarding the conditioning input scores 0.935. M8's JEPA head converged to ~0.91–0.97 — i.e. around
   or barely above the trivial copy baseline. The CFG dropout (p=0.2) is what kept the term non-vacuous,
   on 20% of samples. If revisited, predict further ahead (`w_{t+k}`) or drop `w_t` conditioning.

5. **The VL-reps attach point fails at every λ tried.** See §6.

---

## 5. Caveats (read before quoting these numbers)

- **No pure-BC control has been run.** Every number here is arm-vs-arm. We can say M9 > M8; we *cannot*
  yet say the intent loss beats plain finetuning. **This is the top open item.**
- **M9 changed three things at once** vs M8: removed `w`, swapped MSE→flow, swapped mean→concat. The
  clean isolation of `w` alone is `--intent-flow-decoder mse --intent-flow-target mean`, not yet run
  to completion (the preempt job was cancelled at 4.2k).
- **Single seed, 200 trials/point.** The 6pp M9-vs-M8 gap is not error-barred.
- **M9's 12k checkpoint — the one that scored 79% — has been pruned** (trainer keeps last 3). The
  number is recorded; the weights are gone. Surviving: 24k/26k/28k.
- **PyTorch vs JAX ceiling:** M0 = 96.5% was produced by the JAX pipeline; all PyTorch arms here cap
  around 70–80%. Do not compare across pipelines.
- Stage-1 mismatch (M8 only): the encoder was trained with `window=8` but `precompute_w.py` ran the
  full demo (`c_horizon=1000`), so cached `w` summarizes full history. Never resolved.

---

## 6. The VL-attach failure

`--intent-flow-attach vl` puts the aux gradient into the shared VLM. Both weights tried diverge, with
different shapes:

| λ | behaviour | final `action` |
|---|---|---|
| 0.25 (concat) | healthy to ~7k, then explodes and thrashes 5 → 92 | 24.2 (killed) |
| 0.05 (mean) | healthy to ~6k, jumps to 1.34 at 8k, then **plateaus ~1.0 for 22k steps** | 1.02 |

For reference M9-penult ends at `action = 0.0007` — the λ=0.05 vl run is ~1400× worse and never
recovers. Its 6k checkpoint evals to **0%** (every task 0/20).

The step-0 assertion (`[M9-ATTACH check] VLM params w/ non-zero grad: 563/689`) confirms the gradient
*does* reach the VLM — including SigLIP's patch embedding — so this is not a plumbing bug. Two
independent λ values failing suggests the problem is **structural**, not a tuning issue: pushing an
EE-prediction gradient through the shared VLM corrupts the representations the action head depends on.

Remaining levers if pursued: a separate (much lower) LR on the VLM path, or a warmup delaying the aux
loss until BC settles. Lowering λ further is not promising.

---

## 7. Code map

Nothing below modifies the arms that came before it; each is additive.

| path | role |
|---|---|
| `steer_intent/intent_head.py` | M8 `WSMIntentHead` (CFG + JEPA) |
| `steer_intent/intent_flow_head.py` | M9 `IntentFlowHead` (flow / l1 / mse) |
| `steer_intent/intent_flow_dataset.py` | emits `wsm_intent_target` with **no** `w` cache; concat/mean |
| `steer_intent/networks/` | ported export modules (adaln_zero, workspace_latent, wsm_cfg_cond, jepa_align_head, …) |
| `examples/openpi/train_pi05_cotrain.py` | arms M1–M8 |
| `examples/openpi/train_pi05_m9.py` | fork: + M9 (`--intent-flow`) |
| `examples/openpi/train_pi05_m9_attach.py` | fork: + `--intent-flow-attach {penult,vl}` + step-0 no-op assertion |
| `slurm-scripts/train/pi05/train_pi05_m8_wsm.sbatch` | M8 launcher |
| `slurm-scripts/train/pi05/train_pi05_m9_intent.sbatch` | M9 launcher |
| `slurm-scripts/train/pi05/train_pi05_m9_attach.sbatch` | attach launcher |
| `slurm-scripts/train/pi05/train_pi05_m9_preempt.sbatch` | preempt + `--requeue` + auto-resume |
| `docs/ablations_m8_m9.tex` | ablation grid (LaTeX table) |

The forks exist because a live run trains out of the original and would pick up edits on resume. **Fold
them back together once the arms settle.**

---

## 8. Running things

```bash
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi

# M9 (best arm so far)
OUT=$B/ldahiya_checkpoints/pi05_m9_xxx \
  sbatch slurm-scripts/train/pi05/train_pi05_m9_intent.sbatch

# M9 variants: decoder / target / weight / attach point
OUT=... DECODER=mse TARGET=mean INTENT_W=0.25 ATTACH=penult \
  sbatch slurm-scripts/train/pi05/train_pi05_m9_attach.sbatch

# SR eval — all arms deploy as baseline pi0.5, so NO --intent
python examples/openpi/eval_libero_intent.py \
  --config-name pi05_base_nointent --task-suite libero_goal \
  --num-trials-per-task 20 --norm-stats-from-config --fp32 \
  --checkpoint-dir $B/ldahiya_checkpoints/<run>/<step> --out logs/eval_<tag>.json
```

**SLURM notes.** `normal` QOS caps at 8 GPUs on partition `general`. `maxlab_qos` (16 GPUs) requires
`--partition=maxlab`, whose 2 nodes carry **RTX_PRO_6000, not L40S** — the `--gres=gpu:L40S:N` line
will never match there. `preempt_qos` (24 GPUs, partition `preempt`) is killable; the preempt sbatch
handles it with `--requeue` + auto-resume from the newest complete checkpoint. Caveat: **optimizer
state is not checkpointed**, so every resume resets Adam moments.

---

## 9. Open items

1. **Pure-BC control** — the missing denominator. Nothing here is interpretable in absolute terms
   without it. *(highest priority)*
2. **M9 `mse` + `mean`** — isolates `w` alone; the cancelled preempt run only reached 4.2k.
3. **Eval M9-penult @28k** — did it climb past 79% or plateau/decline like M8?
4. **Stop-grad control** — same head, gradient blocked at the tap. Proves the *supervision* shapes the
   trunk rather than the extra parameters.
5. Decoder (`l1`), target horizon (h ∈ {16, 32} — a prior experiment suggested mean h=32 was best),
   λ sweep on M9 (0.25 was inherited from M8 and the losses are not on the same scale).
6. **RL-token attach** (PI-style read-out token) — the untried middle ground between penult and vl.
7. **robocasa / libero-long** — M9 ports unchanged; this is the strongest argument for the arm.
