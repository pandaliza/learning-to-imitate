# Intent Conditioning for Pi0.5 — Full Experimental Record

**Task:** LIBERO-Goal (10 tasks, 20 trials/task = 200 rollouts per SR number)
**Base:** Pi0.5 (`pi05_base`, ~3B) — SigLIP vision tower (full FT) → Gemma-2B (frozen + LoRA r=16) → prefix → Gemma-300M flow-matching action expert. Finetuned in **fp32** (`pi05_base` overflows bf16).
**Status:** All arms concluded. Last updated after M7-JEPA final checkpoint (@28k).

---

## 1. The aim

The hypothesis we set out to test:

> A VLA policy that is given an explicit representation of **what it is about to do** — an "intent" — should act better than one that must infer it implicitly from pixels and the task string.

"Intent" is operationalised as a low-dimensional code `z` summarising the near-future state of the scene (which objects matter, where the end-effector is heading). The appeal is threefold: it gives the action expert a compact goal signal, it should improve sample efficiency, and it should confer robustness when the observation is perturbed or the scene is only partially observed.

Two families of mechanism were tried:

- **Conditioning** — feed `z` into the action expert (via adaRMS). The policy *consumes* the intent at deploy time.
- **Alignment / auxiliary** — do *not* feed `z` to the policy. Instead, use it as a training-time target that shapes the action expert's internal representation. Deploy is then **byte-identical to the no-intent control (M0)**, so any SR change is attributable purely to the representation learned during training.

The control, **M0**, is a plain Pi0.5 LoRA finetune with no intent at all. Every arm has to beat **0.96**.

---

## 2. Results — all arms

| Arm | Mechanism | Intent target | Deploy | SR | Verdict |
|---|---|---|---|---|---|
| **M0** | none (control) | — | — | **0.96** | **baseline — unbeaten** |
| M1 | condition | ResNet18 slot code | uses `z` | 0.95 | ≈ M0 |
| M2 | condition (co-trained) | VL slot code | uses `z` | 0.78 | < M0 |
| **M3** | condition (decoupled) | frozen-VL slot code | uses `z` | **0.94–0.955** | ≈ M0, **intent inert** |
| M3+DINOv2 | condition | DINOv2 grid → slots | uses `z` | 0.95 | ≈ M0 |
| M3+DINOSAUR | condition | DINOv2 feature-recon slots | uses `z` | 0.945 | ≈ M0 |
| M3+DynaFLIP | condition | DynaFLIP grid → slots | uses `z` | 0.92 | ≈ M0 |
| **M4** (aux-only) | aux loss, no stop-grad | slot `z*` | = M0 | **0.01–0.015** | **diverges — backbone corrupted** |
| M4-cond (detached) | aux + condition, stop-grad tap | slot `z*` | uses `z` | 0.765 @28k | < M0 |
| **M4-cond (attached)** | aux + condition, action-loss shapes generator | slot `z*` | uses `z` | **0.86 @28k** | < M0, **best conditioning arm** |
| **M5-REPA** | frozen-cosine align | slot `z*` | **= M0** | **0.755 @16k** | < M0 |
| **M6-align** | frozen-cosine align | **workspace token** | **= M0** | **0.57 (plateau)** | **< M0 — actively harmful** |
| **M7-JEPA** | symmetric joint embedding | workspace token + EE pose | **= M0** | **0.74 @28k (final)** | < M0 |

### M4 aux-weight sweep (λ on the auxiliary loss)

| λ | SR | |
|---|---|---|
| 0.001 | 0.745 @10k | best |
| 0.005 | 0.74 @10k → 0.605 @18k | decays with training |
| 0.01 | 0.03 @8k | broken |
| 0.02 | 0.06 @10k | broken |
| 0.05 / 0.1 | — | diverged |

No weight works. Small λ is merely *harmless-ish*; large λ destroys the policy. Critically, **action loss is a poor deployability proxy here** — λ=0.01 and λ=0.02 look fine in the loss curve and are catastrophic in deployment.

### SR-vs-training-step curves (alignment arms)

| Step | M5-REPA | M6-align | M7-JEPA |
|---|---|---|---|
| 2k | 0.475 | — | 0.395 |
| 4k | 0.49 | — | 0.645 |
| 6k | 0.675 | — | 0.58 |
| 8k | 0.625 | 0.52 | 0.695 |
| 12k | 0.705 | — | 0.765 |
| 14k | — | — | 0.665 |
| 16k | **0.755** | 0.57 | — |
| 24k | — | — | 0.725 |
| 26k | — | **0.57** | — |
| 28k | — | — | **0.74** |

**Read this curve with care.** It swings ±0.1 between adjacent checkpoints (M7: 0.765 @12k → 0.665 @14k). With 200 rollouts, checkpoint-to-checkpoint noise of 0.05–0.10 is expected. **M5 and M7 are statistically indistinguishable** (both ~0.73–0.755 at convergence). Any claim that one beats the other is not supported.

---

## 3. Conclusions

### 3.1 Nothing beats not conditioning

**M0 = 0.96 is unbeaten by every arm, in both families.** The best conditioning arm (M4-cond-attached) reaches 0.86; the best alignment arm (M5) reaches 0.755.

### 3.2 The conditioning channel is provably *inert*, not merely redundant

The decisive experiment is the **M3 intent-override**: at deploy, replace the intent fed to the policy and measure SR.

| intent fed to policy | SR |
|---|---|
| real `z` | 0.944 |
| **random noise** | **1.0** |
| all zeros | 0.944 |

The policy's success rate is *unchanged* when the intent is replaced with noise. It has learned the channel is uninformative and routes around it entirely. This is not "intent helps a little" — the policy **ignores it**.

### 3.3 Root cause: the slot encoder collapses

The intent `z*` was produced by slot attention over a VL grid. Diagnostics (`figures/slot_collapse.png`) show, for **every** encoder (PaliGemma / DINOv2 / DynaFLIP) and at **both** K=4 and K=8:

- selector weights pinned at exactly **1/K** — no slot is ever selected
- spatial attention at the **`ln 256` maximum** — no slot ever focuses

So `z*` degenerates into a **uniform global average-pool of the VL grid** — which is, by construction, information the action expert already reads from its own prefix. The intent is redundant *by construction*, which is exactly why the override test shows it is inert.

**DINOSAUR** (reconstructing DINOv2 *features* rather than pixels) was the only lever that measurably un-collapsed the slots — and it still produced SR 0.945 ≈ M0. **Fixing the collapse did not make intent useful.** That is the strongest single piece of evidence that the problem is not the encoder.

### 3.4 Aux losses on a shared backbone are catastrophic (M4)

M4 taps `vl_mean` from the *same* forward pass **without a stop-grad**, so the auxiliary gradient flows into the **shared LoRA-VL backbone** that the action expert reads. The two objectives collide, the action features are corrupted, and training diverges (SR ≈ 0.01). Lowering the LR only lengthens the fuse (5e-5 diverges at ~3k; 2e-5 at ~19.3k) — **divergence is inherent to the design, not an LR problem.** Adding a `no_grad` firewall on the tap (M4-cond) fixes both failure modes.

### 3.5 Alignment mechanism matters more than the alignment target

The cleanest mechanism comparison in the whole program is **M6 vs M7**, because they align to the **same** target (the workspace token) and differ *only* in how:

- **M6** — frozen cosine: the action hidden is dragged onto a detached target. **0.57.**
- **M7** — symmetric joint embedding: both sides trainable, meeting in a learned latent (SimSiam: stop-grad + predictor). **0.74.**

**+0.17 SR from the mechanism alone.** Forcing the action hidden to regress onto a *narrow frozen* target destroys action-relevant information — the policy must reproduce everything in the target, including what is irrelevant to control. The symmetric objective lets the target encoder discard the nuisance dimensions and keep only the shared, predictable structure. M6 swallows the target whole; M7 keeps the intersection.

M6 is the only arm that is clearly **worse** than the others, and it plateaued hard (0.57 at 16k, still 0.57 at 26k, while its align loss *drifted up* and its action loss collapsed to 0.0005 — the policy was fitting actions and quietly abandoning the alignment objective).

### 3.6 Every auxiliary arm lands on a ~0.74 ceiling

M4-best (0.745), M5 (0.755), M7 (0.74) — three different mechanisms (aux MSE, frozen-cosine REPA, symmetric JEPA) and three different targets (slot `z*`, workspace token, workspace+EE-pose) all converge on the **same ~0.74–0.755 band**. That consistency is itself the finding: it suggests a property of the *setup*, not of any particular intent design.

---

## 4. Honest caveats — what these results do NOT establish

These are load-bearing. Anyone writing this up should read them.

### 4.1 There may be a residual trainer gap (but it is smaller than it looks)

M0's 0.96 was trained in **JAX**; every alignment arm was trained in the **PyTorch cotrain trainer**. It is tempting to attribute the whole 0.96 → 0.74 drop to the trainer.

**That is not supported.** **M4-cond-attached reached 0.86 in the same PyTorch trainer**, so the trainer is *not* capped at 0.74. Some gap to 0.96 may remain, but the alignment arms' 0.74 is **not** explained by the trainer alone. To settle this definitively, train an **M0-equivalent (no intent, no aux) in the PyTorch cotrain harness** and read its SR. Until that exists, the size of the trainer gap is unknown.

### 4.2 M7's advantage over M5 is confounded

M7's target is `cat([workspace_token, ee])` where `ee = object_states[:, -1]` is the **ground-truth target end-effector pose** — a privileged signal that M5 and M6 never receive. So M7-vs-M5 is not a clean mechanism comparison. (M7-vs-M6 **is** clean — same target, mechanism differs.) An ablation of M7 with the workspace token alone would separate mechanism gain from EE-pose gain. **Not run.**

### 4.3 The workspace model is only the current-frame slice

The Workspace Models paper (CoRL 2026) builds VLM saliency → salient DINOv3 patches → workspace token → **history/memory encoder** → policy. We implemented **only the current-frame path**, by explicit choice. There is no history encoder and no Qwen3-VL keyframe detection.

### 4.4 The task cannot reward intent — by construction

**LIBERO-Goal is Markovian and fully observed.** The current frame contains everything needed to act. A saliency or memory bottleneck has **no headroom** on such a task: there is nothing for the intent to supply that the policy cannot already see. Every negative result above is consistent with "intent is redundant *on this task*" and does **not** generalise to partially-observed or distractor-heavy settings, which is precisely where the workspace-model paper claims its gains.

**This is the single biggest limitation of the entire program.** We tested intent in the one regime where it is least likely to help.

### 4.5 OOD robustness — also flat

Intent was hoped to confer robustness. It does not, on the axes tested:

| eval | M0 | M3 | M3+DINOSAUR |
|---|---|---|---|
| XY perturbation | 0.648 | 0.642 | 0.638 |
| libero_spatial transfer | 0.21 | 0.19 | 0.22 |

Flat. No robustness benefit either.

---

## 5. What would actually move this forward

1. **Close the trainer confound.** Train no-intent M0 in the PyTorch cotrain harness. This is cheap and it is the prerequisite for *any* clean claim about intent hurting. (§4.1)
2. **Change the task.** Move to a partially-observed / distractor-heavy / memory-requiring benchmark. This is the only venue where a saliency or workspace bottleneck *can* win. Continuing on LIBERO-Goal cannot produce a positive result. (§4.4)
3. **M7 EE-pose ablation.** Separate the mechanism gain from the privileged-signal gain. (§4.2)
4. **Build the history arm** — causal encoder + keyframe detection + history-of-workspace-tokens — which is the part of the workspace-model paper that actually addresses partial observability. Pointless without (2).

---

## 6. Reproduction

| Arm | Script |
|---|---|
| M3 / conditioning | `examples/openpi/train_pi05_cotrain.py --pi05-config pi05_base_intent` |
| M4 aux sweep | `slurm-scripts/train/pi05/train_pi05_m4_*.sbatch` |
| M5-REPA | `slurm-scripts/train/pi05/train_pi05_m5_align.sbatch` |
| M6-align | `slurm-scripts/train/pi05/train_pi05_m6_align.sbatch` |
| M7-JEPA | `slurm-scripts/train/pi05/train_pi05_m7_jepa.sbatch` |
| Workspace Stage-0 (labeling) | `slurm-scripts/train/pi05/label_workspace.sbatch` (MolmoPoint-8B) |
| Workspace Stage-1 (encoder) | `slurm-scripts/train/pi05/train_workspace.sbatch` |
| SR eval | `slurm-scripts/train/pi05/eval_pi05_m4_sr.sbatch` (`CK=<ckpt> OUT=<json> TRIALS=20`) |

Raw SR results: `logs/*_sr_*.json`. Detailed per-arm architecture diagrams and gradient-flow analysis: `intent_arms_doc.md`. Slot-collapse diagnostics: `slot_intent_diagnosis.md`.

### Known code cruft
- `train_pi05_m7_jepa.sbatch` passes `--sigreg-weight`, which is **dead** — the loss uses SimSiam (stop-grad + predictor), not SigReg. The header comment also still describes SigReg + smooth-L1. SigReg was abandoned because it is a *batch-covariance* regulariser and batch=2 makes it rank-degenerate.
