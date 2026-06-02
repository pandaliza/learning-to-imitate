# Pi0.5 Fine-tuning with Slot-Intent Conditioning (LIBERO)

Authoritative design record for injecting MIP slot-attention intent into Pi0.5's
action head and fine-tuning on LIBERO. Supersedes the Robomimic-era plan in
[pi05_finetuning.md](pi05_finetuning.md) (which assumed concat-to-backbone and a
slot encoder at eval — both corrected below).

---

## TL;DR

Two stages, **zero JAX porting**, no train/eval distribution gap:

1. **Stage 1 — learn intent reps (MIP, PyTorch).** Train the MIP Config-A
   `flow_intent` model (slot variant) on a broad LIBERO subset
   (spatial + goal + object = 30 tasks, our "LIBERO-90" proxy). This yields two
   components: a `SlotObjectEncoder` (future frames → 64D intent, **training-time
   target generator only**) and an **intent flow map** (`noise → intent | obs`,
   the **deployable `p(z|s)` generator**).

2. **Stage 2 — fine-tune Pi0.5 action head (openpi, JAX).** Precompute a 64D
   `intent` per LIBERO timestep with the **intent flow map** (obs-only), store it
   in the LeRobot dataset, and fine-tune Pi0.5's **action expert only** with intent
   summed into the adaRMS conditioning. At eval, run the **same intent flow map as
   a PyTorch sidecar** to produce intent from the live obs.

The intent the action head sees at train and eval comes from the *same* obs-only
generator → identical distributions, nothing to debug.

---

## Why this shape (key insight)

MIP's `flow_intent` (Config A) is a **two-stage flow model**, and the slot encoder
is *only* a training-time target generator:

- **Training** ([flow_intent_agent.py:356-396](../mip/flow_intent_agent.py#L356-L396)):
  `SlotObjectEncoder` consumes **future frames** → 64D `intent_vec`, supervised by
  an object-state aux loss (and optional recon). That vector is the *regression
  target* for the **intent flow map**, which learns `noise → intent | obs_emb`,
  i.e. `p(z | s)`.
- **Eval** ([flow_intent_agent.py:567-608](../mip/flow_intent_agent.py#L567-L608)):
  there are **no future frames and no slot encoder**. Intent is *sampled* from the
  intent flow map using only the current obs, then `(obs_emb, intent) → action`.

So the deployment question — "where does intent come from with no future frames?"
— is already solved by the **generative intent flow map**, not by a deterministic
predictor (the `IntentPredictor` MLP only serves the "mean" eef variant) and not by
privileged future frames.

This is what lets Pi0.5 be a pure action decoder: we never port slot attention or
the flow map into JAX; we reuse the trusted PyTorch components and only move a 64D
vector across the boundary.

---

## Stage 1 — MIP slot-intent model on LIBERO

**Goal:** train `flow_intent` (slot) so the intent flow map learns a good `p(z|s)`
over a broad task distribution.

- **Data:** `libero_spatial` + `libero_goal` + `libero_object` (30 tasks) as the
  LIBERO-90 proxy. (LIBERO-90 isn't local; this subset is the agreed stand-in.)
- **Config:** `examples/configs/task/libero_goal_suite_image_slot_intent.yaml`
  (`intent_type: slot`, `intent_dim: 64`, `slot_obj_state_dim: 6`,
  `slot_obj_state_key: ee_states`, `slot_image_key: agentview_rgb`). Spatial/object
  suite variants to be added so all three train together.
- **Dataset support:** `LiberoDataset` already emits `intent_frames` (future
  `agentview_rgb`, CHW/255) and `object_states` (normalized future `ee_states`)
  for the slot path
  ([libero_dataset.py:249-257](../mip/datasets/libero_dataset.py#L249-L257)).
- **Launch:** `sbatch slurm-scripts/train/libero/train_libero_goal_suite_image_slot_intent.sbatch`.

**Stopgrad note.** Use the `v8_stopgrad` recipe (`slot_stopgrad_intent: true`):
the slot encoder is shaped only by the aux/recon losses, not co-adapted to the
flow map, so the 64D vector keeps a stable object-grounded meaning. The flow map
still learns `p(z|s)` against that stable target.

**Stage-1 output we carry forward:** the trained **intent flow map** + obs encoder
(used to generate intent from obs alone). The slot encoder itself is not needed in
Stage 2.

---

## Stage 2 — Pi0.5 action-head fine-tune

**Base checkpoint:** `pi05_libero` (already LIBERO-pretrained), at
`/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_libero/params`.

**Freeze / train:** `get_freeze_filter_action_head_only()`
([pi0_config.py](../external/openpi/src/openpi/models/pi0_config.py)) — freezes
SigLIP + PaliGemma expert-0; trains the **action expert + projections** (incl.
`intent_proj`). **No LoRA anywhere** (the action expert is already the small part;
LoRA buys nothing here).

### 2a. Precompute intent → LeRobot

For every LIBERO(-10) timestep, run the Stage-1 **intent flow map** on the current
obs window → 64D intent, and store it as the `intent` field in the LeRobot dataset.

- **Why flow-map (obs-only), not slot-encoder (future)?** It matches the eval
  distribution exactly. This is the principle behind MIP's
  `decoder_uses_sampled_intent` curriculum
  ([flow_intent_agent.py:432](../mip/flow_intent_agent.py#L432)) and your standing
  feedback: *never use GT future intent; use the proxy at both train and eval.*

### 2b. Injection point — adaRMS (v1)

Intent is projected `64 → action_expert.width (1024)` and **summed into the adaRMS
conditioning** alongside the timestep embedding — already scaffolded:

```python
# external/openpi/src/openpi/models/pi0.py  (embed_suffix, pi05 branch)
adarms_cond = time_emb
if self.intent_proj is not None and obs.intent is not None:
    adarms_cond = adarms_cond + self.intent_proj(obs.intent)   # FiLM-style global modulation
```

Plumbing already in place: `Observation.intent` field (model.py),
`intent_proj` Linear + `intent_dim` config field (pi0.py / pi0_config.py).

**Suffix-token variant is round-2, not v1.** It's moderate effort (~15-20 lines
across `embed_suffix`, the action-output slice, and the attention/position
bookkeeping in `sample_actions`
[pi0.py:249-274](../external/openpi/src/openpi/models/pi0.py#L249-L274)); the
output-slice shift is the error-prone part. adaRMS (global FiLM-style modulation)
is the natural match for "intent steers the whole action generation"; only adopt a
token if per-step attention to intent proves necessary.

### 2c. Eval / deployment

Run the Stage-1 intent flow map as a **PyTorch sidecar** on the live LIBERO obs →
64D intent → feed to Pi0.5 via `obs.intent`. No future frames, identical to 2a.

---

## Clean A/B (the actual experiment)

Two TrainConfigs, identical except intent on/off:

| Config | Backbone | Action expert | Intent |
|---|---|---|---|
| `pi05_libero_intent` | frozen | full FT | adaRMS-summed 64D from flow map |
| `pi05_libero_baseline` | frozen | full FT | none |

Same freeze filter, same data, same steps. The only variable is the intent signal.

**Round-2 ablations (documented, not built yet):** suffix-token injection;
LoRA-on-VLM + action-head full FT (lets grounding shift if frozen-backbone caps
performance); oracle-intent ceiling (intent from privileged future frames) to
upper-bound the gain.

---

## Compute estimate

Action-head-only full FT, 30k steps (openpi `pi05_libero` default), on Babel
(no H100s):

- **8× A100_80GB (NVLink) — recommended:** first evaluable ckpt ~6-7 h, full 30k
  ~15-20 h.
- **4× H200** is a strong fallback (fast per-GPU, easier to schedule than 8×A100).
- Avoid L40S/A6000 for multi-GPU training: PCIe-only → sublinear scaling, slower
  wall-clock despite more GPUs.

Likely converges before 30k on 10 tasks; watch the loss curve and early-stop.

---

## Status

- [x] `Observation.intent` field + `from_dict` (model.py)
- [x] `intent_proj` + adaRMS injection (pi0.py)
- [x] `intent_dim` config field + `get_freeze_filter_action_head_only()` (pi0_config.py)
- [x] `LiberoDataset` slot support (`intent_frames` / `object_states`) — verified shapes
- [x] `libero_goal_suite_image_slot_intent.yaml` (per-suite, for debugging)
- [x] `libero_all_suite_image_slot_intent.yaml` (30 tasks) + sbatch — composes + slot data path verified
- [ ] run Stage 1; recover the trained intent flow map
- [ ] MIP → LeRobot intent precompute script (flow-map, obs-only)
- [ ] `pi05_libero_intent` + `pi05_libero_baseline` TrainConfigs
- [ ] eval sidecar (intent flow map → Pi0.5 `obs.intent`)

---

## Open questions

- Which LIBERO-10 task(s) for Stage 2 fine-tune + eval? (LIBERO-Long is the
  standard transfer target.)
- Does the precompute run the flow map deterministically (mean / fixed seed per
  timestep) or sample? Sampling adds intent stochasticity to the dataset; a fixed
  draw is simpler for the first run.
- Sidecar obs alignment: the flow map conditions on MIP's encoder over MIP-format
  obs; confirm the eval-time LIBERO obs is normalized the same way the MIP dataset
  normalizer expects.
