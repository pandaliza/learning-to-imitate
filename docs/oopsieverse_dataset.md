# OopsieVerse Dataset

Source: https://huggingface.co/datasets/ut-robin-lab/oopsieverse-demos
(scope of this doc: `paper_experiments/teleop_data/`, 1.33 GB)

OopsieVerse is a teleop demonstration dataset built to study robot-induced
**damage/safety violations** ("oopsies") during manipulation, and whether
giving a human teleoperator **live corrective feedback** during data
collection reduces those violations. Every task tracks per-object/per-link
"health" and contact-force "damage" alongside the usual state/action stream.

Download it with `mip/datasets/oopsieverse_dataset.py` (see [Downloading](#downloading) below).

## Repo layout

```
paper_experiments/
  teleop_data/
    add_firewood/
      all_data.hdf5              # = without + with, concatenated
      with_live_feedback.hdf5
      without_live_feedback.hdf5
    pick_egg/          ... (same 3 files)
    shelve_item/       ... (same 3 files)
    wipe_countertop/   ... (same 3 files)
```

There is also a `behavior1k/teleop/*_safe.hdf5` / `*_unsafe.hdf5` directory
at the repo root (13 tasks x 2), which is a **different, unrelated tree**
outside `paper_experiments/` -- not covered here.

## Two simulator backends, two HDF5 schemas

The four tasks split evenly across two different simulators, and each
simulator writes a different HDF5 layout. There is **no unified schema**
across all four tasks.

| Task | Backend | Schema | Demos (w/o + w/) | Action dim | State dim |
|---|---|---|---|---|---|
| `add_firewood` | BEHAVIOR-1K / OmniGibson | `behavior1k` | 30 + 30 = 60 | 7 | 358 |
| `shelve_item` | BEHAVIOR-1K / OmniGibson | `behavior1k` | 45 + 45 = 90 | 7 | 3191 |
| `pick_egg` | RoboCasa / robosuite | `robocasa` | 30 + 30 = 60 | 12 | 170 |
| `wipe_countertop` | RoboCasa / robosuite | `robocasa` | 30 + 30 = 60 | 12 | 170 |

`all_data.hdf5` is exactly the concatenation of `without_live_feedback.hdf5`
(demo_0..N-1) and `with_live_feedback.hdf5` (demo_N..2N-1) — confirmed by
demo counts and byte sizes. `with_live_feedback` episodes run ~2-2.6x longer
than `without_live_feedback` episodes on the same task (more steps spent on
corrections).

### Schema A: `behavior1k` (`add_firewood`, `shelve_item`)

```
data/
  demo_0/
    state:      (T+1, D) float32   # opaque, flattened OmniGibson sim-state dump
    state_size: (T+1,)   int64     # byte-length of the state dump per step (mostly constant)
    action:     (T, 7)   float32   # [dx, dy, dz, drx, dry, drz, gripper] IK-controller delta pose + gripper
    reward:     (T,)     float32   # always 0.0 in every demo inspected (DummyTask, no reward defined)
    terminated: (T,)     bool      # always False in every demo inspected
    truncated:  (T,)     bool      # always False in every demo inspected
    init_metadata/            # empty group in every demo inspected
  demo_1/ ...
  attrs:
    config:     JSON str  # full OmniGibson env config (scene, robot, controllers,
                           # damageable objects + their damage_threshold/scale params)
    scene_file: JSON str  # full initial scene state dump (per-object pose/vel/joint state,
                           # used to reconstruct object identities/ordering for `state`)
    n_episodes, n_steps
  demo attrs:
    num_samples: int                # == T+1 (length of `state`)
    health_list_link_names: list[str]  # robot link names tracked for damage
    transitions: '{}'               # empty in every demo inspected
```

**Important caveat**: `state` is *not* a named observation vector. It's
OmniGibson's raw `dump_state()` output — a flat concatenation of
pose/velocity/joint-state for every object in the scene (doors, walls,
ceilings, floors, the fireplace, logs, the robot, ...), in an order that is
only recoverable by cross-referencing `scene_file`/`config` and replaying
through OmniGibson itself. Roughly 60% of the 358 (or 3191 for `shelve_item`)
dims are constant across a whole demo (static furniture); the varying dims
correspond to the robot and the manipulated objects, but there is no
`robot0_eef_pos`-style key to slice out directly from the file alone.
**Decoding this into usable low-dim/image observations requires running the
OmniGibson/BEHAVIOR-1K simulator** (not a dependency of this repo) to
`load_state()` and query cameras/proprioception at each step.

The 7-dim `action` is consistent with a single-arm IK end-effector delta
controller (`arm_0: InverseKinematicsController`) + 1-dim gripper (seen in
the env `config`), matching the Franka Panda used in both `behavior1k` tasks.

### Schema B: `robocasa` (`pick_egg`, `wipe_countertop`)

```
data/
  demo_0/
    actions: (T, 12) float64   # mobile-manipulator delta-pose action (see below)
    dones:   (T,)    bool
    rewards: (T,)    float32   # always 0.0 in every demo inspected
    states:  (T, 170) float64  # raw robosuite/MuJoCo sim state (same opacity caveat as above,
                                # but robosuite's `env.reset_to({"states": ...})` *can* decode this
                                # if robosuite/robocasa is installed)
    obs/
      robot0_eef_pos, robot0_eef_quat, robot0_eef_quat_site
      robot0_base_pos, robot0_base_quat, robot0_base_to_eef_pos, robot0_base_to_eef_quat(_site)
      robot0_gripper_qpos, robot0_gripper_qvel
      robot0_joint_pos(_cos/_sin), robot0_joint_vel, robot0_joint_acc
      robot0_proprio (23,), robot0_proprio-state (68,)
      <object>_pos, <object>_quat, <object>_to_robot0_eef_pos, <object>_to_robot0_eef_quat
        # e.g. egg_pos/egg_quat for pick_egg; sponge_pos/sponge_quat for wipe_countertop
      object-state (14,)          # concatenated task-relevant object state
      health (16,) float32        # per-object/link damage health, 100 -> 0 (see below)
      # NOTE: no image/rgb keys -- this dataset is low-dim proprioception only
    info/
      damage_info: (T,) object   # per-step JSON string, contact forces + damage per body/link
                                  # (drives the `health` values in obs/)
      obs_info: (T,) object
      # wipe_countertop only:
      newly_wiped, proportion_wiped, total_markers, total_wiped
  attrs (per demo): ep_meta (JSON: RoboCasa scene/object/fixture layout),
                    health_list_link_names, model_file, num_samples
```

`health` starts at `[100, 100, ..., 100]` and individual entries drop toward
`0` when the corresponding object/link takes damage (e.g. in a `pick_egg`
demo where the egg is dropped, `health[0]` — the egg's slot — drops to `0`
by the end of the episode while the rest stay at 100). `info/damage_info`
carries the raw per-body contact-force/impact data (`mechanical.damage`,
`impact_forces`, `contacts: [...]`) that health is computed from.

The robot is a Panda-on-Omron-base mobile manipulator (`PandaOmron`); the
12-dim action is a 6-DoF eef pose delta + 1 gripper (matches the value range
of dims 0-6) followed by mobility/base DoFs that are held near `0`/`-1`
(unused) in both tabletop tasks inspected. The layout matches `mip`'s Robomimic
convention (`obs` dict + `actions`/`states`/`rewards`), so these two tasks load
through `RobomimicDataset` — see [Training](#training-state-policies-on-pick_egg--wipe_countertop).

## Downloading

```bash
# one task, one variant (downloads to the shared HuggingFace cache)
python -m mip.datasets.oopsieverse_dataset --task add_firewood --variant all_data

# one task, all three variants, and print a schema summary
python -m mip.datasets.oopsieverse_dataset --task pick_egg --all-variants --inspect

# everything under paper_experiments/teleop_data/ (~1.33 GB)
python -m mip.datasets.oopsieverse_dataset --all-variants --inspect
```

Or programmatically:

```python
from mip.datasets.oopsieverse_dataset import download_oopsieverse, inspect_oopsieverse

path = download_oopsieverse("wipe_countertop", variant="without_live_feedback")
inspect_oopsieverse(path)
```

Files land in the standard `huggingface_hub` cache
(`hf_hub_download(repo_id="ut-robin-lab/oopsieverse-demos", ..., repo_type="dataset")`),
matching the existing `dataset_repo`/`dataset_filename` convention used by
`mip/datasets/robomimic_dataset.py`.

## Analysis: demo counts, episode lengths, and "oopsie" rates

Computed over every demo in all 8 `without_live_feedback`/`with_live_feedback`
files (`all_data.hdf5` skipped — it's just their concatenation).

**270 demos total** across 4 tasks x 2 conditions (150 BEHAVIOR-1K, 120 RoboCasa):

| Task | Condition | Demos | Steps (total) | Ep. length (min/median/max) |
|---|---|---:|---:|---|
| `add_firewood` | without feedback | 30 | 2,835 | 65 / 94 / 127 |
| `add_firewood` | with feedback | 30 | 6,935 | 162 / 239 / 349 |
| `shelve_item` | without feedback | 45 | 13,478 | 206 / 271 / 477 |
| `shelve_item` | with feedback | 45 | 13,761 | 164 / 290 / 659 |
| `pick_egg` | without feedback | 30 | 4,803 | 115 / 155 / 212 |
| `pick_egg` | with feedback | 30 | 7,834 | 174 / 246 / 409 |
| `wipe_countertop` | without feedback | 30 | 9,692 | 143 / 271 / 826 |
| `wipe_countertop` | with feedback | 30 | 9,608 | 110 / 297 / 821 |

Live-feedback episodes run **1.6x-2.4x longer** than non-feedback ones on
the same task (time spent on corrections) — biggest gap on `add_firewood`
(94 -> 239 median steps), smallest on `wipe_countertop` (271 -> 297).
`shelve_item` has 45 demos per condition instead of 30; the other three
tasks all have 30/30.

### "Oopsie" (damage) rates — the RoboCasa tasks (`pick_egg`, `wipe_countertop`)

These are the only two tasks with a usable per-step `health` signal in the
file (0-100 per tracked object/link). Two thresholds:
- **any oopsie**: `min(health) < 100` somewhere in the episode (some damage occurred)
- **severe oopsie**: `min(health) <= 0` (an object/link fully destroyed — e.g.
  the egg breaks, the sponge/wall takes critical damage)

| Task | Condition | Any-oopsie rate | Severe-oopsie rate | Mean min-health |
|---|---|---:|---:|---:|
| `pick_egg` | without feedback | 26/30 = **87%** | 12/30 = **40%** | 34.3 |
| `pick_egg` | with feedback | 11/30 = **37%** | 0/30 = **0%** | 93.9 |
| `wipe_countertop` | without feedback | 29/30 = **97%** | 6/30 = **20%** | 55.5 |
| `wipe_countertop` | with feedback | 24/30 = **80%** | 0/30 = **0%** | 87.5 |

This is the dataset's headline result: **live corrective feedback drives
severe damage (health hitting 0) to exactly 0% on both RoboCasa tasks**,
and roughly halves the any-damage rate on `pick_egg` (87% -> 37%). Task
success itself doesn't obviously suffer — `wipe_countertop`'s final
`proportion_wiped` averages 1.0 (fully wiped) in both conditions.

### BEHAVIOR-1K tasks (`add_firewood`, `shelve_item`): no in-file damage signal

Unlike the RoboCasa tasks, these files carry **no per-step health/damage
array** — only a per-demo `transitions` attribute, which was `{}` (empty)
in *every one of the 150 BEHAVIOR-1K demos checked*, in both conditions.
So while the env `config` defines `damage_threshold`/`damage_scale` params
for the fireplace/logs/robot links (i.e. the simulator *is* tracking
damage), that per-episode outcome isn't surfaced in the HDF5 the way it is
for RoboCasa — recovering it would require replaying each demo's `state`
through OmniGibson. `reward`/`terminated`/`truncated` are likewise always
`0.0`/`False`/`False` for all 150 of these demos, so there's no success or
safety label at all available directly from the file for these two tasks.

## Training state policies on `pick_egg` / `wipe_countertop`

The BEHAVIOR-1K tasks are not trainable here (no named observations; decoding
needs OmniGibson). The two RoboCasa tasks are:

```bash
python examples/train/train_oopsieverse.py task=pick_egg_state          # baseline
python examples/train/train_oopsieverse.py task=pick_egg_state_intent   # + intent conditioning
# same for wipe_countertop_state / wipe_countertop_state_intent
```

- **Data**: both feedback conditions are combined for training; the last 10% of
  demos in each (3 per condition) are held out. One min-max normalizer is shared
  across conditions. Obs = `object-state` (14) + `robot0_eef_pos` (3) +
  `robot0_eef_quat` (4) + `robot0_gripper_qpos` (2); actions are the raw 12-dim
  deltas (`abs_action: false`).
- **Evaluation is offline.** OopsieVerse envs come from the authors' own RoboCasa
  fork and `safety_benchmark` package, which aren't available here, so there is
  no rollout, success rate, or damage rate. Instead, `offline_eval` reports the
  normalized MSE of the executed action chunk (steps `obs_steps-1 ..
  obs_steps-1+act_steps`) on held-out demos, per feedback condition. Intent
  models are scored with three intent sources: the learned `IntentPredictor`,
  the constant-velocity proxy, and the ground-truth future (upper bound).
- **Shared-machine note**: launch with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1`. Without it, each dataloader worker starts full
  thread pools (12 parallel runs reached ~1,800 threads).

### Result: intent conditioning vs baseline

Flow policy, MLP network, 50k steps, batch 256, 3 seeds each. Intent = mean
future eef pose (pos + quat) over the next 8 steps, appended to obs. Numbers are
held-out chunk action MSE (normalized, ×10⁻³, 9-step ODE sampling), mean ± std
over seeds; "t0" is the first executed action only.

| Task | Held-out split | Baseline | Intent (learned) | Intent (const.-velocity) | Intent (oracle) |
|---|---|---:|---:|---:|---:|
| `pick_egg` | without feedback | 22.8 ± 1.3 | 13.9 ± 0.8 | 15.6 ± 0.4 | 9.1 ± 0.6 |
| `pick_egg` | with feedback | 20.9 ± 1.0 | 14.5 ± 0.8 | 14.7 ± 0.4 | 11.1 ± 0.6 |
| `pick_egg` | all | 21.7 ± 0.7 | **14.3 ± 0.7** | 15.0 ± 0.4 | 10.2 ± 0.5 |
| `pick_egg` | all, t0 | 13.3 ± 0.3 | **6.3 ± 0.2** | 6.6 ± 0.2 | 5.6 ± 0.1 |
| `wipe_countertop` | without feedback | 85.2 ± 8.2 | 64.1 ± 1.4 | 65.6 ± 1.6 | 49.9 ± 1.5 |
| `wipe_countertop` | with feedback | 58.9 ± 1.6 | 50.6 ± 3.5 | 50.4 ± 2.7 | 41.9 ± 2.5 |
| `wipe_countertop` | all | 67.9 ± 3.2 | **55.2 ± 2.5** | 55.7 ± 2.1 | 44.7 ± 2.1 |
| `wipe_countertop` | all, t0 | 55.0 ± 4.4 | **37.9 ± 3.2** | 35.5 ± 2.5 | 32.7 ± 2.0 |

- Intent conditioning lowers held-out error on both tasks: −34% (`pick_egg`)
  and −19% (`wipe_countertop`) over the chunk, −53% and −31% on the first
  action. The gap is well outside seed variance and persists from step 5k on.
- **The gain is mostly an explicit-velocity effect, not goal inference.** The
  constant-velocity proxy (extrapolate from the last two obs steps; no learning)
  matches the learned predictor on both tasks, and on `wipe_countertop` it is
  the more accurate intent estimate (MSE 0.66 vs 0.83 ×10⁻³). With 8-step
  lookahead and delta actions, "where the eef is heading" ≈ current velocity.
- The oracle leaves headroom (−10 to −11 points beyond the learned predictor),
  so a stronger intent predictor or a longer `intent_horizon` could help more.
- The benefit is larger on the no-feedback held-out demos than on the feedback
  demos for both tasks, so there's no sign that intent specifically helps
  with correction-heavy demos. With 3 held-out demos per condition, don't read
  much into per-condition differences.
- With 1-step sampling, intent models are *worse* than baseline (`pick_egg`
  167 vs 139), but all models are 6–12× worse at 1 step than at 9.
- Offline action MSE is not task success or damage rate; closed-loop eval needs
  the OopsieVerse simulator.
