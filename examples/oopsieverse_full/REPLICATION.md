# OopsieVerse intent-safety study: replication guide (for an autonomous agent)

Audience: a coding agent on a different machine with a fresh checkout of this repo and no access to the original
host. Every command below comes from the code in `examples/oopsieverse_full/`. Items that could not be verified from
files are marked **UNVERIFIED**. Snapshot: 2026-10-30.

Two placeholders are used throughout:

| Placeholder | Original host path | What lives there |
|---|---|---|
| `$REPO` | `/scr/lizad/personal/learning-to-imitate` | this repo; `outputs/oopsieverse_full_study/` holds frozen artifacts and reports (gitignored) |
| `$OVS` | `/edrive1/lizad/oopsieverse_full_study` | heavy storage: envs, pinned sources, assets, data, runs, reports |

---

## 0. Read this first: three blockers

1. **Absolute paths are hardcoded.** About 100 files under `examples/oopsieverse_full/` hardcode
   `/edrive1/lizad/oopsieverse_full_study` (120 occurrences) and `/scr/lizad/personal/learning-to-imitate` (79).
   Use option A unless you have no root access.
   - **Option A (recommended): recreate the paths with symlinks.**
     ```bash
     sudo mkdir -p /edrive1/lizad /scr/lizad/personal
     sudo ln -s $OVS  /edrive1/lizad/oopsieverse_full_study
     sudo ln -s $REPO /scr/lizad/personal/learning-to-imitate
     ```
   - **Option B: rewrite the paths on a branch.** This changes the code hashes in run keys, which is harmless for a
     fresh replication.
     ```bash
     grep -rlE "/edrive1/lizad/oopsieverse_full_study|/scr/lizad/personal/learning-to-imitate" examples/oopsieverse_full \
       | xargs sed -i "s#/edrive1/lizad/oopsieverse_full_study#$OVS#g; s#/scr/lizad/personal/learning-to-imitate#$REPO#g"
     ```
   - Two files also reference `/scr/lizad/miniconda3/bin/ffmpeg`. It is used only for videos; see pitfall P6.
2. **Frozen artifacts are not in git.** `outputs/` is gitignored and `$OVS` is local storage. Ask the original user for
   the frozen bundle (§4.1) and verify its hashes. Regenerating instead is possible but may not be bit-identical
   (**UNVERIFIED**).
3. **There are no official checkpoints or evaluator.** The pinned oopsieverse source has neither. Every policy here is
   trained from scratch, and the evaluator is ours (`baseline/rollout_eval.py`, metric contract v1.0).

---

## 1. What this reproduces

Rates are task success / primary safe success, in percent. **Primary safe** means task success AND every tracked
entity's final health > 95.

| # | Experiment | Headline numbers (original host) | Report (original path) |
|---|---|---|---|
| E1 | pick_egg 8-arm matrix, 40 train demos, 3 training seeds, 200 test resets × 3 rollout seeds | A-MIX 65.8 / 46.7; I-MEAN 86.2 / 59.9; I-PATH 80.4 / 57.1; I-PATH-G 56.1 / 39.6; AUX-PATH-G 60.9 / 48.1; A-SAFE 24.2 / 21.9; safe-sized mix 57.7 / 45.6; canonical mean 69.4 / 33.8. Holm p vs A-MIX = 0.34 for I-MEAN and I-PATH | `$REPO/outputs/oopsieverse_full_study/matrix/comparison/pick_egg_primary_comparison.md` |
| E2 | wipe_countertop 8-arm matrix, same design, frozen evaluator (counter-label defect, no alias) | every arm 0–14% task (cmean s2 13.7 is the maximum; A-MIX 0–1%) | `$REPO/outputs/.../matrix/eval/per_run_test_tables.md` |
| E3 | wipe v2 calibration (W0 = A-MIX ×3, other arms s0) on P-train / P-devlayout, alias on | pooled W0 18.6 / 2.1 task; I-PATH s0 29.2 / 8.8 | `$OVS/reports/wipe_v2/calibration_w0.md` |
| E4 | wipe Tier 1: W1 spatial-softmax, W2 8×8 tokens, W3 dirt-target intent (seed 0) | P-train / P-devlayout task: W1 27.5 / 1.2; W2 24.2 / 0.0; W3 35.8 / 6.2. All FAIL the bar (P-devlayout ≥30% with Wilson LB >10%) | `$OVS/reports/wipe_v2/tier1_summary.md`, `screen_w{1,2,3}.md` |
| E5 | wipe intent ablation, 40-demo, P-devlayout | I-PATH s0: sampled 8.8 → shuffled 3.8 → zero 0.0. W3: 6.2 → 3.8 → 5.0 | `$OVS/reports/wipe_v2/ablation_intent.md` |
| E6 | wipe P-demo: exact starts of the 40 train demos, 3 rollout seeds | A-MIX s0/s1/s2 13.3 / 45.0 / 20.0; I-PATH s0 37.5; human open-loop replay 57.5 (23/40) | `$OVS/reports/wipe_v2/demo_starts.md` |
| E7 | wipe execution-horizon sweep h ∈ {8,4,1} | P-demo: A-MIX s1 45.0 / 22.5 / 12.5; I-PATH s0 32.5 / 8.8 / 18.8 | `$OVS/reports/wipe_v2/exec_horizon.md` |
| E8 | wipe all-60 matched check: A-MIX and I-PATH on all 60 demos, 3 seeds × 3 rollout seeds | task mean ± sd over seeds. P-demo60: A-MIX 27.6 ± 15.8, I-PATH 48.5 ± 6.6. P-layout60: 21.4 ± 9.2 vs 29.8 ± 1.0. P-random: 14.7 ± 14.5 vs 22.2 ± 3.7 | `$OVS/reports/wipe_v2/all60_eval.md`; protocol Amendment 11 |
| E9 | human open-loop replay on P-demo60 | 33/60 = 55.0% task, 10 primary safe | `$OVS/reports/wipe_v2/replay_demo60.md` |
| E10 | all-60 I-PATH intent ablation on P-layout60 (**still finishing** at snapshot) | sampled 29–31% → shuffled 0–0.8% → zero 0% | `$OVS/reports/wipe_v2/all60_ablation.md` (written on completion) |
| E11 | B1K competence gates, 40-demo A-MIX s0, 30 dev resets | add_firewood 0 successes (26 of 30 scored when the verdict became fixed); shelve_item 1/24 → FAIL | `$OVS/reports/b1k/gate_verdicts_b1k.md`, `b1k_results.md` |
| E12 | B1K add_firewood all-60 gate, 3 seeds (**in progress** at snapshot) | 0, 0 and 1 success in the first ~27 episodes per seed → FAIL on all three | `$OVS/reports/b1k/all60_add_firewood.md` (written on completion) |

**Reproduction tolerance.**
- The batched inference server's actions depend on batch composition (up to 6e-5). Records therefore reproduce
  **in distribution, not bit-exactly per episode**.
- Compare rates with Wilson 95% intervals; the summarisers print them.
- A replication matches if each headline rate falls inside the original Wilson interval, or the two intervals
  overlap substantially.
- Expect training-seed variance to dominate. A-MIX swings 32–68% safe on pick_egg and 11–42% task on wipe across
  seeds.

---

## 2. Hardware and system requirements

| Item | Original host | Minimum to replicate |
|---|---|---|
| GPU | 8× NVIDIA H200 (no RT cores), driver 570.133.20, CUDA 12.8 | Any CUDA 12.x GPU with ≥ 24 GB per training job. Training peaks at ~9–12 GB/job (W2 ~40 GB). Isaac needs an RTX-capable driver stack; it ran on H200 without RT cores |
| CPU | 128 cores, shared | ≥ 32 cores. RoboCasa eval workers are single-threaded MuJoCo processes; we used 24–36 per evaluation job |
| OS | Ubuntu 24.04.2, glibc 2.39 | Linux with EGL (glvnd) for headless MuJoCo rendering |
| Disk | RoboCasa env 7.7 GB; sources + assets 25 GB; caches 8 GB; B1K ~74 GB | ≥ 250 GB free on one large disk for envs, assets, data, runs and logs |

**EGL render parity is host-specific.**
- On the original host, MuJoCo EGL frames were corrupt on GPUs 0 and 2 and correct on GPUs 1 and 3–7.
- Policies consume pixel-exact segmentation, so a corrupt GPU silently changes results.
- Run the parity check on every GPU you intend to render on, and render only on GPUs that PASS:
  ```bash
  source $REPO/examples/oopsieverse_full/baseline/envsetup.sh
  cd $REPO/examples/oopsieverse_full/selection
  for g in 0 1 2 3 4 5 6 7; do
    MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES= $PY render_parity_sel.py egl_gpu${g}_run1
    MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES= CTXFIX=1 $PY render_parity_sel.py egl_gpu${g}_ctxfix_run1
  done
  ```
- PASS means step-path segmentation is pixel-identical to official playback frames (`step_min_seg_equal_frac == 1.0`).
- `selection/gpu_budget.parity_passed_gpus()` reads these JSONs, and the wipe_v2 evaluators use it by default
  (`--render-gpus auto`).
- The reference run dir it compares against defaults to `$OVS/runs/baseline/train/amix_pick_egg_v1_s0`, so train E1
  A-MIX s0 first (§5.1).

---

## 3. Environment setup

### 3.1 RoboCasa / DamageSim study env (everything except B1K)

These are the pinned sources, from `outputs/oopsieverse_full_study/env/env_manifest.json`:

| Source | Remote | Commit | Local modification |
|---|---|---|---|
| oopsieverse | https://github.com/UT-Austin-RobIn/oopsieverse | `a739f99e94f32c3229ae6932299e2e0e8c065480` | none |
| robosuite | https://github.com/ARISE-Initiative/robosuite | `aaa8b9b214ce8e77e82926d677b4d61d55e577ab` (v1.5.2-5) | none |
| robocasa | https://github.com/robocasa/robocasa | `97a40605f4f4e9df8c70ca3a56fe3288b5bd3dc7` (v1.0-6) | `setup.py`: numba pin `==0.61.2` → `>=0.61.2` |

```bash
export OVS=/path/to/big/disk/oopsieverse_full_study; mkdir -p $OVS/{src,env,cache,data,runs,reports}
cd $OVS/src
git clone https://github.com/UT-Austin-RobIn/oopsieverse && git -C oopsieverse checkout a739f99e94f32c3229ae6932299e2e0e8c065480
git clone https://github.com/ARISE-Initiative/robosuite && git -C robosuite checkout aaa8b9b214ce8e77e82926d677b4d61d55e577ab
git clone https://github.com/robocasa/robocasa   && git -C robocasa  checkout 97a40605f4f4e9df8c70ca3a56fe3288b5bd3dc7
sed -i 's/numba==0.61.2/numba>=0.61.2/' robocasa/setup.py           # = oopsieverse install.py patch_robocasa_numba_pin
mkdir -p oopsieverse/externals && ln -s $OVS/src/robosuite oopsieverse/externals/robosuite \
                                && ln -s $OVS/src/robocasa  oopsieverse/externals/robocasa   # layout install.py creates
uv venv $OVS/env/oopsie --python 3.10
source $OVS/env/oopsie/bin/activate
uv pip install -e robosuite && uv pip install -e robocasa && uv pip install -e oopsieverse   # WITH deps (official)
python robocasa/robocasa/scripts/download_kitchen_assets.py      # all types; ~23 GB
python robocasa/robocasa/scripts/setup_macros.py
```

- The exact `sed` pattern and the download script paths are **UNVERIFIED**. The manifest records the intent and
  cites `install.py:79-87, 167-173`. Follow `oopsieverse/install.py` if they differ.
- The asset links come from robocasa @ 97a4060 `box_links_assets.json` (sha256 `918335d0…f7c1d`).
- Expected package versions: mujoco 3.3.1, numpy 2.2.5, scipy 1.15.3, numba 0.62.1, torch 2.7.1+cu126 (pulled in by
  lerobot 0.3.3), h5py 3.16.0, robosuite 1.5.2, robocasa 1.0.0.
- The full lock is in `outputs/oopsieverse_full_study/env/uv_lock.txt`, part of the frozen bundle.

**`baseline/envsetup.sh`** hardcodes `OVS`. With option A from §0 it works as is. Its content:
```bash
export OVS=$OVS PY=$OVS/env/oopsie/bin/python TMPDIR=$OVS/cache/tmp HF_HOME=$OVS/cache/hf \
  XDG_CACHE_HOME=$OVS/cache/xdg TORCH_HOME=$OVS/cache/torch MPLCONFIGDIR=$OVS/cache/mpl NUMBA_CACHE_DIR=$OVS/cache/numba
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d PYNPUT_BACKEND=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```
`__EGL_VENDOR_LIBRARY_DIRS` matters. A conda shell default points at a Mesa-only vendor directory, which gives
software rendering and non-parity frames.

**Official sources stay unmodified.** Our deviations are applied as wrappers, never by editing the pinned sources.
They are recorded as diffs for reference:
- `examples/oopsieverse_full/playback_portability.diff`: path-only remap of the collectors' asset paths
  (`path_remap.py`, `playback_robocasa_portable.py`).
- `examples/oopsieverse_full/baseline/playback_gripper_headless.diff`: the collection-time continuous gripper patch
  plus `has_renderer=False`.

### 3.2 BEHAVIOR-1K env (E11, E12 only)

These come from `outputs/oopsieverse_full_study/b1k/install_manifest.json` and `b1k_feasibility_report.md`.

| Component | Version |
|---|---|
| BEHAVIOR-1K | `UT-Austin-RobIn/BEHAVIOR-1K`, branch `proj/safemanibench`, commit `77d38292b775b15f1a1f99b8a3ee215d3f19c42c` (shelve_item was recorded at `77adbd66`, HEAD~2; that diff touches only `envs/data_wrapper.py`, so use HEAD) |
| OmniGibson / bddl | 3.7.1 / 3.7.x, editable |
| Isaac Sim | 4.5.0.0, the 26 wheels listed in BEHAVIOR-1K `setup.sh`, from https://pypi.nvidia.com; sha256 list in `b1k/isaac_wheels.sha256`. Kit 106.5.0.162521 |
| torch | 2.6.0+cu124 |
| Python | 3.10.19 in a uv venv instead of conda |
| Assets | HF `behavior-1k/zipped-datasets`, revision `9f0d57d4…` (full id in `install_manifest.json`): `behavior-1k-assets-3.7.2rc1.zip` (sha256 `3bd7cdc1…5e8b`), `omnigibson-robot-assets.zip`, `2025-challenge-task-instances.zip` (required by `InteractiveTraversableScene`), `omnigibson.key` |
| EULAs | NVIDIA Isaac Sim EULA and BEHAVIOR Data Bundle EULA (non-commercial). The human must accept them; do not accept them on their behalf |

Deviations from `setup.sh`, all required on the original host:
- **pydantic pin.** Pin `pydantic==2.9.2 pydantic_core==2.23.4` after installing open3d. open3d → dash pulls 2.13.x,
  which crashes `omni.services.core` and segfaults Kit.
- **oopsieverse via `sys.path`.** It is used from the pinned source, not pip-installed. Its deps are installed: open3d,
  tyro, numba==0.62.1.
- **Teleop deps skipped.** telemoma, pyspacemouse and mediapipe are not needed for playback.
- **Full lock:** `outputs/oopsieverse_full_study/b1k/requirements.lock.txt`.

Activation script (on the original host it is `$OVS/env_b1k/b1k_env.sh`, which is not in the repo; recreate it):
```bash
export B1K_ROOT=$OVS UV_CACHE_DIR=$OVS/cache_b1k/uv UV_PYTHON_INSTALL_DIR=$OVS/cache_b1k/python PIP_CACHE_DIR=$OVS/cache_b1k/pip
export HF_HOME=$OVS/cache_b1k/hf TMPDIR=$OVS/cache_b1k/tmp XDG_CACHE_HOME=$OVS/cache_b1k/xdg_cache XDG_DATA_HOME=$OVS/cache_b1k/xdg_data
export OMNIGIBSON_DATA_PATH=$OVS/assets_b1k OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export PATH=$OVS/env_b1k/venv/bin:$PATH VIRTUAL_ENV=$OVS/env_b1k/venv
unset EXP_PATH CARB_APP_PATH ISAAC_PATH CONDA_PREFIX PYTHONPATH
export HOME=$OVS/cache_b1k/home      # Kit writes ~/.cache/ov, ~/.local/share/ov, ~/.nvidia-omniverse here
```
The first Kit boot compiles shaders for about 6 minutes; later starts take about 53 s. See pitfall P4 for the GPU
launch scheme.

---

## 4. Data and frozen artifacts

### 4.1 Frozen bundle (ask the original user to produce it)

On the original host, run this to create a ~10 MB bundle:
```bash
cd / && tar czf oopsie_frozen_bundle.tgz \
  scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/{roster,replay/replay_labels.csv,metric_contract.json,protocol_v1.md,env,catalog,b1k} \
  edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/pools \
  edrive1/lizad/oopsieverse_full_study/runs/matrix/_stats \
  edrive1/lizad/oopsieverse_full_study/reports/wipe_v2/wipe_v2_spec.md \
  edrive1/lizad/oopsieverse_full_study/reports/b1k
```
Unpack it so that the files land at `$REPO/outputs/...` and `$OVS/...`. Then verify the hashes below with
`sha256sum`, and **stop if any differs**.

| File | sha256 |
|---|---|
| `outputs/.../roster/splits_v1.json` | `091ef63d769aa967126d93eadde7439e19013f51d7445a9debc7e06755e00398` |
| `outputs/.../roster/roster_v1.json` | `1945bb580c000b4e469e2c73a3f86670f88832a4f6a44086ce599d11d7ede53c` |
| `outputs/.../roster/reset_banks_v1.json` | `6aa75fbf18cec6f2163cdec9daccebe6f33f3560711e9e8d33ffdcd49dd888df` |
| `outputs/.../metric_contract.json` (v1.0) | `51aa0c798a0241be0399f52ac0295a71d7381734e823140f9dab52a974c65a83` |
| `$OVS/runs/wipe_v2/pools/pools_v1.json` (P-train, P-devlayout, P-random pointer) | `8460ecf6f2736a39470399e6ccc34cf02f6e4c17886a4e3d263f4fa8260b3172` |
| `$OVS/runs/wipe_v2/pools/pools_demo_v1.json` (P-demo, 40) | `ac5539ce55c647f5a1a9e698f1eb997fb3e4e8790dfb488de83b65ba45325d63` |
| `$OVS/runs/wipe_v2/pools/pools_demo60_v1.json` (P-demo60) | `4d157196d3499cd9cddd1411701039e70e9519e242b313c0a789b545f5857f7e` |
| `$OVS/runs/wipe_v2/pools/pools_layout60_v1.json` (P-layout60, 140) | `0ea28057123b4aa2edf3063017e156ed714467f0aecb927255649e8478fc9ad5` |
| `$OVS/reports/b1k/reset_bank_b1k_v1.json` (50 test + 30 dev per task) | `f90a7e322ce4c3143641643d5531ef0899c9e538e23c035f6cf73b56e568ec91` |

The pool files also contain content hashes and pointers into the raw teleop HDF5s, which are re-verified at load time
(`wv2_common.py`, `build_pool_demo.load_demo_resets`). A hash mismatch raises an error. Do not bypass it.

**Without a bundle,** regenerate in this order: §4.3 → §4.4 → §4.5. Freezers refuse to overwrite existing files. The
regenerated hashes may differ from the table, because some files carry timestamps (**UNVERIFIED** whether contents
match exactly). Report any such difference.

### 4.2 Raw demonstrations

Source: HF dataset **`ut-robin-lab/oopsieverse-demos`**, revision **`c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3`**
(`catalog/inventory_tasks_and_data.py:33-34`).
```bash
huggingface-cli download ut-robin-lab/oopsieverse-demos --repo-type dataset \
  --revision c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3 --local-dir $OVS/data/oopsieverse-demos \
  --include "paper_experiments/teleop_data/*"
```
- Layout: `paper_experiments/teleop_data/{pick_egg,wipe_countertop,add_firewood,shelve_item}/{with,without}_live_feedback.hdf5`,
  plus `all_data.hdf5`.
- Episode counts: pick_egg 30 + 30, wipe_countertop 30 + 30, add_firewood 30 + 30, shelve_item 45 + 45 (90, not 60).
- Per-file LFS sha256 and size are in `outputs/.../catalog/hf_file_listing.json`. Example: pick_egg
  `with_live_feedback.hdf5` is 68,447,413 bytes, sha256 `cebf3adb…8cd7b0`. Verify every file against it.
- Each RoboCasa demo carries attrs `ep_meta`, `model_file`, `health_list_link_names` and datasets `actions [T,12]`,
  `states`, `obs/*`, `info/*`.

### 4.3 Catalog, replay labels, splits

Skip this section if you have the bundle; it is needed only to regenerate frozen files.
1. `$PY -I catalog/inventory_tasks_and_data.py --src $OVS/src/oopsieverse --data $OVS/data/oopsieverse-demos --out $REPO/outputs/oopsieverse_full_study/catalog`
2. Replay labels: `$PY replay.py --jobs JOBS.json --workers 32 --out OUT.jsonl`, with modes `force` and `open`, then
   `analyze_replay.py`. The exact job list used is **UNVERIFIED**; `outputs/.../replay/replay_sample.json` records it.
3. `$PY freeze_splits.py` → `roster/splits_v1.json`. This is an exhaustive grouped 70/15/15 search, so it is
   deterministic.

Splits v1 per task:

| | pick_egg | wipe_countertop |
|---|---|---|
| train | 40 (20 fb / 20 nofb) | 40 (10 fb / 30 nofb), 3 dirt layouts |
| dev / test | 10 / 10 | 10 / 10 |
| A_SAFE_train | 19 | 5 |
| safe-sized mix | 19 | 5 |

### 4.4 Official playback → training HDF5

```bash
bash $REPO/examples/oopsieverse_full/baseline/run_playback_all.sh 24
```
- This produces one process per demo for pick_egg and wipe, writing to `$OVS/data/playback/parts/<task>__<cond>__demo_<i>.hdf5`.
- Edit `GPUS=(6 7)` in the script to two parity-passing GPUs. It is the official playback plus a path remap,
  collection-time gripper and headless mode.
- Split files (ExternalLinks to the parts) are built on demand by `matrix/train_matrix.py:data_file()` →
  `baseline/split_data.build_split_hdf5` / `matrix/data.build_subset_hdf5`.

### 4.5 Reset banks and pools

- **RoboCasa bank:** `$PY build_reset_banks.py derive`, then `enumerate --workers 32`, then `tests`, then
  `analyze_reset_banks.py` → `roster/reset_banks_v1.json`.
  - Per task: eval_dev 100, eval_test 200, sel_* (selection only) and reserve 300.
  - Seeds come from `KeyedRNG('oopsie_full_v1')` (blake2b), so the bank is deterministic.
- **wipe v2 pools.** Run all of these in `wipe_v2/`, each command in sequence:
  - `$PY build_pools.py build`, `verify`, `freeze`
  - `$PY build_pool_demo.py build`, `freeze`
  - `$PY build_pools_all60.py demo60-build`, `demo60-freeze`, `layout60-build`, `layout60-verify`, `layout60-freeze`
  - Render on one parity GPU: `MUJOCO_EGL_DEVICE_ID=g CUDA_VISIBLE_DEVICES=`.
- **B1K bank:** run `b1k/build_reset_bank_b1k.py TASK POOL SEED0 N OUT_DIR [--determinism K]` per shard, then
  `b1k/freeze_reset_bank_b1k.py`. Test seeds are 100000–100049 and dev seeds 200000–200029.

---

## 5. Experiments, in dependency order

Shared conventions:
- `source $REPO/examples/oopsieverse_full/baseline/envsetup.sh` first.
- Run long jobs under `nohup nice -n 10`, with logs on the big disk.
- Every trainer and evaluator is resumable: rerun the same command, and finished records or checkpoints are skipped.
- Training: 30,000 steps, batch 64, AdamW lr 3e-4, wd 1e-5, clip 5, fp32, no EMA, Beta(1.5,1) flow times, Euler-16
  sampling. These are the official values from `learning/configs/*.yaml`.
- `--workers 4` and `--act-ckpt` are throughput-only and verified bit-identical.

### 5.1 E1 + E2: policy matrix (pick_egg and wipe 40-demo)

Train 8 arms × 2 tasks × 3 seeds = 48 runs. Arms: `amix asafe imean ipath ipathg auxpathg ssmix cmean`.
```bash
cd $REPO/examples/oopsieverse_full/matrix
# one run:
CUDA_VISIBLE_DEVICES=<g> $PY train_matrix.py --arm ipath --task pick_egg --seed 0 --workers 4 --save-every 10000
# or the resumable queue (pick_egg primary arms → wipe primary arms → controls):
nohup $PY queue_runner.py --slots 0:3,1:3 --max-jobs 6 --workers 4 > $OVS/runs/matrix/_queue/runner.log 2>&1 &
```
- Run dirs are `$OVS/runs/matrix/<arm>_<task>_v1_s<seed>/`, holding `meta.json`, `ckpt/step_30000.pth` and `DONE.json`.
- Wall time on the original host: median 4.2 h per run (0.4–14 h under contention).
- On the original host A-MIX pick_egg s0 was adopted from `$OVS/runs/baseline/train/amix_pick_egg_v1_s0`
  (`baseline/train_amix.py`, bit-identical recipe). Running `train_matrix.py --arm amix` is equivalent.

Evaluate each run on 200 eval_test resets × rollout seeds 0, 1, 2 (600 episodes), timeout 650 (pick_egg) or 1250 (wipe):
```bash
$PY rollout_eval_matrix.py --run-dir $OVS/runs/matrix/<run_id> --task pick_egg --resets test --rollout-seeds 0,1,2 \
    --out-dir $OVS/runs/matrix_eval/<run_id>/test_s30000_<run_key> --workers 8 --ckpt step_30000.pth \
    --render-gpu <parity_gpu> --infer-gpu <gpu>
# the daemon that ran this originally: nohup $PY eval_runner.py --workers 8 &   (it first runs a smoke gate on *_smoke500 runs)
$PY summarize_eval.py <eval_dir> <out_json>; $PY summarize_eval.py --index
$PY compare_arms.py --task pick_egg          # → matrix/comparison/pick_egg_primary_comparison.md
```
- Throughput was about 400–700 episodes/h per evaluation job.
- The E2 wipe rows use the frozen evaluator, which has the counter-label defect (P1), so they are 0–14%. Do not
  "fix" this, because it is what the main-study tables measure.

### 5.2 E3–E5, E7: wipe v2 (40-demo checkpoints from E2)

Run in `cd $REPO/examples/oopsieverse_full/wipe_v2`. `wipe_v2/README.md` has full details.
- **E3 calibration:** `bash run_calibration.sh`. That is W0 = `amix_wipe_countertop_v1_s{0,1,2}` on
  P-train, P-devlayout and P-random, plus the other arms at seed 0 on P-train and P-devlayout, with rollout seeds
  0 and 1. Then `$PY summarize_wipe_v2.py --calibration`.
- **E4 Tier 1:**
  1. `$PY extract_markers.py with/demo_10 …` for every wipe demo. It writes per-marker wipe steps for W3 to
     `$OVS/runs/wipe_v2/markers/`.
  2. Train: `CUDA_VISIBLE_DEVICES=<g> $PY train_v2.py --arm {w1,w2,w3} --seed 0 --workers 4 --act-ckpt --save-every 1000`.
     Run dirs: `$OVS/runs/wipe_v2/train/w{1,2,3}_wipe_countertop_v2_s0`. W2 needs about 40 GB and activation
     checkpointing. Alternatively run `nohup $PY queue_v2.py &`.
  3. Evaluate: `$PY eval_wipe_v2.py --run-dir RUN --pools P-train,P-devlayout --rollout-seeds 0,1 --ckpt step_30000.pth --max-workers 32 --label <run_id>`.
  4. Summarise: `$PY summarize_wipe_v2.py --run-dir RUN`. Pass bar (predeclared): P-devlayout ≥ 30% with
     Wilson LB > 10%, and P-train ≥ pooled W0 (18.6%).
- **E5 ablation:** `bash run_ablation.sh` (I-PATH s0 and W3 s0 on P-devlayout, modes shuffled and zero, rollout seeds
  0 and 1). Then `$PY summarize_wipe_v2.py --ablation`.
- **E6 P-demo:** `$PY eval_wipe_v2_demo.py --run-dir RUN … --pools P-demo --rollout-seeds 0,1,2` for A-MIX
  s0/s1/s2 and I-PATH s0. Then `$PY summarize_wipe_v2.py --demo-starts`.
  - Human replay reference: `replay.py` mode `open` on the 40 train demos (`runs/replay/open_teleop/*.npz`).
- **E7 horizon:** `bash run_horizon.sh` (h = 4 then 1; h = 8 reuses existing records). Then
  `$PY summarize_wipe_v2.py --exec-horizon`.

### 5.3 E8–E10: wipe all-60 matched check

```bash
cd $REPO/examples/oopsieverse_full/wipe_v2
# bit-identity check first (200 steps, must equal runs/matrix/amix_wipe_countertop_v1_s0):
$PY train_all60.py --data40 --arm amix --task wipe_countertop --seed 0 --num-steps 200 --label verify \
    --run-id amix_wipe_countertop_data40_s0__verify200 --no-resume --runs-root $OVS/runs/wipe_v2/all60/verify --workers 4 --act-ckpt
# 6 trainings (amix, ipath × seeds 0,1,2), or `nohup $PY queue_all60.py &`:
CUDA_VISIBLE_DEVICES=<g> $PY train_all60.py --arm ipath --task wipe_countertop --seed 0 \
    --run-id ipath_wipe_countertop_all60_s0 --runs-root $OVS/runs/wipe_v2/all60/train --workers 4 --act-ckpt --save-every 1000
bash run_all60_eval.sh            # 3 stages; P-demo60,P-layout60,P-random × rollout seeds 0,1,2; ~3.7 h per stage here
T=$OVS/runs/wipe_v2/all60/train; M=$OVS/runs/matrix
$PY summarize_wipe_v2.py --all60 $(for a in amix ipath; do for s in 0 1 2; do echo --run-dir $T/${a}_wipe_countertop_all60_s$s; done; done) \
    --run-dir $M/amix_wipe_countertop_v1_s1 --run-dir $M/ipath_wipe_countertop_v1_s0 --out-md $OVS/reports/wipe_v2/all60_eval.md
# E9 human replay ceiling (edit MUJOCO_EGL_DEVICE_ID="6" at the top of the file to a parity GPU):
$PY replay_demo60.py run && $PY replay_demo60.py report
# E10 ablation (edit --render-gpus/--infer-gpu 3 in the script to a parity GPU):
bash run_all60_ablation.sh && $PY summarize_all60_ablation.py --out-md $OVS/reports/wipe_v2/all60_ablation.md
```
- Training wall time was ~14.8 h per run on the original host under heavy contention.
- Predeclared reading: A-MIX all-60 ≳ 80% would mean the setup explained the gap; ≲ 50% means wipe is not a
  functioning baseline. The original result was the latter.

### 5.4 E11–E12: BEHAVIOR-1K (diagnostic)

Run in `cd $REPO/examples/oopsieverse_full/b1k`. Image replay runs in the B1K env; training runs in the RoboCasa
study env.
1. Image replay of all demos: `replay_b1k_images.py --with_images …` (official low-resolution sensors). For
   shelve_item, use the Temperature shim (pitfall P3).
2. Labels: `$PY -I build_b1k_labels.py --task T --playback … --timing … --out labels.csv`.
3. Parts: `prepare_b1k_parts.py` with `--shift 0`, the official pairing chosen by the alignment audit.
4. Splits: `freeze_splits_b1k.py`. Reset bank: §4.5.
5. E11, 40-demo:
   - Train: `CUDA_VISIBLE_DEVICES=<g> $PY train_b1k.py --arm amix --task {add_firewood,shelve_item} --seed 0 --workers 4`.
     Wall time was 6,402 s / 7,638 s.
   - Gate: `python eval_b1k.py --run-dir RUN --task T --pool dev --ids 0-29 --rollout-seed 0 --out OUT`, then
     `$PY summarize_b1k.py gate T`.
   - PASS requires ≥ 10 successes and ≥ 5 damaging successes out of 30.
6. E12, all-60 add_firewood:
   - Build the data: `$PY train_b1k_all60.py --build-data-only`.
   - Train with `bash launch_all60.sh <seed> <gpu>` for seeds 0, 1 and 2. It refuses to start unless the GPU has
     16 GB free. Wall time was about 34,200 s each.
   - Gate: `bash run_gate_all60.sh <parity_gpu> <seed>`. The script hardcodes the CVD mapping for GPUs 3, 4 and 6;
     adapt it per P4.
   - Summarise: `$PY summarize_all60.py --seeds 0,1,2` → `reports/b1k/all60_add_firewood.md`.
   - Under load, Isaac evaluation ran at 300–1300 s per episode.

### 5.5 Not required

- `selection/` is the test-state selection study, `wipe_diag/` the earlier wipe diagnosis, `dt_check.py` the dt FIXME
  check. None of them feed the headline numbers above.
- `wipe_v2/render_rollout_video.py` and the demo renders are illustrations only.

---

## 6. Known pitfalls

| ID | Symptom | Cause | Fix used in the study |
|---|---|---|---|
| P1 | wipe policies see the counter as "unknown" at evaluation | A plain reset names the counter `Counter`, while training seg (playback) names it `DamageableCounter`; the substring vocabulary misses it (protocol Amendment 7.2) | All wipe_v2 evaluators apply `wv2_common.alias_seg_lut`. The E2 frozen evaluator does NOT, and that is intentional for the main-study tables |
| P2 | First policy frame shows the previous episode; EEF off by up to 0.9 m | robosuite returns cached observables after `set_state`. Also, `DamageableWipeCounter.reset` returns pre-IK obs | After any restore: `env._update_observables(force=True); env.get_observations()`. Built into `build_pool_demo.restore_and_check`, the all-60 evaluators and `rollout_eval` |
| P3 | shelve_item replay: "Unable to find AG object" every step; wrong grasp | Recorded states are 3191 floats, runtime is 3192 (damagesim adds a robot `Temperature`) | Wrapper shim inserts `Temperature = 23.0` before the robot AG block, with hard size checks (`replay_b1k_shim.diff`, `shelve_shim_validation.md`) |
| P4 | Isaac: "No device could be created / GPUs do not support RayTracing", rc 139, DEVICE_LOST | `CUDA_VISIBLE_DEVICES` set naively, or a bad GPU | Never set CVD to a single GPU. Select with `OMNIGIBSON_GPU_ID=g` and Kit args `--/renderer/activeGpu=g --/physics/cudaDevice=g`, multiGpu off. When CVD is needed, hide only the forbidden GPUs and keep g at its own index (original: g=3 → `1,2,4,3`; g=4 → `1,2,3,5,4`; g=6 → `1,2,3,4,5,7,6`). Probe every GPU first (`probe_isaac.py`, `guarded_probe*.sh`). Isaac holds ~830 MiB on every visible GPU |
| P5 | Kit segfault at startup in `omni.services.core` | pydantic 2.13 pulled in by open3d → dash | `pydantic==2.9.2 pydantic_core==2.23.4` |
| P6 | ffmpeg "Unknown encoder libx264" | conda ffmpeg built without x264 | Use the `imageio-ffmpeg` bundled binary (`imageio_ffmpeg.write_frames`) |
| P7 | Supervisors or status writes crash with ENOSPC | small system disk filled | Put every log, status, record and cache on the big disk. Set `TMPDIR` (envsetup.sh does) |
| P8 | Eval supervisor dies with an nvidia-smi timeout under load | GPU budget polling calls nvidia-smi | Rerun the same command; it resumes from records. The original zero-mode ablation was relaunched once |
| P9 | Same (reset, seed) gives a slightly different episode | batched inference round-off (≤ 6e-5); batch-1 CPU video renders differ too | Compare distributions, not episodes |
| P10 | `empty PNG file` during wipe resets | robosuite writes dirt textures to a fixed `/tmp/robosuite_temp_tex/`, and concurrent processes race on it | Treat it as a retryable infrastructure error, never as a policy failure; the evaluators already retry |
| P11 | Corrupt or software-rendered frames, wrong segmentation | bad EGL GPU, or the Mesa vendor dir | Parity test (§2) and `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d` |
| P12 | `REFUSE` or run-key mismatch on resume | code, data or config changed under an existing run dir | Use a fresh run dir. Never delete a `DONE.json` to force a retrain into an old key |

---

## 7. Rules the original study followed (do not "fix" these)

- **Frozen artifacts are never edited:** splits, roster, metric contract, reset banks, pools and labels. Every
  builder refuses to overwrite. New conditions get new files and a dated amendment.
- **Official sources stay at the pinned commits, unmodified.** Deviations are wrappers or shims, recorded as diffs.
- **Evaluator files are hashed into eval keys.** New behaviour goes in a new sibling entry point (the
  `eval_wipe_v2_*.py` pattern) so existing keys stay valid.
- **Pass criteria and readings were declared before results.** See `protocol_v1.md`, `wipe_v2_spec.md` and the
  `b1k_diag_protocol.md` addenda. Do not relax the bars: W1–W3 fail the 30% unseen bar, and the B1K gates fail the
  10/30 bar.
- **Each primary arm is a single change from A-MIX,** with the same data, recipe and seeds. A-MIX checkpoints are
  official training math bit-for-bit (`matrix/arm_verification.md`).
- **The wipe test-demo layouts (g04/g05) and the eval_test bank were never used for screening.** Only the all-60
  check uses g04 and g05.

---

## 8. Checklist and what to report back

1. [ ] Paths resolved (§0 option A or B). Frozen bundle unpacked and **all §4.1 hashes match**.
2. [ ] RoboCasa env built at the pinned commits; package versions as in §3.1.
3. [ ] Raw data downloaded at the pinned revision; per-file LFS sha256 matches `hf_file_listing.json`.
4. [ ] Render parity run on every GPU; the PASS list recorded.
5. [ ] Playback parts built: 120 RoboCasa demos (2 tasks × 2 conditions × 30), none missing a `.done` marker.
6. [ ] E1/E2: 48 runs trained and evaluated (600 episodes each); `compare_arms.py --task pick_egg` table produced.
7. [ ] E3–E7 wipe v2 tables produced.
8. [ ] E8–E10 all-60 tables produced; the bit-identity check passed.
9. [ ] (Optional) B1K env built; E11/E12 gates run.

Report back:
- **Environment:** GPU model, driver, parity PASS list, any version deviations.
- **Hashes:** any frozen-artifact or data hash mismatch, and what you did about it.
- **Results:** for each experiment E1–E12, your table next to the original numbers in §1, with Wilson intervals, the
  training wall time and the episode count. State whether each headline rate falls inside the original interval.
- **Runs:** the number of infrastructure errors per run, kept separate from policy failures.
- **Deviations:** every command you changed, and why.
