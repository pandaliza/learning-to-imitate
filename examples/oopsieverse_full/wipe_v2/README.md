# wipe_countertop v2 screening: evaluation side

Spec: `/edrive1/lizad/oopsieverse_full_study/reports/wipe_v2/wipe_v2_spec.md` (Amendment 8). Data, logs and status live
under `/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/`. Reports live under `.../reports/wipe_v2/`.

| File | Role |
|---|---|
| `eval_wipe_v2.py` | Evaluator wrapper. It uses the frozen `run_episode` with the counter-label alias, the frozen pools and multi-GPU rendering. Policies load through `load_policy(run_dir, device, ckpt)`; the header documents the interface. |
| `wv2_common.py` | Paths, layout groups, alias LUT, pool loading (hash-checked). |
| `build_pools.py` | Builds, verifies and freezes `runs/wipe_v2/pools/pools_v1.json` (already frozen; refuses to overwrite it). |
| `verify_alias.py` | Pixel-level check of the alias against training frames and playback reloads. |
| `summarize_wipe_v2.py` | Per-pool tables, Wilson CIs, pass criterion; `--calibration` writes `reports/wipe_v2/calibration_w0.md`. |
| `run_calibration.sh` | The W0 and other-arm calibration jobs. |

## Evaluate a new run-dir on the pools

```bash
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh   # sets $PY, TMPDIR, EGL
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
RUN=/edrive1/lizad/oopsieverse_full_study/runs/<your_runs>/<run_id>      # holds meta.json + ckpt/step_30000.pth
nohup nice -n 10 $PY eval_wipe_v2.py --run-dir $RUN --pools P-train,P-devlayout --rollout-seeds 0,1 \
    --ckpt step_30000.pth --max-workers 32 --label <run_id> \
    > /edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs/<run_id>.eval.log 2>&1 &
# progress:  cat /edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/_status/<run_id>.json
# stop:      touch /edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/eval/STOP   (then rm it; re-run the same command to resume)
$PY summarize_wipe_v2.py --run-dir $RUN          # tables + pass criterion vs W0 s0 P-train
```

Notes:
- **Screening pools.** The screening pools are P-train (60 resets × 2 seeds = 120 episodes) and P-devlayout (40 × 2 = 80). Add `P-random` only for the post-pass test phase. The spec allows that phase only under a dated amendment, together with the test-demo layouts.
- **Several runs at once.** Repeat `--run-dir` to evaluate several runs in one job. Each run gets its own inference server, and the runs share the render workers.
- **GPUs.** Rendering uses the parity-passed GPUs from `selection/gpu_budget.parity_passed_gpus()` (currently 3/4/6/7) and keeps at least 8 GB free on each one. Inference picks the GPU with the most free memory; override it with `--infer-gpu N`. `--max-workers` is the CPU cap and must be ≤ 40. It is shared across concurrent jobs only through the GPU ledger, so split it yourself when you run jobs side by side.
- **Records.** Records go to `runs/wipe_v2/eval/<run_id>/<pool>__<ckpt>__<evalkey>/<reset>__r<seed>.json`. The eval key hashes the evaluator code, `v2_registry.py`, the pools hash, the run key and the checkpoint sha256. Changing any of these starts a fresh directory.
- **Custom architectures (W1/W2/...).** Create `wipe_v2/v2_registry.py`:

  ```python
  from eval_wipe_v2 import register

  @register("w1")                      # == meta.json["arm"]
  def load_w1(run_dir, device, ckpt):
      ...                              # build model, load ckpt, .to(device).eval()
      return obj                       # .seg_keys, .cfg.num_inference_steps, .intent_nfe, .decode(seg, prop, x0, z)
  ```

  `decode` must return **denormalised** actions `[B, 8, 12]` (and intents or None). Intent-conditioned arms (W3/W4) that
  follow the I-PATH convention need no registry entry. They only need `meta["intent"] = {"mode": "condition", "n_tok", "tok_dim", "intent_nfe", ...}`
  and an `IntentCFMPolicy`-compatible checkpoint. Keep `meta["policy_config"]` loadable by the official `PolicyConfig` and put extra fields in `meta["v2"]`.
- **Smoke test.** For a first check of a new arm, add `--limit 1 --pools P-devlayout --rollout-seeds 0`. That runs 1 episode, and its record is a valid one.

## Rollout-time intent ablation (spec "Secondary analysis")

`eval_wipe_v2_ablate.py` takes the same CLI as `eval_wipe_v2.py`, plus `--intent-ablation {sampled,shuffled,zero}`. It is a separate entry point because `eval_wipe_v2.py` hashes itself into every eval key, so editing it would re-key all existing runs.
- `sampled` is the default. It calls `eval_wipe_v2.main()` unchanged, so the keys and directories are those of the main runs and no extra episodes are run.
- `shuffled` and `zero` work only for condition-mode intent arms. The ablation mode is part of the key, and records go to `runs/wipe_v2/ablation/<run_id>/`. Every record carries `intent_ablation` and `intent_ablation_provenance`.
- `zero` uses the training-mean token, computed from the training data and cached in `runs/wipe_v2/ablation/`.

```bash
nohup nice -n 10 $PY eval_wipe_v2_ablate.py --intent-ablation shuffled --run-dir $RUN --pools P-devlayout \
    --rollout-seeds 0,1 --max-workers 16 --label <run_id>__shuffled > $LOG 2>&1 &
```

## P-demo pool and execution-horizon sweep (spec addenda, 2026-10-26)

**P-demo.** `build_pool_demo.py build|freeze` built the frozen `runs/wipe_v2/pools/pools_demo_v1.json`, which holds the exact starts of the 40 training demos.
- The reset is `apply_reset` (ep_meta + model_file + states[0]) followed by an observable refresh, because robosuite returns cached observables after `set_state`.
- Evaluate with `eval_wipe_v2_demo.py`. It takes the same CLI as `eval_wipe_v2.py`, accepts only `--pools P-demo`, and needs no edits to the hashed evaluator files.
- Report: `summarize_wipe_v2.py --demo-starts`, which writes `reports/wipe_v2/demo_starts.md`.

**Execution horizon.** `eval_wipe_v2_horizon.py --exec-horizon H` takes the same CLI, any pool including P-demo, and H from 1 to 8.
- H is part of the key, and records go to `runs/wipe_v2/horizon/<run_id>/`.
- Records add `exec_horizon` and the smoothness fields.
- `run_horizon.sh` runs the sweep. Report: `summarize_wipe_v2.py --exec-horizon`, which writes `reports/wipe_v2/exec_horizon.md`.

## Matched-condition baseline check (spec addendum, 2026-10-27)

**Pools.** `build_pools_all60.py` builds, verifies and freezes the two pools under `runs/wipe_v2/pools/`, each with a `.sha256` file and mode 0444.
- `pools_demo60_v1.json` (P-demo60) holds the exact starts of all 60 wipe demos. The reset is the P-demo restore plus the observable refresh, and every entry carries its `layout_group` over g00..g06.
  - Commands: `demo60-build` (resumable, with a shuffled second pass), then `demo60-freeze`.
- `pools_layout60_v1.json` (P-layout60) holds sticky-ep_meta resets, 20 per layout across all 7 layouts (140 in total).
  - Commands: `layout60-build`, `layout60-verify` (a fresh process, shuffled re-reset), then `layout60-freeze`.

**Evaluate.** `eval_wipe_v2_all60.py` takes the same CLI as `eval_wipe_v2.py` and accepts `--pools` from P-demo60, P-layout60 and P-random.
- The alias is on, the timeout is 1250 steps, and the pool-file hashes plus this entry point's sha are part of the eval key.
- Records go to `runs/wipe_v2/all60/eval/<run_id>/<pool>__<ckpt>__<key>/`.
```
source examples/oopsieverse_full/baseline/envsetup.sh; cd examples/oopsieverse_full/wipe_v2
RUN=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/all60/train/amix_wipe_countertop_all60_s0
nohup nice -n 10 $PY eval_wipe_v2_all60.py --run-dir $RUN --pools P-demo60,P-layout60,P-random \
    --rollout-seeds 0,1,2 --max-workers 32 --total-workers 36 --label all60_s0 \
    > /edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs/all60_s0.log 2>&1 &
```
- Summarize with `summarize_wipe_v2.py --all60 --run-dir RUN [--run-dir ...] --out-md reports/wipe_v2/<name>.md`.
