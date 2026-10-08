# Oops workshop: spatial-robustness runbook (cluster)

Spec: `docs/oops_workshop_robustness_spec.md`. Code: `examples/openpi/oops/`. Launchers: `slurm-scripts/oops/`.
Comparison: **A = M0** `pi05_base_nointent/base_nointent_goal_run1/29999` vs
**I = M1** `pi05_base_intent/base_intent_goal_run1/29999` + ResNet slot-intent sidecar
(`libero-goal-suite-image/slot_intent/models/model_best.pt`). M3 (frozen-VL intent) is optional/secondary.

All commands run from the repo root of the cluster checkout (`/home/ldahiya/max_vla/much-ado-about-noising`)
after syncing this branch's `examples/openpi/oops/`, `examples/openpi/eval_libero_intent.py`, `slurm-scripts/oops/`.
Outputs go to `$OOPS=/data/user_data/ldahiya/oops_runs` (override with `OOPS=...`).

| # | step | command | cost | gate |
|---|---|---|---|---|
| 0 | tests | `pytest examples/openpi/oops/oops_test.py` | s | must pass |
| 1 | inventory + historical audit | `python examples/openpi/oops/audit_existing_runs.py --out-dir docs/oops_workshop/audit --logs-dir logs` | min | both primary checkpoints + sidecar exist; same base (pi05_base), same budget |
| 2 | **freeze** validity manifest (policy-free) | `sbatch slurm-scripts/oops/oops_00_manifest.sbatch` | ~20-40 CPU-min | refuses to overwrite; never rebuilt after outcomes are seen |
| 2b | review manifest | `python -c` / open `$OOPS/manifest_pv1/cases.jsonl`; compare counts with the local provisional manifest (docs/oops_workshop/manifest_local_provisional/) | min | invalid reasons look physical on a few cases |
| 3 | native parity, both arms | `ARM=M0 sbatch slurm-scripts/oops/oops_10_parity.sbatch`; `ARM=M1 sbatch ...` | 2x ~10 GPU-min | `PARITY: PASS` in both logs; any mismatch blocks step 5 |
| 4 | sanity batch + throughput | `MAX_CASES=2 sbatch --array=0-1 slurm-scripts/oops/oops_20_eval.sbatch` | ~10 GPU-min | records appear in `$OOPS/episodes_pv1/{M0,M1}/`; note `wall_s` per episode |
| 5 | campaign (resumable) | `sbatch slurm-scripts/oops/oops_20_eval.sbatch` (16 jobs = 8 tasks x 2 arms, interleaved) | 504 cases/arm, est. 15-30 s/episode -> ~2-4 GPU-h/arm | stop submitting new jobs at **01:00** EDT |
| 6 | provisional analysis (any time) | `PROVISIONAL=1 bash slurm-scripts/oops/oops_30_analyze.sh` | s | figures/tables regenerate from records |
| 7 | freeze evidence | `bash slurm-scripts/oops/oops_30_analyze.sh` at **02:00** EDT | s | copy `docs/oops_workshop/results/` into the paper |

Resume: re-submit the same array; completed `(case_id, arm, checkpoint_hash, evaluator_hash, policy_seed)` keys are
skipped. Editing `eval_spatial_shifts.py`/`oops_common.py` changes `evaluator_hash` -> nothing is reused (by design);
the analysis refuses to mix identities within an arm.

If the queue is slow: the array order is task-major with arms interleaved; incomplete tasks drop out of the fixed
macro task set automatically (listed in `summary.json:tasks_dropped_from_macro`). Report them as not run; do not
re-order to favour particular tasks.

Optional secondary arm (only after M0/M1 complete): `ARMS="M0 M3" sbatch --array=1,3,5,7,9,11,13,15 ...`
(odd indices = the second arm; M0 records are reused because their keys match) then
`INTENT_ARM=M3 OUT=docs/oops_workshop/results_m3 bash slurm-scripts/oops/oops_30_analyze.sh`.

## Local substitute study: robomimic square-mh-state (what was actually run, 2026-10-05)

The pi0.5 checkpoints are cluster-only, so the submission's rollout evidence comes from the local matched MIP pair
(baseline vs flow-intent). See `docs/oops_workshop/square/methods_facts.md`. CPU only, from the repo root:

```bash
.venv/bin/python -m pytest -q examples/openpi/oops/oops_test.py
# 1. freeze the physical-case manifest (policy-free; ~10 min)
.venv/bin/python examples/openpi/oops/square_build_manifest.py --out-dir oops_runs/square_manifest_pv1 --n-resets 50
# 2. parity: bitwise trace + statistical cohort (delta=0, adapter vs untouched native reset)
.venv/bin/python examples/openpi/oops/square_parity.py --manifest-dir oops_runs/square_manifest_pv1 --seeds 0,1 --out oops_runs/square_parity.json
.venv/bin/python examples/openpi/oops/square_eval.py --manifest-dir oops_runs/square_manifest_pv1 --out-dir oops_runs/square_parity_pv1 --workers 24 --deltas 0.0 --policy-seeds 0,1,2
.venv/bin/python examples/openpi/oops/square_eval.py --manifest-dir oops_runs/square_manifest_pv1 --out-dir oops_runs/square_parity_pv1 --workers 24 --deltas 0.0 --policy-seeds 0,1,2 --native-reset
# 3. campaign: 1,550 cases x 2 arms (~30-40 min on 24 workers; resumable)
.venv/bin/python examples/openpi/oops/square_eval.py --manifest-dir oops_runs/square_manifest_pv1 --out-dir oops_runs/square_episodes_pv1 --workers 24
# 4. analysis (all paper numbers/figures)
.venv/bin/python examples/openpi/oops/analyze_robustness.py --manifest-dir oops_runs/square_manifest_pv1 \
    --episodes-dir oops_runs/square_episodes_pv1 --baseline-arm A_baseline --intent-arm I_flow_intent \
    --baseline-label "Baseline flow policy" --intent-label "Flow-intent policy" --training-seeds "1 (seed 0)" \
    --deltas 0.0,0.02,0.05,0.10 --out-dir docs/oops_workshop/square/results
.venv/bin/python examples/openpi/oops/square_breakdowns.py --manifest-dir oops_runs/square_manifest_pv1 \
    --episodes-dir oops_runs/square_episodes_pv1 --out-dir docs/oops_workshop/square/results
```
# 5. secondary: policy seeds 1-2 (both arms), then pooled 3-seed analysis
.venv/bin/python examples/openpi/oops/square_eval.py --manifest-dir oops_runs/square_manifest_pv1 --out-dir oops_runs/square_episodes_pv1_seeds12 --workers 24 --policy-seeds 1,2
# concatenate per-arm JSONL of square_episodes_pv1 + _seeds12 into square_episodes_pv1_allseeds/<arm>/, then rerun
# analyze_robustness.py / square_breakdowns.py with --out-dir docs/oops_workshop/square/results_3seeds
