"""Matched spatial-shift rollouts for ONE arm over a frozen physical-case manifest.

Minimal extension of the native evaluator (examples/openpi/eval_libero_intent.py): same env class, resolution,
10-step dummy settle, 300-step cap, replan-every-5, observation construction, intent sidecars and success
semantics (episode ends at the first step the env reports success; success = "ever achieved").
What it adds, and nothing else:
  - initial states come from build_case_manifest.py (validity decided before any policy ran); the loaded
    state's hash is checked against the manifest, and delta=0 states against the benchmark init file
  - per-episode deterministic seeding shared by both arms (oops_common.episode_seed)
  - one durable JSONL record per episode, resume, one deterministic retry of infrastructure errors
  - non-finite policy output is recorded as a policy failure (not an infrastructure error)
  - post-settle object position is compared with the manifest (flags a physics-stack mismatch)
  - outcome-independent video rule; optional per-replan action dumps for the native parity check

Arms (flags exactly as the native SR launchers used them, slurm-scripts/train/pi05/eval_pi05_base_ab.sbatch,
eval_pi05_m3_sr.sbatch) are selected with --preset; any flag can be overridden.

  python examples/openpi/oops/eval_spatial_shifts.py --preset M0 \
      --manifest-dir oops_runs/manifest_pv1 --out-dir oops_runs/episodes
"""

import argparse
import collections
import functools
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import oops_common as oc  # noqa: E402

B = "/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi"
PRESETS = {
    # baseline: no intent channel
    "M0": dict(config_name="pi05_base_nointent",
               checkpoint_dir=f"{B}/ldahiya_checkpoints/pi05_base_nointent/base_nointent_goal_run1/29999"),
    # frozen/decoupled ResNet slot-intent sidecar -> adaRMS intent conditioning
    "M1": dict(config_name="pi05_base_intent",
               checkpoint_dir=f"{B}/ldahiya_checkpoints/pi05_base_intent/base_intent_goal_run1/29999",
               intent=True, intent_task_config="libero_goal_suite_image_slot_intent",
               intent_ckpt="/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt"),
    # frozen-VL decoupled generator (secondary)
    "M3": dict(config_name="pi05_base_intent_m3",
               checkpoint_dir=f"{B}/ldahiya_checkpoints/pi05_base_intent_m3/base_intent_m3_goal_run1/29999",
               vl_cotrain=True, intent_task_config="libero_goal_suite_image_slot_intent_vl",
               intent_stack=f"{B}/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt",
               frozen_base_weights=f"{B}/pi05_base_pytorch", frozen_base_config="pi05_base_nointent"),
}


def _quat2axisangle(quat):  # identical to the native evaluator
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize(img_hwc, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR))


def _video_frame(obs):
    agv = np.ascontiguousarray(obs["agentview_image"][::-1])
    wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return np.concatenate([agv, wr], axis=1)


def _save_video(path, frames, fps=20):
    import imageio
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def evaluator_hash():
    h = hashlib.sha256()
    for p in (HERE / "eval_spatial_shifts.py", HERE / "oops_common.py"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_commit():
    try:
        c = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "examples/openpi"],
                               capture_output=True, text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_policy(args):
    """Policy + intent construction copied from eval_libero_intent.py (M0 / M1 sidecar / M3 VL paths)."""
    sys.path.insert(0, str(REPO / "external/openpi/src"))
    import torch
    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    train_config = _config.get_config(args.config_name)
    norm_stats = None
    if args.norm_stats_from_config:
        from openpi.training import checkpoints as _checkpoints
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, norm_stats=norm_stats)
    if args.fp32 and not args.frozen_base_weights:
        policy._model = policy._model.float()

    gen = None
    if args.intent:
        from mip.pi05_intent import IntentGenerator
        assert args.intent_ckpt, "--intent requires --intent-ckpt"
        gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir, device=args.device)

    vl = None
    if args.vl_cotrain:
        import openpi.models.model as _model
        from mip.pi05_intent import CotrainIntentModule
        assert args.intent_stack, "--vl-cotrain requires --intent-stack"
        cot = CotrainIntentModule(args.intent_task_config, args.config_dir, device=args.device, vl_obs=True)
        stack = torch.load(args.intent_stack, map_location=args.device, weights_only=False)
        vlw = stack["vl_proj"]["weight"].shape[1]
        cot.vl_proj(torch.zeros(1, vlw, device=args.device))
        cot.vl_proj.load_state_dict(stack["vl_proj"])
        cot.agent.intent_flow_map.load_state_dict(stack["intent_flow_map"])
        cot.eval()
        vl_model = policy._model
        if args.frozen_base_weights:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
            import safetensors.torch as _stt
            fb_cfg = _config.get_config(args.frozen_base_config)
            vl_model = PI0Pytorch(fb_cfg.model).to(args.device)
            _stt.load_model(vl_model, str(pathlib.Path(args.frozen_base_weights) / "model.safetensors"), strict=False)
            vl_model = vl_model.float().eval()

        def _to_obs(inp):
            def conv(v):
                if isinstance(v, dict):
                    return {k: conv(vv) for k, vv in v.items()}
                t = torch.as_tensor(np.asarray(v), device=args.device)
                return (t.float() if t.is_floating_point() else t)[None]
            return _model.Observation.from_dict(conv(inp))

        def _vl_intent(element):
            el = dict(element)
            el["intent"] = np.zeros(cot.intent_dim, dtype=np.float32)
            obs_t = _to_obs(policy._input_transform(el))
            with torch.no_grad():
                z = cot.vl_sample_intent(vl_model.vl_image_features(obs_t))
            return z[0].detach().cpu().numpy().astype(np.float32)
        vl = _vl_intent
    return policy, gen, vl


def video_selected(rec, task_rank):
    """Outcome-independent: reset 0, delta=0 plus two fixed directions per nonzero delta, rotated by task."""
    if rec["reset_id"] != 0:
        return False
    if rec["delta_m"] == 0.0:
        return True
    return rec["angle_idx"] in (task_rank % 10, (task_rank + 5) % 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--arm", default=None, help="arm label written to every record (default: preset name)")
    ap.add_argument("--config-name")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--intent", action="store_true", default=None)
    ap.add_argument("--intent-task-config")
    ap.add_argument("--intent-ckpt")
    ap.add_argument("--vl-cotrain", action="store_true", default=None)
    ap.add_argument("--intent-stack")
    ap.add_argument("--frozen-base-weights")
    ap.add_argument("--frozen-base-config")
    ap.add_argument("--norm-stats-from-config", action="store_true", default=None)
    ap.add_argument("--fp32", action="store_true", default=None)
    ap.add_argument("--config-dir", default=str(REPO / "examples/configs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7, help="env.seed (as native); per-episode seeds come from the case")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default=None, help="'|'-separated task_name slugs (default: all in manifest)")
    ap.add_argument("--reset-ids", default=None)
    ap.add_argument("--deltas", default=None)
    ap.add_argument("--policy-seeds", default="0", help="comma-separated policy seed indices per physical case")
    ap.add_argument("--max-cases", type=int, default=None, help="sanity batch: stop after N new episodes")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--dump-actions", default=None, help="dir: save per-replan action chunks (+intent) per episode")
    ap.add_argument("--full-checkpoint-hash", action="store_true", help="full sha256 instead of head/tail")
    args = ap.parse_args()

    preset = PRESETS.get(args.preset, {})
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    for k in ("intent", "vl_cotrain", "norm_stats_from_config", "fp32"):
        setattr(args, k, bool(getattr(args, k)))
    if args.frozen_base_config is None:
        args.frozen_base_config = "pi05_base_nointent"
    assert args.config_name and args.checkpoint_dir, "need --preset or --config-name/--checkpoint-dir"
    arm = args.arm or args.preset
    assert arm, "need --arm when no --preset is given"

    # ---- manifest ----
    mdir = pathlib.Path(args.manifest_dir)
    cases = oc.read_jsonl(mdir / "cases.jsonl")
    npz = np.load(mdir / "init_states.npz")
    states = npz["states"]
    assert list(npz["case_ids"]) == [c["case_id"] for c in cases], "manifest states/cases out of sync"
    meta = json.loads((mdir / "manifest_meta.json").read_text())
    assert meta["config_version"] == oc.CONFIG_VERSION, f"manifest {meta['config_version']} != {oc.CONFIG_VERSION}"
    if args.tasks:
        keep = set(args.tasks.split("|"))
        cases = [c for c in cases if c["task_name"] in keep]
    if args.reset_ids:
        keep = {int(x) for x in args.reset_ids.split(",")}
        cases = [c for c in cases if c["reset_id"] in keep]
    if args.deltas:
        keep = {round(float(x), 6) for x in args.deltas.split(",")}
        cases = [c for c in cases if round(c["delta_m"], 6) in keep]
    policy_seeds = [int(x) for x in args.policy_seeds.split(",")]

    # ---- provenance ----
    ckpt_fp, _ = oc.dir_fingerprint(args.checkpoint_dir, full=args.full_checkpoint_hash)
    side = args.intent_ckpt or args.intent_stack
    side_fp = oc.file_fingerprint(side, full=args.full_checkpoint_hash) if side else None
    ev_hash = evaluator_hash()
    prov = {"arm": arm, "preset": args.preset, "config_name": args.config_name, "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_hash": ckpt_fp, "intent_sidecar": side, "intent_sidecar_hash": side_fp,
            "intent_mode": "resnet_sidecar" if args.intent else ("vl_stack" if args.vl_cotrain else "none"),
            "frozen_base_weights": args.frozen_base_weights if args.vl_cotrain else None,
            "fp32": args.fp32, "norm_stats_from_config": args.norm_stats_from_config,
            "replan_steps": args.replan_steps, "evaluator_hash": ev_hash, "git_commit": git_commit(),
            "manifest_dir": str(mdir.resolve()), "validity_version": meta["validity_rules"]["version"],
            "host": platform.node()}
    out = pathlib.Path(args.out_dir)
    (out / arm).mkdir(parents=True, exist_ok=True)
    (out / arm / f"provenance_{ev_hash}_{int(time.time())}.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2), flush=True)

    def rkey(case_id, ps):
        return f"{case_id}#{arm}#{ckpt_fp}#{ev_hash}#s{ps}"

    # ---- policy (loaded once) + LIBERO ----
    import torch
    torch.load = functools.partial(torch.load, weights_only=False)
    import robosuite
    if tuple(int(x) for x in robosuite.__version__.split(".")[:2]) >= (1, 5):
        import mip.envs.libero._robosuite_compat  # noqa: F401
    policy, gen, vl = build_policy(args)
    obs_steps = gen.obs_steps if gen is not None else 1
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[oc.SUITE]()

    def mip_obs(obs):
        st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
        agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)
        wr = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
        return st, agv, wr

    by_task = collections.OrderedDict()
    for c in cases:
        by_task.setdefault(c["task_name"], []).append(c)
    task_rank = {t: i for i, t in enumerate(sorted({c["task_name"] for c in cases}))}

    n_new = 0
    for tname, tcases in by_task.items():
        rec_path = out / arm / f"{tname}.jsonl"
        done_keys = {r["result_key"] for r in oc.read_jsonl(rec_path)}
        todo = [(c, ps) for c in tcases for ps in policy_seeds if rkey(c["case_id"], ps) not in done_keys]
        print(f"[{arm}] {tname}: {len(tcases) * len(policy_seeds)} episodes, {len(todo)} to run", flush=True)
        if not todo:
            continue
        tid = tcases[0]["suite_task_id"]
        task = suite.get_task(tid)
        assert str(task.language).strip().lower() == tcases[0]["instruction"], "task id/instruction mismatch"
        suite_inits = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        def make_env():
            e = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=oc.ENV_RES, camera_widths=oc.ENV_RES)
            try:
                e.seed(args.seed)
            except (TypeError, AttributeError):
                pass
            return e
        env = make_env()
        xyz_idx = oc.flat_xyz_index(env.env.sim, tcases[0]["moved_joint"])
        assert xyz_idx == tcases[0]["flat_xyz_index"], "flat index differs from manifest"

        def run_episode(c, ps):
            state = np.asarray(states[c["state_row"]], dtype=np.float64)
            assert oc.state_hash(state) == c["initialized_state_hash"], "stored state hash mismatch"
            base_ok = oc.state_hash(np.asarray(suite_inits[c["reset_id"]], dtype=np.float64))
            if c["delta_m"] == 0.0:
                assert base_ok == c["initialized_state_hash"], "benchmark init file differs from manifest"
            seed = oc.episode_seed(c["instruction"], c["reset_id"], c["delta_m"], c["angle_idx"], ps)
            oc.seed_everything(seed, policy)
            env.reset()
            obs = env.set_init_state(state)
            plan = collections.deque()
            win = collections.deque(maxlen=obs_steps)
            record = args.video_dir is not None and ps == policy_seeds[0] and video_selected(c, task_rank[tname])
            frames = [] if record else None
            dump = [] if args.dump_actions else None
            done, success_step, policy_failure, settled = False, None, None, None
            min_d, steps = float("inf"), 0
            for t in range(oc.MAX_STEPS + oc.NUM_STEPS_WAIT):
                if t < oc.NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(oc.LIBERO_DUMMY_ACTION)
                    continue
                if settled is None:
                    settled = np.asarray(env.get_sim_state(), dtype=np.float64)[xyz_idx:xyz_idx + 3].copy()
                if not plan:
                    element = {
                        "observation/image": obs["agentview_image"],
                        "observation/wrist_image": obs["robot0_eye_in_hand_image"],
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32),
                        "prompt": str(task.language),
                    }
                    if gen is not None:
                        st, a128, w128 = mip_obs(obs)
                        win.append((st, a128, w128))
                        while len(win) < obs_steps:
                            win.appendleft(win[0])
                        sw = np.stack([w[0] for w in win])[None]
                        iw = {"agentview_rgb": np.stack([w[1] for w in win])[None],
                              "eye_in_hand_rgb": np.stack([w[2] for w in win])[None]}
                        element["intent"] = gen.intent_for_windows(sw, iw)[0].astype(np.float32)
                    if vl is not None:
                        element["intent"] = vl(element)
                    action_chunk = np.asarray(policy.infer(element)["actions"])
                    if dump is not None:
                        dump.append({"t": t, "actions": action_chunk.copy(),
                                     "intent": element.get("intent"), "state": element["observation/state"].copy()})
                    if not np.all(np.isfinite(action_chunk)):
                        policy_failure = "nonfinite_action"
                        break
                    plan.extend(action_chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                steps += 1
                obj = np.asarray(env.get_sim_state(), dtype=np.float64)[xyz_idx:xyz_idx + 3]
                min_d = min(min_d, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - obj)))
                if record:
                    frames.append(_video_frame(obs))
                if gen is not None:
                    win.append(mip_obs(obs))
                if done:
                    success_step = steps
                    break
            vpath = None
            if record and frames:
                vdir = pathlib.Path(args.video_dir) / arm
                vdir.mkdir(parents=True, exist_ok=True)
                vpath = str(vdir / f"{c['case_id'].replace('|', '__')}_s{ps}.mp4")
                _save_video(vpath, frames)
            if dump is not None:
                ddir = pathlib.Path(args.dump_actions) / arm
                ddir.mkdir(parents=True, exist_ok=True)
                np.save(ddir / f"{c['task_name']}_r{c['reset_id']}_d{c['delta_m']:.3f}_a{c['angle_idx']}_s{ps}.npy",
                        np.array(dump + [{"done": bool(done), "steps": steps}], dtype=object), allow_pickle=True)
            ms = c.get("settled_xyz")
            mismatch = None if (settled is None or ms is None) else float(np.linalg.norm(settled - np.asarray(ms)))
            return {"success": bool(done) and policy_failure is None, "success_step": success_step, "steps": steps,
                    "timeout": (not done) and policy_failure is None and steps >= oc.MAX_STEPS,
                    "policy_failure": policy_failure, "episode_seed": seed,
                    "settled_xyz": None if settled is None else settled.tolist(),
                    "manifest_settle_mismatch_m": mismatch, "min_eef_obj_dist_m": min_d, "video_path": vpath}

        for c, ps in todo:
            if args.max_cases is not None and n_new >= args.max_cases:
                break
            t0 = time.time()
            res, err = None, None
            for attempt in range(2):  # one deterministic retry of infrastructure failures
                try:
                    res = run_episode(c, ps)
                    err = None
                    break
                except AssertionError:
                    raise  # provenance/identity violations stop the campaign
                except Exception:
                    err = traceback.format_exc(limit=3)
                    print(f"[{arm}] infra error on {c['case_id']} attempt {attempt}: {err}", flush=True)
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = make_env()
            rec = {"result_key": rkey(c["case_id"], ps), "cohort_id": f"{oc.EXPERIMENT}/{oc.CONFIG_VERSION}",
                   "case_id": c["case_id"], "arm": arm, "policy_seed": ps, "task_name": tname,
                   "reset_id": c["reset_id"], "delta_m": c["delta_m"], "angle_idx": c["angle_idx"],
                   "angle_rad": c["angle_rad"], "initialized_state_hash": c["initialized_state_hash"],
                   "validity_status": c["validity_status"], "validity_reason": c["validity_reason"],
                   "validity_version": c["validity_version"], "checkpoint_hash": ckpt_fp,
                   "evaluator_hash": ev_hash, "config_name": args.config_name, "wall_s": round(time.time() - t0, 1),
                   "infrastructure_error": err}
            if res is not None:
                rec.update(res)
            oc.append_jsonl(rec_path, rec)
            n_new += 1
            print(f"[{arm}] {c['case_id']} s{ps} -> {'INFRA' if err else ('OK' if rec['success'] else 'fail')}"
                  f" steps={rec.get('steps')} {rec['wall_s']}s", flush=True)
        env.close()
        if args.max_cases is not None and n_new >= args.max_cases:
            break
    print(f"[{arm}] done: {n_new} new episodes -> {out / arm}")


if __name__ == "__main__":
    main()
