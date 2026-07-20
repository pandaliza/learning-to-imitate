"""Evaluate a Pi0.5 checkpoint on the LIBERO goal suite, with optional slot-intent.

Direct (in-process) rollout — adapts external/openpi/examples/libero/main.py to load
the policy locally via create_trained_policy and (optionally) inject a 64D intent
sampled from the MIP flow map (mip.pi05_intent.IntentGenerator) at each replan.

Per-step obs follows openpi's LIBERO convention:
  state = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]   (8D, matches our norm stats)
For the intent sidecar (MIP flow map) we additionally build the MIP obs window:
  state15 = [eef_pos, axisangle, gripper_qpos, joint_pos]  + agentview/eye_in_hand @128.

Outputs per-task + mean success rate to stdout and a JSON file.

Usage (baseline, no intent):
  python examples/openpi/eval_libero_intent.py --config-name pi05_libero \
    --checkpoint-dir /data/.../pi05_libero --out logs/eval_baseline.json
Usage (intent):
  python examples/openpi/eval_libero_intent.py --config-name pi05_libero_intent \
    --checkpoint-dir /data/.../slot_intent_goal_run1/18000 --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \
    --out logs/eval_intent.json
"""

import argparse
import collections
import functools
import json
import math
import pathlib

import numpy as np
import torch

# Must run before any LIBERO env import: patches robosuite 1.5.x API back to what
# LIBERO 1.4 expects (load_controller_config, robot/gripper shims). Same fix MIP uses.
import mip.envs.libero._robosuite_compat  # noqa: F401, E402

# LIBERO's *.pruned_init init-state files predate torch's weights_only=True default;
# load them as trusted local pickles (the MIP flow-map loader already does this).
torch.load = functools.partial(torch.load, weights_only=False)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = 300  # libero_goal: longest demo ~270 steps
NUM_STEPS_WAIT = 10


def _quat2axisangle(quat):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--num-trials-per-task", type=int, default=20)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rotate-images", action="store_true",
                    help="rotate 180 (off-the-shelf pi05_libero/PI convention); omit for our finetune")
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", default=None)
    ap.add_argument("--norm-stats-from-config", action="store_true",
                    help="load norm stats from the config's assets dir instead of checkpoint/assets "
                         "(for PyTorch co-train checkpoints, which save model.safetensors without assets/)")
    # VL co-train eval: intent comes from the model's OWN tapped VL features -> co-trained
    # vl_proj + flow map (intent_stack.pt), not the ResNet IntentGenerator sidecar.
    ap.add_argument("--vl-cotrain", action="store_true",
                    help="VL-grounded intent: tap the model's PaliGemma image tokens -> vl_proj -> flow map")
    ap.add_argument("--intent-stack", default=None, help="intent_stack.pt (vl_proj + intent_flow_map) for --vl-cotrain")
    ap.add_argument("--fp32", action="store_true", help="force the policy model to float32 (pi05_base overflows bf16)")
    # M3 (decoupled VL): the generator was trained on FROZEN pi05_base VL, so at deploy the VL tap
    # must use a frozen pi05_base -- NOT the finetuned policy's VL. Load it separately for the tap.
    ap.add_argument("--frozen-base-weights", default=None,
                    help="M3: dir of a frozen pi05_base PyTorch (model.safetensors) used ONLY for the VL tap")
    ap.add_argument("--frozen-base-config", default="pi05_base_nointent",
                    help="config to build the frozen-base PyTorch model architecture")
    # DINOv2/DynaFLIP arms: the slot-intent generator was trained on an ALTERNATIVE encoder's
    # agentview patch grid (not pi05 VL), so the deploy VL tap must run THAT encoder per step.
    ap.add_argument("--encoder-tap", default=None, choices=["dinov2", "dynaflip"],
                    help="tap an alt vision encoder (agentview-only) for the intent generator instead of pi05 VL")
    ap.add_argument("--config-dir", default="examples/configs")
    # FIXED-M4: the deployable generator is an aux MLP (vl_mean -> z_hat), not the flow map.
    ap.add_argument("--aux-head", default=None,
                    help="FIXED-M4: aux_head.pt; z_hat = aux_head(pi05 vl_mean) fed to the action head")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    np.random.seed(args.seed)

    # ---- load policy ----
    train_config = _config.get_config(args.config_name)
    norm_stats = None
    if args.norm_stats_from_config:
        from openpi.training import checkpoints as _checkpoints
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, norm_stats=norm_stats)
    if args.fp32 and not args.frozen_base_weights:
        policy._model = policy._model.float()  # pi05_base overflows bf16 -> fp32 forward (ALL paths)

    # ---- optional intent generator (MIP flow map sidecar) ----
    gen = None
    if args.intent:
        from mip.pi05_intent import IntentGenerator
        assert args.intent_ckpt, "--intent requires --intent-ckpt"
        gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir, device=args.device)
    obs_steps = gen.obs_steps if gen is not None else 1

    # ---- VL co-train intent: tap the model's own PaliGemma image tokens -> co-trained
    #      vl_proj -> flow map (intent_stack.pt). No ResNet sidecar. ----
    vl = None
    if args.vl_cotrain:
        import openpi.models.model as _model
        from mip.pi05_intent import CotrainIntentModule
        assert args.intent_stack or args.aux_head, "--vl-cotrain requires --intent-stack or --aux-head"
        cot = CotrainIntentModule(args.intent_task_config, args.config_dir, device=args.device, vl_obs=True)
        if args.intent_stack:  # flow-map generator (M2/M3/DINOv2); FIXED-M4 uses --aux-head instead
            stack = torch.load(args.intent_stack, map_location=args.device, weights_only=False)
            vlw = stack["vl_proj"]["weight"].shape[1]
            cot.vl_proj(torch.zeros(1, vlw, device=args.device))         # materialize LazyLinear before load
            cot.vl_proj.load_state_dict(stack["vl_proj"])
            cot.agent.intent_flow_map.load_state_dict(stack["intent_flow_map"])
        cot.eval()
        # M3: load a FROZEN pi05_base for the VL tap (decoupled generator was trained on frozen-base
        # VL, not the finetuned policy's VL). Falls back to the policy's own VL (M2 co-train) if unset.
        vl_model = policy._model
        if args.frozen_base_weights:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
            import safetensors.torch as _stt
            fb_cfg = _config.get_config(args.frozen_base_config)
            vl_model = PI0Pytorch(fb_cfg.model).to(args.device)
            _stt.load_model(vl_model, str(pathlib.Path(args.frozen_base_weights) / "model.safetensors"), strict=False)
            vl_model = vl_model.float().eval()

        # DINOv2/DynaFLIP arms: tap that encoder's AGENTVIEW patch grid (the generator was trained on
        # it), not pi05 VL. Reuse the precompute featurizer (same model/preprocess as the cache).
        featurize = None
        if args.encoder_tap:
            from precompute_encoder_grids import _load_encoder
            featurize = _load_encoder(args.encoder_tap, args.device)

        # FIXED-M4: rebuild the aux MLP generator (vl_mean -> z_hat). Input dim from the saved 1st layer.
        aux_head_eval = None
        if args.aux_head:
            _ah = torch.load(args.aux_head, map_location=args.device, weights_only=False)["aux_head"]
            _in = _ah["0.weight"].shape[1]
            aux_head_eval = torch.nn.Sequential(
                torch.nn.Linear(_in, 512), torch.nn.GELU(), torch.nn.Linear(512, cot.intent_dim)).to(args.device)
            aux_head_eval.load_state_dict(_ah)
            aux_head_eval.eval()

        def _to_obs(inp):
            def conv(v):
                if isinstance(v, dict):
                    return {k: conv(vv) for k, vv in v.items()}
                t = torch.as_tensor(np.asarray(v), device=args.device)
                return (t.float() if t.is_floating_point() else t)[None]  # add batch dim
            return _model.Observation.from_dict(conv(inp))

        def _vl_intent(element):
            if aux_head_eval is not None:  # FIXED-M4: aux MLP generator on the pi05 vl_mean
                el = dict(element); el["intent"] = np.zeros(cot.intent_dim, dtype=np.float32)
                obs_t = _to_obs(policy._input_transform(el))
                with torch.no_grad():
                    z = aux_head_eval(vl_model.vl_image_features(obs_t))
            elif featurize is not None:  # alt-encoder agentview grid -> mean -> generator
                grid = featurize([element["observation/image"]])      # (1, N, dim)
                with torch.no_grad():
                    z = cot.vl_sample_intent(grid.mean(dim=1).to(args.device).float())
            else:                      # pi05 VL tap (M2 co-train / M3 frozen-base)
                el = dict(element); el["intent"] = np.zeros(cot.intent_dim, dtype=np.float32)
                obs_t = _to_obs(policy._input_transform(el))
                with torch.no_grad():
                    z = cot.vl_sample_intent(vl_model.vl_image_features(obs_t))
            return z[0].detach().cpu().numpy().astype(np.float32)
        vl = _vl_intent

    # ---- LIBERO suite ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.n_tasks

    def mip_obs(obs):
        """Build (state15, agentview128_chw, wrist128_chw) for the flow map, MIP convention."""
        st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
        agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)        # CHW uint8
        wr = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
        return st, agv, wr

    results = {}
    total_ep, total_succ = 0, 0
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)  # robosuite 1.5.x may not expose a callable seed(); init states fix the reset anyway
        except (TypeError, AttributeError):
            pass
        t_ep, t_succ = 0, 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            plan = collections.deque()
            win = collections.deque(maxlen=obs_steps)
            done = False
            for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue
                if not plan:
                    agv = obs["agentview_image"]
                    wr = obs["robot0_eye_in_hand_image"]
                    if args.rotate_images:
                        agv = np.ascontiguousarray(agv[::-1, ::-1])
                        wr = np.ascontiguousarray(wr[::-1, ::-1])
                    element = {
                        "observation/image": agv,
                        "observation/wrist_image": wr,
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
                        sw = np.stack([w[0] for w in win])[None]          # (1,To,15)
                        iw = {"agentview_rgb": np.stack([w[1] for w in win])[None],
                              "eye_in_hand_rgb": np.stack([w[2] for w in win])[None]}
                        element["intent"] = gen.intent_for_windows(sw, iw)[0].astype(np.float32)
                    if vl is not None:
                        element["intent"] = vl(element)
                    action_chunk = np.asarray(policy.infer(element)["actions"])
                    plan.extend(action_chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                if gen is not None:  # keep the obs window fresh between replans
                    win.append(mip_obs(obs))
                if done:
                    break
            t_ep += 1
            t_succ += int(done)
            total_ep += 1
            total_succ += int(done)
        env.close()
        sr = t_succ / max(t_ep, 1)
        results[str(task.language)] = sr
        print(f"[{args.task_suite}] task {task_id} SR={sr:.2f} ({t_succ}/{t_ep}) :: {task.language}", flush=True)

    mean_sr = total_succ / max(total_ep, 1)
    summary = {"config": args.config_name, "checkpoint": args.checkpoint_dir, "intent": args.intent,
               "task_suite": args.task_suite, "num_trials_per_task": args.num_trials_per_task,
               "mean_sr": mean_sr, "total": f"{total_succ}/{total_ep}", "per_task": results}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {args.config_name} intent={args.intent} MEAN SR = {mean_sr:.3f} ({total_succ}/{total_ep}) -> {args.out}")


if __name__ == "__main__":
    main()
