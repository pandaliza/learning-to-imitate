"""Evaluate Pi0.5 (with/without slot-intent) under radial XY perturbation of the target object.

For each (task, δ) pair, shifts the target object's XY position by δ metres in
evenly-spaced directions before each episode and measures success rate.  Running
both configs side-by-side produces the SR-vs-δ degradation curves that test whether
intent conditioning helps generalise to OOD object positions.

Training-distribution note: across all libero_goal init states, object x-coordinates
have zero variance (fixed per-object column on the table); y-coordinates vary with
std ≈ 0.01 m.  A radial perturbation in XY therefore probes two distinct regimes:
  - y component alone: in-distribution direction, OOD at δ ≳ 0.03 m (3 σ)
  - x component:       immediately OOD at any δ > 0

Tasks 0 ("open middle drawer") and 7 ("turn on stove") are skipped: their targets
are a slide joint / hinge joint, not a free-body object with an XY position.

Qpos layout (identical across all 10 libero_goal tasks):
  state[9:12]   — akita_black_bowl  xyz
  state[16:19]  — cream_cheese      xyz
  state[23:26]  — wine_bottle       xyz
  state[30:33]  — plate             xyz

Usage (no-intent baseline):
  python examples/openpi/eval_libero_perturbed.py \\
    --config-name pi05_libero_nointent \\
    --checkpoint-dir /data/.../nointent_goal_run1/30000 \\
    --out logs/perturb_nointent.json

Usage (slot-intent conditioned):
  python examples/openpi/eval_libero_perturbed.py \\
    --config-name pi05_libero_intent \\
    --checkpoint-dir /data/.../slot_intent_goal_run1/30000 \\
    --intent \\
    --intent-task-config libero_goal_suite_image_slot_intent \\
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \\
    --out logs/perturb_intent.json

Then plot the two JSON files together to get the SR-vs-δ comparison.
"""

import argparse
import collections
import functools
import json
import math
import pathlib
import sys

import numpy as np
import torch

import mip.envs.libero._robosuite_compat  # noqa: F401
torch.load = functools.partial(torch.load, weights_only=False)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = 300
NUM_STEPS_WAIT = 10

# qpos index of the xyz start for each free-body object (free joint = 7 qpos: xyz + quat)
OBJECT_QPOS_XYZ = {
    "bowl":         9,
    "cream_cheese": 16,
    "wine_bottle":  23,
    "plate":        30,
}

# task_id -> (object_name, qpos_xyz_start) to perturb; None = skip
# Target free-joint NAME per task; the flat-state index is resolved from the built env's
# sim.model at runtime (jnt_qposadr + 1 for the leading `time`). Hardcoded indices are
# forbidden here: an off-by-one (qpos addr vs flat idx) silently reclassified every failure
# as no_reach and half-applied the perturbation for the entire first M10 campaign.
TASK_TARGET = {
    0: None,                                            # "open middle drawer" — drawer is a slide joint
    1: ("bowl", "akita_black_bowl_1_joint0"),           # "put the bowl on the stove"
    2: ("wine_bottle", "wine_bottle_1_joint0"),         # "put the wine bottle on top of the cabinet"
    3: ("bowl", "akita_black_bowl_1_joint0"),           # "open top drawer and put bowl inside"
    4: ("bowl", "akita_black_bowl_1_joint0"),           # "put the bowl on top of the cabinet"
    5: ("plate", "plate_1_joint0"),                     # "push the plate to the front of the stove"
    6: ("cream_cheese", "cream_cheese_1_joint0"),       # "put the cream cheese in the bowl"
    7: None,                                            # "turn on the stove" — button is a hinge joint
    8: ("bowl", "akita_black_bowl_1_joint0"),           # "put the bowl on the plate"
    9: ("wine_bottle", "wine_bottle_1_joint0"),         # "put the wine bottle on the rack"
}


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize(img_hwc, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR))


def _perturb(state, qpos_xyz_start, dx, dy):
    """Copy state and shift the target object's XY by (dx, dy) metres."""
    s = state.copy()
    s[qpos_xyz_start]     += dx
    s[qpos_xyz_start + 1] += dy
    return s


def _object_xyz(env, qpos_xyz_start):
    """Live object xyz from the flattened sim state (call after the settle steps).

    get_sim_state() returns the SAME [time, qpos, qvel] layout that the init_state the
    perturbation edits uses, so indexing it at qpos_xyz_start is guaranteed consistent with
    _perturb / req_xyz (robosuite's flat state has a leading `time`, so this is NOT data.qpos)."""
    flat = np.asarray(env.get_sim_state(), dtype=np.float64)
    return flat[qpos_xyz_start:qpos_xyz_start + 3].copy()


def _eef_obj_dist(obs, obj_xyz):
    """Gripper-site to object distance (m) for the reach test."""
    return float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - obj_xyz))


def _video_frame(obs):
    """Human-viewable agentview+wrist side-by-side frame (robosuite renders are bottom-up)."""
    agv = np.ascontiguousarray(obs["agentview_image"][::-1])
    wr  = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return np.concatenate([agv, wr], axis=1)


def _save_video(path, frames, fps=20):
    import imageio
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def _failure_class(done, reached, grasped):
    """Decompose a pick-and-place episode into the stage that broke.
      success   — task completed
      no_reach  — gripper never got near the object (localisation / approach failure)
      no_grasp  — reached but never lifted it (grasp failure)
      no_place  — lifted it but didn't complete the task (transport / placement failure)"""
    if done:
        return "success"
    if not reached:
        return "no_reach"
    if not grasped:
        return "no_grasp"
    return "no_place"


def _robot_base_xy(env):
    """Best-effort robot-base XY (world xpos) for the reach check; None if no body resolves."""
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return None
    for name in ("robot0_base", "robot0_link0", "base"):
        try:
            return np.asarray(sim.data.get_body_xpos(name)[:2], dtype=np.float64).copy()
        except Exception:
            continue
    return None


def _validity(req_xyz, settled_xyz, robot_xy, args):
    """Physics-grounded validity of a perturbed placement, read back after settling.

    Returns (valid, reason, metrics). Both arms share init states, so this does NOT
    affect the intent-vs-control Δ; it quantifies how many placements are physically
    realizable so the absolute SR curve can be interpreted (and an SR-over-valid reported).
      - xy_drift: object shoved from its requested XY by contact resolution => collision/ejection
      - z_drop:   object below its requested Z after settling                => fell off the table
      - reach:    requested XY beyond --max-reach from the robot base        => out of workspace
    """
    xy_drift = float(np.hypot(settled_xyz[0] - req_xyz[0], settled_xyz[1] - req_xyz[1]))
    z_drop   = float(req_xyz[2] - settled_xyz[2])
    reach    = float(np.hypot(req_xyz[0] - robot_xy[0], req_xyz[1] - robot_xy[1])) if robot_xy is not None else float("nan")
    valid, reason = True, "ok"
    if xy_drift > args.xy_eject_tol:
        valid, reason = False, "collision"
    elif z_drop > args.z_fall_tol:
        valid, reason = False, "fell"
    elif args.max_reach is not None and robot_xy is not None and reach > args.max_reach:
        valid, reason = False, "unreachable"
    return valid, reason, {"xy_drift": xy_drift, "z_drop": z_drop, "reach": reach}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task IDs (0-9); default = all with a perturbable target")
    ap.add_argument("--task-target-qpos", type=int, default=None,
                    help="override TASK_TARGET for all tasks: qpos index of target object xyz "
                         "(e.g. 9 for bowl). Use when running a suite other than libero_goal.")
    ap.add_argument("--deltas", default="0.0,0.02,0.05,0.10,0.15,0.20",
                    help="comma-separated perturbation magnitudes in metres")
    ap.add_argument("--num-trials-per-delta", type=int, default=10,
                    help="episodes per (task, delta); directions are evenly spaced in [0, 2π)")
    ap.add_argument("--axis", default="radial", choices=["radial", "x", "y"],
                    help="radial=evenly-spaced directions; x/y=perturb only that axis (±δ, "
                         "trials split between + and - sign). x is immediately OOD (zero train "
                         "variance); y is in-distribution to ~3σ≈0.03 m.")
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    # M10 co-prediction: route sampling through sample_actions_copred (deploy graph has intent
    # tokens). Mirrors eval_libero_intent.py; needs a copred checkpoint (--config-name pi05_base_copred).
    ap.add_argument("--schedule", default=None, choices=["s1", "s2", "s3"],
                    help="M10: s1 intent-first, s2 joint, s3 action-first control")
    ap.add_argument("--num-intent-steps", type=int, default=4, help="M10: K_I Euler steps for the intent phase")
    ap.add_argument("--copred-mask", default=None, choices=["j", "t", "b1"],
                    help="M10: override the config's suffix mask variant to match the checkpoint")
    ap.add_argument("--rotate-images", action="store_true",
                    help="rotate 180° (needed for off-the-shelf pi05_libero; omit for our finetune)")
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--intent-override", default="flow", choices=["flow", "zero", "random"],
                    help="flow=normal sidecar; zero=all-zeros intent; random=Gaussian noise intent")
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", default=None)
    # VL co-train: deployable intent = the model's own PaliGemma image tokens -> co-trained
    # vl_proj -> flow map (intent_stack.pt). No ResNet IntentGenerator sidecar.
    ap.add_argument("--vl-cotrain", action="store_true",
                    help="generate intent from the co-trained VL flow map (intent_stack.pt)")
    ap.add_argument("--intent-stack", default=None,
                    help="intent_stack.pt (vl_proj + intent_flow_map) for --vl-cotrain")
    ap.add_argument("--frozen-base-weights", default=None,
                    help="M3: dir of a frozen pi05_base PyTorch (model.safetensors) used ONLY for the VL tap")
    ap.add_argument("--frozen-base-config", default="pi05_base_nointent",
                    help="config to build the frozen-base PyTorch model architecture")
    ap.add_argument("--encoder-tap", default=None, choices=["dinov2", "dynaflip"],
                    help="DINOv2/DynaFLIP arms: tap that encoder's agentview grid for the intent "
                         "generator (instead of the pi05 VL tap)")
    ap.add_argument("--aux-head", default=None,
                    help="FIXED-M4: aux_head.pt; z_hat = aux_head(pi05 vl_mean) fed to the action head "
                         "(used instead of --intent-stack)")
    ap.add_argument("--norm-stats-from-config", action="store_true",
                    help="load norm stats from the config assets (co-train ckpt dir has no assets/)")
    ap.add_argument("--fp32", action="store_true",
                    help="force the policy model to float32 (pi05_base overflows bf16)")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--xy-eject-tol", type=float, default=0.04,
                    help="max object XY drift (m) from requested after settling; above = ejected by collision")
    ap.add_argument("--z-fall-tol", type=float, default=0.03,
                    help="max object Z drop (m) after settling; above = fell off the table")
    ap.add_argument("--max-reach", type=float, default=None,
                    help="if set, flag requested XY beyond this radius (m) from the robot base as out-of-workspace")
    ap.add_argument("--reach-thresh", type=float, default=0.08,
                    help="gripper-object distance (m) below which the object counts as reached")
    ap.add_argument("--lift-thresh", type=float, default=0.04,
                    help="object rise (m) above its settled height that counts as a grasp")
    ap.add_argument("--video-dir", default=None,
                    help="if set, save agentview+wrist rollout mp4s here (filename tags the failure stage)")
    ap.add_argument("--videos-per-cell", type=int, default=2,
                    help="max rollout videos to save per (task, delta)")
    ap.add_argument("--out", default="logs/eval_perturbed.json")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    n = args.num_trials_per_delta

    sys.path.insert(0, "external/openpi/src")
    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    np.random.seed(args.seed)

    train_config = _config.get_config(args.config_name)
    if args.copred_mask:
        import dataclasses
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, copred_mask=args.copred_mask))
    norm_stats = None
    if args.norm_stats_from_config:
        from openpi.training import checkpoints as _checkpoints
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, norm_stats=norm_stats)
    if args.schedule:
        # M10: swap the sampler for the co-prediction schedule driver (same as eval_libero_intent.py);
        # the rest of the policy pipeline (transforms, normalization, chunking) is untouched.
        assert int(getattr(policy._model, "copred_h", 0)) > 0, \
            "--schedule requires a co-prediction checkpoint (config with copred_h > 0)"
        policy._sample_actions = functools.partial(
            policy._model.sample_actions_copred,
            schedule=args.schedule, num_intent_steps=args.num_intent_steps)
        print(f"[M10] perturbed eval via sample_actions_copred schedule={args.schedule}", flush=True)

    gen = None
    if args.intent:
        from mip.pi05_intent import IntentGenerator
        assert args.intent_ckpt, "--intent requires --intent-ckpt"
        gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir,
                              device=args.device)
    obs_steps = gen.obs_steps if gen is not None else 1

    # VL co-train deployable intent: tap the model's PaliGemma image tokens -> co-trained
    # vl_proj -> flow map (intent_stack.pt). Mirrors eval_libero_intent.py's --vl-cotrain.
    vl = None
    vl_dim = 0
    if args.vl_cotrain:
        import openpi.models.model as _model
        from mip.pi05_intent import CotrainIntentModule
        assert args.intent_stack or args.aux_head, "--vl-cotrain requires --intent-stack or --aux-head"
        if args.fp32 and not args.frozen_base_weights:
            policy._model = policy._model.float()        # pi05_base overflows bf16 -> fp32 forward
        cot = CotrainIntentModule(args.intent_task_config, args.config_dir, device=args.device, vl_obs=True)
        if args.intent_stack:  # flow-map generator (M2/M3/DINOv2); FIXED-M4 uses --aux-head instead
            stack = torch.load(args.intent_stack, map_location=args.device, weights_only=False)
            vlw = stack["vl_proj"]["weight"].shape[1]
            cot.vl_proj(torch.zeros(1, vlw, device=args.device))   # materialize LazyLinear before load
            cot.vl_proj.load_state_dict(stack["vl_proj"])
            cot.agent.intent_flow_map.load_state_dict(stack["intent_flow_map"])
        cot.eval()
        vl_dim = int(cot.intent_dim)
        # FIXED-M4: aux MLP generator (vl_mean -> z_hat). Input dim from the saved 1st layer.
        aux_head_eval = None
        if args.aux_head:
            _ah = torch.load(args.aux_head, map_location=args.device, weights_only=False)["aux_head"]
            _in = _ah["0.weight"].shape[1]
            aux_head_eval = torch.nn.Sequential(
                torch.nn.Linear(_in, 512), torch.nn.GELU(), torch.nn.Linear(512, vl_dim)).to(args.device)
            aux_head_eval.load_state_dict(_ah)
            aux_head_eval.eval()
        # M3: frozen pi05_base for the VL tap (decoupled generator trained on frozen-base VL, not
        # the finetuned policy's VL). Falls back to the policy's own VL (M2 co-train) if unset.
        vl_model = policy._model
        if args.frozen_base_weights:
            import os as _os
            import openpi.training.config as _cfgmod
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
            import safetensors.torch as _stt
            fb_cfg = _cfgmod.get_config(args.frozen_base_config)
            vl_model = PI0Pytorch(fb_cfg.model).to(args.device)
            _stt.load_model(vl_model, _os.path.join(args.frozen_base_weights, "model.safetensors"), strict=False)
            vl_model = vl_model.float().eval()

        # DINOv2/DynaFLIP arms: tap that encoder's AGENTVIEW patch grid (the generator was trained
        # on it), not pi05 VL. Reuse the precompute featurizer (same model/preprocess as the cache).
        featurize = None
        if args.encoder_tap:
            from precompute_encoder_grids import _load_encoder
            featurize = _load_encoder(args.encoder_tap, args.device)

        def _to_obs(inp):
            def conv(v):
                if isinstance(v, dict):
                    return {k: conv(vv) for k, vv in v.items()}
                t = torch.as_tensor(np.asarray(v), device=args.device)
                return (t.float() if t.is_floating_point() else t)[None]   # add batch dim
            return _model.Observation.from_dict(conv(inp))

        def _vl_intent(element):
            if aux_head_eval is not None:                          # FIXED-M4: aux MLP on the pi05 vl_mean
                el = dict(element); el["intent"] = np.zeros(vl_dim, dtype=np.float32)
                obs_t = _to_obs(policy._input_transform(el))
                with torch.no_grad():
                    z = aux_head_eval(vl_model.vl_image_features(obs_t))
            elif featurize is not None:                            # alt-encoder agentview grid -> mean -> generator
                grid = featurize([element["observation/image"]])   # (1, N, dim)
                with torch.no_grad():
                    z = cot.vl_sample_intent(grid.mean(dim=1).to(args.device).float())
            else:                                                  # pi05 VL tap (M2 co-train / M3 frozen-base)
                el = dict(element); el["intent"] = np.zeros(vl_dim, dtype=np.float32)
                obs_t = _to_obs(policy._input_transform(el))
                with torch.no_grad():
                    z = cot.vl_sample_intent(vl_model.vl_image_features(obs_t))
            return z[0].detach().cpu().numpy().astype(np.float32)
        vl = _vl_intent

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()

    if args.tasks is not None:
        task_ids = [int(x) for x in args.tasks.split(",")]
    else:
        task_ids = [tid for tid in range(suite.n_tasks) if TASK_TARGET[tid] is not None]

    # Evenly-spaced directions: trial i uses the same angle at every δ, so curves are
    # directly comparable (same spatial direction probed at all magnitudes). For --axis x/y,
    # the displacement is confined to one axis with the sign alternating per trial.
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)

    def _disp(trial_i, delta):
        if args.axis == "radial":
            return delta * math.cos(angles[trial_i]), delta * math.sin(angles[trial_i])
        sign = 1.0 if trial_i % 2 == 0 else -1.0
        return (sign * delta, 0.0) if args.axis == "x" else (0.0, sign * delta)

    def _mip_obs(obs):
        st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
        agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)
        wr  = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
        return st, agv, wr

    # results[delta_str][task_id_str] = {sr, succ, ep, task, target_obj, delta}
    results: dict = {str(d): {} for d in deltas}

    for task_id in task_ids:
        task = suite.get_task(task_id)
        target = TASK_TARGET.get(task_id)
        if target is None and args.task_target_qpos is None:
            print(f"[task {task_id}] SKIP — no free-joint target :: {task.language}", flush=True)
            continue

        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)
        except (TypeError, AttributeError):
            pass
        env.reset()
        if args.task_target_qpos is not None:
            obj_name, qpos_start = "override", args.task_target_qpos
        else:
            obj_name, target_joint = target
            sim = env.env.sim
            jid = sim.model.joint_name2id(target_joint)
            assert sim.model.jnt_type[jid] == 0, f"{target_joint} is not a free joint"
            # flat sim state is [time, qpos, qvel] -> flat idx = qpos addr + 1
            qpos_start = int(sim.model.jnt_qposadr[jid]) + 1
            print(f"[task {task_id}] target={obj_name} joint={target_joint} flat_idx={qpos_start}", flush=True)
        robot_xy = _robot_base_xy(env)  # fixed per env; used for the optional reach check
        print(f"[task {task_id}] robot_base_xy={None if robot_xy is None else robot_xy.round(3).tolist()}  "
              f"tol(xy_eject={args.xy_eject_tol}, z_fall={args.z_fall_tol}, max_reach={args.max_reach})", flush=True)

        settled_ref = {}  # trial_i -> settled unperturbed xyz (validity baseline; cancels the settle drop)
        for delta in deltas:
            t_ep = t_succ = t_valid = t_succ_valid = 0
            reasons = collections.Counter()
            fails = collections.Counter()   # failure-stage breakdown over VALID trials only
            n_vid = 0                        # videos saved for this (task, delta) cell
            for trial_i in range(n):
                record = args.video_dir is not None and n_vid < args.videos_per_cell
                frames = [] if record else None
                base = init_states[trial_i % len(init_states)]
                if delta > 0.0:
                    dx, dy = _disp(trial_i, delta)
                    init_state = _perturb(base, qpos_start, dx, dy)
                else:
                    dx = dy = 0.0
                    init_state = base
                # Validity baseline: the SETTLED unperturbed pose (+ requested shift), not the raw
                # init pose — init states start objects above the surface, and the settle drop
                # would otherwise read as "fell" on every trial. Falls back to the raw init pose
                # for delta orderings that hit delta>0 before 0.0.
                ref = settled_ref.get(trial_i % len(init_states))
                if ref is not None:
                    req_xyz = ref + np.array([dx, dy, 0.0])
                else:
                    req_xyz = np.asarray(init_state[qpos_start:qpos_start + 3], dtype=np.float64).copy()

                env.reset()
                obs = env.set_init_state(init_state)
                plan = collections.deque()
                win  = collections.deque(maxlen=obs_steps)
                done = False
                settled_xyz = None
                obj_zmax = None          # max object height seen during the policy phase
                min_reach = float("inf") # closest the gripper got to the object

                for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        continue
                    if settled_xyz is None:
                        settled_xyz = _object_xyz(env, qpos_start)  # read once, right after settling
                        obj_zmax = float(settled_xyz[2])
                        if delta == 0.0 and trial_i % len(init_states) not in settled_ref:
                            settled_ref[trial_i % len(init_states)] = settled_xyz.copy()
                    if not plan:
                        agv = obs["agentview_image"]
                        wr  = obs["robot0_eye_in_hand_image"]
                        if args.rotate_images:
                            agv = np.ascontiguousarray(agv[::-1, ::-1])
                            wr  = np.ascontiguousarray(wr[::-1, ::-1])
                        element = {
                            "observation/image":       agv,
                            "observation/wrist_image": wr,
                            "observation/state": np.concatenate([
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ]).astype(np.float32),
                            "prompt": str(task.language),
                        }
                        if gen is not None:
                            st, a128, w128 = _mip_obs(obs)
                            win.append((st, a128, w128))
                            while len(win) < obs_steps:
                                win.appendleft(win[0])
                            if args.intent_override == "zero":
                                element["intent"] = np.zeros(gen.intent_dim, dtype=np.float32)
                            elif args.intent_override == "random":
                                element["intent"] = np.random.randn(gen.intent_dim).astype(np.float32)
                            else:
                                sw = np.stack([w[0] for w in win])[None]      # (1, To, 15)
                                iw = {
                                    "agentview_rgb":  np.stack([w[1] for w in win])[None],
                                    "eye_in_hand_rgb": np.stack([w[2] for w in win])[None],
                                }
                                element["intent"] = gen.intent_for_windows(sw, iw)[0].astype(np.float32)
                        if vl is not None:
                            if args.intent_override == "zero":
                                element["intent"] = np.zeros(vl_dim, dtype=np.float32)
                            elif args.intent_override == "random":
                                element["intent"] = np.random.randn(vl_dim).astype(np.float32)
                            else:
                                element["intent"] = vl(element)   # co-trained VL flow-map intent
                        action_chunk = np.asarray(policy.infer(element)["actions"])
                        plan.extend(action_chunk[: args.replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    cur_obj = _object_xyz(env, qpos_start)
                    obj_zmax  = max(obj_zmax, float(cur_obj[2]))
                    min_reach = min(min_reach, _eef_obj_dist(obs, cur_obj))
                    if record:
                        frames.append(_video_frame(obs))
                    if gen is not None:
                        win.append(_mip_obs(obs))
                    if done:
                        break

                if settled_xyz is None:                       # episode ended during the settle window
                    settled_xyz = _object_xyz(env, qpos_start)
                    obj_zmax = float(settled_xyz[2])
                reached = min_reach < args.reach_thresh
                grasped = (obj_zmax - float(settled_xyz[2])) > args.lift_thresh
                fclass  = _failure_class(done, reached, grasped)
                if delta == 0.0:
                    valid, reason = True, "ok"   # unperturbed placement is valid by definition
                else:
                    valid, reason, _ = _validity(req_xyz, settled_xyz, robot_xy, args)
                reasons[reason] += 0 if valid else 1

                if record and frames:
                    tag = fclass if valid else f"invalid_{reason}"
                    vpath = pathlib.Path(args.video_dir) / (
                        f"{args.config_name}_t{task_id}_d{delta:.2f}_trial{trial_i}_{tag}.mp4")
                    vpath.parent.mkdir(parents=True, exist_ok=True)
                    _save_video(str(vpath), frames)
                    n_vid += 1

                t_ep   += 1
                t_succ += int(done)
                if valid:
                    t_valid      += 1
                    t_succ_valid += int(done)
                    fails[fclass] += 1   # stage breakdown only over physically-valid trials

            sr = t_succ / max(t_ep, 1)
            sr_valid = t_succ_valid / max(t_valid, 1)
            valid_frac = t_valid / max(t_ep, 1)
            results[str(delta)][str(task_id)] = {
                "sr": sr, "succ": t_succ, "ep": t_ep,
                "sr_valid": sr_valid, "n_valid": t_valid, "valid_frac": valid_frac,
                "invalid_reasons": dict(reasons),
                "fail_breakdown": dict(fails),   # over valid trials: success/no_reach/no_grasp/no_place
                "task": task.language, "target_obj": obj_name, "delta": delta,
            }
            fb = "  ".join(f"{k}={fails[k]}" for k in ("success", "no_reach", "no_grasp", "no_place") if fails[k])
            print(f"[task {task_id}] δ={delta:.2f}  SR={sr:.2f} ({t_succ}/{t_ep})  "
                  f"valid={valid_frac:.2f} SR|valid={sr_valid:.2f} ({t_succ_valid}/{t_valid})  "
                  f"[{fb}]"
                  f"{'' if not reasons else '  ' + dict(reasons).__repr__()}  "
                  f"obj={obj_name} :: {task.language}", flush=True)

        env.close()

    # Aggregate: mean SR across tasks at each δ (over all trials, and over valid-only trials)
    for d in deltas:
        task_vals = [v for v in results[str(d)].values() if isinstance(v, dict)]
        task_srs       = [v["sr"] for v in task_vals]
        task_srs_valid = [v["sr_valid"] for v in task_vals if v["n_valid"] > 0]
        task_vfrac     = [v["valid_frac"] for v in task_vals]
        results[str(d)]["_mean_sr"]         = float(np.mean(task_srs)) if task_srs else float("nan")
        results[str(d)]["_mean_sr_valid"]   = float(np.mean(task_srs_valid)) if task_srs_valid else float("nan")
        results[str(d)]["_mean_valid_frac"] = float(np.mean(task_vfrac)) if task_vfrac else float("nan")

    out_data = {
        "config":               args.config_name,
        "checkpoint":           args.checkpoint_dir,
        "intent":               args.intent,
        "task_suite":           args.task_suite,
        "num_trials_per_delta": n,
        "deltas":               deltas,
        "results":              results,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n=== {args.config_name}  intent={args.intent} ===")
    print(f"{'delta':>8}  {'mean_SR':>8}  {'SR|valid':>9}  {'valid':>6}  tasks")
    for d in deltas:
        mean_sr    = results[str(d)].get("_mean_sr", float("nan"))
        mean_srv   = results[str(d)].get("_mean_sr_valid", float("nan"))
        mean_vfrac = results[str(d)].get("_mean_valid_frac", float("nan"))
        per_task = "  ".join(
            f"{tid}:{v['sr']:.2f}"
            for tid, v in results[str(d)].items()
            if not tid.startswith("_")
        )
        print(f"{d:8.2f}  {mean_sr:8.3f}  {mean_srv:9.3f}  {mean_vfrac:6.2f}  {per_task}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
