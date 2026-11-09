"""Rollout ablation for a trained BC-RNN intent policy: is the intent input actually used?

Runs N episodes with fixed per-episode seeds (identical initial states across modes) in one of
  normal   - the policy's own predicted intent (as in training-time evals)
  shuffle  - the intent fed to the LSTM is replaced, at every step, by a random row of the sidecar's
             intent targets (realistic values, unrelated to the current state)
  constant - replaced by the mean intent target (a fixed vector, no information)
The predictor still runs; only what the LSTM receives changes.

Usage (PYTHONPATH must start with external/robomimic_dexmg, as in scripts/dexmg_bcrnn_run.sh):
  python eval_intent_ablation.py --ckpt C.pth --sidecar S.hdf5 --intent_key hand_intent --dataset D.hdf5
                                 --mode shuffle --n 50 --out OUT.json [--match_dataset_scene]
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import train_bcrnn as TB  # noqa: E402

import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.train_utils as TrainUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--intent_key", default="intent")
    p.add_argument("--intent_input", default="proprio", choices=["proprio", "image"])
    p.add_argument("--dataset", required=True, help="original dataset (scene source for --match_dataset_scene)")
    p.add_argument("--mode", required=True, choices=["normal", "shuffle", "constant"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--match_dataset_scene", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    FileUtils.load_dict_from_checkpoint = lambda ckpt_path: torch.load(
        os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if not hasattr(TrainUtils, "SubprocVectorEnv"):  # run_rollout isinstance-checks it; tianshou not installed
        TrainUtils.SubprocVectorEnv = type("SubprocVectorEnv", (), {})
    TB.patch_env_creation(args.dataset, args.match_dataset_scene)
    TB.setup_intent(args.sidecar, args.intent_key, args.intent_input)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    with h5py.File(args.sidecar, "r") as f:
        pool = np.concatenate([f["data"][d][args.intent_key][()] for d in f["data"].keys()])
    pool_t = torch.from_numpy(pool).float().to(device)
    if args.mode == "shuffle":
        TB.IntentObsEncoder.OVERRIDE = ("shuffle", pool_t)
    elif args.mode == "constant":
        TB.IntentObsEncoder.OVERRIDE = ("constant", pool_t.mean(0))

    _, ckpt = FileUtils.algo_name_from_checkpoint(ckpt_path=args.ckpt)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    horizon = args.horizon or config.experiment.rollout.horizon
    shape_meta = ckpt["shape_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=ckpt["env_metadata"], render=False, render_offscreen=True,
        use_image_obs=shape_meta["use_images"], use_depth_obs=shape_meta["use_depths"])
    env = EnvUtils.wrap_env_from_config(env, config=config)

    succ, t0 = [], time.time()
    for ep in range(args.n):
        np.random.seed(args.seed0 + ep)
        torch.manual_seed(args.seed0 + ep)
        r = TrainUtils.run_rollout(policy=policy, env=env, horizon=horizon, terminate_on_success=True)
        succ.append(float(r["Success_Rate"][0]))
        print(f"[{args.mode}] ep {ep}: success {succ[-1]:.0f}  running {np.mean(succ):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = dict(ckpt=args.ckpt, mode=args.mode, n=args.n, seed0=args.seed0, horizon=horizon,
               success_rate=float(np.mean(succ)), successes=succ, intent_key=args.intent_key,
               intent_input=args.intent_input)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[{args.mode}] success {out['success_rate']:.3f} over {args.n} -> {args.out}")


if __name__ == "__main__":
    main()
