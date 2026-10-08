"""Native A-MIX reproduction: the OFFICIAL OopsieVerse CFM trainer, unchanged, on a split-filtered
playback HDF5.

What is official and untouched (src/oopsieverse/learning @ a739f99):
  PlaybackDataset, CFMPolicy/PolicyConfig, CFMTrainer.train() loop, AdamW, grad clip, Beta(1.5,1)
  time sampler, Euler-16 sampler from N(0,I), action min-max to [0,1], config YAML values.

What this wrapper adds (all outside the training math):
  1. Seeding (torch/numpy/random): the official trainer sets no seed.
  2. data_path -> split-filtered playback HDF5 (split_data.py) instead of the full file.
  3. wandb -> local JSONL shim (official loop calls wandb.log; we capture it with wall-clock).
  4. evaluate(): the official evaluate() runs on the train set (eval_dataset defaults to train);
     we keep that value and additionally run the SAME official evaluate() on the dev split
     (normalised with train statistics, same class vocabulary) and log it as eval/dev_mse_loss.
  5. Checkpoint sidecar (meta.json): class vocabulary, action min/max, config, data hashes, seed,
     parameter counts, git pins. The official checkpoint stores only state_dict + optimizer.

Usage:
  CUDA_VISIBLE_DEVICES=6 python train_amix.py --task pick_egg --split-version v1 --seed 0 \
      --run-name amix_pick_egg_v1_s0 [--num-steps 30000] [--label smoke]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from split_data import build_split_hdf5  # noqa: E402

OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
sys.path.insert(0, OOPSIE)
RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/baseline/train"
CONFIGS = {"pick_egg": "learning/configs/pick_egg.yaml", "wipe_countertop": "learning/configs/wipe_counter.yaml"}


class JsonlWandb:
    """Minimal stand-in for the wandb module used by cfm_trainer (init/log/finish)."""

    def __init__(self, path):
        self.path = path
        self.t0 = None
        self.f = None

    def init(self, **kw):
        self.f = open(self.path, "a")
        self.t0 = time.time()
        self.f.write(json.dumps({"event": "init", "config": kw.get("config"), "time": self.t0}) + "\n")
        self.f.flush()
        return self

    def log(self, d):
        rec = {k: (float(v) if torch.is_tensor(v) else v) for k, v in d.items()}
        rec["wall_s"] = time.time() - self.t0
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def finish(self):
        self.f.write(json.dumps({"event": "finish", "wall_s": time.time() - self.t0}) + "\n")
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--split-version", default="v1", choices=["v1", "draft"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--num-steps", type=int, default=None, help="override (smoke only)")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--label", default="final", help="'final' or 'smoke'")
    ap.add_argument("--no-dev-eval", action="store_true")
    args = ap.parse_args()
    if args.split_version == "draft":
        assert args.label == "smoke", "draft split may only be used for smoke runs"

    os.chdir(OOPSIE)
    import learning.train_eval.cfm_trainer as T
    from learning.dataset import PlaybackDataset
    from learning.models import CFMPolicy, PolicyConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.join(RUNS, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    raw = T.load_yaml_config(CONFIGS[args.task])
    train_path, train_eps, split_meta = build_split_hdf5(args.task, "train", args.split_version)
    dev_path, dev_eps, _ = build_split_hdf5(args.task, "dev", args.split_version)
    raw["data_path"] = train_path
    raw["save_dir"] = os.path.join(run_dir, "ckpt")
    raw["use_wandb"] = True  # routed to the JSONL shim below
    raw["wandb_run_name"] = args.run_name
    if args.num_steps is not None:
        raw["num_steps"] = args.num_steps
    if args.save_every is not None:
        raw["save_every"] = args.save_every
    cfg = T.trainer_config_from_mapping(raw)

    shim = JsonlWandb(os.path.join(run_dir, "metrics.jsonl"))
    T.wandb = shim
    T.WANDB_AVAILABLE = True

    t_data = time.time()
    ds = PlaybackDataset(
        data_path=cfg.data_path, frame_stack=cfg.frame_stack, action_chunk_size=cfg.action_chunk,
        seg_img_size=(cfg.seg_img_size, cfg.seg_img_size), normalize_action=cfg.normalize_action,
        objects_of_interest=cfg.objects_of_interest, obs_keys=cfg.obs_keys)
    dev = None
    if not args.no_dev_eval and dev_eps:
        dev = PlaybackDataset(
            data_path=dev_path, frame_stack=cfg.frame_stack, action_chunk_size=cfg.action_chunk,
            seg_img_size=(cfg.seg_img_size, cfg.seg_img_size), normalize_action=cfg.normalize_action,
            objects_of_interest=cfg.objects_of_interest, obs_keys=cfg.obs_keys)
        assert dev.class_to_id == ds.class_to_id, (dev.class_to_id, ds.class_to_id)
        dev.action_min, dev.action_max = ds.action_min, ds.action_max  # train statistics only
    t_data = time.time() - t_data

    pcfg = PolicyConfig(
        state_dim=ds.state_dim, action_dim=ds.action_dim, num_seg_views=ds.num_seg_views,
        num_seg_classes=max(ds.num_seg_classes, 1), frame_stack=cfg.frame_stack,
        action_chunk_size=cfg.action_chunk, seg_img_height=cfg.seg_img_size, seg_img_width=cfg.seg_img_size,
        action_min=ds.action_min if cfg.normalize_action else None,
        action_max=ds.action_max if cfg.normalize_action else None)
    policy = CFMPolicy(pcfg)
    policy.print_parameter_summary()
    pc = policy.get_parameter_count()

    class Trainer(T.CFMTrainer):
        @torch.no_grad()
        def evaluate(self):
            v = super().evaluate()  # official: on eval_dataset (= train set)
            if dev is not None:
                saved = self.eval_dataset
                self.eval_dataset = dev
                d = super().evaluate()
                self.eval_dataset = saved
                if self.wandb_run:
                    T.wandb.log({"eval/dev_mse_loss": float(d), "eval/train_mse_loss": float(v)})
            return v

    meta = {
        "run_name": args.run_name, "label": args.label, "task": args.task, "seed": args.seed,
        "split": split_meta, "train_episodes": [f"{t}/{c}/{d}" for t, c, d in train_eps],
        "dev_episodes": [f"{t}/{c}/{d}" for t, c, d in dev_eps],
        "train_hdf5": train_path, "dev_hdf5": dev_path,
        "trainer_config": {k: v for k, v in vars(cfg).items()},
        "policy_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(pcfg).items()},
        "class_to_id": ds.class_to_id, "action_min": ds.action_min.tolist(), "action_max": ds.action_max.tolist(),
        "n_train_samples": len(ds), "n_dev_samples": len(dev) if dev is not None else 0,
        "parameter_count": pc, "dataset_build_s": t_data,
        "official_code": {"repo": OOPSIE, "sha": "a739f99e94f32c3229ae6932299e2e0e8c065480"},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "fp32 (official; no autocast)", "ema": "none (official)",
    }
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1, default=str)

    trainer = Trainer(config=cfg, policy=policy, train_dataset=ds, args_dict=meta["trainer_config"])
    t0 = time.time()
    trainer.train(resume_step=0)
    meta["train_wall_s"] = time.time() - t0
    meta["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1, default=str)
    print("TRAIN_DONE", meta["train_wall_s"])


if __name__ == "__main__":
    main()
