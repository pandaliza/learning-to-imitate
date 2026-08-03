"""M11 trainer: intent-action CO-PREDICTION on pi0.5 action expert for RoboCasa (docs/co_prediction.md).

Fork of train_pi05_m10.py per the arm convention (live runs train out of the originals). Ported to
RoboCasa LeRobot dataset (NVIDIA PhysicalAI Kitchen, PandaOmron):
  - state 16D, action 12D (padded to 32)
  - 2 cameras (agentview_left -> base slot, eye_in_hand -> wrist slot)
  - H=10 action horizon
  - intent: h=8 eef-relative waypoints, d_I=7 (pos3 + quat4, vs LIBERO's 6)

Uses RobocasaCopredDataset when available (steer_intent/robocasa_copred_dataset.py), or a fake
dataset for smoke testing. Dataset must provide the same batch interface as IntentFlowDataset:
  - "obs": {"state": (B, To, 16), "agentview_rgb": (B, To, 3, H, W), "eye_in_hand_rgb": (B, To, 3, H, W)}
  - "action": (B, A, 12)
  - "wsm_intent_target": (B, h, 7)
  - "task_id": (B,)
  - normalizers for both

Arms (docs/co_prediction.md section 6):
  A0  pure BC (no intent anything)           python train_pi05_m11.py --intent-weight 0.0 --train-intent-only False
  A1  tied      tau_I = tau_A                --tied
  A2/A3/A4 decoupled (one ckpt, three schedules at eval)
  B1  adaLN     --copred-mask b1
  xattn        --copred-mask xattn

  python examples/openpi/train_pi05_m11.py \\
    --pi05-config pi05_robocasa_copred \\
    --slot-task-config libero_goal_suite_image_slot_intent_vl \\
    --pi05-weights .../pi05_base_pytorch \\
    --intent-weight 1.0 --lookahead-stride 2 \\
    --steps 30000 --batch-size 32 --accum-steps 1 --out .../pi05_m11_copred
"""

import argparse
import os
import re
import shutil

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from steer_intent.intent_flow_dataset import make_intent_flow_dataset


def _set_lora_requires_grad(model):
    """LoRA-VL + full action head (incl. the fresh copred intent projections): freeze ONLY the
    PaliGemma LLM base. Mirrors train_pi05_m10.py."""
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        is_llm_base = ("language_model" in name) and ("lora_" not in name)
        p.requires_grad_(not is_llm_base)
        n_train += 0 if is_llm_base else p.numel()
        n_freeze += p.numel() if is_llm_base else 0
    print(f"[freeze-lora] trainable={n_train/1e6:.1f}M  frozen(LLM base)={n_freeze/1e6:.1f}M")


def _build_pi05_transforms(pi05_config):
    data_config = pi05_config.data.create(pi05_config.assets_dirs, pi05_config.model)
    norm_stats = data_config.norm_stats
    # openpi returns None SILENTLY when assets/<config>/<asset_id>/norm_stats.json is missing, and
    # Normalize(None) is a no-op -> the model trains on RAW states/actions while eval normalizes
    # (train/eval mismatch, ~0% SR). This exact failure burned the first M10 launch (2026-07-20).
    # UNLESS --test-fake-dataset is set (smoke test), in which case we bypass this guard.
    if norm_stats is None and not os.environ.get("TEST_FAKE_DATASET"):
        raise FileNotFoundError(
            f"norm stats missing for config '{pi05_config.name}' -- expected under "
            f"{pi05_config.assets_dirs}. Symlink assets/{pi05_config.name} -> pi05_base_nointent."
        )
    return _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_robocasa_copred")
    ap.add_argument("--slot-task-config", default="libero_goal_suite_image_slot_intent_vl")
    ap.add_argument("--pi05-weights", required=True, help="pi05_base PyTorch params dir (safetensors)")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    # --- co-prediction knobs (docs/co_prediction.md section 3) ---
    ap.add_argument("--intent-weight", type=float, default=1.0, help="w_I on the intent flow-matching loss")
    ap.add_argument("--train-intent-only", type=lambda x: x.lower() != "false", default=True,
                    help="False to run A0 pure-BC (no intent gradient); True for copred training")
    ap.add_argument("--tied", action="store_true", help="A1 ablation: tau_I = tau_A (no decoupling)")
    ap.add_argument("--strat-clean-p", type=float, default=0.25,
                    help="P(force tau_I = 0): clean-intent conditioning cell (intent loss masked)")
    ap.add_argument("--strat-noise-p", type=float, default=0.10,
                    help="P(force tau_I = 1): uninformative-intent cell (prevents over-reliance)")
    ap.add_argument("--lookahead-stride", type=int, default=2,
                    help="Delta: I_k = eef(t + k*Delta); 2 reaches t+16, past the H=10 chunk")
    ap.add_argument("--copred-mask", default=None, choices=["j", "t", "b1", "xattn"],
                    help="override the config's suffix mask variant (C1: t; B1 adaLN0: b1; "
                         "gated cross-attn: xattn)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--test-fake-dataset", action="store_true",
                    help="Smoke test: use a fake dataset instead of real RobocasaCopredDataset")
    args = ap.parse_args()

    # Set env var for the norm-stats guard to skip when testing with fake dataset
    if args.test_fake_dataset:
        os.environ["TEST_FAKE_DATASET"] = "1"

    # --- DDP setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK); single-GPU if unset ---
    ddp = "RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        device = f"cuda:{local_rank}"
    else:
        local_rank, world_size, rank, device = 0, 1, 0, args.device
    is_main = rank == 0

    # --- Pi0.5 model (PyTorch, copred_h > 0) ---
    pi05_config = _config.get_config(args.pi05_config)
    if args.copred_mask:
        import dataclasses
        pi05_config = dataclasses.replace(
            pi05_config, model=dataclasses.replace(pi05_config.model, copred_mask=args.copred_mask))
    copred_h = int(pi05_config.model.copred_h)
    copred_intent_dim = int(pi05_config.model.copred_intent_dim)
    assert copred_h > 0, f"{args.pi05_config} must set copred_h > 0"
    model = PI0Pytorch(pi05_config.model).to(device)
    import safetensors.torch
    start_step = 0
    if args.resume_from:
        safetensors.torch.load_model(model, os.path.join(args.resume_from, "model.safetensors"), strict=False)
        start_step = int(os.path.basename(os.path.normpath(args.resume_from)))
        print(f"[resume] loaded {args.resume_from} -> start_step {start_step}", flush=True)
    else:
        # strict=False: the base ckpt has no intent_in_proj / intent_out_proj (fresh, out zero-init).
        safetensors.torch.load_model(model, os.path.join(args.pi05_weights, "model.safetensors"), strict=False)
    if "lora" in pi05_config.model.paligemma_variant:
        _set_lora_requires_grad(model)
    else:
        raise NotImplementedError("M11 recipe is LoRA-VL + full action head (pi05_robocasa_copred)")
    if os.environ.get("FORCE_FP32"):
        model = model.float()
        print("[debug] model forced to float32", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    # --- Dataset: RobocasaCopredDataset when available, or fake for smoke test ---
    if args.test_fake_dataset:
        # Smoke test with fake dataset: no Hydra config needed
        ds = _make_fake_robocasa_dataset(copred_h, copred_intent_dim)
        if is_main:
            print(f"[FAKE-DATASET] h={copred_h} d_I={copred_intent_dim}", flush=True)
    else:
        # Real training: RobocasaCopredDataset (D1 provides this)
        try:
            from steer_intent.robocasa_copred_dataset import RobocasaCopredDataset
            # D1 builds this with the same batch interface as IntentFlowDataset
            ds = RobocasaCopredDataset(
                root_dir=os.environ.get(
                    "ROBOCASA_DATA_ROOT",
                    "/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target"),
                norm_stats_path="assets/pi05_robocasa_copred/robocasa/norm_stats.json",
                action_horizon=int(pi05_config.model.action_horizon),
                lookahead_stride=args.lookahead_stride,
                intent_horizon=copred_h * args.lookahead_stride,
            )
        except ImportError:
            raise ImportError(
                "RobocasaCopredDataset not found. D1 agent should provide steer_intent/robocasa_copred_dataset.py "
                "before running real training. Use --test-fake-dataset for smoke testing."
            )

    # Task langs for prompts: RoboCasa task names from the dataset's task-id map
    # ("TurnOnElectricKettle" -> "turn on electric kettle"); no LIBERO Hydra config involved.
    if not args.test_fake_dataset:
        _id2name = {v: k for k, v in ds._task_id_map.items()}
        _task_langs = [re.sub(r"(?<!^)(?=[A-Z])", " ", _id2name[i]).lower()
                       for i in range(len(_id2name))]
    else:
        # Fake dataset: synthesize task langs (fake task_id is idx % 10, so cover all 10)
        _task_langs = [f"task_{i}" for i in range(10)]

    if is_main:
        print(f"[prompts] {len(_task_langs)} per-task prompts, e.g. '{_task_langs[0]}'", flush=True)

    # Probe dataset to verify shapes
    _probe = ds[0]["wsm_intent_target"]
    assert _probe.shape[0] == copred_h, f"waypoints {_probe.shape[0]} != copred_h {copred_h}"
    assert _probe.shape[1] == copred_intent_dim, \
        f"intent dim {_probe.shape[1]} != copred_intent_dim {copred_intent_dim}"
    if is_main:
        print(f"[M11-COPRED-ROBOCASA] h={copred_h} d_I={copred_intent_dim} stride={args.lookahead_stride} "
              f"mask={pi05_config.model.copred_mask} tied={args.tied} w_I={args.intent_weight} "
              f"train_intent={args.train_intent_only} strat=({args.strat_clean_p},{args.strat_noise_p})", flush=True)

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    try:
        _avail_cpus = len(os.sched_getaffinity(0))  # respects slurm/cgroup CPU limits
    except AttributeError:
        _avail_cpus = os.cpu_count() or 8
    n_workers = 0 if args.test_fake_dataset else min(8, _avail_cpus)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=n_workers, drop_last=True)
    tfm = _build_pi05_transforms(pi05_config)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    for _ in range(start_step):
        sched.step()

    obs_steps = int(ds.obs_steps)
    action_horizon = int(pi05_config.model.action_horizon)

    def build_observation(batch):
        """Per-sample openpi transform of the Pi0.5 fields, then collate -> Observation, actions.
        Same un-normalize dance as train_pi05_m10.py (openpi's Normalize must run exactly once)."""
        st = ds.normalizer["obs"]["state"].unnormalize(batch["obs"]["state"].numpy())
        agv = batch["obs"]["agentview_rgb"].numpy()
        wr = batch["obs"]["eye_in_hand_rgb"].numpy()
        act = ds.normalizer["action"].unnormalize(batch["action"].numpy())
        tid = batch["task_id"].numpy()
        B = st.shape[0]
        elems = []
        cur = obs_steps - 1
        for i in range(B):
            el = {
                "image": (agv[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
                "wrist_image": (wr[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
                "state": st[i, cur].astype(np.float32),  # RoboCasa: full 16D state (stats are 16-dim; openpi Normalize pads)
                "actions": act[i, :action_horizon].astype(np.float32),
                "prompt": _task_langs[int(tid[i])],
            }
            elems.append(tfm(el))
        coll = {k: np.stack([e[k] for e in elems]) if not isinstance(elems[0][k], dict)
                else {kk: np.stack([e[k][kk] for e in elems]) for kk in elems[0][k]}
                for k in elems[0]}

        def _to_t(v):
            t = torch.as_tensor(v, device=device)
            return t.float() if t.is_floating_point() else t

        obs = _model.Observation.from_dict({
            k: (_to_t(v) if not isinstance(v, dict) else {kk: _to_t(vv) for kk, vv in v.items()})
            for k, v in coll.items()})
        actions = _to_t(coll["actions"])
        return obs, actions

    step = start_step
    accum = max(1, args.accum_steps)
    micro = 0
    epoch = 0
    opt.zero_grad()
    last = {}
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            observation, actions = build_observation(batch)
            itgt = batch["wsm_intent_target"].to(device).float()  # (B, h, d_I), eef-normalized

            B = actions.shape[0]
            time = raw_model.sample_time(B, actions.device)

            # A0 pure-BC mode: skip intent gradient entirely (zero intent weight, no intent loss)
            if not args.train_intent_only:
                # A0: forward with no intent targets -> no intent loss, actions trained as pure BC
                action_loss = model(observation, actions, time=time)
                if isinstance(action_loss, tuple):
                    action_loss = action_loss[0]  # Extract action loss from (action, intent) tuple
                action_l = action_loss.mean()
                intent_l = torch.tensor(0.0, device=device)
                total = action_l
            else:
                # Copred training (A1/A2/A3/etc): compute both losses
                if args.tied:
                    intent_time = time.clone()  # A1: one shared tau, no decoupling
                else:
                    # Independent draw + stratified forcing (section 3.1). sample_time is in
                    # [0.001, 1.0), so forced 0/1 mark the two stratified cells unambiguously.
                    intent_time = raw_model.sample_time(B, actions.device)
                    u = torch.rand(B, device=actions.device)
                    intent_time = torch.where(u < args.strat_clean_p, torch.zeros_like(intent_time), intent_time)
                    intent_time = torch.where(u > 1.0 - args.strat_noise_p, torch.ones_like(intent_time), intent_time)

                action_loss, intent_loss = model(
                    observation, actions, time=time, intent_targets=itgt, intent_time=intent_time)
                action_l = action_loss.mean()
                # Mask the forced tau_I=0 cell: nothing to denoise there, the intent tokens acted
                # purely as (clean) conditioning. .mean() per block = the per-dim normalization of 3.2.
                m = (intent_time > 0).float()
                intent_l = (intent_loss.mean(dim=(1, 2)) * m).sum() / m.sum().clamp(min=1.0)
                total = action_l + args.intent_weight * intent_l

            (total / accum).backward()
            last = {"action": float(action_l), "intent": float(intent_l), "total": float(total)}
            micro += 1
            if micro % accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

            if is_main and step % int(os.environ.get("PRINT_EVERY", 100)) == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in last.items())
                print(f"step {step}: {msg}", flush=True)
            if is_main and step > 0 and step % args.save_every == 0:
                d = os.path.join(args.out, str(step)); os.makedirs(d, exist_ok=True)
                # Everything (incl. intent projections) lives in the model -- single artifact.
                safetensors.torch.save_model(raw_model, os.path.join(d, "model.safetensors"))
                _stats_dst = os.path.join(d, "assets", "robocasa")
                os.makedirs(_stats_dst, exist_ok=True)
                shutil.copy("assets/pi05_robocasa_copred/robocasa/norm_stats.json", _stats_dst)
                print(f"[ckpt] step {step} -> {d}", flush=True)
                kept = sorted(int(x) for x in os.listdir(args.out) if x.isdigit())
                for old in kept[:-3]:
                    shutil.rmtree(os.path.join(args.out, str(old)), ignore_errors=True)
            step += 1
            if step >= args.steps:
                break

    # Save final checkpoint on exit (fix open item 5: off-by-one in original m10)
    if is_main and step > start_step:
        d = os.path.join(args.out, str(step)); os.makedirs(d, exist_ok=True)
        safetensors.torch.save_model(raw_model, os.path.join(d, "model.safetensors"))
        # Replicate norm stats into the checkpoint so openpi's create_trained_policy finds them
        _stats_dst = os.path.join(d, "assets", "robocasa")
        os.makedirs(_stats_dst, exist_ok=True)
        shutil.copy("assets/pi05_robocasa_copred/robocasa/norm_stats.json", _stats_dst)
        print(f"[ckpt-final] step {step} -> {d}", flush=True)
        if torch.cuda.is_available():
            print(f"[mem] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    if ddp:
        dist.destroy_process_group()


def _make_fake_robocasa_dataset(copred_h, copred_intent_dim):
    """Fake RoboCasa dataset for smoke testing (no real data, random tensors with correct shapes).

    Batch interface matches RobocasaCopredDataset (built by D1):
      - "obs": {"state": (B, To, 16), "agentview_rgb": (B, To, 3, H, W), "eye_in_hand_rgb": (B, To, 3, H, W)}
      - "action": (B, A, 12)
      - "wsm_intent_target": (B, copred_h, copred_intent_dim)
      - "task_id": (B,)
      - normalizer: {"obs": {"state": ...}, "action": ...} -- no-op normalizers
    """
    class FakeRobocasaDataset:
        def __init__(self, copred_h, copred_intent_dim, size=128, obs_steps=2, action_steps=10):
            self.copred_h = copred_h
            self.copred_intent_dim = copred_intent_dim
            self.size = size
            self.obs_steps = obs_steps
            self.action_steps = action_steps

            # No-op normalizers (smoke test passes through)
            class NoOpNorm:
                def normalize(self, x):
                    return x
                def unnormalize(self, x):
                    return x

            self.normalizer = {
                "obs": {"state": NoOpNorm()},
                "action": NoOpNorm(),
            }

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            B_fake = 1  # Single sample; will be batched by DataLoader
            return {
                "obs": {
                    "state": torch.randn(self.obs_steps, 16, dtype=torch.float32),  # RoboCasa: 16D state
                    "agentview_rgb": torch.rand(self.obs_steps, 3, 224, 224, dtype=torch.float32),
                    "eye_in_hand_rgb": torch.rand(self.obs_steps, 3, 224, 224, dtype=torch.float32),
                },
                "action": torch.randn(self.action_steps, 12, dtype=torch.float32),  # RoboCasa: 12D action
                "wsm_intent_target": torch.randn(self.copred_h, self.copred_intent_dim, dtype=torch.float32),
                "task_id": torch.tensor(idx % 10, dtype=torch.long),
            }

    return FakeRobocasaDataset(copred_h, copred_intent_dim)


if __name__ == "__main__":
    main()
