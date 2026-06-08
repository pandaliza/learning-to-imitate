"""End-to-end co-training: Pi0.5 action head + MIP slot-intent stack (PyTorch path).

Joint training of:
  - MIP slot encoder (future frames -> intent) + flow map (obs -> intent for eval)
  - Pi0.5's action expert (consumes intent via the adaRMS hook in pi0_pytorch.py)
under one optimizer:  action_loss + flow_loss + aux_loss + recon_loss.

Data source is MIP's LiberoDataset (slot config) — it natively yields the MIP-format
obs (15D state + 128px images), `intent_frames`, `object_states`, and `action` that the
CotrainIntentModule needs; Pi0.5's observation is derived from the same batch and run
through openpi's transforms (LiberoInputs/Normalize/tokenize/resize/pad).

STATUS: first draft — needs a debug run to shake out the openpi transform/Observation
+ PyTorch freeze/weight-load integration (like the decoupled path did). See
docs/pi05_slotintent_finetuning.md.

Usage:
  python examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_libero_intent \
    --slot-task-config libero_goal_suite_image_slot_intent \
    --warmstart /data/.../slot_intent/models/model_best.pt \
    --steps 30000 --batch-size 32 --out /data/.../openpi-checkpoints/pi05_cotrain
"""

import argparse
import os
import re

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from mip.datasets.libero_dataset import make_dataset
from mip.pi05_intent import CotrainIntentModule

# Train the action expert ('_1' gemma params) + suffix projections (incl. intent_proj);
# freeze SigLIP (img) + PaliGemma expert-0. Mirrors get_freeze_filter_action_head_only().
_TRAINABLE = re.compile(r"(llm.*_1.*)|(action_in_proj)|(action_out_proj)|(time_mlp)|(intent_proj)")
_FROZEN = re.compile(r"(img\.)|(llm)")  # frozen unless matched by _TRAINABLE


def _set_action_head_only_requires_grad(model):
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        train = bool(_TRAINABLE.search(name)) or (not _FROZEN.search(name))
        p.requires_grad_(train)
        n_train += p.numel() if train else 0
        n_freeze += 0 if train else p.numel()
    print(f"[freeze] trainable={n_train/1e6:.1f}M  frozen={n_freeze/1e6:.1f}M")


def _build_pi05_transforms(pi05_config):
    """openpi input transforms that turn a libero element dict -> Observation fields."""
    data_config = pi05_config.data.create(pi05_config.assets_dirs, pi05_config.model)
    norm_stats = data_config.norm_stats
    return _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,       # LiberoInputs (intent passthrough)
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,       # tokenize prompt, resize 224, pad to 32
    ])


def _prompt_for(task_config, batch_idx):
    # MIP LiberoDataset doesn't carry language; derive a generic prompt per suite.
    # (Refine to per-task once dataset carries task_id/language.)
    return "complete the task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_libero_intent")
    ap.add_argument("--slot-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--warmstart", default=None, help="Stage-1 slot ckpt to warm-start the intent stack")
    ap.add_argument("--pi05-weights", required=True, help="pi05_libero params dir (safetensors) to init Pi0.5")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=32, help="per-step micro-batch")
    ap.add_argument("--accum-steps", type=int, default=1, help="grad accumulation; effective batch = batch_size * accum_steps")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # --- DDP setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK); single-GPU if unset ---
    ddp = "RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)          # claim this rank's GPU BEFORE NCCL init
        dist.init_process_group(backend="nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        device = f"cuda:{local_rank}"
    else:
        local_rank, world_size, rank, device = 0, 1, 0, args.device
    is_main = rank == 0

    # --- Pi0.5 model (PyTorch) ---
    pi05_config = _config.get_config(args.pi05_config)
    model = PI0Pytorch(pi05_config.model).to(device)
    import safetensors.torch
    safetensors.torch.load_model(model, os.path.join(args.pi05_weights, "model.safetensors"), strict=False)
    _set_action_head_only_requires_grad(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()  # fit a 3B model on one GPU
    model.train()
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    # --- MIP slot-intent stack (co-trained); replicated per rank, grads all-reduced manually ---
    cotrain = CotrainIntentModule(args.slot_task_config, args.config_dir, device=device, warmstart_ckpt=args.warmstart)
    cotrain.train()
    obs_steps = int(cotrain.cfg.task.obs_steps)  # MIP encoder consumes the first obs_steps frames
    action_horizon = int(pi05_config.model.action_horizon)  # Pi0.5 predicts this many steps (10); MIP chunk is 16

    # --- Data: MIP LiberoDataset (slot) gives obs / intent_frames / object_states / action ---
    ds = make_dataset(cotrain.cfg.task)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=8, drop_last=True)
    tfm = _build_pi05_transforms(pi05_config)

    # --- Optimizer over trainable Pi0.5 params + the whole intent stack ---
    params = [p for p in model.parameters() if p.requires_grad] + cotrain.train_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def build_observation(batch, intent):
        """Per-sample openpi transform of the Pi0.5 fields, then collate -> Observation, actions."""
        st = batch["obs"]["state"].numpy()            # (B, To, 15) normalized
        agv = batch["obs"]["agentview_rgb"].numpy()    # (B, To, C, H, W) in [0,1]
        wr = batch["obs"]["eye_in_hand_rgb"].numpy()
        act = batch["action"].numpy()                  # (B, horizon, 7)
        intent_np = intent.detach().cpu().numpy()
        B = st.shape[0]
        elems = []
        cur = obs_steps - 1  # current-obs frame (last of the obs window, not a future frame)
        for i in range(B):
            # Raw LeRobot keys — the RepackTransform (first in the chain) remaps these to observation/*.
            elems.append(tfm({
                "image": (agv[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),  # current frame HWC
                "wrist_image": (wr[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
                "state": st[i, cur, :8].astype(np.float32),   # ee(6)+gripper(2)
                "actions": act[i, :action_horizon].astype(np.float32),  # Pi0.5 horizon (10), not MIP's 16
                "prompt": _prompt_for(cotrain.cfg.task, i),
                "intent": intent_np[i].astype(np.float32),
            }))
        coll = {k: np.stack([e[k] for e in elems]) if not isinstance(elems[0][k], dict)
                else {kk: np.stack([e[k][kk] for e in elems]) for kk in elems[0][k]}
                for k in elems[0]}

        def _to_t(v):  # float arrays -> float32 (model is float32); keep int/bool dtypes
            t = torch.as_tensor(v, device=device)
            return t.float() if t.is_floating_point() else t

        obs = _model.Observation.from_dict({
            k: (_to_t(v) if not isinstance(v, dict) else {kk: _to_t(vv) for kk, vv in v.items()})
            for k, v in coll.items()})
        actions = _to_t(coll["actions"])
        return obs, actions

    step = 0          # optimizer steps (effective-batch updates), matches the other arms' step count
    accum = max(1, args.accum_steps)
    micro = 0
    epoch = 0
    delta_t = torch.zeros(args.batch_size, device=device)  # flow-matching warmup delta (0 = base)
    opt.zero_grad()
    last = {}
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            intent_frames = batch["intent_frames"].to(device)
            object_states = batch["object_states"].to(device)
            obs_mip = {k: v[:, :obs_steps].to(device) for k, v in batch["obs"].items()}
            intent, ilosses = cotrain.intent_and_losses(intent_frames, object_states, obs_mip, delta_t)

            observation, actions = build_observation(batch, intent)
            action_loss = model(observation, actions)
            if isinstance(action_loss, (list, tuple)):
                action_loss = action_loss[0]
            action_loss = action_loss.mean()

            total = action_loss + ilosses["flow"] + args.aux_weight * ilosses["aux"]
            if "recon" in ilosses:
                total = total + ilosses["recon"]

            (total / accum).backward()  # scale so accumulated grad ~ effective-batch grad
            last = {"action": float(action_loss), **{k: float(v) for k, v in ilosses.items()}, "total": float(total)}
            micro += 1
            if micro % accum != 0:
                continue  # keep accumulating; one optimizer step per `accum` micro-batches

            # Pi0.5 grads are auto-synced by DDP on backward; manually all-reduce the
            # intent-stack grads (it's not DDP-wrapped — called via MIP agent methods).
            if ddp:
                for p in cotrain.train_parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad /= world_size
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

            if is_main and step % 100 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in last.items())
                print(f"step {step}: {msg}", flush=True)
            if is_main and step > 0 and step % args.save_every == 0:
                d = os.path.join(args.out, str(step)); os.makedirs(d, exist_ok=True)
                safetensors.torch.save_model(raw_model, os.path.join(d, "model.safetensors"))
                torch.save({  # actual intent-stack weights (so future runs can resume)
                    "encoder": cotrain.agent.encoder.state_dict(),
                    "slot_encoder": cotrain.agent.slot_encoder.state_dict(),
                    "intent_flow_map": cotrain.agent.intent_flow_map.state_dict(),
                    "step": step,
                }, os.path.join(d, "intent_stack.pt"))
                print(f"[ckpt] step {step} -> {d}", flush=True)
            step += 1
            if step >= args.steps:
                break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
