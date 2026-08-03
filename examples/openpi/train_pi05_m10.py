"""M10 trainer: intent-action CO-PREDICTION in the pi0.5 action expert (docs/co_prediction.md).

Fork of train_pi05_m9.py per the arm convention (live runs train out of the originals). Unlike
M1-M9 this arm has NO auxiliary head: the h=8 intent waypoint tokens live inside PI0Pytorch
(copred_h > 0), are denoised jointly with the action chunk under an INDEPENDENT noise level, and
ship in model.safetensors. Deploy graph therefore includes the intent tokens -- eval with
eval_libero_intent.py --schedule {s1,s2,s3}, NOT the baseline no-intent path.

Arms (docs/co_prediction.md section 6):
  A1 tied      --tied              tau_I = tau_A every sample (co-training alone)
  A2/A3/A4     (default)           decoupled + stratified; one checkpoint, three eval schedules
  C1 two-stream --copred-mask t
  B1 adaLN     --copred-mask b1    intent -> pooled adaRMS on action tokens, no token attention
  xattn        --copred-mask xattn intent -> gated cross-attn on action tokens (vs B1 adaLN0)

  python examples/openpi/train_pi05_m10.py \
    --pi05-config pi05_base_copred \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights .../pi05_base_pytorch \
    --intent-weight 1.0 --lookahead-stride 2 \
    --steps 30000 --batch-size 2 --accum-steps 4 --out .../pi05_m10_copred
"""

import argparse
import os
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
    PaliGemma LLM base. Mirrors train_pi05_m9.py."""
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
    if norm_stats is None:
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
    ap.add_argument("--pi05-config", default="pi05_base_copred")
    ap.add_argument("--slot-task-config", default="libero_goal_suite_image_slot_intent_vl")
    ap.add_argument("--pi05-weights", required=True, help="pi05_base PyTorch params dir (safetensors)")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    # --- co-prediction knobs (docs/co_prediction.md section 3) ---
    ap.add_argument("--intent-weight", type=float, default=1.0, help="w_I on the intent flow-matching loss")
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
    args = ap.parse_args()

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
        raise NotImplementedError("M10 recipe is LoRA-VL + full action head (pi05_base_copred)")
    if os.environ.get("FORCE_FP32"):
        model = model.float()
        print("[debug] model forced to float32", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    # --- Task config (hydra) -> dataset with strided intent targets ---
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(args.config_dir)):
        cfg = compose(config_name="main", overrides=[f"task={args.slot_task_config}", "network=mlp_flow_intent"])
    with open_dict(cfg.task):
        # intent_horizon is the REACH in env steps (sizes the sample window upstream);
        # h waypoints x stride Delta must fit inside it.
        cfg.task.task_id_conditioning = True
        cfg.task.intent_horizon = copred_h * args.lookahead_stride
    _task_langs = [os.path.basename(p).replace("_demo.hdf5", "").replace(".hdf5", "").replace("_", " ")
                   for p in cfg.task.dataset_paths]
    if is_main:
        print(f"[prompts] {len(_task_langs)} per-task prompts, e.g. '{_task_langs[0]}'", flush=True)

    ds = make_intent_flow_dataset(cfg.task, intent_target_mode="traj", lookahead_stride=args.lookahead_stride)
    _probe = ds[0]["wsm_intent_target"]
    assert _probe.shape[0] == copred_h, f"waypoints {_probe.shape[0]} != copred_h {copred_h}"
    assert _probe.shape[1] == int(pi05_config.model.copred_intent_dim), \
        f"eef dim {_probe.shape[1]} != copred_intent_dim {pi05_config.model.copred_intent_dim}"
    if is_main:
        print(f"[M10-COPRED] h={copred_h} d_I={_probe.shape[1]} stride={args.lookahead_stride} "
              f"mask={pi05_config.model.copred_mask} tied={args.tied} w_I={args.intent_weight} "
              f"strat=({args.strat_clean_p},{args.strat_noise_p})", flush=True)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=8, drop_last=True)
    tfm = _build_pi05_transforms(pi05_config)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    for _ in range(start_step):
        sched.step()

    obs_steps = int(cfg.task.obs_steps)
    action_horizon = int(pi05_config.model.action_horizon)

    def build_observation(batch):
        """Per-sample openpi transform of the Pi0.5 fields, then collate -> Observation, actions.
        Same un-normalize dance as train_pi05_m9.py (openpi's Normalize must run exactly once)."""
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
                "state": st[i, cur, :8].astype(np.float32),
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

            if is_main and step % 100 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in last.items())
                print(f"step {step}: {msg}", flush=True)
            if is_main and step > 0 and step % args.save_every == 0:
                d = os.path.join(args.out, str(step)); os.makedirs(d, exist_ok=True)
                # Everything (incl. intent projections) lives in the model -- single artifact.
                safetensors.torch.save_model(raw_model, os.path.join(d, "model.safetensors"))
                print(f"[ckpt] step {step} -> {d}", flush=True)
                kept = sorted(int(x) for x in os.listdir(args.out) if x.isdigit())
                for old in kept[:-3]:
                    shutil.rmtree(os.path.join(args.out, str(old)), ignore_errors=True)
            step += 1
            if step >= args.steps:
                break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
