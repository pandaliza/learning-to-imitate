"""Stage-1 (decoupled) training of the VL-grounded slot-intent stack.

Trains the slot encoder (frozen-VL agentview grid -> z*) + the flow-map generator
(conditioned on the current-obs frozen-VL mean) on the precomputed VL cache, with NO pi05
model in the loop (VL is precomputed -> cheap). This is the "learn intent first, then
condition" (decoupled) recipe, identical to the M1 frozen-intent setup except the slot
encoder and generator are VL-grounded instead of ResNet18.

Saves intent_stack.pt (slot_encoder + intent_flow_map + vl_proj). Stage-2 then finetunes
the pi05 action head on the FROZEN intent (the existing M1 mechanism); eval samples the
deployable z_hat via mip.pi05_intent.CotrainIntentModule.vl_sample_intent (vl_proj + flow map).

Single-GPU:  python examples/openpi/train_slot_intent_vl.py --out /data/.../slot_intent_vl
Multi-GPU:   torchrun --standalone --nproc_per_node=8 examples/openpi/train_slot_intent_vl.py --out ...
             (the intent stack is replicated per rank; its grads are all-reduced manually.)
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mip.datasets.libero_dataset import make_dataset
from mip.pi05_intent import CotrainIntentModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-task-config", default="libero_goal_suite_image_slot_intent_vl")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=16, help="per-GPU micro-batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmstart", default=None,
                    help="resume Stage-1 from an intent_stack_*.pt (loads slot_encoder/flow_map/vl_proj + step)")
    args = ap.parse_args()

    # --- DDP setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; single-GPU if unset) ---
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

    if is_main:
        os.makedirs(args.out, exist_ok=True)
    cot = CotrainIntentModule(args.slot_task_config, args.config_dir, device=device, vl_obs=True)
    cot.train()
    assert cot.agent.slot_encoder.vl_input, "config must set slot_vl_input=true"

    ds = make_dataset(cot.cfg.task)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    # Materialise vl_proj (LazyLinear infers VL width on first call) before the optimizer.
    cot.vl_proj(torch.zeros(1, int(cot.cfg.task.slot_vl_dim), device=device))
    start_step = 0
    if args.warmstart:  # resume Stage-1: load slot encoder + flow map + vl_proj and continue from its step
        _ws = torch.load(args.warmstart, map_location=device, weights_only=False)
        cot.agent.slot_encoder.load_state_dict(_ws["slot_encoder"])
        cot.agent.intent_flow_map.load_state_dict(_ws["intent_flow_map"])
        cot.vl_proj.load_state_dict(_ws["vl_proj"])
        start_step = int(_ws.get("step", 0))
        if is_main:
            print(f"[resume] {args.warmstart} -> start_step {start_step}", flush=True)
    params = cot.train_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def save(tag, step):
        if not is_main:
            return
        torch.save({
            "slot_encoder": cot.agent.slot_encoder.state_dict(),
            "intent_flow_map": cot.agent.intent_flow_map.state_dict(),
            "vl_proj": cot.vl_proj.state_dict(),
            "step": step,
        }, os.path.join(args.out, f"intent_stack_{tag}.pt"))

    step, best, ema, epoch = start_step, float("inf"), None, 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            grids = batch["intent_vl_grids"].to(device, non_blocking=True)   # (B,K,256,2048)
            frames = batch["intent_frames"].to(device, non_blocking=True)    # (B,K,3,128,128) recon tgt
            obj = batch["object_states"].to(device, non_blocking=True)       # (B,K,6)
            vlm = batch["vl_obs_mean"].to(device, non_blocking=True)         # (B,2048)
            delta_t = torch.zeros(grids.shape[0], device=device)
            _, losses = cot.vl_intent_and_losses(frames, obj, vlm, delta_t, intent_grids=grids)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            if ddp:  # the intent stack isn't DDP-wrapped -> all-reduce its grads manually
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad /= world_size
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            lv = float(loss.detach())
            ema = lv if ema is None else 0.99 * ema + 0.01 * lv
            if is_main and step % 100 == 0:
                msg = " ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
                print(f"step {step}: {msg} total={lv:.4f} ema={ema:.4f}", flush=True)
            if is_main and step > 0 and step % args.save_every == 0:
                save(str(step), step)
                if ema < best:
                    best = ema
                    save("best", step)
            step += 1
            if step >= args.steps:
                break
    save("final", step)
    if is_main:
        print(f"[done] {step} steps, best_ema={best:.4f} -> {args.out}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
