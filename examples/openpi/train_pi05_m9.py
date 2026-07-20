"""M9 trainer: pi0.5 + the STRIPPED intent head (no workspace token, no CFG, no JEPA).

Fork of train_pi05_cotrain.py, carrying every arm it had (M1-M8) plus M9 (--intent-flow). It is a copy
rather than an edit because a live M8 run trains out of the original and would pick up changes on
resume; once the M9 arm settles the two should be folded back together.

M9 in one line: M8 minus w. Same penultimate tap, same future-EEF target family, same train-time-only
contract -- but no stage-1 encoder, no MolmoPoint labels, no w precompute, no CFG conditioning, no JEPA.
It tests whether the aux SUPERVISION alone shapes the trunk (the workspace machinery being incidental in
a memory-free setting like LIBERO), and it is the arm that ports to robocasa / libero-long unmodified.
Deploy graph is baseline pi0.5 -- eval with eval_libero_intent.py WITHOUT --intent, same as M8.

  python examples/openpi/train_pi05_m9.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights .../pi05_base_pytorch \
    --intent-flow --intent-flow-decoder flow --intent-flow-weight 0.25 \
    --steps 30000 --batch-size 2 --accum-steps 4 --out .../pi05_m9_flow

--- original docstring ---

End-to-end co-training: Pi0.5 action head + MIP slot-intent stack (PyTorch path).

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
import dataclasses
import os
import re
import shutil

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
from steer_intent.intent_head import WSMIntentHead
from steer_intent.intent_flow_dataset import make_intent_flow_dataset
from steer_intent.intent_flow_head import IntentFlowHead

# Train the action expert ('_1' gemma params) + suffix projections (incl. intent_proj);
# freeze SigLIP (img) + PaliGemma expert-0. Mirrors get_freeze_filter_action_head_only().
_TRAINABLE = re.compile(r"(llm.*_1.*)|(action_in_proj)|(action_out_proj)|(time_mlp)|(intent_proj)")
_FROZEN = re.compile(r"(img\.)|(llm)")  # frozen unless matched by _TRAINABLE


def _sigreg(z, eps=1e-4):
    """SIGReg-style collapse regularizer (VICReg variance+covariance form). With a joint/symmetric
    JEPA alignment, both embedding heads could collapse to a constant; this keeps the batch
    embedding full-rank (unit-variance dims + decorrelated dims). z: (B, D) -> scalar."""
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    var_term = torch.relu(1.0 - std).mean()               # push each dim toward unit variance
    B, D = z.shape
    cov = (z.T @ z) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_term = (off ** 2).sum() / D                        # decorrelate dimensions (off-diagonal)
    return var_term + cov_term


def _set_action_head_only_requires_grad(model):
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        train = bool(_TRAINABLE.search(name)) or (not _FROZEN.search(name))
        p.requires_grad_(train)
        n_train += p.numel() if train else 0
        n_freeze += 0 if train else p.numel()
    print(f"[freeze] trainable={n_train/1e6:.1f}M  frozen={n_freeze/1e6:.1f}M")


def _set_lora_requires_grad(model):
    """LoRA-VL + full action head: freeze ONLY the PaliGemma LLM base (the 2B gemma);
    train LoRA adapters + SigLIP vision tower + projectors + action expert + suffix
    projections. Mirrors the JAX get_freeze_filter() for gemma_2b_lora (freezes the gemma
    base, leaves img/lora/expert trainable)."""
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        # The PaliGemma LLM base = the language_model weights that are NOT lora adapters.
        is_llm_base = ("language_model" in name) and ("lora_" not in name)
        p.requires_grad_(not is_llm_base)
        n_train += 0 if is_llm_base else p.numel()
        n_freeze += p.numel() if is_llm_base else 0
    print(f"[freeze-lora] trainable={n_train/1e6:.1f}M  frozen(LLM base)={n_freeze/1e6:.1f}M")


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
    # VL-grounded intent (Phase 1): flow map conditions on the live LoRA-VL mean image-token
    # feature (tapped from the Pi0.5 prefix) instead of the MIP ResNet18 obs encoder.
    ap.add_argument("--vl-intent", action="store_true",
                    help="condition the intent flow map on the tapped LoRA-VL obs feature")
    # Gradient knobs (default = V1): detached z_hat into the action head + stop-grad VL tap.
    ap.add_argument("--intent-attach", action="store_true",
                    help="V2/V3: keep z_hat attached into the action head (action loss shapes the flow map)")
    ap.add_argument("--vl-grad", action="store_true",
                    help="V3: let gradients flow into the VL backbone via the tap (else stop-grad / no_grad)")
    # M4 (aux-loss intent): NO conditioning. An aux head on the live LoRA-VL mean is trained to
    # PREDICT the frozen VL slot target z* (anticipatory representation); the action path never
    # sees an intent. Use with --pi05-config pi05_base_nointent + --aux-intent-stack.
    ap.add_argument("--aux-intent", action="store_true", help="M4: aux-loss intent (predict z*, no conditioning)")
    ap.add_argument("--aux-intent-stack", default=None,
                    help="M4: intent_stack_*.pt providing the FROZEN VL slot encoder for the z* target")
    ap.add_argument("--aux-intent-weight", type=float, default=1.0, help="M4: weight on the z*-prediction aux loss")
    # FIXED-M4: make the aux head a deployable GENERATOR -- feed its z_hat to the action head
    # (train AND deploy) so intent is actually usable. Tap is stop-grad + z_hat detached into the
    # action head => no backbone corruption (the failure mode of aux-only M4). Use pi05_base_intent.
    ap.add_argument("--aux-condition", action="store_true",
                    help="FIXED-M4: condition the action head on the aux head's z_hat (co-trained conditioning)")
    # REPA-align (M5): align the ACTION EXPERT's penultimate hidden to the frozen slot target z*
    # via a projector + cosine loss (Yu et al. 2024, "Representation Alignment for Generation").
    # NO conditioning -> deploy == M0 (training-only). Gradient is localized to the action expert
    # (+ projector) and only diffuses into VL via cross-attention; bounded cosine loss -> avoids the
    # aux-M4 backbone-corruption failure. Use with --pi05-config pi05_base_nointent + --align-intent-stack.
    ap.add_argument("--align-intent", action="store_true",
                    help="M5/REPA: align action-expert penultimate hidden to z* (cosine, no conditioning)")
    ap.add_argument("--align-intent-stack", default=None,
                    help="M5: intent_stack_*.pt providing the FROZEN slot encoder for the z* target")
    ap.add_argument("--align-workspace-stack", default=None,
                    help="M6-align: workspace_stack_*.pt (frozen workspace encoder); target = workspace "
                         "token w (MolmoPoint-supervised, saliency-selective) instead of the slot z*")
    ap.add_argument("--align-weight", type=float, default=0.5, help="M5: weight on the cosine alignment loss")
    # M7-JEPA (Sarvesh): SYMMETRIC joint-embedding alignment. A trainable predictor (action side) and a
    # trainable target encoder (workspace token w + target-EE-pose) meet in a joint latent; SigReg keeps
    # it from collapsing. Needs --align-intent + --align-workspace-stack. Target EE pose = object_states[-1].
    ap.add_argument("--align-jepa", action="store_true",
                    help="M7: joint (both-trainable) JEPA align of action hidden to [workspace w ; target EE pose] + SigReg")
    ap.add_argument("--sigreg-weight", type=float, default=1.0, help="M7: weight on the SigReg collapse regularizer")
    ap.add_argument("--jepa-dim", type=int, default=256, help="M7: joint embedding dim D")
    ap.add_argument("--jepa-lr", type=float, default=1e-3,
                    help="M7: LR for the FRESH JEPA heads (base pi05 params keep --lr; 2e-5 is far too "
                         "slow for random-init heads -> SigReg can't expand them -> collapse)")
    # M8-WSM (steer_intent): a 3-layer MLP on the action-expert PENULTIMATE predicts INTENT (future EEF).
    # w_t (frozen causal workspace latent, past-only) CFG-conditions MLP layer 1; w_{t+1} is a JEPA
    # alignment target for MLP layer 2. Both losses are TRAIN-TIME ONLY (deploy == M0; w never in the
    # inference graph). Needs --wsm-w-cache-dir (precompute_w.py output). No conditioning of the action head.
    ap.add_argument("--wsm-intent", action="store_true",
                    help="M8: workspace-conditioned intent head (CFG w_t + JEPA w_{t+1} + intent regression)")
    ap.add_argument("--wsm-w-cache-dir", default=None,
                    help="M8: dir of per-demo w.npy (frozen causal workspace latents from precompute_w.py)")
    ap.add_argument("--wsm-w-dim", type=int, default=512, help="M8: workspace token dim")
    ap.add_argument("--wsm-jepa-weight", type=float, default=1.0, help="M8: weight on the JEPA (w_{t+1}) align loss")
    ap.add_argument("--wsm-intent-weight", type=float, default=1.0, help="M8: weight on the intent-regression loss")
    ap.add_argument("--wsm-sigreg-weight", type=float, default=0.0, help="M8: weight on SIGReg isotropy (0=off)")
    # M9 (steer_intent): the STRIPPED head -- same penultimate tap + future-EEF target as M8, but no w
    # anywhere (no CFG condition, no JEPA target, no stage-1/precompute). Needs no --wsm-w-cache-dir.
    ap.add_argument("--intent-flow", action="store_true",
                    help="M9: intent head on the action-expert penultimate, no workspace token")
    ap.add_argument("--intent-flow-decoder", default="flow", choices=["flow", "l1", "mse"],
                    help="M9: flow-matching decoder (distributional) vs l1 (median) vs mse (=M8's objective)")
    ap.add_argument("--intent-flow-weight", type=float, default=0.25,
                    help="M9: weight on the intent aux loss (w=2.0 over-regularized M8; 0.25 was best)")
    ap.add_argument("--intent-flow-target", default="concat", choices=["concat", "mean"],
                    help="M9: h-step EEF trajectory (concat) vs mean over the horizon (=M8's target)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default=None,
                    help="resume a cotrain run from a ckpt dir (model.safetensors [+ aux_head.pt]); "
                         "start step parsed from the dir name, LR schedule fast-forwarded to it")
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
    start_step = 0
    if args.resume_from:  # continue a stopped run from its checkpoint (model + step)
        safetensors.torch.load_model(model, os.path.join(args.resume_from, "model.safetensors"), strict=False)
        start_step = int(os.path.basename(os.path.normpath(args.resume_from)))
        print(f"[resume] loaded {args.resume_from} -> start_step {start_step}", flush=True)
    else:
        safetensors.torch.load_model(model, os.path.join(args.pi05_weights, "model.safetensors"), strict=False)
    if "lora" in pi05_config.model.paligemma_variant:
        _set_lora_requires_grad(model)        # LoRA-VL + full action head (gemma base frozen)
    else:
        _set_action_head_only_requires_grad(model)
    if os.environ.get("FORCE_FP32"):          # debug: run the forward in fp32 (bf16-overflow check)
        model = model.float()
        print("[debug] model forced to float32", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()  # fit a 3B model on one GPU
    model.train()
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    # --- MIP slot-intent stack (co-trained); replicated per rank, grads all-reduced manually ---
    aux_intent = args.aux_intent
    align_jepa = args.align_jepa
    align_intent = args.align_intent or align_jepa    # JEPA reuses the align tap/loop machinery
    wsm_intent = args.wsm_intent                       # M8: workspace-conditioned intent head
    intent_flow = args.intent_flow                     # M9: stripped intent head (no w)
    cotrain = CotrainIntentModule(args.slot_task_config, args.config_dir, device=device,
                                  warmstart_ckpt=args.warmstart, vl_obs=args.vl_intent or aux_intent or align_intent)
    aux_head = None
    align_proj = None
    jepa_pred = jepa_target = jepa_h = None
    wsm_head = None
    intent_flow_head = None
    # M8: inject the frozen w cache into the dataset config so make_dataset windows w with the frames.
    if wsm_intent:
        from omegaconf import open_dict
        with open_dict(cotrain.cfg.task):
            cotrain.cfg.task.wsm_w_cache_dir = args.wsm_w_cache_dir
    # M6-align: align to the frozen WORKSPACE encoder's token (MolmoPoint-supervised, saliency-selective)
    # instead of the slot z*. Same REPA mechanism; only the target source + its dim differ.
    align_ws = align_intent and args.align_workspace_stack is not None
    _wsenc, _align_tdim = None, cotrain.intent_dim
    if align_ws:
        from workspace_models import WorkspaceConfig, WorkspaceModel
        _ws = torch.load(args.align_workspace_stack, map_location=device, weights_only=False)
        _wscfg = WorkspaceConfig(**_ws["cfg"])
        _wsenc = WorkspaceModel(_wscfg).encoder.to(device).eval()
        _wsenc.load_state_dict(_ws["encoder"])
        for _p in _wsenc.parameters():
            _p.requires_grad_(False)
        _align_tdim = _wscfg.hidden_dim
        if is_main:
            print(f"[M6-align] frozen workspace encoder (step {_ws.get('step')}) -> target dim {_align_tdim}", flush=True)
    elif aux_intent or align_intent:
        # M4/M5: load the FROZEN slot encoder (the z* target source). vl_proj / flow_map are unused
        # (no generator) -> leave untouched. Only the aux head / projector (+ pi05 action head) train.
        _st = torch.load(args.aux_intent_stack if aux_intent else args.align_intent_stack,
                         map_location=device, weights_only=False)
        cotrain.agent.slot_encoder.load_state_dict(_st["slot_encoder"])
        cotrain.agent.slot_encoder.eval()
        for _p in cotrain.agent.slot_encoder.parameters():  # only the slot encoder produces z* (frozen)
            _p.requires_grad_(False)
    if aux_intent:
        # M4: a fresh aux head predicts z* from the live LoRA-VL mean.
        aux_head = torch.nn.Sequential(
            torch.nn.LazyLinear(512), torch.nn.GELU(), torch.nn.Linear(512, cotrain.intent_dim),
        ).to(device)
        aux_head.train()
        if is_main:
            print(f"[M4] frozen VL slot encoder (step {_st.get('step')}); aux head -> z* dim {cotrain.intent_dim}", flush=True)
    elif align_jepa:
        # M7-JEPA (Sarvesh), SimSiam form: both encoders train; a predictor + stop-grad prevent collapse
        # WITHOUT batch statistics. (SigReg/VICReg need a big batch to estimate covariance -> useless at
        # pi05's batch=2; SimSiam's stop-grad+predictor asymmetry is batch-independent.)
        #   enc_a: action hidden -> z_a ;  enc_b: [workspace w ; EE pose] -> z_b ;  predictor h(.)
        jepa_pred = torch.nn.Sequential(   # enc_a (action side)
            torch.nn.LazyLinear(512), torch.nn.GELU(), torch.nn.Linear(512, args.jepa_dim)).to(device)
        jepa_target = torch.nn.Sequential(  # enc_b ([workspace w ; EE pose] side)
            torch.nn.LazyLinear(512), torch.nn.GELU(), torch.nn.Linear(512, args.jepa_dim)).to(device)
        jepa_h = torch.nn.Sequential(       # SimSiam predictor (bottleneck)
            torch.nn.Linear(args.jepa_dim, args.jepa_dim // 4), torch.nn.GELU(),
            torch.nn.Linear(args.jepa_dim // 4, args.jepa_dim)).to(device)
        jepa_pred.train(); jepa_target.train(); jepa_h.train()
        if is_main:
            print(f"[M7-JEPA SimSiam] joint dim {args.jepa_dim}, align_w {args.align_weight}; "
                  f"stop-grad+predictor anti-collapse (batch-independent)", flush=True)
    elif align_intent:
        # M5/M6-align/REPA: an asymmetric projector maps the pooled action-expert penultimate hidden
        # to the target embedding space; cosine-aligned to the frozen target (stop-grad).
        align_proj = torch.nn.Sequential(
            torch.nn.LazyLinear(512), torch.nn.GELU(), torch.nn.Linear(512, _align_tdim),
        ).to(device)
        align_proj.train()
        if is_main:
            _tag = "M6-align (workspace)" if align_ws else "M5 (slot z*)"
            print(f"[{_tag}] REPA projector -> target dim {_align_tdim}, weight {args.align_weight}", flush=True)
    elif wsm_intent or intent_flow:
        pass  # M8/M9 heads are built after ds exists (need the intent-target dim); slot stack unused
    else:
        cotrain.train()
    obs_steps = int(cotrain.cfg.task.obs_steps)  # MIP encoder consumes the first obs_steps frames
    action_horizon = int(pi05_config.model.action_horizon)  # Pi0.5 predicts this many steps (10); MIP chunk is 16

    # --- Data: MIP LiberoDataset (slot) gives obs / intent_frames / object_states / action ---
    # Per-task language prompts: the LiberoDataset emits a per-sample task_id (HDF5 file
    # index) when task_id_conditioning is on; map it to the task language from the filenames
    # so the policy trains on REAL per-task prompts instead of a constant "complete the task"
    # (the constant prompt made the policy task-blind -> ~0% SR at eval).
    from omegaconf import open_dict
    with open_dict(cotrain.cfg.task):
        cotrain.cfg.task.task_id_conditioning = True
    _task_langs = [os.path.basename(p).replace("_demo.hdf5", "").replace(".hdf5", "").replace("_", " ")
                   for p in cotrain.cfg.task.dataset_paths]
    if is_main:
        print(f"[prompts] {len(_task_langs)} per-task prompts, e.g. '{_task_langs[0]}'", flush=True)
    # M9 needs wsm_intent_target WITHOUT a w cache (M8 only emits it alongside w), and wants the h-step
    # trajectory option; the subclass adds exactly that on top of the stock factory.
    ds = (make_intent_flow_dataset(cotrain.cfg.task, intent_target_mode=args.intent_flow_target)
          if intent_flow else make_dataset(cotrain.cfg.task))
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=8, drop_last=True)
    tfm = _build_pi05_transforms(pi05_config)

    # --- Optimizer over trainable Pi0.5 params + the whole intent stack ---
    params = [p for p in model.parameters() if p.requires_grad]
    if aux_intent:
        # The aux head reads the pi05 PaliGemma vl_mean (gemma_2b prefix width = 2048), which is
        # INDEPENDENT of slot_vl_dim (the slot-encoder input dim, e.g. 768 for DINOv2/DynaFLIP). They
        # coincide only when the slot encoder reads PaliGemma grids; decouple so cross-encoder z*
        # targets (e.g. DINOSAUR/DINOv2) work.
        _vlmean_dim = 2048
        if args.aux_condition:  # FIXED-M4 generator: vl_mean -> z_hat (no proprio; matches M3's deployable recipe)
            aux_head(torch.zeros(1, _vlmean_dim, device=device))
        else:                   # M4 aux-only: [vl_mean ; current proprio] -> z*
            _proprio_dim = int(ds[0]["obs"]["state"].shape[-1])
            aux_head(torch.zeros(1, _vlmean_dim + _proprio_dim, device=device))
        params = params + list(aux_head.parameters())   # slot encoder frozen; only the aux head trains
        if args.resume_from and os.path.exists(os.path.join(args.resume_from, "aux_head.pt")):
            aux_head.load_state_dict(torch.load(os.path.join(args.resume_from, "aux_head.pt"),
                                                map_location=device, weights_only=False)["aux_head"])
            print("[resume] loaded aux_head.pt", flush=True)
    elif align_jepa:
        # M7: predictor at action-expert width; target encoder at [workspace w ; EE pose] width.
        _expert_width = int(raw_model.action_out_proj.in_features)
        _ee_dim = int(ds[0]["object_states"].shape[-1])        # target EE pose dim (ee_states = 6D)
        jepa_pred(torch.zeros(1, _expert_width, device=device))
        jepa_target(torch.zeros(1, _align_tdim + _ee_dim, device=device))
        params = params + list(jepa_pred.parameters()) + list(jepa_target.parameters()) + list(jepa_h.parameters())
        if args.resume_from and os.path.exists(os.path.join(args.resume_from, "jepa.pt")):
            _j = torch.load(os.path.join(args.resume_from, "jepa.pt"), map_location=device, weights_only=False)
            jepa_pred.load_state_dict(_j["pred"]); jepa_target.load_state_dict(_j["target"]); jepa_h.load_state_dict(_j["h"])
            print("[resume] loaded jepa.pt", flush=True)
    elif align_intent:
        # Materialise the projector's LazyLinear at the action-expert width (action_out_proj in-dim).
        _expert_width = int(raw_model.action_out_proj.in_features)
        align_proj(torch.zeros(1, _expert_width, device=device))
        params = params + list(align_proj.parameters())  # slot encoder frozen; only the projector trains
        if args.resume_from and os.path.exists(os.path.join(args.resume_from, "align_proj.pt")):
            align_proj.load_state_dict(torch.load(os.path.join(args.resume_from, "align_proj.pt"),
                                                  map_location=device, weights_only=False)["align_proj"])
            print("[resume] loaded align_proj.pt", flush=True)
    elif wsm_intent:
        # M8: 3-layer intent head on the action-expert penultimate (width = action_out_proj in-dim);
        # intent target dim from the dataset's wsm_intent_target (future-EEF pose).
        _expert_width = int(raw_model.action_out_proj.in_features)
        _intent_dim = int(ds[0]["wsm_intent_target"].shape[-1])
        wsm_head = WSMIntentHead(penult_dim=_expert_width, intent_dim=_intent_dim, w_dim=args.wsm_w_dim,
                                 jepa_weight=args.wsm_jepa_weight, intent_weight=args.wsm_intent_weight,
                                 sigreg_weight=args.wsm_sigreg_weight).to(device)
        wsm_head.train()
        params = params + list(wsm_head.parameters())
        if args.resume_from and os.path.exists(os.path.join(args.resume_from, "wsm_head.pt")):
            wsm_head.load_state_dict(torch.load(os.path.join(args.resume_from, "wsm_head.pt"),
                                                map_location=device, weights_only=False)["wsm_head"])
            print("[resume] loaded wsm_head.pt", flush=True)
        if is_main:
            print(f"[M8-WSM] penult_dim={_expert_width} intent_dim={_intent_dim} w_dim={args.wsm_w_dim} "
                  f"jepa={args.wsm_jepa_weight} intent={args.wsm_intent_weight} sigreg={args.wsm_sigreg_weight}",
                  flush=True)
    elif intent_flow:
        # M9: same tap/width as M8, but the head sees only the penultimate -- no w in or out.
        _expert_width = int(raw_model.action_out_proj.in_features)
        _intent_dim = int(ds[0]["wsm_intent_target"].shape[-1])
        intent_flow_head = IntentFlowHead(penult_dim=_expert_width, intent_dim=_intent_dim,
                                          decoder=args.intent_flow_decoder,
                                          weight=args.intent_flow_weight).to(device)
        intent_flow_head.train()
        params = params + list(intent_flow_head.parameters())
        if args.resume_from and os.path.exists(os.path.join(args.resume_from, "intent_flow_head.pt")):
            intent_flow_head.load_state_dict(torch.load(os.path.join(args.resume_from, "intent_flow_head.pt"),
                                                        map_location=device, weights_only=False)["intent_flow_head"])
            print("[resume] loaded intent_flow_head.pt", flush=True)
        if is_main:
            print(f"[M9-INTENT] penult_dim={_expert_width} intent_dim={_intent_dim} "
                  f"decoder={args.intent_flow_decoder} target={args.intent_flow_target} "
                  f"weight={args.intent_flow_weight} (no w, no CFG, no JEPA)", flush=True)
    else:
        params = params + cotrain.train_parameters()
    if align_jepa:  # fresh heads get a high LR; the pretrained pi05 backbone keeps the finetune LR
        _head = list(jepa_pred.parameters()) + list(jepa_target.parameters()) + list(jepa_h.parameters())
        _hids = {id(p) for p in _head}
        _base = [p for p in params if id(p) not in _hids]
        opt = torch.optim.AdamW([{"params": _base, "lr": args.lr},
                                 {"params": _head, "lr": args.jepa_lr}], weight_decay=1e-4)
        if is_main:
            print(f"[M7-JEPA] base LR {args.lr}, JEPA-head LR {args.jepa_lr}", flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    for _ in range(start_step):
        sched.step()  # fast-forward LR schedule to the resume point

    def build_observation(batch, intent):
        """Per-sample openpi transform of the Pi0.5 fields, then collate -> Observation, actions."""
        # The MIP LiberoDataset returns state/action already MinMax-normalized; un-normalize
        # back to RAW physical units so openpi's Normalize (which the env eval also applies)
        # runs exactly ONCE. (Feeding the pre-normalized values caused double-normalization ->
        # train/eval state+action distribution mismatch -> 0% SR.)
        st = ds.normalizer["obs"]["state"].unnormalize(batch["obs"]["state"].numpy())  # (B,To,15) raw
        agv = batch["obs"]["agentview_rgb"].numpy()    # (B, To, C, H, W) in [0,1]
        wr = batch["obs"]["eye_in_hand_rgb"].numpy()
        act = ds.normalizer["action"].unnormalize(batch["action"].numpy())             # (B,horizon,7) raw
        tid = batch["task_id"].numpy()                 # (B,) per-sample task index -> real prompt
        intent_np = intent.detach().cpu().numpy() if intent is not None else None  # None -> M4 (no conditioning)
        B = st.shape[0]
        elems = []
        cur = obs_steps - 1  # current-obs frame (last of the obs window, not a future frame)
        for i in range(B):
            # Raw LeRobot keys — the RepackTransform (first in the chain) remaps these to observation/*.
            el = {
                "image": (agv[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),  # current frame HWC
                "wrist_image": (wr[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
                "state": st[i, cur, :8].astype(np.float32),   # ee(6)+gripper(2)
                "actions": act[i, :action_horizon].astype(np.float32),  # Pi0.5 horizon (10), not MIP's 16
                "prompt": _task_langs[int(tid[i])],
            }
            if intent_np is not None:
                el["intent"] = intent_np[i].astype(np.float32)
            elems.append(tfm(el))
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

    step = start_step  # optimizer steps; resume continues from the checkpoint's step (else 0)
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

            if aux_intent and args.aux_condition:
                # FIXED-M4: the aux head is a deployable GENERATOR. Tap vl_mean (stop-grad -> the aux
                # loss never touches the backbone), predict z_hat, and CONDITION the action head on a
                # DETACHED z_hat (action loss can't fight the generator). Same chain at deploy.
                placeholder = torch.zeros(intent_frames.shape[0], cotrain.intent_dim, device=device)
                observation, actions = build_observation(batch, placeholder)
                with torch.no_grad():
                    vl_mean = raw_model.vl_image_features(observation)
                z_hat = aux_head(vl_mean)
                # --intent-attach: feed z_hat ATTACHED so the action loss also shapes the generator
                # (action head pulls useful intents -> avoids an inert channel). Backbone stays safe
                # via the no_grad tap above. Default: detached (M3-style, generator trained by aux only).
                observation = dataclasses.replace(
                    observation, intent=(z_hat if args.intent_attach else z_hat.detach()))
            elif aux_intent or align_intent or wsm_intent or intent_flow:
                # M4/M5/M8/M9: NO conditioning of the action head. The aux head / REPA projector / wsm /
                # intent head reads a tap from the SAME action forward (one pass -> vanilla DDP).
                observation, actions = build_observation(batch, None)
            elif args.vl_intent:
                # VL-grounded intent: build obs (placeholder intent) -> tap the live LoRA-VL
                # mean image-token feature -> train the flow map to match z* -> sample the
                # DEPLOYABLE z_hat and feed THAT to the action head (z_hat is the same at
                # train and eval, unlike M3 which fed the privileged future z*).
                placeholder = torch.zeros(intent_frames.shape[0], cotrain.intent_dim, device=device)
                observation, actions = build_observation(batch, placeholder)
                if args.vl_grad:                                  # V3: grad into the VL backbone
                    vl_mean = raw_model.vl_image_features(observation)
                else:                                             # V1/V2: stop-grad the VL tap
                    with torch.no_grad():
                        vl_mean = raw_model.vl_image_features(observation)
                _, ilosses = cotrain.vl_intent_and_losses(intent_frames, object_states, vl_mean, delta_t)
                if args.intent_attach:                            # V2/V3: action loss shapes the flow map
                    z_hat = cotrain.vl_sample_intent(vl_mean)
                else:                                             # V1: detached into the action head
                    with torch.no_grad():
                        z_hat = cotrain.vl_sample_intent(vl_mean)
                observation = dataclasses.replace(observation, intent=z_hat)  # bypass build's numpy detach
            else:
                obs_mip = {k: v[:, :obs_steps].to(device) for k, v in batch["obs"].items()}
                intent, ilosses = cotrain.intent_and_losses(intent_frames, object_states, obs_mip, delta_t)
                observation, actions = build_observation(batch, intent)

            if aux_intent and not args.aux_condition:
                action_loss, vl_mean = model(observation, actions, return_vl_mean=True)  # single VL pass
            elif align_intent or wsm_intent or intent_flow:
                # single forward -> action loss + the penultimate hidden (B, horizon, expert_width)
                action_loss, act_hidden = model(observation, actions, return_action_hidden=True)
            else:
                action_loss = model(observation, actions)  # FIXED-M4: conditioned on z_hat (tapped above)
            if isinstance(action_loss, (list, tuple)):
                action_loss = action_loss[0]
            action_loss = action_loss.mean()

            if aux_intent:  # predict the frozen VL slot target z* (the anticipatory intent code)
                with torch.no_grad():
                    z_star = cotrain.agent.slot_encoder(batch["intent_vl_grids"].to(device))[0]  # (B, dim)
                if args.aux_condition:  # FIXED-M4: generator's z_hat (already computed) regresses to z*
                    ilosses = {"auxz": torch.nn.functional.mse_loss(z_hat, z_star)}
                else:                   # M4 aux-only: [same-forward vl_mean ; current proprio] -> z*
                    state_cur = batch["obs"]["state"][:, obs_steps - 1].to(device).float()  # (B, state_dim)
                    ilosses = {"auxz": torch.nn.functional.mse_loss(
                        aux_head(torch.cat([vl_mean, state_cur], dim=-1)), z_star)}
            elif align_jepa:  # M7 (SimSiam): symmetric predict-the-other + stop-grad -> collapse-free, batch-free
                with torch.no_grad():                                        # frozen workspace token (slots)
                    w = _wsenc(batch["intent_vl_grids"][:, 0].to(device)).squeeze(1)   # (B, 768)
                ee = batch["object_states"][:, -1].to(device).float()        # (B, ee_dim) target EE pose
                z_a = jepa_pred(act_hidden.mean(dim=1))                       # action embedding (B, D)
                z_b = jepa_target(torch.cat([w, ee], dim=-1))                # [workspace ; EE pose] embedding
                _cos = torch.nn.functional.cosine_similarity
                align_l = -0.5 * (_cos(jepa_h(z_a), z_b.detach(), dim=-1).mean()
                                  + _cos(jepa_h(z_b), z_a.detach(), dim=-1).mean())
                zstd = 0.5 * (torch.nn.functional.normalize(z_a, dim=-1).std(0).mean()
                              + torch.nn.functional.normalize(z_b, dim=-1).std(0).mean())  # collapse monitor
                ilosses = {"align": align_l, "zstd": zstd.detach()}
            elif align_intent:  # REPA: cosine-align pooled action hidden -> projector -> to target (stop-grad)
                with torch.no_grad():
                    if align_ws:  # M6-align: workspace token of the (first) future-frame DINOv2 grid
                        z_star = _wsenc(batch["intent_vl_grids"][:, 0].to(device)).squeeze(1)  # (B, hidden)
                    else:         # M5: slot z*
                        z_star = cotrain.agent.slot_encoder(batch["intent_vl_grids"].to(device))[0]  # (B, dim)
                z_hat = align_proj(act_hidden.mean(dim=1))  # pool over horizon -> (B, target_dim)
                ilosses = {"align": (1.0 - torch.nn.functional.cosine_similarity(z_hat, z_star, dim=-1)).mean()}
            elif wsm_intent:  # M8: w_t CFG-conditions the intent head; w_{t+1} JEPA-aligns MLP layer 2
                w_t = batch["wsm_w_t"].to(device).float()               # (B, w_dim)  past-only, conditioning input
                w_next = batch["wsm_w_next"].to(device).float()         # (B, w_dim)  JEPA target (stop-grad)
                itgt = batch["wsm_intent_target"].to(device).float()    # (B, eef_dim) future-EEF regression target
                _, wsm_total, wsm_m = wsm_head(act_hidden, w_t, w_next, itgt, global_step=step)
                ilosses = {"wsm": wsm_total, **wsm_m}
            elif intent_flow:  # M9: penultimate -> intent. No w read anywhere in this arm.
                itgt = batch["wsm_intent_target"].to(device).float()   # (B, h*eef_dim) or (B, eef_dim)
                _, if_total, if_m = intent_flow_head(act_hidden, itgt, global_step=step)
                ilosses = {"intent_flow": if_total, **if_m}

            if aux_intent:
                total = action_loss + args.aux_intent_weight * ilosses["auxz"]
            elif align_jepa:
                total = action_loss + args.align_weight * ilosses["align"]  # SimSiam: stop-grad handles collapse
            elif align_intent:
                total = action_loss + args.align_weight * ilosses["align"]
            elif wsm_intent:  # additive: action BC + (intent-regression + JEPA + optional SIGReg), weighted inside the head
                total = action_loss + ilosses["wsm"]
            elif intent_flow:  # additive: action BC + intent aux (weighted inside the head)
                total = action_loss + ilosses["intent_flow"]
            else:
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
                # all-reduce the non-DDP trainable grads (aux head for M4, projector for M5, else the stack)
                _extra = (aux_head.parameters() if aux_intent
                          else (list(jepa_pred.parameters()) + list(jepa_target.parameters()) + list(jepa_h.parameters())) if align_jepa
                          else align_proj.parameters() if align_intent
                          else wsm_head.parameters() if wsm_intent
                          else intent_flow_head.parameters() if intent_flow
                          else cotrain.train_parameters())
                for p in _extra:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad /= world_size
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()  # cosine LR decay (one step per optimizer update)
            opt.zero_grad()

            if is_main and step % 100 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in last.items())
                print(f"step {step}: {msg}", flush=True)
            if is_main and step > 0 and step % args.save_every == 0:
                d = os.path.join(args.out, str(step)); os.makedirs(d, exist_ok=True)
                safetensors.torch.save_model(raw_model, os.path.join(d, "model.safetensors"))
                if aux_intent:
                    # M4: only the aux head is new (training-only; deploy uses just the action head).
                    torch.save({"aux_head": aux_head.state_dict(), "step": step}, os.path.join(d, "aux_head.pt"))
                elif align_jepa:
                    # M7: two encoders + predictor are new (training-only; deploy == M0).
                    torch.save({"pred": jepa_pred.state_dict(), "target": jepa_target.state_dict(),
                                "h": jepa_h.state_dict(), "step": step}, os.path.join(d, "jepa.pt"))
                elif align_intent:
                    # M5/REPA: only the projector is new (training-only; deploy == M0, just the action head).
                    torch.save({"align_proj": align_proj.state_dict(), "step": step},
                               os.path.join(d, "align_proj.pt"))
                elif wsm_intent:
                    # M8: only the intent head is new (training-only; deploy == M0, just the action head).
                    torch.save({"wsm_head": wsm_head.state_dict(), "step": step},
                               os.path.join(d, "wsm_head.pt"))
                elif intent_flow:
                    # M9: only the intent head is new (training-only; deploy == M0, just the action head).
                    torch.save({"intent_flow_head": intent_flow_head.state_dict(), "step": step},
                               os.path.join(d, "intent_flow_head.pt"))
                else:
                    stack = {  # actual intent-stack weights (so future runs can resume)
                        "slot_encoder": cotrain.agent.slot_encoder.state_dict(),
                        "intent_flow_map": cotrain.agent.intent_flow_map.state_dict(),
                        "step": step,
                    }
                    # VL mode trains vl_proj (the obs encoder); ResNet path saves agent.encoder.
                    stack["vl_proj" if cotrain.vl_obs else "encoder"] = (
                        cotrain.vl_proj if cotrain.vl_obs else cotrain.agent.encoder
                    ).state_dict()
                    torch.save(stack, os.path.join(d, "intent_stack.pt"))
                print(f"[ckpt] step {step} -> {d}", flush=True)
                # Prune to the last 3 checkpoints (fp32 model.safetensors is ~15GB each).
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
