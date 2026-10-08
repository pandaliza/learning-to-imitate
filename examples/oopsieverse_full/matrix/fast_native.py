"""Evaluation-only: CFMPolicy.generate_action with the observation encoders evaluated ONCE per chunk instead of at
each of the 16 Euler steps (the official loop recomputes identical seg/state tokens every step in eval mode).
Same ops in the same order on the same inputs -> verified bitwise identical to the official generate_action
(verify_fast_native.json). Used only by the evaluator's inference server to cut GPU-0 load (training shares GPU 0)."""
import torch
from learning.utils import denormalize_action


@torch.no_grad()
def generate_action_cached(model, seg_images, state, init_noise):
    assert not model.training
    cfg = model.config
    B = state.shape[0]
    device = state.device
    seg_tokens = model.encode_segmentation(seg_images)
    state_tokens = model.encode_state(state)
    mask = model.construct_mask(device)
    x = init_noise.detach().clone()
    dt = 1.0 / cfg.num_inference_steps
    for step in range(cfg.num_inference_steps):
        t = torch.full((B,), step * dt, device=device)
        t_tokens = model.encode_timestep(t)
        action_tokens = model.encode_action(x)
        action_tokens += t_tokens
        sequence = torch.cat([seg_tokens, state_tokens, action_tokens], dim=1)
        sequence = sequence + model.pos_encoding(sequence.shape[1], B, device)
        for block in model.transformer_blocks:
            sequence = block(sequence, attn_mask=mask)["output"]
        sequence = model.final_norm(sequence)
        velocity = model.action_head(sequence[:, -cfg.action_chunk_size:])
        x = x + velocity * dt
    if cfg.action_min is not None and cfg.action_max is not None:
        x = denormalize_action(x, model.action_min, model.action_max)
    return x
