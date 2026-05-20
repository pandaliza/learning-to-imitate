# Slot-Intent Architecture: Coding Session Notes

## Overview

This session covered the slot-attention based intent conditioning framework for robotic manipulation policies, including architecture deep-dives, visualization, ablations, and cross-task experimental results.

---

## Architecture: SlotObjectEncoder

```
Future frames (B, 8, 3, H, W)
    ↓ ResNet18 layer2 → (B×8, 128, 11, 11)
    ↓ Linear 128→64 → tokens (B×8, 121, 64)
    ↓ SlotAttention (K=3 slots, 3 iters)
    ↓ Soft selector (Linear 64→1 → softmax over K)
    ↓ Mean over 8 frames
→ Intent vector (B, 64)
```

### Slot Attention (detail)

Scaled dot-product attention with softmax over **slots** (not positions), so slots compete for each spatial position:

- **Q** = slot vectors → Linear → `(K, 64)`
- **K, V** = spatial tokens → Linear → `(121, 64)`
- `attn = softmax_over_K(Q·Kᵀ / √64)` → `(K, 121)`
- Slot update = weighted sum of V, passed through **GRU** (Gated Recurrent Unit) to refine across 3 iterations
- GRU = RNN cell with forget/update gates; slot vector is the hidden state, attention output is the input

### Auxiliary Losses

- **Aux regression loss**: `obj_regressor (64→14)` predicts GT object low-dim state from selected slot → MSE
- **Reconstruction loss** (SpatialBroadcastDecoder): each slot tiled to image resolution + coord channels → CNN → per-slot RGB + alpha → softmax masks → composite reconstruction → MSE vs input frame

### Gradient Flow

No stopgrad on `intent_vec` before the interpolant — slot encoder receives gradients from **both**:
1. Flow loss (via `x_t = α_t·z + β_t·intent_vec`)
2. Aux regression loss

The action decoder receives `intent_vec.detach()` — action loss only trains the MLP decoder, not the slot encoder.

---

## Experimental Results: lift-mh-image (v1 → v8)

| Variant | SR | CNN | Num slots | Recon loss | Soft selector |
|---------|-----|-----|-----------|------------|---------------|
| v1 | 0% | layer4 | 1 | ✗ | ✗ |
| v2 | 0% | layer4 | 1 | ✗ | ✗ |
| v3 | 0% | layer3 | 1 | ✗ | ✗ |
| v4 | 0% | layer3 | 2 | ✗ | ✗ |
| v5 | 40% | layer3 | 2 | ✗ | ✗ |
| v6 | 64% | layer3 | 3 | ✗ | ✗ |
| v7 | 100% | layer3 | 3 | ✓ | ✓ |
| v8 | 100% | layer2 | 3 | ✓ | ✓ |

Key insight: reconstruction loss + soft selector (v7) was the critical change enabling 100% SR. Layer2 (v8) kept the same SR at 3× faster training (11×11 vs 6×6 feature grid, smaller spatial tokens).

---

## CNN-Intent Ablation

Simpler baseline: ResNet18 → global avg pool → Linear → mean over k frames → 64D intent (no slot attention, no aux loss).

**Result: 0% SR on lift-mh-image** vs slot-intent 100%.

Global pooling collapses spatial structure — "average scene appearance" is too entangled to serve as a useful intent conditioning signal. Slot attention's spatial competition preserves localized structure the flow model can learn from.

---

## Cross-Task Results

### can-mh-image

| Run | Best SR |
|-----|---------|
| baseline | 91.7% |
| flow_intent | 77.1% |
| slot_intent | 29.2% |

### tool-hang-image

| Run | Best SR |
|-----|---------|
| baseline (ResNet50, bs64) | 54.2% |
| flow_intent (ResNet50, bs64) | 41.7% |
| flow_intent curriculum 100k | 4.2% |
| slot_intent (bs16) | 0% |
| slot_intent (bs64, in progress) | 0% |

### Why intent conditioning hurts on harder tasks

1. **Larger train/eval intent gap**: at train time the slot encoder sees actual future frames → precise 64D intent; at eval the flow ODE samples from noise → approximate intent. Lift-mh is forgiving; can-mh and tool-hang require more precise conditioning to beat a baseline.
2. **Weaker slot grounding on complex scenes**: lift-mh has a clean scene with a bright red cube. Can-mh (two objects, multimodal strategies) and tool-hang (cluttered, side-view, tool geometry) produce noisier slot vectors — the action decoder gets a worse conditioning signal than no intent at all.

---

## Visualizations

### Slot Attention Maps (`rollouts/slot_attn_v8.png`)

Generated via `examples/viz_slot_attention.py`:
1. Sample k=8 future frames from a demo trajectory (HDF5)
2. Run slot encoder with `return_attn=True` → attention weights `(B×k, K, 121)`
3. Upsample 11×11 attention → 84×84 via bilinear interpolation, blend as colormap overlay
4. Grid: raw frames on top, one colored row per slot below

**Observation**: Slot 0 = table/background, Slot 1 = red cube, Slot 2 = robot arm. Slot 1 shows tight localization on the cube (the aux loss is grounding it).

### Slot Reconstructions (`rollouts/slot_attn_v8_recon.png`)

Added `--show-recon` flag and `run_recon()` function to viz script. Runs SpatialBroadcastDecoder internals directly to extract per-slot RGB and masks.

**Observation**: Reconstructions are dark/smeared — slots are not achieving clean object-level decomposition. The SBD provides spatial competition pressure but the decoder is underpowered (4-layer conv). SR gains came from richer spatial features, not clean slot decomposition.

---

## Key Code Locations

| Component | File | Notes |
|-----------|------|-------|
| SlotAttention, SpatialBroadcastDecoder, SlotObjectEncoder | `mip/networks/slot_attention.py` | Full architecture |
| Slot intent training step | `mip/flow_intent_agent.py:355-399` | flow_loss + aux_loss, no stopgrad on intent |
| Slot intent eval (ODE sampling) | `mip/flow_intent_agent.py:572-605` | randn → ODE → intent → action |
| Attention visualization | `examples/viz_slot_attention.py` | `--show-recon` flag added this session |
| v8 task config | `examples/configs/task/lift_mh_image_slot_intent_v8.yaml` | layer2, 3 slots, recon, soft selector |
