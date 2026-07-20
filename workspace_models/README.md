# workspace_models

Current-frame re-implementation of **Workspace Models: Lightweight Robotic Memory via
Saliency-Driven Supervision** (CoRL 2026), for use as a **saliency-supervised intent token**
for Pi0.5 — an alternative to the slot-attention / DINOSAUR intent.

## Why this (vs slot-intent)
Our slot-intent (M2/M3) reconstructs the *full* DINO feature grid → generic scene content →
**redundant** with what PaliGemma already encodes. A workspace token is instead distilled from a
**VLM-curated salient patch subset** (target object / what-to-focus-on-now), so its content is
*task-selective*, which is the actual lever our null results pointed at.

## Scope of this implementation
- **Current-frame only.** The paper's encoder is causal over `o_{1:t}` (long-horizon memory);
  here the encoder sees a single frame's patches + a learned workspace slot. This captures the
  "what to focus on now" half — the object-isolation signal — without the temporal memory
  machinery (LIBERO-Goal is ~Markovian, so that half has little to bite on).
- **Backbone-agnostic core.** The model consumes DINO **patch tokens** `(B, N, feat_dim)`.
  Produce them live via `backbone.DinoV3Backbone`, or precompute grids offline (DINOSAUR-style).
- **VLM saliency labeling is external** (the data long-pole). The model consumes the salient set
  as `(target_patches, target_mask)`; producing it (Qwen3-VL keyframe/event detection +
  MolmoPoint open-vocab pointing → DINO patches) is a separate offline pipeline — not included.

## Files (modular)
| file | role | paper |
|---|---|---|
| `config.py` | `WorkspaceConfig` w/ Table 3 defaults | Table 3 |
| `backbone.py` | frozen DINOv3 → patch tokens (optional) | §3, Table 3 "Model" |
| `encoder.py` | `AttentionPooler` + `WorkspaceEncoder` → workspace token | Fig 2, §3 |
| `decoder.py` | DETR set-reconstruction decoder (feature + occupancy heads) | §3.1 |
| `matching.py` | Hungarian matching, no grad through σ | §3.1, Eq. under 3.1 |
| `losses.py` | `L_feat` + `L_active` under the matching | Eq. 3.1 / 3.2 |
| `model.py` | `WorkspaceModel` (train `forward`, deploy `encode`) + smoke test | — |

## Usage
```python
from workspace_models import WorkspaceModel, WorkspaceConfig

cfg = WorkspaceConfig(proprio_dim=8)          # match your DINO dim / patch count / proprio
model = WorkspaceModel(cfg)

# train (supervision = VLM-curated salient patches):
w, losses = model(patch_tokens, target_patches, target_mask, proprio)   # (B, N, D), (B, P, D), (B, P)
losses["total"].backward()

# deploy (no decoder, no VLM):
w = model.encode(patch_tokens, proprio)       # (B, num_workspace_tokens, hidden_dim)
```
Smoke test: `python -m workspace_models.model`.

## Pipeline (built + smoke-tested)

```bash
# 1) LABEL — point at task objects over cached DINO grids -> salient patch indices (offline VLM).
#    Omit --molmo to dry-run with MockPointer.
python -m workspace_models.saliency.libero_adapter \
    --hdf5 /path/to/libero_goal/*_demo.hdf5 --vl-cache-dir .../vl_cache_goal_dinov2 \
    --out-dir .../workspace_salient_goal --sample-rate 5 --max-patches 8 --molmo <molmo_id>

# 2) STAGE-1 — train the workspace encoder on (grid, salient set); saves workspace_stack_*.pt.
python examples/openpi/train_workspace.py \
    --vl-cache-dir .../vl_cache_goal_dinov2 --salient-dir .../workspace_salient_goal \
    --out .../workspace_stack_goal --num-patches 196 --max-patches 8
```

- `saliency/libero_adapter.py` — labeling over HDF5 + grid caches → per-demo salient-index `.npz`.
- `dataset.py` `WorkspaceGridDataset` — pairs current-frame grids with salient sets for Stage-1.
- `examples/openpi/train_workspace.py` — Stage-1 trainer (warmup+cosine, grad-clip, per Table 3).

**Stage-2 (TODO):** freeze the encoder, project the frozen `w` (hidden_dim → `intent_dim`), and
condition Pi0.5 via the existing M3 path; at deploy run `encode()` live (like `--encoder-tap`).

Before a real labeling run: (i) confirm the MolmoPoint checkpoint id, (ii) hand-curate the object
list per task (`objects_per_task=`) — `objects_from_task_language` is a `the/a → noun` heuristic.

## Notes on faithfulness
- Loss weights (`feature_loss_weight=1.0`, `existence_loss_weight=0.01`) and matching costs
  (both `1.0`) follow Table 3; `feature_loss_weight` is `2.0` for DrawerRecall/BalanceBar.
- `max_patches` (decoder query slots, `m`) defaults to 8 ("Slots 8"); `num_workspace_tokens=1`
  ("Workspace tokens per step 1", the bottleneck — ablations show more tokens hurt).
- The encoder uses bidirectional self-attention over `[pooled X ; proprio ; z]` (single frame);
  the paper's causal temporal mask is unnecessary here and is deliberately omitted.
