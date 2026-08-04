# B0/B1 Feature Cache — Quick Start

**Before running this:** feature cache must be pre-extracted (see Full-Dataset Extraction below).

---

## 1. Loading Features in Training Code

### Option A: Simple per-episode loader
```python
from steer_intent.feature_cache import FeatureCache

cache = FeatureCache(
    cache_root="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/pooled",
    variant="pooled"
)

# Load one episode
data = cache.load_episode(task="TurnOnElectricKettle", episode_stem="episode_000000")

# data contains:
# - "base": (N_frames, 1152) fp16
# - "wrist": (N_frames, 1152) fp16
# - "lang": (1152,) fp16
# - "num_frames": int

print(data["base"].shape)  # (225, 1152)
print(data["lang"].shape)  # (1152,)
```

### Option B: In a DataLoader (example)
```python
import torch
from pathlib import Path
from steer_intent.feature_cache import FeatureCache

class CachedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, cache_root, task_names, variant="pooled"):
        self.cache = FeatureCache(cache_root, variant=variant)
        self.task_names = task_names
        self.episodes = self._scan_episodes(cache_root)
    
    def _scan_episodes(self, cache_root):
        """Scan for all cached episodes."""
        episodes = []
        features_dir = Path(cache_root) / "features"
        for task_dir in features_dir.iterdir():
            if task_dir.is_dir() and task_dir.name in self.task_names:
                for npz_file in task_dir.glob("*.npz"):
                    episode_stem = npz_file.stem
                    episodes.append((task_dir.name, episode_stem))
        return episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        task, episode_stem = self.episodes[idx]
        data = self.cache.load_episode(task, episode_stem)
        
        # Convert to torch tensors
        return {
            "base": torch.from_numpy(data["base"]),
            "wrist": torch.from_numpy(data["wrist"]),
            "lang": torch.from_numpy(data["lang"]),
        }

# Usage in training
dataset = CachedFeaturesDataset(
    cache_root="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/pooled",
    task_names=["TurnOnElectricKettle", "PickPlaceCounterToCabinet", ...],
)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    base_feats = batch["base"]  # (batch_size, N_frames, 1152)
    wrist_feats = batch["wrist"]  # (batch_size, N_frames, 1152)
    lang_feats = batch["lang"]  # (batch_size, 1152)
    # ... training forward pass
```

---

## 2. Available Variants

### Pooled (Recommended for B0/B1)
- **Shape:** `base=(N_frames, 1152)`, `wrist=(N_frames, 1152)`
- **Cache size:** ~7.4 GB (all 9 tasks, 4543 episodes, 1.87M frames)
- **Use:** B0/B1 baseline with frozen feature encoder
- **Integration:** Straightforward concatenation or attention over camera features

```python
# Example: concatenate both cameras
combined_feat = torch.cat([base, wrist], dim=-1)  # (N_frames, 2304)
```

### Grid16 (Optional, future work)
- **Shape:** `base=(N_frames, 16, 1152)`, `wrist=(N_frames, 16, 1152)`
- **Cache size:** ~8.4 GB
- **Use:** Spatial reasoning (e.g., attend to specific image regions)
- **Integration:** Requires vision transformer or spatial attention layers

```python
# Example: multi-head attention over spatial grid
from torch.nn import MultiheadAttention
attn = MultiheadAttention(embed_dim=1152, num_heads=8)
attended, _ = attn(grid_feat, grid_feat, grid_feat)  # (N_frames, 16, 1152)
```

---

## 3. Full-Dataset Extraction (One-Time Setup)

### Prerequisites
1. Ensure `.venv` environment is set up with `pip install -e .`
2. HF_HOME will be auto-set to `/data/user_data/ldahiya/hf_cache`
3. GPU available (job will queue/pend otherwise)

### Launch Extraction

**For pooled variant** (recommended first):
```bash
sbatch --export=NONE slurm-scripts/b0b1/extract_features_full.sbatch pooled
```

**For grid16 variant** (optional):
```bash
sbatch --export=NONE slurm-scripts/b0b1/extract_features_full.sbatch grid16
```

### Monitor Progress
```bash
# Check job status
squeue -u ldahiya | grep features_full

# View logs (streaming)
tail -f logs/b0b1_features_full_<jobid>.log
```

### Expected Output
- **Cache location:** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>/`
- **Manifest:** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>/manifest.json`
- **Features:** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>/features/<task>/<episode>.npz`
- **Time:** ~50 hours on 1 GPU
- **Size:** 7.4 GB (pooled) or 8.4 GB (grid16)

### Resume & Restart
- Scripts auto-skip existing `.npz` files (resume-safe)
- To re-extract: manually delete specific `.npz` files or entire task dir, then relaunch

---

## 4. Feature Specifications (SigLIP-SO400M)

| Property | Value |
|----------|-------|
| **Vision Encoder** | google/siglip-so400m-patch14-384 |
| **Vision Feature Dim** | 1152 |
| **Text/Lang Feature Dim** | 1152 |
| **Input Image Size** | 384×384 pixels |
| **Spatial Token Grid** | 24×24 (576 patches + 1 CLS) → pooled to 4×4 for grid16 |
| **Dtype** | float16 (2 bytes/value) |
| **Cameras** | robot0_agentview_left ("base") + robot0_eye_in_hand ("wrist") |
| **Tasks** | 9 (TurnOnElectricKettle, PickPlaceCounterToCabinet, PickPlaceCounterToStove, SlideDishwasherRack, KettleBoiling, LoadDishwasher, PrepareCoffee, PreSoakPan, WashLettuce) |

---

## 5. Troubleshooting

### Error: "Cache not found"
```
FileNotFoundError: Feature cache not found: /path/to/features/TurnOnElectricKettle/episode_000000.npz
```
**Solution:** Run full-dataset extraction first (`sbatch ... extract_features_full.sbatch pooled`)

### Error: "Manifest not found"
```
FileNotFoundError: manifest.json not found at /path/to/b0b1_features/pooled/manifest.json
```
**Solution:** Extraction incomplete or wrong path. Check cache root ends in `pooled/` or `grid16/`

### Out of memory during extraction
- GPU OOM is unlikely (SigLIP is 400M params, uses ~10 GB)
- CPU OOM: reduce video decoding batch size (unlikely in practice)
- Disk full: check `/data/group_data/maxlab/common_datasets/pandaliza/` has ~10 GB free

### Slow extraction (< 5 frames/sec)
- GPU not in use? Check CUDA_VISIBLE_DEVICES and sbatch logs
- Network I/O bottleneck: acceptable, extraction is I/O-bound (mp4 decoding)
- Expected throughput: 10 frames/sec → ~50 hours for 1.87M frames

---

## 6. Files & Locations

### Core Code
- **Extraction:** `steer_intent/vl_feature_cache.py` (VLFeatureExtractor, extract_all_features)
- **Reader:** `steer_intent/feature_cache.py` (FeatureCache class)

### Test & Verification
- **Model check:** `steer_intent/test_vl_model_access.py`
- **End-to-end test:** `steer_intent/test_feature_extraction.py`
- **Test cache (reference):** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features_test/`

### SLURM Scripts
- **Full extraction:** `slurm-scripts/b0b1/extract_features_full.sbatch`
- **Shard test (reference):** `slurm-scripts/b0b1/extract_features_shard_test.sbatch`

### Documentation
- **Full report:** `docs/b0b1/feature_cache_report.md`
- **This guide:** `docs/b0b1/feature_cache_quickstart.md`

---

## 7. Next: Training B0/B1

Once cache is ready, integrate into your training code:

1. Load FeatureCache
2. Instantiate CachedFeaturesDataset
3. In training loop: replace image encoding with cached feature loading
4. Expected speedup: 100× faster than on-the-fly SigLIP encoding

Example timeline:
- **Before cache:** images → encoder (~50ms per batch) → training
- **After cache:** cache lookup (~1ms per batch) → training
- **Speedup:** 50 hours training time → ~30 minutes model init + 30 minutes training on cached features
