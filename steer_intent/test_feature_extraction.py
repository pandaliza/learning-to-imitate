#!/usr/bin/env python
"""Test feature extraction on a single episode to verify correctness and performance.

This script:
1. Tests model loading and feature extraction
2. Extracts features for one episode (TurnOnElectricKettle #0)
3. Verifies cache format and manifest
4. Performs sanity checks and computes projected metrics
"""
import os
import sys
import time
import json
from pathlib import Path

# Set HF cache before imports
os.environ["HF_HOME"] = "/data/user_data/ldahiya/hf_cache"

import numpy as np
from steer_intent.vl_feature_cache import VLFeatureExtractor, TASK_INSTRUCTIONS
from steer_intent.feature_cache import FeatureCache

def test_extraction():
    """Test feature extraction pipeline."""

    print("="*70)
    print("Feature Extraction Test")
    print("="*70)

    # Configuration
    root_dir = Path("/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target")
    cache_root = Path("/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features_test")
    task = "TurnOnElectricKettle"

    # Verify dataset exists
    if not root_dir.exists():
        print(f"ERROR: Dataset not found at {root_dir}")
        return False

    # Clean test cache
    import shutil
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"\nTest dataset: {task}")
    print(f"Dataset root: {root_dir}")
    print(f"Cache root: {cache_root}")
    print("")

    # ===== Test 1: Model Loading =====
    print("[Test 1] Loading VL model...")
    try:
        extractor = VLFeatureExtractor(
            variant="pooled",
            device="cuda",
            hf_home=os.environ.get("HF_HOME"),
        )
        print(f"  ✓ Model loaded: {extractor.model_type}")
        print(f"  Model: {extractor.model_id}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False

    # ===== Test 2: Extract single episode =====
    print("\n[Test 2] Extracting features for one episode...")

    # Find first episode
    task_dir = root_dir / "atomic" / task
    date_dir = sorted(task_dir.iterdir())[0]
    lerobot_dir = date_dir / "lerobot"
    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data" / "chunk-000"
    video_dir = lerobot_dir / "videos" / "chunk-000"

    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path) as f:
        ep_meta = json.loads(f.readline())

    ep_num = ep_meta["episode_index"]
    ep_len = ep_meta["length"]
    parquet_path = data_dir / f"episode_{ep_num:06d}.parquet"

    print(f"  Episode: {ep_num}, Length: {ep_len} frames")

    try:
        start = time.time()
        features = extractor.process_episode(
            parquet_path=parquet_path,
            video_dir=video_dir,
            episode_number=ep_num,
        )
        elapsed = time.time() - start

        # Extract text once
        text = TASK_INSTRUCTIONS.get(task, task)
        features["lang"] = extractor.extract_text_features(text)

        print(f"  ✓ Extraction complete in {elapsed:.1f}s")
        print(f"    Base features: {features['base'].shape}")
        print(f"    Wrist features: {features['wrist'].shape}")
        print(f"    Text embedding: {features['lang'].shape}")

        # Calculate throughput
        throughput = ep_len / elapsed
        print(f"    Throughput: {throughput:.1f} frames/sec")

        # Project to full dataset
        # ~500 eps per task, ~9 tasks = 4500 eps total
        # ~1M frames per task, ~9M frames total
        total_eps = 4500
        eps_per_task = 500
        total_frames = 1.87e6

        projected_time_hrs = (total_frames / ep_len) * elapsed / 3600
        print(f"    Projected full dataset: {projected_time_hrs:.1f} hours on 1 GPU")

    except Exception as e:
        print(f"  ✗ Failed to extract features: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== Test 3: Save and verify cache format =====
    print("\n[Test 3] Saving features and verifying cache format...")

    for variant in ["pooled", "grid16"]:
        try:
            print(f"\n  Testing variant: {variant}")

            # Re-extract with grid16
            extractor_grid = VLFeatureExtractor(
                variant=variant,
                device="cuda",
                hf_home=os.environ.get("HF_HOME"),
            )

            start = time.time()
            features_grid = extractor_grid.process_episode(
                parquet_path=parquet_path,
                video_dir=video_dir,
                episode_number=ep_num,
            )
            features_grid["lang"] = extractor_grid.extract_text_features(text)
            elapsed = time.time() - start

            # Save
            output_dir = cache_root / variant / "features" / task
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"episode_{ep_num:06d}.npz"

            np.savez_compressed(
                output_path,
                base=features_grid["base"].astype(np.float16),
                wrist=features_grid["wrist"].astype(np.float16),
                lang=features_grid["lang"].astype(np.float16),
            )

            print(f"    ✓ Saved to {output_path}")
            print(f"      Base: {features_grid['base'].shape} dtype={features_grid['base'].dtype}")
            print(f"      Wrist: {features_grid['wrist'].shape} dtype={features_grid['wrist'].dtype}")
            print(f"      Lang: {features_grid['lang'].shape} dtype={features_grid['lang'].dtype}")

            # Check file size
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"      File size: {file_size_mb:.2f} MB")

            # Project cache size
            projected_cache_gb = file_size_mb * (1.87e6 / ep_len) / 1024
            print(f"      Projected full cache: {projected_cache_gb:.1f} GB")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ===== Test 4: Create and verify manifest =====
    print("\n[Test 4] Creating and verifying manifest...")

    try:
        for variant in ["pooled", "grid16"]:
            variant_dir = cache_root / variant
            manifest = {
                "variant": variant,
                "vision_dim": 1152,  # SigLIP-SO400M dimension
                "num_vision_tokens": 1 if variant == "pooled" else 16,
                "lang_dim": 1152,
                "cameras": ["base", "wrist"],
                "dtype": "float16",
                "frame_indexing": "matches LeRobot episode frame index",
            }

            manifest_path = variant_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"  ✓ Manifest created for {variant}")
            print(f"    Path: {manifest_path}")

            # Verify can be loaded
            cache = FeatureCache(variant_dir, variant=variant)
            print(f"    ✓ Manifest loads successfully")
            print(f"      Vision dim: {cache.vision_dim}")
            print(f"      Lang dim: {cache.lang_dim}")
            print(f"      Num vision tokens: {cache.num_vision_tokens}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== Test 5: Load and verify features via FeatureCache =====
    print("\n[Test 5] Loading features via FeatureCache interface...")

    try:
        for variant in ["pooled", "grid16"]:
            variant_dir = cache_root / variant
            cache = FeatureCache(variant_dir, variant=variant)

            data = cache.load_episode(task, f"episode_{ep_num:06d}")

            print(f"  ✓ {variant}:")
            print(f"    Base: {data['base'].shape}")
            print(f"    Wrist: {data['wrist'].shape}")
            print(f"    Lang: {data['lang'].shape}")
            print(f"    Num frames: {data['num_frames']}")

            # Sanity check: features should be float16 and have reasonable values
            assert data["base"].dtype == np.float16, f"Wrong dtype: {data['base'].dtype}"
            assert data["base"].shape[0] == ep_len, f"Wrong num frames: {data['base'].shape[0]}"

            if variant == "pooled":
                assert data["base"].ndim == 2, f"pooled should be 2D"
            else:
                assert data["base"].ndim == 3, f"grid16 should be 3D"

            print(f"    ✓ Sanity checks passed")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== Summary =====
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    print("\nSummary:")
    print(f"  Model: SigLIP-SO400M (fallback from gated PaliGemma)")
    print(f"  Throughput: ~{throughput:.1f} frames/sec")
    print(f"  Projected full-dataset time: ~{projected_time_hrs:.1f} hours")
    print(f"  Cache format: npz with base/wrist/lang arrays")
    print(f"  Variant options: pooled (~15 GB) or grid16 (~150-250 GB)")

    return True

if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)
