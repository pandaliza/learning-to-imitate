#!/usr/bin/env python
"""Quick test to verify PaliGemma or SigLIP model access."""
import os
import sys

# Set HF cache
os.environ["HF_HOME"] = "/data/user_data/ldahiya/hf_cache"

print("Testing VL model access...")
print(f"HF_HOME={os.environ.get('HF_HOME')}")

try:
    from transformers import AutoModel, AutoProcessor, PaliGemmaProcessor

    print("\n1. Attempting to load PaliGemma-3B...")
    try:
        processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
        print("   ✓ PaliGemmaProcessor loaded")

        # Don't load full model in test (saves time/memory)
        print("   ✓ PaliGemma-3B is available via HuggingFace")
        model_used = "paligemma"
    except Exception as e:
        print(f"   ✗ PaliGemma load failed: {e}")
        model_used = None

    if model_used is None:
        print("\n2. Attempting fallback to SigLIP...")
        try:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            print("   ✓ SigLIP-SO400M is available via HuggingFace")
            model_used = "siglip"
        except Exception as e:
            print(f"   ✗ SigLIP load failed: {e}")
            model_used = None

    if model_used:
        print(f"\n✓ Using {model_used.upper()} for feature extraction")
        sys.exit(0)
    else:
        print("\n✗ Neither PaliGemma nor SigLIP available")
        sys.exit(1)

except ImportError as e:
    print(f"ERROR: transformers not installed: {e}")
    sys.exit(1)
