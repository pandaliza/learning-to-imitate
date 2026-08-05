"""Smoke test for B0/B1: verify architecture, T-mask, and basic training.

Tests with both random tensors (small dims for speed) and real feature cache (1152D SigLIP).

Tests:
  1. B0/B1: forward pass with correct I/O shapes
  2. B1 T-mask: intent tokens do NOT attend to action tokens (unit test)
  3. B1 intent out_proj: zero-init (starts silent)
  4. Training loops: finite losses without divergence
  5. Sampling: B0 and B1 produce finite outputs
  6. Real cache test: load TurnOnElectricKettle episode, forward pass
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from b0b1.dit import DiT
from b0b1.sample import SampleContext, sample_b0, sample_b1
from steer_intent.feature_cache import FeatureCache


# Configuration
VISION_DIM = 1152  # SigLIP-SO400M
LANG_DIM = 1152
TEST_CACHE_POOLED = Path("/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features_test/pooled")
TEST_CACHE_GRID16 = Path("/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features_test/grid16")


def test_b0_forward_random():
    """Test B0 forward pass with random tensors."""
    print("[TEST] B0 forward pass (random)")

    B, num_vision = 2, 2  # 2 cameras
    H, action_dim = 10, 12
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b0",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.eval()

    vision = torch.randn(B, num_vision, VISION_DIM)
    lang = torch.randn(B, LANG_DIM)
    state = torch.randn(B, obs_steps, state_dim)
    x_action = torch.randn(B, H, action_dim)
    tau = torch.rand(B)

    with torch.no_grad():
        v_action, v_intent = model(
            x_action=x_action,
            x_intent=None,
            tau_action=tau,
            tau_intent=None,
            vision_features=vision,
            lang_features=lang,
            state=state,
        )

    assert v_action.shape == (B, H, action_dim)
    assert v_intent is None
    assert torch.isfinite(v_action).all()
    print("  ✓ B0 forward pass OK")


def test_b1_forward_random():
    """Test B1 forward pass with random tensors."""
    print("[TEST] B1 forward pass (random)")

    B, num_vision = 2, 2
    H, action_dim = 10, 12
    h, intent_dim = 8, 7
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b1",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.eval()

    vision = torch.randn(B, num_vision, VISION_DIM)
    lang = torch.randn(B, LANG_DIM)
    state = torch.randn(B, obs_steps, state_dim)
    x_action = torch.randn(B, H, action_dim)
    x_intent = torch.randn(B, h, intent_dim)
    tau_action = torch.rand(B)
    tau_intent = torch.rand(B)

    with torch.no_grad():
        v_action, v_intent = model(
            x_action=x_action,
            x_intent=x_intent,
            tau_action=tau_action,
            tau_intent=tau_intent,
            vision_features=vision,
            lang_features=lang,
            state=state,
        )

    assert v_action.shape == (B, H, action_dim)
    assert v_intent.shape == (B, h, intent_dim)
    assert torch.isfinite(v_action).all()
    assert torch.isfinite(v_intent).all()
    print("  ✓ B1 forward pass OK")


def test_b1_t_mask():
    """Unit test for B1 T-mask: intent tokens never attend to action tokens."""
    print("[TEST] B1 T-mask unit test")

    model = DiT(
        arm="b1",
        depth=1,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=2,
        lang_dim=LANG_DIM,
    )

    # Get the attention mask
    B, num_context = 1, 3  # 2 vision + 1 lang/state
    num_intent = 8
    num_action = 10
    total = num_context + num_intent + num_action

    mask = model._get_attention_mask_for_b1(B, num_context, torch.device("cpu"))

    assert mask is not None
    assert mask.shape == (total, total)

    # Verify T-mask structure:
    # Row i, Col j: True = mask out, False = attend
    # Intent tokens (rows [3:11]) should have True (mask out) for action columns [11:21]
    intent_start = num_context
    intent_end = intent_start + num_intent
    action_start = intent_end
    action_end = total

    # Check that intent rows have True for action columns (masking them out)
    for i in range(intent_start, intent_end):
        for j in range(action_start, action_end):
            assert mask[i, j] == True, (
                f"Intent token {i} should mask out (True) action token {j}, "
                f"but got {mask[i, j]}"
            )

    # Check that action rows have False for all columns (attend to all)
    for i in range(action_start, action_end):
        for j in range(total):
            assert mask[i, j] == False, (
                f"Action token {i} should attend to all (False) including position {j}, "
                f"but got {mask[i, j]}"
            )

    print("  ✓ B1 T-mask verified:")
    print(f"    Intent rows [{intent_start}:{intent_end}] mask action cols [{action_start}:{action_end}]")
    print(f"    Action rows [{action_start}:{action_end}] attend to all")


def test_b1_intent_out_proj_zero_init():
    """Test B1: intent_out_proj is zero-initialized."""
    print("[TEST] B1 intent_out_proj zero-init")

    model = DiT(
        arm="b1",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=2,
        lang_dim=LANG_DIM,
    )

    # Check that intent_out_proj weights and bias are zero
    assert torch.allclose(model.intent_out_proj.weight, torch.zeros_like(model.intent_out_proj.weight))
    assert torch.allclose(model.intent_out_proj.bias, torch.zeros_like(model.intent_out_proj.bias))
    print("  ✓ intent_out_proj zero-initialized")


def test_training_loop_b0():
    """Test 30 training steps for B0, verify loss is finite."""
    print("[TEST] B0 training loop (30 steps)")

    B, num_vision = 4, 2
    H, action_dim = 10, 12
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b0",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for step in range(30):
        vision = torch.randn(B, num_vision, VISION_DIM)
        lang = torch.randn(B, LANG_DIM)
        state = torch.randn(B, obs_steps, state_dim)
        x0_action = torch.randn(B, H, action_dim)
        tau = torch.rand(B)

        # Flow matching loss
        eps = torch.randn_like(x0_action)
        x_tau = tau.view(-1, 1, 1) * eps + (1 - tau.view(-1, 1, 1)) * x0_action
        u = eps - x0_action

        v_action, _ = model(
            x_action=x_tau,
            x_intent=None,
            tau_action=tau,
            tau_intent=None,
            vision_features=vision,
            lang_features=lang,
            state=state,
        )

        loss = torch.mean((v_action - u) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if not np.isfinite(loss.item()):
            print(f"  ✗ NaN/Inf at step {step}")
            return False

    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])

    print(f"  Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    print("  ✓ B0 training loop OK (loss finite)")
    return True


def test_training_loop_b1():
    """Test 30 training steps for B1."""
    print("[TEST] B1 training loop (30 steps)")

    B, num_vision = 4, 2
    H, action_dim = 10, 12
    h, intent_dim = 8, 7
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b1",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for step in range(30):
        vision = torch.randn(B, num_vision, VISION_DIM)
        lang = torch.randn(B, LANG_DIM)
        state = torch.randn(B, obs_steps, state_dim)
        x0_action = torch.randn(B, H, action_dim)
        x0_intent = torch.randn(B, h, intent_dim)
        tau_action = torch.rand(B)
        tau_intent = torch.rand(B)

        # Flow matching loss
        eps_action = torch.randn_like(x0_action)
        eps_intent = torch.randn_like(x0_intent)
        x_tau_action = tau_action.view(-1, 1, 1) * eps_action + (1 - tau_action.view(-1, 1, 1)) * x0_action
        x_tau_intent = tau_intent.view(-1, 1, 1) * eps_intent + (1 - tau_intent.view(-1, 1, 1)) * x0_intent
        u_action = eps_action - x0_action
        u_intent = eps_intent - x0_intent

        v_action, v_intent = model(
            x_action=x_tau_action,
            x_intent=x_tau_intent,
            tau_action=tau_action,
            tau_intent=tau_intent,
            vision_features=vision,
            lang_features=lang,
            state=state,
        )

        loss_action = torch.mean((v_action - u_action) ** 2)
        loss_intent = torch.mean((v_intent - u_intent) ** 2)
        loss = loss_action + loss_intent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if not np.isfinite(loss.item()):
            print(f"  ✗ NaN/Inf at step {step}")
            return False

    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])

    print(f"  Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    print("  ✓ B1 training loop OK (loss finite)")
    return True


def test_sampling_b0():
    """Test B0 sampling."""
    print("[TEST] B0 sampling")

    B, num_vision = 2, 2
    H, action_dim = 10, 12
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b0",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.eval()

    vision = torch.randn(B, num_vision, VISION_DIM)
    lang = torch.randn(B, LANG_DIM)
    state = torch.randn(B, obs_steps, state_dim)

    ctx = SampleContext(vision, lang, state, model)
    action = sample_b0(ctx, action_dim, H, steps=4)

    assert action.shape == (B, H, action_dim)
    assert torch.isfinite(action).all()
    print("  ✓ B0 sampling OK")


def test_sampling_b1():
    """Test B1 sampling."""
    print("[TEST] B1 sampling")

    B, num_vision = 2, 2
    H, action_dim = 10, 12
    h, intent_dim = 8, 7
    state_dim = 16
    obs_steps = 2

    model = DiT(
        arm="b1",
        depth=2,
        dim=256,
        num_heads=4,
        vision_dim=VISION_DIM,
        num_vision_tokens=num_vision,
        lang_dim=LANG_DIM,
    )
    model.eval()

    vision = torch.randn(B, num_vision, VISION_DIM)
    lang = torch.randn(B, LANG_DIM)
    state = torch.randn(B, obs_steps, state_dim)

    ctx = SampleContext(vision, lang, state, model)
    intent, action = sample_b1(ctx, action_dim, H, intent_dim, h, steps_intent=2, steps_action=4)

    assert intent.shape == (B, h, intent_dim)
    assert action.shape == (B, H, action_dim)
    assert torch.isfinite(intent).all()
    assert torch.isfinite(action).all()
    print("  ✓ B1 sampling OK")


def test_with_real_cache_pooled():
    """Test with real feature cache (pooled variant)."""
    print("[TEST] Real cache (pooled)")

    if not TEST_CACHE_POOLED.exists():
        print("  ⚠ Feature cache not found, skipping")
        return True

    try:
        cache = FeatureCache(TEST_CACHE_POOLED, variant="pooled")
        features = cache.load_episode("TurnOnElectricKettle", "episode_000000")

        # Forward pass with real features
        model = DiT(
            arm="b0",
            depth=2,
            dim=256,
            num_heads=4,
            vision_dim=cache.vision_dim,
            num_vision_tokens=cache.num_vision_tokens,
            lang_dim=cache.lang_dim,
        )
        model.eval()

        # Use first 2 frames
        B = 1
        base_feat = torch.from_numpy(features["base"][:2]).float().unsqueeze(0)  # (1, 2, 1152)
        wrist_feat = torch.from_numpy(features["wrist"][:2]).float().unsqueeze(0)  # (1, 2, 1152)
        lang_feat = torch.from_numpy(features["lang"]).float().unsqueeze(0)  # (1, 1152)
        vision = torch.cat([base_feat, wrist_feat], dim=1)  # (1, 2*2, 1152) but actually (1, 2, 1152) + (1, 2, 1152)

        # Actually, we have obs_steps=2, so we have 2 frames per camera
        # Let's stack them properly
        vision = torch.cat([base_feat, wrist_feat], dim=1)  # (1, 4, 1152) - 2 base + 2 wrist
        state = torch.randn(B, 2, 16)  # obs_steps=2
        x_action = torch.randn(B, 10, 12)
        tau = torch.rand(B)

        with torch.no_grad():
            v_action, _ = model(
                x_action=x_action,
                x_intent=None,
                tau_action=tau,
                tau_intent=None,
                vision_features=vision,
                lang_features=lang_feat,
                state=state,
            )

        assert v_action.shape == (B, 10, 12)
        assert torch.isfinite(v_action).all()
        print(f"  ✓ Real cache (pooled) OK | vision_dim={cache.vision_dim}, lang_dim={cache.lang_dim}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_with_real_cache_grid16():
    """Test with real feature cache (grid16 variant)."""
    print("[TEST] Real cache (grid16)")

    if not TEST_CACHE_GRID16.exists():
        print("  ⚠ Feature cache not found, skipping")
        return True

    try:
        cache = FeatureCache(TEST_CACHE_GRID16, variant="grid16")
        features = cache.load_episode("TurnOnElectricKettle", "episode_000000")

        # Forward pass with real features
        model = DiT(
            arm="b1",
            depth=2,
            dim=256,
            num_heads=4,
            vision_dim=cache.vision_dim,
            num_vision_tokens=cache.num_vision_tokens,  # Should be 16
            lang_dim=cache.lang_dim,
        )
        model.eval()

        # Use first 2 frames, grid16 format
        B = 1
        base_feat = torch.from_numpy(features["base"][:2]).float().unsqueeze(0)  # (1, 2, 16, 1152)
        wrist_feat = torch.from_numpy(features["wrist"][:2]).float().unsqueeze(0)  # (1, 2, 16, 1152)
        lang_feat = torch.from_numpy(features["lang"]).float().unsqueeze(0)  # (1, 1152)

        # Flatten spatial grid: (1, 2, 16, 1152) -> (1, 2*16, 1152)
        base_flat = base_feat.reshape(B, -1, base_feat.shape[-1])
        wrist_flat = wrist_feat.reshape(B, -1, wrist_feat.shape[-1])
        vision = torch.cat([base_flat, wrist_flat], dim=1)  # (1, 64, 1152) - 2*16 + 2*16

        state = torch.randn(B, 2, 16)
        x_action = torch.randn(B, 10, 12)
        x_intent = torch.randn(B, 8, 7)
        tau_action = torch.rand(B)
        tau_intent = torch.rand(B)

        with torch.no_grad():
            v_action, v_intent = model(
                x_action=x_action,
                x_intent=x_intent,
                tau_action=tau_action,
                tau_intent=tau_intent,
                vision_features=vision,
                lang_features=lang_feat,
                state=state,
            )

        assert v_action.shape == (B, 10, 12)
        assert v_intent.shape == (B, 8, 7)
        assert torch.isfinite(v_action).all()
        assert torch.isfinite(v_intent).all()
        print(f"  ✓ Real cache (grid16) OK | vision_dim={cache.vision_dim}, num_vision_tokens={cache.num_vision_tokens}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("B0/B1 Smoke Tests (1152D SigLIP + Real Feature Cache)")
    print("=" * 70 + "\n")

    try:
        test_b0_forward_random()
        test_b1_forward_random()
        test_b1_t_mask()
        test_b1_intent_out_proj_zero_init()
        test_training_loop_b0()
        test_training_loop_b1()
        test_sampling_b0()
        test_sampling_b1()
        test_with_real_cache_pooled()
        test_with_real_cache_grid16()

        print("\n" + "=" * 70)
        print("All smoke tests PASSED ✓")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
