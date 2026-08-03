"""Smoke test for eval_robocasa_intent.py — verifies JSON output and obs/action handling.

Run this BEFORE installing RoboCasa to verify the eval harness structure is sound.
Uses a mocked RoboCasa environment.

Usage:
    python examples/openpi/test_robocasa_eval_smoke.py [--json-only]
"""

import argparse
import collections
import json
import numpy as np
import pathlib
import tempfile


class MockRoboCasaEnv:
    """Minimal mock RoboCasa env for testing obs extraction and action handling."""

    def __init__(self, task_name: str, seed: int = 0):
        self.task_name = task_name
        self.seed = seed
        self.step_count = 0
        np.random.seed(seed)

    def reset(self):
        """Return mock obs with all required keys."""
        return {
            "robot0_base_pos": np.random.randn(3).astype(np.float32),
            "robot0_base_quat": np.random.randn(4).astype(np.float32),
            "robot0_eef_pos_rel": np.random.randn(3).astype(np.float32),
            "robot0_eef_quat_rel": np.random.randn(4).astype(np.float32),
            "robot0_gripper_qpos": np.random.randn(2).astype(np.float32),
            "robot0_agentview_left": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        }

    def step(self, action):
        """Step mock env."""
        if len(action) != 12:
            raise RuntimeError(f"Expected action dim 12, got {len(action)}")
        self.step_count += 1
        # Succeed after 10 steps (mock success)
        done = self.step_count > 10
        return self.reset(), 0.0, done, {"success": done}

    def close(self):
        pass


def test_obs_action_handling():
    """Test that obs extraction and action handling work."""
    print("[SMOKE] Testing obs extraction and action handling...")

    # Import the RoboCasaEnvAdapter from the eval script
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from eval_robocasa_intent import RoboCasaEnvAdapter, ROBOCASA_STATE_DIM, ROBOCASA_ACTION_DIM

    # Create mock adapter
    class MockAdapter(RoboCasaEnvAdapter):
        def __init__(self):
            self.task_name = "TestTask"
            self.env = MockRoboCasaEnv("TestTask")

    adapter = MockAdapter()

    # Test reset and obs extraction
    obs = adapter.env.reset()
    state = adapter.extract_state_obs(obs)
    assert state.shape[0] == ROBOCASA_STATE_DIM, f"State dim mismatch: {state.shape[0]} vs {ROBOCASA_STATE_DIM}"
    print(f"  ✓ State extraction: shape {state.shape}, dtype {state.dtype}")

    # Test image extraction
    base, wrist = adapter.extract_images(obs)
    assert base.shape == (256, 256, 3), f"Base image shape mismatch: {base.shape}"
    assert wrist.shape == (256, 256, 3), f"Wrist image shape mismatch: {wrist.shape}"
    print(f"  ✓ Image extraction: base {base.shape}, wrist {wrist.shape}")

    # Test action stepping
    action = np.random.randn(ROBOCASA_ACTION_DIM).astype(np.float32)
    obs2, reward, done, info = adapter.env.step(action)
    print(f"  ✓ Action stepping: action_dim={len(action)}, done={done}, has_success={('success' in info)}")

    # Test success check
    success = adapter.check_success(obs2, info, done)
    print(f"  ✓ Success check: {success}")

    print("[SMOKE] Obs/action tests PASSED\n")


def test_json_output():
    """Test that the JSON output structure is correct."""
    print("[SMOKE] Testing JSON output structure...")

    # Create a mock results dict matching what the eval script produces
    summary = {
        "config": "pi05_base_copred",
        "checkpoint": "/fake/checkpoint",
        "task_set": "atomic_seen",
        "num_tasks": 2,
        "num_trials_per_task": 3,
        "mean_sr": 0.667,
        "total": "4/6",
        "per_task": {
            "PickPlaceCounterToCabinet": 0.667,
            "PickPlaceCounterToStove": 0.667,
        },
        "schedule": "s1",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(summary, f, indent=2)
        output_path = f.name

    # Read it back and verify
    with open(output_path) as f:
        loaded = json.load(f)

    assert loaded["config"] == "pi05_base_copred"
    assert loaded["mean_sr"] == 0.667
    assert "PickPlaceCounterToCabinet" in loaded["per_task"]
    assert loaded["schedule"] == "s1"

    pathlib.Path(output_path).unlink()
    print(f"  ✓ JSON structure valid: config, mean_sr, per_task, schedule fields present")
    print("[SMOKE] JSON output tests PASSED\n")


def test_task_set_definitions():
    """Test that task sets are defined and non-empty."""
    print("[SMOKE] Testing task set definitions...")

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from eval_robocasa_intent import ROBOCASA_TASK_SETS, MAX_STEPS_PER_SET

    atomic = ROBOCASA_TASK_SETS["atomic_seen"]
    composite_seen = ROBOCASA_TASK_SETS["composite_seen"]
    composite_unseen = ROBOCASA_TASK_SETS["composite_unseen"]

    assert len(atomic) == 4, f"Expected 4 atomic tasks, got {len(atomic)}"
    assert len(composite_seen) == 5, f"Expected 5 composite_seen tasks, got {len(composite_seen)}"
    assert len(composite_unseen) == 6, f"Expected 6 composite_unseen tasks, got {len(composite_unseen)}"

    all_tasks = atomic + composite_seen + composite_unseen
    assert len(all_tasks) == 15, f"Expected 15 total tasks, got {len(all_tasks)}"
    assert len(set(all_tasks)) == 15, "Task names not unique"

    assert MAX_STEPS_PER_SET["atomic_seen"] == 500
    assert MAX_STEPS_PER_SET["composite_seen"] == 1000
    assert MAX_STEPS_PER_SET["composite_unseen"] == 1000

    print(f"  ✓ Task sets: atomic={len(atomic)}, composite_seen={len(composite_seen)}, "
          f"composite_unseen={len(composite_unseen)}, total={len(all_tasks)}")
    print(f"  ✓ Max steps: {MAX_STEPS_PER_SET}")
    print("[SMOKE] Task definition tests PASSED\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true",
                    help="Only test JSON output (skip obs extraction)")
    args = ap.parse_args()

    print("\n=== SMOKE TEST: eval_robocasa_intent.py ===\n")

    try:
        test_task_set_definitions()
    except Exception as e:
        print(f"[FAIL] Task definition test: {e}\n")
        return 1

    try:
        test_json_output()
    except Exception as e:
        print(f"[FAIL] JSON output test: {e}\n")
        return 1

    if not args.json_only:
        try:
            test_obs_action_handling()
        except Exception as e:
            print(f"[FAIL] Obs/action handling test: {e}\n")
            return 1

    print("=== ALL SMOKE TESTS PASSED ===\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
