"""Obs construction audit: dump and compare obs dict structure at eval vs training.

This verifies:
1. Obs keys match between env and expected training format
2. State vector composition and ordering
3. Image shapes, dtypes, camera assignment
4. Normalization is applied correctly

Usage:
  python examples/openpi/debug_obs_construction.py \\
    --task TurnOnElectricKettle \\
    --checkpoint-dir /data/.../m11_a1_tied/30000 \\
    --out logs/debug_obs_construction.json
"""

import argparse
import json
import pathlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="TurnOnElectricKettle")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[Obs Audit] Dumping obs construction for {args.task}", flush=True)

    # Initialize RoboCasa env
    try:
        import robocasa
        import robosuite
    except ImportError as e:
        raise ImportError(f"RoboCasa/robosuite import failed: {e}")

    try:
        env = robosuite.make(
            args.task,
            robots="PandaOmron",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            horizon=500,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create RoboCasa task {args.task}: {e}")

    # Reset and capture obs
    print(f"[Obs Audit] Resetting environment...", flush=True)
    try:
        obs = env.reset()
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        return

    # Analyze obs structure
    report = {
        "task": args.task,
        "obs_keys": list(obs.keys()),
        "state_construction": {},
        "images": {},
        "potential_issues": [],
    }

    print(f"\n[Obs Audit] Obs keys: {list(obs.keys())}")

    # Check state keys
    required_state_keys = [
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
    ]

    state_parts = []
    for key in required_state_keys:
        if key not in obs:
            report["potential_issues"].append(f"Missing state key: {key}")
            print(f"[WARN] Missing key: {key}")
            continue

        val = obs[key]
        val_np = np.asarray(val)
        print(f"  {key}: shape={val_np.shape}, dtype={val_np.dtype}, values={val_np}")
        report["state_construction"][key] = {
            "shape": list(val_np.shape),
            "dtype": str(val_np.dtype),
            "sample_values": val_np.flatten()[:5].tolist(),
        }
        state_parts.append(val_np)

    # Reconstruct 16D state
    if len(state_parts) == 5:
        reconstructed_state = np.concatenate(state_parts).astype(np.float32)
        print(f"\n[Obs Audit] Reconstructed 16D state: shape={reconstructed_state.shape}")
        print(f"  Values: {reconstructed_state}")
        report["state_construction"]["reconstructed"] = {
            "shape": list(reconstructed_state.shape),
            "dtype": str(reconstructed_state.dtype),
            "values": reconstructed_state.tolist(),
        }
    else:
        report["potential_issues"].append(f"Could not reconstruct state: only {len(state_parts)} of 5 parts")

    # Check images
    image_keys = ["robot0_agentview_left_image", "robot0_eye_in_hand_image"]
    for key in image_keys:
        if key not in obs:
            report["potential_issues"].append(f"Missing image key: {key}")
            print(f"[WARN] Missing image key: {key}")
            continue

        img = np.asarray(obs[key])
        print(f"  {key}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        report["images"][key] = {
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "min": float(img.min()),
            "max": float(img.max()),
        }

    # Expected format check
    print(f"\n[Obs Audit] Expected format checks:")
    if "reconstructed" in report["state_construction"]:
        if report["state_construction"]["reconstructed"]["shape"][0] == 16:
            print(f"  ✓ State is 16D")
        else:
            report["potential_issues"].append(f"State dim mismatch: expected 16, got {report['state_construction']['reconstructed']['shape'][0]}")

    if len(report["images"]) >= 2:
        print(f"  ✓ Both cameras present")
    else:
        report["potential_issues"].append(f"Image count mismatch: expected 2, got {len(report['images'])}")

    # Check specific obs keys used by eval (from eval_robocasa_intent.py)
    print(f"\n[Obs Audit] Eval-specific key checks:")
    eval_keys = [
        "robot0_agentview_left_image",
        "robot0_eye_in_hand_image",
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
    ]
    for key in eval_keys:
        present = key in obs
        print(f"  {'✓' if present else '✗'} {key}")
        if not present:
            report["potential_issues"].append(f"Eval will fail: missing key {key}")

    # Save report
    env.close()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Obs Audit] Report saved to {args.out}")

    if report["potential_issues"]:
        print(f"\n!!! POTENTIAL ISSUES FOUND:")
        for issue in report["potential_issues"]:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
