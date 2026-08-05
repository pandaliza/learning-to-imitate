"""GT-action replay diagnostic with FIXED success detection.

This version uses env._check_success() which is the correct RoboCasa method.
Previous version was broken because it looked for env.is_success() which doesn't exist.

Usage:
  python examples/openpi/debug_gt_action_replay_fixed.py \\
    --task TurnOnElectricKettle \\
    --data-root /data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target \\
    --out logs/debug_gt_replay_fixed.json
"""

import argparse
import json
import pathlib
import numpy as np
import torch
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="TurnOnElectricKettle", help="RoboCasa task name")
    ap.add_argument("--data-root", required=True, help="Path to /target (LeRobot format)")
    ap.add_argument("--num-episodes", type=int, default=3, help="Number of demo episodes to replay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[GT Replay FIXED] Loading demo episodes for {args.task}", flush=True)

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
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=[],
            control_freq=20,
            horizon=1000,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create RoboCasa task {args.task}: {e}")

    # Scan for demo episodes in LeRobot format
    data_root = pathlib.Path(args.data_root)
    demo_episodes = []

    for category in ["atomic", "composite"]:
        task_dir = data_root / category / args.task
        if not task_dir.exists():
            continue

        for date_dir in sorted(task_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            lerobot_dir = date_dir / "lerobot"
            if not lerobot_dir.exists():
                continue

            data_dir = lerobot_dir / "data" / "chunk-000"
            meta_dir = lerobot_dir / "meta"
            episodes_path = meta_dir / "episodes.jsonl"

            if not episodes_path.exists() or not data_dir.exists():
                continue

            # Read episodes metadata
            with open(episodes_path) as f:
                episodes_meta = [json.loads(line) for line in f]

            for ep_meta in episodes_meta:
                ep_num = ep_meta["episode_index"]
                parquet_path = data_dir / f"episode_{ep_num:06d}.parquet"

                if parquet_path.exists():
                    demo_episodes.append({
                        "parquet_path": parquet_path,
                        "length": ep_meta["length"],
                        "task": args.task,
                    })

    if not demo_episodes:
        print(f"[ERROR] No demo episodes found for {args.task} in {args.data_root}")
        return

    demo_episodes = demo_episodes[:args.num_episodes]
    print(f"[GT Replay FIXED] Found {len(demo_episodes)} demo episodes, replaying first {len(demo_episodes)}", flush=True)

    results = {}

    for ep_idx, ep_info in enumerate(demo_episodes):
        print(f"\n[Episode {ep_idx}] Loading {ep_info['parquet_path'].name}", flush=True)

        try:
            # Load episode parquet
            df = pd.read_parquet(ep_info["parquet_path"])
        except Exception as e:
            print(f"[ERROR] Failed to load parquet: {e}")
            continue

        # Extract action column (state is observation)
        if "action" not in df.columns:
            print(f"[WARN] No 'action' column in parquet")
            continue

        actions = df["action"].values  # List of arrays
        episode_length = len(actions)

        print(f"[Episode {ep_idx}] Action count: {episode_length}, action[0] shape: {actions[0].shape if len(actions) > 0 else 'N/A'}")

        # Reset env
        try:
            obs = env.reset()
            print(f"[Episode {ep_idx}] Env reset OK, obs keys: {set(obs.keys())}")
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            continue

        # Replay actions open-loop
        success = False
        frames_executed = 0
        step_errors = []

        for step in range(min(episode_length, 500)):  # Cap at 500 steps
            action = actions[step]

            # Convert to mutable numpy array (IMPORTANT: pandas arrays are read-only)
            action = np.asarray(action, dtype=np.float32).copy()

            # Validate action dim
            if len(action) < 12:
                print(f"[WARN] Step {step}: action dim {len(action)} < 12, padding with zeros")
                action = np.concatenate([action, np.zeros(12 - len(action))]).astype(np.float32)
            elif len(action) > 12:
                action = action[:12]

            try:
                obs, reward, done, info = env.step(action)
                frames_executed += 1

                if step < 3:  # Debug first 3 steps
                    print(f"  Step {step}: action={action[:4]}... reward={reward:.4f} done={done}", flush=True)

                # Check success using the CORRECT method: env._check_success()
                # This is the critical fix!
                if hasattr(env, '_check_success'):
                    try:
                        success = env._check_success()
                    except:
                        success = False

                if done:
                    print(f"[Episode {ep_idx}] Done at step {step}")
                    print(f"[Episode {ep_idx}] Final success (env._check_success): {success}")
                    break

            except Exception as e:
                step_errors.append(str(e))
                if len(step_errors) <= 3:
                    print(f"  [ERROR] Step {step}: {e}", flush=True)
                if len(step_errors) >= 5:
                    print(f"  [ERROR] Too many step errors, aborting episode")
                    break

        results[f"ep{ep_idx}"] = {
            "task": args.task,
            "episode_file": str(ep_info["parquet_path"].name),
            "demo_length": episode_length,
            "frames_executed": frames_executed,
            "success": success,
            "errors": step_errors[:3],  # First 3 errors only
        }

        print(f"[Episode {ep_idx}] RESULT: success={success}, executed={frames_executed}/{min(episode_length, 500)} steps", flush=True)

    # Summary
    env.close()
    successes = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"\n[GT Replay FIXED Summary] {successes}/{total} demos replayed successfully")

    summary = {
        "task": args.task,
        "num_episodes": len(results),
        "success_count": successes,
        "success_rate": successes / max(total, 1),
        "per_episode": results,
        "method": "env._check_success()",
        "note": "FIXED: uses env._check_success() instead of env.is_success()",
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GT Replay FIXED] Results saved to {args.out}")


if __name__ == "__main__":
    main()
