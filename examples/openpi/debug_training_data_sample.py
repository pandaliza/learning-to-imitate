"""Sample training data to see actual action ranges and values.

This helps determine if:
1. Action dim 4 really has range [-2.0, 0.0] in demos
2. Actions 0-3 (base motion) are truly bounded [-1, 1]
3. The quantile stats reflect the actual demo data

Usage:
  python examples/openpi/debug_training_data_sample.py \\
    --data-root /data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target \\
    --out logs/debug_training_data_stats.json
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=100, help="Max episodes to sample")
    args = ap.parse_args()

    print(f"[Data Sample] Scanning RoboCasa dataset at {args.data_root}", flush=True)

    data_root = pathlib.Path(args.data_root)

    # Default 9 train tasks
    task_names = [
        "TurnOnElectricKettle",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "SlideDishwasherRack",
        "KettleBoiling",
        "LoadDishwasher",
        "PrepareCoffee",
        "PreSoakPan",
        "WashLettuce",
    ]

    all_actions = []
    all_states = []
    episode_count = 0

    for task_name in task_names:
        for category in ["atomic", "composite"]:
            task_dir = data_root / category / task_name
            if not task_dir.exists():
                continue

            for date_dir in sorted(task_dir.iterdir()):
                if not date_dir.is_dir() or episode_count >= args.max_episodes:
                    continue

                lerobot_dir = date_dir / "lerobot"
                if not lerobot_dir.exists():
                    continue

                data_dir = lerobot_dir / "data" / "chunk-000"
                meta_dir = lerobot_dir / "meta"
                episodes_path = meta_dir / "episodes.jsonl"

                if not episodes_path.exists() or not data_dir.exists():
                    continue

                with open(episodes_path) as f:
                    episodes_meta = [json.loads(line) for line in f]

                for ep_meta in episodes_meta:
                    if episode_count >= args.max_episodes:
                        break

                    ep_num = ep_meta["episode_index"]
                    parquet_path = data_dir / f"episode_{ep_num:06d}.parquet"

                    if not parquet_path.exists():
                        continue

                    try:
                        df = pd.read_parquet(parquet_path)
                    except Exception as e:
                        print(f"[WARN] Failed to load {parquet_path}: {e}")
                        continue

                    if "action" not in df.columns:
                        continue

                    # Extract actions
                    actions = np.array([np.asarray(a) for a in df["action"].values])
                    if len(actions) > 0 and len(actions[0]) >= 12:
                        actions = actions[:, :12]  # Take first 12 dims
                        all_actions.append(actions)

                    if "observation.state" in df.columns:
                        try:
                            states = np.array([np.asarray(s) for s in df["observation.state"].values])
                            all_states.append(states)
                        except:
                            pass

                    episode_count += 1
                    print(f"[Data Sample] Loaded {episode_count} episodes ({len(all_actions)} action arrays)", flush=True)

    if not all_actions:
        print("[ERROR] No action data found!")
        return

    # Concatenate all actions
    all_actions_array = np.concatenate(all_actions, axis=0)
    print(f"\n[Data Sample] Total action samples: {len(all_actions_array)}", flush=True)

    # Compute statistics per dimension
    report = {
        "episodes_sampled": episode_count,
        "total_samples": len(all_actions_array),
        "action_dim": 12,
        "per_dimension": {},
    }

    print("\n=== ACTION STATISTICS (12D) ===")
    for i in range(12):
        dim_data = all_actions_array[:, i]
        stats = {
            "mean": float(np.mean(dim_data)),
            "std": float(np.std(dim_data)),
            "q01": float(np.quantile(dim_data, 0.01)),
            "q05": float(np.quantile(dim_data, 0.05)),
            "q50": float(np.quantile(dim_data, 0.50)),
            "q95": float(np.quantile(dim_data, 0.95)),
            "q99": float(np.quantile(dim_data, 0.99)),
            "min": float(np.min(dim_data)),
            "max": float(np.max(dim_data)),
            "is_constant": float(np.std(dim_data)) < 1e-6,
        }
        report["per_dimension"][f"action[{i}]"] = stats

        const_mark = " [CONSTANT]" if stats["is_constant"] else ""
        print(
            f"  dim[{i}]: mean={stats['mean']:8.4f}, std={stats['std']:8.4f}, "
            f"[q01={stats['q01']:7.4f}, q99={stats['q99']:7.4f}] {const_mark}"
        )

    # Compare with our norm_stats
    print("\n=== COMPARISON WITH NORM_STATS.JSON ===")
    norm_stats_path = pathlib.Path("assets/pi05_robocasa_copred/robocasa/norm_stats.json")
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        ns_actions = norm_stats["norm_stats"]["actions"]
        for i in range(12):
            our_q01 = report["per_dimension"][f"action[{i}]"]["q01"]
            our_q99 = report["per_dimension"][f"action[{i}]"]["q99"]
            ns_q01 = ns_actions["q01"][i]
            ns_q99 = ns_actions["q99"][i]

            diff_q01 = abs(our_q01 - ns_q01)
            diff_q99 = abs(our_q99 - ns_q99)

            match = "✓" if diff_q01 < 0.1 and diff_q99 < 0.1 else "✗"
            print(
                f"  dim[{i}]: {match} data=[{our_q01:7.4f}, {our_q99:7.4f}] "
                f"vs norm_stats=[{ns_q01:7.4f}, {ns_q99:7.4f}]"
            )

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Data Sample] Report saved to {args.out}")


if __name__ == "__main__":
    main()
