"""Normalization audit: verify which norm_stats.json file was used at train vs eval time.

Traces the exact normalization path and compares quantile ranges to determine if
there's a train/eval mismatch in degenerate dimensions (base_motion, control_mode).

Usage:
  python examples/openpi/debug_norm_audit.py \\
    --pi05-config pi05_robocasa_copred \\
    --out logs/debug_norm_audit.json
"""

import argparse
import json
import pathlib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_robocasa_copred")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--assets-dir", default="assets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[Norm Audit] Tracing normalization for config {args.pi05_config}", flush=True)

    # Load openpi config
    try:
        import openpi.training.config as _config
    except ImportError as e:
        raise ImportError(f"openpi import failed: {e}")

    config = _config.get_config(args.pi05_config)
    print(f"[Norm Audit] Config loaded: {config.name}", flush=True)
    print(f"[Norm Audit] Assets dirs: {config.assets_dirs}", flush=True)
    print(f"[Norm Audit] Asset ID: {config.data.assets.asset_id}", flush=True)

    # Try to load norm stats the same way training does
    from openpi.training import checkpoints as _checkpoints
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(config.assets_dirs, data_config.asset_id)

    if norm_stats is None:
        print("[ERROR] norm_stats is None! Training would have failed the guard.")
        return

    print(f"\n[Norm Audit] Successfully loaded norm_stats")
    print(f"[Norm Audit] State shape: {len(norm_stats['state']['mean'])}D")
    print(f"[Norm Audit] Action shape: {len(norm_stats['actions']['mean'])}D")

    # Check for degenerate dimensions
    print("\n=== DEGENERATE STATE DIMENSIONS (q01 ≈ q99) ===")
    state_q01 = np.array(norm_stats["state"]["q01"])
    state_q99 = np.array(norm_stats["state"]["q99"])
    state_std = np.array(norm_stats["state"]["std"])

    for i in range(len(state_q01)):
        q_range = state_q99[i] - state_q01[i]
        if abs(q_range) < 1e-3:  # Essentially constant
            print(f"  State[{i}]: q01={state_q01[i]:.6f}, q99={state_q99[i]:.6f}, range={q_range:.6f}, std={state_std[i]:.6f}")

    print("\n=== DEGENERATE ACTION DIMENSIONS (q01 ≈ q99) ===")
    action_q01 = np.array(norm_stats["actions"]["q01"])
    action_q99 = np.array(norm_stats["actions"]["q99"])
    action_mean = np.array(norm_stats["actions"]["mean"])
    action_std = np.array(norm_stats["actions"]["std"])

    for i in range(len(action_q01)):
        q_range = action_q99[i] - action_q01[i]
        if abs(q_range) < 1e-3:  # Essentially constant
            print(f"  Action[{i}]: q01={action_q01[i]:.6f}, q99={action_q99[i]:.6f}, range={q_range:.6f}, mean={action_mean[i]:.6f}, std={action_std[i]:.6f}")

    # Highlight the problematic dims we see in eval logs
    print("\n=== FOCUS: ACTION DIMS 0-4 (base motion + control mode) ===")
    for i in range(min(5, len(action_q01))):
        q_range = action_q99[i] - action_q01[i]
        degenerate = abs(q_range) < 1e-3
        print(f"  Action[{i}]: q01={action_q01[i]:.6f}, q99={action_q99[i]:.6f}, range={q_range:.6f} {'[DEGENERATE]' if degenerate else ''}")

    # Save detailed report
    report = {
        "config": args.pi05_config,
        "assets_dirs": config.assets_dirs,
        "asset_id": data_config.asset_id,
        "norm_stats_loaded": True,
        "state_dim": len(state_q01),
        "action_dim": len(action_q01),
        "degenerate_state_dims": [],
        "degenerate_action_dims": [],
        "warning": None,
    }

    # Identify degenerate dims
    for i in range(len(state_q01)):
        q_range = state_q99[i] - state_q01[i]
        if abs(q_range) < 1e-3:
            report["degenerate_state_dims"].append(i)

    for i in range(len(action_q01)):
        q_range = action_q99[i] - action_q01[i]
        if abs(q_range) < 1e-3:
            report["degenerate_action_dims"].append(i)

    if report["degenerate_action_dims"]:
        report["warning"] = (
            f"CRITICAL: Action dims {report['degenerate_action_dims']} have degenerate quantiles! "
            "Quantile normalization would map all values to the same point, causing loss spikes. "
            "This matches the observed 0% SR pattern."
        )

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Norm Audit] Report saved to {args.out}")

    if report["warning"]:
        print(f"\n!!! {report['warning']}")


if __name__ == "__main__":
    main()
