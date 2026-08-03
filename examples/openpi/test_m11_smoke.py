"""Smoke test for M11 (RoboCasa co-prediction trainer).

Runs a few training steps with a fake dataset to verify:
  1. Shape compatibility (state 16D, action 12D, intent 7D)
  2. Co-prediction forward pass (both action and intent losses computed)
  3. Backward pass and gradient updates work
  4. Both tied and decoupled noise schedules work

Test scenario: A0 pure-BC, A1 tied, A2 decoupled (joint mask).
Requires: --pi05-weights path to pi05_base PyTorch checkpoint.

Usage:
  python examples/openpi/test_m11_smoke.py \\
    --pi05-weights /path/to/pi05_base/params \\
    --out /tmp/m11_smoke_test

Expected output:
  - Loss decreases over steps (backward pass working)
  - No shape mismatches (data pipeline correct)
  - Peak GPU memory recorded (if GPU available)
"""

import argparse
import os
import sys
import tempfile
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-weights", required=True, help="path to pi05_base PyTorch params")
    ap.add_argument("--out", default="/tmp/m11_smoke_test")
    ap.add_argument("--steps", type=int, default=5, help="number of training steps for smoke test")
    ap.add_argument("--batch-size", type=int, default=2, help="batch size for smoke test")
    ap.add_argument("--device", default="cuda" if os.path.exists("/proc/driver/nvidia") else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Test configurations: A0 (pure BC), A1 (tied), A2 (decoupled)
    test_configs = [
        ("A0_pure_bc", {
            "--intent-weight": "0.0",
            "--train-intent-only": "False",
        }),
        ("A1_tied", {
            "--intent-weight": "1.0",
            "--tied": "",
            "--train-intent-only": "True",
        }),
        ("A2_decoupled_joint", {
            "--intent-weight": "1.0",
            "--train-intent-only": "True",
        }),
    ]

    results = []
    for test_name, extra_args in test_configs:
        print(f"\n{'='*60}")
        print(f"Running smoke test: {test_name}")
        print(f"{'='*60}")

        out_dir = os.path.join(args.out, test_name)
        os.makedirs(out_dir, exist_ok=True)

        # Build command
        cmd = [
            sys.executable, "examples/openpi/train_pi05_m11.py",
            "--pi05-config", "pi05_robocasa_copred",
            "--pi05-weights", args.pi05_weights,
            "--config-dir", "examples/configs",
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--save-every", str(max(args.steps + 1, 100)),  # Don't save during smoke test
            "--out", out_dir,
            "--device", args.device,
            "--test-fake-dataset",
        ]

        # Add test-specific args
        for key, val in extra_args.items():
            cmd.append(key)
            if val:
                cmd.append(val)

        print(f"Command: {' '.join(cmd)}\n")

        # Run the trainer
        try:
            result = subprocess.run(cmd, cwd="/home/ldahiya/max_vla/much-ado-about-noising", timeout=600)
            if result.returncode == 0:
                print(f"\n[OK] {test_name} completed successfully")
                results.append((test_name, "PASS"))
            else:
                print(f"\n[FAIL] {test_name} exited with code {result.returncode}")
                results.append((test_name, "FAIL"))
        except subprocess.TimeoutExpired:
            print(f"\n[TIMEOUT] {test_name} timed out after 600s")
            results.append((test_name, "TIMEOUT"))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            results.append((test_name, "ERROR"))

    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for test_name, status in results:
        print(f"{test_name:30} {status:10}")

    all_pass = all(status == "PASS" for _, status in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
