"""A1 native parity test (Task 3).

Run the native evaluator path and the adapter with identical resets and noise.
Confirm predicted chunks match and rollouts match (10 episodes per arm).
Compare to saved performance (~62% for ACTION, ~64% for INTENT).

Author: Claude Haiku 4.5
Date: 2026-10-05
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.torch_utils import set_seed


def load_config_from_yaml() -> Config:
    """Load configuration for square_mh_state task."""
    # Would normally be done via Hydra
    # For now, we'll create a config object manually
    # This should be replaced with actual Hydra loading
    pass


def test_a1_parity():
    """Run A1 parity test.

    Steps:
    1. Load both ACTION and INTENT checkpoints
    2. Create environments and dataset
    3. Run 10 episodes per arm
    4. Compare predicted actions (normalized and unnormalized)
    5. Compare success rates
    6. Write results to CSV and JSON
    """

    output_dir = Path("outputs/intent_selection/audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("A1 NATIVE PARITY TEST")
    print("=" * 60)

    # Placeholder: actual implementation would:
    # 1. Create adapters for both checkpoints
    # 2. Load environments with fixed reset seeds
    # 3. Run evaluation with matched noise injection
    # 4. Log predicted actions and rollout outcomes
    # 5. Compare results

    results = {
        "test": "A1 native parity",
        "action_checkpoint": "checkpoints/square_mh_state_flow_mlp_512_h10_seed0_success62.pt",
        "intent_checkpoint": "checkpoints/square_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success64.pt",
        "n_episodes_per_arm": 10,
        "status": "placeholder - awaiting implementation after package installation",
        "expected_action_sr": 0.62,
        "expected_intent_sr": 0.64,
    }

    output_file = output_dir / "a1_parity_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPlaceholder results written to: {output_file}")
    return 0


if __name__ == "__main__":
    exit(test_a1_parity())
