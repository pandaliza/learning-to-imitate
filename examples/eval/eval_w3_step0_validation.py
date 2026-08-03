"""W3 step-0 validation harness for square-mh-state BC policies.

Validates the two base policies (plain flow BC and flow-intent BC) before RL.
Metrics:
  (a) 50-rollout SR for each (target window 30–70%)
  (b) VDR/diversity probe at critical states
  (c) Intent-intervention probe: sample K intents, measure endpoint dispersion

Usage (post-training, when checkpoints exist):
    python examples/eval/eval_w3_step0_validation.py \\
        --baseline-ckpt checkpoints/square_mh_baseline.pt \\
        --flow-intent-ckpt checkpoints/square_mh_flow_intent.pt \\
        --n-rollouts 50 \\
        --out results/w3_step0_validation.json

Or use the wrapper for metric (a) only:
    python examples/eval/eval_success_rate.py \\
        --run "baseline:ckpt:task=square_mh_state" \\
        --run "flow_intent:ckpt:task=square_mh_state_flow_intent:+network.arch_variant=flow_intent" \\
        --n-rollouts 50 \\
        --out results/w3_step0_sr.csv
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from mip.torch_utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="W3 step-0 validation: baseline + flow-intent on square-mh-state. "
                    "Placeholder for SR/VDR/intent-intervention probes."
    )
    parser.add_argument("--baseline-ckpt", required=True, help="Path to baseline checkpoint")
    parser.add_argument("--flow-intent-ckpt", required=True, help="Path to flow-intent checkpoint")
    parser.add_argument("--task", default="square_mh_state", help="Task config name")
    parser.add_argument("--n-rollouts", type=int, default=50, help="Number of rollouts for SR")
    parser.add_argument("--n-critical", type=int, default=10, help="Number of critical states for probes")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Placeholder results documenting the validation gates and their usage
    results = {
        "w3_step0_validation": {
            "task": args.task,
            "n_rollouts": args.n_rollouts,
            "n_critical_states": args.n_critical,
            "baseline_checkpoint": str(args.baseline_ckpt),
            "flow_intent_checkpoint": str(args.flow_intent_ckpt),
            "status": "scaffold_ready_for_implementation",
            "gates": {
                "gate_a_sr": {
                    "description": "Both base policies land in ~30-70% SR (difficulty-matched for fair RL comparison)",
                    "metric": "success_rate",
                    "threshold": "[0.30, 0.70]",
                    "tool": "examples/eval/eval_success_rate.py",
                },
                "gate_b_vdr": {
                    "description": "Flow-intent retains structured diversity (VDR > 0.35 on square-mh, prior: 0.71 on square-ph)",
                    "metric": "vdr",
                    "threshold": "> 0.35",
                    "tool": "examples/collect/collect_multimodal_viz.py + analyze_multimodal_clustering.py",
                },
                "gate_c_intent_intervention": {
                    "description": "Intent interventions cause observable behavior changes (endpoint dispersion from intent > from action noise alone)",
                    "metric": "intent_dispersion_ratio",
                    "threshold": "> 2.0",
                    "tool": "examples/eval/eval_w3_step0_validation.py (implementation)",
                },
            },
            "usage_notes": [
                "Metric (a): Use eval_success_rate.py with correct --run specs (see docstring).",
                "Metric (b): Use collect_multimodal_viz.py for K=50 samples, N=10 critical states, then analyze_multimodal_clustering.py.",
                "Metric (c): This script implementation (intent ODE sampling → endpoint dispersion) deferred to post-training.",
                "All gate results should be reported with 95% CIs; no go/no-go on marginal passes.",
            ]
        }
    }

    # Save structure
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"W3 Step-0 Validation Scaffold Ready")
    print(f"{'='*70}")
    print(f"\nGate checks documented in: {args.out}")
    print(f"\nExecute metrics in order:")
    print(f"  1. Metric (a) — SR: python examples/eval/eval_success_rate.py")
    print(f"  2. Metric (b) — VDR: python examples/collect/collect_multimodal_viz.py")
    print(f"  3. Metric (c) — Intent: Complete implementation in eval_w3_step0_validation.py")
    print(f"\nThresholds for gate pass:")
    print(f"  (a) Both SRs in [30%, 70%]")
    print(f"  (b) Flow-intent VDR > 0.35")
    print(f"  (c) Intent dispersion / action-noise dispersion > 2.0")
    print(f"\nFallback: If gates fail, use reduced-demo or shifted-reset (see §3.1 of intent_dsrl_plan_v2.md)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
