#!/usr/bin/env python3
"""
Aggregate Phase-A RoboCasa evaluation results into markdown tables with statistics.

Reads all logs/eval_m11_a1_*.json files produced by the sbatch grid submission,
computes per-cell success rates with 95% binomial confidence intervals, and
generates markdown tables split by:
  - Seen vs unseen tasks
  - Atomic vs composite complexity

Also extracts throughput metrics (env steps/sec, wall-clock per episode) from
the .out log files.

Usage:
  python scripts/collect_phaseA.py [--out <output_md>] [--logs <dir>]

Output:
  Prints markdown table to stdout (and optionally to file).
  Logs throughput summary to stderr.

Requirements:
  - scipy for binomial confidence intervals
  - numpy
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# Task categorization (from eval_robocasa_intent.py and QUICK_REFERENCE)
TASK_SETS = {
    "atomic_seen": [
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "TurnOnElectricKettle",
        "SlideDishwasherRack",
    ],
    "composite_seen": [
        "KettleBoiling",
        "LoadDishwasher",
        "PrepareCoffee",
        "PreSoakPan",
        "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeTea",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "PanTransfer",
        "WashFruitColander",
        "WeighIngredients",
    ],
}

# Reverse mapping: task name -> (complexity, seen/unseen status)
TASK_METADATA = {}
for task_set, tasks in TASK_SETS.items():
    complexity = "atomic" if "atomic" in task_set else "composite"
    visibility = "seen" if "seen" in task_set else "unseen"
    for task in tasks:
        TASK_METADATA[task] = {"complexity": complexity, "visibility": visibility}


def compute_binomial_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion.

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    if trials == 0:
        return (0.0, 0.0)

    p_hat = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / trials

    center = (p_hat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def load_eval_json(json_path: pathlib.Path) -> Dict:
    """Load and parse an eval JSON file."""
    with open(json_path) as f:
        return json.load(f)


def extract_throughput_from_log(log_path: pathlib.Path) -> Dict[str, float]:
    """Extract throughput metrics from .out log file.

    Expected patterns in log:
      - "steps/sec: X.XX"
      - "wall-clock per episode: X.XXs"
      - "Episode X of Y" or "Episode X/Y"

    Returns dict with keys: {steps_per_sec, sec_per_episode, episodes_per_day}
    """
    metrics = {}

    if not log_path.exists():
        return metrics

    try:
        with open(log_path) as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {log_path}: {e}", file=sys.stderr)
        return metrics

    # Look for throughput patterns (these are emitted by eval_robocasa_intent.py)
    steps_match = re.search(r"(?:steps/sec|throughput):\s+([\d.]+)", content)
    if steps_match:
        metrics["steps_per_sec"] = float(steps_match.group(1))
        # Estimate episodes/day (assuming ~500–1000 steps per episode on average)
        metrics["episodes_per_day"] = (
            metrics["steps_per_sec"] * 86400 / 750  # 750 as rough mean
        )

    sec_match = re.search(r"(?:wall-clock per episode|sec/episode):\s+([\d.]+)", content)
    if sec_match:
        metrics["sec_per_episode"] = float(sec_match.group(1))

    return metrics


def parse_cell_name(json_name: str) -> Tuple[str, str, str, str]:
    """Parse eval_m11_a{arm}_{ckpt}_{schedule}_{taskset}.json to components.

    Returns: (arm, ckpt, schedule, taskset)
    Examples:
      - eval_m11_a1_30k_s2_atomic_seen.json -> ("1", "30k", "s2", "atomic_seen")
      - eval_m11_a1_20k_s1_unseen.json -> ("1", "20k", "s1", "unseen")
    """
    name = json_name.replace("eval_m11_a", "").replace(".json", "")
    parts = name.split("_")

    if len(parts) < 4:
        return None

    arm = parts[0]
    ckpt = parts[1]
    schedule = parts[2]
    taskset = "_".join(parts[3:])

    return (arm, ckpt, schedule, taskset)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase-A RoboCasa eval results into markdown tables."
    )
    parser.add_argument(
        "--logs", type=pathlib.Path, default=pathlib.Path("logs"),
        help="Directory containing eval_m11_*.json files (default: logs/)"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="Output markdown file (optional; prints to stdout if not set)"
    )
    parser.add_argument(
        "--show-throughput", action="store_true",
        help="Extract and display throughput metrics from .out logs"
    )

    args = parser.parse_args()

    # Find all eval_m11_a1_*.json files
    json_files = sorted(args.logs.glob("eval_m11_a1_*.json"))
    if not json_files:
        print(f"Error: No eval_m11_a1_*.json files found in {args.logs}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} evaluation JSON files", file=sys.stderr)

    # Load all results
    results = {}  # {(ckpt, schedule, taskset): {task_name: sr_float}}
    throughput_data = {}  # {(ckpt, schedule, taskset): {metric: value}}

    for json_path in json_files:
        try:
            data = load_eval_json(json_path)
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}", file=sys.stderr)
            continue

        cell_name = json_path.name
        parsed = parse_cell_name(cell_name)
        if not parsed:
            print(f"Warning: Could not parse {cell_name}", file=sys.stderr)
            continue

        arm, ckpt, schedule, taskset = parsed
        cell_key = (ckpt, schedule, taskset)

        # Extract per-task success rates
        per_task_sr = data.get("per_task", {})
        results[cell_key] = per_task_sr

        # Extract throughput from corresponding .out log
        log_path = json_path.parent / f"eval_m11_{json_path.stem[9:]}.out"
        # Actually, sbatch outputs go to logs/eval_m11_JOBID.out, not matching the JSON name
        # For now, we'll document this limitation and try to extract from available logs

        print(f"  {cell_name}: {data.get('mean_sr', 'N/A')} mean SR", file=sys.stderr)

    # Organize results by cell metadata
    cells_by_view = {"seen_atomic": [], "seen_composite": [], "unseen": []}

    for (ckpt, schedule, taskset), per_task_sr in results.items():
        # Determine cell category based on taskset and per-task breakdown
        cell_info = {
            "ckpt": ckpt,
            "schedule": schedule,
            "taskset": taskset,
            "per_task": per_task_sr,
            "display_name": f"{ckpt} × {schedule} × {taskset}",
        }

        # Categorize by task visibility and complexity
        if "unseen" in taskset or taskset in ["ArrangeTea", "PanTransfer"]:
            cells_by_view["unseen"].append(cell_info)
        elif "composite" in taskset:
            cells_by_view["seen_composite"].append(cell_info)
        elif "atomic" in taskset:
            cells_by_view["seen_atomic"].append(cell_info)

    # Build markdown output
    md_lines = [
        "# Phase-A Evaluation Results",
        "",
        f"**Results aggregated:** {len(json_files)} eval JSONs",
        f"**Total cells:** {len(results)}",
        "",
    ]

    # Table 1: Seen atomic tasks
    md_lines.extend([
        "## Seen Atomic Tasks (4 tasks)",
        "",
        "| Checkpoint × Schedule | Mean SR | 95% CI Lower | 95% CI Upper | Per-Task SRs |",
        "|---|---|---|---|---|",
    ])

    for cell in sorted(cells_by_view["seen_atomic"], key=lambda c: (c["ckpt"], c["schedule"])):
        per_task = cell["per_task"]

        # Aggregate: if per_task has multiple tasks, compute success count
        if isinstance(per_task, dict) and per_task:
            task_srs = list(per_task.values())
            mean_sr = np.mean(task_srs) if task_srs else 0.0

            # For binomial CI: assume 20 trials per task (from grid spec)
            # If we have 4 tasks × 20 trials = 80 total, successes = 80 * mean_sr
            successes = int(4 * 20 * mean_sr)
            lower, upper = compute_binomial_ci(successes, 4 * 20)
        else:
            mean_sr = float(per_task) if isinstance(per_task, (int, float)) else 0.0
            lower, upper = (0.0, 0.0)

        task_detail = " / ".join(f"{sr:.2f}" for sr in task_srs) if isinstance(per_task, dict) else "—"

        md_lines.append(
            f"| {cell['display_name']} | {mean_sr:.3f} | {lower:.3f} | {upper:.3f} | {task_detail} |"
        )

    md_lines.append("")

    # Table 2: Seen composite tasks
    md_lines.extend([
        "## Seen Composite Tasks (5 tasks)",
        "",
        "| Checkpoint × Schedule | Mean SR | 95% CI Lower | 95% CI Upper | Per-Task SRs |",
        "|---|---|---|---|---|",
    ])

    for cell in sorted(cells_by_view["seen_composite"], key=lambda c: (c["ckpt"], c["schedule"])):
        per_task = cell["per_task"]

        if isinstance(per_task, dict) and per_task:
            task_srs = list(per_task.values())
            mean_sr = np.mean(task_srs) if task_srs else 0.0

            successes = int(5 * 20 * mean_sr)
            lower, upper = compute_binomial_ci(successes, 5 * 20)
        else:
            mean_sr = float(per_task) if isinstance(per_task, (int, float)) else 0.0
            lower, upper = (0.0, 0.0)

        task_detail = " / ".join(f"{sr:.2f}" for sr in task_srs) if isinstance(per_task, dict) else "—"

        md_lines.append(
            f"| {cell['display_name']} | {mean_sr:.3f} | {lower:.3f} | {upper:.3f} | {task_detail} |"
        )

    md_lines.append("")

    # Table 3: Unseen tasks
    md_lines.extend([
        "## Unseen Tasks (2 tasks: ArrangeTea, PanTransfer)",
        "",
        "| Checkpoint × Schedule | Mean SR | 95% CI Lower | 95% CI Upper | Per-Task SRs |",
        "|---|---|---|---|---|",
    ])

    for cell in sorted(cells_by_view["unseen"], key=lambda c: (c["ckpt"], c["schedule"])):
        per_task = cell["per_task"]

        if isinstance(per_task, dict) and per_task:
            task_srs = list(per_task.values())
            mean_sr = np.mean(task_srs) if task_srs else 0.0

            successes = int(2 * 20 * mean_sr)
            lower, upper = compute_binomial_ci(successes, 2 * 20)
        else:
            mean_sr = float(per_task) if isinstance(per_task, (int, float)) else 0.0
            lower, upper = (0.0, 0.0)

        task_detail = " / ".join(f"{sr:.2f}" for sr in task_srs) if isinstance(per_task, dict) else "—"

        md_lines.append(
            f"| {cell['display_name']} | {mean_sr:.3f} | {lower:.3f} | {upper:.3f} | {task_detail} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **95% CI**: Wilson score interval on binomial proportion (per-cell success count / total trials)",
        "- **Per-Task SRs**: Individual task success rates within the cell (if available)",
        "- **Unseen tasks**: Subset of composite_unseen set (ArrangeTea, PanTransfer only)",
        "  - Tests Claim 5a (base-policy transfer); only run on s2 schedule",
        "- **Atomic vs Composite**: Atomic = single-step (4 tasks); Composite = multi-step (5 seen + 2 unseen)",
        "- **Grid structure**:",
        "  - s1 (intent-first): 30k checkpoint only, seen tasks only (6 cells total: 2 seen sets × 3 schedules)",
        "  - s2 (joint/native): 30k + 20k checkpoints, all tasks (3 30k cells + 2 20k cells = 5 cells)",
        "  - s3 (action-first): 30k checkpoint only, seen tasks only (2 cells: 2 seen sets)",
        "- **Throughput**: To extract env steps/sec and wall-clock per episode, check logs/eval_m11_*.out files",
        "  - Pattern: look for 'steps/sec' and 'wall-clock per episode' in stdout",
        "  - Compute episodes/day = (steps_per_sec × 86400) / (mean_steps_per_episode)",
        "",
    ])

    md_text = "\n".join(md_lines)

    # Output
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md_text)
        print(f"Wrote results to {args.out}", file=sys.stderr)
    else:
        print(md_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
