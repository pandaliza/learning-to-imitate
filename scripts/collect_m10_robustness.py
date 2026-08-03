#!/usr/bin/env python3
"""
Collect M10 robustness results from three evaluation campaigns (spatial, perturb, faildiag)
across four arms (A0, A1, A2, A3).

Globs for JSON files matching patterns:
  - logs/eval_spatial_*.json
  - logs/eval_perturb_*.json
  - logs/eval_faildiag_*.json

Renders three tables: spatial-transfer, perturbation, faildiag.
Missing files render as '—'. Supports --markdown for GitHub-flavored output.
"""

import argparse
import glob
import json
import pathlib
import re
import sys
from collections import defaultdict


def extract_arm(filename):
    """Extract arm name (A0/A1/A2/A3) from filename."""
    match = re.search(r'_(A[0-3])\b', filename)
    return match.group(1) if match else None


def load_json(filepath):
    """Load JSON file, return None on failure."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collect_spatial_results(pattern="logs/eval_spatial_*.json"):
    """Load all spatial JSON files, keyed by arm."""
    results = {}
    for path in sorted(glob.glob(pattern)):
        arm = extract_arm(pathlib.Path(path).name)
        if arm:
            data = load_json(path)
            if data:
                results[arm] = data
    return results


def collect_perturb_results(pattern="logs/eval_perturb_*.json"):
    """Load all perturb JSON files, keyed by arm."""
    results = {}
    for path in sorted(glob.glob(pattern)):
        arm = extract_arm(pathlib.Path(path).name)
        if arm:
            data = load_json(path)
            if data:
                results[arm] = data
    return results


def collect_faildiag_results(pattern="logs/eval_faildiag_*.json"):
    """Load all faildiag JSON files, keyed by arm."""
    results = {}
    for path in sorted(glob.glob(pattern)):
        arm = extract_arm(pathlib.Path(path).name)
        if arm:
            data = load_json(path)
            if data:
                results[arm] = data
    return results


def format_value(val, fmt=".3f"):
    """Format numeric value or return '—' for None."""
    if val is None:
        return "—"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return "—"


def print_spatial_table(spatial_results, markdown=False):
    """Print spatial transfer table: arm x mean_sr with per-task rows."""
    print("\n" + "="*80)
    print("SPATIAL TRANSFER (libero_goal -> zero-shot)")
    print("="*80)

    arms = sorted(spatial_results.keys())
    if not arms:
        print("No spatial results found.")
        return

    # Collect all tasks across all arms
    all_tasks = set()
    for data in spatial_results.values():
        if data and "per_task" in data:
            all_tasks.update(data["per_task"].keys())
    all_tasks = sorted(all_tasks)

    if markdown:
        # Markdown table header
        header = "| Task | " + " | ".join(arms) + " |"
        separator = "|" + "|".join(["---"] * (len(arms) + 1)) + "|"
        print(header)
        print(separator)

        # Mean SR row
        mean_row = "| **Mean SR** |"
        for arm in arms:
            val = spatial_results.get(arm, {}).get("mean_sr", None)
            mean_row += f" {format_value(val)} |"
        print(mean_row)

        # Per-task rows
        for task in all_tasks:
            row = f"| {task} |"
            for arm in arms:
                data = spatial_results.get(arm, {})
                val = data.get("per_task", {}).get(task, None)
                row += f" {format_value(val)} |"
            print(row)
    else:
        # Plain text table
        col_width = 50
        arm_width = 12

        # Header
        header = f"{'Task':<{col_width}}" + "".join(f"{arm:>{arm_width}}" for arm in arms)
        print(header)
        print("-" * len(header))

        # Mean SR row
        mean_row = f"{'MEAN SR':<{col_width}}"
        for arm in arms:
            val = spatial_results.get(arm, {}).get("mean_sr", None)
            mean_row += f"{format_value(val):>{arm_width}}"
        print(mean_row)

        # Per-task rows
        for task in all_tasks:
            row = f"{task:<{col_width}}"
            for arm in arms:
                data = spatial_results.get(arm, {})
                val = data.get("per_task", {}).get(task, None)
                row += f"{format_value(val):>{arm_width}}"
            print(row)


def print_perturb_table(perturb_results, markdown=False):
    """Print perturbation table: arm x delta -> SR, plus per-task detail."""
    print("\n" + "="*80)
    print("PERTURBATION ROBUSTNESS (object position perturbation)")
    print("="*80)

    arms = sorted(perturb_results.keys())
    if not arms:
        print("No perturbation results found.")
        return

    # Collect deltas (assume consistent across arms)
    deltas = []
    for data in perturb_results.values():
        if data and "deltas" in data:
            deltas = data["deltas"]
            break

    if not deltas:
        print("No delta values found in perturbation results.")
        return

    delta_strs = [str(d) for d in deltas]

    if markdown:
        # Markdown table: arm x delta
        header = "| Δ (m) | " + " | ".join(arms) + " |"
        separator = "|" + "|".join(["---"] * (len(arms) + 1)) + "|"
        print(header)
        print(separator)

        for delta_str in delta_strs:
            row = f"| {delta_str} |"
            for arm in arms:
                data = perturb_results.get(arm, {})
                results_dict = data.get("results", {}).get(delta_str, {})
                mean_sr = results_dict.get("_mean_sr", None)
                row += f" {format_value(mean_sr)} |"
            print(row)
    else:
        # Plain text table
        delta_width = 12
        arm_width = 12

        # Header
        header = f"{'Delta (m)':<{delta_width}}" + "".join(f"{arm:>{arm_width}}" for arm in arms)
        print(header)
        print("-" * len(header))

        # Rows by delta
        for delta_str in delta_strs:
            row = f"{delta_str:<{delta_width}}"
            for arm in arms:
                data = perturb_results.get(arm, {})
                results_dict = data.get("results", {}).get(delta_str, {})
                mean_sr = results_dict.get("_mean_sr", None)
                row += f"{format_value(mean_sr):>{arm_width}}"
            print(row)

    # Per-task detail section
    print("\n--- Per-Task Breakdown ---")
    for arm in arms:
        print(f"\n{arm}:")
        data = perturb_results.get(arm, {})
        results_dict = data.get("results", {})
        deltas_list = data.get("deltas", [])

        if not results_dict:
            print("  No results")
            continue

        # Collect all task IDs
        all_task_ids = set()
        for delta_data in results_dict.values():
            if isinstance(delta_data, dict):
                all_task_ids.update(k for k in delta_data.keys() if not k.startswith("_"))
        all_task_ids = sorted(all_task_ids, key=lambda x: int(x) if x.isdigit() else float('inf'))

        for task_id in all_task_ids:
            srs = []
            for delta_str in [str(d) for d in deltas_list]:
                task_data = results_dict.get(delta_str, {}).get(task_id, {})
                sr = task_data.get("sr", None)
                srs.append(format_value(sr, fmt=".2f"))

            # Get task name from any delta that has it
            task_name = None
            for delta_str in [str(d) for d in deltas_list]:
                task_data = results_dict.get(delta_str, {}).get(task_id, {})
                if "task" in task_data:
                    task_name = task_data["task"]
                    break

            task_label = f"Task {task_id}" if not task_name else task_name[:30]
            sr_str = "  ".join(srs)
            print(f"  {task_label}: {sr_str}")


def print_faildiag_table(faildiag_results, markdown=False):
    """Print faildiag table in whatever structure its JSONs have."""
    print("\n" + "="*80)
    print("FAILURE DIAGNOSIS")
    print("="*80)

    arms = sorted(faildiag_results.keys())
    if not arms:
        print("No failure diagnosis results found.")
        return

    if markdown:
        print("| Arm | Result |")
        print("|---|---|")
        for arm in arms:
            data = faildiag_results.get(arm)
            if data is None:
                print(f"| {arm} | — |")
            else:
                # Pretty-print a snippet of the JSON
                summary = json.dumps(data, indent=2)
                # Truncate for markdown
                lines = summary.split('\n')[:10]
                snippet = '\n'.join(lines)
                if len(summary.split('\n')) > 10:
                    snippet += '\n...'
                print(f"| {arm} | ```\n{snippet}\n``` |")
    else:
        # Plain text format
        for arm in arms:
            print(f"\n{arm}:")
            data = faildiag_results.get(arm)
            if data is None:
                print("  —")
            else:
                # Pretty-print the JSON with indentation
                summary = json.dumps(data, indent=2)
                for line in summary.split('\n'):
                    print(f"  {line}")


def main():
    ap = argparse.ArgumentParser(
        description="Collect M10 robustness results from spatial, perturb, and faildiag campaigns."
    )
    ap.add_argument("--markdown", action="store_true",
                    help="Emit GitHub-flavored markdown instead of plain text")
    ap.add_argument("--spatial-pattern", default="logs/eval_spatial_*.json")
    ap.add_argument("--perturb-pattern", default="logs/eval_perturb_*.json")
    ap.add_argument("--faildiag-pattern", default="logs/eval_faildiag_*.json")
    args = ap.parse_args()

    spatial = collect_spatial_results(args.spatial_pattern)
    perturb = collect_perturb_results(args.perturb_pattern)
    faildiag = collect_faildiag_results(args.faildiag_pattern)

    print_spatial_table(spatial, markdown=args.markdown)
    print_perturb_table(perturb, markdown=args.markdown)
    print_faildiag_table(faildiag, markdown=args.markdown)

    if args.markdown:
        print("\n---\n*Generated by collect_m10_robustness.py*")


if __name__ == "__main__":
    main()
