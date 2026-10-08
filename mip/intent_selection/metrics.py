"""Metrics and analysis module for intent selection experiment.

Implements oracle availability (M2), cross-validated simulator selector (M3),
random-candidate success (M1), and bootstrap confidence intervals with paired differences.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from mip.intent_selection.rng import KeyedRNG

# ============================================================================
# Configuration & Thresholds
# ============================================================================

DEFAULT_EFFECT_THRESHOLD = 0.05
DEFAULT_CI_WIDTH_THRESHOLD = 0.10
DEFAULT_CEILING_THRESHOLD = 0.95
DEFAULT_FLOOR_THRESHOLD = 0.05
DEFAULT_M3_GAIN_THRESHOLD = 0.02


# ============================================================================
# OutcomeRecord Schema and Validator
# ============================================================================

@dataclass
class OutcomeRecord:
    """Schema for outcome records from intent selection experiment."""
    experiment_id: str
    config_hash: str
    checkpoint_pair_id: str
    arm: str
    checkpoint_hash: str
    split: str
    stratum: str
    parent_reset_id: int
    state_id: int
    source_policy: str
    elapsed_steps: int
    candidate_id: int
    candidate_seed: int
    decoder_seed: int
    continuation_id: int
    tail_mode: str
    success: bool
    termination_reason: str
    steps_executed: int
    success_step: int | None
    return_value: float
    candidate_latency_ms: float
    tail_latency_ms: float
    error_status: str
    candidate_hash: str


class OutcomeRecordValidator:
    """Validates outcome records DataFrame."""

    REQUIRED_COLUMNS = {
        'experiment_id', 'config_hash', 'checkpoint_pair_id', 'arm',
        'checkpoint_hash', 'split', 'stratum', 'parent_reset_id', 'state_id',
        'source_policy', 'elapsed_steps', 'candidate_id', 'candidate_seed',
        'decoder_seed', 'continuation_id', 'tail_mode', 'success',
        'termination_reason', 'steps_executed', 'success_step', 'return',
        'candidate_latency_ms', 'tail_latency_ms', 'error_status', 'candidate_hash'
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []
        self.warnings = []

    def validate(self) -> tuple[bool, list[str], list[str]]:
        """Validate records. Returns (is_valid, errors, warnings)."""
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")
            return False, self.errors, self.warnings

        unresolved = self.df[self.df['error_status'] == 'unresolved']
        if len(unresolved) > 0:
            self.errors.append(f"Unresolved error_status in {len(unresolved)} rows")

        bad_errors = self.df[~self.df['error_status'].isin(['ok', 'unresolved'])]
        if len(bad_errors) > 0:
            self.errors.append(f"Invalid error_status values: {bad_errors['error_status'].unique().tolist()}")

        key_cols = [
            'experiment_id', 'config_hash', 'checkpoint_pair_id', 'arm',
            'split', 'stratum', 'parent_reset_id', 'state_id', 'candidate_id',
            'continuation_id', 'tail_mode'
        ]
        duplicates = self.df.duplicated(subset=key_cols, keep=False)
        if duplicates.any():
            self.errors.append(f"Duplicate keys found in {duplicates.sum() // 2} row pairs")

        self._check_complete_grids()

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _check_complete_grids(self):
        """Check that candidate×continuation grids are complete per state."""
        for (arm, state_id, tail_mode), group in (
            self.df.groupby(['arm', 'state_id', 'tail_mode'])
        ):
            candidates = set(group['candidate_id'].unique())
            continuations = set(group['continuation_id'].unique())

            for cand_id in candidates:
                cand_data = group[group['candidate_id'] == cand_id]
                cand_continuations = set(cand_data['continuation_id'].unique())
                if cand_continuations != continuations:
                    missing = continuations - cand_continuations
                    self.errors.append(
                        f"Incomplete grid: arm={arm}, state={state_id}, "
                        f"candidate={cand_id} missing continuations {missing}"
                    )


# ============================================================================
# M1: Random-Candidate Success
# ============================================================================

def compute_m1_random_success(
    df: pd.DataFrame,
    stratum: str = 'mid',
    tail_mode: str = 'native'
) -> float:
    """Compute M1: random-candidate success p_s."""
    subset = df[(df['stratum'] == stratum) & (df['tail_mode'] == tail_mode)]
    state_successes = subset.groupby('state_id')['success'].mean()
    return float(state_successes.mean())


# ============================================================================
# M2: Oracle Availability
# ============================================================================

def _stable_choose_log(n: int, k: int) -> float:
    """Compute log(choose(n, k)) stably using lgamma."""
    if n < k or k < 0:
        return -np.inf
    if k == 0:
        return 0.0
    return special.loggamma(n + 1) - special.loggamma(k + 1) - special.loggamma(n - k + 1)


def compute_oracle_availability(
    successes_per_candidate: np.ndarray,
    k: int
) -> float:
    """Compute oracle availability C_{s,r}(K)."""
    m = int(successes_per_candidate.sum())
    M = len(successes_per_candidate)

    if k > M:
        raise ValueError(f"K={k} exceeds candidate pool size M={M}")
    if m == M:
        return 1.0
    if m == 0:
        return 0.0
    if k == M:
        return float(m > 0)

    log_choose_M_minus_m_K = _stable_choose_log(M - m, k)
    log_choose_M_K = _stable_choose_log(M, k)

    if log_choose_M_minus_m_K == -np.inf:
        return 1.0

    log_ratio = log_choose_M_minus_m_K - log_choose_M_K
    ratio = np.exp(log_ratio)
    return float(1.0 - ratio)


def compute_m2_oracle_availability(
    df: pd.DataFrame,
    arm: str,
    stratum: str = 'mid',
    tail_mode: str = 'native',
    k_values: list[int] | None = None
) -> dict[int, float]:
    """Compute M2: oracle availability for an arm."""
    if k_values is None:
        k_values = [1, 4, 16]

    subset = df[
        (df['arm'] == arm) &
        (df['stratum'] == stratum) &
        (df['tail_mode'] == tail_mode)
    ]

    final_results = {}
    for k in k_values:
        state_averages = []
        for state_id in subset['state_id'].unique():
            state_group = subset[subset['state_id'] == state_id]
            state_c_values = []
            for cont_id in state_group['continuation_id'].unique():
                cont_group = state_group[state_group['continuation_id'] == cont_id]
                successes = cont_group.groupby('candidate_id')['success'].any().values
                state_c_values.append(compute_oracle_availability(successes, k))
            state_averages.append(np.mean(state_c_values))
        final_results[k] = float(np.mean(state_averages))

    return final_results


# ============================================================================
# M3: Cross-Validated Simulator Selector
# ============================================================================

def _seeded_shuffle(items: list[int], seed: int) -> list[int]:
    """Deterministically shuffle items using a seed."""
    rng = np.random.RandomState(seed)
    result = items.copy()
    rng.shuffle(result)
    return result


def _compute_analytic_tie_integrated_score(
    tied_candidates: list[int],
    success_by_cand_rep: dict[int, dict[int, bool]],
    heldout_replicates: list[int]
) -> float:
    """Compute analytic expectation over uniform tie-breaking."""
    if not tied_candidates:
        return np.nan

    heldout_scores = [
        np.mean([success_by_cand_rep[cand_id][rep_id] for rep_id in heldout_replicates])
        for cand_id in tied_candidates
    ]
    return float(np.mean(heldout_scores))


def compute_m3_cross_validated_selector(
    df: pd.DataFrame,
    arm: str,
    state_id: int,
    experiment_id: str,
    checkpoint_pair_id: str,
    tail_mode: str = 'native',
    selection_replicates: list[int] | None = None,
    heldout_replicates: list[int] | None = None,
    k_values: list[int] | None = None,
    n_subsets_k4: int = 128,
    split_sensitivity: bool = False
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Compute M3: cross-validated selector with split sensitivity.
    Returns: (primary_results, split_sensitivity_results)
    """
    if (selection_replicates is None) != (heldout_replicates is None):
        raise ValueError("pass both selection_replicates and heldout_replicates, or neither")
    if selection_replicates is None:
        # Preregistered split (spec §8 M3): {0,1}/{2,3} for R=4, {0}/{1} preview for R=2.
        reps = sorted(df.loc[df['state_id'] == state_id, 'continuation_id'].unique())
        if len(reps) >= 4:
            selection_replicates, heldout_replicates = reps[:2], reps[2:4]
        elif len(reps) == 2:
            selection_replicates, heldout_replicates = reps[:1], reps[1:]
        else:
            raise ValueError(f"M3 needs R=2 or R>=4 replicates, got {len(reps)}")
    if k_values is None:
        k_values = [1, 4, 16]

    rng_obj = KeyedRNG(experiment_id)

    subset = df[
        (df['arm'] == arm) &
        (df['state_id'] == state_id) &
        (df['tail_mode'] == tail_mode)
    ]

    if len(subset) == 0:
        empty_result = {k: {
            'selected_success': np.nan,
            'random_baseline': np.nan,
            'gain': np.nan,
            'tie_fraction': 0.0,
            'analytic_tied': np.nan,
        } for k in k_values}
        return empty_result, empty_result

    all_candidates = sorted(subset['candidate_id'].unique())
    M = len(all_candidates)

    success_by_cand_rep = {}
    for cand_id in all_candidates:
        success_by_cand_rep[cand_id] = {}
        for cont_id in selection_replicates + heldout_replicates:
            result = subset[
                (subset['candidate_id'] == cand_id) &
                (subset['continuation_id'] == cont_id)
            ]
            if len(result) != 1:
                raise ValueError(
                    f"M3 needs exactly one record for state {state_id}, arm {arm}, "
                    f"candidate {cand_id}, replicate {cont_id}; found {len(result)}"
                )
            success_by_cand_rep[cand_id][cont_id] = bool(result['success'].iloc[0])

    primary_results = _compute_m3_split(
        all_candidates, M, success_by_cand_rep, selection_replicates,
        heldout_replicates, k_values, rng_obj, arm, checkpoint_pair_id,
        state_id, n_subsets_k4
    )

    if split_sensitivity:
        sensitivity_results = _compute_m3_split(
            all_candidates, M, success_by_cand_rep, heldout_replicates,
            selection_replicates, k_values, rng_obj, arm, checkpoint_pair_id,
            state_id, n_subsets_k4
        )
    else:
        sensitivity_results = {}

    return primary_results, sensitivity_results


def _compute_m3_split(
    all_candidates: list[int],
    M: int,
    success_by_cand_rep: dict[int, dict[int, bool]],
    selection_replicates: list[int],
    heldout_replicates: list[int],
    k_values: list[int],
    rng_obj: KeyedRNG,
    arm: str,
    checkpoint_pair_id: str,
    state_id: int,
    n_subsets_k4: int
) -> dict[int, dict[str, float]]:
    """Compute M3 for a specific selection/held-out split."""
    results = {}

    for k in k_values:
        if k > M:
            k = M

        if k == 1:
            subsets_to_eval = [[cand] for cand in all_candidates]
        elif k == M or k == 16:
            subsets_to_eval = [all_candidates]
        elif k == 4:
            # Subsets are shared across arms at a state (paired design), so arm is not keyed.
            subset_seed = rng_obj.derive_seed(
                'm3_subsets',
                checkpoint_pair=checkpoint_pair_id,
                state_id=state_id,
            )
            rng = np.random.RandomState(seed=subset_seed)
            subsets_to_eval = []
            seen = set()
            max_attempts = n_subsets_k4 * 10
            attempts = 0
            while len(subsets_to_eval) < n_subsets_k4 and attempts < max_attempts:
                subset_idx = rng.choice(M, size=k, replace=False)
                subset_tuple = tuple(sorted([all_candidates[i] for i in subset_idx]))
                if subset_tuple not in seen:
                    subsets_to_eval.append((list(subset_tuple), len(seen)))
                    seen.add(subset_tuple)
                attempts += 1
        else:
            subsets_to_eval = [
                (list(comb), idx) for idx, comb in enumerate(combinations(all_candidates, k))
            ]

        selected_successes = []
        random_successes = []
        analytic_scores = []
        tie_counts = 0

        for subset_data in subsets_to_eval:
            if isinstance(subset_data, tuple):
                subset_candidates, subset_idx = subset_data
            else:
                subset_candidates = subset_data
                subset_idx = 0

            selection_scores = {}
            for cand_id in subset_candidates:
                sel_success = np.mean([
                    success_by_cand_rep[cand_id][rep_id]
                    for rep_id in selection_replicates
                ])
                selection_scores[cand_id] = sel_success

            best_score = max(selection_scores.values())
            tied = [cid for cid, sc in selection_scores.items() if sc == best_score]

            if len(tied) > 1:
                tie_counts += 1
                # Tie-break seed is keyed per subset and never depends on held-out outcomes.
                tie_seed = rng_obj.derive_seed(
                    'm3_ties',
                    checkpoint_pair=checkpoint_pair_id,
                    state_id=state_id,
                    arm=arm,
                    subset_index=subset_idx,
                )
                j_star = _seeded_shuffle(tied, seed=tie_seed)[0]
            else:
                j_star = tied[0]

            heldout_success = np.mean([
                success_by_cand_rep[j_star][rep_id]
                for rep_id in heldout_replicates
            ])
            selected_successes.append(heldout_success)

            analytic_score = _compute_analytic_tie_integrated_score(
                tied, success_by_cand_rep, heldout_replicates
            )
            analytic_scores.append(analytic_score)

            random_success = np.mean([
                np.mean([
                    success_by_cand_rep[cand_id][rep_id]
                    for rep_id in heldout_replicates
                ])
                for cand_id in subset_candidates
            ])
            random_successes.append(random_success)

        mean_selected = float(np.mean(selected_successes))
        mean_random = float(np.mean(random_successes))
        mean_analytic = float(np.mean(analytic_scores))
        gain = mean_selected - mean_random
        tie_fraction = tie_counts / len(subsets_to_eval) if subsets_to_eval else 0.0

        results[k] = {
            'selected_success': mean_selected,
            'random_baseline': mean_random,
            'gain': gain,
            'tie_fraction': tie_fraction,
            'analytic_tied': mean_analytic,
        }

    return results


# ============================================================================
# Bootstrap Confidence Intervals with Paired Differences
# ============================================================================

def _bootstrap_paired_metric(
    df: pd.DataFrame,
    metric_fn,
    stratum: str = 'mid',
    tail_mode: str = 'native',
    arm1: str = 'INTENT',
    arm2: str = 'ACTION',
    n_resamples: int = 2000,
    ci: float = 0.95,
    bootstrap_seed: int = None
) -> dict[str, float]:
    """Compute bootstrap CI for paired difference (arm1 - arm2) on same resampled parents.

    CRITICAL FIX: Assigns unique boot_copy id to each drawn parent to prevent groupby deduplication.

    Args:
        bootstrap_seed: REQUIRED (no default). Must be explicit.

    Returns dict with 'mean', 'ci_lower', 'ci_upper', 'std' of the paired difference.
    """
    if bootstrap_seed is None:
        raise ValueError("bootstrap_seed is required (no default)")

    subset = df[(df['stratum'] == stratum) & (df['tail_mode'] == tail_mode)]
    parents = sorted(subset['parent_reset_id'].unique())

    if len(parents) == 0:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}

    rng = np.random.RandomState(bootstrap_seed)
    bootstrap_diffs = []
    skipped_resamples = 0

    for _ in range(n_resamples):
        resampled_parents = rng.choice(parents, size=len(parents), replace=True)

        # CRITICAL FIX: Assign unique boot_copy to each drawn parent
        resampled_rows = []
        for boot_copy, parent_id in enumerate(resampled_parents):
            parent_rows = subset[subset['parent_reset_id'] == parent_id].copy()
            parent_rows['boot_copy'] = boot_copy
            # Modify state_id and parent_reset_id to include boot_copy to avoid deduplication
            parent_rows['state_id'] = parent_rows['state_id'].astype(str) + f'#{boot_copy}'
            parent_rows['parent_reset_id'] = parent_rows['parent_reset_id'].astype(str) + f'#{boot_copy}'
            resampled_rows.append(parent_rows)

        bootstrap_subset = pd.concat(resampled_rows, ignore_index=True)

        arm1_subset = bootstrap_subset[bootstrap_subset['arm'] == arm1]
        arm2_subset = bootstrap_subset[bootstrap_subset['arm'] == arm2]

        if len(arm1_subset) == 0 or len(arm2_subset) == 0:
            skipped_resamples += 1
            continue

        val1 = metric_fn(arm1_subset, stratum, tail_mode)
        val2 = metric_fn(arm2_subset, stratum, tail_mode)

        if not (np.isnan(val1) or np.isnan(val2)):
            diff = val1 - val2
            bootstrap_diffs.append(diff)

    if len(bootstrap_diffs) == 0:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

    return {
        'mean': float(np.mean(bootstrap_diffs)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(bootstrap_diffs)),
        'n_skipped': skipped_resamples,
    }


# ============================================================================
# Decision Helper
# ============================================================================

def apply_practical_rule(
    c_intent_k16: float,
    c_action_k16: float,
    paired_diff_k16_ci: tuple[float, float],
    m3_gain: float,
    m3_gain_ci: tuple[float, float],
    c_intent_k1: float,
    c_action_k1: float,
    c_intent_gain_k16: float | None = None,
    m3_selected_intent: float | None = None,
    m3_selected_action: float | None = None,
    effect_threshold: float = DEFAULT_EFFECT_THRESHOLD,
    ci_width_threshold: float = DEFAULT_CI_WIDTH_THRESHOLD,
    ceiling_threshold: float = DEFAULT_CEILING_THRESHOLD,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    m3_gain_threshold: float = DEFAULT_M3_GAIN_THRESHOLD,
) -> tuple[str, str]:
    """Apply the practical continuation rule from spec §10 to the primary endpoint.

    m3_gain/m3_gain_ci: INTENT's M3 selected-minus-random gain at K=16.
    c_intent_gain_k16: INTENT's own C(16) - C(1).
    m3_selected_intent/action: each arm's M3 held-out selected success at K=16.
    """
    paired_diff = c_intent_k16 - c_action_k16
    ci_lo, ci_hi = paired_diff_k16_ci
    ci_width = ci_hi - ci_lo
    ci_str = f'[{ci_lo:+.3f}, {ci_hi:+.3f}]'
    m3_str = f'M3 gain {m3_gain:.3f} (95% CI [{m3_gain_ci[0]:+.3f}, {m3_gain_ci[1]:+.3f}])'

    if c_intent_k1 >= ceiling_threshold and c_action_k1 >= ceiling_threshold:
        return 'inconclusive', (
            f'Ceiling effect: both arms ≥{ceiling_threshold} at K=1 '
            f'(INTENT={c_intent_k1:.3f}, ACTION={c_action_k1:.3f})'
        )
    if c_intent_k1 <= floor_threshold and c_action_k1 <= floor_threshold:
        return 'inconclusive', (
            f'Floor effect: both arms ≤{floor_threshold} at K=1 '
            f'(INTENT={c_intent_k1:.3f}, ACTION={c_action_k1:.3f})'
        )

    # A positive lower bound is evidence even when the interval is wide.
    if paired_diff >= effect_threshold and ci_lo > 0:
        if m3_gain_ci[0] > 0:
            return 'promising_coverage', (
                f'INTENT C(16) advantage {paired_diff:+.3f} (95% CI {ci_str}) with credible {m3_str}'
            )
        return 'inconclusive', (
            f'INTENT C(16) advantage {paired_diff:+.3f} (95% CI {ci_str}) but {m3_str} is not credible'
        )

    if (
        c_intent_gain_k16 is not None
        and m3_selected_intent is not None
        and m3_selected_action is not None
        and c_intent_gain_k16 >= effect_threshold
        and m3_selected_intent < m3_selected_action
    ):
        return 'useful_headroom', (
            f'INTENT gains {c_intent_gain_k16:.3f} from C(1) to C(16) but its selected success '
            f'{m3_selected_intent:.3f} still trails ACTION {m3_selected_action:.3f}'
        )

    if ci_width > ci_width_threshold:
        return 'inconclusive', (
            f'Wide interval for C(16) paired difference {paired_diff:+.3f}: '
            f'95% CI {ci_str} (width {ci_width:.3f} > {ci_width_threshold:.3f})'
        )

    if abs(paired_diff) < effect_threshold / 2 and m3_gain_ci[0] > m3_gain_threshold:
        return 'promising_selectability', (
            f'Similar C(16) (diff {paired_diff:+.3f}, 95% CI {ci_str}) but {m3_str}'
        )

    if ci_hi < effect_threshold:
        return 'no_advantage', (
            f'C(16) paired difference {paired_diff:+.3f} with narrow 95% CI {ci_str} '
            f'excluding a {effect_threshold:.2f} advantage'
        )

    return 'inconclusive', f'No clear pattern: C(16) diff {paired_diff:+.3f}, 95% CI {ci_str}, {m3_str}'


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze(
    manifest: dict[str, Any],
    records: pd.DataFrame,
    n_resamples: int = 2000,
    bootstrap_seed: int | None = None,
    continuation_replicates: int = 4,
    report_k_values: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Analyze intent selection experiment results."""
    experiment_id = manifest.get('experiment_id', 'unknown')

    if bootstrap_seed is None:
        rng_obj = KeyedRNG(experiment_id)
        bootstrap_seed = rng_obj.derive_seed('bootstrap', state_id=0)

    validator = OutcomeRecordValidator(records)
    is_valid, errors, warnings = validator.validate()

    if not is_valid:
        raise ValueError(f"Validation failed: {errors}")

    if warnings:
        for w in warnings:
            print(f"Warning: {w}")

    pool_size = int(records.groupby(['arm', 'state_id', 'tail_mode'])['candidate_id'].nunique().min())
    if report_k_values is None:
        report_k_values = sorted({k for k in (1, 4, 16) if k < pool_size} | {pool_size})
    elif max(report_k_values) > pool_size:
        raise ValueError(f"report_k_values {report_k_values} exceed candidate pool size {pool_size}")

    output_tables = {}

    max_replicates = records['continuation_id'].max() + 1
    is_exploratory = (max_replicates < 4)

    state_metrics = []
    for stratum in records['stratum'].unique():
        for tail_mode in records['tail_mode'].unique():
            for state_id in records[records['stratum'] == stratum]['state_id'].unique():
                for arm in records['arm'].unique():
                    subset = records[
                        (records['stratum'] == stratum) &
                        (records['tail_mode'] == tail_mode) &
                        (records['state_id'] == state_id) &
                        (records['arm'] == arm)
                    ]

                    if len(subset) == 0:
                        continue

                    m1_success = float(subset['success'].mean())
                    # C_{s,r}(K) per continuation replicate, then averaged over r (spec §8 M2).
                    # (Was: any() pooled over replicates, which overstates C.)
                    m2_values = {
                        k: float(np.mean([
                            compute_oracle_availability(
                                g.groupby('candidate_id')['success'].any().values, k)
                            for _, g in subset.groupby('continuation_id')
                        ]))
                        for k in report_k_values
                    }

                    row = {
                        'stratum': stratum,
                        'tail_mode': tail_mode,
                        'state_id': state_id,
                        'arm': arm,
                        'm1_random_success': m1_success,
                    }
                    for k in report_k_values:
                        row[f'c_k{k}'] = m2_values[k]

                    state_metrics.append(row)

    if state_metrics:
        output_tables['metrics_by_state'] = pd.DataFrame(state_metrics)

    aggregate_rows = []
    for stratum in records['stratum'].unique():
        for tail_mode in records['tail_mode'].unique():
            for arm in records['arm'].unique():
                # M1 must be computed per arm (was: pooled over both arms).
                m1 = compute_m1_random_success(records[records['arm'] == arm], stratum, tail_mode)
                m2_dict = compute_m2_oracle_availability(
                    records, arm, stratum, tail_mode, k_values=report_k_values
                )

                row = {
                    'stratum': stratum,
                    'tail_mode': tail_mode,
                    'arm': arm,
                    'm1_random_success': m1,
                }
                for k in report_k_values:
                    row[f'c_k{k}'] = m2_dict[k]
                row['exploratory'] = is_exploratory

                aggregate_rows.append(row)

    # The action arm may be labelled 'ACTION' or a variant such as 'ACTION_STOCH'.
    other_arms = [a for a in records['arm'].unique() if a != 'INTENT']
    action_arm = 'ACTION' if 'ACTION' in other_arms else (other_arms[0] if len(other_arms) == 1 else None)
    if action_arm is not None and 'INTENT' in records['arm'].unique():
        for stratum in records['stratum'].unique():
            for tail_mode in records['tail_mode'].unique():
                def m2_for_arm(df_arm, _, __):
                    # Mean over states of mean over replicates of C_{s,r}(16) (spec §8 M2).
                    # (Was: candidate ids pooled across all states and replicates.)
                    if len(df_arm) == 0:
                        return np.nan
                    k16 = min(16, int(df_arm.groupby(['state_id', 'continuation_id'])['candidate_id'].nunique().min()))
                    per_state = [
                        np.mean([
                            compute_oracle_availability(
                                g_r.groupby('candidate_id')['success'].any().values, k16)
                            for _, g_r in g_s.groupby('continuation_id')
                        ])
                        for _, g_s in df_arm.groupby('state_id')
                    ]
                    return float(np.mean(per_state))

                paired_ci = _bootstrap_paired_metric(
                    records,
                    m2_for_arm,
                    stratum=stratum,
                    tail_mode=tail_mode,
                    arm1='INTENT',
                    arm2=action_arm,
                    n_resamples=n_resamples,
                    bootstrap_seed=bootstrap_seed,
                )

                aggregate_rows.append({
                    'stratum': stratum,
                    'tail_mode': tail_mode,
                    'arm': f'INTENT-{action_arm}_difference',
                    'c_k16': paired_ci['mean'],
                    'c_k16_ci_lower': paired_ci['ci_lower'],
                    'c_k16_ci_upper': paired_ci['ci_upper'],
                    'exploratory': is_exploratory,
                })

    if aggregate_rows:
        output_tables['aggregate_metrics'] = pd.DataFrame(aggregate_rows)

    for cp_id in records['checkpoint_pair_id'].unique():
        cp_records = records[records['checkpoint_pair_id'] == cp_id]
        cp_metrics = []

        for stratum in cp_records['stratum'].unique():
            for tail_mode in cp_records['tail_mode'].unique():
                for arm in cp_records['arm'].unique():
                    subset = cp_records[
                        (cp_records['stratum'] == stratum) &
                        (cp_records['tail_mode'] == tail_mode) &
                        (cp_records['arm'] == arm)
                    ]

                    if len(subset) == 0:
                        continue

                    m1 = float(subset['success'].mean())
                    m2_dict = compute_m2_oracle_availability(
                        subset, arm, stratum, tail_mode, k_values=report_k_values)

                    cp_metrics.append({
                        'checkpoint_pair_id': cp_id,
                        'stratum': stratum,
                        'tail_mode': tail_mode,
                        'arm': arm,
                        'm1_random_success': m1,
                        **{f'c_k{k}': m2_dict[k] for k in report_k_values},
                    })

        if cp_metrics:
            output_tables[f'metrics_seed_{cp_id}'] = pd.DataFrame(cp_metrics)

    return output_tables


def write_results_to_csv(
    tables: dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """Write analysis results to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    csv_mapping = {
        'metrics_by_state': 'metrics_by_state.csv',
        'aggregate_metrics': 'aggregate_metrics.csv',
    }

    for table_name, csv_filename in csv_mapping.items():
        if table_name in tables:
            filepath = os.path.join(output_dir, csv_filename)
            tables[table_name].to_csv(filepath, index=False)
            print(f"Wrote {filepath}")

    for table_name, df in tables.items():
        if table_name.startswith('metrics_seed_'):
            csv_filename = f"{table_name}.csv"
            filepath = os.path.join(output_dir, csv_filename)
            df.to_csv(filepath, index=False)
            print(f"Wrote {filepath}")
